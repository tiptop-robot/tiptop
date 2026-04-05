"""Offline H5 evaluation mode for TiPToP.

Loads observations from H5 files (pi-sim-evals format) and runs perception + planning
without a real robot, saving a serialized plan JSON file for downstream evaluation.
"""

import asyncio
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import aiohttp
import h5py
import numpy as np
import rerun as rr
import tyro
from scipy.spatial.transform import Rotation

from tiptop.config import tiptop_cfg
from tiptop.motion_planning import build_curobo_solvers
from tiptop.perception.cameras import Frame
from tiptop.planning import build_tamp_config, run_planning, save_tiptop_plan, serialize_plan
from tiptop.recording import save_run_metadata, save_run_outputs
from tiptop.tiptop_run import Observation, run_perception
from tiptop.utils import (
    add_file_handler,
    check_cutamp_version,
    get_robot_rerun,
    print_tiptop_banner,
    remove_file_handler,
    setup_logging,
)

_log = logging.getLogger(__name__)


def load_h5_observation(h5_path: Path) -> Observation:
    """Load an observation from an H5 file (pi-sim-evals format).

    Expected fields: rgb, depth, intrinsic_matrix, pos_w, quat_w_ros (w,x,y,z), q_init.
    """
    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"H5 file not found: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        rgb = f["rgb"][:]
        depth = f["depth"][:]
        pos_w = f["pos_w"][:]
        quat_w_ros = f["quat_w_ros"][:]  # stored as [w, x, y, z] in this format
        intrinsics = f["intrinsic_matrix"][:]
        q_init = np.atleast_1d(f["q_init"][()])
        if q_init.ndim != 1 or q_init.shape[0] < 2:
            raise ValueError(
                f"q_init in H5 file has unexpected shape {q_init.shape} (value: {q_init}). "
                f"Expected a 1D array of joint positions (e.g., 7 for Franka). "
                f"The H5 file may have been generated incorrectly — re-run save_h5_obs.py to regenerate it."
            )

    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth.squeeze(-1)

    if rgb.dtype != np.uint8:
        rgb = (rgb * 255.0).clip(0, 255).astype(np.uint8) if rgb.max() <= 1.0 else rgb.astype(np.uint8)

    # Preprocess depth: remove invalid values and truncate range
    depth = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    depth[depth < 0] = 0.0
    depth[depth > tiptop_cfg().perception.depth_trunc_m] = 0.0

    # Build world_from_cam: quat_w_ros is [w, x, y, z] in this h5 format
    quat_xyzw = np.array([quat_w_ros[1], quat_w_ros[2], quat_w_ros[3], quat_w_ros[0]], dtype=np.float32)
    R = Rotation.from_quat(quat_xyzw).as_matrix()
    world_from_cam = np.eye(4, dtype=np.float32)
    world_from_cam[:3, :3] = R
    world_from_cam[:3, 3] = pos_w.astype(np.float32)
    world_from_cam[:3, 3] -= np.array([0.0, 0.0, -0.015], dtype=np.float32)  # calibration offset for this h5 format

    frame = Frame(
        serial="static",
        timestamp=0.0,
        rgb=rgb.astype(np.uint8),
        intrinsics=intrinsics.astype(np.float32),
        depth=depth,
    )
    return Observation(frame=frame, world_from_cam=world_from_cam, q_init=q_init.astype(np.float32))


def run_tiptop_h5(
    h5_path: str,
    task_instruction: str,
    output_dir: str = "tiptop_h5_outputs",
    max_planning_time: float = 60.0,
    opt_steps_per_skeleton: int = 500,
    num_particles: int = 256,
    cutamp_visualize: bool = False,
    rr_spawn: bool = True,
):
    """
    TiPToP offline runner. Loads an H5 observation file and runs perception + planning,
    saving a serialized plan JSON file for downstream evaluation.

    Args:
        h5_path: Path to H5 observation file.
        task_instruction: Task instruction (e.g. 'put the cube in the bowl').
        output_dir: Top-level directory to save outputs; a timestamped subdirectory is created per run.
        max_planning_time: Maximum time to spend planning with cuTAMP across all skeletons (approximate).
        opt_steps_per_skeleton: Number of optimization steps per skeleton in cuTAMP.
        num_particles: Number of particles for cuTAMP; decrease if running out of GPU memory.
        cutamp_visualize: Whether to visualize cuTAMP optimization.
        rr_spawn: Whether to spawn a Rerun viewer.
    """
    assert max_planning_time > 0
    assert opt_steps_per_skeleton > 0
    assert num_particles > 0

    if not task_instruction:
        raise ValueError("--task-instruction is required")

    print_tiptop_banner()
    check_cutamp_version()

    cfg = tiptop_cfg()
    config = build_tamp_config(
        num_particles=num_particles,
        max_planning_time=max_planning_time,
        opt_steps=opt_steps_per_skeleton,
        robot_type=cfg.robot.type,
        time_dilation_factor=cfg.robot.time_dilation_factor,
        collision_activation_distance=0.0,
        enable_visualizer=cutamp_visualize,
    )

    setup_logging(level=logging.INFO)
    ik_solver, motion_gen, _ = build_curobo_solvers(num_particles, config.coll_n_spheres, include_workspace=False)

    rr.init("tiptop_h5_run", spawn=rr_spawn)
    observation = load_h5_observation(Path(h5_path))
    robot_rr = get_robot_rerun()
    robot_rr.set_joint_positions(observation.q_init)

    timestamp = datetime.now()
    save_dir = Path(output_dir) / timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    save_dir.mkdir(parents=True, exist_ok=True)
    file_handler = add_file_handler(save_dir / "tiptop_run.log")

    try:

        async def _run_perception():
            connector = aiohttp.TCPConnector(limit=10, force_close=True)
            timeout = aiohttp.ClientTimeout(total=60.0)
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                return await run_perception(
                    session,
                    observation,
                    task_instruction,
                    save_dir,
                    depth_estimator=None,
                    gripper_mask=None,
                    include_workspace=False,
                )

        _log.info("Running perception pipeline...")
        perception_start = time.perf_counter()
        env, all_surfaces, processed_scene, grounded_atoms = asyncio.run(_run_perception())
        perception_duration = time.perf_counter() - perception_start
        _log.info("Planning with cuTAMP...")
        cutamp_plan = None
        planning_duration = None
        failure_reason = None
        try:
            cutamp_plan, planning_duration, failure_reason = run_planning(
                env,
                config,
                observation.q_init,
                ik_solver,
                processed_scene.grasps,
                motion_gen,
                all_surfaces,
                experiment_dir=save_dir / "cutamp",
            )
        finally:
            save_run_outputs(save_dir, env, processed_scene.grasps)
            save_run_metadata(
                save_dir=save_dir,
                timestamp=timestamp.isoformat(timespec="seconds"),
                task_instruction=task_instruction,
                q_at_capture=observation.q_init,
                world_from_cam=observation.world_from_cam,
                perception_duration=perception_duration,
                grounded_atoms=grounded_atoms,
                planning_success=cutamp_plan is not None,
                planning_failure_reason=failure_reason,
                planning_duration=planning_duration,
            )

        if cutamp_plan is not None:
            plan_path = save_dir / "tiptop_plan.json"
            save_tiptop_plan(serialize_plan(cutamp_plan, observation.q_init), plan_path)
            _log.info(f"Saved TiPToP plan to {plan_path}")
        else:
            _log.warning(f"No plan found: {failure_reason}")

        _log.info(f"Saved outputs to {save_dir}")
    finally:
        remove_file_handler(file_handler)
        rr.disconnect()

    return save_dir


def entrypoint():
    """CLI entrypoint wrapper. Calls run_tiptop_h5 and force-exits to avoid GPU cleanup segfaults."""
    try:
        tyro.cli(run_tiptop_h5)
    except Exception:
        _log.exception("TiPToP run failed")
        os._exit(1)
    else:
        os._exit(0)


if __name__ == "__main__":
    entrypoint()
