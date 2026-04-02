#!/usr/bin/env python3
"""Websocket server for tiptop perception + planning pipeline.

This server exposes the full tiptop pipeline (detection, segmentation, grasp generation,
and motion planning) over websocket, allowing clients running different Python versions
to query for trajectory plans.

Usage:
    pixi run python -m tiptop.websocket_server --port 8765
"""

import asyncio
import http
import json
import logging
import time
import traceback
from datetime import datetime
from pathlib import Path

import aiohttp
import msgpack_numpy
import numpy as np
import rerun as rr
import tyro
import websockets.asyncio.server as ws_server
import websockets.frames
from curobo.geom.types import WorldConfig
from curobo.wrap.reacher.ik_solver import IKSolver
from curobo.wrap.reacher.motion_gen import MotionGen

from tiptop.config import tiptop_cfg
from tiptop.motion_planning import build_curobo_solvers
from tiptop.perception.cameras import Frame
from tiptop.planning import build_tamp_config, run_planning, save_tiptop_plan, serialize_plan
from tiptop.recording import save_run_metadata, save_run_outputs
from tiptop.tiptop_run import Observation, run_perception
from tiptop.utils import NumpyEncoder, add_file_handler, get_robot_rerun, print_tiptop_banner, remove_file_handler, setup_logging

_log = logging.getLogger(__name__)


class TiptopPlanningServer:
    """Websocket server for tiptop perception + planning pipeline.

    Follows the same pattern as openpi's WebsocketPolicyServer.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        num_particles: int = 256,
        max_planning_time: float = 60.0,
        rerun_mode: str = "stream",
        include_workspace: bool = False,
    ) -> None:
        if rerun_mode not in {"stream", "save"}:
            raise ValueError(f"Invalid rerun mode: {rerun_mode}")

        self._host = host
        self._port = port
        self._rerun_mode = rerun_mode
        self._include_workspace = include_workspace
        self._metadata = {
            "server": "tiptop",
            "version": "0.1.0",
            "num_particles": num_particles,
            "max_planning_time": max_planning_time,
        }
        self._cfg = tiptop_cfg()

        # Motion planning components (initialized during warmup)
        self._ik_solver: IKSolver | None = None
        self._motion_gen: MotionGen | None = None
        self._initial_world_cfg: WorldConfig | None = None
        self._config = build_tamp_config(
            num_particles=num_particles,
            max_planning_time=max_planning_time,
            opt_steps=500,
            robot_type=self._cfg.robot.type,
            time_dilation_factor=self._cfg.robot.time_dilation_factor,
        )
        self._output_dir = Path("tiptop_server_outputs")

    def _reset_motion_planning(self) -> None:
        """Reset collision world to initial state to clear stale cached state between runs."""
        self._ik_solver.update_world(self._initial_world_cfg)
        self._motion_gen.update_world(self._initial_world_cfg)
        self._motion_gen.reset(reset_seed=False)

    @staticmethod
    def _health_check(connection: ws_server.ServerConnection, request: ws_server.Request) -> ws_server.Response | None:
        """Handle health check requests."""
        if request.path == "/health":
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        return None

    def serve_forever(self) -> None:
        """Run the server forever (blocking)."""
        asyncio.run(self.run())

    async def run(self) -> None:
        """Async main loop."""
        # Warm up motion planning on startup
        _log.info("Setting up motion planning...")
        self._ik_solver, self._motion_gen, self._initial_world_cfg = build_curobo_solvers(
            self._config.num_particles,
            self._config.coll_n_spheres,
            include_workspace=self._include_workspace,
        )

        _log.info(f"Starting TiptopPlanningServer on ws://{self._host}:{self._port}")
        async with ws_server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=self._health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: ws_server.ServerConnection) -> None:
        """Handle a websocket connection."""
        _log.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        # Send metadata on connection
        await websocket.send(packer.pack(self._metadata))

        while True:
            try:
                start_time = time.monotonic()

                # Receive observation from client
                raw_data = await websocket.recv()
                obs = msgpack_numpy.unpackb(raw_data)

                _log.info(f"Received planning request: task='{obs.get('task', 'unknown')}'")

                # Run the full pipeline
                infer_start = time.monotonic()
                result = await self._run_pipeline(obs)
                infer_time = time.monotonic() - infer_start

                # Add timing info
                result["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                    "total_ms": (time.monotonic() - start_time) * 1000,
                }

                # Send result back as JSON
                await websocket.send(json.dumps(result, cls=NumpyEncoder))
                _log.info(f"Sent planning result: success={result['success']}, time={infer_time:.2f}s")

            except websockets.ConnectionClosed:
                _log.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                error_msg = traceback.format_exc()
                _log.error(f"Error processing request: {error_msg}")
                await websocket.send(error_msg)
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

    async def _run_pipeline(self, obs: dict) -> dict:
        """Run the full perception + planning pipeline.

        Args:
            obs: Dictionary containing:
                - rgb: uint8 array (H, W, 3)
                - depth: float32 array (H, W) in meters
                - intrinsics: float32 array (3, 3)
                - world_from_cam: float32 array (4, 4)
                - task: str
                - q_init: float32 array (7,) joint positions

        Returns:
            Dictionary containing:
                - success: bool
                - plan: list of plan steps (trajectory or gripper actions)
                - error: str or None
        """
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        iso_timestamp = now.isoformat(timespec="seconds")
        save_dir = self._output_dir / timestamp
        save_dir.mkdir(parents=True, exist_ok=True)
        file_handler = add_file_handler(save_dir / "tiptop_run.log")

        # Extract and preprocess observation
        rgb = obs["rgb"].astype(np.uint8)
        depth = obs["depth"].copy().astype(np.float32)
        K = obs["intrinsics"].astype(np.float32)
        world_from_cam = obs["world_from_cam"].astype(np.float32)
        task_instruction = obs["task"]
        q_init = obs["q_init"]

        # Preprocess depth: remove invalid values and truncate range
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth[depth < 0] = 0.0
        depth[depth > self._cfg.perception.depth_trunc_m] = 0.0

        frame = Frame(serial="static", timestamp=0.0, rgb=rgb, intrinsics=K, depth=depth)
        observation = Observation(frame=frame, world_from_cam=world_from_cam, q_init=q_init)

        env = None
        processed_scene = None
        grounded_atoms = None
        perception_duration = None
        planning_duration = None
        failure_reason = None
        cutamp_plan = None
        try:
            # Reset collision world to clear stale cuTAMP state from previous run
            self._reset_motion_planning()

            # Initialize rerun if enabled (idempotent)
            rr.init("tiptop_server", recording_id=timestamp, spawn=self._rerun_mode == "stream")
            if self._rerun_mode == "save":
                rrd_path = save_dir / "tiptop.rrd"
                rr.save(rrd_path)
                _log.info(f"Saving Rerun stream to {rrd_path}")
            rerun_robot = get_robot_rerun()

            _log.info(f"Processing: RGB shape={rgb.shape}, depth shape={depth.shape}, task='{task_instruction}'")
            rerun_robot.set_joint_positions(q_init)

            connector = aiohttp.TCPConnector(limit=10, force_close=True)
            timeout = aiohttp.ClientTimeout(total=120.0)
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                perception_start = time.monotonic()
                env, all_surfaces, processed_scene, grounded_atoms = await run_perception(
                    session,
                    observation,
                    task_instruction,
                    save_dir,
                    depth_estimator=None,
                    gripper_mask=None,
                    include_workspace=self._include_workspace,
                )
                perception_duration = time.monotonic() - perception_start
            # Log camera intrinsics and pose to rerun if enabled
            rr.log("cam", rr.Pinhole(image_from_camera=K))
            rr.log(
                "cam",
                rr.Transform3D(translation=world_from_cam[:3, 3], mat3x3=world_from_cam[:3, :3], axis_length=0.05),
            )

            # Run cuTAMP planning
            _log.info("Running cuTAMP planning...")
            cutamp_plan, planning_duration, failure_reason = await asyncio.to_thread(
                run_planning,
                env,
                self._config,
                q_init,
                self._ik_solver,
                processed_scene.grasps,
                self._motion_gen,
                all_surfaces,
            )

            if cutamp_plan is None:
                return {
                    "success": False,
                    "plan": None,
                    "error": f"cuTAMP failed to find a plan: {failure_reason}",
                }

            serialized_plan = serialize_plan(cutamp_plan, q_init)
            plan_path = save_dir / "tiptop_plan.json"
            save_tiptop_plan(serialized_plan, plan_path)
            _log.info(f"Saved outputs to {save_dir}")

            return {
                "success": True,
                "plan": serialized_plan,
                "error": None,
            }

        except Exception as e:
            _log.error(f"Pipeline error: {e}", exc_info=True)
            if not failure_reason:
                failure_reason = str(e)
            return {
                "success": False,
                "plan": None,
                "error": str(e),
            }
        finally:
            if env is not None and processed_scene is not None and observation is not None:
                save_run_outputs(save_dir, env, processed_scene.grasps)
                save_run_metadata(
                    save_dir=save_dir,
                    timestamp=iso_timestamp,
                    task_instruction=task_instruction,
                    q_at_capture=observation.q_init,
                    world_from_cam=observation.world_from_cam,
                    perception_duration=perception_duration,
                    grounded_atoms=grounded_atoms,
                    planning_success=cutamp_plan is not None,
                    planning_failure_reason=failure_reason,
                    planning_duration=planning_duration,
                )
            if file_handler is not None:
                remove_file_handler(file_handler)


def _run_server(
    host: str = "0.0.0.0",
    port: int = 8765,
    num_particles: int = 256,
    max_planning_time: float = 60.0,
    rerun_mode: str = "stream",
    include_workspace: bool = False,
) -> None:
    """Tiptop websocket planning server.

    Args:
        host: Host to bind to.
        port: Port to bind to.
        num_particles: Number of particles for cuTAMP.
        max_planning_time: Max planning time in seconds.
        rerun_mode: Rerun visualization mode. 'stream' spawns the Rerun viewer; 'save' writes .rrd files to disk.
        include_workspace: If True, include real-robot workspace cuboids in the collision world.
    """
    print_tiptop_banner()
    setup_logging()
    logging.getLogger("websockets.server").setLevel(logging.INFO)

    server = TiptopPlanningServer(
        host=host,
        port=port,
        num_particles=num_particles,
        max_planning_time=max_planning_time,
        rerun_mode=rerun_mode,
        include_workspace=include_workspace,
    )
    server.serve_forever()


def entrypoint():
    tyro.cli(_run_server)


if __name__ == "__main__":
    entrypoint()
