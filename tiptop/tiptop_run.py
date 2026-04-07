import asyncio
import logging
import shutil
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import aiohttp
import numpy as np
import open3d as o3d
import rerun as rr
import tyro
from curobo.geom.types import Cuboid, Mesh
from curobo.types.base import TensorDeviceType
from curobo.wrap.reacher.ik_solver import IKSolver
from curobo.wrap.reacher.motion_gen import MotionGen
from cutamp.config import TAMPConfiguration
from cutamp.envs import TAMPEnvironment
from cutamp.tamp_domain import HandEmpty, On
from cutamp.utils.rerun_utils import log_curobo_mesh_to_rerun
from jaxtyping import Bool, Float
from scipy.spatial import KDTree

from tiptop.config import load_calibration, tiptop_cfg
from tiptop.execute_plan import execute_cutamp_plan
from tiptop.motion_planning import build_curobo_solvers, go_to_capture
from tiptop.perception.cameras import (
    Camera,
    DepthEstimator,
    Frame,
    ZedCamera,
    get_depth_estimator,
    get_external_camera,
    get_hand_camera,
)
from tiptop.perception.m2t2 import m2t2_to_tiptop_transform
from tiptop.perception.sam2 import sam2_client
from tiptop.perception.segmentation import segment_pointcloud_by_masks, segment_table_with_ransac
from tiptop.perception.utils import convert_mesh_to_aabb_cuboid, convert_trimesh_box_to_curobo_cuboid, convert_trimesh_to_curobo_mesh
from tiptop.perception_wrapper import detect_and_segment, predict_depth_and_grasps
from tiptop.planning import build_tamp_config, run_planning, save_tiptop_plan, serialize_plan
from tiptop.recording import (
    record_cameras,
    save_perception_outputs,
    save_run_metadata,
    save_run_outputs,
)
from tiptop.utils import (
    RobotClient,
    add_file_handler,
    check_cutamp_version,
    get_robot_client,
    get_robot_rerun,
    load_gripper_mask,
    print_tiptop_banner,
    remove_file_handler,
    setup_logging,
)
from tiptop.viz_utils import get_gripper_mesh, get_heatmap
from tiptop.workspace import workspace_cuboids

_log = logging.getLogger(__name__)
tensor_args = TensorDeviceType()

_executor_pool = None


class UserExitException(Exception):
    """Raised when user explicitly requests to exit."""


@dataclass(frozen=True)
class Observation:
    """Snapshot of sensor data and robot state needed for one perception+planning run."""

    frame: Frame
    world_from_cam: Float[np.ndarray, "4 4"]
    q_init: Float[np.ndarray | list, "n"]


@dataclass(frozen=True)
class _DemoContainer:
    """Container for storing things needed for the live robot demo."""

    robot: RobotClient
    cam: Camera
    external_cam: Camera | None
    enable_recording: bool
    ee_from_cam: Float[np.ndarray, "4 4"]
    depth_estimator: DepthEstimator

    gripper_mask: Bool[np.ndarray, "h w"]

    ik_solver: IKSolver
    motion_gen: MotionGen


@dataclass
class ProcessedScene:
    """Processed 3D scene ready for TAMP."""

    table_cuboid: Cuboid
    object_meshes: dict[str, Mesh]
    object_pcds: dict[str, o3d.geometry.PointCloud]
    grasps: dict[str, dict]  # Label -> grasp data with tensor versions


def capture_live_observation(container: _DemoContainer) -> Observation:
    """Read robot joint positions and compute world_from_cam via forward kinematics."""
    q_curr = container.robot.get_joint_positions()
    q_curr_pt = tensor_args.to_device(q_curr)
    world_from_ee = container.motion_gen.kinematics.get_state(q_curr_pt).ee_pose.get_numpy_matrix()[0]
    world_from_cam = world_from_ee @ container.ee_from_cam
    frame = container.cam.read_camera()
    return Observation(frame=frame, world_from_cam=world_from_cam, q_init=q_curr)


def get_demo_container(
    num_particles: int, num_spheres: int, collision_activation_distance: float, enable_recording: bool = False
) -> _DemoContainer:
    """Cache and warm-up everything needed for the live demo."""
    _log.info("Starting demo warmup...")
    client = get_robot_client()

    # Setup cameras
    cam = get_hand_camera()
    external_cam = get_external_camera()
    ee_from_cam = load_calibration(cam.serial)

    # External camera for recording (if enabled)
    if enable_recording:
        if not isinstance(cam, ZedCamera):
            raise NotImplementedError(f"Recording requires a ZED hand camera, got {type(cam).__name__}")
        if not isinstance(external_cam, ZedCamera):
            raise NotImplementedError(f"Recording requires a ZED external camera, got {type(external_cam).__name__}")

    # Create depth estimator once — closed over camera intrinsics
    # Cache the SAM2 client
    sam2_client()

    # Warm-up IK solver and motion generator
    ik_solver, motion_gen, _ = build_curobo_solvers(num_particles, num_spheres, collision_activation_distance)
    return _DemoContainer(
        robot=client,
        cam=cam,
        external_cam=external_cam,
        enable_recording=enable_recording,
        ee_from_cam=ee_from_cam,
        depth_estimator=get_depth_estimator(cam),
        gripper_mask=load_gripper_mask(),
        ik_solver=ik_solver,
        motion_gen=motion_gen,
    )


async def check_server_health(session: aiohttp.ClientSession):
    """Check health of FoundationStereo and M2T2 server."""
    from tiptop.perception.foundation_stereo import check_health_status as fs_check_health_status
    from tiptop.perception.m2t2 import check_health_status as m2t2_check_health_status

    cfg = tiptop_cfg()
    await asyncio.gather(
        fs_check_health_status(session, cfg.perception.foundation_stereo.url),
        m2t2_check_health_status(session, cfg.perception.m2t2.url),
    )
    _log.info("Server health checks successful!")


def _label_rollout(save_dir: Path, output_dir: str, date_str: str, timestamp: str) -> None:
    """Prompt user to label rollout as success/failure, moving it out of eval/. Loops on invalid input."""
    try:
        while True:
            user_input = (
                input(
                    "\nWas the execution successful? Enter 'y' for success, 'n' for failure, or leave empty to skip: "
                )
                .strip()
                .lower()
            )
            if user_input == "y":
                dest = Path(output_dir) / "success" / date_str / timestamp
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(save_dir, dest)
                _log.info(f"Moved rollout to success directory: {dest}")
                return
            elif user_input == "n":
                dest = Path(output_dir) / "failure" / date_str / timestamp
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(save_dir, dest)
                _log.info(f"Moved rollout to failure directory: {dest}")
                return
            elif user_input == "":
                _log.info(f"Keeping rollout in eval directory: {save_dir}")
                return
            else:
                print("Invalid input. Please enter 'y', 'n', or leave empty to skip.")
    except EOFError:
        _log.info("No input received, keeping rollout in eval directory")


def _get_task_instruction() -> str:
    task_instruction = ""
    while not task_instruction:
        try:
            task_instruction = input(
                "Enter task instruction (e.g., 'place the red cup on the table', or 'exit' to quit): "
            ).strip()
            if task_instruction.lower() == "exit":
                raise UserExitException("User requested exit")
        except KeyboardInterrupt:
            raise UserExitException("User interrupted with Ctrl+C")

    return task_instruction


def create_tamp_environment(
    object_meshes: dict[str, Mesh], table_cuboid: Cuboid, grounded_atoms: list[dict], include_workspace: bool
) -> tuple[TAMPEnvironment, list[Cuboid | Mesh]]:
    def _normalize_surface_label(label: str) -> str:
        """Map Gemini's table aliases onto the canonical TAMP table object."""
        if label in {"table_surface", "tabletop", "table_top", "counter", "countertop", "work_surface"}:
            return table_cuboid.name
        return label

    _log.info(f"Object mesh labels: {list(object_meshes.keys())}")

    # Identify which objects are used as surfaces (second arg in on(x, y))
    surface_labels = set()
    for atom in grounded_atoms:
        if atom["predicate"] == "on" and len(atom["args"]) == 2:
            surface_labels.add(_normalize_surface_label(atom["args"][1]))

    # Separate movables and surfaces
    movables = []
    surfaces = []
    for label, mesh in object_meshes.items():
        if label in surface_labels:
            # Use AABB cuboid for surfaces: convex hull meshes cause false collisions when
            # other objects rest on or near the surface (they appear inside the hull volume).
            surfaces.append(convert_mesh_to_aabb_cuboid(mesh))
        else:
            movables.append(mesh)
    _log.info(f"Movables: {[m.name for m in movables]}")
    _log.info(f"Surfaces: {[s.name for s in surfaces]}")

    # Create goal state from grounded atoms
    goal_state = {HandEmpty.ground()}
    for atom in grounded_atoms:
        if atom["predicate"] == "on" and len(atom["args"]) == 2:
            movable_label, surface_label = atom["args"]
            surface_label = _normalize_surface_label(surface_label)
            goal_state.add(On.ground(movable_label, surface_label))
            _log.info(f"Goal: {movable_label} on {surface_label}")

    # All surfaces include table and detected surface objects
    all_surfaces = [table_cuboid, *surfaces]
    statics = list(workspace_cuboids()) if include_workspace else []
    for surface in all_surfaces:
        statics.append(surface)

    # Create TAMP environment
    env = TAMPEnvironment(
        name="tiptop_cutamp",
        movables=movables,
        statics=statics,
        type_to_objects={"Movable": movables, "Surface": all_surfaces},
        goal_state=frozenset(goal_state),
    )
    _log.info(f"Created TAMP environment with {len(movables)} movables, {len(all_surfaces)} surfaces")
    return env, all_surfaces


def process_scene_geometry(
    xyz_map: np.ndarray,
    rgb_map: np.ndarray,
    masks: np.ndarray,
    bboxes: list,
    grasps: dict,
    object_pcds: dict[str, o3d.geometry.PointCloud] | None = None,
) -> ProcessedScene:
    """Process perception results into 3D scene geometry for TAMP.

    Args:
        xyz_map: World-space XYZ coordinates (H, W, 3)
        rgb_map: RGB image (H, W, 3) in 0-255 range
        masks: Segmentation masks from SAM2
        bboxes: Bounding boxes from Gemini
        grasps: Grasp predictions from M2T2
        object_pcds: Optional pre-computed object point clouds

    Returns:
        ProcessedScene with table cuboid, object meshes, pcds, and filtered grasps
    """
    # Segment table with RANSAC (returns trimesh Box)
    table_trimesh = segment_table_with_ransac(xyz_map, rgb_map, masks)
    table_cuboid = convert_trimesh_box_to_curobo_cuboid(table_trimesh, name="table")
    log_curobo_mesh_to_rerun("world/table", table_cuboid.get_mesh(), static_transform=True)

    # For filtering to table plane height
    config = TAMPConfiguration()
    table_top_z = table_trimesh.bounds[1, 2] + config.world_activation_distance + config.coll_sphere_radius * 2
    object_trimeshes, object_pcds_computed = segment_pointcloud_by_masks(
        xyz_map,
        rgb_map,
        masks,
        bboxes,
        table_top_z,
        return_pcd=True,
        erode_pixels=tiptop_cfg().perception.mask_erosion_pixels,
    )

    # Use provided point clouds if available, otherwise use computed ones
    if object_pcds is None:
        object_pcds = object_pcds_computed

    # Associate grasps with objects by checking contact point proximity
    # Build a single KDTree from all object points with label tracking
    obj_labels = list(object_pcds.keys())
    all_points = []
    point_to_label = []  # Maps each point index to its object label
    for label, pcd in object_pcds.items():
        obj_points = np.asarray(pcd.points)
        all_points.append(obj_points)
        point_to_label.extend([label] * len(obj_points))

    all_points = np.vstack(all_points)
    point_to_label = np.array(point_to_label)
    combined_kdtree = KDTree(all_points)

    # Re-associate grasps to objects based on contact point proximity
    # Collect all valid grasps in flat arrays first
    all_poses, all_confs, all_contacts, all_labels = [], [], [], []
    for _, grasp_dict in grasps.items():
        poses, confs, contacts = grasp_dict["poses"], grasp_dict["confidences"], grasp_dict["contacts"]
        if len(contacts) == 0:
            continue

        dists, nearest_idxs = combined_kdtree.query(contacts)
        nearest_labels = point_to_label[nearest_idxs]
        within_thresh = dists < tiptop_cfg().perception.contact_threshold_m
        all_poses.append(poses[within_thresh])
        all_confs.append(confs[within_thresh])
        all_contacts.append(contacts[within_thresh])
        all_labels.append(nearest_labels[within_thresh])

    # Group by object label using boolean masks
    filtered_grasps = {}
    if all_poses:
        all_poses = np.concatenate(all_poses)
        all_confs = np.concatenate(all_confs)
        all_contacts = np.concatenate(all_contacts)
        all_labels = np.concatenate(all_labels)

        for label in obj_labels:
            mask = all_labels == label
            filtered_grasps[label] = {
                "poses": all_poses[mask],
                "confidences": all_confs[mask],
                "contacts": all_contacts[mask],
            }
            count = mask.sum()
            if count > 0:
                _log.info(
                    f"Object {label}: Associated {count} grasps (within {tiptop_cfg().perception.contact_threshold_m * 100:.1f}cm)"
                )
            else:
                _log.warning(f"Object {label}: No grasps within threshold")
    else:
        for label in obj_labels:
            filtered_grasps[label] = {
                "poses": np.array([]).reshape(0, 4, 4),
                "confidences": np.array([]),
                "contacts": np.array([]).reshape(0, 0, 3),
            }
            _log.warning(f"Object {label}: No grasps within threshold")

    gripper_mesh = get_gripper_mesh()
    vertices = np.asarray(gripper_mesh.vertices)
    vertices_hom = np.c_[vertices, np.ones(len(vertices))]  # Add homogeneous coordinate
    faces = np.asarray(gripper_mesh.triangles)
    viz_grasp_dur = 0.0

    # Convert trimesh objects to cuRobo meshes and log to Rerun
    object_meshes = {}
    for label, trimesh_obj in object_trimeshes.items():
        curobo_mesh = convert_trimesh_to_curobo_mesh(trimesh_obj, label)
        object_meshes[label] = curobo_mesh
        label_clean = label.replace(" ", "-")
        log_curobo_mesh_to_rerun(f"world/objects/{label_clean}", curobo_mesh.get_mesh(), static_transform=True)

        # Log the point cloud
        pcd = object_pcds[label]
        rr.log(f"world/obj_pcd/{label_clean}", rr.Points3D(positions=pcd.points, colors=pcd.colors))

        # Transform grasps to tcp frame
        grasp_dict = filtered_grasps[label]
        world_from_obj = np.eye(4)
        curobo_pose = np.array(curobo_mesh.pose)
        assert np.allclose(curobo_pose[3:], np.array([1.0, 0.0, 0.0, 0.0]))
        world_from_obj[:3, 3] = curobo_pose[:3]
        obj_from_world = np.linalg.inv(world_from_obj)

        world_from_grasp = grasp_dict["poses"] @ m2t2_to_tiptop_transform()
        obj_from_grasp = obj_from_world @ world_from_grasp
        filtered_grasps[label]["grasps_obj"] = tensor_args.to_device(obj_from_grasp)
        filtered_grasps[label]["confidences_pt"] = tensor_args.to_device(filtered_grasps[label]["confidences"])

        if len(world_from_grasp) == 0:
            continue

        # Visualize the resulting grasps
        viz_start = time.perf_counter()
        my_vertices_hom = vertices_hom.copy()

        # Convert to tiptop convention and select top grasps
        grasp_poses = world_from_grasp[:30]
        confidences = filtered_grasps[label]["confidences"][:30]
        transformed_verts = np.einsum("nij,mj->nmi", grasp_poses, my_vertices_hom)[..., :3]
        colors = get_heatmap(confidences)

        for grasp_idx, (verts, color) in enumerate(zip(transformed_verts, colors)):
            rr.log(
                f"world/grasps/{label}/{grasp_idx:04d}",
                rr.Mesh3D(
                    vertex_positions=verts, triangle_indices=faces, vertex_colors=np.tile(color, (len(verts), 1))
                ),
                static=True,
            )
        viz_grasp_dur += time.perf_counter() - viz_start

    _log.info(f"Visualizing grasps took: {viz_grasp_dur:.2f}s")
    return ProcessedScene(
        table_cuboid=table_cuboid,
        object_meshes=object_meshes,
        object_pcds=object_pcds,
        grasps=filtered_grasps,
    )


async def run_perception(
    session: aiohttp.ClientSession,
    observation: Observation,
    task_instruction: str,
    save_dir: Path,
    depth_estimator: DepthEstimator | None = None,
    gripper_mask: Bool[np.ndarray, "h w"] | None = None,
    include_workspace: bool = True,
    log_to_rerun: bool = True,
) -> tuple[TAMPEnvironment, list, ProcessedScene, list[dict]]:
    start_time = time.perf_counter()

    frame = observation.frame
    rgb = frame.rgb
    if log_to_rerun:
        rr.log("rgb", rr.Image(rgb))

    # Run depth+grasps and detection concurrently
    depth_results, detection_results = await asyncio.gather(
        predict_depth_and_grasps(
            session,
            frame,
            observation.world_from_cam,
            tiptop_cfg().perception.voxel_downsample_size,
            depth_estimator=depth_estimator,
            gripper_mask=gripper_mask,
        ),
        detect_and_segment(rgb, task_instruction),
    )
    _log.info(f"Capturing observation and running perception APIs took {time.perf_counter() - start_time:.2f}s")

    # Save results (ProcessPoolExecutor for live mode, default thread pool for h5 mode)
    loop = asyncio.get_running_loop()
    save_future = loop.run_in_executor(
        _executor_pool,
        save_perception_outputs,
        rgb,
        frame.intrinsics,
        depth_results["depth_map"],
        depth_results["xyz_map"],
        depth_results["rgb_map"],
        detection_results["bboxes"],
        detection_results["masks"],
        save_dir,
        gripper_mask,
    )

    if log_to_rerun:
        rr.log(
            "world/pcd",
            rr.Points3D(
                positions=depth_results["xyz_map"].reshape(-1, 3), colors=depth_results["rgb_map"].reshape(-1, 3)
            ),
        )

    # Run scene geometry processing while saving
    proc_st = time.perf_counter()
    process_coroutine = asyncio.to_thread(
        process_scene_geometry,
        depth_results["xyz_map"],
        depth_results["rgb_map"],
        detection_results["masks"],
        detection_results["bboxes"],
        depth_results["grasps"],
    )
    processed_scene, save_result = await asyncio.gather(process_coroutine, save_future)

    if log_to_rerun:
        bbox_viz, masks_viz = save_result
        rr.log("bboxes", rr.Image(bbox_viz))
        rr.log("masks", rr.Image(masks_viz))

    env, all_surfaces = create_tamp_environment(
        processed_scene.object_meshes,
        processed_scene.table_cuboid,
        detection_results["grounded_atoms"],
        include_workspace,
    )
    _log.info(f"Processing scene and perception results took {time.perf_counter() - proc_st:.2f}s")
    _log.info(f"Perception pipeline completed, took {time.perf_counter() - start_time:.2f}s")
    return env, all_surfaces, processed_scene, detection_results["grounded_atoms"]


async def async_entrypoint(container: _DemoContainer, config: TAMPConfiguration, output_dir: str, execute_plan: bool):
    """Main async entrypoint for the live robot demo."""
    cfg = tiptop_cfg()

    # Force TCP handshake for every request
    connector = aiohttp.TCPConnector(limit=10, force_close=True)
    async with aiohttp.ClientSession(connector=connector) as session:
        while True:
            try:
                _log.debug(f"Preparing TiPToP for next run...")
                await check_server_health(session)

                # Go to capture pose and ask user for instruction
                _log.debug("Moving robot to capture joint positions")
                go_to_capture(time_dilation_factor=cfg.robot.time_dilation_factor, motion_gen=container.motion_gen)
                task_instruction = _get_task_instruction()  # Let UserExitException propagate
                _log.info(f"User entered instruction: {task_instruction}")

                now = datetime.now()
                timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
                iso_timestamp = now.isoformat(timespec="seconds")
                date_str = now.strftime("%Y-%m-%d")
                rr.init("tiptop_run", recording_id=timestamp, spawn=True)
                # Log workspace for visualization purposes
                robot_rr = get_robot_rerun()
                for obj in workspace_cuboids():
                    log_curobo_mesh_to_rerun(f"world/workspace/{obj.name}", obj.get_mesh(), static_transform=True)

                save_dir = Path(output_dir) / "eval" / timestamp
                _log.info(f"Saving logs, results, and visualizations to {save_dir}")

                # Add log file handler for this run
                file_handler = add_file_handler(save_dir / "tiptop_run.log")
                try:
                    # Capture robot state and compute camera pose
                    observation = capture_live_observation(container)
                    robot_rr.set_joint_positions(observation.q_init)

                    # Now we're ready! Start timing
                    _log.info("Running Perception...")
                    perception_start = time.perf_counter()
                    env, all_surfaces, processed_scene, grounded_atoms = await run_perception(
                        session,
                        observation,
                        task_instruction,
                        save_dir,
                        depth_estimator=container.depth_estimator,
                        gripper_mask=container.gripper_mask,
                    )
                    perception_duration = time.perf_counter() - perception_start

                    cutamp_plan = None
                    planning_duration = None
                    failure_reason = None
                    try:
                        _log.info("Running Planning...")
                        cutamp_plan, planning_duration, failure_reason = run_planning(
                            env,
                            config,
                            q_init=observation.q_init,
                            ik_solver=container.ik_solver,
                            grasps=processed_scene.grasps,
                            motion_gen=container.motion_gen,
                            all_surfaces=all_surfaces,
                            experiment_dir=save_dir / "cutamp",
                        )
                        _log.info(f"Perception and cuTAMP planning took: {perception_duration + planning_duration:.2f}s")
                        if cutamp_plan is not None:
                            plan_path = save_dir / "tiptop_plan.json"
                            save_tiptop_plan(serialize_plan(cutamp_plan, observation.q_init), plan_path)
                            _log.info(f"Saved TiPToP plan to {plan_path}")

                        if cutamp_plan is not None and execute_plan:
                            _log.info("Executing plan...")
                            # Execute with optional recording
                            if container.enable_recording:
                                cameras_to_record = [
                                    (
                                        container.external_cam,
                                        save_dir / "external_cam.svo",
                                        save_dir / "external_cam.mp4",
                                    ),
                                ]
                                if isinstance(container.cam, ZedCamera):
                                    cameras_to_record.append(
                                        (container.cam, save_dir / "hand_cam.svo", save_dir / "hand_cam.mp4"),
                                    )
                                with record_cameras(cameras_to_record):
                                    execute_cutamp_plan(cutamp_plan, client=container.robot)
                            else:
                                execute_cutamp_plan(cutamp_plan, client=container.robot)
                            _log.info("Finished executing plan!")
                        elif cutamp_plan is not None:
                            _log.info("Skipping cuTAMP plan execution on real robot")
                        else:
                            _log.warning(f"No plan found: {failure_reason}")

                        _log.debug(f"Finished run for instruction: {task_instruction}")
                    finally:
                        # Always save env, grasps, metadata, and artifacts regardless of success
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
                        _log.info(f"Logs, results, and visualizations saved to {save_dir}")

                    if execute_plan:
                        _label_rollout(save_dir, output_dir, date_str, timestamp)
                except Exception:
                    _log.exception("TiPToP run failed")
                    raise
                finally:
                    # Always remove the file handler after the run
                    remove_file_handler(file_handler)
            except (UserExitException, KeyboardInterrupt) as e:
                if isinstance(e, KeyboardInterrupt):
                    _log.info("Interrupted by user (Ctrl+C)")
                else:
                    _log.info("User requested exit")
                break


def _sync_entrypoint(
    output_dir: str = "tiptop_outputs",
    max_planning_time: float = 60.0,
    opt_steps_per_skeleton: int = 500,
    execute_plan: bool = True,
    cutamp_visualize: bool = False,
    num_particles: int = 256,
    enable_recording: bool = False,
):
    """
    TiPToP live robot runner. Runs continuously on the real robot.

    Args:
        output_dir: Top-level directory to save outputs to; a timestamped subdirectory is created per run.
        max_planning_time: Maximum time to spend planning with cuTAMP across all skeletons (approximate).
        opt_steps_per_skeleton: Number of optimization steps per skeleton in cuTAMP.
        execute_plan: Whether to execute the plan on the real robot.
        cutamp_visualize: Whether to visualize cuTAMP optimization.
        num_particles: Number of particles for cuTAMP; decrease if running out of GPU memory.
        enable_recording: Whether to record external camera video during execution.
    """
    assert max_planning_time > 0
    assert opt_steps_per_skeleton > 0
    assert num_particles > 0

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

    global _executor_pool
    setup_logging(level=logging.DEBUG)

    container = get_demo_container(num_particles, config.coll_n_spheres, 0.0, enable_recording)
    # Workers ignore SIGINT so only the main process handles Ctrl+C for clean shutdown
    _executor_pool = ProcessPoolExecutor(
        max_workers=4, initializer=signal.signal, initargs=(signal.SIGINT, signal.SIG_IGN)
    )

    exit_code = 1
    try:
        asyncio.run(async_entrypoint(container, config, output_dir, execute_plan))
        exit_code = 0
    except (UserExitException, KeyboardInterrupt) as e:
        if isinstance(e, KeyboardInterrupt):
            _log.info("Interrupted during startup/shutdown (Ctrl+C)")
        else:
            _log.debug("Exit detected")
        exit_code = 0
    finally:
        if container is not None:
            _log.debug("Tearing down cameras and robot...")
            container.cam.close()
            if container.external_cam is not None:
                container.external_cam.close()
            container.robot.close()
        if _executor_pool is not None:
            _executor_pool.shutdown(wait=False, cancel_futures=True)
        sys.exit(exit_code)


def entrypoint():
    tyro.cli(_sync_entrypoint)


if __name__ == "__main__":
    entrypoint()
