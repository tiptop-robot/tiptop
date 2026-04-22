import json
import logging
import re
from pathlib import Path
import cv2
import dill
import open3d as o3d
import numpy as np
import rerun as rr
import torch
from PIL import Image
from omegaconf import OmegaConf

from curobo.types.base import TensorDeviceType
from cutamp.envs.utils import TAMPEnvironment
from cutamp.robots import load_robot_container
from cutamp.robots.utils import RerunRobot
from cutamp.utils.common import pose_list_to_mat4x4
from cutamp.utils.rerun_utils import log_curobo_pose_to_rerun, curobo_to_rerun, log_curobo_mesh_to_rerun
from tiptop.perception.m2t2 import m2t2_to_tiptop_transform
from tiptop.planning import load_tiptop_plan
from tiptop.utils import get_robot_rerun, setup_logging
from tiptop.viz_utils import get_heatmap, get_gripper_mesh

_log = logging.getLogger(__name__)


def viz_grasps(grasps, num_grasps_per_object: int):
    """Visualize grasps"""
    gripper_mesh = get_gripper_mesh()
    vertices = np.asarray(gripper_mesh.vertices)
    vertices_hom = np.c_[vertices, np.ones(len(vertices))]  # Add homogeneous coordinate
    faces = np.asarray(gripper_mesh.triangles)

    for obj_name, grasp_dict in grasps.items():
        num_grasps = len(grasp_dict["poses"])
        if num_grasps == 0:
            continue

        world_from_grasp = grasp_dict["poses"][:num_grasps_per_object] @ m2t2_to_tiptop_transform()
        confidences = grasp_dict["confidences"][:num_grasps_per_object]

        transformed_verts = np.einsum("nij,mj->nmi", world_from_grasp, vertices_hom)[..., :3]
        colors = get_heatmap(confidences)

        for grasp_idx, (verts, color) in enumerate(zip(transformed_verts, colors)):
            rr.log(
                f"grasps/{obj_name}/{grasp_idx:04d}",
                rr.Mesh3D(
                    vertex_positions=verts, triangle_indices=faces, vertex_colors=np.tile(color, (len(verts), 1))
                ),
                static=True,
            )


def viz_tiptop_plan(tiptop_plan: dict, cutamp_env: TAMPEnvironment, robot_rr: RerunRobot, robot_type: str) -> None:
    """Visualize a TiPToP plan on the tiptop_execution timeline, including object poses while grasped."""
    kin_model = load_robot_container(robot_type, TensorDeviceType()).kin_model
    device = kin_model.tensor_args.device

    # Set initial object poses and robot position
    curr_time = 0.0
    timeline = "tiptop_execution"
    rr.set_time(timeline, duration=curr_time)
    robot_rr.set_joint_positions(tiptop_plan["q_init"])
    obj_to_current_pose = {
        obj.name: pose_list_to_mat4x4(obj.pose).numpy() for obj in cutamp_env.movables + cutamp_env.statics
    }
    for obj, mat4x4 in obj_to_current_pose.items():
        rr.log(f"world/{obj}", rr.Transform3D(translation=mat4x4[:3, 3], mat3x3=mat4x4[:3, :3]))

    grasped_obj: str | None = None
    ee_from_obj: np.ndarray | None = None  # (4, 4) rigid transform from EE to grasped object
    last_q: np.ndarray = tiptop_plan["q_init"]

    for action_dict in tiptop_plan.get("steps", []):
        if action_dict["type"] == "trajectory":
            traj = torch.tensor(action_dict["positions"], device=device)
            dt = action_dict["dt"]
            end_time = curr_time + len(traj) * dt
            times = [rr.TimeColumn(timeline, duration=np.linspace(curr_time, end_time, len(traj)))]
            for key, columns in robot_rr.get_rr_columns(traj).items():
                rr.send_columns(key, indexes=times * len(columns), columns=columns)

            if grasped_obj is not None:
                world_from_ee = kin_model.get_state(traj).ee_pose.get_matrix().cpu().numpy()
                world_from_obj = world_from_ee @ ee_from_obj
                obj_cols = rr.Transform3D.columns(
                    mat3x3=world_from_obj[:, :3, :3], translation=world_from_obj[:, :3, 3]
                )
                rr.send_columns(f"world/{grasped_obj}", indexes=times * len(obj_cols), columns=obj_cols)
                obj_to_current_pose[grasped_obj] = world_from_obj[-1]

            curr_time = end_time
            last_q = action_dict["positions"][-1]

        elif action_dict["type"] == "gripper":
            if action_dict["action"] == "close":
                # Parse object name from label e.g. "Pick(crackers_in_wrapper, grasp1, q1)"
                match = re.match(r"\w+\((\w+)", action_dict["label"])
                if match is None:
                    raise ValueError(f"Could not parse object name from label: {action_dict['label']}")
                grasped_obj = match.group(1)
                world_from_ee = (
                    kin_model.get_state(torch.tensor(last_q, device=device)[None]).ee_pose.get_matrix()[0].cpu().numpy()
                )
                ee_from_obj = np.linalg.inv(world_from_ee) @ obj_to_current_pose[grasped_obj]
            elif action_dict["action"] == "open":
                grasped_obj = None
                ee_from_obj = None
            else:
                raise ValueError(f"Unknown gripper action: {action_dict['action']}")
        else:
            raise ValueError(f"Unknown action type in tiptop plan: {action_dict['type']}")


def viz_tiptop_run(
    run_dir: str,
    visualize_grasps: bool = True,
    visualize_plan: bool = True,
    num_grasps_per_object: int = 30,
    log_transform_arrows: bool = True,
) -> None:
    """
    Visualize TiPToP outputs (perception and plan) from a saved run directory in Rerun.

    Args:
        run_dir: Path to a saved TiPToP run directory (contains metadata.json, rgb.png, etc.).
        visualize_grasps: Whether to visualize the M2T2 grasp candidates.
        visualize_plan: Whether to visualize the TiPToP plan trajectory, including object poses while grasped.
        num_grasps_per_object: Maximum number of grasp candidates to display per object.
        log_transform_arrows: Whether to log coordinate frame arrows on object transforms.
    """
    setup_logging()
    run_dir = Path(run_dir)
    perception_dir = run_dir / "perception"

    # Load metadata
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Could not find metadata.json in {run_dir}")
    with open(run_dir / "metadata.json") as f:
        metadata = json.load(f)
    if metadata["version"] != "1.0.0":
        raise NotImplementedError(f"Version {metadata['version']} not supported")
    _log.info(f"Task instruction: {metadata['task_instruction']}")
    _log.info(f"Grounded Goal Atoms: {metadata['perception']['grounded_atoms']}")
    # Start rerun for visualization
    rr.init(application_id="viz_tiptop_outputs", spawn=True)

    # Load tiptop config from this run
    tiptop_cfg = OmegaConf.load(run_dir / "tiptop.yml")
    robot_type = tiptop_cfg["robot"]["type"]
    robot_rr = get_robot_rerun(robot_type=robot_type)
    robot_rr.set_joint_positions(metadata["observation"]["q_at_capture"])

    # Log camera and RGB
    world_from_cam = np.array(metadata["observation"]["world_from_cam"])
    rr.log("cam", rr.Transform3D(mat3x3=world_from_cam[:3, :3], translation=world_from_cam[:3, 3]), static=True)
    with open(perception_dir / "intrinsics.json", "r") as f:
        intrinsics = json.load(f)
    K = np.array(intrinsics["intrinsics"])
    rr.log("cam", rr.Pinhole(image_from_camera=K), static=True)
    rgb = Image.open(run_dir / "rgb.png")
    rr.log("cam/rgb", rr.Image(rgb), static=True)

    # Mask out depth where gripper is present
    depth = cv2.imread(str(perception_dir / "depth.png"), cv2.IMREAD_UNCHANGED)
    gripper_mask_path = perception_dir / "gripper_mask.png"
    if gripper_mask_path.exists():
        gripper_mask = cv2.imread(str(perception_dir / "gripper_mask.png"), cv2.IMREAD_GRAYSCALE)
        depth[gripper_mask == 255] = 0
    else:
        _log.warning("Gripper mask not found, using full depth")
    rr.log("cam/depth", rr.DepthImage(depth, meter=1000.0), static=True)

    # Bounding boxes and mask visualization
    bboxes_viz = Image.open(run_dir / "bboxes_viz.png")
    masks_viz = Image.open(run_dir / "masks_viz.png")
    rr.log("bboxes_viz", rr.Image(bboxes_viz), static=True)
    rr.log("masks_viz", rr.Image(masks_viz), static=True)

    # Point cloud
    pcd = o3d.io.read_point_cloud(perception_dir / "pointcloud.ply")
    rr.log("pcd", rr.Points3D(positions=pcd.points, colors=pcd.colors), static=True)

    # Grasps
    if visualize_grasps:
        grasps = torch.load(perception_dir / "grasps.pt", weights_only=False, map_location="cpu")
        viz_grasps(grasps, num_grasps_per_object)

    # Load TAMPEnvironment using dill (variant of pickle)
    cutamp_env = None
    try:
        with open(perception_dir / "cutamp_env.pkl", "rb") as f:
            cutamp_env = dill.load(f)

        # Log all the objects
        for obj in cutamp_env.movables:
            log_curobo_pose_to_rerun(f"world/{obj.name}", obj, static_transform=False, log_arrows=log_transform_arrows)
            rr.log(f"world/{obj.name}/mesh", curobo_to_rerun(obj.get_mesh(), compute_vertex_normals=True), static=True)
        for obj in cutamp_env.statics:
            log_curobo_mesh_to_rerun(
                f"world/{obj.name}", obj.get_mesh(), static_transform=True, log_arrows=log_transform_arrows
            )
    except Exception as e:
        _log.warning(f"Failed to load cutamp_env.pkl, skipping (source may have changed): {e}")

    if not visualize_plan:
        return

    # Load and visualize the tiptop plan
    tiptop_plan_path = run_dir / "tiptop_plan.json"
    if not tiptop_plan_path.exists():
        _log.warning(f"Could not find tiptop_plan.json in {run_dir}")
        return
    tiptop_plan = load_tiptop_plan(tiptop_plan_path)
    if tiptop_plan["version"] != "1.0.0":
        raise NotImplementedError(f"TiPToP plan version {tiptop_plan['version']} not supported")

    if cutamp_env is None:
        _log.warning("Cannot visualize plan without cutamp_env, skipping")
        return

    viz_tiptop_plan(tiptop_plan, cutamp_env, robot_rr, robot_type)


def viz_tiptop_run_entrypoint():
    import tyro

    tyro.cli(viz_tiptop_run)


if __name__ == "__main__":
    viz_tiptop_run_entrypoint()
