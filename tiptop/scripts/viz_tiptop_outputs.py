import json
import logging
from pathlib import Path
import cv2
import dill
import open3d as o3d
import numpy as np
import rerun as rr
import torch
from PIL import Image
from omegaconf import OmegaConf

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


def viz_tiptop_outputs(outputs_dir: str, visualize_grasps: bool = True, num_grasps_per_object: int = 30, log_transform_arrows: bool = True):
    setup_logging()
    outputs_dir = Path(outputs_dir)
    perception_dir = outputs_dir / "perception"

    # Load metadata
    metadata_path = outputs_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Could not find metadata.json in {outputs_dir}")
    with open(outputs_dir / "metadata.json") as f:
        metadata = json.load(f)
    if metadata["version"] != "1.0.0":
        raise NotImplementedError(f"Version {metadata['version']} not supported")
    _log.info(f"Task instruction: {metadata['task_instruction']}")
    _log.info(f"Grounded Goal Atoms: {metadata['perception']['grounded_atoms']}")
    # Start rerun for visualization
    rr.init(application_id="viz_tiptop_outputs",spawn=True)

    # Load tiptop config from this run
    tiptop_cfg = OmegaConf.load(outputs_dir / "tiptop.yml")
    robot_type = tiptop_cfg["robot"]["type"]
    robot_rr = get_robot_rerun(robot_type=robot_type)
    robot_rr.set_joint_positions(metadata["observation"]["q_at_capture"])

    # Log camera and RGB
    world_from_cam = np.array(metadata["observation"]["world_from_cam"])
    rr.log("cam", rr.Transform3D(mat3x3=world_from_cam[:3, :3], translation=world_from_cam[:3, 3]))
    with open(perception_dir / "intrinsics.json", "r") as f:
        intrinsics = json.load(f)
    K = np.array(intrinsics["intrinsics"])
    rr.log("cam", rr.Pinhole(image_from_camera=K))
    rgb = Image.open(outputs_dir / "rgb.png")
    rr.log("cam/rgb", rr.Image(rgb))

    # Mask out depth where gripper is present
    gripper_mask = cv2.imread(str(perception_dir / "gripper_mask.png"), cv2.IMREAD_GRAYSCALE)
    depth = cv2.imread(str(perception_dir / "depth.png"), cv2.IMREAD_UNCHANGED)
    depth[gripper_mask == 255] = 0
    rr.log("cam/depth", rr.DepthImage(depth, meter=1000.0))

    # Bounding boxes and mask visualization
    bboxes_viz = Image.open(outputs_dir / "bboxes_viz.png")
    masks_viz = Image.open(outputs_dir / "masks_viz.png")
    rr.log("bboxes_viz", rr.Image(bboxes_viz))
    rr.log("masks_viz", rr.Image(masks_viz))

    # Point cloud
    pcd = o3d.io.read_point_cloud(perception_dir / "pointcloud.ply")
    rr.log("pcd", rr.Points3D(positions=pcd.points, colors=pcd.colors))

    # Grasps
    if visualize_grasps:
        grasps = torch.load(perception_dir / "grasps.pt", weights_only=False, map_location="cpu")
        viz_grasps(grasps, num_grasps_per_object)

    # Log tiptop plan
    tiptop_plan_path = outputs_dir / "tiptop_plan.json"
    if not tiptop_plan_path.exists():
        _log.warning(f"Could not find tiptop_plan.json in {outputs_dir}")
        tiptop_plan = {}
    else:
        tiptop_plan = load_tiptop_plan(tiptop_plan_path)
    if tiptop_plan["version"] != "1.0.0":
        raise NotImplementedError(f"TiPToP plan version {metadata['version']} not supported")

    # Load TAMPEnvironment using dill (variant of pickle)
    try:
        with open(perception_dir / "cutamp_env.pkl", "rb") as f:
            cutamp_env = dill.load(f)

        # Log all the objects
        for obj in cutamp_env.movables:
            log_curobo_pose_to_rerun(
                f"world/{obj.name}", obj, static_transform=False, log_arrows=log_transform_arrows
            )
            rr.log(
                f"world/{obj.name}/mesh", curobo_to_rerun(obj.get_mesh(), compute_vertex_normals=True),
                static=True
            )
        for obj in cutamp_env.statics:
            log_curobo_mesh_to_rerun(f"world/{obj.name}", obj.get_mesh(), static_transform=True,
                                     log_arrows=log_transform_arrows)
    except Exception as e:
        _log.warning(f"Failed to load cutamp_env.pkl, skipping (source may have changed): {e}")

    start_time = 0.0
    timeline = "tiptop_execution"

    rr.set_time(timeline, duration=start_time)
    robot_rr.set_joint_positions(tiptop_plan["q_init"])

    obj_to_current_pose = {obj.name: pose_list_to_mat4x4(obj.pose) for obj in cutamp_env.movables + cutamp_env.statics}
    for obj, mat4x4 in obj_to_current_pose.items():
        rr.log(f"world/{obj}", rr.Transform3D(translation=mat4x4[:3, 3], mat3x3=mat4x4[:3, :3]))

    for action_dict in tiptop_plan.get("steps", []):
        if action_dict["type"] == "trajectory":
            traj = torch.tensor(action_dict["positions"])
            dt = action_dict["dt"]
            end_time = start_time + len(traj) * dt
            times = [rr.TimeColumn(timeline, duration=np.linspace(start_time, end_time, len(traj)))]
            key_to_columns = robot_rr.get_rr_columns(traj)
            for key, columns in key_to_columns.items():
                rr.send_columns(key, indexes=times * len(columns), columns=columns)
            start_time = end_time
        else:
            print(action_dict["type"])

            robot_state = world.kin_model.get_state(plan.position)
            world_from_ee = robot_state.ee_pose.get_matrix()
            world_from_obj = world_from_ee @ ee_from_obj
            ts = visualizer.log_joint_trajectory_with_mat4x4(
                traj=plan.position,
                mat4x4_key=f"world/{obj}",
                mat4x4=world_from_obj,
                timeline=timeline,
                start_time=ts,
                dt=dt,
            )
    print()


def viz_tiptop_outputs_entrypoint():
    import tyro

    # tyro.cli(viz_tiptop_outputs)
    outputs_dir = "/home/labubu/workspace/tiptop/tiptop_outputs/success/2026-03-31/2026-03-31_19-34-47"
    viz_tiptop_outputs(outputs_dir, visualize_grasps=True)


if __name__ == "__main__":
    viz_tiptop_outputs_entrypoint()
