import asyncio
import logging

import aiohttp
import numpy as np
import open3d as o3d
import rerun as rr
from curobo.types.base import TensorDeviceType
from cutamp.robots import load_fr3_robotiq_container, load_ur5_container

from tiptop.config import load_calibration, tiptop_cfg
from tiptop.perception.cameras import (
    RealsenseFrame,
    get_depth_estimator,
    get_hand_camera,
)
from tiptop.perception.m2t2 import generate_grasps, m2t2_to_tiptop_transform
from tiptop.perception.utils import depth_to_xyz
from tiptop.utils import get_robot_client, get_robot_rerun, load_gripper_mask, patch_log_level, setup_logging
from tiptop.viz_utils import get_gripper_mesh, get_heatmap

_log = logging.getLogger(__name__)


async def _run_demo(num_grasps_per_object: int):
    client = get_robot_client()
    robot_rr = get_robot_rerun()

    cfg = tiptop_cfg()
    tensor_args = TensorDeviceType()
    with patch_log_level("curobo", logging.ERROR):
        if cfg.robot.type == "fr3_robotiq":
            robot_container = load_fr3_robotiq_container(tensor_args)
        elif cfg.robot.type == "ur5":
            robot_container = load_ur5_container(tensor_args)
        else:
            raise ValueError(f"Unknown robot type: {cfg.robot.type}")

    # Load gripper mask for masking out depth
    gripper_mask = load_gripper_mask()

    # Setup camera and depth estimator
    cam = get_hand_camera()
    depth_estimator = get_depth_estimator(cam)
    ee_from_cam = load_calibration(cam.serial)

    # Extract the camera frame
    frame = cam.read_camera()
    K = frame.intrinsics
    q_curr = client.get_joint_positions()
    robot_rr.set_joint_positions(q_curr)
    q_curr_pt = tensor_args.to_device(q_curr)
    world_from_ee = robot_container.kin_model.get_state(q_curr_pt).ee_pose.get_numpy_matrix()[0]
    world_from_cam = world_from_ee @ ee_from_cam
    rr.log("cam", rr.Pinhole(image_from_camera=K))
    rr.log("cam", rr.Transform3D(translation=world_from_cam[:3, 3], mat3x3=world_from_cam[:3, :3]))
    _log.info("Successfully retrieved camera frame and robot joint positions")

    rgb = frame.rgb
    rr.log("cam/rgb", rr.Image(rgb))

    if isinstance(frame, RealsenseFrame):
        ir_left, ir_right = frame.ir_left, frame.ir_right
        ir_left_rgb = np.stack([ir_left, ir_left, ir_left], axis=-1)
        ir_right_rgb = np.stack([ir_right, ir_right, ir_right], axis=-1)
        rr.log("left_ir", rr.Image(ir_left_rgb))
        rr.log("right_ir", rr.Image(ir_right_rgb))

    # Blended image with the gripper mask
    left_blended = rgb.copy()
    left_blended[gripper_mask] = left_blended[gripper_mask] * 0.7 + np.array([255, 0, 0]) * 0.3
    rr.log("blended", rr.Image(left_blended))

    # Predict depth with FoundationStereo
    _log.info("Running FoundationStereo to predict depth")
    async with aiohttp.ClientSession() as session:
        pred_depth = await depth_estimator(session, frame)
    rr.log("cam/depth", rr.DepthImage(pred_depth, meter=1.0))

    # Project to point cloud and set gripper mask to zeros
    xyz_map = depth_to_xyz(pred_depth, K)  # in cam frame
    xyz_map = xyz_map @ world_from_cam[:3, :3].T + world_from_cam[:3, 3]
    xyz_map[gripper_mask] = np.nan
    rgb_map = rgb.copy()
    rr.log("pcd", rr.Points3D(positions=xyz_map.reshape(-1, 3), colors=rgb_map.reshape(-1, 3)))

    # Downsample the points
    o3d_pcd = o3d.geometry.PointCloud()
    valid_mask = ~np.isnan(xyz_map).any(axis=-1)
    xyz_valid = xyz_map[valid_mask].reshape(-1, 3)
    rgb_valid = rgb_map[valid_mask].reshape(-1, 3) / 255.0  # o3d requires 0-1 float
    o3d_pcd.points = o3d.utility.Vector3dVector(xyz_valid)
    o3d_pcd.colors = o3d.utility.Vector3dVector(rgb_valid)
    o3d_pcd = o3d_pcd.voxel_down_sample(voxel_size=0.0075)
    rr.log("pcd_downsampled", rr.Points3D(positions=o3d_pcd.points, colors=o3d_pcd.colors))

    # Now run M2T2 on downsampled point cloud
    _log.info("Running M2T2 to generate grasps")
    grasps = generate_grasps(
        server_url=cfg.perception.m2t2.url, scene_xyz=np.asarray(o3d_pcd.points), scene_rgb=np.asarray(o3d_pcd.colors)
    )

    # Visualize the resulting grasps
    gripper_mesh = get_gripper_mesh()
    vertices = np.asarray(gripper_mesh.vertices)
    vertices_hom = np.c_[vertices, np.ones(len(vertices))]  # Add homogeneous coordinate
    faces = np.asarray(gripper_mesh.triangles)
    m2t2_to_tiptop = m2t2_to_tiptop_transform()

    for obj_name, grasps_dict in grasps.items():
        # Convert to tiptop convention and select top grasps
        grasp_poses = grasps_dict["poses"][:num_grasps_per_object] @ m2t2_to_tiptop
        confidences = grasps_dict["confidences"][:num_grasps_per_object]
        transformed_verts = np.einsum("nij,mj->nmi", grasp_poses, vertices_hom)[..., :3]
        colors = get_heatmap(confidences)

        for grasp_idx, (verts, color) in enumerate(zip(transformed_verts, colors)):
            rr.log(
                f"grasps/{obj_name}/{grasp_idx:04d}",
                rr.Mesh3D(
                    vertex_positions=verts, triangle_indices=faces, vertex_colors=np.tile(color, (len(verts), 1))
                ),
            )

    _log.info("Visualized M2T2 grasps")
    _log.info("Perception demo complete!")


def perception_demo(num_grasps_per_object: int = 32, rr_spawn: bool = True):
    """Demo for FoundationStereo and M2T2."""
    setup_logging()
    rr.init("perception_demo", spawn=rr_spawn)
    asyncio.run(_run_demo(num_grasps_per_object))


if __name__ == "__main__":
    perception_demo()
