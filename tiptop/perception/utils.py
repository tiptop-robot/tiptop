import numpy as np
import open3d as o3d
import trimesh
from curobo.geom.types import Cuboid, Mesh
from jaxtyping import Float


def depth_to_xyz(
    depth: Float[np.ndarray, "h w"],
    K: Float[np.ndarray, "3 3"],
) -> Float[np.ndarray, "h w 3"]:
    """Convert depth map to XYZ point cloud using camera intrinsics.

    Uses the pinhole camera model:
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        Z = depth

    Args:
        depth: Depth map in meters.
        K: Camera intrinsics matrix (3x3).

    Returns:
        XYZ map where each pixel contains (X, Y, Z) coordinates in meters.
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    h, w = depth.shape

    # Create pixel coordinate grids using broadcasting
    u = np.arange(w, dtype=np.float32)  # (w,)
    v = np.arange(h, dtype=np.float32)  # (h,)
    u_grid, v_grid = np.meshgrid(u, v)  # Both (h, w)

    # Compute XYZ using vectorized operations
    z = depth
    x = (u_grid - cx) * z / fx
    y = (v_grid - cy) * z / fy

    # Stack into (h, w, 3) array
    xyz = np.stack([x, y, z], axis=-1)
    return xyz


def get_o3d_pcd(
    xyz_map: Float[np.ndarray, "*n 3"], rgb_map: Float[np.ndarray, "*n 3"], voxel_size: float | None = None
) -> o3d.geometry.PointCloud:
    """Get open3d point cloud."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_map.reshape(-1, 3))
    pcd.colors = o3d.utility.Vector3dVector(rgb_map.reshape(-1, 3))
    if voxel_size is not None:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return pcd


def convert_trimesh_box_to_curobo_cuboid(trimesh_box: trimesh.primitives.Box, name: str) -> Cuboid:
    """Convert trimesh Box to cuRobo Cuboid."""
    # Get box properties
    center = trimesh_box.centroid
    extents = trimesh_box.extents
    color = trimesh_box.visual.main_color[:3] if hasattr(trimesh_box.visual, "main_color") else [200, 200, 200]

    # Create cuRobo Cuboid
    cuboid = Cuboid(
        name=name,
        dims=[float(extents[0]), float(extents[1]), float(extents[2])],
        pose=[float(center[0]), float(center[1]), float(center[2]), 1.0, 0.0, 0.0, 0.0],
        color=list(color),
    )
    return cuboid


def convert_trimesh_to_curobo_mesh(trimesh_obj: trimesh.Trimesh, label: str) -> Mesh:
    """Convert trimesh object to cuRobo Mesh."""
    vertices = np.asarray(trimesh_obj.vertices)
    centroid = vertices.mean(0)
    vertices_centered = vertices - centroid

    # Get colors
    if hasattr(trimesh_obj.visual, "vertex_colors") and trimesh_obj.visual.vertex_colors is not None:
        vertex_colors = np.asarray(trimesh_obj.visual.vertex_colors)[:, :3] / 255.0
    else:
        # Use a default color
        vertex_colors = np.ones((len(vertices), 3)) * 0.5

    # Create cuRobo Mesh
    mesh = Mesh(
        name=label,
        vertices=vertices_centered.tolist(),
        faces=np.asarray(trimesh_obj.faces).tolist(),
        vertex_colors=vertex_colors.tolist(),
        pose=[float(centroid[0]), float(centroid[1]), float(centroid[2]), 1.0, 0.0, 0.0, 0.0],
    )
    return mesh
