"""Per-object shape completion via RecGen."""

import logging
import time

import aiohttp
import cv2
import numpy as np
import open3d as o3d
import trimesh
from jaxtyping import Bool, Float, UInt8

from tiptop.perception.recgen import generate_shape_async

_log = logging.getLogger(__name__)


async def reconstruct_objects_with_recgen(
    session: aiohttp.ClientSession,
    server_url: str,
    rgb_cam: UInt8[np.ndarray, "h w 3"],
    depth_cam: Float[np.ndarray, "h w"],
    masks: Bool[np.ndarray, "n h w"],
    bboxes: list[dict],
    intrinsics: Float[np.ndarray, "3 3"],
    world_from_cam: Float[np.ndarray, "4 4"],
    xyz_world: Float[np.ndarray, "h w 3"],
    rgb_world: Float[np.ndarray, "h w 3"],
    erode_pixels: int,
    seed: int = 42,
    target_faces: int | None = None,
) -> tuple[dict[str, trimesh.Trimesh], dict[str, o3d.geometry.PointCloud]]:
    """Run RecGen on each object mask sequentially.

    The mesh is reconstructed in the camera frame and transformed to world frame
    via world_from_cam. The point cloud is built from xyz_world (the partial depth
    observation); the caller is responsible for any table-relative filtering.

    Returns (object_meshes, object_pcds) keyed by bbox["label"].
    """
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    masks_2d = masks.astype(bool)
    if len(bboxes) != masks_2d.shape[0]:
        raise ValueError(f"bboxes ({len(bboxes)}) and masks ({masks_2d.shape[0]}) length mismatch")

    object_meshes: dict[str, trimesh.Trimesh] = {}
    object_pcds: dict[str, o3d.geometry.PointCloud] = {}

    for mask_2d, bbox in zip(masks_2d, bboxes):
        label = bbox["label"]

        # Erode to handle depth edge noise; fall back to the un-eroded mask if too few
        # valid points remain (e.g. thin objects like knives).
        original_mask = mask_2d
        if erode_pixels > 0:
            kernel = np.ones((erode_pixels * 2 + 1, erode_pixels * 2 + 1), np.uint8)
            mask_2d = cv2.erode(mask_2d.astype(np.uint8), kernel, iterations=1).astype(bool)

        xyz_obj = xyz_world[mask_2d]
        rgb_obj = rgb_world[mask_2d]
        valid = ~np.isnan(xyz_obj).any(axis=1)
        xyz_obj = xyz_obj[valid]
        rgb_obj = rgb_obj[valid]

        if len(xyz_obj) < 10 and erode_pixels > 0:
            _log.warning(f"{label}: too few points ({len(xyz_obj)}) after erosion; retrying with erode_pixels=0")
            mask_2d = original_mask
            xyz_obj = xyz_world[mask_2d]
            rgb_obj = rgb_world[mask_2d]
            valid = ~np.isnan(xyz_obj).any(axis=1)
            xyz_obj = xyz_obj[valid]
            rgb_obj = rgb_obj[valid]

        if len(xyz_obj) < 10:
            _log.warning(f"Skipping {label}: only {len(xyz_obj)} valid depth points")
            continue

        t0 = time.perf_counter()
        payload = await generate_shape_async(
            session,
            server_url,
            rgb=rgb_cam,
            depth=depth_cam,
            mask=mask_2d,
            intrinsics=intrinsics,
            seed=seed,
            target_faces=target_faces,
        )
        recgen_s = time.perf_counter() - t0

        verts_cam = np.asarray(payload["vertices"], dtype=np.float64)
        verts_hom = np.c_[verts_cam, np.ones(len(verts_cam))]
        verts_world = (world_from_cam @ verts_hom.T).T[:, :3]

        mesh_kwargs = {
            "vertices": verts_world,
            "faces": np.asarray(payload["faces"]),
            "process": False,
        }
        if "vertex_colors" in payload:
            mesh_kwargs["vertex_colors"] = np.asarray(payload["vertex_colors"])
        mesh = trimesh.Trimesh(**mesh_kwargs)
        mesh.metadata = {"name": label}
        object_meshes[label] = mesh

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_obj)
        pcd.colors = o3d.utility.Vector3dVector(rgb_obj)
        object_pcds[label] = pcd

        _log.info(
            f"RecGen {label}: {recgen_s:.2f}s, {len(mesh.vertices)} verts, "
            f"{len(mesh.faces)} faces; pcd={len(pcd.points)} pts"
        )

    return object_meshes, object_pcds
