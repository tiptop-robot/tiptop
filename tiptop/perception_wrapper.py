import asyncio
import logging
import time

import aiohttp
import numpy as np
from jaxtyping import Bool, Float, UInt8
from PIL import Image

from tiptop.config import tiptop_cfg
from tiptop.perception.cameras import (
    DepthEstimator,
    Frame,
)
from tiptop.perception.m2t2 import generate_grasps_async
from tiptop.perception.utils import depth_to_xyz, get_o3d_pcd

_log = logging.getLogger(__name__)


async def detect_and_segment(rgb: UInt8[np.ndarray, "h w 3"], task_instruction: str) -> dict:
    """Run VLM detection and SAM2 segmentation pipeline."""
    rgb_pil = Image.fromarray(rgb)
    rgb_pil_resized = rgb_pil.resize((800, int(800 * rgb_pil.size[1] / rgb_pil.size[0])), Image.Resampling.LANCZOS)
    _log.info(
        f"Resized image from {rgb_pil.size[1]}x{rgb_pil.size[0]} to {rgb_pil_resized.size[1]}x{rgb_pil_resized.size[0]}"
    )

    async def _detect():
        from tiptop.perception.gemini import detect_and_translate_async

        _log.info(f"Starting Gemini object detection")
        _st = time.perf_counter()
        _bboxes, _grounded_atoms = await detect_and_translate_async(rgb_pil_resized, task_instruction)
        _dur = time.perf_counter() - _st
        _log.info(f"Gemini detection took {_dur:.2f}s ({len(_bboxes)} objects, {len(_grounded_atoms)} atoms)")
        return _bboxes, _grounded_atoms

    def _segment(_bboxes: list[dict]):
        from tiptop.perception.sam2 import sam2_segment_objects

        _log.info(f"Starting SAM2 object segmentation with Gemini masks")
        _st = time.perf_counter()
        # TODO: async version of this?
        _masks = sam2_segment_objects(rgb_pil, _bboxes)
        _dur = time.perf_counter() - _st
        _log.info(f"SAM2 segmentation took {_dur:.2f}s ({len(_masks)} masks)")
        return _masks

    bboxes, grounded_atoms = await _detect()

    # Sanitize labels: replace spaces with underscores for downstream compatibility
    for bbox in bboxes:
        bbox["label"] = bbox["label"].replace(" ", "_")
    for atom in grounded_atoms:
        atom["args"] = [arg.replace(" ", "_") for arg in atom["args"]]

    masks = await asyncio.to_thread(_segment, bboxes)

    return {"bboxes": bboxes, "masks": masks, "grounded_atoms": grounded_atoms}


async def predict_depth_and_grasps(
    session: aiohttp.ClientSession,
    frame: Frame,
    world_from_cam: Float[np.ndarray, "4 4"],
    downsample_voxel_size: float,
    depth_estimator: DepthEstimator | None = None,
    gripper_mask: Bool[np.ndarray, "h w 3"] | None = None,
) -> dict:
    """Predict depth map using FoundationStereo and grasps using M2T2. Uses depth_estimator if provided, otherwise uses frame.depth."""
    cfg = tiptop_cfg()

    # Get depth map — use estimator (e.g. FoundationStereo) or fall back to onboard sensor depth
    if depth_estimator is not None:
        depth_map = await depth_estimator(session, frame)
    else:
        if frame.depth is None:
            raise RuntimeError(
                "No depth available: depth_estimator is None and frame.depth is not set. "
                "Either provide a depth_estimator or ensure the camera captures hardware depth."
            )
        _log.warning("No depth_estimator provided, falling back to hardware depth")
        depth_map = frame.depth

    # Convert to point cloud in world frame
    K = frame.intrinsics
    xyz_map = depth_to_xyz(depth_map, K)
    xyz_map = xyz_map @ world_from_cam[:3, :3].T + world_from_cam[:3, 3]
    if gripper_mask is not None:
        xyz_map[gripper_mask] = 0.0
    rgb_map = frame.rgb.astype(np.float32) / 255.0  # make it float with [0, 1]

    # Create open3d point cloud and downsample
    pcd = await asyncio.to_thread(
        get_o3d_pcd,
        xyz_map,
        rgb_map,
        downsample_voxel_size,
    )
    xyz_downsampled = np.asarray(pcd.points)
    rgb_downsampled = np.asarray(pcd.colors)

    # Predict grasps using M2T2
    grasps = await generate_grasps_async(
        session,
        cfg.perception.m2t2.url,
        scene_xyz=xyz_downsampled,
        scene_rgb=rgb_downsampled,
        apply_bounds=cfg.perception.m2t2.apply_bounds,
    )

    return {
        "depth_map": depth_map,
        # (h, w, 3) for xyz, rgb, and valid mask map
        "xyz_map": xyz_map,
        "rgb_map": rgb_map,
        # (n, 3) for downsampled point cloud
        "xyz_downsampled": xyz_downsampled,
        "rgb_downsampled": rgb_downsampled,
        "pcd_downsampled": pcd,
        # grasps
        "grasps": grasps,
    }
