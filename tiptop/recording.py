import json
import logging
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import cv2
import dill
import numpy as np
import open3d as o3d
import torch
from jaxtyping import Float, UInt8
from PIL import Image

from tiptop.perception.cameras.zed_camera import ZedCamera, convert_svo_to_mp4
from tiptop.perception.utils import get_o3d_pcd
from tiptop.perception.visualization import visualize_detections, visualize_masks

_log = logging.getLogger(__name__)


@contextmanager
def record_cameras(recordings: list[tuple[ZedCamera, Path, Path | None]]) -> Generator[None, None, None]:
    """Context manager for recording multiple ZED cameras simultaneously.

    All cameras stop collecting frames at the same time on exit, then MP4
    conversion runs sequentially afterwards.

    Args:
        recordings: List of (camera, svo_path, mp4_path) tuples
    """
    stop_events: list[threading.Event] = []
    threads: list[threading.Thread] = []

    for camera, svo_path, _ in recordings:
        stop_event = threading.Event()

        def recording_loop(cam=camera, event=stop_event):
            while not event.is_set():
                try:
                    cam.read_camera()
                except Exception as e:
                    _log.error(f"Error grabbing frame during recording: {e}")
                    break

        camera.start_recording(str(svo_path))
        thread = threading.Thread(target=recording_loop)
        thread.start()
        _log.info(f"Started recording camera {get_serial(camera)} to {svo_path}")
        stop_events.append(stop_event)
        threads.append(thread)

    try:
        yield
    finally:
        # Signal all cameras to stop simultaneously so recordings are the same length
        for event in stop_events:
            event.set()
        for (camera, svo_path, _), thread in zip(recordings, threads):
            thread.join(timeout=3.0)
            camera.stop_recording()
            _log.info(f"Stopped recording camera {get_serial(camera)}")

        # Convert to MP4 after all cameras have stopped
        for camera, svo_path, mp4_path in recordings:
            if mp4_path is None:
                continue
            actual_svo_path = svo_path
            if not svo_path.exists():
                svo2_path = svo_path.with_suffix(".svo2")
                if svo2_path.exists():
                    _log.debug(f"SVO actually written by ZED SDK to {svo2_path.name}")
                    actual_svo_path = svo2_path
            if actual_svo_path.exists():
                convert_svo_to_mp4(actual_svo_path, mp4_path)
            else:
                raise FileNotFoundError(
                    f"Recording failed: SVO file not found at {svo_path} or {svo_path.with_suffix('.svo2')}"
                )


def save_perception_outputs(
    rgb: UInt8[np.ndarray, "h w 3"],
    intrinsics_matrix: Float[np.ndarray, "3 3"],
    depth_map: Float[np.ndarray, "h w"],
    xyz_map: Float[np.ndarray, "n 3"],
    rgb_map: Float[np.ndarray, "n 3"],
    bboxes: list[dict],
    masks: np.ndarray,
    task_instruction: str,
    save_dir: Path,
):
    """Save perception outputs to disk."""
    start_time = time.perf_counter()
    save_dir.mkdir(parents=True, exist_ok=True)

    # Task instruction
    with open(save_dir / "task_instruction.txt", "w", encoding="utf-8") as f:
        f.write(task_instruction)

    # Camera image
    cv2.imwrite(str(save_dir / "rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    # Intrinsics
    intrinsics_dict = {"intrinsics": intrinsics_matrix.tolist()}
    with open(save_dir / "intrinsics.json", "w") as f:
        json.dump(intrinsics_dict, f, indent=2)

    # Convert depth from meters to millimeters for uint16 storage
    depth_mm = depth_map * 1000.0
    depth_mm = np.clip(depth_mm, 0, 65535)
    depth_uint16 = depth_mm.astype(np.uint16)
    cv2.imwrite(str(save_dir / "depth.png"), depth_uint16)

    # Create point cloud and write
    pcd = get_o3d_pcd(xyz_map, rgb_map)
    o3d.io.write_point_cloud(str(save_dir / "pointcloud.ply"), pcd)

    # Write bboxes
    rgb_pil = Image.fromarray(rgb)
    bbox_viz = visualize_detections(rgb_pil, bboxes, output_path=str(save_dir / "bboxes_viz.png"), show_plot=False)
    with open(save_dir / "bboxes.json", "w") as f:
        json.dump(bboxes, f, indent=2)

    # Write masks
    masks_viz = visualize_masks(rgb_pil, masks, bboxes)
    cv2.imwrite(str(save_dir / "masks_viz.png"), cv2.cvtColor(masks_viz, cv2.COLOR_RGB2BGR))
    masks_bool = masks > 0.5
    np.savez_compressed(str(save_dir / "masks.npz"), masks_bool)  # masks are sparse so can compress

    save_dur = time.perf_counter() - start_time
    _log.info(f"Saved perception outputs to {save_dir} in {save_dur:.2f}s")
    return bbox_viz, masks_viz


def save_run_outputs(save_dir: Path, env, grasps: dict) -> None:
    """Save cuTAMP environment and grasps to disk."""
    with open(save_dir / "cutamp_env.pkl", "wb") as f:
        dill.dump(env, f)
    _log.info(f"Saved cutamp env to {save_dir}/cutamp_env.pkl")

    torch.save(grasps, save_dir / "grasps.pt")
    _log.info(f"Saved grasps to {save_dir}/grasps.pt")
