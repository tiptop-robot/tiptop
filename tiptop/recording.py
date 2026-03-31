import json
import logging
import shutil
import subprocess
import threading
import time
from contextlib import contextmanager
from functools import cache
from pathlib import Path
from typing import Generator

import cv2
import dill
import numpy as np
import open3d as o3d
import torch
from jaxtyping import Bool, Float, UInt8
from PIL import Image

from tiptop.config import tiptop_config_path
from tiptop.perception.cameras.zed_camera import ZedCamera, convert_svo_to_mp4
from tiptop.perception.utils import get_o3d_pcd
from tiptop.perception.visualization import visualize_detections, visualize_masks

_log = logging.getLogger(__name__)


@cache
def _get_git_root() -> Path:
    """Return the repository root."""
    return Path(subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=Path(__file__).parent,
        stderr=subprocess.DEVNULL,
        text=True,
    ).strip())


def _collect_git_info() -> dict:
    """Return git commit hash and dirty status for metadata.

    pixi.lock is excluded from the dirty check as it's not relevant for debugging
    """
    try:
        root = _get_git_root()
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        porcelain = subprocess.check_output(
            ["git", "status", "--porcelain", "--", ".", ":(exclude)pixi.lock"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        dirty = bool(porcelain.strip())
        return {"commit": commit, "dirty": dirty, "porcelain": porcelain.strip() if dirty else None}
    except (FileNotFoundError, subprocess.CalledProcessError):
        _log.warning("Failed to collect git info", exc_info=True)
        return {"commit": None, "dirty": None, "porcelain": None}


def _get_git_diff() -> str | None:
    """Return the full git diff against HEAD, excluding pixi.lock, or None if unavailable or empty."""
    try:
        root = _get_git_root()
        diff = subprocess.check_output(
            ["git", "diff", "HEAD", "--", ".", ":(exclude)pixi.lock"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return diff if diff else None
    except (FileNotFoundError, subprocess.CalledProcessError):
        _log.warning("Failed to get git diff", exc_info=True)
        return None


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
        _log.info(f"Started recording camera {camera.serial} to {svo_path}")
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
            _log.info(f"Stopped recording camera {camera.serial}")

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
    save_dir: Path,
    gripper_mask: Bool[np.ndarray, "h w"] | None = None,
):
    """Save perception outputs to disk.

    Visualization files (rgb.png, bboxes_viz.png, masks_viz.png) are saved at the top
    level of save_dir for quick access. Raw data files go into save_dir/perception/.
    """
    start_time = time.perf_counter()
    save_dir.mkdir(parents=True, exist_ok=True)
    perception_dir = save_dir / "perception"
    perception_dir.mkdir(exist_ok=True)

    # Camera image
    cv2.imwrite(str(save_dir / "rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    # Intrinsics
    intrinsics_dict = {"intrinsics": intrinsics_matrix.tolist()}
    with open(perception_dir / "intrinsics.json", "w") as f:
        json.dump(intrinsics_dict, f, indent=2)

    # Convert depth from meters to millimeters for uint16 storage
    depth_mm = depth_map * 1000.0
    depth_mm = np.clip(depth_mm, 0, 65535)
    depth_uint16 = depth_mm.astype(np.uint16)
    cv2.imwrite(str(perception_dir / "depth.png"), depth_uint16)

    # Create point cloud and write
    pcd = get_o3d_pcd(xyz_map, rgb_map)
    o3d.io.write_point_cloud(str(perception_dir / "pointcloud.ply"), pcd)

    # Write bboxes
    rgb_pil = Image.fromarray(rgb)
    bbox_viz = visualize_detections(rgb_pil, bboxes, output_path=str(save_dir / "bboxes_viz.png"), show_plot=False)
    with open(perception_dir / "bboxes.json", "w") as f:
        json.dump(bboxes, f, indent=2)

    # Write masks
    masks_viz = visualize_masks(rgb_pil, masks, bboxes)
    cv2.imwrite(str(save_dir / "masks_viz.png"), cv2.cvtColor(masks_viz, cv2.COLOR_RGB2BGR))
    masks_bool = masks > 0.5
    np.savez_compressed(str(perception_dir / "masks.npz"), masks_bool)  # masks are sparse so can compress

    if gripper_mask is not None:
        gripper_mask_img = Image.fromarray(gripper_mask.astype(np.uint8) * 255)
        gripper_mask_img.save(str(perception_dir / "gripper_mask.png"))

    save_dur = time.perf_counter() - start_time
    _log.info(f"Saved perception outputs to {save_dir} in {save_dur:.2f}s")
    return bbox_viz, masks_viz


def save_run_outputs(save_dir: Path, env, grasps: dict) -> None:
    """Save cuTAMP environment, grasps, and run artifacts (config) to disk."""
    # Save cutamp environment and grasps
    perception_dir = save_dir / "perception"
    perception_dir.mkdir(parents=True, exist_ok=True)
    with open(perception_dir / "cutamp_env.pkl", "wb") as f:
        dill.dump(env, f)
    _log.info(f"Saved cutamp env to {perception_dir}/cutamp_env.pkl")

    torch.save(grasps, perception_dir / "grasps.pt")
    _log.info(f"Saved grasps to {perception_dir}/grasps.pt")

    # tiptop config for reproducibility
    shutil.copy2(tiptop_config_path, save_dir / "tiptop.yml")
    _log.info(f"Saved tiptop config to {save_dir}/tiptop.yml")


def save_run_metadata(
    save_dir: Path,
    timestamp: str,
    task_instruction: str,
    q_at_capture: np.ndarray,
    world_from_cam: np.ndarray,
    perception_duration: float | None,
    grounded_atoms: list[dict] | None,
    planning_success: bool | None,
    planning_failure_reason: str | None,
    planning_duration: float | None,
) -> None:
    """Save structured run metadata to metadata.json."""
    git_info = _collect_git_info()
    if git_info["dirty"]:
        diff = _get_git_diff()
        if diff:
            (save_dir / "git.diff").write_text(diff, encoding="utf-8")

    metadata = {
        "task_instruction": task_instruction,
        "timestamp": timestamp,
        "observation": {
            "q_at_capture": q_at_capture,
            "world_from_cam": world_from_cam.tolist(),
        },
        "perception": {
            "grounded_atoms": grounded_atoms,
            "duration": round(perception_duration, 3) if perception_duration is not None else None,
        },
        "planning": {
            "success": planning_success,
            "failure_reason": planning_failure_reason,
            "duration": round(planning_duration, 3) if planning_duration is not None else None,
        },
        "version": "1.0.0",
        "git": git_info,
    }
    with open(save_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    _log.info(f"Saved run metadata to {save_dir}/metadata.json")


