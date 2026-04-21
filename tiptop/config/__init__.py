import json
import os
from pathlib import Path

import numpy as np
from jaxtyping import Float
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.transform import Rotation

config_dir = Path(__file__).parent
config_assets_dir = config_dir / "assets"
tiptop_config_path = config_dir / "tiptop.yml"
calib_info_path = config_assets_dir / "calibration_info.json"

_cached_cfg = None  # Cache for lazy loading


def tiptop_cfg(force_reload: bool = False) -> DictConfig:
    """Load TiPToP config from file."""
    global _cached_cfg
    if _cached_cfg is None or force_reload:
        _cached_cfg = OmegaConf.load(tiptop_config_path)
        # Merge CLI overrides from sys.argv
        cli = OmegaConf.from_cli()
        _cached_cfg = OmegaConf.merge(_cached_cfg, cli)
    return _cached_cfg


def set_tiptop_cfg_from_file(cfg_path: Path) -> DictConfig:
    """Override the cached TiPToP config with one loaded from a specific file.

    Used by reruns to restore the config that was active during the original run,
    so robot/camera/perception settings stay consistent with the saved data.
    CLI overrides from sys.argv are re-applied on top.
    """
    global _cached_cfg
    _cached_cfg = OmegaConf.load(cfg_path)
    cli = OmegaConf.from_cli()
    _cached_cfg = OmegaConf.merge(_cached_cfg, cli)
    return _cached_cfg


def load_calibration_info():
    if not os.path.exists(calib_info_path):
        raise FileNotFoundError(f"{calib_info_path} not found.")
    with open(calib_info_path, "r") as f:
        calibration_info = json.load(f)
    return calibration_info


def load_calibration(cam_key: str) -> Float[np.ndarray, "4 4"]:
    """Load camera calibration 4x4 transform for a given camera serial."""
    calibration_dict = load_calibration_info()
    if cam_key not in calibration_dict:
        raise ValueError(f"{cam_key} not found in {calib_info_path}")

    pose_vec = calibration_dict[cam_key]["pose"]
    xyz, rpy = pose_vec[:3], pose_vec[3:]
    cam2frame = np.eye(4)
    cam2frame[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    cam2frame[:3, 3] = xyz
    return cam2frame


def update_calibration_info(cam_key: str, pose: np.ndarray):
    """Update calibration info with new camera pose.

    Args:
        cam_key: Camera identifier (e.g., "16779706_left")
        pose: 6DOF pose vector [x, y, z, roll, pitch, yaw]
    """
    import time

    # Load existing calibration info or create empty dict
    if os.path.exists(calib_info_path):
        calibration_dict = load_calibration_info()
    else:
        calibration_dict = {}

    # Update with new pose and timestamp
    calibration_dict[cam_key] = {
        "pose": pose.tolist() if isinstance(pose, np.ndarray) else list(pose),
        "timestamp": time.time(),
    }

    # Write back to file
    with open(calib_info_path, "w") as f:
        json.dump(calibration_dict, f, indent=2)

    print(f"Updated calibration for {cam_key}")
