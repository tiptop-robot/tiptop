from typing import Awaitable, Callable, Protocol

import aiohttp
import numpy as np

from tiptop.config import tiptop_cfg
from tiptop.perception.cameras.frame import Frame
from tiptop.perception.cameras.rs_camera import (
    RealsenseCamera,
    RealsenseFrame,
    RealsenseIntrinsics,
    rs_infer_depth_async,
)
from tiptop.perception.cameras.zed_camera import (
    ZedCamera,
    ZedFrame,
    ZedIntrinsics,
    zed_infer_depth_async,
)

# Callable that takes an aiohttp session and a camera frame and returns a float depth map (H, W) in metres.
DepthEstimator = Callable[[aiohttp.ClientSession, Frame], Awaitable[np.ndarray]]


class Camera(Protocol):
    serial: str

    def read_camera(self) -> Frame: ...
    def close(self) -> None: ...


def get_depth_estimator(cam: Camera) -> DepthEstimator:
    """Get the appropriate FoundationStereo depth estimator for the given camera. Call once per camera."""
    if isinstance(cam, ZedCamera):
        intrinsics = cam.get_intrinsics()

        async def _zed_estimate(session: aiohttp.ClientSession, f: Frame) -> np.ndarray:
            return await zed_infer_depth_async(session, f, intrinsics)  # type: ignore[arg-type]

        return _zed_estimate
    elif isinstance(cam, RealsenseCamera):
        intrinsics = cam.get_intrinsics()

        async def _rs_estimate(session: aiohttp.ClientSession, f: Frame) -> np.ndarray:
            return await rs_infer_depth_async(session, f, intrinsics)  # type: ignore[arg-type]

        return _rs_estimate
    else:
        raise ValueError(f"No depth estimator available for camera type: {type(cam).__name__}")


def _get_zed_camera(cam_cfg, depth: bool = False, pointcloud: bool = False) -> ZedCamera:
    """Create a ZedCamera from config."""
    serial = str(cam_cfg.serial)
    flip = cam_cfg.get("flip", False)
    resolution = cam_cfg.get("resolution", "HD720")
    fps = cam_cfg.get("fps", 60)
    return ZedCamera(serial, resolution=resolution, fps=fps, flip=flip, depth=depth, pointcloud=pointcloud)


def get_hand_camera(depth: bool = False) -> Camera:
    """Get the hand camera by serial number."""
    cfg = tiptop_cfg()
    cam_cfg = cfg.cameras.hand
    cam_type = cam_cfg.type
    if cam_type == "zed":
        return _get_zed_camera(cam_cfg, depth=depth)
    elif cam_type == "realsense":
        return RealsenseCamera(str(cam_cfg.serial), enable_depth=depth)
    else:
        raise ValueError(f"Unknown camera type: {cam_type}")


def get_external_camera() -> Camera:
    """Get the external camera by serial number."""
    cfg = tiptop_cfg()
    cam_cfg = cfg.cameras.external
    cam_type = cam_cfg.type
    if cam_type == "zed":
        return _get_zed_camera(cam_cfg)
    elif cam_type == "realsense":
        return RealsenseCamera(str(cam_cfg.serial))
    else:
        raise ValueError(f"Unknown camera type: {cam_type}")
