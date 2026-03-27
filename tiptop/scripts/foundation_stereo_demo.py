import asyncio
import logging

import aiohttp
import rerun as rr

from tiptop.perception.cameras import get_depth_estimator, get_hand_camera
from tiptop.perception.utils import depth_to_xyz
from tiptop.utils import setup_logging

_log = logging.getLogger(__name__)


async def _run_demo():
    # Setup camera
    cam = get_hand_camera()
    depth_estimator = get_depth_estimator(cam)

    # Read frame
    frame = cam.read_camera()
    K = frame.intrinsics
    rr.log("cam", rr.Pinhole(image_from_camera=K))
    rgb = frame.rgb
    rr.log("cam/rgb", rr.Image(rgb))

    # Predict depth with FoundationStereo
    _log.info("Running FoundationStereo to predict depth")
    async with aiohttp.ClientSession() as session:
        pred_depth = await depth_estimator(session, frame)
    rr.log("cam/depth", rr.DepthImage(pred_depth, meter=1.0))

    # Project to point cloud and set gripper mask to zeros
    xyz_map = depth_to_xyz(pred_depth, K)  # in cam frame
    rgb_map = rgb.copy()
    rr.log("pcd", rr.Points3D(positions=xyz_map.reshape(-1, 3), colors=rgb_map.reshape(-1, 3)))
    _log.info("FoundationStereo demo complete!")


def foundation_stereo_demo(rr_spawn: bool = True):
    """Demo for FoundationStereo."""
    setup_logging()
    rr.init("foundation_stereo_demo", spawn=rr_spawn)
    asyncio.run(_run_demo())


if __name__ == "__main__":
    foundation_stereo_demo()
