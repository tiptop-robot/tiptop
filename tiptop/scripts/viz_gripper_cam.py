"""Visualize the gripper (hand camera) in cv2."""

import logging

import cv2
import numpy as np

from tiptop.perception.cameras import get_hand_camera
from tiptop.utils import load_gripper_mask, setup_logging

_log = logging.getLogger(__name__)


def viz_gripper_cam(show_gripper_mask: bool = True):
    """
    Visualize the gripper (hand camera) camera feed.

    Args:
        show_gripper_mask: Whether to overlay the gripper mask.
    """
    setup_logging()
    cam = get_hand_camera()

    # Load gripper mask for overlaying
    gripper_mask = load_gripper_mask() if show_gripper_mask else None

    frame = cam.read_camera()
    height, width, *_ = frame.bgr.shape

    # Create cv2 window
    window_name = "Hand Camera"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)
    _log.info("Press 'q' to exit")

    try:
        while True:
            # Read camera frame
            frame = cam.read_camera()
            bgr = frame.bgr
            viz_frame = bgr.copy()
            cv2.putText(viz_frame, "Press 'q' to quit", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if show_gripper_mask:
                # Optionally overlay gripper mask in red
                viz_frame[gripper_mask] = viz_frame[gripper_mask] * 0.7 + np.array([0, 0, 255]) * 0.3
            cv2.imshow(window_name, viz_frame)
            try:
                key = cv2.waitKey(1) & 0xFF
            except cv2.error:
                key = 0xFF  # continue without processing if fail

            if key == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()


def viz_gripper_cam_entrypoint():
    import tyro

    tyro.cli(viz_gripper_cam)


if __name__ == "__main__":
    viz_gripper_cam_entrypoint()
