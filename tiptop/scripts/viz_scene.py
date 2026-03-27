import logging
import os.path
from typing import Literal

import cv2

from tiptop.perception.cameras import ZedCamera, get_external_camera, get_hand_camera
from tiptop.utils import setup_logging

_log = logging.getLogger(__name__)


def viz_scene(
    cam_type: Literal["hand", "external"] = "external",
    ref_scene: str | None = None,
    alpha: float = 0.5,
):
    """
    Visualize scene with the camera, and optionally blend in the reference scene image to allow easier resetting of the
    scene.

    Args:
        cam_type: Camera type to use for visualization.
        ref_scene: Path to reference scene image for blending.
        alpha: Blending weight for reference image (0.0 = current only, 1.0 = reference only).
    """
    setup_logging()

    # Load reference scene image if specified
    ref_image = None
    if ref_scene:
        if not os.path.exists(ref_scene):
            raise FileNotFoundError(f"Reference scene not found: {ref_scene}")
        ref_image = cv2.imread(ref_scene)
        _log.info(f"Loaded reference image from {ref_scene}")

    # Setup camera
    if cam_type == "hand":
        cam = get_hand_camera()
    elif cam_type == "external":
        cam = get_external_camera()
    else:
        raise ValueError(f"Unknown camera type: {cam_type}")

    # Grab a frame to determine height and width
    frame = cam.read_camera()
    cam_resolution = frame.bgr.shape[:2]
    height, width = cam_resolution

    # Resize reference image
    if ref_image is not None and cam_resolution != ref_image.shape[:2]:
        ref_resolution = ref_image.shape[:2]
        ref_image = cv2.resize(ref_image, (width, height))
        _log.info(f"Resized reference image from {ref_resolution} to {cam_resolution}")

    window_name = "Visualize Scene"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)
    _log.info(f"Starting live view... Press 'q' to quit.")

    try:
        while True:
            frame = cam.read_camera()
            bgr = frame.bgr
            viz_frame = bgr.copy()

            if ref_image is not None:
                viz_frame = cv2.addWeighted(viz_frame, 1 - alpha, ref_image, alpha, 0.0)

            cv2.putText(viz_frame, "Press 'q' to quit", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow(window_name, viz_frame)

            # Handle keyboard input
            try:
                key = cv2.waitKey(1) & 0xFF
            except cv2.error:
                key = 0xFF

            if key == ord("q"):
                _log.info("Quitting...")
                break
    except KeyboardInterrupt:
        _log.debug("Caught keyboard interrupt")
    finally:
        cv2.destroyAllWindows()


def viz_scene_entrypoint():
    import tyro

    tyro.cli(viz_scene)


if __name__ == "__main__":
    viz_scene_entrypoint()
