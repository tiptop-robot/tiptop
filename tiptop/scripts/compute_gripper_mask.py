"""Compute gripper mask using Gemini detection + SAM segmentation."""

import logging

import cv2
import numpy as np
from tiptop.perception.gemini import gemini_client, load_json
from PIL import Image
from scipy.ndimage import binary_dilation, binary_fill_holes

from tiptop.perception.cameras import get_hand_camera
from tiptop.perception.sam2 import sam2_segment_objects
from tiptop.utils import gripper_mask_path, setup_logging

_log = logging.getLogger(__name__)

_prompt = """
Detect the robot gripper in the image. Include the whole left and right gripper fingers and gripper body in a single 
detection. Return bounding boxes as a JSON array with labels. Never return masks or code fencing. Limit to 1 object, 
which is the gripper.

The format should be as follows: [{"box_2d": [ymin, xmin, ymax, xmax], "label": <label for the object>}] normalized to 
0-1000. The values in box_2d must only be integers""".strip()


def compute_gripper_mask(gemini_model: str = "gemini-robotics-er-1.5-preview", dilation_iters: int = 8):
    """
    Compute gripper mask using Gemini detection and SAM segmentation.

    Args:
        gemini_model: Gemini model to use for gripper detection.
        dilation_iters: Number of binary dilation iterations to apply to the mask.
    """
    setup_logging()

    # Setup hand camera and read a frame
    cam = get_hand_camera()
    frame = cam.read_camera()
    rgb = frame.rgb
    rgb_pil = Image.fromarray(rgb)

    # Query Gemini to get gripper bounding box
    client = gemini_client()
    response = client.models.generate_content(model=gemini_model, contents=[rgb_pil, _prompt])
    bboxes: list = load_json(response.text)
    if len(bboxes) == 0:
        raise RuntimeError("No gripper detected! Try adjusting the camera view or prompt.")
    elif len(bboxes) > 1:
        _log.warning(f"Found multiple detections! Using the first one, make sure to validate the gripper mask")

    gripper_bbox = bboxes[0]
    _log.info(f"Using detection: {gripper_bbox['label']}")

    # Now use SAM to get segmentation mask
    masks = sam2_segment_objects(rgb_pil, [gripper_bbox])
    _log.debug(f"SAM masks shape: {masks.shape}")
    gripper_mask = masks[0].squeeze().astype(bool)

    # Post-process the mask by filling holes then dilating, as we get edge noise from depth sensor or prediction
    gripper_mask = binary_fill_holes(gripper_mask)
    gripper_mask = binary_dilation(gripper_mask, iterations=dilation_iters)

    # Visualize the new gripper mask
    overlay = rgb.copy()
    overlay[gripper_mask] = overlay[gripper_mask] * 0.7 + np.array([255, 0, 0]) * 0.3
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    overlay_bgr = cv2.putText(
        overlay_bgr,
        "Verify the gripper mask. Save? Press 'y' or 'n'.",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(0, 255, 0),
        thickness=2,
    )

    window_name = "Gripper Mask"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, overlay_bgr.shape[1], overlay_bgr.shape[0])
    save_mask = False
    while True:
        cv2.imshow(window_name, overlay_bgr)
        key = cv2.waitKey(1000)
        if key == ord("n"):
            _log.info("Key 'n' detected, not saving mask")
            break
        elif key == ord("y"):
            save_mask = True
            break
        else:
            _log.debug("Must press 's' or 'q'")

    if save_mask:
        # Save as binary PNG
        mask_image = Image.fromarray((gripper_mask.astype(np.uint8) * 255))
        mask_image.save(gripper_mask_path)
        _log.info(f"Saved gripper mask to {gripper_mask_path}")


def compute_gripper_mask_entrypoint():
    import tyro

    tyro.cli(compute_gripper_mask)


if __name__ == "__main__":
    compute_gripper_mask_entrypoint()
