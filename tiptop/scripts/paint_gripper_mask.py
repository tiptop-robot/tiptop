"""Manually paint gripper mask with interactive drawing tool."""

import logging

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_fill_holes

from tiptop.perception.cameras import get_hand_camera
from tiptop.utils import gripper_mask_path, load_gripper_mask, setup_logging

_log = logging.getLogger(__name__)


class MaskPainter:
    """Interactive mask painting tool using OpenCV."""

    def __init__(self, rgb: np.ndarray, brush_size: int = 20):
        """
        Initialize the mask painter.

        Args:
            rgb: RGB image (H, W, 3) to paint mask on.
            brush_size: Initial brush size in pixels.
        """
        self.rgb = rgb
        self.mask = np.zeros(rgb.shape[:2], dtype=bool)
        self.brush_size = brush_size
        self.drawing = False
        self.mode = "draw"  # 'draw' or 'erase'

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing/erasing."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            if self.mode == "draw":
                self._paint_at(x, y)
            else:
                self._erase_at(x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            if self.mode == "draw":
                self._paint_at(x, y)
            else:
                self._erase_at(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False

    def _paint_at(self, x: int, y: int):
        """Paint mask at the given pixel location."""
        h, w = self.mask.shape
        y1 = max(0, y - self.brush_size // 2)
        y2 = min(h, y + self.brush_size // 2)
        x1 = max(0, x - self.brush_size // 2)
        x2 = min(w, x + self.brush_size // 2)

        # Create circular brush
        yy, xx = np.ogrid[y1:y2, x1:x2]
        circle = (xx - x) ** 2 + (yy - y) ** 2 <= (self.brush_size // 2) ** 2
        self.mask[y1:y2, x1:x2][circle] = True

    def _erase_at(self, x: int, y: int):
        """Erase mask at the given pixel location."""
        h, w = self.mask.shape
        y1 = max(0, y - self.brush_size // 2)
        y2 = min(h, y + self.brush_size // 2)
        x1 = max(0, x - self.brush_size // 2)
        x2 = min(w, x + self.brush_size // 2)

        # Create circular brush
        yy, xx = np.ogrid[y1:y2, x1:x2]
        circle = (xx - x) ** 2 + (yy - y) ** 2 <= (self.brush_size // 2) ** 2
        self.mask[y1:y2, x1:x2][circle] = False

    def get_overlay(self) -> np.ndarray:
        """Get RGB image with mask overlay in red."""
        overlay = self.rgb.copy()
        overlay[self.mask] = overlay[self.mask] * 0.7 + np.array([255, 0, 0]) * 0.3
        return overlay

    def fill_holes(self):
        """Fill holes in the mask."""
        self.mask = binary_fill_holes(self.mask)
        _log.info("Filled holes in mask")

    def dilate(self, iterations: int = 8):
        """Dilate the mask."""
        self.mask = binary_dilation(self.mask, iterations=iterations)
        _log.info(f"Dilated mask with {iterations} iterations")

    def clear(self):
        """Clear the entire mask."""
        self.mask = np.zeros_like(self.mask)
        _log.info("Cleared mask")


def paint_gripper_mask(initial_brush_size: int = 20, dilation_iters: int = 8):
    """
    Interactively paint gripper mask on camera image.

    Controls:
    - Left Click + Drag: Paint or erase (depending on mode)
    - 'e': Toggle between draw and erase modes
    - '+' or '=': Increase brush size
    - '-' or '_': Decrease brush size
    - 'f': Fill holes in mask
    - 'd': Dilate mask
    - 'c': Clear mask
    - 'y': Save mask and exit
    - 'n' or 'q': Exit without saving

    Args:
        initial_brush_size: Initial brush size in pixels.
        dilation_iters: Number of dilation iterations when pressing 'd'.
    """
    setup_logging()

    # Setup hand camera and read a frame
    _log.info("Capturing image from hand camera...")
    cam = get_hand_camera()
    frame = cam.read_camera()
    rgb = frame.rgb

    # Create mask painter
    painter = MaskPainter(rgb, brush_size=initial_brush_size)

    # Load existing mask if it exists
    if gripper_mask_path.exists():
        _log.info(f"Loading existing mask from {gripper_mask_path}")
        existing_mask = load_gripper_mask()
        # Ensure mask dimensions match the current image
        if existing_mask.shape == painter.mask.shape:
            painter.mask = existing_mask
            _log.info("Existing mask loaded successfully")
        else:
            _log.warning(
                f"Existing mask shape {existing_mask.shape} doesn't match "
                f"current image shape {painter.mask.shape}. Starting with empty mask."
            )
    else:
        _log.info("No existing mask found. Starting with empty mask.")

    # Setup window
    window_name = "Paint Gripper Mask"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, painter.mouse_callback)

    _log.info("Interactive mask painting started")
    _log.info("Controls:")
    _log.info("  Left Click + Drag: Paint or erase")
    _log.info("  'e': Toggle draw/erase mode")
    _log.info("  '+' or '=': Increase brush size")
    _log.info("  '-' or '_': Decrease brush size")
    _log.info("  'f': Fill holes")
    _log.info("  'd': Dilate mask")
    _log.info("  'c': Clear mask")
    _log.info("  'y': Save and exit")
    _log.info("  'n' or 'q': Exit without saving")

    save_mask = False
    while True:
        # Get overlay and add instructions
        overlay = painter.get_overlay()
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

        # Add text instructions
        mode_text = "DRAW" if painter.mode == "draw" else "ERASE"
        mode_color = (0, 255, 0) if painter.mode == "draw" else (0, 0, 255)
        instructions = [
            (f"Mode: {mode_text} (press 'e' to toggle)", mode_color),
            (f"Brush Size: {painter.brush_size}px", (0, 255, 0)),
            ("Left Click+Drag: Paint/Erase", (0, 255, 0)),
            ("+/-: Size | f: Fill | d: Dilate | c: Clear", (0, 255, 0)),
            ("y: Save | n/q: Cancel", (0, 255, 0)),
        ]
        for i, (text, color) in enumerate(instructions):
            cv2.putText(
                overlay_bgr,
                text,
                (10, 30 + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.6,
                color=color,
                thickness=2,
            )

        # Resize window to match image
        h, w = overlay_bgr.shape[:2]
        cv2.resizeWindow(window_name, w, h)
        cv2.imshow(window_name, overlay_bgr)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF

        if key == ord("y"):
            save_mask = True
            _log.info("Saving mask...")
            break
        elif key in [ord("n"), ord("q")]:
            _log.info("Exiting without saving")
            break
        elif key == ord("e"):
            painter.mode = "erase" if painter.mode == "draw" else "draw"
            _log.info(f"Mode: {painter.mode.upper()}")
        elif key in [ord("+"), ord("=")]:
            painter.brush_size = min(100, painter.brush_size + 5)
            _log.info(f"Brush size: {painter.brush_size}px")
        elif key in [ord("-"), ord("_")]:
            painter.brush_size = max(5, painter.brush_size - 5)
            _log.info(f"Brush size: {painter.brush_size}px")
        elif key == ord("f"):
            painter.fill_holes()
        elif key == ord("d"):
            painter.dilate(iterations=dilation_iters)
        elif key == ord("c"):
            painter.clear()

    cv2.destroyAllWindows()

    if save_mask:
        # Save as binary PNG (same format as compute_gripper_mask.py)
        mask_image = Image.fromarray((painter.mask.astype(np.uint8) * 255))
        mask_image.save(gripper_mask_path)
        _log.info(f"Saved gripper mask to {gripper_mask_path}")
    else:
        _log.info("Mask not saved")


def paint_gripper_mask_entrypoint():
    import tyro

    tyro.cli(paint_gripper_mask)


if __name__ == "__main__":
    paint_gripper_mask_entrypoint()
