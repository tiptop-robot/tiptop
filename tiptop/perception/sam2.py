"""SAM2 segmentation — local predictor and remote HTTP client."""

import base64
import io
import logging
import os
from functools import cache

import numpy as np
import requests
import torch.cuda
from PIL import Image
from jaxtyping import Float
from tqdm import tqdm

from tiptop.config import tiptop_cfg
from tiptop.utils import get_tiptop_cache_dir

_log = logging.getLogger(__name__)

_SAM2_BASE_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824"


def download_sam2_checkpoint(model_name: str = "sam2.1_hiera_large.pt") -> str:
    """Download SAM2 checkpoint if it doesn't already exist."""
    model_url = os.path.join(_SAM2_BASE_URL, model_name)
    dest_path = get_tiptop_cache_dir() / model_name

    if dest_path.exists():
        _log.debug(f"SAM2 checkpoint {model_name} already exists at {dest_path}.")
        return dest_path

    _log.info(f"Downloading SAM2 checkpoint from {model_url} to {dest_path}.")
    response = requests.get(model_url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 KB

    with (
        open(dest_path, "wb") as file,
        tqdm(total=total_size, unit="iB", unit_scale=True, desc=model_name) as progress_bar,
    ):
        for data in response.iter_content(block_size):
            file.write(data)
            progress_bar.update(len(data))

    _log.info(f"SAM2 checkpoint {model_name} downloaded successfully.")
    return dest_path


@cache
def _sam2_predictor(checkpoint: str, device: str):
    """Load and cache the SAM2 image predictor."""
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    config = os.environ.get("SAM2_CONFIG", "configs/sam2.1/sam2.1_hiera_l.yaml")
    _log.info(f"Loading SAM2 with checkpoint={checkpoint}, config={config}, device={device}")
    predictor = SAM2ImagePredictor(build_sam2(config, checkpoint, device=device))
    _log.info("Successfully loaded SAM2")
    return predictor


def _segment_local(image: Image.Image, boxes: np.ndarray, checkpoint: str) -> tuple[np.ndarray, np.ndarray]:
    """Run SAM2 segmentation locally."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = _sam2_predictor(checkpoint, device)
    predictor.set_image(image)
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=boxes,
        multimask_output=False,
    )
    return masks, scores


def _segment_remote(image: Image.Image, boxes: np.ndarray, server_url: str) -> tuple[np.ndarray, np.ndarray]:
    """Run SAM2 segmentation via remote server."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    payload = {"image_base64": base64.b64encode(buffer.getvalue()).decode(), "boxes": boxes.tolist()}

    try:
        response = requests.post(f"{server_url}/segment", json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()

        masks = np.array([
            [np.load(io.BytesIO(base64.b64decode(m))) for m in mask_batch]
            for mask_batch in result["masks"]
        ])
        return masks, np.array(result["scores"])

    except Exception as e:
        _log.error(f"Remote SAM2 segmentation failed: {e}")
        raise e


def sam2_client() -> None:
    """Warm up SAM2: pre-load the local predictor or verify the remote server is reachable."""
    cfg = tiptop_cfg()
    mode = cfg.perception.sam.mode
    if mode == "local":
        checkpoint = str(download_sam2_checkpoint())
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _sam2_predictor(checkpoint, device)
    elif mode == "remote":
        server_url = cfg.perception.sam.url
        try:
            requests.get(f"{server_url}/health", timeout=5).raise_for_status()
            _log.info("Successfully connected to SAM2 server")
        except Exception as e:
            _log.warning(f"Failed to connect to SAM2 server: {e}")
    else:
        raise ValueError(f"Invalid SAM2 mode: {mode}. Must be 'local' or 'remote'")


def sam2_segment_objects(
    rgb_pil: Image.Image,
    detection_results: list[dict],
) -> Float[np.ndarray, "n 1 h w"]:
    """Segment detection results from Gemini with SAM2.

    Args:
        rgb_pil: PIL Image to segment.
        detection_results: List of detection dicts from Gemini, each with a 'box_2d' key
                           in [ymin, xmin, ymax, xmax] format normalized to 0-1000.

    Returns:
        Segmentation masks of shape (N, 1, H, W).
    """
    cfg = tiptop_cfg()
    mode = cfg.perception.sam.mode

    # Convert Gemini bbox format [ymin, xmin, ymax, xmax] (0-1000) to SAM2 [x0, y0, x1, y1] (pixels)
    img_height, img_width = rgb_pil.height, rgb_pil.width
    boxes = np.array([
        [
            (xmin / 1000.0) * img_width,
            (ymin / 1000.0) * img_height,
            (xmax / 1000.0) * img_width,
            (ymax / 1000.0) * img_height,
        ]
        for detection in detection_results
        if len(box_2d := detection.get("box_2d", [])) == 4
        for ymin, xmin, ymax, xmax in [box_2d]
    ])

    _log.info(f"Segmenting {len(boxes)} objects using SAM2 in {mode} mode")
    if mode == "local":
        masks, scores = _segment_local(rgb_pil, boxes, str(download_sam2_checkpoint()))
    else:
        masks, scores = _segment_remote(rgb_pil, boxes, cfg.perception.sam.url)
    _log.info(f"Generated {len(masks)} segmentation masks, shape: {masks.shape}")

    if masks.ndim == 3:
        masks = masks[None]
    return masks
