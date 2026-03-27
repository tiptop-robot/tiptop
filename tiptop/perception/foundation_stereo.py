import io
import logging
import os.path
import time

import aiohttp
import cv2
import numpy as np
import requests
from jaxtyping import Float, UInt8

from tiptop.utils import ServerHealthCheckError

_log = logging.getLogger(__name__)


def _encode_images_to_png(
    left_rgb: UInt8[np.ndarray, "h w 3"], right_rgb: UInt8[np.ndarray, "h w 3"]
) -> tuple[bytes, bytes]:
    if left_rgb.shape != right_rgb.shape:
        raise ValueError(f"Expected shape of left_rgb {left_rgb.shape} to match right_rgb {right_rgb.shape}")
    if not (left_rgb.dtype == right_rgb.dtype == np.uint8):
        raise ValueError(f"Expected uint8 dtype for left_rgb ({left_rgb.dtype}) and right_rgb ({right_rgb.dtype})")

    # Since we're encoding with cv2 need to convert to BGR first
    _, left_bytes = cv2.imencode(".png", cv2.cvtColor(left_rgb, cv2.COLOR_RGB2BGR))
    _, right_bytes = cv2.imencode(".png", cv2.cvtColor(right_rgb, cv2.COLOR_RGB2BGR))
    return left_bytes.tobytes(), right_bytes.tobytes()


def _decode_depth_response(content: bytes) -> Float[np.ndarray, "h w"]:
    """Decode depth map from NPZ response content."""
    buffer = io.BytesIO(content)
    return np.load(buffer)["depth"]


def infer_depth(
    server_url: str,
    left_rgb: UInt8[np.ndarray, "h w 3"],
    right_rgb: UInt8[np.ndarray, "h w 3"],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    baseline: float,
) -> Float[np.ndarray, "h w"]:
    """Predict depth given a stereo pair using FoundationStereo (synchronous version)."""
    start_time = time.perf_counter()
    left_bytes, right_bytes = _encode_images_to_png(left_rgb, right_rgb)
    files = {
        "left_image": ("left.png", left_bytes, "image/png"),
        "right_image": ("right.png", right_bytes, "image/png"),
    }
    data = {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "baseline": baseline, "scale": 1.0, "hiera": 0, "valid_iters": 32}

    infer_endpoint = os.path.join(server_url.rstrip("/"), "infer")
    _log.debug(f"Sending inference request to FoundationStereo server at {infer_endpoint}")
    response = requests.post(infer_endpoint, files=files, data=data)
    if response.status_code != 200:
        raise RuntimeError(
            f"FoundationStereo request failed with status code {response.status_code}. Response: {response.text}"
        )

    depth = _decode_depth_response(response.content)
    duration = time.perf_counter() - start_time
    _log.info(f"FoundationStereo depth map={depth.shape}, inference time={duration:.2f}s")
    return depth


async def infer_depth_async(
    session: aiohttp.ClientSession,
    server_url: str,
    left_rgb: UInt8[np.ndarray, "h w 3"],
    right_rgb: UInt8[np.ndarray, "h w 3"],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    baseline: float,
) -> Float[np.ndarray, "h w"]:
    """Predict depth given a stereo pair using FoundationStereo (async version)."""
    start_time = time.perf_counter()
    left_bytes, right_bytes = _encode_images_to_png(left_rgb, right_rgb)

    # Create FormData for multipart upload
    data = aiohttp.FormData()
    data.add_field("left_image", left_bytes, filename="left.png", content_type="image/png")
    data.add_field("right_image", right_bytes, filename="right.png", content_type="image/png")
    data.add_field("fx", str(fx))
    data.add_field("fy", str(fy))
    data.add_field("cx", str(cx))
    data.add_field("cy", str(cy))
    data.add_field("baseline", str(baseline))
    data.add_field("scale", "1.0")
    data.add_field("hiera", "0")
    data.add_field("valid_iters", "32")

    # Call the server
    infer_endpoint = os.path.join(server_url.rstrip("/"), "infer")
    _log.debug(f"Sending inference request to FoundationStereo server at {infer_endpoint}")

    async with session.post(infer_endpoint, data=data, timeout=aiohttp.ClientTimeout(total=10.0)) as response:
        if response.status != 200:
            text = await response.text()
            raise RuntimeError(f"FoundationStereo request failed with status code {response.status}. Response: {text}")
        content = await response.read()

    # Decode response
    depth = _decode_depth_response(content)

    duration = time.perf_counter() - start_time
    _log.info(f"FoundationStereo depth map={depth.shape}, inference time={duration:.2f}s")
    return depth


async def check_health_status(session: aiohttp.ClientSession, server_url: str):
    """Calls the FoundationStereo server health status endpoint."""
    endpoint = os.path.join(server_url.rstrip("/"), "health")
    try:
        async with session.get(endpoint, timeout=5.0) as response:
            response.raise_for_status()
            health_data = await response.json()
            status = health_data["status"]

            if status != "healthy":
                _log.error(f"FoundationStereo health check failed at {server_url}")
                raise ServerHealthCheckError(f"{server_url} returned status: {status}")

            _log.info(f"✓ FoundationStereo server is healthy")
    except aiohttp.ClientError as e:
        _log.error(f"Health check failed for FoundationStereo")
        raise ServerHealthCheckError(f"FoundationStereo is unreachable: {e}") from e
