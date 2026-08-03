import logging
import os

import aiohttp
import msgpack
import msgpack_numpy
import numpy as np
from jaxtyping import Bool, Float, UInt8

from tiptop.utils import ServerHealthCheckError

_log = logging.getLogger(__name__)


async def generate_shape_async(
    session: aiohttp.ClientSession,
    server_url: str,
    rgb: UInt8[np.ndarray, "h w 3"],
    depth: Float[np.ndarray, "h w"],
    mask: Bool[np.ndarray, "h w"],
    intrinsics: Float[np.ndarray, "3 3"],
    seed: int = 42,
    target_faces: int | None = None,
    timeout: float = 180.0,
) -> dict:
    """Run RecGen single-view reconstruction for one object via HTTP.

    If target_faces is provided, the server-side quadric edge-collapse decimator
    runs after inference and the response carries the decimated mesh.

    Args:
        session: aiohttp session for the POST.
        server_url: RecGen server base URL.
        rgb: Camera-frame RGB image.
        depth: Camera-frame depth (meters).
        mask: Object mask (non-zero = object).
        intrinsics: Camera intrinsics.
        seed: RecGen seed for reproducible reconstruction.
        target_faces: If set, mesh is server-side decimated to this many faces.
        timeout: Per-request timeout in seconds.

    Returns:
        Dict with keys:
            "vertices":      (N, 3) float32 — mesh vertices in the camera frame.
            "faces":         (M, 3) int32 — triangle indices.
            "vertex_colors": (N, 4) uint8 — per-vertex RGBA (optional, may be absent).
            "pose_matrix":   (4, 4) float64 — object-to-camera pose.
            "pose_quat":     (7,) float64 — same pose as [tx, ty, tz, qx, qy, qz, qw].
    """
    payload = {
        "rgb": rgb,
        "depth": depth,
        "mask": mask,
        "intrinsics": intrinsics.astype(np.float64),
        "seed": int(seed),
    }
    if target_faces is not None:
        payload["target_faces"] = int(target_faces)
    body = msgpack.packb(payload, default=msgpack_numpy.encode, use_bin_type=True)
    endpoint = os.path.join(server_url.rstrip("/"), "generate")

    _log.debug(f"Sending inference request to RecGen server at {endpoint}")
    async with session.post(
        endpoint,
        data=body,
        headers={"Content-Type": "application/x-msgpack"},
        timeout=aiohttp.ClientTimeout(total=timeout),
    ) as response:
        response.raise_for_status()
        return msgpack.unpackb(await response.read(), object_hook=msgpack_numpy.decode, raw=False)


async def check_health_status(session: aiohttp.ClientSession, server_url: str):
    """Calls the RecGen server health status endpoint."""
    endpoint = os.path.join(server_url.rstrip("/"), "health")
    try:
        async with session.get(endpoint, timeout=aiohttp.ClientTimeout(total=5.0)) as response:
            response.raise_for_status()
            health_data = await response.json()
            status = health_data.get("status")

            if status != "ok":
                _log.error(f"RecGen health check failed at {server_url}")
                raise ServerHealthCheckError(f"{server_url} returned status: {status}")

            if not health_data.get("pipeline_loaded"):
                _log.warning(f"RecGen pipeline not loaded yet at {server_url}; first request will block")
            _log.info("✓ RecGen server is healthy")
    except aiohttp.ClientError as e:
        _log.error("Health check failed for RecGen")
        raise ServerHealthCheckError(f"RecGen is unreachable: {e}") from e
