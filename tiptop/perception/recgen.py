import logging
import os
import time

import aiohttp
import msgpack
import msgpack_numpy
import numpy as np
from jaxtyping import Bool, Float, UInt8

from tiptop.utils import ServerHealthCheckError

msgpack_numpy.patch()

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
    timeout: float = 600.0,
) -> dict:
    """Run RecGen single-view reconstruction on one object via HTTP.

    Returns the unpacked msgpack payload with keys ``vertices``, ``faces``,
    ``pose_matrix``, ``pose_quat``, and optionally ``vertex_colors``. The mesh
    is expressed in the camera frame.

    If ``target_faces`` is provided, the server-side quadric edge-collapse
    decimator runs after inference and the response carries the decimated mesh.
    """
    payload = {
        "rgb": np.ascontiguousarray(rgb),
        "depth": np.ascontiguousarray(depth),
        "mask": np.ascontiguousarray((mask > 0).astype(np.uint8)),
        "intrinsics": np.ascontiguousarray(intrinsics, dtype=np.float64),
        "seed": int(seed),
    }
    if target_faces is not None:
        payload["target_faces"] = int(target_faces)
    body = msgpack.packb(payload, use_bin_type=True)
    endpoint = os.path.join(server_url.rstrip("/"), "generate")

    start_time = time.perf_counter()
    _log.debug(f"Sending inference request to RecGen server at {endpoint}")
    async with session.post(
        endpoint,
        data=body,
        headers={"Content-Type": "application/x-msgpack"},
        timeout=aiohttp.ClientTimeout(total=timeout),
    ) as response:
        response.raise_for_status()
        result = msgpack.unpackb(await response.read(), raw=False)
    duration = time.perf_counter() - start_time
    _log.info(f"RecGen inference time={duration:.2f}s")
    return result


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
