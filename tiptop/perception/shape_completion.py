"""Per-object shape completion via RecGen."""

import asyncio
import logging
import time

import aiohttp
import numpy as np
import trimesh
from jaxtyping import Bool, Float, UInt8

from tiptop.perception.recgen import generate_shape_async

_log = logging.getLogger(__name__)


async def reconstruct_objects_with_recgen(
    session: aiohttp.ClientSession,
    server_url: str,
    rgb_cam: UInt8[np.ndarray, "h w 3"],
    depth_cam: Float[np.ndarray, "h w"],
    masks: Bool[np.ndarray, "n h w"],
    bboxes: list[dict],
    intrinsics: Float[np.ndarray, "3 3"],
    world_from_cam: Float[np.ndarray, "4 4"],
    seed: int = 42,
    target_faces: int | None = None,
    concurrency: int = 4,
) -> dict[str, trimesh.Trimesh]:
    """
    Reconstruct each detected object's world-frame mesh with RecGen, concurrently.

    Requests are dispatched together and bounded by ``concurrency`` (set near the server's GPU count); the server fans
    them across GPUs. Each mesh is reconstructed in the camera frame and transformed to world frame via world_from_cam.
    Raises RuntimeError if any object's request fails, rather than silently dropping it. The masked depth point cloud is
    intentionally not computed here: RecGen needs only the mask, and the downstream pipeline derives object point clouds
    from the masks itself.

    Args:
        session: Shared aiohttp session.
        server_url: RecGen server base URL.
        rgb_cam: Camera-frame RGB image.
        depth_cam: Camera-frame depth (meters).
        masks: Per-object segmentation masks.
        bboxes: Per-object bbox dicts; only "label" is used here.
        intrinsics: Camera intrinsics.
        world_from_cam: Camera-to-world transform applied to each mesh.
        seed: RecGen seed for reproducible reconstruction.
        target_faces: If set, server-side decimate each mesh to this many faces.
        concurrency: Max in-flight requests.

    Returns:
        Dict with object meshes in world frame, keyed by bbox["label"].
    """
    _log.warning(f"RecGen is EXPERIMENTAL and slow: reconstructing {len(bboxes)} object(s), concurrency={concurrency}")
    t_start = time.perf_counter()
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    # RecGen takes the un-eroded mask; erosion degrades its reconstructions. The point cloud path in
    # process_scene_geometry still erodes, where suppressing depth edge noise is what matters.
    masks_2d = masks.astype(bool)
    if len(bboxes) != masks_2d.shape[0]:
        raise ValueError(f"bboxes ({len(bboxes)}) and masks ({masks_2d.shape[0]}) length mismatch")

    sem = asyncio.Semaphore(max(1, concurrency))

    async def request_shape_completion(_label: str, _mask: Bool[np.ndarray, "h w"]) -> dict:
        """POST one object to RecGen under the semaphore; raise with context on failure."""
        async with sem:
            _t_request = time.perf_counter()
            try:
                _response = await generate_shape_async(
                    session,
                    server_url=server_url,
                    rgb=rgb_cam,
                    depth=depth_cam,
                    mask=_mask,
                    intrinsics=intrinsics,
                    seed=seed,
                    target_faces=target_faces,
                )
            except Exception as e:
                raise RuntimeError(f"RecGen reconstruction failed for object '{_label}': {e}") from e
        _log.debug(f"RecGen for {_label} took {time.perf_counter() - _t_request:.2f}s")
        return _response

    # Dispatch every object's request concurrently; gather preserves order, so each response lines up with its bbox
    labels = [bbox["label"] for bbox in bboxes]
    responses = await asyncio.gather(*(request_shape_completion(label, mask) for label, mask in zip(labels, masks_2d)))

    # Transform the reconstructed meshes from the camera frame into the world frame.
    object_meshes: dict[str, trimesh.Trimesh] = {}
    for label, response in zip(labels, responses):
        verts_cam = np.asarray(response["vertices"], dtype=np.float64)
        verts_hom = np.c_[verts_cam, np.ones(len(verts_cam))]
        verts_world = (world_from_cam @ verts_hom.T).T[:, :3]

        mesh_kwargs = {"vertices": verts_world, "faces": np.asarray(response["faces"]), "process": False}
        if "vertex_colors" in response:
            mesh_kwargs["vertex_colors"] = np.asarray(response["vertex_colors"])
        mesh = trimesh.Trimesh(**mesh_kwargs)
        mesh.metadata = {"name": label}
        object_meshes[label] = mesh
        _log.info(f"RecGen {label}: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

    _log.info(f"RecGen reconstructed {len(object_meshes)} object(s) in {time.perf_counter() - t_start:.2f}s")
    return object_meshes
