import logging
from typing import Dict

import cv2
import numpy as np
import open3d as o3d
import trimesh

# Set up logging
_log = logging.getLogger(__name__)


def aabb_to_cuboid(aabb: np.ndarray, name: str) -> trimesh.primitives.Box:
    """Convert AABB to trimesh Box.

    Args:
        aabb: Axis-aligned bounding box as np.ndarray of shape (2, 3)
              where aabb[0] is min point and aabb[1] is max point
        name: Name to associate with the box

    Returns:
        A trimesh.primitives.Box representing the AABB
    """
    # Calculate box dimensions
    extents = aabb[1] - aabb[0]  # [width, depth, height]
    center = aabb.mean(0)  # Center point

    # Create a box centered at origin with the right dimensions
    box = trimesh.primitives.Box(extents=extents)

    # Move box to the correct position
    box.apply_translation(center)

    # Store name as metadata
    box.metadata = {"name": name}

    # Set a default color (light gray)
    box.visual.face_colors = [200, 200, 200, 255]

    return box


def _object_contact_points(xyz_world: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Estimate the 3D contact point of each object with the surface it rests on.

    For each object mask, takes the bottom 10th-percentile z points and returns their centroid.
    These points approximate where the object touches the table.

    Args:
        xyz_world: (H, W, 3) structured point cloud in world frame.
        masks: (N, 1, H, W) segmentation masks from SAM.

    Returns:
        (M, 3) array of contact points, one per object with enough valid points (M <= N).
    """
    masks_2d = masks.squeeze(1).astype(bool)  # (N, H, W)
    contacts = []
    for mask in masks_2d:
        obj_xyz = xyz_world[mask]
        valid = ~np.isnan(obj_xyz).any(axis=1)
        if valid.sum() < 10:
            continue
        obj_xyz = obj_xyz[valid]
        z_thresh = np.percentile(obj_xyz[:, 2], 10)
        bottom_pts = obj_xyz[obj_xyz[:, 2] <= z_thresh]
        contacts.append(bottom_pts.mean(axis=0))
    return np.array(contacts) if contacts else np.empty((0, 3))


def segment_table_with_ransac(
    xyz_world: np.ndarray,
    rgb: np.ndarray,
    masks: np.ndarray,
    valid_mask: np.ndarray = None,
    max_planes: int = 5,
    contact_threshold: float = 0.03,
) -> trimesh.primitives.Box:
    """Segment the table by finding the plane that the most detected objects rest on.

    Runs iterative RANSAC to find candidate planes, then scores each by how many object
    contact points (bottom of each object's point cloud) lie within `contact_threshold`
    of the plane. The winning plane is the table surface.

    Args:
        xyz_world: (H, W, 3) structured point cloud in world frame.
        rgb: (H, W, 3) RGB colors in [0, 1].
        masks: (N, 1, H, W) segmentation masks from SAM for detected objects.
        valid_mask: Optional boolean mask of valid points over the spatial dims.
        max_planes: Maximum number of RANSAC iterations to run.
        contact_threshold: Distance (metres) within which an object contact point
                           is considered to lie on a candidate plane.

    Returns:
        table_box: trimesh.primitives.Box representing the table.
    """
    if len(xyz_world.shape) != 3:
        raise ValueError(f"Expected structured (H, W, 3) point cloud, got shape {xyz_world.shape}")

    # Estimate where each object contacts its supporting surface
    contact_pts = _object_contact_points(xyz_world, masks)
    if len(contact_pts) == 0:
        raise RuntimeError("No object contact points found — ensure objects are detected before calling this function.")
    _log.debug(f"Object contact points (world frame):\n{contact_pts}")

    # Build point cloud from valid points
    if valid_mask is None:
        valid_mask = ~np.isnan(xyz_world).any(axis=2)
    xyz_valid = xyz_world[valid_mask]
    rgb_valid = rgb[valid_mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_valid)
    pcd.colors = o3d.utility.Vector3dVector(rgb_valid)
    voxel_size = 0.005
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    # Iterative RANSAC: score each candidate plane by number of objects resting on it
    remaining_pcd = pcd
    best_score = -1
    best_pcd = None

    for i in range(max_planes):
        if len(remaining_pcd.points) < 50:
            break

        plane_model, inlier_idxs = remaining_pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
        a, b, c, d = plane_model
        norm = np.linalg.norm([a, b, c])

        # Distance from each contact point to this plane
        dists = np.abs(contact_pts @ np.array([a, b, c]) + d) / norm
        score = int((dists < contact_threshold).sum())

        inlier_pcd = remaining_pcd.select_by_index(inlier_idxs)
        _log.debug(f"Plane {i}: model={plane_model}, objects_on_plane={score}/{len(contact_pts)}")

        if score > best_score:
            best_score = score
            best_pcd = inlier_pcd

        remaining_pcd = remaining_pcd.select_by_index(inlier_idxs, invert=True)

    if best_pcd is None or best_score == 0:
        raise RuntimeError(
            f"No plane found with objects resting on it (tried {max_planes} planes, "
            f"{len(contact_pts)} contact points, threshold={contact_threshold}m)."
        )

    _log.info(f"Selected table plane with {best_score}/{len(contact_pts)} objects")
    table_pcd = best_pcd

    # Remove statistical outliers to eliminate distant points that happen to lie on the plane
    table_pcd, _ = table_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # DBSCAN clustering: keep only the largest cluster (the actual table surface).
    # eps is tied to voxel_size so no independent hyperparameter is introduced.
    dbscan_eps = 3 * voxel_size
    labels = np.array(table_pcd.cluster_dbscan(eps=dbscan_eps, min_points=10))
    n_clusters = int(labels.max()) + 1 if len(labels) > 0 and labels.max() >= 0 else 0
    if n_clusters == 0:
        raise RuntimeError("DBSCAN found no clusters in table plane inliers after outlier removal.")
    if n_clusters > 2:
        _log.warning(
            f"DBSCAN found {n_clusters} clusters in table plane inliers — expected 1–2. "
            f"Keeping largest cluster; point cloud may have significant noise."
        )
    else:
        _log.debug(f"DBSCAN found {n_clusters} cluster(s) in table plane inliers.")
    largest_label = int(np.bincount(labels[labels >= 0]).argmax())
    table_pcd = table_pcd.select_by_index(np.where(labels == largest_label)[0])

    # Get table AABB using percentile-based bounds to handle remaining outliers
    table_pts = np.asarray(table_pcd.points)
    # Use 2nd and 98th percentiles for XY to avoid extreme outliers while keeping most of table
    xy_min = np.percentile(table_pts[:, :2], 2, axis=0)
    xy_max = np.percentile(table_pts[:, :2], 98, axis=0)
    # Use actual min/max for Z since height is well-defined by RANSAC
    z_min = table_pts[:, 2].min()
    z_max = table_pts[:, 2].max()

    table_aabb = np.stack([np.append(xy_min, z_min), np.append(xy_max, z_max)])
    surface_z = table_pts[:, 2].mean()

    # Create table box
    table_box = aabb_to_cuboid(table_aabb, "table")

    # Adjust height position so the top of the box aligns with detected surface
    # We need to adjust the transform directly since trimesh.Box works differently
    extents = table_box.extents
    table_center = table_box.center_mass
    # Offset the box down so its top surface aligns with the detected plane
    height_offset = surface_z - table_center[2] - extents[2] / 2 - 0.02  # small offset
    table_box.apply_translation([0, 0, height_offset])

    # Set color from point cloud
    table_color = (np.asarray(table_pcd.colors).mean(0) * 255).astype(np.uint8)
    table_color_rgba = np.append(table_color, 255)
    table_box.visual.face_colors = table_color_rgba

    _log.info(f"Table surface at z = {surface_z:.3f}, dims = {table_box.extents}")
    return table_box


def augment_with_base_projections(
    points: np.ndarray, colors: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray | None]:
    """Augment a point cloud by appending a copy of each point flattened to the minimum z of the input.

    Args:
        points: (N, 3) points in world coordinates.
        colors: Optional (N, 3) or (N, 4) colors per point.

    Returns:
        Tuple of (2N, 3) augmented points and (2N, C) augmented colors (or None).
    """
    min_z = points[:, 2].min()

    projected_points = points.copy()
    projected_points[:, 2] = min_z
    augmented_points = np.vstack([points, projected_points])
    if colors is not None:
        augmented_colors = np.vstack([colors, colors])
    else:
        augmented_colors = None

    return augmented_points, augmented_colors


def segment_pointcloud_by_masks(
    xyz_world: np.ndarray,
    rgb: np.ndarray,
    masks: np.ndarray,
    bboxes: list[dict],
    max_z: float,
    return_pcd: bool = False,
    erode_pixels: int = 0,
) -> dict[str, trimesh.Trimesh] | tuple[dict[str, trimesh.Trimesh], dict]:
    """Segment pointcloud using object masks.

    Args:
        xyz_world: Point cloud in world frame with shape (H, W, 3) matching the masks
        rgb: RGB colors corresponding to the point cloud with shape (H, W, 3)
        masks: (num_objects, 1, H, W) segmentation masks from SAM
        bboxes: List of bbox dictionaries with 'label' and 'box_2d' keys
        max_z: Maximum z value for filtering points
        return_pcd: Whether to return point clouds in addition to meshes
        erode_pixels: Number of pixels to erode the mask by to handle depth edge noise. Default is 0 (no erosion).

    Returns:
        Dictionary mapping object labels to trimesh.Trimesh objects
    """
    object_meshes = {}
    object_pcds = {}
    masks_2d = masks.squeeze(1).astype(bool)  # (num_objects, H, W)

    # Check that we have a structured pointcloud
    if len(xyz_world.shape) != 3 or xyz_world.shape[2] != 3:
        raise ValueError(
            f"Expected structured pointcloud with shape (H, W, 3), got {xyz_world.shape}. "
            f"Flattened pointclouds are not supported as they cannot be directly masked."
        )

    # Handle case where we have more masks than bounding boxes
    if masks.shape[0] > len(bboxes):
        _log.warning(
            f"More masks ({masks.shape[0]}) than bounding boxes ({len(bboxes)}). "
            f"Selecting best masks for each bbox based on IoU."
        )

        # Extract 2D bounding boxes from the detection results
        bbox_coords = []
        for bbox in bboxes:
            # Extract coordinates from the box_2d field
            if "box_2d" in bbox and len(bbox["box_2d"]) == 4:
                # Assuming box_2d is [ymin, xmin, ymax, xmax] normalized to 0-1000
                ymin, xmin, ymax, xmax = bbox["box_2d"]
                # Convert to image coordinates based on mask dimensions
                h, w = masks.shape[2:]
                x_min = int((xmin / 1000.0) * w)
                y_min = int((ymin / 1000.0) * h)
                x_max = int((xmax / 1000.0) * w)
                y_max = int((ymax / 1000.0) * h)
                bbox_coords.append([x_min, y_min, x_max, y_max])
            else:
                # If bbox format is invalid, use a dummy bbox
                _log.warning(f"Invalid bbox format for {bbox.get('label', 'unknown')}")
                bbox_coords.append([0, 0, 10, 10])  # Small dummy box to ensure lowest IoU

        # Calculate IoU between each mask and each bbox to find best matches
        best_mask_indices = []

        for bbox_idx, bbox_coord in enumerate(bbox_coords):
            x_min, y_min, x_max, y_max = bbox_coord
            # Create a binary mask for the bbox
            bbox_mask = np.zeros((masks.shape[2], masks.shape[3]), dtype=bool)
            bbox_mask[y_min:y_max, x_min:x_max] = True

            # Calculate IoU for each mask with this bbox
            best_iou = -1
            best_idx = -1

            for mask_idx, mask in enumerate(masks_2d):
                # Skip if this mask has already been assigned
                if mask_idx in best_mask_indices:
                    continue

                # Calculate intersection and union
                intersection = np.logical_and(mask, bbox_mask).sum()
                union = np.logical_or(mask, bbox_mask).sum()
                iou = intersection / union if union > 0 else 0

                # Update if better IoU is found
                if iou > best_iou:
                    best_iou = iou
                    best_idx = mask_idx

            if best_idx >= 0:
                best_mask_indices.append(best_idx)
            else:
                # If no good match is found (all masks already assigned),
                # use any unassigned mask
                remaining_indices = [i for i in range(len(masks_2d)) if i not in best_mask_indices]
                if remaining_indices:
                    best_mask_indices.append(remaining_indices[0])
                else:
                    # No masks left - this should not happen with more masks than boxes
                    _log.warning(f"No mask available for bbox {bbox_idx}")
                    best_mask_indices.append(0)  # Use first mask as fallback

        _log.info(f"Selected mask indices {best_mask_indices} for {len(bboxes)} bounding boxes")

        # Create a new masks array with just the selected masks
        selected_masks = np.zeros((len(best_mask_indices), 1, masks.shape[2], masks.shape[3]), dtype=masks.dtype)
        for i, idx in enumerate(best_mask_indices):
            selected_masks[i, 0] = masks[idx, 0]
        masks = selected_masks
        masks_2d = masks.squeeze(1).astype(bool)  # Update masks_2d with selected masks

    # Process each mask and create a mesh for each object
    for mask_2d, bbox in zip(masks_2d, bboxes):
        label = bbox["label"]

        # Erode the mask to handle depth edge noise
        if erode_pixels > 0:
            kernel = np.ones((erode_pixels * 2 + 1, erode_pixels * 2 + 1), np.uint8)
            mask_2d = cv2.erode(mask_2d.astype(np.uint8), kernel, iterations=1).astype(bool)

        # Get points for this object using the mask
        xyz_obj = xyz_world[mask_2d]
        rgb_obj = rgb[mask_2d]

        # Filter out invalid points
        valid = ~np.isnan(xyz_obj).any(axis=1)
        xyz_obj = xyz_obj[valid]
        rgb_obj = rgb_obj[valid]

        if len(xyz_obj) < 10:
            _log.warning(f"Skipping {label}: too few points ({len(xyz_obj)})")
            continue

        z_mask = xyz_obj[..., 2] > max_z
        xyz_proj, rgb_proj = augment_with_base_projections(xyz_obj[z_mask], rgb_obj[z_mask])

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_proj)
        pcd.colors = o3d.utility.Vector3dVector(rgb_proj)

        # Remove outliers
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
        object_pcds[label] = pcd

        # Compute convex hull
        try:
            hull, _ = pcd.compute_convex_hull()
            vertices = np.asarray(hull.vertices)
            centroid = vertices.mean(0)

            # Get hull faces as triangles
            faces = np.asarray(hull.triangles)

            # Convert to trimesh
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)

            # Add metadata
            mesh.metadata = {"name": label, "centroid": centroid.tolist()}

            # Set color from average of point cloud
            mean_color = np.asarray(pcd.colors).mean(0)
            color_rgba = np.append(mean_color * 255, 255).astype(np.uint8)
            mesh.visual.face_colors = color_rgba

            # Store the object in the dictionary
            object_meshes[label] = mesh
            _log.info(f"Created mesh for {label}: {len(pcd.points)} pts, centroid={centroid}")

        except Exception as e:
            _log.warning(f"Failed to create mesh for {label}: {e}")

    if return_pcd:
        return object_meshes, object_pcds
    else:
        return object_meshes
