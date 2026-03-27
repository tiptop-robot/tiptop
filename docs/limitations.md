# Limitations

The TiPToP [project page](https://tiptop-robot.github.io/#extensions) and [implementation notes](https://tiptop-robot.github.io/implementation#limitations) discuss known limitations and potential extensions. We document some additional ones below to help set expectations and guide future development. [Contributions](contributing.md) to help overcome any of these are welcome!

#### No object-on-object stacking

TiPToP cannot stack multiple objects atop one another, so tasks such as block stacking are currently not supported. This is due to the current implementation of cuTAMP and is not a conceptual limitation of the system. We are working on updates to support this.

#### Table-plane assumption

The perception pipeline assumes a single dominant table surface, detected via iterative RANSAC scored by how many object contact points lie on each candidate plane. This fails or degrades when the workspace surface is not flat or when objects are distributed across multiple surfaces (e.g., a shelf above a table). Workspaces without a clear table-like surface are not supported. See `segment_table_with_ransac` in `tiptop/perception/segmentation.py`.

One approach to overcome this is to extend the Gemini VLM object detection to also detect surfaces (e.g., shelves, trays, table regions), which could be used to extract mutliple support surfaces for the scene representation.


#### Partial observability and convex hull geometry

TiPToP represents each object as the convex hull of its observed point cloud. Even if multiple faces of an object are visible, the convex hull wraps the entire observed geometry, so the "On" predicate used by the planner (cuTAMP) reduces to placing objects on top of the hull. For an open box, for instance, the planner cannot distinguish "inside the box" from "on top of the box" since both are surfaces of the same convex hull.

See `augment_with_base_projections` and `segment_pointcloud_by_masks` in `tiptop/perception/segmentation.py`, and `stable_placement_costs` in `cutamp/cost_function.py`.

Improving the perception pipeline to use part-based segmentation to represent distinct object surfaces would be one approach for addressing this.

#### Oriented bounding boxes and placement bounds

cuTAMP uses oriented bounding boxes (OBBs) to determine valid placement regions for the "On" predicate. For flat or irregular objects such as plates, the OBB can extend beyond the actual physical surface, causing the planner to generate placement poses that overhang the edge of the object. One fix would be to use the 2D convex hull of the object's point cloud projected onto the support plane as the placement region, which would tightly follow the actual observed surface shape. See `cutamp/utils/obb.py` and `cutamp/cost_function.py`.

#### Grasp generation failures and fallback grasper

M2T2 does not always generate grasps for every object in the scene, particularly when the scene is cluttered: large objects dominate the point cloud and smaller or partially occluded objects may receive few or no grasp candidates. When this happens, TiPToP falls back to a heuristic 4-DOF grasp sampler in cuTAMP, which is less reliable and can produce poor grasps on objects with unusual geometry. Improvements to M2T2 or the point cloud preprocessing would help. See `grasp_4dof_sampler` in `cutamp/particle_initialization.py` for the fallback sampler.
