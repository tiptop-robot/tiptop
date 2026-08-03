# Experimental Features

TiPToP includes two experimental features, both **disabled by default**. They are less thoroughly tested than the core pipeline, and their configuration may change between releases.

```{warning}
Experimental features are not covered by the integration test suite. Enable them for research and evaluation rather than for reliable operation.
```

```{tip}
We really value feedback on these features. If you hit problems running them, or they work well for a task we haven't tried, please [open a GitHub issue](https://github.com/tiptop-robot/tiptop/issues).
```

## Place Next To

Supports goals of the form "place X next to Y" via the `Near` predicate in cuTAMP. Without it, TiPToP only supports placing objects **on** a surface. Examples:

- *"place the mug next to the coffee jar"*
- *"put the apple beside the cereal box"*
- *"put the marker adjacent to the notebook"*
- *"put the screwdrivers near the toolbox"* (moves several objects, one `near` goal each)

This is the feature behind the Place-NextTo task in our [MolmoSpaces evaluation](blogs/molmospaces-inference-time-search/en.md), where TiPToP placed first on that task with 38.0% over 1000 episodes.

Enable it in `tiptop/config/tiptop.yml`:

```yaml
experimental:
  pick_place_next_to: true
```

This requires cuTAMP 0.0.6 or newer, which provides the predicate. TiPToP checks the installed version on startup and fails with a clear message if it is too old.

Enabling the flag switches Gemini to a prompt that can emit `near` atoms when it translates your instruction, and turns on near-placement handling in the planner.

Under the hood this adds cuTAMP's `PlaceNear` operator, defined in [`cutamp/tamp_domain.py`](https://github.com/tiptop-robot/cuTAMP/blob/main/cutamp/tamp_domain.py). It is a normal `Place` plus a `NearPlacement` constraint, so sampling and motion planning are unchanged and the reference object only enters through the cost. That cost is computed in `near_placement_costs` in [`cutamp/cost_function.py`](https://github.com/tiptop-robot/cuTAMP/blob/main/cutamp/cost_function.py): it penalizes the center-to-center xy distance between the object and its reference in one direction only, evaluated at the placement timestep so the reference's pose at that moment is used. The distance threshold is half of each object's largest xy extent plus a fixed gap, so larger objects get a proportionally larger allowance. The constraint counts as satisfied within cuTAMP's default tolerance of 5cm, set by [`default_constraint_to_tol`](https://github.com/tiptop-robot/cuTAMP/blob/main/cutamp/scripts/utils.py).

## RecGen Shape Completion

By default TiPToP represents each object as the convex hull of its observed point cloud. The hull is what cuTAMP uses for collision checking, by sampling collision spheres from the mesh surface, and for placement bounds via the object's bounding box. See *Partial observability and convex hull geometry* in [Limitations](limitations.md).

[RecGen](https://reconstruction-by-generation.github.io/) reconstructs a complete mesh per object, replacing the convex hull. TiPToP sends it the RGB image, the depth map, the object's segmentation mask from SAM-2, and the camera intrinsics, as one request per object. The masked depth point cloud is still used to associate grasps with objects, so grasping behavior is unchanged.

### Setup

RecGen runs as a microservice, like M2T2 and FoundationStereo. Follow the setup instructions at [github.com/williamshen-nz/recgen](https://github.com/williamshen-nz/recgen), and see the [RecGen project page](https://reconstruction-by-generation.github.io/) for background on the method.

RecGen runs once per object, and each reconstruction occupies one GPU for its whole duration. The RecGen gateway starts one worker per GPU and TiPToP dispatches all of a scene's requests at once, so a scene with no more objects than the server has GPUs reconstructs in a single pass and costs about as much as its slowest object. Beyond that, the extra requests queue for a free worker. More GPUs therefore speed up multi-object scenes, but do not make any individual object faster.

Capture viewpoint also matters a lot. Reconstructions degrade on top-down views, and the `q_capture` shipped in `tiptop.yml` is fairly top-down. A more front-facing pose gives noticeably better results, for example:

```yaml
robot:
  q_capture: [0.350, -1.427, 0.495, -2.734, 0.012, 1.990, -2.446]
```

These joint values are specific to our setup, so treat them as a starting point and check the resulting view with [`viz-gripper-cam`](command-reference.md#viz-gripper-cam).

### Configuration

```yaml
perception:
  recgen:
    url: "http://<endpoint>:18324"
    enabled: true
    target_faces: 10000
    concurrency: 6
```

| Key | Description |
|---|---|
| `url` | RecGen gateway address. The gateway listens on port 18324 by default, so normally you only need to fill in your host. Checked at startup when `enabled` is true. |
| `enabled` | Use RecGen instead of convex hulls. |
| `target_faces` | Target face count per object, applied by server-side decimation. Set to `null` to disable, which returns very large meshes. |
| `concurrency` | Maximum in-flight requests. Set at or slightly above the server's GPU count. |

`target_faces` affects the mesh used for visualization and static-world collision only. cuTAMP samples collision spheres for movable objects from the mesh surface, which is insensitive to how finely the mesh is tessellated, so raising it does not change planning behavior.

### Runtime

Reconstruction time depends on object complexity and on your GPUs, so treat these as rough numbers.

Our server has 4x RTX 3090s, so it runs 4 requests in parallel. A single object takes about 9 seconds, and scenes of up to 4 objects finish in roughly that same time overall, 9 to 13 seconds in our runs. With more than 4 objects the extra requests wait for a free GPU, so 5 and 6 object scenes took 17 to 24 seconds.
