# Experimental Features

TiPToP includes two experimental features, both **disabled by default**. They are less battle-tested than the default pipeline and their configuration may change between releases.

```{warning}
Experimental features are not covered by the integration test suite. Enable them for research and evaluation rather than for reliable operation.
```

```{tip}
We really value feedback on these features. If you hit problems running them, or they work well for a task we haven't tried, please [open a GitHub issue](https://github.com/tiptop-robot/tiptop/issues) — reports from real setups are what move these from experimental to default. See [Contributing](contributing.md) for the issue templates.
```

## Place Next To

Supports goals of the form "place X next to Y" — for example *"put the yellow block next to the orange block"* — via the `Near` predicate in cuTAMP. Without it, TiPToP only supports placing objects **on** a surface.

Enable it in `tiptop/config/tiptop.yml`:

```yaml
experimental:
  pick_place_next_to: true
```

This requires cuTAMP 0.0.6 or newer, which provides the predicate. TiPToP checks the installed version on startup and fails with a clear message if it is too old.

Enabling the flag switches Gemini to a prompt that can emit `near` atoms when it translates your instruction, and turns on near-placement handling in the planner. The placement distance tolerance defaults to cuTAMP's `NearPlacement` value of 5cm; see `default_constraint_to_tol` in cuTAMP's `cutamp/scripts/utils.py` to change it.

## RecGen Shape Completion

By default TiPToP represents each object as the convex hull of its observed point cloud. As described under *Partial observability and convex hull geometry* in [Limitations](limitations.md), the hull wraps all observed geometry, so the planner cannot distinguish "inside the box" from "on top of the box".

[RecGen](https://reconstruction-by-generation.github.io/) reconstructs a complete mesh for each object from a single RGB-D view, replacing the convex hull. The masked depth point cloud is still used to associate grasps with objects, so grasping behaviour is unchanged.

### Setup

RecGen runs as a microservice, like M2T2 and FoundationStereo. Follow the setup instructions at [github.com/williamshen-nz/recgen](https://github.com/williamshen-nz/recgen), and see the [RecGen project page](https://reconstruction-by-generation.github.io/) for background on the method.

We recommend running it on a multi-GPU machine: TiPToP generates one completion per object and dispatches those requests concurrently, so the server fans them across available GPUs.

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
| `url` | RecGen server address. Checked at startup when `enabled` is true. |
| `enabled` | Use RecGen instead of convex hulls. |
| `target_faces` | Target face count per object, applied by server-side decimation. Set to `null` to disable, which returns very large meshes. |
| `concurrency` | Maximum in-flight requests. Set at or slightly above the server's GPU count. |

`target_faces` affects the mesh used for visualisation and static-world collision only. cuTAMP samples collision spheres for movable objects from the mesh surface, which is insensitive to how finely the mesh is tessellated, so raising it does not change planning behaviour.

### Caveats

RecGen adds roughly 10-20s per object, which is why convex hulls remain the default.

Reconstruction quality varies, and the failure modes below are RecGen-side rather than integration problems:

- **Camera viewpoint matters.** Reconstructions degrade on top-down views. A more front-facing capture pose gives noticeably better results.
- **Occluded objects reconstruct poorly.** A partially occluded object can come back over-extended along one axis, which may then fail cuTAMP's stable-placement constraint.
- **Multi-shell reconstructions.** RecGen occasionally returns two overlapping shells for one object. Decimation cannot merge geometrically distinct surfaces, so these meshes stay large.
