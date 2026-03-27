# Adding a New Embodiment

```{important}
These docs are a work in progress and are likely **not** comprehensive.  We plan to make adding new embodiments easier over time. Contributions are welcome: if you run into trouble, please reach out for help.
```

TiPToP supports adding new robots and cameras through its config-driven architecture. All robot and camera selection flows through `tiptop/config/tiptop.yml`, with dispatch logic in a handful of files. This guide walks through each step, using the UR5 + RealSense port as a concrete reference.

```{tip}
For every step below, look at the existing implementations as working examples: the UR5 integration in `tiptop/ur5/` and the Panda/FR3 integration via `bamboo-franka-client` in `tiptop/utils.py`. For cuTAMP, see `cutamp/robots/ur5.py`, `cutamp/robots/franka_robotiq.py`, and `cutamp/robots/franka.py`.
```

---

## Adding a New Robot

### Step 1: Create a config YAML

In `tiptop/config/tiptop.yml` fill in the robot-specific fields. The key fields are:

```yaml
robot:
  type: your_robot        # Unique string identifier
  dof: 6                  # Degrees of freedom (excluding gripper)
  host: "192.168.1.2"     # IP address or hostname for robot connection
  time_dilation_factor: 0.2  # Speed scaling (0.0-1.0), start low for safety
  q_home: [...]           # Home joint position angles (in radians typically)
  q_capture: [...]        # Image capture joint configuration (in radians typically)
```

### Step 2: Add cuTAMP and cuRobo assets

cuRobo and cuTAMP need robot description files for motion planning. You'll need to create the following in `cutamp/robots/assets/`:

1. **URDF** — Your robot arm + gripper + camera mount as a single URDF. The camera mount should be modeled so the motion planner accounts for it during collision checking — it doesn't need to be attached as a fixed joint on the tool frame specifically, just present and accurately represented. See `ur5e_robotiq_2f_85.urdf`.

2. **cuRobo YAML config** — Defines collision spheres for each link along, links to ignore for self-collision checks, retract configurations, etc. See `ur5e_robotiq_2f_85.yml`. Some of the fields that matter are:

   - `urdf_path` - path to the URDF file for the robot
   - `base_link` - name of the base link in the URDF
   - `ee_link` - name of the end effector link in the URDF
   - `collision_spheres` — spheres approximating each link's geometry (see below for tools)
   - `retract_config` — a safe joint configuration to retract to

   To generate collision spheres, we recommend [Ballpark](https://github.com/chungmin99/ballpark) — we've had good experience with it and it includes a web visualizer that makes tuning spheres straightforward. [Foam](https://github.com/CoMMALab/foam) is another option.

   ```{note}
   The [cuRobo robot configuration tutorial](https://curobo.org/tutorials/1_robot_configuration.html) has useful reference material, but you **do not** need to install IsaacSim/IsaacLab or follow the full tutorial. You only need the YAML config file — look at the existing configs in `cutamp/robots/assets/` and adapt from there.
   ```

3. **Gripper collision spheres** — A `.pt` file containing a `(num_spheres, 4)` tensor where each row is `[x, y, z, radius]`. These are used by cuTAMP for collision checking in particle initialization. The frame convention is easiest to understand by looking at existing examples such as `franka_gripper_spheres.pt` and `robotiq_2f_85_gripper_spheres.pt` alongside the corresponding robot module.

   To visualize and verify your gripper spheres, run the robot module (e.g., `python -m cutamp.robots.ur5`), which renders the spheres overlaid on the robot in Rerun.

### Step 3: Create the cuTAMP robot module

Create a robot module at `cutamp/robots/your_robot.py` that exposes the same set of functions as the existing modules. Your module needs to provide cuRobo config loading, IK solver construction, gripper sphere loading, Rerun visualization, and a kinematics container. See these examples:

- `cutamp/robots/ur5.py` — UR5 with Robotiq gripper
- `cutamp/robots/franka_robotiq.py` — FR3/Panda with Robotiq gripper
- `cutamp/robots/franka.py` — FR3/Panda with Franka gripper

You can verify your robot loads correctly by running the module (e.g., `python -m cutamp.robots.ur5`), which should display the robot in Rerun.

#### Registering with cuTAMP

You'll need to register your robot's string identifier (e.g., `"your_robot"`) with cuTAMP so that `TAMPConfiguration(robot="your_robot")` works. This requires:

1. Add your robot module to `cutamp/robots/__init__.py` and register it in the robot registry there.
2. In `cutamp/config.py`, add your robot type to both the `Literal` type on the `robot` field in `TAMPConfiguration` and the set check in `validate_tamp_config`.

Look at how existing robots are handled in both files and follow the same pattern.

As part of registering your robot, you'll also need to define the **`tool_from_ee` transform** — the offset from the kinematic end-effector (`ee_link`) to the actual tool center point (e.g., the midpoint between the gripper fingertips). Get this wrong and grasps will be offset or rotated. This is defined inside your `load_your_container` function in `cutamp/robots/__init__.py` — refer closely to the existing implementations there.

#### Debugging tips

- **Test the IK solver in isolation.** Before running full TAMP, solve IK for a few known reachable poses to confirm the solver converges. If it doesn't, check your URDF joint limits and collision spheres.
- **Collision spheres too large or too small?** If the planner can't find any valid plans, your collision spheres may be too conservative — this is most impactful on the gripper side, where spheres that are too large will prevent the planner from finding valid grasps near objects. Visualize them in Rerun to check coverage.
- **Run cuTAMP demos before full TiPToP runs.** Use the `cutamp-demo` flag to test your robot's planning in isolation with the right embodiment wired up, before running the full TiPToP pipeline.

### Step 4: Write a robot client

Create a new module under `tiptop/` for your robot (e.g., `tiptop/your_robot/`). Your client must implement the same interface used by the existing robot clients. Compare these two to see the common contract:

- `tiptop/ur5/ur5_client.py` — full in-repo implementation (UR5 + Robotiq gripper via RTDE)
- `tiptop/utils.py` → `get_bamboo_client()` — Panda/FR3 integration point (external `bamboo-franka-client` package)

The `RobotClient` type union and `get_robot_client()` dispatch in `tiptop/utils.py` show how both clients are used interchangeably.

#### Things to keep in mind

- **Trajectory interpolation.** Trajectories from cuRobo's MotionGen arrive at variable dt — you may need to interpolate to your robot's servo rate (e.g., the UR5 client interpolates to 125Hz).
- **Safety checks.** Verify the first waypoint is close to the robot's current position before executing. The UR5 client uses a 1-degree threshold.
- **Return format.** The execution layer in `tiptop/execute_plan.py` expects `{"success": True}` on success and `{"success": False, "error": "..."}` on failure.
- **Gripper parameters.** `speed` and `force` are normalized floats in `[0, 1]`. Scale to your gripper's native units.
- **Factory function.** Provide a cached factory (e.g., `get_your_robot_client()`) that reads config and returns a singleton. Use lazy imports for your SDK so it's not required when running with a different robot type.

If your robot SDK requires additional packages, add an optional dependency group in `pyproject.toml` (see the existing `ur5` group for reference).

### Step 5: Wire up the tiptop dispatch

Several files dispatch on `cfg.robot.type` to select the right robot client, IK solver, motion planner, etc. Add `elif` branches for your robot in each one — follow the existing patterns for UR5 and Panda:

1. **`tiptop/utils.py`** — `RobotClient` type union, `get_robot_client()`, `get_robot_rerun()`
2. **`tiptop/motion_planning.py`** — `get_ik_solver()`, `get_motion_gen()`
3. **`tiptop/workspace.py`** — `workspace_cuboids()` (see Step 6)
4. **`tiptop/scripts/viz_calibration.py`** — container loading for calibration visualization
5. **`tiptop/scripts/perception_demo.py`** — container loading for perception demo

```{note}
This may not be a comprehensive list. We welcome contributions that help us simplify adding a new embodiment!
```

### Step 6: Define workspace obstacles

Add a function in `tiptop/workspace.py` that returns cuboid obstacles for your physical setup:

```python
def your_robot_workspace() -> tuple[Cuboid, ...]:
    table = Cuboid("table", dims=[0.6, 0.9, 0.02], pose=[0, 0, -0.01, *unit_quat], color=[200, 180, 130])
    # Add walls, ceiling, nearby furniture, etc.
    return (table,)
```

Then add an `elif` branch in `workspace_cuboids()`:

```python
elif cfg.robot.type == "your_robot":
    return your_robot_workspace()
```

Visualize the result to iterate on obstacle placement:

```bash
python tiptop/workspace.py
```

```{important}
It's better to overapproximate workspace obstacles than underapproximate them. Include anything the robot could collide with: tables, walls, monitors, camera mounts, etc.
```

### Step 7: Calibrate and verify

#### Wrist-camera calibration

If you're using a wrist-mounted camera, you need to calibrate the `ee_from_cam` transform (the fixed offset from the end-effector frame to the camera frame). Run:

```bash
calibrate-wrist-cam
```

You may need to make changes to `calibrate_wrist_cam.py` to ensure the calibration works properly with your camera setup. The calibration result is stored per camera serial number in `tiptop/config/assets/calibration_info.json` as a 6-DOF pose `[x, y, z, roll, pitch, yaw]`.

#### Verification checklist

With your config active and the robot powered on:

```bash
# 1. Check robot connection
get-joint-positions

# 2. Test gripper
gripper-open
gripper-close

# 3. Test motion planning (this will move the robot!)
go-home

# 4. Verify camera calibration — visualizes point cloud overlaid on robot in Rerun
viz-calibration
```

### Troubleshooting

#### cuTAMP can't find any valid plans

- **Incorrect `tool_from_ee`**: If this transform is wrong, the sampled grasp and placement poses will be offset or rotated relative to the object or gripper, causing the planner to fail to find valid plans. Re-check how it's defined in `cutamp/robots/__init__.py` and compare against existing robot modules.
- **Collision spheres too large.** If the planner rejects every candidate, your collision spheres may be overly conservative. Visualize them in Rerun and shrink any that extend well beyond the actual link geometry.
- **Workspace obstacles overlapping the robot.** Check that your `workspace_cuboids()` don't intersect the robot's base or links at the home configuration.
- **IK solver not converging.** Test the IK solver in isolation for a known reachable pose. If it fails, check that `ee_link` in the cuRobo config matches what your URDF defines and that joint limits are correct.

#### cuTAMP plans succeed but execution fails

- **Controller not tracking the trajectory accurately enough.** The robot may not be able to faithfully follow the planned trajectory. This typically requires improving the low-level controller implementation itself, e.g. tuning controller gains.
- **Joint limits mismatch.** The planner generates configurations that exceed your real robot's limits. Compare the URDF joint limits against your robot's actual limits.

---

## Adding a New Camera

We already provide support for RealSense and ZED cameras, so this section is if you need to add a new camera type.

The camera interface uses a `Frame` base class and a `Camera` protocol, so adding a new camera is mostly about subclassing `Frame` and implementing the protocol. See `tiptop/perception/cameras/rs_camera.py` (RealSense) and `tiptop/perception/cameras/zed_camera.py` (ZED) for complete examples.

### Step 1: Write a camera driver

Create a new file in `tiptop/perception/cameras/` (e.g., `your_camera.py`). You need three things: a frame subclass, an intrinsics dataclass, and a camera class.

#### Frame subclass

All frames extend the `Frame` base class in `tiptop/perception/cameras/frame.py`:

```python
@dataclass(frozen=True)
class Frame:
    serial: str
    timestamp: float
    rgb: UInt8[np.ndarray, "h w 3"]
    intrinsics: Float[np.ndarray, "3 3"]  # Camera intrinsics matrix
    depth: Float[np.ndarray, "h w"] | None = None  # Onboard sensor depth in metres
```

`Frame` also provides a `bgr` property that converts `rgb` to BGR automatically. The rest of the codebase accesses `frame.rgb`, `frame.bgr`, `frame.depth`, and `frame.intrinsics` directly — there are no dispatcher functions to update.

Your subclass adds camera-specific fields needed for stereo depth inference. Use `frozen=True, kw_only=True`:

```python
@dataclass(frozen=True, kw_only=True)
class YourFrame(Frame):
    # Add stereo fields needed for FoundationStereo
    left_rgb: UInt8[np.ndarray, "h w 3"] | None = None
    right_rgb: UInt8[np.ndarray, "h w 3"] | None = None
```

The stereo pair can be RGB (like ZED) or IR converted to 3-channel (like RealSense). If your camera doesn't have stereo, you'll need an alternative depth source.

#### Intrinsics dataclass

A frozen dataclass holding camera calibration parameters for stereo depth inference and wrist-camera calibration. The fields you need depend on your camera — here are the two existing examples:

```python
# ZED — stereo pair is the left/right RGB cameras
@dataclass(frozen=True)
class ZedIntrinsics:
    K_left: Float[np.ndarray, "3 3"]
    K_right: Float[np.ndarray, "3 3"]
    distortion_left: Float[np.ndarray, "12"]
    distortion_right: Float[np.ndarray, "12"]
    baseline: float  # Meters

# RealSense — stereo pair is the IR cameras, which differ from the color camera
@dataclass(frozen=True)
class RealsenseIntrinsics:
    K_color: Float[np.ndarray, "3 3"]
    K_ir: Float[np.ndarray, "3 3"]
    baseline_ir: float  # Meters
    T_color_from_ir: Float[np.ndarray, "4 4"]  # Needed to warp IR depth onto color grid
    distortion_color: Float[np.ndarray, "5"]
```

At minimum you need: intrinsics for the stereo camera pair, a baseline in meters, and distortion coefficients for `calibrate-wrist-cam`.

#### Camera class

Your camera class must satisfy the `Camera` protocol defined in `cameras/__init__.py`:

```python
class Camera(Protocol):
    serial: str
    def read_camera(self) -> Frame: ...
    def close(self) -> None: ...
```

A typical implementation:

```python
class YourCamera:
    def __init__(self, serial: str, enable_depth: bool = False):
        self.serial = serial
        # Initialize your camera SDK here

    def read_camera(self) -> YourFrame:
        # Capture and return a frame
        # Must populate: serial, timestamp, rgb, intrinsics (3x3 matrix), and
        # optionally depth (in metres) plus any camera-specific stereo fields
        ...

    @cache
    def get_intrinsics(self) -> YourIntrinsics:
        # Return intrinsics (cached since they don't change)
        ...

    def close(self):
        # Release camera resources
        ...
```

```{important}
`read_camera()` should be fast — avoid expensive conversions here since it runs in the capture loop. Note that `intrinsics` on the `Frame` is the color camera's 3x3 matrix (e.g., `K_left` for ZED, `K_color` for RealSense), while `get_intrinsics()` returns the full intrinsics dataclass used by depth inference.
```

### Step 2: Add depth inference functions

FoundationStereo takes a rectified stereo pair and returns a depth map in meters. You need to write sync and async wrappers that prepare your camera's stereo data for the FoundationStereo server.

The FoundationStereo API (`tiptop/perception/foundation_stereo.py`) expects:

| Parameter | Type | Description |
|-----------|------|-------------|
| `left_rgb` | `uint8 (h, w, 3)` | Left stereo image (RGB) |
| `right_rgb` | `uint8 (h, w, 3)` | Right stereo image (RGB) |
| `fx`, `fy`, `cx`, `cy` | `float` | Intrinsics of the **left** stereo camera |
| `baseline` | `float` | Stereo baseline in meters |

```python
def your_cam_infer_depth(
    frame: YourFrame, intrinsics: YourIntrinsics
) -> Float[np.ndarray, "h w"]:
    """Returns depth in meters aligned to the color image."""
    from tiptop.perception.foundation_stereo import infer_depth

    cfg = tiptop_cfg()
    depth = infer_depth(
        cfg.perception.foundation_stereo.url,
        left_rgb=frame.left_rgb,
        right_rgb=frame.right_rgb,
        fx=intrinsics.K_color[0, 0],
        fy=intrinsics.K_color[1, 1],
        cx=intrinsics.K_color[0, 2],
        cy=intrinsics.K_color[1, 2],
        baseline=intrinsics.baseline,
    )
    return depth
```

The async version follows the same pattern but takes an `aiohttp.ClientSession` as the first argument and uses `infer_depth_async`:

```python
async def your_cam_infer_depth_async(
    session: aiohttp.ClientSession,
    frame: YourFrame,
    intrinsics: YourIntrinsics,
) -> Float[np.ndarray, "h w"]:
    from tiptop.perception.foundation_stereo import infer_depth_async

    cfg = tiptop_cfg()
    depth = await infer_depth_async(
        session,
        cfg.perception.foundation_stereo.url,
        left_rgb=frame.left_rgb,
        right_rgb=frame.right_rgb,
        fx=intrinsics.K_color[0, 0],
        fy=intrinsics.K_color[1, 1],
        cx=intrinsics.K_color[0, 2],
        cy=intrinsics.K_color[1, 2],
        baseline=intrinsics.baseline,
    )
    return depth
```

```{note}
If your stereo pair is not from the color camera (e.g. RealSense uses IR stereo), you need to warp the resulting depth map onto the color pixel grid. See `_depth_ir_to_color()` in `rs_camera.py` for an example.
```

### Step 3: Wire up camera dispatch

In `tiptop/perception/cameras/__init__.py`, you only need to touch two things:

**1. `get_depth_estimator()`** — add an `isinstance` branch that returns an async depth estimation callable for your camera:

```python
elif isinstance(cam, YourCamera):
    intrinsics = cam.get_intrinsics()

    async def _your_estimate(session: aiohttp.ClientSession, f: Frame) -> np.ndarray:
        return await your_cam_infer_depth_async(session, f, intrinsics)

    return _your_estimate
```

**2. `get_hand_camera()` / `get_external_camera()`** — add an `elif` branch to instantiate your camera from config:

```python
elif cam_type == "your_camera":
    return YourCamera(str(cam_cfg.serial), enable_depth=depth)
```

That's it — since all frames share the `Frame` base class, there are no per-frame dispatchers to update for RGB, BGR, depth, intrinsics, or point clouds.

### Step 4: Update config

Set the camera type in your config YAML:

```yaml
cameras:
  hand:
    serial: "your-serial"
    type: your_camera
  external:
    serial: "your-external-serial"
    type: your_camera
```

### Step 5: Verify

```bash
# Check camera feed for hand camera
viz-gripper-cam

# Check camera feed for external camera
viz-scene
```
