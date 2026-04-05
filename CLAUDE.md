# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TiPToP is a Task and Motion Planning (TAMP) system for robots with a modular architecture consisting of three sequential components:

1. **Perception Module** - Processes RGB images and natural language to build object-centric 3D representations using FoundationStereo (depth), Gemini Robotics-ER 1.5 (VLM), SAM-2 (segmentation), and M2T2 (grasp detection)
2. **Planning Module** - Uses cuTAMP, a GPU-parallelized TAMP solver that optimizes thousands of candidate pick-and-place plans in parallel
3. **Execution Module** - Executes trajectories using joint impedance control

More details: https://tiptop-robot.github.io/

## Development Commands

### Installation

We use [pixi](https://pixi.prefix.dev/) for environment and dependency management:

```bash
# Install all dependencies (PyTorch, cuRobo deps, etc.)
pixi install

# Install external dependencies (ZED SDK, cuRobo, cuTAMP)
pixi run setup-all
```

### Activating the Environment

All development and CLI commands must be run inside the pixi shell:

```bash
# Activate the pixi environment
pixi shell

# Now you can run all tiptop commands directly
```

### CLI Scripts

The package provides several CLI entry points (defined in `pyproject.toml`). First activate the environment with `pixi shell`, then run commands directly:

```bash
# Setup and calibration
compute-gripper-mask       # Gemini + SAM2 gripper detection

# Visualization
viz-calibration            # Rerun visualization of camera calibration
viz-gripper-cam            # View gripper camera feed
viz-scene                  # Visualize scene

# Robot control
go-home                    # Move robot to home configuration
go-to-capture              # Move robot to image capture configuration
gripper-open               # Open gripper
gripper-close              # Close gripper
```

### Documentation

Documentation tasks can be run directly with `pixi run`:

```bash
# Install Sphinx and doc dependencies
pixi run docs-install

# Build HTML documentation
pixi run docs-build

# Serve with live reload (for development)
pixi run docs-serve

# Clean build artifacts
pixi run docs-clean
```

The documentation is automatically built and deployed via ReadTheDocs using Python 3.11.

## Architecture Overview

### Package Structure

The `tiptop/` package is organized into focused modules:

- **`config/`** - Centralized configuration management using OmegaConf
  - `tiptop.yml` contains robot, camera, and perception service settings
  - `calibration_info.json` stores camera-to-end-effector transforms
  - `tiptop_cfg()` provides lazy-loaded config with CLI override support

- **`perception/`** - Perception pipeline with microservices architecture
  - `foundation_stereo.py` - Depth estimation via HTTP to remote server
  - `m2t2.py` - Grasp detection via HTTP to remote server
  - `sam2.py` - Object segmentation (local or remote)
  - `zed_camera.py` - ZED stereo camera interface
  - All services support both sync and async (`_async`) APIs

- **`scripts/`** - CLI entry points for setup, visualization, and robot control

- **`motion_planning.py`** - Motion planning using cuRobo's MotionGen

- **`warm_start.py`** - Initializes and warm-ups IK solver and MotionGen

- **`tiptop_demo.py`** - Main end-to-end demo pipeline

### Data Flow

The system follows this pipeline:

```
1. Image Capture
   └─> ZED camera (RGB-D stereo)

2. Perception (Async where possible)
   ├─> FoundationStereo (depth refinement)
   ├─> Depth-to-3D projection (point cloud)
   ├─> Gemini (object detection from RGB)
   ├─> SAM2 (object segmentation)
   └─> M2T2 (grasp generation from point cloud)

3. Task Planning
   ├─> Create TAMP environment (objects + surfaces)
   ├─> Parse instruction with Gemini
   └─> Run cuTAMP for motion plans

4. Execution
   ├─> IK solver for grasp poses
   ├─> MotionGen for collision-aware trajectories
   └─> Bamboo Franka client for robot control

5. Visualization
   └─> Rerun for real-time 3D visualization
```

### Key Integration Points

- **Bamboo Franka Controller** (`bamboo-franka-controller`) - Joint position queries and trajectory execution with impedance control
- **cuRobo** - GPU-accelerated motion planning and collision checking
- **cuTAMP** - Task-level planning with stream-based constraint solving (external dependency, not in `pyproject.toml`)
- **Rerun** - 3D visualization of robot state, sensor data, and plans
- **OmegaConf** - Configuration management with YAML files and CLI overrides

### Coordinate Frame Conventions

Transformation matrices follow the pattern `target_from_source`:
- `world_from_cam` - Transforms camera coordinates to world coordinates
- `obj_from_grasp` - Transforms grasp frame to object frame
- `gripper_from_cam` - Transforms camera frame to gripper frame

Calibration data in `calibration_info.json` stores 4x4 homogeneous transforms for camera extrinsics.

## Microservices Architecture

TiPToP uses a microservices-based architecture where external components run as separate HTTP servers:

### M2T2 (Grasp Detection)
Generates 6-DOF grasps from point clouds. Installation:

```bash
cd $TiPToP_DIR
git clone git@github.com:williamshen-nz/m2t2-private.git M2T2
cd M2T2
conda env create -n TiPToP-m2t2 python=3.10 -y
conda activate TiPToP-m2t2

# Install PyTorch matching your CUDA version (check with: nvcc --version)
# Example for CUDA 12.8:
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
  --index-url https://download.pytorch.org/whl/cu128

pip install pointnet2_ops/
pip install -r requirements.txt
pip install .

# Download weights
git clone https://huggingface.co/wentao-yuan/m2t2 weights
```

### FoundationStereo (Depth Estimation)
Predicts depth maps from stereo camera images (Zed). Installation:

```bash
cd $TiPToP_DIR
git clone git@github.com:williamshen-nz/FoundationStereo-private.git FoundationStereo
cd FoundationStereo
conda env create -f environment.yml
conda run -n TiPToP-foundation_stereo pip install flash-attn
conda activate TiPToP-foundation_stereo

# Download checkpoints (non-commercial version)
pip install gdown
gdown <link-url>
```

## Coding Style and Principles

### Function-based Design
- **Prefer functions over classes**: Use standalone functions and functional programming patterns where possible
- Only use classes when managing stateful operations (e.g., `ParticleInitializer`, `RolloutFunction`, `TAMPWorld`)
- Classes that act as callable function containers should implement `__call__()` to maintain functional interface

### Documentation
- Use concise single-line docstrings for simple functions
- Only add detailed docstrings when the function behavior is non-trivial or requires explanation
- Avoid redundant documentation that simply restates the function name

### Code Organization
- Break complex logic into focused, single-purpose functions
- Keep functions cohesive - extract logical units like transformation computations or data processing
- Use descriptive function names that clearly indicate purpose (e.g., `get_world_from_gripper`, `transform_pointcloud_to_world`)
- Avoid over-fragmenting code into too many tiny functions

### Naming Conventions
- Use descriptive variable names that indicate transformations: `world_from_cam`, `obj_from_grasp`, `gripper_from_cam`
- Follow the pattern `target_from_source` for transformation matrices
- Use `_fn` suffix for higher-order functions (e.g., `grasp_to_mat4x4_fn`)

### Type Hints
- Use type hints for function parameters and return values
- Leverage `jaxtyping` for tensor dimensions (e.g., `Float[torch.Tensor, "batch dim"]`)
- Define TypedDict classes for complex return structures (see `Rollout` in `rollout.py`)

### Error Handling
- Validate inputs at the start of functions with clear error messages
- Use `raise ValueError` or `raise RuntimeError` with descriptive messages
- Include context in error messages (e.g., which object/parameter caused the issue)

### Imports
- Group imports: standard library, third-party, local imports
- Use absolute imports from `tiptop` package root
- Avoid wildcard imports
- Avoid local imports inside functions for standard modules; import at module level

### String Formatting
- Always use f-strings for string formatting; avoid `%s` or `.format()`

### Code Quality Tools
- **Ruff** configured with line length 120
- Pre-commit hooks enforce import sorting (`ruff --select I --fix`) and formatting (`ruff-format`)
- Hooks only apply to `tiptop/` directory
- `pre-commit` is a pixi dependency and is registered automatically via `pixi run setup-all`

## Working Style

- **Discuss before implementing** — When the approach isn't obvious (e.g., where to put docs, how to structure config), talk through options before writing code
- **Comments and docstrings must be accurate** — Don't write comments that describe implementation details irrelevant to the reader or are vaguely wrong. If a comment doesn't add real information, drop it
- **Don't add dead code paths** — If every case goes down the same branch, don't add the other branch "just in case." Keep what's tested, remove what isn't

## Documentation Style Guide

When working on documentation:
- Be concise without making the writing feel unnatural
- Handle edge cases appropriately but not exhaustively
- Avoid excessive verbosity
- Use MyST-Parser Markdown features (colon fences, admonitions, etc.)
- Follow the existing black & white theme aesthetic
- API documentation will be auto-generated from docstrings once source code is added

## Key Concepts

The core TAMP concepts used throughout the codebase:

- **Streams** - Generate continuous values during planning (poses, grasps, paths)
- **Environment** - Manages world state including objects and obstacles
- **Robot** - Represents robot kinematics and capabilities
- **Problem** - Combines environment, robot, and goal into a complete TAMP problem
- **Solution** - Contains success status, action sequence, and cost

The planning approach is iterative: symbolic search finds candidate task plans, motion validation checks feasibility, and refinement occurs when motions fail.
