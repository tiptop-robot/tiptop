# Running in Simulation

TiPToP supports two simulation workflows using [droid-sim-evals](https://github.com/tiptop-robot/droid-sim-evals):

1. **Websocket server mode** — the simulator connects to a running TiPToP server, sends an observation, and receives a trajectory in real time.
2. **Offline H5 mode** — observations are captured to an H5 file ahead of time, TiPToP processes the file and writes a trajectory, and the simulator replays it independently. Use this for batch evaluation or to decouple data collection from planning.

```{important}
This guide assumes you have completed the [installation instructions](installation.md) for TiPToP and M2T2. FoundationStereo is not required — we use ground truth depth from the simulator.
```

## Setup

The [droid-sim-evals](https://github.com/tiptop-robot/droid-sim-evals) simulator requires Ubuntu 22.04 or later (IsaacLab does not support Ubuntu 20.04). The simulator uses [`uv`](https://github.com/astral-sh/uv) for dependency management. Install it if you don't have it already:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone the simulator repo (including submodules) and install its dependencies:

```bash
cd $TIPTOP_DIR
git clone --recurse-submodules https://github.com/tiptop-robot/droid-sim-evals.git
cd droid-sim-evals
uv sync
```

Next, download the simulation assets:

```bash
curl -O https://pi-sim-assets.s3.us-east-1.amazonaws.com/assets.zip 
unzip assets.zip
```

This downloads **5 scenes** (scene IDs 1–5) as USD files into the `assets/` directory. Each scene has multiple variants that place objects in different configurations:

| Scene | Variants     |
|-------|--------------|
| 1     | 10 (0–9)     |
| 2     | 10 (0–9)     |
| 3     | 11 (0–8, 10–11) |
| 4     | 10 (0–9)     |
| 5     | 10 (0–9)     |

To see what the scenes look like, and also an example natural language command in each scene, look at the [README for the sim evals repo](https://github.com/tiptop-robot/droid-sim-evals).


Set your Google API key (required for Gemini, which TiPToP uses for object detection and task parsing). You can generate one following the instructions at [https://ai.google.dev/gemini-api/docs/api-key](https://ai.google.dev/gemini-api/docs/api-key).

```bash
export GOOGLE_API_KEY="your-api-key"
```

In all simulation workflows, M2T2 must be running. Start it first:

```bash
cd $TIPTOP_DIR/M2T2
pixi run server
```

## Websocket server mode

In this mode, the simulator connects to TiPToP over a websocket, sends an observation, and receives a full trajectory in response.

Start the TiPToP websocket server from the `tiptop` directory:

```bash
cd $TIPTOP_DIR/tiptop
pixi run tiptop-server
```

Then, run the simulator in headless mode:

```bash
cd $TIPTOP_DIR/droid-sim-evals
uv run tiptop_eval.py --scene <scene_id> --variant <variant_id> --instruction "<instruction>"
```

Replace `<scene_id>` with the scene number (1–5) and `<variant_id>` with the variant number (e.g. 0).

To visualize execution in IsaacLab, add `--headless False`:

```bash
uv run tiptop_eval.py --scene <scene_id> --variant <variant_id> --instruction "<instruction>" --headless False
```

## Offline H5 mode

In this mode, observations are stored in an H5 file ahead of time. TiPToP reads the file, runs perception and planning, and writes a trajectory that the simulator can replay independently.

### Generate an H5 observation file

An example H5 file is provided at `droid-sim-evals/tiptop_assets/tiptop_scene1_obs.h5` to get started quickly.

To generate your own from a specific scene and variant:

```bash
cd $TIPTOP_DIR/droid-sim-evals
uv run save_h5_obs.py --scene <scene_id> --variant <variant_id> --output tiptop_assets/your-save-path.h5
```

### Run TiPToP on the H5 file

Run TiPToP on the H5 observation file to produce a trajectory. An example trajectory is available at `droid-sim-evals/tiptop_assets/tiptop_scene1_plan.json`.

To run on your own files:

```bash
cd $TIPTOP_DIR/tiptop
pixi shell
tiptop-h5 \
  --h5-path /path/to/input/obs.h5 \
  --task-instruction "your instruction here" \
  --output-dir /path/to/output/dir
```

### Replay the trajectory in the simulator

Once TiPToP has written the trajectory, replay it in the simulator. To use the example trajectory:

```bash
cd $TIPTOP_DIR/droid-sim-evals
uv run replay_json_traj.py --json-path tiptop_assets/tiptop_scene1_plan.json --scene 1 --variant 0
```

To visualize in IsaacLab, add `--headless False`:

```bash
uv run replay_json_traj.py --json-path tiptop_assets/tiptop_scene1_plan.json --scene 1 --variant 0 --headless False
```
