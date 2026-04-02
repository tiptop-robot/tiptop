"""Integration test for the TiPToP H5 pipeline.

Requires all external perception services (FoundationStereo, SAM2, M2T2, Gemini) to be running
and API keys to be configured. Run explicitly with:

    pytest tests/test_tiptop_h5.py -m integration -v
"""

import json
from pathlib import Path

import pytest

H5_PATH = Path(__file__).parent / "tiptop_obs.h5"
TASK_INSTRUCTION = "Put the cube in the bowl"


@pytest.mark.integration
def test_tiptop_h5_pipeline(tmp_path):
    from tiptop.tiptop_h5 import run_tiptop_h5

    run_tiptop_h5(
        h5_path=str(H5_PATH),
        task_instruction=TASK_INSTRUCTION,
        output_dir=str(tmp_path),
        cutamp_visualize=False,
        rr_spawn=False,
    )

    # Find the timestamped output subdirectory
    output_dirs = list(tmp_path.iterdir())
    assert len(output_dirs) == 1, f"Expected one output directory, got: {output_dirs}"
    save_dir = output_dirs[0]

    # Metadata should always be written, even if planning fails
    metadata_path = save_dir / "metadata.json"
    assert metadata_path.exists(), "metadata.json was not written"
    with open(metadata_path) as f:
        metadata = json.load(f)

    assert metadata["task_instruction"] == TASK_INSTRUCTION
    assert metadata["version"] == "1.0.0"

    # Perception must have found at least one object
    grounded_atoms = metadata["perception"]["grounded_atoms"]
    assert len(grounded_atoms) > 0, "Perception found no grounded atoms"

    # Core perception outputs must exist
    perception_dir = save_dir / "perception"
    assert (perception_dir / "pointcloud.ply").exists(), "pointcloud.ply missing"
    assert (perception_dir / "cutamp_env.pkl").exists(), "cutamp_env.pkl missing"
    assert (perception_dir / "grasps.pt").exists(), "grasps.pt missing"

    # Plan should be found for this scenario
    assert metadata["planning_success"], (
        f"Planning failed: {metadata.get('planning_failure_reason')}"
    )
    assert (save_dir / "tiptop_plan.json").exists(), "tiptop_plan.json missing despite planning success"
