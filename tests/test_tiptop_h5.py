"""Integration tests for the TiPToP H5 pipeline. Run with: pixi run test-integration"""

import json

import pytest

# (h5 filename, task instruction, expect_planning_success)
SCENES = [
    ("tiptop_scene1_obs.h5", "Put the Rubik's cube in the bowl.", True),
    ("tiptop_scene2_obs.h5", "Put the can in the mug.", True),
    ("tiptop_scene3_obs.h5", "Put the banana in the bin.", True),
    ("tiptop_scene4_obs.h5", "Put the banana in the bowl.", True),
    ("tiptop_scene5_obs.h5", "Put 3 blocks in the bowl.", True),
]


@pytest.mark.integration
@pytest.mark.parametrize(
    "h5_filename, task_instruction, expect_planning_success",
    SCENES,
    ids=[f"scene{i}" for i in range(1, len(SCENES) + 1)],
)
def test_tiptop_h5_pipeline(tmp_path, h5_assets, h5_filename, task_instruction, expect_planning_success):
    h5_path = h5_assets / h5_filename
    assert h5_path.exists(), f"Test asset not found: {h5_path}"

    # Local import to avoid slow transitive imports affecting other tests
    from tiptop.tiptop_h5 import run_tiptop_h5

    run_tiptop_h5(
        h5_path=str(h5_path),
        task_instruction=task_instruction,
        output_dir=str(tmp_path),
        max_planning_time=10.0,
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

    assert metadata["task_instruction"] == task_instruction
    assert metadata["version"] == "1.0.0"

    # Perception must have found at least one object
    grounded_atoms = metadata["perception"]["grounded_atoms"]
    assert len(grounded_atoms) > 0, "Perception found no grounded atoms"

    # Core perception outputs must exist
    perception_dir = save_dir / "perception"
    assert (perception_dir / "pointcloud.ply").exists(), "pointcloud.ply missing"
    assert (perception_dir / "cutamp_env.pkl").exists(), "cutamp_env.pkl missing"
    assert (perception_dir / "grasps.pt").exists(), "grasps.pt missing"

    # Planning success — some cluttered scenes may not find a plan within the time budget
    planning = metadata["planning"]
    if expect_planning_success:
        assert planning["success"], f"Planning failed: {planning.get('failure_reason')}"
        assert (save_dir / "tiptop_plan.json").exists(), "tiptop_plan.json missing despite planning success"
    else:
        assert not planning["success"], "Planning unexpectedly succeeded"
        assert not (save_dir / "tiptop_plan.json").exists(), "tiptop_plan.json unexpectedly created"
