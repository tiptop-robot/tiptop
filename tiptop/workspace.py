from dataclasses import replace
from functools import cache

from curobo.geom.types import Cuboid
from curobo.types.base import TensorDeviceType
from cutamp.envs.utils import unit_quat
from cutamp.utils.rerun_utils import log_curobo_mesh_to_rerun

from tiptop.config import tiptop_cfg
from tiptop.utils import get_robot_rerun

tensor_args = TensorDeviceType()

# The workspace to motion plan against for collisions


def fr3_workspace() -> tuple[Cuboid, ...]:
    vention_table = Cuboid(
        "vention",
        dims=[0.86, 1.24, 0.7],
        pose=[0.19 - 0.86 / 2, 0.525, -0.35, *unit_quat],
        color=[2, 48, 117],
    )
    nishanth = Cuboid(
        "nishanth",
        dims=[2.5, 0.25, 1.8],
        pose=[-0.25, -0.55, 0.0, *unit_quat],
        color=[255, 0, 255],
    )
    ipad = Cuboid(
        "ipad",
        dims=[0.25, 0.3, 0.2],
        pose=[0.05, 0.25, 0.1, *unit_quat],
        color=[255, 0, 0],
    )
    camera = Cuboid(
        name="camera",
        dims=[0.36, 0.2, 0.76],
        pose=[0.0, 0.525, 0.76 / 2, *unit_quat],
        color=[0, 87, 217],
    )
    obstacles = (vention_table, camera, nishanth, ipad)
    return obstacles


def ur5_workspace() -> tuple[Cuboid, ...]:
    pedestal_table = Cuboid(
        "pedestal_table",
        dims=[0.48, 0.87, 0.02],
        pose=[-0.05, 0, -0.01, *unit_quat],
        color=[222, 184, 135],
    )
    # ceiling is more for avoiding crazy large motions
    ceiling = Cuboid("ceiling", dims=[1.5, 2.0, 0.01], pose=[0.5, 0.0, 1.0, *unit_quat], color=[255, 255, 255])

    # humans work to the left and right of the UR5...
    wall_left = Cuboid("wall_left", dims=[1.5, 0.1, 1.0], pose=[0.0, 0.8, 0.2, *unit_quat], color=[225, 225, 225])
    wall_right = Cuboid("wall_right", dims=[1.5, 0.1, 1.0], pose=[0.0, -0.8, 0.2, *unit_quat], color=[225, 225, 225])
    obstacles = (pedestal_table, ceiling, wall_left, wall_right)
    return obstacles


@cache
def workspace_cuboids() -> tuple[Cuboid, ...]:
    """Return workspace cuboids for the configured robot, with names prefixed by 'workspace_' to avoid collisions with objects discovered by actual perception."""
    cfg = tiptop_cfg()
    if cfg.robot.type == "fr3_robotiq":
        cuboids = fr3_workspace()
    elif cfg.robot.type == "panda":
        cuboids = fr3_workspace()
    elif cfg.robot.type == "fr3":
        cuboids = fr3_workspace()
    elif cfg.robot.type == "panda_robotiq":
        cuboids = fr3_workspace()
    elif cfg.robot.type == "ur5":
        cuboids = ur5_workspace()
    else:
        raise ValueError(f"Unknown robot type: {cfg.robot.type}")
    return tuple(replace(c, name=f"workspace_{c.name}") for c in cuboids)


if __name__ == "__main__":
    import rerun as rr

    rr.init("robot_workspace", spawn=True)
    get_robot_rerun()
    for obj in workspace_cuboids():
        log_curobo_mesh_to_rerun(f"world/{obj.name}", obj.get_mesh(), static_transform=True)
