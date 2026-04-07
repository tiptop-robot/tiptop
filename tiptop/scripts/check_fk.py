"""Diagnostic script: compute cuRobo FK at given joint positions and print the EE pose.

Usage (from pixi shell):
    check-fk --q 0.0 -0.628 0.0 -2.513 0.0 1.885 0.0

Compare the printed EE pose against what MolmoSpaces reports for the same joint config
to detect FK model mismatches between cuRobo and the sim.
"""

import logging

import numpy as np
import tyro
from curobo.types.base import TensorDeviceType

from tiptop.motion_planning import build_curobo_solvers
from tiptop.planning import build_tamp_config
from tiptop.utils import setup_logging


def check_fk(q: list[float]) -> None:
    """Print the cuRobo FK result for the given joint positions.

    Args:
        q: Joint positions (7 values for FR3).
    """
    setup_logging()
    log = logging.getLogger(__name__)

    cfg_q = np.array(q, dtype=np.float32)
    log.info(f"Computing FK for q = {cfg_q.tolist()}")

    from tiptop.config import tiptop_cfg
    cfg = tiptop_cfg()

    tamp_config = build_tamp_config(
        num_particles=1,
        max_planning_time=60.0,
        opt_steps=1,
        robot_type=cfg.robot.type,
        time_dilation_factor=cfg.robot.time_dilation_factor,
    )
    _, motion_gen, _ = build_curobo_solvers(
        num_particles=1,
        num_spheres=tamp_config.coll_n_spheres,
        include_workspace=False,
    )

    tensor_args = TensorDeviceType()
    q_pt = tensor_args.to_device(cfg_q)
    ee_pose = motion_gen.kinematics.get_state(q_pt).ee_pose

    pos = ee_pose.position.cpu().numpy().squeeze()
    quat = ee_pose.quaternion.cpu().numpy().squeeze()  # [qw, qx, qy, qz]

    print("\n=== cuRobo FK result ===")
    print(f"  EE link : {cfg.robot.type} -> grasp_frame")
    print(f"  Position: x={pos[0]:.4f}  y={pos[1]:.4f}  z={pos[2]:.4f}  (metres, robot base frame)")
    print(f"  Quat    : qw={quat[0]:.4f}  qx={quat[1]:.4f}  qy={quat[2]:.4f}  qz={quat[3]:.4f}")
    print()
    print("Compare this to MolmoSpaces's reported EE position at the same joint config.")
    print("A mismatch confirms a FK model difference between cuRobo and the sim.")


def entrypoint():
    tyro.cli(check_fk)


if __name__ == "__main__":
    entrypoint()
