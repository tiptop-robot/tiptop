from __future__ import annotations

import logging
import time
from functools import cache
from typing import List

import numpy as np

_log = logging.getLogger(__name__)

_MAX_INITIAL_ERROR = np.deg2rad(1.0)  # 1 degree


class UR5Client:
    """Client for UR with Robotiq gripper."""

    def __init__(self, ip_addr: str):
        try:
            import rtde_control
            import rtde_receive
        except ImportError as e:
            raise ImportError("Missing UR RTDE dependencies. You can run `pip install ur_rtde`") from e

        from tiptop.ur5.robotiq_gripper import RobotiqGripper

        self.rtde_r = rtde_receive.RTDEReceiveInterface(ip_addr)
        try:
            self.rtde_c = rtde_control.RTDEControlInterface(ip_addr)
        except RuntimeError as e:
            _log.warning("First attempt to connect to UR failed. Retrying...", exc_info=True)
            try:
                self.rtde_c = rtde_control.RTDEControlInterface(ip_addr)  # Try again
            except RuntimeError:
                _log.error("Second attempt to connect to UR failed.", exc_info=True)
                raise  # Reraise the error after second failure

        self.gripper = RobotiqGripper()
        self.gripper.connect(ip_addr, 63352)
        if not self.gripper.is_active():
            _log.debug("Activating Robotiq gripper")
            self.gripper.activate()

    def get_joint_positions(self) -> List[float]:
        return list(self.rtde_r.getActualQ())

    def open_gripper(self, speed: float = 1.0, force: float = 0.1):
        """Blocking call to open gripper."""
        speed = int(speed * 255)
        force = int(force * 255)
        result = self.gripper.move_and_wait_for_pos(self.gripper.get_open_position(), speed=speed, force=force)
        return {"success": result[0]}

    def close_gripper(self, speed: float = 1.0, force: float = 0.1):
        """Blocking call to close gripper."""
        speed = int(speed * 255)
        force = int(force * 255)
        result = self.gripper.move_and_wait_for_pos(self.gripper.get_closed_position(), speed=speed, force=force)
        return {"success": result[0]}

    def execute_joint_impedance_path(self, joint_confs, joint_vels, durations):
        """Execute a joint trajectory using servoJ with interpolation to 125Hz.

        Args:
            joint_confs: numpy array of shape [N, 6], joint angles in radians
            joint_vels: numpy array of shape [N, 6], joint velocities in rad/s
            durations: list of float, timestep durations

        Returns:
            dict: {"success": bool}
        """
        # Handle empty trajectories
        if len(joint_confs) == 0:
            return {"success": True}

        # Safety check: Validate array shapes
        if joint_confs.shape[1] != 6:
            raise ValueError(f"Expected 6 joints, got {joint_confs.shape[1]}")
        if joint_vels.shape != joint_confs.shape:
            raise ValueError(f"Velocity shape {joint_vels.shape} doesn't match position shape {joint_confs.shape}")

        # Safety check: Initial position should be close to current position
        q_current = np.array(self.rtde_r.getActualQ())
        q_initial = joint_confs[0]
        position_error = np.linalg.norm(q_current - q_initial)

        if position_error > _MAX_INITIAL_ERROR:
            raise RuntimeError(
                f"Initial trajectory position is too far from current position: "
                f"{position_error:.4f} rad (max allowed: {_MAX_INITIAL_ERROR:.4f} rad). "
                f"Current: {q_current}, Target: {q_initial}"
            )

        # Handle single waypoint - just move there
        if len(joint_confs) == 1:
            success = self.rtde_c.moveJ(joint_confs[0].tolist())
            return {"success": success}

        # Interpolate trajectory to 125Hz for smooth servoJ execution
        dt_traj = durations[0]
        dt_servo = 1.0 / 125.0  # 125Hz control rate
        t_original = np.arange(len(joint_confs)) * dt_traj
        t_total = t_original[-1]
        t_interp = np.arange(0, t_total, dt_servo)

        # Interpolate each joint
        joint_confs_interp = np.array([np.interp(t_interp, t_original, joint_confs[:, j]) for j in range(6)]).T

        # Add dwell waypoints at end for settling (prevents click sound at end)
        num_dwell = 30
        dwell_waypoints = np.repeat(joint_confs_interp[-1:, :], num_dwell, axis=0)
        joint_confs_interp = np.vstack([joint_confs_interp, dwell_waypoints])

        _log.info(f"Trajectory info: {len(joint_confs)} waypoints, dt={dt_traj:.4f}s, expected time={t_total:.4f}s")

        # servoJ parameters (see: https://www.universal-robots.com/articles/ur/programming/servoj-command/)
        velocity = 0.5  # Not used by servoJ but required
        acceleration = 0.5  # Not used by servoJ but required
        lookahead_time = 0.05  # Smoothing lookahead
        gain_normal = 400  # Position gain for trajectory following
        gain_settling = 300  # Slightly lower gain for smooth settling at end

        # Control loop!
        start_time = time.monotonic()
        num_trajectory_waypoints = len(joint_confs_interp) - num_dwell
        try:
            for i, q in enumerate(joint_confs_interp):
                loop_start = time.monotonic()
                # Use lower gain during settling phase for smoother stop
                gain = gain_settling if i >= num_trajectory_waypoints else gain_normal
                self.rtde_c.servoJ(q.tolist(), velocity, acceleration, dt_servo, lookahead_time, gain)

                # Check for protective stop or emergency stop
                safety_status = self.rtde_r.getSafetyMode()
                if safety_status not in [1]:  # 1 = NORMAL mode
                    raise RuntimeError(
                        f"Robot entered unsafe state (safety mode {safety_status}) at waypoint {i}/{len(joint_confs_interp)}"
                    )

                # Sleep for remaining time to maintain dt_servo rate
                elapsed = time.monotonic() - loop_start
                sleep_time = max(0, dt_servo - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        except Exception as e:
            # Ensure servo mode is stopped even if there's an error
            self.rtde_c.servoStop()
            raise RuntimeError(f"Trajectory execution failed: {e}") from e

        actual_time = time.monotonic() - start_time
        _log.info(f"Actual execution time: {actual_time:.4f}s (expected {t_total:.4f}s)")

        # Stop servo mode
        self.rtde_c.servoStop()

        # Check final position error
        desired_final = joint_confs[-1]
        actual_final = np.array(self.rtde_r.getActualQ())
        error = actual_final - desired_final
        error_norm = np.linalg.norm(error)
        _log.info(f"Final position error: {error_norm:.6f} rad (max joint error: {np.max(np.abs(error)):.6f} rad)")

        return {"success": True}

    def close(self):
        """Disconnect from the robot and gripper."""
        self.gripper.disconnect()
        self.rtde_c.disconnect()
        self.rtde_r.disconnect()


@cache
def get_ur5_client() -> UR5Client:
    from tiptop.config import tiptop_cfg

    return UR5Client(tiptop_cfg().robot.host)
