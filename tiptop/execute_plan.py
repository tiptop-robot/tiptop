import logging
import time

from tiptop.utils import RobotClient, get_robot_client

_log = logging.getLogger(__name__)


class ExecutionFailure(Exception):
    """Failure in executing plan on robot."""


def execute_cutamp_plan(cutamp_plan: list[dict], client: RobotClient | None = None) -> None:
    """Execute the plan from cuTAMP on the real robot."""
    if client is None:
        client = get_robot_client()

    start_time = time.perf_counter()
    for step, action_dict in enumerate(cutamp_plan):
        action_start_time = time.perf_counter()
        action_type = action_dict["type"]
        action_label = action_dict["label"]

        # Form log message
        msg = f"Executing step {step + 1}/{len(cutamp_plan)}: {action_label}. Action type: {action_dict['type']}"
        if action_type == "gripper":
            msg += f" ({action_dict['action']})"
        elif action_type == "trajectory":
            msg += f" ({len(action_dict['plan'].position)} waypoints)"
        else:
            raise ValueError(f"Unknown action type in cuTAMP plan: {action_dict['type']}")
        _log.info(msg)

        # Now execute the actions
        if action_type == "gripper":
            action = action_dict["action"]
            if action == "open":
                result = client.open_gripper(speed=1.0)
            elif action == "close":
                result = client.close_gripper(speed=1.0)
            else:
                raise ValueError(f"Unknown gripper action: {action}")

        elif action_type == "trajectory":
            # Extract joint position and velocity waypoints for the trajectory
            waypoints = action_dict["plan"].position.cpu().numpy()
            velocities = action_dict["plan"].velocity.cpu().numpy()
            timings = [action_dict["dt"]] * len(waypoints)
            result = client.execute_joint_impedance_path(
                joint_confs=waypoints, joint_vels=velocities, durations=timings
            )

        else:
            raise ValueError(f"Unexpected action type in cuTAMP plan: {action_dict['type']}")

        # Raise error if execution failed
        if result is None:
            raise RuntimeError("Fatal error: result should not be None")
        if not result["success"]:
            raise ExecutionFailure(result["error"])

        action_duration = time.perf_counter() - action_start_time
        _log.debug(f"Executing {action_type} action took {action_duration:.2f}s")

    # Now we're done executing plan open-loop without any failures on the controller side
    duration = time.perf_counter() - start_time
    _log.info(f"Real robot execution took {duration:.2f}s")
