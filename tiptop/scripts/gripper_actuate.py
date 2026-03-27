import logging

import tyro

from tiptop.utils import get_robot_client, setup_logging

_log = logging.getLogger(__name__)


def gripper_open(speed: float = 1.0, force: float = 0.1):
    """Open the Robotiq gripper."""
    client = get_robot_client()
    client.open_gripper(speed=speed, force=force)
    _log.info(f"Gripper opened with speed={speed}, force={force}")


def gripper_close(speed: float = 1.0, force: float = 0.1):
    """Close the Robotiq gripper."""
    client = get_robot_client()
    client.close_gripper(speed=speed, force=force)
    _log.info(f"Gripper closed with speed={speed}, force={force}")


def gripper_open_entrypoint():
    setup_logging()
    tyro.cli(gripper_open)


def gripper_close_entrypoint():
    setup_logging()
    tyro.cli(gripper_close)
