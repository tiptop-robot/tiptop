import logging

import tyro

from tiptop.config import tiptop_cfg
from tiptop.utils import setup_logging

_log = logging.getLogger(__name__)
_cfg_time_dilation_factor = tiptop_cfg().robot.time_dilation_factor


def go_to_home_entrypoint():
    def _entrypoint(time_dilation_factor: float = _cfg_time_dilation_factor):
        """move robot to home joint positions."""
        from tiptop.motion_planning import go_to_home

        setup_logging()
        _log.info("Going to home configuration")
        go_to_home(time_dilation_factor=time_dilation_factor)

    tyro.cli(_entrypoint)


def go_to_capture_entrypoint():
    def _entrypoint(time_dilation_factor: float = _cfg_time_dilation_factor):
        """Move robot to capture joint positions."""
        from tiptop.motion_planning import go_to_capture

        setup_logging()
        _log.info("Going to capture configuration")
        go_to_capture(time_dilation_factor=time_dilation_factor)

    tyro.cli(_entrypoint)
