import json
import logging
import sys
from contextlib import contextmanager
from functools import cache
from pathlib import Path

import numpy as np
from bamboo.client import BambooFrankaClient
from cutamp.robots import (
    RerunRobot,
    load_fr3_franka_rerun,
    load_fr3_robotiq_rerun,
    load_franka_rerun,
    load_panda_robotiq_rerun,
    load_ur5_rerun,
)
from jaxtyping import Bool
from PIL import Image

from tiptop.config import config_dir, tiptop_cfg
from tiptop.ur5.ur5_client import UR5Client

gripper_mask_path = config_dir / "assets" / "gripper_mask.png"

REQUIRED_CUTAMP_VERSION = "0.0.4"


def check_cutamp_version() -> None:
    """Raise RuntimeError if the installed cuTAMP version does not match REQUIRED_CUTAMP_VERSION."""
    try:
        import cutamp

        version = cutamp.__version__
    except AttributeError:
        version = "<0.0.2"
    if version != REQUIRED_CUTAMP_VERSION:
        raise RuntimeError(
            f"cuTAMP version mismatch: required {REQUIRED_CUTAMP_VERSION}, found {version}. "
            "Please run: pixi run install-cutamp"
        )


class ServerHealthCheckError(Exception):
    """Raised when a server health check fails."""


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


@cache
def get_tiptop_cache_dir() -> Path:
    import tiptop

    top_level_dir = Path(tiptop.__file__).parent
    cache_dir = top_level_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@cache
def get_bamboo_client() -> BambooFrankaClient:
    """Get a BambooFrankaClient instance using the current config."""
    cfg = tiptop_cfg()

    if cfg.robot.type in {"fr3_robotiq", "panda_robotiq"}:
        gripper = "robotiq"
    elif cfg.robot.type in {"panda", "fr3"}:
        gripper = "franka"
    else:
        raise ValueError(f"Unknown robot type {cfg.robot.type}.")

    return BambooFrankaClient(
        control_port=cfg.robot.port, server_ip=cfg.robot.host, gripper_port=cfg.robot.gripper_port, gripper_type=gripper
    )


RobotClient = BambooFrankaClient | UR5Client


def get_robot_client() -> RobotClient:
    """Get a RobotClient instance for the robot."""
    cfg = tiptop_cfg()
    if cfg.robot.type in {"fr3_robotiq", "panda_robotiq", "panda", "fr3"}:
        return get_bamboo_client()
    elif cfg.robot.type == "ur5":
        from tiptop.ur5.ur5_client import get_ur5_client

        return get_ur5_client()
    else:
        raise ValueError(f"Unknown robot type: {cfg.robot.type}")


def get_robot_rerun(robot_type: str | None = None) -> RerunRobot:
    """Get a RerunRobot instance for the robot."""
    if robot_type is None:
        robot_type = tiptop_cfg().robot.type
    if robot_type == "fr3_robotiq":
        return load_fr3_robotiq_rerun()
    elif robot_type == "panda_robotiq":
        return load_panda_robotiq_rerun()
    elif robot_type == "panda":
        return load_franka_rerun()
    elif robot_type == "fr3":
        return load_fr3_franka_rerun()
    elif robot_type == "ur5":
        return load_ur5_rerun()
    else:
        raise ValueError(f"Unknown robot type: {robot_type}")


def load_gripper_mask() -> Bool[np.ndarray, "h w"]:
    """Load the binary mask for the gripper."""
    gripper_mask_img = Image.open(gripper_mask_path)
    gripper_mask = np.array(gripper_mask_img).astype(bool)
    return gripper_mask


def setup_logging(level: int = logging.INFO):
    # Ensure stdout and stderr use UTF-8 encoding to handle Unicode characters
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

    # Define a custom formatter with ANSI escape codes for colors
    class CustomFormatter(logging.Formatter):
        level_colors = {
            "DEBUG": "\033[34m",  # Blue
            "INFO": "\033[32m",  # Green
            "WARNING": "\033[33m",  # Yellow
            "ERROR": "\033[31m",  # Red
            "CRITICAL": "\033[35m",  # Magenta
        }
        reset = "\033[0m"

        def format(self, record):
            # Format without modifying the record - build colored levelname locally
            log_color = self.level_colors.get(record.levelname, "")
            colored_levelname = f"{log_color}{record.levelname}{self.reset}"

            # Manually build the formatted string with colors
            log_time = self.formatTime(record, self.datefmt)
            message = f"{log_time} - {record.name} - {colored_levelname} - {record.getMessage()}"
            if record.exc_info:
                message = message + "\n" + self.formatException(record.exc_info)
            return message

    # Define the log format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = CustomFormatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")

    # Configure the root logger (force reconfiguration)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove only default StreamHandlers (stdout/stderr), keep other handlers (files, etc.)
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler) and handler.stream in (sys.stdout, sys.stderr):
            root_logger.removeHandler(handler)

    # Pin explicitly so a later add_file_handler() bumping root doesn't change console
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.setLevel(level)
    root_logger.addHandler(handler)

    # Bamboo can be INFO level
    logging.getLogger("bamboo").setLevel(logging.INFO)

    # Silence noisy third-party library loggers
    noisy_modules = (
        "asyncio",
        "charset_normalizer",
        "curobo",
        "google_genai",
        "hpack",
        "httpcore",
        "httpx",
        "hydra",
        "matplotlib",
        "PIL",
        "trimesh",
        "urllib3",
        "yourdfpy",
    )
    for module in noisy_modules:
        logging.getLogger(module).setLevel(logging.WARNING)


def add_file_handler(log_file: Path, level: int = logging.DEBUG) -> logging.FileHandler:
    """Add a file handler to the root logger.

    Args:
        log_file: Path to the log file to create
        level: Logging level for the file handler

    Returns:
        The FileHandler instance so it can be removed later
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Create file handler with plain formatting (no colors) and UTF-8 encoding
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(level)

    # Use same format as console but without colors
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(formatter)

    # Lower root if the file handler is more verbose, so records reach it
    root_logger = logging.getLogger()
    if level < root_logger.level:
        root_logger.setLevel(level)
    root_logger.addHandler(file_handler)
    return file_handler


def remove_file_handler(handler: logging.FileHandler):
    """Remove a file handler from the root logger and close it.

    Args:
        handler: The FileHandler to remove
    """
    root_logger = logging.getLogger()
    root_logger.removeHandler(handler)
    handler.close()


def print_tiptop_banner():
    """Print TiPToP ASCII art banner with ice cream."""
    banner = """
    ████████╗ ██╗ ██████╗ ████████╗  ██████╗  ██████╗
    ╚══██╔══╝ ██║ ██╔══██╗╚══██╔══╝ ██╔═══██╗ ██╔══██╗
       ██║    ██║ ██████╔╝   ██║    ██║   ██║ ██████╔╝
       ██║    ██║ ██╔═══╝    ██║    ██║   ██║ ██╔═══╝
       ██║    ██║ ██║        ██║    ╚██████╔╝ ██║
       ╚═╝    ╚═╝ ╚═╝        ╚═╝     ╚═════╝  ╚═╝
    ═══════════════════════════════════════════════════
       TiPToP is a Planner That just works on Pixels
    ═══════════════════════════════════════════════════
    """
    print(banner)


@contextmanager
def patch_log_level(logger_name: str, log_level: int):
    """Context manager to temporarily override the log level."""
    logger = logging.getLogger(logger_name)
    og_level = logger.getEffectiveLevel()
    logger.setLevel(log_level)
    yield
    logger.setLevel(og_level)
