"""Modified from https://github.com/droid-dataset/droid/blob/main/droid/camera_utils/camera_readers/zed_camera.py"""

import logging
import time
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import aiohttp
import cv2
import numpy as np
from jaxtyping import Float, UInt8
from tqdm import tqdm

from tiptop.config import tiptop_cfg
from tiptop.perception.cameras.frame import Frame

_log = logging.getLogger(__name__)


def time_ms():
    return time.time_ns() // 1_000_000


@dataclass(frozen=True, kw_only=True)
class ZedFrame(Frame):
    """Frame from ZED which also includes raw stereo pair and point cloud."""

    left_bgra: UInt8[np.ndarray, "h w 4"]  # BGRA left uint8
    right_bgra: UInt8[np.ndarray, "h w 4"]  # BGRA right uint8
    pointcloud: Float[np.ndarray, "h w 4"] | None = None  # Hardware pointcloud float32 millimeters


@dataclass(frozen=True)
class ZedIntrinsics:
    """Intrinsics for ZED camera."""

    K_left: Float[np.ndarray, "3 3"]  # Left RGB camera matrix
    K_right: Float[np.ndarray, "3 3"]  # Right RGB camera matrix
    distortion_left: Float[np.ndarray, "12"]
    distortion_right: Float[np.ndarray, "12"]
    baseline: float  # Meters


def _custom_params(resolution: str, fps: int, flip: bool = False):
    """Build camera init params from explicit resolution and fps strings."""
    import pyzed.sl as sl

    resolution_map = {
        "HD720": sl.RESOLUTION.HD720,
        "HD1080": sl.RESOLUTION.HD1080,
        "HD2K": sl.RESOLUTION.HD2K,
    }
    if resolution not in resolution_map:
        raise ValueError(f"Unknown resolution '{resolution}', expected one of {list(resolution_map)}")

    return dict(
        depth_minimum_distance=0.1,
        camera_resolution=resolution_map[resolution],
        depth_stabilization=False,
        camera_fps=fps,
        camera_image_flip=sl.FLIP_MODE.ON if flip else sl.FLIP_MODE.OFF,
    )


class ZedCamera:
    def __init__(
        self,
        serial: str,
        depth: bool = False,
        pointcloud: bool = False,
        resolution: str = "HD720",
        fps: int = 60,
        flip: bool = False,
    ):
        import pyzed.sl as sl

        start_time = time.perf_counter()
        self.serial = serial
        self._enable_depth = depth
        self._enable_pcd = pointcloud
        self._is_recording = False

        # Initialize readers
        self._cam = sl.Camera()
        self._left_img = sl.Mat()
        self._right_img = sl.Mat()
        self._depth = sl.Mat()
        self._pcd = sl.Mat()
        self._runtime = sl.RuntimeParameters()

        # Open the camera
        _log.info(f"Opening ZED camera {self.serial}")
        self._params = _custom_params(resolution, fps, flip=flip)
        sl_params = sl.InitParameters(**self._params)
        sl_params.set_from_serial_number(int(self.serial))
        status = self._cam.open(sl_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED camera (s/n: {self.serial}) failed to open")

        # Cache the intrinsics
        _ = self.get_intrinsics()
        init_dur = time.perf_counter() - start_time
        _log.info(f"ZED camera (s/n: {self.serial}) initialization complete, took {init_dur:.2f}s")

    @cache
    def get_intrinsics(self) -> ZedIntrinsics:
        calib_params = self._cam.get_camera_information().camera_configuration.calibration_parameters
        baseline_mm = calib_params.get_camera_baseline()
        baseline = baseline_mm / 1000.0

        left = calib_params.left_cam
        K_left = np.array([[left.fx, 0.0, left.cx], [0.0, left.fy, left.cy], [0.0, 0.0, 1.0]])
        distortion_left = np.array(left.disto)

        right = calib_params.right_cam
        K_right = np.array([[right.fx, 0.0, right.cx], [0.0, right.fy, right.cy], [0.0, 0.0, 1.0]])
        distortion_right = np.array(right.disto)

        return ZedIntrinsics(
            K_left=K_left,
            K_right=K_right,
            distortion_left=distortion_left,
            distortion_right=distortion_right,
            baseline=baseline,
        )

    def read_camera(self) -> ZedFrame:
        import pyzed.sl as sl

        # Grab a frame
        err = self._cam.grab(self._runtime)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to grab frame from ZED camera {self.serial}")

        # Retrieve data
        timestamp = self._cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds()
        self._cam.retrieve_image(self._left_img, sl.VIEW.LEFT)
        left_bgra = self._left_img.get_data()
        self._cam.retrieve_image(self._right_img, sl.VIEW.RIGHT)
        right_bgra = self._right_img.get_data()

        depth = None
        if self._enable_depth:
            self._cam.retrieve_measure(self._depth, sl.MEASURE.DEPTH)
            depth = (self._depth.get_data() / 1000.0).astype(np.float32)  # mm to m

        pointcloud = None
        if self._enable_pcd:
            self._cam.retrieve_measure(self._pcd, sl.MEASURE.XYZ)
            pointcloud = self._pcd.get_data()

        return ZedFrame(
            serial=self.serial,
            timestamp=timestamp,
            rgb=cv2.cvtColor(left_bgra, cv2.COLOR_BGRA2RGB),
            intrinsics=self.get_intrinsics().K_left,
            depth=depth,
            left_bgra=left_bgra,
            right_bgra=right_bgra,
            pointcloud=pointcloud,
        )

    def start_recording(self, filename: str):
        """Start recording camera stream to SVO file using H265 compression."""
        import pyzed.sl as sl

        if not filename.endswith(".svo"):
            raise ValueError("Recording filename must end with .svo")

        recording_param = sl.RecordingParameters(filename, sl.SVO_COMPRESSION_MODE.H265)
        err = self._cam.enable_recording(recording_param)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to enable recording to {filename}")
        self._is_recording = True
        _log.info(f"Started recording to {filename}")

    def stop_recording(self):
        """Stop recording camera stream."""
        if self._is_recording:
            self._cam.disable_recording()
            self._is_recording = False
            _log.info("Stopped recording")

    def close(self):
        self.stop_recording()
        self._cam.close()


def zed_infer_depth(
    frame: ZedFrame,
    intrinsics: ZedIntrinsics,
) -> Float[np.ndarray, "h w"]:
    """Estimate depth from Zed frame and intrinsics using FoundationStereo. Synchronous version."""
    from tiptop.perception.foundation_stereo import infer_depth

    cfg = tiptop_cfg()
    K_left = intrinsics.K_left
    depth = infer_depth(
        cfg.perception.foundation_stereo.url,
        left_rgb=cv2.cvtColor(frame.left_bgra, cv2.COLOR_BGRA2RGB),
        right_rgb=cv2.cvtColor(frame.right_bgra, cv2.COLOR_BGRA2RGB),
        fx=K_left[0, 0],
        fy=K_left[1, 1],
        cx=K_left[0, 2],
        cy=K_left[1, 2],
        baseline=intrinsics.baseline,
    )
    return depth


async def zed_infer_depth_async(
    session: aiohttp.ClientSession,
    frame: ZedFrame,
    intrinsics: ZedIntrinsics,
) -> Float[np.ndarray, "h w"]:
    """Estimate depth from Zed frame and intrinsics using FoundationStereo. Async version."""
    from tiptop.perception.foundation_stereo import infer_depth_async

    cfg = tiptop_cfg()
    K_left = intrinsics.K_left
    depth = await infer_depth_async(
        session,
        cfg.perception.foundation_stereo.url,
        left_rgb=cv2.cvtColor(frame.left_bgra, cv2.COLOR_BGRA2RGB),
        right_rgb=cv2.cvtColor(frame.right_bgra, cv2.COLOR_BGRA2RGB),
        fx=K_left[0, 0],
        fy=K_left[1, 1],
        cx=K_left[0, 2],
        cy=K_left[1, 2],
        baseline=intrinsics.baseline,
    )
    return depth


def convert_svo_to_mp4(svo_path: Path, mp4_path: Path, crf: int = 20):
    """Convert SVO file to MP4 video (left RGB only) using ffmpeg for efficient compression."""
    import shutil
    import subprocess

    import pyzed.sl as sl

    if not svo_path.exists():
        raise FileNotFoundError(f"SVO file not found: {svo_path}")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH, required for SVO to MP4 conversion")

    # Open SVO file
    _log.info(f"Converting {svo_path} to {mp4_path}")
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(str(svo_path))
    init_params.svo_real_time_mode = False
    zed = sl.Camera()
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open SVO file: {err}")

    image_size = zed.get_camera_information().camera_configuration.resolution
    width = image_size.width
    height = image_size.height
    fps = zed.get_camera_information().camera_configuration.fps

    # Use ffmpeg with CRF for much smaller files with good quality
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        str(mp4_path),
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    # Pipe frames from SVO to ffmpeg
    left_image = sl.Mat()
    rt_param = sl.RuntimeParameters()
    nb_frames = zed.get_svo_number_of_frames()
    try:
        with tqdm(total=nb_frames, desc="Converting SVO to MP4", unit="frame") as pbar:
            while True:
                err = zed.grab(rt_param)
                if err == sl.ERROR_CODE.SUCCESS:
                    zed.retrieve_image(left_image, sl.VIEW.LEFT)
                    frame_bgra = left_image.get_data()
                    frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
                    try:
                        proc.stdin.write(frame_bgr.tobytes())
                    except BrokenPipeError:
                        break  # ffmpeg died; returncode check below will raise
                    pbar.update(1)
                elif err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                    break
                else:
                    raise RuntimeError(f"Failed to grab frame at position {zed.get_svo_position()}: {err}")
    finally:
        proc.stdin.close()
        stderr = proc.stderr.read()
        proc.wait()
        zed.close()

    if proc.returncode != 0:
        stderr_msg = stderr.decode(errors="replace").strip()
        raise RuntimeError(f"ffmpeg exited with code {proc.returncode}: {stderr_msg}")
    _log.info(f"Conversion complete: {mp4_path}")
