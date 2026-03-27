from dataclasses import dataclass

import cv2
import numpy as np
from jaxtyping import Float, UInt8


@dataclass(frozen=True)
class Frame:
    serial: str
    timestamp: float
    rgb: UInt8[np.ndarray, "h w 3"]
    intrinsics: Float[np.ndarray, "3 3"]  # Camera intrinsics matrix
    depth: Float[np.ndarray, "h w"] | None = None  # Onboard sensor depth in metres (optional)

    @property
    def bgr(self) -> UInt8[np.ndarray, "h w 3"]:
        return cv2.cvtColor(self.rgb, cv2.COLOR_RGB2BGR)
