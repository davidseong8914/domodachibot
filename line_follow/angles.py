"""Steering angle from image geometry and circular error metrics.

Convention (image coordinates, y down):
  - Origin O is bottom center: (width/2, height - 1).
  - Target P is a pixel above the robot (desired path direction).
  - Vector v = P - O. Forward in the image is up (-y).
  - theta = atan2(v.x, -v.y): 0 = straight ahead, positive = turn toward +x (right in image).
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import numpy.typing as npt


def bottom_center_origin(width: int, height: int) -> Tuple[int, int]:
    return (width // 2, height - 1)


def theta_from_origin_target(
    width: int, height: int, target_x: float, target_y: float
) -> float:
    """Signed heading in radians from bottom-center origin to target pixel."""
    ox, oy = bottom_center_origin(width, height)
    vx = float(target_x) - ox
    vy = float(target_y) - oy
    return math.atan2(vx, -vy)


def angular_diff_rad(a: float, b: float) -> float:
    """Signed difference a - b wrapped to (-pi, pi]."""
    d = a - b
    return (d + math.pi) % (2.0 * math.pi) - math.pi


def circular_mae_rad(predictions: npt.NDArray[np.floating], ground_truth: npt.NDArray[np.floating]) -> float:
    """Mean absolute angular error; each error is shortest arc magnitude in radians."""
    if predictions.shape != ground_truth.shape:
        raise ValueError("predictions and ground_truth must have the same shape")
    diff = np.asarray(predictions, dtype=np.float64).ravel() - np.asarray(ground_truth, dtype=np.float64).ravel()
    wrapped = (diff + math.pi) % (2.0 * math.pi) - math.pi
    return float(np.mean(np.abs(wrapped)))


def draw_ray(
    image_bgr: npt.NDArray[np.uint8],
    origin: Tuple[int, int],
    theta_rad: float,
    length: float,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> npt.NDArray[np.uint8]:
    """Draw a ray from origin at heading theta_rad (same convention as theta_from_origin_target)."""
    import cv2

    ox, oy = origin
    # Unit direction: forward is -y in image space, right is +x.
    dx = math.sin(theta_rad)
    dy = -math.cos(theta_rad)
    x2 = int(round(ox + dx * length))
    y2 = int(round(oy + dy * length))
    out = image_bgr.copy()
    cv2.line(out, (ox, oy), (x2, y2), color, thickness, lineType=cv2.LINE_AA)
    return out
