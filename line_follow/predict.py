"""Steering prediction from a single BGR frame (experiment sandbox)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt

from line_follow.angles import angular_diff_rad, bottom_center_origin, theta_from_origin_target

# Fraction of image height from the top used as look-ahead (road + line ahead of the robot).
LOOKAHEAD_FRACTION = 0.55


@dataclass(frozen=True)
class LookaheadSteering:
    """Outputs from the look-ahead Otsu pipeline (for eval / debugging)."""

    theta_rad: float
    """Binary mask for debugging: component and/or column-stripe votes (ROI-sized)."""
    mask_u8: npt.NDArray[np.uint8]
    otsu_threshold: float
    roi_h: int


def _lookahead_roi_height(image_h: int) -> int:
    return max(1, int(math.ceil(image_h * LOOKAHEAD_FRACTION)))


def _largest_true_segment(mask_1d: npt.NDArray[np.bool_]) -> Optional[slice]:
    best: Optional[Tuple[int, int]] = None
    best_len = 0
    n = int(mask_1d.shape[0])
    i = 0
    while i < n:
        if not bool(mask_1d[i]):
            i += 1
            continue
        j = i
        while j < n and bool(mask_1d[j]):
            j += 1
        if j - i > best_len:
            best_len = j - i
            best = (i, j)
        i = j
    if best is None:
        return None
    return slice(best[0], best[1])


def _column_stripe_theta_and_mask(
    v_raw: npt.NDArray[np.uint8],
    full_h: int,
    full_w: int,
) -> Tuple[Optional[float], npt.NDArray[np.uint8]]:
    """
    Per-column peak brightness after wide horizontal blur: follows a thin vertical stripe
    even when the Otsu blob breaks under shadow (fragmented line).
    """
    roi_h, w = v_raw.shape
    dbg = np.zeros((roi_h, w), dtype=np.uint8)
    vb = cv2.GaussianBlur(v_raw, (9, 27), 0)
    col_peak = np.max(vb, axis=0).astype(np.float64)
    row_max = np.argmax(vb, axis=0)

    t = float(np.percentile(col_peak, 66))
    active = col_peak >= t
    min_run = max(8, w // 22)
    seg = _largest_true_segment(active)
    if seg is None or (seg.stop - seg.start) < min_run:
        return None, dbg

    cols = np.arange(seg.start, seg.stop, dtype=np.float64)
    peaks = col_peak[seg]
    rows = row_max[seg].astype(np.float64)
    wts = peaks * peaks
    sw = float(np.sum(wts))
    if sw < 1e-6:
        return None, dbg
    cx = float(np.sum(cols * wts) / sw)
    cy = float(np.sum(rows * wts) / sw)

    kpt = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    for j in range(seg.start, seg.stop):
        r = int(row_max[j])
        dbg[r, j] = 255
    dbg = cv2.dilate(dbg, kpt, iterations=1)

    return theta_from_origin_target(full_w, full_h, cx, cy), dbg


def _component_steering(
    v_raw: npt.NDArray[np.uint8],
    roi_h: int,
    w: int,
    h: int,
) -> Tuple[float, npt.NDArray[np.uint8], float]:
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    v_eq = clahe.apply(v_raw)

    inv = 255 - v_eq
    blur = cv2.GaussianBlur(inv, (5, 5), 0)
    otsu_val, mask_line = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_line = cv2.morphologyEx(mask_line, cv2.MORPH_OPEN, k, iterations=1)

    roi_pixels = float(roi_h * w)
    min_area = max(60, int(0.00015 * roi_pixels))
    v_floor = float(np.clip(np.percentile(v_raw, 45) + 18, 92, 185))
    cx_image = 0.5 * float(w)

    num, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask_line, connectivity=8)
    best_label = -1
    best_score = -1.0
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        left = int(stats[i, cv2.CC_STAT_LEFT])
        comp = labels == i
        mean_v = float(np.mean(v_raw[comp]))
        if mean_v < v_floor:
            continue
        aspect = bh / max(float(bw), 1.0)
        elong = min(aspect, 10.0)
        cx_comp = left + 0.5 * float(bw)
        dist_n = abs(cx_comp - cx_image) / max(0.5 * w, 1.0)
        center_prior = 1.0 + 0.55 * math.exp(-5.0 * dist_n * dist_n)
        score = mean_v * math.log1p(area) * (1.0 + 0.2 * elong) * center_prior
        if score > best_score:
            best_score = score
            best_label = i

    if best_label >= 0:
        final = ((labels == best_label).astype(np.uint8)) * 255
        m = cv2.moments(final)
        if m["m00"] < 1e-3:
            return 0.0, final, float(otsu_val)
        cx = float(m["m10"] / m["m00"])
        cy = float(m["m01"] / m["m00"])
        return theta_from_origin_target(w, h, cx, cy), final, float(otsu_val)

    m = cv2.moments(mask_line)
    if m["m00"] < 1e-3:
        return 0.0, mask_line, float(otsu_val)
    cx = float(m["m10"] / m["m00"])
    cy = float(m["m01"] / m["m00"])
    return theta_from_origin_target(w, h, cx, cy), mask_line, float(otsu_val)


def _fuse_component_and_column(
    theta_comp: float,
    theta_col: Optional[float],
    shadow_frac: float,
) -> float:
    if theta_col is None:
        return theta_comp
    diff = abs(angular_diff_rad(theta_col, theta_comp))
    # Heavy shadow / uneven lighting: column stripe tracks local maxima, less slab-like than Otsu blobs.
    if shadow_frac >= 0.17:
        w_col = float(np.clip(0.32 + 1.15 * shadow_frac, 0.32, 0.82))
        sx = (1.0 - w_col) * math.sin(theta_comp) + w_col * math.sin(theta_col)
        sy = (1.0 - w_col) * (-math.cos(theta_comp)) + w_col * (-math.cos(theta_col))
        return math.atan2(sx, -sy)
    if diff <= math.radians(24):
        sx = 0.5 * (math.sin(theta_comp) + math.sin(theta_col))
        sy = 0.5 * (-math.cos(theta_comp) - math.cos(theta_col))
        return math.atan2(sx, -sy)
    return theta_comp


def lookahead_steering(image_bgr: npt.NDArray[np.uint8]) -> LookaheadSteering:
    """
    Hybrid look-ahead:

    1) Otsu on inverted CLAHE-V (INV) + connected components filtered by *raw* V (paint vs shadow).
    2) Column-wise peak on blurred raw V inside the largest bright run (thin vertical line under
       broken segmentation / shadow overlap).
    3) Shadow-aware fusion: more column weight when a large ROI fraction is very dark.
    """
    h, w = image_bgr.shape[:2]
    roi_h = _lookahead_roi_height(h)
    roi = image_bgr[0:roi_h, :]
    if roi.size == 0:
        return LookaheadSteering(0.0, np.zeros((0, w), dtype=np.uint8), 0.0, roi_h)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    v_raw = hsv[:, :, 2]

    shadow_frac = float(np.mean(v_raw < np.percentile(v_raw, 38)))

    theta_comp, mask_comp, otsu_val = _component_steering(v_raw, roi_h, w, h)
    theta_col, col_mask = _column_stripe_theta_and_mask(v_raw, h, w)

    theta = _fuse_component_and_column(theta_comp, theta_col, shadow_frac)
    mask_out = np.maximum(mask_comp, col_mask)

    return LookaheadSteering(theta, mask_out, otsu_val, roi_h)


def predict_steering(image_bgr: npt.NDArray[np.uint8]) -> float:
    """Same convention as line_follow.angles.theta_from_origin_target."""
    return lookahead_steering(image_bgr).theta_rad


def lookahead_debug_overlay(image_bgr: npt.NDArray[np.uint8], res: LookaheadSteering) -> npt.NDArray[np.uint8]:
    h, w = image_bgr.shape[:2]
    vis = image_bgr.copy()
    roi_h = res.roi_h
    if res.mask_u8.size == 0:
        return vis

    m = cv2.moments(res.mask_u8)
    if m["m00"] >= 1e-3:
        cx_roi = m["m10"] / m["m00"]
        cy_roi = m["m01"] / m["m00"]
        icx = int(round(cx_roi))
        icy = int(round(cy_roi))
        cv2.circle(vis, (icx, icy), 8, (0, 255, 255), -1, lineType=cv2.LINE_AA)

    cv2.rectangle(vis, (0, 0), (w - 1, roi_h - 1), (255, 0, 0), 2, lineType=cv2.LINE_AA)
    ox, oy = bottom_center_origin(w, h)
    length = min(w, h) * 0.45
    th = res.theta_rad
    dx = math.sin(th)
    dy = -math.cos(th)
    x2 = int(round(ox + dx * length))
    y2 = int(round(oy + dy * length))
    cv2.line(vis, (ox, oy), (x2, y2), (0, 255, 0), 2, lineType=cv2.LINE_AA)
    return vis


def predict_steering_with_debug(
    image_bgr: npt.NDArray[np.uint8],
) -> Tuple[float, npt.NDArray[np.uint8]]:
    """Returns (theta, BGR visualization: look-ahead box, centroid, predicted ray)."""
    res = lookahead_steering(image_bgr)
    return res.theta_rad, lookahead_debug_overlay(image_bgr, res)
