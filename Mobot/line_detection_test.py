"""
CV Parameter Tuning Script
===========================

Browse images and tune CV parameters live with keyboard shortcuts.
Use this to find good BLOCK_SIZE, THRESH_C, and CLOSE_KERNEL_SIZE values
before copying them into mobot_line_follower.py.

Usage:
    python tune_params.py --image_dir ./new_field_img
    python tune_params.py --image ./path/to/single.jpg

Controls:
    d / right arrow : next image
    a / left arrow  : previous image
    b / B           : BLOCK_SIZE    + / - 10
    c / C           : THRESH_C      + / - 1
    k / K           : CLOSE_KERNEL  + / - 2
    r               : reset to defaults
    q               : quit
"""

from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass

import cv2
import numpy as np


# =============================================================================
# Tunable parameters (adjust with keyboard live)
# =============================================================================

ROI_BOTTOM_FRACTION = 0.45
SIDE_MARGIN_FRACTION = 0.35

BLUR_KERNEL = 9
BLOCK_SIZE = 201
THRESH_C = -4
MORPH_KERNEL = 5
CLOSE_KERNEL_SIZE = 15

BAND_HEIGHT = 20
BAND_FRACTIONS = (0.70, 0.45)

MIN_WHITE_PIXELS_PER_COL = 3
MIN_SEGMENT_WIDTH = 12
CENTER_BIAS_WEIGHT = 2.0
MAX_SEGMENT_GAP = 10


# =============================================================================
# CV pipeline (same as mobot_line_follower.py)
# =============================================================================

@dataclass
class BandDetection:
    y_start: int
    x_center: int | None
    segment: tuple[int, int] | None
    occupancy: np.ndarray


def preprocess(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (BLUR_KERNEL, BLUR_KERNEL), 0)


def segment_line(blurred_gray):
    thresh = cv2.adaptiveThreshold(
        blurred_gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        BLOCK_SIZE, THRESH_C,
    )
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (MORPH_KERNEL, MORPH_KERNEL)
    )
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (CLOSE_KERNEL_SIZE, CLOSE_KERNEL_SIZE)
    )
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, open_kernel)
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_kernel)
    return cleaned


def contiguous_segments(binary_1d):
    segments = []
    start = None
    for i, value in enumerate(binary_1d):
        if value and start is None:
            start = i
        elif not value and start is not None:
            segments.append((start, i - 1))
            start = None
    if start is not None:
        segments.append((start, len(binary_1d) - 1))
    return segments


def merge_close_segments(segments, max_gap=MAX_SEGMENT_GAP):
    if not segments:
        return []
    merged = [segments[0]]
    for start, end in segments[1:]:
        last_start, last_end = merged[-1]
        if start - last_end - 1 <= max_gap:
            merged[-1] = (last_start, end)
        else:
            merged.append((start, end))
    return merged


def choose_segment(segments, reference_x):
    valid = [s for s in segments if (s[1] - s[0] + 1) >= MIN_SEGMENT_WIDTH]
    if not valid:
        return None
    best, best_score = None, -np.inf
    for start, end in valid:
        width = end - start + 1
        center = 0.5 * (start + end)
        score = width - CENTER_BIAS_WEIGHT * abs(center - reference_x)
        if score > best_score:
            best_score = score
            best = (start, end)
    return best


def find_line_in_band(mask, y_start, band_height, reference_x):
    strip = mask[y_start:y_start + band_height, :]
    col_counts = np.sum(strip > 0, axis=0)
    occupancy = col_counts >= MIN_WHITE_PIXELS_PER_COL

    segments = contiguous_segments(occupancy)
    segments = merge_close_segments(segments, MAX_SEGMENT_GAP)
    chosen = choose_segment(segments, reference_x)

    if chosen is None:
        return BandDetection(y_start=y_start, x_center=None, segment=None, occupancy=col_counts)

    start, end = chosen
    xs = np.arange(start, end + 1)
    weights = col_counts[start:end + 1].astype(np.float32)
    if np.sum(weights) > 0:
        x_center = int(np.round(np.sum(xs * weights) / np.sum(weights)))
    else:
        x_center = (start + end) // 2

    return BandDetection(y_start=y_start, x_center=x_center, segment=chosen, occupancy=col_counts)


def process_frame(frame_bgr):
    h, w = frame_bgr.shape[:2]
    roi_y0 = int(h * (1.0 - ROI_BOTTOM_FRACTION))
    side_margin = int(w * SIDE_MARGIN_FRACTION)
    roi_x0, roi_x1 = side_margin, w - side_margin

    roi = frame_bgr[roi_y0:, roi_x0:roi_x1]
    roi_h, roi_w = roi.shape[:2]

    blurred = preprocess(roi)
    mask = segment_line(blurred)

    band_y_starts = [min(max(int(roi_h * frac), 0), roi_h - BAND_HEIGHT) for frac in BAND_FRACTIONS]
    image_center_x = roi_w // 2
    detections = []

    reference_x = image_center_x
    for i, y_start in enumerate(band_y_starts):
        det = find_line_in_band(mask, y_start, BAND_HEIGHT, reference_x)
        detections.append(det)
        if i == 0 and det.x_center is not None:
            reference_x = det.x_center

    near = detections[0]
    error = None if near.x_center is None else near.x_center - image_center_x

    return {
        "roi": roi, "mask": mask, "detections": detections,
        "image_center_x": image_center_x, "error": error,
    }


def visualize(results):
    roi = results["roi"].copy()
    mask_color = cv2.cvtColor(results["mask"], cv2.COLOR_GRAY2BGR)
    h, w = roi.shape[:2]
    cx = results["image_center_x"]

    cv2.line(roi, (cx, 0), (cx, h), (0, 255, 0), 2)
    cv2.line(mask_color, (cx, 0), (cx, h), (0, 255, 0), 2)

    band_colors = [(255, 140, 0), (0, 140, 255)]
    band_names = ["NEAR", "FAR"]

    for i, det in enumerate(results["detections"]):
        color = band_colors[i % len(band_colors)]
        label = band_names[i] if i < len(band_names) else f"B{i}"
        y0 = det.y_start

        cv2.rectangle(roi, (0, y0), (w, y0 + BAND_HEIGHT), color, 2)
        cv2.rectangle(mask_color, (0, y0), (w, y0 + BAND_HEIGHT), color, 2)

        if det.segment is not None:
            sx, ex = det.segment
            cy = y0 + BAND_HEIGHT // 2
            cv2.line(roi, (sx, cy), (ex, cy), (255, 0, 255), 2)

        if det.x_center is not None:
            pt = (det.x_center, y0 + BAND_HEIGHT // 2)
            cv2.circle(roi, pt, 7, (0, 0, 255), -1)
            cv2.circle(mask_color, pt, 7, (0, 0, 255), -1)

    err = results["error"]
    info = f"Err:{err:+d}px" if err is not None else "LINE LOST"
    cv2.putText(roi, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return np.hstack([roi, mask_color])


# =============================================================================
# Main loop
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Tune CV parameters on track photos")
    parser.add_argument("--image", type=str, help="Path to one image")
    parser.add_argument("--image_dir", type=str, help="Directory of images")
    args = parser.parse_args()

    if args.image:
        image_paths = [args.image]
    elif args.image_dir:
        image_paths = sorted(
            glob.glob(os.path.join(args.image_dir, "*.jpg"))
            + glob.glob(os.path.join(args.image_dir, "*.png"))
            + glob.glob(os.path.join(args.image_dir, "*.jpeg"))
        )
    else:
        print("Use --image or --image_dir")
        return

    if not image_paths:
        print("No images found")
        return

    print(f"Found {len(image_paths)} image(s)")
    print("Controls:")
    print("  d / right arrow : next image")
    print("  a / left arrow  : previous image")
    print("  b / B           : BLOCK_SIZE    + / - 10")
    print("  c / C           : THRESH_C      + / - 1")
    print("  k / K           : CLOSE_KERNEL  + / - 2")
    print("  r               : reset to defaults")
    print("  q               : quit")
    print()

    global BLOCK_SIZE, THRESH_C, CLOSE_KERNEL_SIZE
    defaults = (BLOCK_SIZE, THRESH_C, CLOSE_KERNEL_SIZE)

    idx = 0
    while True:
        img = cv2.imread(image_paths[idx])
        if img is None:
            idx = (idx + 1) % len(image_paths)
            continue

        img_small = cv2.resize(img, (640, 480))
        results = process_frame(img_small)
        vis = visualize(results)

        footer = (
            f"[{idx+1}/{len(image_paths)}] "
            f"BLOCK={BLOCK_SIZE}  C={THRESH_C}  CLOSE={CLOSE_KERNEL_SIZE}  "
            f"| {os.path.basename(image_paths[idx])}"
        )
        cv2.putText(vis, footer, (10, vis.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Param Tuning", vis)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            break
        elif key in (83, ord('d')):
            idx = (idx + 1) % len(image_paths)
        elif key in (81, ord('a')):
            idx = (idx - 1) % len(image_paths)

        # BLOCK_SIZE
        elif key == ord('b'):
            BLOCK_SIZE = min(BLOCK_SIZE + 10, 401)
            if BLOCK_SIZE % 2 == 0:
                BLOCK_SIZE += 1
            print(f"BLOCK_SIZE = {BLOCK_SIZE}")
        elif key == ord('B'):
            BLOCK_SIZE = max(BLOCK_SIZE - 10, 11)
            if BLOCK_SIZE % 2 == 0:
                BLOCK_SIZE += 1
            print(f"BLOCK_SIZE = {BLOCK_SIZE}")

        # THRESH_C
        elif key == ord('c'):
            THRESH_C = min(THRESH_C + 1, 10)
            print(f"THRESH_C = {THRESH_C}")
        elif key == ord('C'):
            THRESH_C = max(THRESH_C - 1, -30)
            print(f"THRESH_C = {THRESH_C}")

        # CLOSE_KERNEL_SIZE
        elif key == ord('k'):
            CLOSE_KERNEL_SIZE = min(CLOSE_KERNEL_SIZE + 2, 101)
            print(f"CLOSE_KERNEL_SIZE = {CLOSE_KERNEL_SIZE}")
        elif key == ord('K'):
            CLOSE_KERNEL_SIZE = max(CLOSE_KERNEL_SIZE - 2, 3)
            print(f"CLOSE_KERNEL_SIZE = {CLOSE_KERNEL_SIZE}")

        # Reset
        elif key == ord('r'):
            BLOCK_SIZE, THRESH_C, CLOSE_KERNEL_SIZE = defaults
            print("Reset to defaults")

    cv2.destroyAllWindows()
    print("\nFinal parameters (copy to mobot_line_follower.py):")
    print(f"  BLOCK_SIZE = {BLOCK_SIZE}")
    print(f"  THRESH_C = {THRESH_C}")
    print(f"  CLOSE_KERNEL_SIZE = {CLOSE_KERNEL_SIZE}")


if __name__ == "__main__":
    main()