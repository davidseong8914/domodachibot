#!/usr/bin/env python3
"""Run SteerNet on each frame of a video; draw steering ray + theta text; write MP4."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from line_follow.angles import bottom_center_origin, draw_ray  # noqa: E402
from line_follow.learned_predict import predict_steering_learned, unload_cache  # noqa: E402

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(x, **_):
        return x


def main() -> None:
    p = argparse.ArgumentParser(description="SteerNet video overlay (cyan ray from bottom-center)")
    p.add_argument("video", type=Path, help="Input video path")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output MP4 (default: <input_stem>_nn_overlay.mp4 next to input)",
    )
    p.add_argument(
        "--weights",
        type=Path,
        default=REPO_ROOT / "line_follow" / "weights" / "steer.pt",
        help="SteerNet checkpoint from train_steer",
    )
    p.add_argument(
        "--smooth",
        type=float,
        default=None,
        metavar="ALPHA",
        help="Optional temporal smooth in (0,1]: blend direction vectors vs previous frame",
    )
    args = p.parse_args()
    if args.smooth is not None and not (0.0 < args.smooth <= 1.0):
        print("error: --smooth must be in (0, 1].", file=sys.stderr)
        sys.exit(2)

    vin = args.video.resolve()
    if not vin.is_file():
        print(f"error: not a file: {vin}", file=sys.stderr)
        sys.exit(1)
    wpath = args.weights.resolve()
    if not wpath.is_file():
        print(f"error: weights not found: {wpath}", file=sys.stderr)
        sys.exit(1)

    out = args.output
    if out is None:
        out = vin.parent / f"{vin.stem}_nn_overlay.mp4"
    else:
        out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    unload_cache()
    cap = cv2.VideoCapture(str(vin))
    if not cap.isOpened():
        print(f"error: could not open video: {vin}", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-3:
        fps = 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if nframes <= 0:
        nframes = None

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, fps, (w, h))
    if not writer.isOpened():
        print("error: could not open VideoWriter (try different --fourcc)", file=sys.stderr)
        cap.release()
        sys.exit(1)

    prev_vx: float | None = None
    prev_vy: float | None = None
    fi = 0
    pbar = tqdm(total=nframes if nframes and nframes > 0 else None, unit="fr", desc="nn_overlay")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        pred_raw = predict_steering_learned(frame, wpath)
        if args.smooth is not None and prev_vx is not None and prev_vy is not None:
            a = float(args.smooth)
            nx = math.sin(pred_raw)
            ny = -math.cos(pred_raw)
            vx = (1.0 - a) * prev_vx + a * nx
            vy = (1.0 - a) * prev_vy + a * ny
            pred = math.atan2(vx, -vy)
        else:
            pred = pred_raw
        prev_vx = math.sin(pred)
        prev_vy = -math.cos(pred)

        ox, oy = bottom_center_origin(w, h)
        ray_len = min(w, h) * 0.5
        vis = draw_ray(frame, (ox, oy), pred, ray_len, color=(255, 255, 0), thickness=3)

        deg = math.degrees(pred)
        label = f"theta {deg:+.1f} deg"
        cv2.putText(
            vis,
            label,
            (12, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            label,
            (10, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (40, 40, 40),
            1,
            cv2.LINE_AA,
        )

        writer.write(vis)
        fi += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    writer.release()
    print(f"wrote {out}  ({fi} frames)")


if __name__ == "__main__":
    main()
