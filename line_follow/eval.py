"""Evaluate predict_steering against labels.jsonl."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

# Repo root = parent of line_follow/
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from line_follow.angles import (  # noqa: E402
    angular_diff_rad,
    bottom_center_origin,
    circular_mae_rad,
    draw_ray,
    theta_from_origin_target,
)
from line_follow.learned_predict import predict_steering_learned  # noqa: E402
from line_follow.predict import lookahead_debug_overlay, lookahead_steering  # noqa: E402


def load_labels_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def resolve_image_path(row: Dict[str, Any], labels_file: Path) -> Path:
    raw = row["image"]
    p = Path(raw)
    if p.is_absolute():
        return p
    # Prefer path relative to repo root
    candidate = REPO_ROOT / p
    if candidate.is_file():
        return candidate
    return (labels_file.parent / p).resolve()


def ground_truth_theta(row: Dict[str, Any], width: int, height: int) -> float:
    if "target_px" in row:
        tx, ty = row["target_px"]
        return theta_from_origin_target(width, height, float(tx), float(ty))
    return float(row["theta_rad"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate steering predictor vs labels.jsonl")
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Path to labels.jsonl (paths inside are relative to repo root)",
    )
    parser.add_argument(
        "--viz",
        type=Path,
        default=None,
        help="If set, write comparison images (GT green, pred magenta)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Use debug prediction overlay (ROI, centroid) for viz output",
    )
    parser.add_argument(
        "--no-save-masks",
        action="store_true",
        help="With --viz, skip writing binary masks and otsu_thresholds.jsonl",
    )
    parser.add_argument(
        "--smooth",
        type=float,
        default=None,
        metavar="ALPHA",
        help="Optional temporal smoothing over label order: 0<ALPHA<=1 (higher = trust new frame more). "
        "Uses direction-vector blend between consecutive predictions (helps camera shake on sequences).",
    )
    parser.add_argument(
        "--nn-weights",
        type=Path,
        default=None,
        help="Path to SteerNet checkpoint (.pt) from train_steer.py; uses neural predictor instead of classical",
    )
    args = parser.parse_args()
    if args.smooth is not None and not (0.0 < args.smooth <= 1.0):
        print("error: --smooth ALPHA must be in (0, 1].", file=sys.stderr)
        sys.exit(2)
    if args.nn_weights is not None and args.debug:
        print("warning: --debug is ignored when using --nn-weights", file=sys.stderr)
    labels_path = args.labels.resolve()
    rows = load_labels_jsonl(labels_path)
    if not rows:
        print("No label rows found.", file=sys.stderr)
        sys.exit(1)

    preds: List[float] = []
    gts: List[float] = []
    errors: List[tuple[str, float]] = []

    if args.viz is not None:
        args.viz.mkdir(parents=True, exist_ok=True)
        masks_dir = args.viz / "masks"
        if not args.no_save_masks:
            masks_dir.mkdir(parents=True, exist_ok=True)
            thresholds_path = masks_dir / "otsu_thresholds.jsonl"
            thresholds_path.write_text("", encoding="utf-8")

    prev_vx: float | None = None
    prev_vy: float | None = None
    use_nn = args.nn_weights is not None
    nn_ckpt = args.nn_weights.resolve() if use_nn else None
    if use_nn and not nn_ckpt.is_file():
        print(f"error: --nn-weights not found: {nn_ckpt}", file=sys.stderr)
        sys.exit(2)

    for row in rows:
        img_path = resolve_image_path(row, labels_path)
        if not img_path.is_file():
            print(f"warning: missing image {img_path}", file=sys.stderr)
            continue
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"warning: could not read {img_path}", file=sys.stderr)
            continue
        h, w = bgr.shape[:2]
        gt = ground_truth_theta(row, w, h)
        if use_nn:
            pred_raw = predict_steering_learned(bgr, nn_ckpt)
            res = None
        else:
            res = lookahead_steering(bgr)
            pred_raw = res.theta_rad
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

        if args.debug and res is not None:
            vis = lookahead_debug_overlay(bgr, res)
        else:
            vis = bgr.copy()

        preds.append(pred)
        gts.append(gt)
        err = angular_diff_rad(pred, gt)
        rel = row.get("image", str(img_path))
        errors.append((rel, abs(err)))

        if args.viz is not None:
            ox, oy = bottom_center_origin(w, h)
            ray_len = min(w, h) * 0.5
            vis = draw_ray(vis, (ox, oy), gt, ray_len, color=(0, 255, 0), thickness=3)
            # Classical: magenta; +smooth: orange; neural: cyan (BGR).
            if use_nn:
                pred_color = (255, 255, 0)
            elif args.smooth is not None:
                pred_color = (0, 165, 255)
            else:
                pred_color = (255, 0, 255)
            vis = draw_ray(vis, (ox, oy), pred, ray_len, color=pred_color, thickness=2)
            stem = Path(rel).stem
            out_path = args.viz / f"compare_{stem}.jpg"
            cv2.imwrite(str(out_path), vis)
            if not args.no_save_masks and res is not None and res.mask_u8.size > 0:
                mask_path = masks_dir / f"bin_{stem}.png"
                cv2.imwrite(str(mask_path), res.mask_u8)
                rec = {
                    "image": rel,
                    "otsu_on_blurred_inv_v": res.otsu_threshold,
                    "roi_h": res.roi_h,
                    "mask_path": f"masks/bin_{stem}.png",
                }
                with (masks_dir / "otsu_thresholds.jsonl").open("a", encoding="utf-8") as tf:
                    tf.write(json.dumps(rec, sort_keys=True) + "\n")

    if not preds:
        print("No valid labeled images evaluated.", file=sys.stderr)
        sys.exit(1)

    pred_arr = np.array(preds, dtype=np.float64)
    gt_arr = np.array(gts, dtype=np.float64)
    mae_rad = circular_mae_rad(pred_arr, gt_arr)
    mae_deg = math.degrees(mae_rad)
    signed = np.array([angular_diff_rad(float(p), float(g)) for p, g in zip(preds, gts)])
    bias_rad = float(np.mean(signed))
    bias_deg = math.degrees(bias_rad)

    errors.sort(key=lambda x: -x[1])
    worst = errors[: min(5, len(errors))]

    print(f"N = {len(preds)}")
    if use_nn:
        print(f"predictor: SteerNet ({nn_ckpt})")
    if args.smooth is not None:
        print(f"temporal smooth ALPHA = {args.smooth} (blend in label-file order)")
    print(f"circular MAE = {mae_deg:.2f} deg ({mae_rad:.4f} rad)")
    print(
        f"mean signed error = {bias_deg:.2f} deg ({bias_rad:.4f} rad)  "
        f"(pred minus GT; positive means predictor angle is CCW of GT on the image plane)"
    )
    print("worst |error|:")
    for name, e in worst:
        print(f"  {math.degrees(e):.2f} deg  {name}")

    if args.viz is not None:
        print(f"wrote viz to {args.viz}")
        if not args.no_save_masks:
            print(f"wrote binary masks and masks/otsu_thresholds.jsonl under {args.viz}")


if __name__ == "__main__":
    main()
