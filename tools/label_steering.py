#!/usr/bin/env python3
"""Interactive labeling: click target pixel; steering angle from bottom-center origin.

Keys: n next, p previous, s save, q quit. Mouse: left click sets target.

Paths written to labels.jsonl are relative to the repository root when possible.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from line_follow.angles import (  # noqa: E402
    bottom_center_origin,
    draw_ray,
    theta_from_origin_target,
)


def load_labels_map(labels_path: Path) -> Dict[str, Dict[str, Any]]:
    if not labels_path.is_file():
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    with labels_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = row["image"]
            out[key] = row
    return out


def write_labels_jsonl(labels_path: Path, rows_by_image: Dict[str, Dict[str, Any]], image_keys_in_order: List[str]) -> None:
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with labels_path.open("w", encoding="utf-8") as f:
        for key in image_keys_in_order:
            if key in rows_by_image:
                f.write(json.dumps(rows_by_image[key], sort_keys=True) + "\n")


def rel_path_for_label(abs_path: Path, repo_root: Path) -> str:
    try:
        return str(abs_path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(abs_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Click to label steering target; writes labels.jsonl")
    parser.add_argument(
        "--frames-dir",
        type=Path,
        required=True,
        help="Directory containing frame images",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Output labels.jsonl path",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=1280,
        help="Max width for display (image coords mapped back for saving)",
    )
    args = parser.parse_args()

    frames_dir = args.frames_dir.resolve()
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = sorted(
        p for p in frames_dir.iterdir() if p.suffix.lower() in exts and p.is_file()
    )
    if not files:
        print(f"No images found in {frames_dir}", file=sys.stderr)
        sys.exit(1)

    labels_path = args.labels.resolve()
    rows = load_labels_map(labels_path)
    image_keys_in_order = [rel_path_for_label(p, REPO_ROOT) for p in files]

    state: Dict[str, Any] = {
        "index": 0,
        "target": None,  # (x, y) in full image coords, or None
        "drag": False,
    }

    def current_key() -> str:
        return image_keys_in_order[state["index"]]

    def current_abs_path() -> Path:
        return files[state["index"]]

    def load_row_for_current() -> None:
        key = current_key()
        row = rows.get(key)
        if row and "target_px" in row:
            state["target"] = tuple(row["target_px"])
        else:
            state["target"] = None

    load_row_for_current()

    win = "label_steering"

    def on_mouse(event: int, x: int, y: int, flags: int, param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            img = param["img"]
            sx = float(param["scale_x"])
            sy = float(param["scale_y"])
            ix = int(round(x / sx))
            iy = int(round(y / sy))
            ix = max(0, min(img.shape[1] - 1, ix))
            iy = max(0, min(img.shape[0] - 1, iy))
            state["target"] = (ix, iy)

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    print("n: next  p: prev  s: save all  q: quit  click: set target")

    try:
        while True:
            path = current_abs_path()
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                print(f"Could not read {path}", file=sys.stderr)
                state["index"] = min(state["index"] + 1, len(files) - 1)
                load_row_for_current()
                continue

            h, w = img.shape[:2]
            scale = min(1.0, args.max_width / w)
            disp_w = max(1, int(w * scale))
            disp_h = max(1, int(h * scale))
            scale_x = disp_w / w
            scale_y = disp_h / h

            vis = img.copy()
            ox, oy = bottom_center_origin(w, h)
            cv2.circle(vis, (ox, oy), 6, (0, 0, 255), -1, lineType=cv2.LINE_AA)

            if state["target"] is not None:
                tx, ty = state["target"]
                th = theta_from_origin_target(w, h, tx, ty)
                ray_len = min(w, h) * 0.55
                vis = draw_ray(vis, (ox, oy), th, ray_len, color=(0, 255, 0), thickness=3)
                cv2.circle(vis, (int(tx), int(ty)), 6, (255, 128, 0), -1, lineType=cv2.LINE_AA)

            disp = cv2.resize(vis, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
            title = f"{state['index']+1}/{len(files)}  {current_key()}"
            cv2.setWindowTitle(win, title)
            cv2.imshow(win, disp)
            cv2.setMouseCallback(
                win,
                on_mouse,
                param={"img": img, "scale_x": scale_x, "scale_y": scale_y},
            )

            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
            if key == ord("n"):
                state["index"] = min(state["index"] + 1, len(files) - 1)
                load_row_for_current()
            elif key == ord("p"):
                state["index"] = max(state["index"] - 1, 0)
                load_row_for_current()
            elif key == ord("s"):
                key_cur = current_key()
                if state["target"] is None:
                    print("No target set for this frame (click first).", file=sys.stderr)
                else:
                    im_save = cv2.imread(str(current_abs_path()), cv2.IMREAD_COLOR)
                    if im_save is None:
                        print(f"Could not read {current_abs_path()} for save.", file=sys.stderr)
                    else:
                        sh, sw = im_save.shape[:2]
                        ox_c, oy_c = bottom_center_origin(sw, sh)
                        tx, ty = state["target"]
                        rows[key_cur] = {
                            "image": key_cur,
                            "origin": [ox_c, oy_c],
                            "target_px": [int(tx), int(ty)],
                            "theta_rad": theta_from_origin_target(sw, sh, tx, ty),
                        }
                        write_labels_jsonl(labels_path, rows, image_keys_in_order)
                        print(f"Wrote {labels_path} ({sum(1 for k in image_keys_in_order if k in rows)} labeled)")

    except KeyboardInterrupt:
        print("\nExiting (labels already on disk if you pressed s).", file=sys.stderr)
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
