"""Run SteerNet checkpoint for steering angle (same theta convention as line_follow.angles)."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import numpy.typing as npt
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from line_follow.steer_net import SteerNet  # noqa: E402

_cached_path: Optional[Path] = None
_cached_model: Optional[torch.nn.Module] = None
_cached_meta: Optional[dict] = None


def load_steer_net(ckpt_path: Path, device: torch.device | None = None) -> tuple[SteerNet, dict, torch.device]:
    global _cached_path, _cached_model, _cached_meta
    ckpt_path = ckpt_path.resolve()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if _cached_path == ckpt_path and _cached_model is not None and _cached_meta is not None:
        return _cached_model, _cached_meta, device  # type: ignore[return-value]

    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    meta = dict(blob.get("meta", {}))
    img_h = int(meta.get("img_h", 120))
    img_w = int(meta.get("img_w", 160))
    meta["img_h"] = img_h
    meta["img_w"] = img_w
    model = SteerNet().to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    _cached_path = ckpt_path
    _cached_model = model
    _cached_meta = meta
    return model, meta, device


def warmup_steer_net(ckpt_path: Path | str, device: torch.device | None = None) -> None:
    """One forward pass so the first real camera frame is not paying cold-start cost."""
    predict_steering_learned(
        np.zeros((64, 64, 3), dtype=np.uint8), ckpt_path, device=device
    )


def predict_steering_learned(
    image_bgr: npt.NDArray[np.uint8],
    ckpt_path: Path | str,
    device: torch.device | None = None,
) -> float:
    """BGR uint8 full frame -> theta_rad."""
    model, meta, dev = load_steer_net(Path(ckpt_path), device)
    h, w = int(meta["img_h"]), int(meta["img_w"])
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # INTER_LINEAR is faster than INTER_AREA on Pi for downscales; small effect on angle.
    rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    x = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0).div_(255.0)
    x = x.to(dev, non_blocking=False)
    with torch.inference_mode():
        out = model(x)[0]
    s, c = out[0].item(), out[1].item()
    return math.atan2(s, c)


def unload_cache() -> None:
    global _cached_path, _cached_model, _cached_meta
    _cached_path = None
    _cached_model = None
    _cached_meta = None
