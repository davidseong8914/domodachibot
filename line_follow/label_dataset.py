"""Load steering labels from JSONL for PyTorch training."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from line_follow.angles import theta_from_origin_target
from line_follow.imagenet_norm import normalize_rgb_01chw


def load_labels_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def resolve_image_path(row: Dict[str, Any], labels_file: Path, repo_root: Path) -> Path:
    raw = row["image"]
    p = Path(raw)
    if p.is_absolute():
        return p
    cand = repo_root / p
    if cand.is_file():
        return cand
    return (labels_file.parent / p).resolve()


def theta_from_row(row: Dict[str, Any], width: int, height: int) -> float:
    if "target_px" in row:
        tx, ty = row["target_px"]
        return theta_from_origin_target(width, height, float(tx), float(ty))
    return float(row["theta_rad"])


def _augment_rgb_uint8(rgb: np.ndarray) -> np.ndarray:
    """Photometric augment only: HSV/RGB jitter, desat, blur, noise (no geometric warps)."""
    out = rgb
    if random.random() < 0.85:
        hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV).astype(np.float32)
        dh = random.uniform(-10.0, 10.0)
        hsv[:, :, 0] = (hsv[:, :, 0] + dh) % 180.0
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.82, 1.18), 0.0, 255.0)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(0.72, 1.28), 0.0, 255.0)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    if random.random() < 0.45:
        gains = np.array(
            [random.uniform(0.90, 1.10), random.uniform(0.90, 1.10), random.uniform(0.90, 1.10)],
            dtype=np.float32,
        ).reshape(1, 1, 3)
        out = np.clip(out.astype(np.float32) * gains, 0.0, 255.0).astype(np.uint8)
    if random.random() < 0.18:
        gray = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
        gray3 = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
        a = random.uniform(0.0, 0.22)
        out = np.clip(out.astype(np.float32) * (1.0 - a) + gray3 * a, 0.0, 255.0).astype(np.uint8)
    if random.random() < 0.38:
        k = random.choice([3, 5])
        out = cv2.GaussianBlur(out, (k, k), 0)
    if random.random() < 0.42:
        sigma = random.uniform(2.0, 9.0)
        noise = np.random.normal(0.0, sigma, out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)
    return out


class SteeringLabelDataset(Dataset):
    """BGR images resized to (H, W); inputs ImageNet-normalized RGB; targets (sin(theta), cos(theta))."""

    def __init__(
        self,
        rows: List[Dict[str, Any]],
        labels_path: Path,
        repo_root: Path,
        img_h: int,
        img_w: int,
        augment: bool = False,
    ) -> None:
        self._items: List[Tuple[Path, float, str]] = []
        self.img_h = img_h
        self.img_w = img_w
        self.augment = augment
        for row in rows:
            p = resolve_image_path(row, labels_path, repo_root)
            if not p.is_file():
                continue
            im = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if im is None:
                continue
            h, w = im.shape[:2]
            th = theta_from_row(row, w, h)
            key = str(row["image"])
            self._items.append((p, th, key))
        if not self._items:
            raise ValueError("No valid labeled images found.")

    def image_keys(self) -> List[str]:
        return [t[2] for t in self._items]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path, th, _key = self._items[idx]
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            bgr = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA)
        if self.augment:
            rgb = _augment_rgb_uint8(rgb)
        x = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        x = normalize_rgb_01chw(x)
        s, c = math.sin(th), math.cos(th)
        y = torch.tensor([s, c], dtype=torch.float32)
        return x, y
