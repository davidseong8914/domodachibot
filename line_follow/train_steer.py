#!/usr/bin/env python3
"""Train SteerNet from labels.jsonl (sin/cos steering targets)."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(it, **_kw):
        return it

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from line_follow.label_dataset import SteeringLabelDataset, load_labels_jsonl  # noqa: E402
from line_follow.splits import split_by_path_substrings, split_random  # noqa: E402
from line_follow.steer_net import SteerNet  # noqa: E402
from line_follow.torch_device import pick_device  # noqa: E402


def angular_mae_deg(pred_theta: np.ndarray, gt_theta: np.ndarray) -> float:
    d = pred_theta - gt_theta
    d = (d + math.pi) % (2 * math.pi) - math.pi
    return float(np.mean(np.abs(d)) * 180.0 / math.pi)


@torch.no_grad()
def eval_loader(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    n = 0
    preds: list[float] = []
    gts: list[float] = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        out = model(x)
        loss_sum += F.mse_loss(out, y, reduction="sum").item()
        n += x.shape[0]
        out_cpu = out.cpu().numpy()
        y_cpu = y.cpu().numpy()
        for j in range(out_cpu.shape[0]):
            s, c = float(out_cpu[j, 0]), float(out_cpu[j, 1])
            preds.append(math.atan2(s, c))
            sg, cg = float(y_cpu[j, 0]), float(y_cpu[j, 1])
            gts.append(math.atan2(sg, cg))
    loss = loss_sum / max(n, 1)
    mae_deg = angular_mae_deg(np.array(preds), np.array(gts))
    return loss, mae_deg


def main() -> None:
    parser = argparse.ArgumentParser(description="Train steering CNN from labels.jsonl")
    parser.add_argument("--labels", type=Path, required=True, help="labels.jsonl path")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root for resolving relative image paths",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Fine-tuning epochs (default 40 suits ImageNet-pretrained backbone + ~1k–2k labels)",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="AdamW lr (default 1e-4 for MobileNetV3-Small fine-tuning; use --lr 1e-3 for a small scratch CNN)",
    )
    parser.add_argument(
        "--split",
        choices=["random", "clip"],
        default="random",
        help="random: by --val-frac/--test-frac; clip: by path substrings",
    )
    parser.add_argument("--val-frac", type=float, default=0.15, help="Used when --split random")
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.15,
        help="Hold-out test fraction for --split random; use 0 for train/val only",
    )
    parser.add_argument(
        "--clip-val",
        action="append",
        default=[],
        metavar="SUBSTR",
        help="Image path substring -> val (repeatable). Used with --split clip",
    )
    parser.add_argument(
        "--clip-test",
        action="append",
        default=[],
        metavar="SUBSTR",
        help="Image path substring -> test (repeatable). Used with --split clip",
    )
    parser.add_argument("--img-h", type=int, default=120)
    parser.add_argument("--img-w", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "line_follow" / "weights" / "steer.pt",
        help="Output checkpoint (.pt)",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Train-only photometric jitter: HSV, RGB gains, partial desat, Gaussian blur, noise (no rotation)",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        metavar="N",
        help="Print metrics every N epochs (use 1 to see every epoch)",
    )
    parser.add_argument(
        "--csv-log",
        type=Path,
        default=None,
        help="Write epoch,train_mse,val_mse,val_mae_deg,lr to this CSV (overwrites each run)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="auto: CUDA if available, else Apple Silicon MPS, else CPU",
    )
    args = parser.parse_args()

    if args.epochs < 1:
        print("error: --epochs must be >= 1", file=sys.stderr)
        sys.exit(2)
    if args.batch_size < 1:
        print("error: --batch-size must be >= 1", file=sys.stderr)
        sys.exit(2)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    labels_path = args.labels.resolve()
    repo_root = args.repo_root.resolve()
    rows = load_labels_jsonl(labels_path)
    full_eval = SteeringLabelDataset(rows, labels_path, repo_root, args.img_h, args.img_w, augment=False)
    n = len(full_eval)
    if n < 8:
        print(f"Need at least 8 samples, got {n}", file=sys.stderr)
        sys.exit(1)

    keys = full_eval.image_keys()
    if args.split == "clip":
        if not args.clip_val or not args.clip_test:
            print("error: --split clip requires at least one --clip-val and one --clip-test", file=sys.stderr)
            sys.exit(2)
        try:
            train_set, val_set, test_set = split_by_path_substrings(keys, args.clip_test, args.clip_val)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            sys.exit(2)
    else:
        train_set, val_set, test_set = split_random(n, args.val_frac, args.test_frac, args.seed)

    train_idx = sorted(train_set)
    val_idx = sorted(val_set)
    test_idx = sorted(test_set)

    if args.augment:
        full_train = SteeringLabelDataset(rows, labels_path, repo_root, args.img_h, args.img_w, augment=True)
        train_ds = Subset(full_train, train_idx)
    else:
        train_ds = Subset(full_eval, train_idx)
    val_ds = Subset(full_eval, val_idx)
    train_loader = DataLoader(
        train_ds, batch_size=min(args.batch_size, len(train_ds)), shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=min(args.batch_size, len(val_ds)), shuffle=False, drop_last=False
    )
    test_loader: DataLoader | None = None
    if test_idx:
        test_ds = Subset(full_eval, test_idx)
        test_loader = DataLoader(
            test_ds, batch_size=min(args.batch_size, len(test_ds)), shuffle=False, drop_last=False
        )

    if args.device == "auto":
        device = pick_device()
    elif args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            print("error: --device cuda but CUDA is not available", file=sys.stderr)
            sys.exit(2)
        device = torch.device("cuda")
    else:
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            print("error: --device mps but MPS is not available", file=sys.stderr)
            sys.exit(2)
        device = torch.device("mps")
    print(f"device: {device}", flush=True)
    model = SteerNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))

    best_val = float("inf")
    best_state = None

    log_every = max(1, args.log_every)
    csv_file = None
    csv_writer: csv.DictWriter | None = None
    if args.csv_log is not None:
        args.csv_log.parent.mkdir(parents=True, exist_ok=True)
        csv_file = args.csv_log.open("w", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(
            csv_file, fieldnames=["epoch", "train_mse", "val_mse", "val_mae_deg", "lr"]
        )
        csv_writer.writeheader()

    try:
        for epoch in tqdm(range(1, args.epochs + 1), desc="SteerNet", unit="ep"):
            model.train()
            train_loss = 0.0
            tn = 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                out = model(x)
                loss = F.mse_loss(out, y)
                loss.backward()
                opt.step()
                train_loss += loss.item() * x.shape[0]
                tn += x.shape[0]
            sched.step()
            vl, vmae = eval_loader(model, val_loader, device)
            tl = train_loss / max(tn, 1)
            lr = float(opt.param_groups[0]["lr"])
            if vl < best_val:
                best_val = vl
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if csv_writer is not None:
                csv_writer.writerow(
                    {
                        "epoch": epoch,
                        "train_mse": f"{tl:.8f}",
                        "val_mse": f"{vl:.8f}",
                        "val_mae_deg": f"{vmae:.6f}",
                        "lr": f"{lr:.8e}",
                    }
                )
                csv_file.flush()
            if epoch % log_every == 0 or epoch == 1 or epoch == args.epochs:
                print(
                    f"epoch {epoch:4d}  train_mse {tl:.6f}  val_mse {vl:.6f}  "
                    f"val_mae_deg {vmae:.2f}  lr {lr:.2e}  best_val_mse {best_val:.6f}"
                )
    finally:
        if csv_file is not None:
            csv_file.close()

    if best_state is not None:
        model.load_state_dict(best_state)

    test_mse = None
    test_mae_deg = None
    if test_loader is not None:
        test_mse, test_mae_deg = eval_loader(model, test_loader, device)
        print(f"hold-out test:  test_mse {test_mse:.6f}  test_mae_deg {test_mae_deg:.2f}")
    else:
        print("no hold-out test set (--test-frac 0 or empty test split)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        labels_rel = str(labels_path.relative_to(repo_root))
    except ValueError:
        labels_rel = str(labels_path)
    meta = {
        "img_h": args.img_h,
        "img_w": args.img_w,
        "labels": labels_rel,
        "split": args.split,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "val_mse": best_val,
        "test_mse": test_mse,
        "test_mae_deg": test_mae_deg,
        "clip_val": args.clip_val if args.split == "clip" else None,
        "clip_test": args.clip_test if args.split == "clip" else None,
        "val_frac": args.val_frac if args.split == "random" else None,
        "test_frac": args.test_frac if args.split == "random" else None,
        "seed": args.seed,
        "augment": args.augment,
        "log_every": args.log_every,
        "csv_log": str(args.csv_log) if args.csv_log is not None else None,
    }
    torch.save(
        {
            "model": model.state_dict(),
            "meta": meta,
        },
        args.out,
    )
    with args.out.with_suffix(".meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
