#!/usr/bin/env python3
"""Delete frame_*.jpg under a directory by index rules (updates disk only; run sync_labels next)."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def frame_index(path: Path) -> int | None:
    m = re.match(r"frame_(\d+)\.jpg$", path.name, re.I)
    return int(m.group(1)) if m else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Prune extracted JPEG frames by numeric index")
    parser.add_argument("--dir", type=Path, required=True, help="Directory containing frame_*.jpg")
    parser.add_argument(
        "--max-index",
        type=int,
        default=None,
        metavar="N",
        help="Delete frames with index > N (keep N and below)",
    )
    parser.add_argument(
        "--drop-multiples-of",
        type=int,
        default=None,
        metavar="K",
        help="Delete frames where index %% K == 0 (e.g. K=3 removes 3,6,9,...)",
    )
    parser.add_argument("-n", "--dry-run", action="store_true", help="Print actions only")
    args = parser.parse_args()

    d = args.dir.resolve()
    if not d.is_dir():
        print(f"error: not a directory: {d}", file=sys.stderr)
        sys.exit(1)
    if args.max_index is None and args.drop_multiples_of is None:
        print("error: pass at least one of --max-index or --drop-multiples-of", file=sys.stderr)
        sys.exit(2)

    removed = 0
    for p in sorted(d.glob("frame_*.jpg")):
        idx = frame_index(p)
        if idx is None:
            continue
        kill = False
        if args.max_index is not None and idx > args.max_index:
            kill = True
        if args.drop_multiples_of is not None and args.drop_multiples_of > 0:
            if idx % args.drop_multiples_of == 0:
                kill = True
        if kill:
            if args.dry_run:
                print(f"would remove {p.name}")
            else:
                p.unlink()
            removed += 1

    print(f"{'would remove' if args.dry_run else 'removed'} {removed} file(s) under {d}")


if __name__ == "__main__":
    main()
