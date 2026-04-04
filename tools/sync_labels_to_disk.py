#!/usr/bin/env python3
"""Drop label rows whose image path does not resolve to an existing file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter labels.jsonl to existing images only")
    parser.add_argument("--labels", type=Path, required=True, help="labels.jsonl path")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    labels_path = args.labels.resolve()
    repo_root = args.repo_root.resolve()
    if not labels_path.is_file():
        print(f"error: missing {labels_path}", file=sys.stderr)
        sys.exit(1)

    kept: list[dict] = []
    dropped = 0
    with labels_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            raw = row["image"]
            p = Path(raw)
            if p.is_absolute():
                cand = p
            else:
                cand = repo_root / p
                if not cand.is_file():
                    cand = labels_path.parent / p
            if cand.is_file():
                kept.append(row)
            else:
                dropped += 1

    with labels_path.open("w", encoding="utf-8") as out:
        for row in kept:
            out.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"{labels_path}: kept {len(kept)}, dropped {dropped}")


if __name__ == "__main__":
    main()
