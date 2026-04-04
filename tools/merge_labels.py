#!/usr/bin/env python3
"""Merge multiple labels.jsonl files into one (by image key; later files override)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge JSONL label files into one")
    parser.add_argument("inputs", nargs="+", type=Path, help="labels.jsonl files in merge order")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output labels.jsonl")
    args = parser.parse_args()

    by_image: Dict[str, Dict[str, Any]] = {}
    for path in args.inputs:
        if not path.is_file():
            print(f"warning: skip missing {path}", file=sys.stderr)
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                by_image[row["image"]] = row

    keys: List[str] = sorted(by_image.keys())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for k in keys:
            out.write(json.dumps(by_image[k], sort_keys=True) + "\n")
    print(f"Wrote {len(keys)} rows to {args.output}")


if __name__ == "__main__":
    main()
