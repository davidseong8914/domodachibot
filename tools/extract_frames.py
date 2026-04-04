#!/usr/bin/env python3
"""Extract still frames from a video at a fixed frame rate using FFmpeg.

Default fps=0.2 selects one frame every 5 seconds (1/5 Hz).

Angle / labeling convention for this repo lives in line_follow.angles (used by other tools).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract frames from video via FFmpeg (e.g. 1 frame every 5 s with --fps 0.2)."
    )
    parser.add_argument("video", type=Path, help="Input video file")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for extracted images (created if missing)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=0.2,
        help="Output sampling rate in Hz (default: 0.2 = one frame every 5 seconds)",
    )
    parser.add_argument(
        "--pattern",
        default="frame_%04d.jpg",
        help="Output filename pattern (default: frame_%%04d.jpg)",
    )
    parser.add_argument(
        "-q:v",
        dest="jpeg_q",
        type=int,
        default=2,
        help="JPEG quality 1-31, lower is better (ffmpeg -q:v)",
    )
    args = parser.parse_args()
    video = args.video.resolve()
    if not video.is_file():
        print(f"error: video not found: {video}", file=sys.stderr)
        sys.exit(1)
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pattern = out_dir / args.pattern

    vf = f"fps={args.fps}"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video),
        "-vf",
        vf,
        "-q:v",
        str(args.jpeg_q),
        str(out_pattern),
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    paths = sorted(out_dir.glob("frame_*.jpg"))
    if not paths:
        paths = sorted(out_dir.glob("*.jpg"))
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
