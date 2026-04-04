#!/usr/bin/env bash
# Build a "display" MP4 from iPhone MOV/MP4 without applying rotation metadata:
#   -noautorotate keeps encoded pixel layout as stored
# Then crop (default: centered square 1080x1080 from 1920x1080).
#
# Usage:
#   ./tools/iphone_flat_crop_display.sh INPUT.MOV OUTPUT_display.mp4
# Override crop filter:
#   CROP="1280:720:320:180" ./tools/iphone_flat_crop_display.sh in.mov out.mp4

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${1:?input video}"
OUT="${2:?output mp4}"
CROP="${CROP:-1080:1080:(iw-ow)/2:(ih-oh)/2}"

ffmpeg -y -noautorotate -i "$SRC" \
  -vf "crop=${CROP}" \
  -c:v libx264 -crf 20 -preset fast -movflags +faststart -an \
  "$OUT"
echo "Wrote $OUT"
