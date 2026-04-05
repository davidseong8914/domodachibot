#!/usr/bin/env python3
"""
Line follower on Pi: Picamera2 + SteerNet (.pt) + same motor/servo stack as mobot_line_follower_headless.

Repo root must be on PYTHONPATH (run from repo root or from this folder — see below).

SSH on Pi (typical):
  cd ~/domodachibot   # or your clone path
  source .venv/bin/activate   # venv with torch + opencv + picamera2
  python3 mobot_nn_line_follower.py --weights line_follow/weights/steer.pt
  # or: python3 Mobot/mobot_nn_line_follower.py  (if kept under Mobot/)

Training stays on the Mac; copy steer.pt to the Pi (scp/USB). Install PyTorch for Pi arm64 if needed.

Flags:
  --steer-sign   Flip if robot turns the wrong way (+1 or -1).
  --theta-gain   Scale: servo_offset ≈ steer_sign * degrees(theta) * gain (before clamp).
  --simulate     No GPIO (prints steering only); still needs Pi camera unless you hack in a webcam.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

# Repo root = directory that contains line_follow/ (works if this file is in root or in Mobot/)
_here = Path(__file__).resolve().parent
REPO_ROOT = _here if (_here / "line_follow").is_dir() else _here.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# Colleague scripts live in Mobot/
_mobot_dir = REPO_ROOT / "Mobot"
if _mobot_dir.is_dir() and str(_mobot_dir) not in sys.path:
    sys.path.insert(0, str(_mobot_dir))

from line_follow.learned_predict import predict_steering_learned  # noqa: E402

from mobot_line_follower_headless import (  # noqa: E402
    LENS_POSITION,
    MotorController,
    NORMAL_SPEED,
    SERVO_MAX_OFFSET,
    SLOW_SPEED,
)


def run_live(weights: Path, steer_sign: float, theta_gain: float, simulate: bool) -> None:
    from picamera2 import Picamera2
    from libcamera import controls as libcam_controls

    print("Initializing camera...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "BGR888"}
    )
    picam2.configure(config)
    picam2.start()

    print("Waiting for auto-exposure to settle...")
    time.sleep(2)

    picam2.set_controls(
        {
            "AfMode": libcam_controls.AfModeEnum.Manual,
            "LensPosition": LENS_POSITION,
        }
    )
    time.sleep(0.5)
    print(f"Camera ready — LensPosition: {LENS_POSITION}")

    motors = MotorController(simulate=simulate)
    wpath = weights.resolve()
    print(f"SteerNet weights: {wpath}")
    print("Running. Ctrl+C to stop.\n")

    frame_count = 0
    try:
        while True:
            frame = picam2.capture_array("main")
            theta = predict_steering_learned(frame, wpath)

            offset = steer_sign * math.degrees(theta) * theta_gain
            offset = max(-SERVO_MAX_OFFSET, min(SERVO_MAX_OFFSET, offset))

            if abs(offset) > 25:
                speed = SLOW_SPEED
            else:
                speed = NORMAL_SPEED

            motors.set_steering(offset)
            motors.set_drive_speed(int(speed))

            frame_count += 1
            if frame_count % 30 == 0:
                print(
                    f"[{frame_count}] theta={math.degrees(theta):+6.2f}°  "
                    f"servo_off={offset:+.1f}°  spd={int(speed)}"
                )

            time.sleep(0.03)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        motors.stop()
        motors.cleanup()
        picam2.stop()
        print("Shutdown complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pi line follow with SteerNet checkpoint")
    parser.add_argument(
        "--weights",
        type=Path,
        default=REPO_ROOT / "line_follow" / "weights" / "steer.pt",
        help="Path to steer.pt (from train_steer on Mac)",
    )
    parser.add_argument(
        "--steer-sign",
        type=float,
        default=1.0,
        help="+1 default; set -1 if robot steers opposite to the line",
    )
    parser.add_argument(
        "--theta-gain",
        type=float,
        default=1.0,
        help="Multiply steering offset from |theta| in degrees (tune if turns too weak/strong)",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Do not touch GPIO (still captures camera on Pi)",
    )
    args = parser.parse_args()

    if not args.weights.is_file():
        print(f"error: weights not found: {args.weights.resolve()}", file=sys.stderr)
        sys.exit(1)

    run_live(args.weights, args.steer_sign, args.theta_gain, args.simulate)


if __name__ == "__main__":
    main()
