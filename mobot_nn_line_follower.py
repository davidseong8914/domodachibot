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
  --preview-port Serve browser preview (MJPEG) on 0.0.0.0:PORT; open http://127.0.0.1:PORT/ on the Pi
                 or ssh -L PORT:127.0.0.1:PORT pi@PI ... then http://127.0.0.1:PORT/ on your laptop.
"""

from __future__ import annotations

import argparse
import math
import socket
from typing import Any
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

# Repo root = directory that contains line_follow/ (works if this file is in root or in Mobot/)
_here = Path(__file__).resolve().parent
REPO_ROOT = _here if (_here / "line_follow").is_dir() else _here.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# Colleague scripts live in Mobot/
_mobot_dir = REPO_ROOT / "Mobot"
if _mobot_dir.is_dir() and str(_mobot_dir) not in sys.path:
    sys.path.insert(0, str(_mobot_dir))

from line_follow.angles import bottom_center_origin, draw_ray  # noqa: E402
from line_follow.learned_predict import predict_steering_learned  # noqa: E402

from mobot_line_follower_headless import (  # noqa: E402
    LENS_POSITION,
    MotorController,
    NORMAL_SPEED,
    SERVO_MAX_OFFSET,
    SLOW_SPEED,
)

_BOUNDARY = b"--jpgboundary\r\n"
_PREVIEW_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>NN line follow preview</title></head>
<body style="margin:0;background:#111;color:#ccc;font-family:sans-serif;">
<p style="margin:8px;">SteerNet live view (MJPEG). The robot uses the same camera feed.</p>
<img src="/stream" style="max-width:100%;height:auto;" alt="camera" />
</body></html>
""".encode("utf-8")


class _PreviewState:
    __slots__ = ("_lock", "_jpg")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jpg: bytes | None = None

    def set_jpeg(self, data: bytes) -> None:
        with self._lock:
            self._jpg = data

    def get_jpeg(self) -> bytes | None:
        with self._lock:
            return self._jpg


def _start_preview_http(port: int, state: _PreviewState) -> ThreadingHTTPServer:
    state_ref = state

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *_args: object) -> None:
            return

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(_PREVIEW_HTML)
                return
            if self.path == "/stream":
                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    "multipart/x-mixed-replace; boundary=jpgboundary",
                )
                self.send_header("Cache-Control", "no-store, no-cache")
                self.send_header("Pragma", "no-cache")
                self.end_headers()
                try:
                    while True:
                        jpg = state_ref.get_jpeg()
                        if jpg:
                            self.wfile.write(
                                _BOUNDARY
                                + b"Content-Type: image/jpeg\r\n"
                                + b"Content-Length: "
                                + str(len(jpg)).encode("ascii")
                                + b"\r\n\r\n"
                                + jpg
                                + b"\r\n"
                            )
                        time.sleep(0.05)
                except (BrokenPipeError, ConnectionResetError):
                    return
            self.send_error(404)

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _encode_preview_frame(
    frame_bgr: Any, theta_rad: float, offset_deg: float, speed: float
) -> bytes:
    h, w = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
    ox, oy = bottom_center_origin(w, h)
    ray_len = min(w, h) * 0.5
    # Same heading convention as training / nn_overlay_video.py (from bottom-center origin).
    vis = draw_ray(
        frame_bgr, (ox, oy), theta_rad, ray_len, color=(255, 255, 0), thickness=3
    )
    tdeg = math.degrees(theta_rad)
    cv2.putText(
        vis,
        f"theta={tdeg:+.1f}deg  servo={offset_deg:+.1f}deg  spd={int(speed)}",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    ok, buf = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return b""
    return buf.tobytes()


def run_live(
    weights: Path,
    steer_sign: float,
    theta_gain: float,
    simulate: bool,
    preview_port: int,
) -> None:
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

    preview_server: ThreadingHTTPServer | None = None
    preview_state: _PreviewState | None = None
    if preview_port > 0:
        preview_state = _PreviewState()
        preview_server = _start_preview_http(preview_port, preview_state)
        host = socket.gethostname()
        print(
            f"Browser preview: http://127.0.0.1:{preview_port}/  "
            f"(on this Pi)  —  LAN: http://{host}:{preview_port}/  "
            f"(if reachable). SSH tunnel: ssh -L {preview_port}:127.0.0.1:{preview_port} pi@PI"
        )

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

            if preview_state is not None:
                preview_state.set_jpeg(
                    _encode_preview_frame(frame, theta, offset, speed)
                )

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
        if preview_server is not None:
            preview_server.shutdown()
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
    parser.add_argument(
        "--preview-port",
        type=int,
        default=0,
        metavar="PORT",
        help="HTTP MJPEG preview on 0.0.0.0:PORT (0 = disabled). Example: --preview-port 8080",
    )
    args = parser.parse_args()

    if not args.weights.is_file():
        print(f"error: weights not found: {args.weights.resolve()}", file=sys.stderr)
        sys.exit(1)

    if args.preview_port < 0 or args.preview_port > 65535:
        print("error: --preview-port must be in 0..65535", file=sys.stderr)
        sys.exit(1)

    run_live(
        args.weights,
        args.steer_sign,
        args.theta_gain,
        args.simulate,
        args.preview_port,
    )


if __name__ == "__main__":
    main()
