#!/usr/bin/env python3
"""
Pi line follower — primary on-robot runner: Picamera2 + SteerNet (.pt).

Run from the **repo root** (this file lives next to `line_follow/`, not under Mobot/).

  cd /path/to/domodachibot
  source venv/bin/activate   # torch + opencv + picamera2 (+ system libcamera)
  python3 mobot_nn_line_follower.py --weights line_follow/weights/steer.pt

`Mobot/*` scripts are classical CV / legacy helpers; tune speeds here (`NORMAL_SPEED` /
`SLOW_SPEED`), not in `mobot_line_follower_headless.py`.

Training on Mac; copy `steer.pt` to the Pi. Motor wiring matches `MotorController` in Mobot.

Flags:
  --steer-sign   +1 default. If line is RIGHT of center but theta/ray point LEFT, try -1 (see line_follow/angles.py).
  --theta-gain   Scale: servo_offset ≈ steer_sign * degrees(theta) * gain (before clamp).
  --simulate     No GPIO; still uses the Pi camera.
  --frame-sleep  Extra seconds after each loop (default 0). Use e.g. 0.01 to cap CPU.
  --stream       HTTP MJPEG (browser or ffplay); no ffmpeg; same camera as line follow.
  --stream-port  Port for --stream (default 8888).
  --steer-only   Drive motor always off; steer from NN only (for bench testing).
  --steer-test-mult  Multiply servo command in --steer-only (default 2.0 = more obvious motion).
"""

from __future__ import annotations

import argparse
import http.server
import math
import queue
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (OSError, ValueError):
        pass

import cv2
import numpy as np

# Repo root (this script is at domodachibot/mobot_nn_line_follower.py).
_script_dir = Path(__file__).resolve().parent
REPO_ROOT = _script_dir if (_script_dir / "line_follow").is_dir() else _script_dir.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
_MOBOT_DIR = REPO_ROOT / "Mobot"
if str(_MOBOT_DIR) not in sys.path:
    sys.path.insert(0, str(_MOBOT_DIR))

print(
    "Loading PyTorch + SteerNet (first time can take 30–90s on a Pi; not hung)…",
    flush=True,
)
from line_follow.angles import bottom_center_origin, draw_ray  # noqa: E402
from line_follow.learned_predict import (  # noqa: E402
    predict_steering_learned,
    warmup_steer_net,
)

from mobot_line_follower_headless import (  # noqa: E402 — GPIO + lens only; speeds are below
    LENS_POSITION,
    MotorController,
    SERVO_MAX_OFFSET,
)
from release_pi_camera_pipeline import release_pi_camera_pipeline  # noqa: E402

# Drive PWM (0–255): NN follower speeds live here only (not tied to CV scripts in Mobot/).
NORMAL_SPEED = 40
SLOW_SPEED = 40
# In --steer-only mode: servo offset is multiplied by this after theta_gain (default = “double sensitivity”).
STEER_TEST_MULT_DEFAULT = 2.0

# BGR frames from Picamera2 (must match preview main size when streaming).
STREAM_WIDTH = 640
STREAM_HEIGHT = 480
STREAM_JPEG_QUALITY = 82


@dataclass
class _MjpegState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    jpeg: bytes = field(default_factory=bytes)


def _print_camera_busy_help() -> None:
    print(
        "\nThe Pi camera is already in use (only one app can open it at a time).\n\n"
        "0) By default this script runs release_pi_camera_pipeline() first (stops PipeWire and\n"
        "   SIGTERM/SIGKILLs other processes holding /dev/video*). If you used\n"
        "   --no-pipeline-release, retry without it. Otherwise another user/process may hold the device.\n\n"
        "1) Find and kill the holder (often NOT named mobot_nn_line_follower):\n"
        "     pgrep -af python\n"
        "     sudo lsof /dev/video* 2>/dev/null\n"
        "     sudo fuser -v /dev/video0\n"
        "   Then: kill PID   or   pkill -f motor_line_follower_vnc   or   pkill -f picamera2\n"
        "   Wait 1–2s, then retry. If still stuck: sudo kill -9 PID\n\n"
        "2) Close the browser tab / ffplay showing the MJPEG stream (old run may still\n"
        "   have released the camera — if not, step 1).\n\n"
        "3) PipeWire (often still grabs the camera after a plain stop):\n"
        "     systemctl --user stop wireplumber pipewire pipewire-pulse \\\n"
        "       pipewire.socket pipewire-pulse.socket 2>/dev/null\n"
        "     sudo lsof /dev/video0   # should be empty; if pipewire is back, force:\n"
        "     pkill -9 pipewire; pkill -9 wireplumber; sleep 2\n"
        "   Hard block socket activation until you are done:\n"
        "     systemctl --user mask pipewire.socket 2>/dev/null\n"
        "   Restore desktop audio/camera portal afterward:\n"
        "     systemctl --user unmask pipewire.socket\n"
        "     systemctl --user start pipewire.socket pipewire pipewire-pulse.socket pipewire-pulse wireplumber\n\n"
        "4) Portal / VNC stack sometimes uses the camera:\n"
        "     systemctl --user stop xdg-desktop-portal xdg-desktop-portal-gtk 2>/dev/null\n\n"
        "5) Other:\n"
        "     pkill rpicam-vid 2>/dev/null; pkill rpicam-hello 2>/dev/null\n"
        "   Still stuck: reboot the Pi, then run mobot before opening desktop camera apps.\n\n",
        file=sys.stderr,
    )


def _guess_lan_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.2)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "YOUR_PI_IP"


def _run_mjpeg_encoder(
    frame_q: queue.Queue[np.ndarray],
    state: _MjpegState,
    stop: threading.Event,
) -> None:
    while not stop.is_set():
        try:
            fr = frame_q.get(timeout=0.25)
        except queue.Empty:
            continue
        if fr.shape[1] != STREAM_WIDTH or fr.shape[0] != STREAM_HEIGHT:
            fr = cv2.resize(
                fr,
                (STREAM_WIDTH, STREAM_HEIGHT),
                interpolation=cv2.INTER_LINEAR,
            )
        ok, buf = cv2.imencode(
            ".jpg",
            fr,
            [int(cv2.IMWRITE_JPEG_QUALITY), STREAM_JPEG_QUALITY],
        )
        if ok:
            blob = buf.tobytes()
            with state.lock:
                state.jpeg = blob


def _make_mjpeg_handler(state: _MjpegState):
    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path not in ("/", "/index.html", "/video", "/mjpeg"):
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header(
                "Content-Type",
                "multipart/x-mixed-replace; boundary=frame",
            )
            self.end_headers()
            try:
                while True:
                    with state.lock:
                        jpg = state.jpeg
                    if jpg:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(
                            f"Content-Length: {len(jpg)}\r\n\r\n".encode("ascii")
                        )
                        self.wfile.write(jpg)
                        self.wfile.write(b"\r\n")
                    time.sleep(1 / 30.0)
            except (
                BrokenPipeError,
                ConnectionResetError,
                ConnectionAbortedError,
                TimeoutError,
            ):
                pass

        def log_message(self, _format: str, *_args: object) -> None:
            pass

    return _Handler


def _overlay_stream_frame(
    bgr: np.ndarray,
    theta_rad: float,
    servo_deg: float,
    speed: int,
    frame_idx: int,
    steer_sign: float,
) -> np.ndarray:
    """
    Both rays use the same *bearing* theta_net (image-plane heading per angles.py: toward the line).
    Magenta = fixed length (raw NN direction). Green = same angle, length scales with |servo_cmd|
    so it does not misuse wheel degrees as an image angle (--steer-sign flips servo sign only).
    """
    vis = bgr.copy()
    h, w = vis.shape[:2]
    cx = w // 2
    origin = bottom_center_origin(w, h)
    ray_len = float(min(w, h) * 0.35)

    cv2.line(vis, (cx, 0), (cx, h), (60, 60, 60), 1, cv2.LINE_AA)

    vis = draw_ray(vis, origin, theta_rad, ray_len * 0.5, (255, 0, 255), 1)
    strength = max(0.12, min(1.0, abs(servo_deg) / float(SERVO_MAX_OFFSET)))
    green_len = ray_len * strength
    vis = draw_ray(vis, origin, theta_rad, green_len, (0, 255, 0), 3)

    ox, oy = origin
    tip_x = int(round(ox + green_len * math.sin(theta_rad)))
    tip_y = int(round(oy - green_len * math.cos(theta_rad)))
    cv2.circle(vis, (ox, oy), 7, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(vis, (tip_x, tip_y), 5, (0, 255, 255), -1, cv2.LINE_AA)

    bar_h = 98
    cv2.rectangle(vis, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.rectangle(vis, (0, 0), (w, bar_h), (200, 200, 200), 1)
    y0 = 22
    fs = 0.52
    col = (0, 255, 255)
    edge = (0, 0, 0)
    lines = [
        f"theta_net {math.degrees(theta_rad):+.2f} deg   steer_sign {steer_sign:+.0f}  "
        f"(+theta => RIGHT per angles.py)",
        f"servo_cmd {servo_deg:+.1f} deg (clamp +/-{SERVO_MAX_OFFSET})   drive_PWM {speed}",
        f"frame {frame_idx} | same bearing; green len ~ |servo|/max",
        "Rays = theta_net in image; steer_sign changes servo only (not ray angle).",
    ]
    for i, t in enumerate(lines):
        y = y0 + i * 22
        cv2.putText(vis, t, (8, y), cv2.FONT_HERSHEY_SIMPLEX, fs, edge, 3, cv2.LINE_AA)
        cv2.putText(vis, t, (8, y), cv2.FONT_HERSHEY_SIMPLEX, fs, col, 1, cv2.LINE_AA)
    return vis


def run_live(
    weights: Path,
    steer_sign: float,
    theta_gain: float,
    simulate: bool,
    frame_sleep: float,
    stream: bool,
    stream_port: int,
    steer_only: bool,
    steer_test_mult: float,
    release_pipeline: bool = True,
) -> None:
    from picamera2 import Picamera2
    from libcamera import controls as libcam_controls

    if release_pipeline:
        print(
            "Releasing camera pipeline (prior users + PipeWire / rpicam)…",
            flush=True,
        )
        release_pi_camera_pipeline()

    print("Initializing camera…", flush=True)
    try:
        picam2 = Picamera2()
    except RuntimeError as e:
        err = str(e).lower()
        if "busy" in err or "did not complete" in err or "acquire" in err:
            print(f"error: {e}", file=sys.stderr)
            _print_camera_busy_help()
            sys.exit(1)
        raise
    # Fewer buffers + no frame queue → fresher frames, lower end-to-end latency.
    config = picam2.create_preview_configuration(
        main={"size": (STREAM_WIDTH, STREAM_HEIGHT), "format": "BGR888"},
        buffer_count=2,
        queue=False,
    )
    print("Configuring Picamera2…", flush=True)
    picam2.configure(config)
    picam2.start()
    print("Camera started. Waiting 2s for auto-exposure…", flush=True)
    time.sleep(2)

    picam2.set_controls(
        {
            "AfMode": libcam_controls.AfModeEnum.Manual,
            "LensPosition": LENS_POSITION,
        }
    )
    time.sleep(0.5)
    print(f"Camera ready — LensPosition: {LENS_POSITION}", flush=True)

    stream_httpd: http.server.ThreadingHTTPServer | None = None
    stream_q: queue.Queue[np.ndarray] | None = None
    stream_stop: threading.Event | None = None
    stream_thread: threading.Thread | None = None
    if stream:
        stream_state = _MjpegState()
        stream_q = queue.Queue(maxsize=1)
        stream_stop = threading.Event()
        stream_thread = threading.Thread(
            target=_run_mjpeg_encoder,
            args=(stream_q, stream_state, stream_stop),
            daemon=True,
        )
        stream_thread.start()
        handler = _make_mjpeg_handler(stream_state)
        try:
            stream_httpd = http.server.ThreadingHTTPServer(
                ("0.0.0.0", stream_port), handler
            )
        except OSError as e:
            print(
                f"error: cannot bind HTTP stream on port {stream_port}: {e}",
                file=sys.stderr,
            )
            stream_stop.set()
            stream_thread.join(timeout=2.0)
            picam2.stop()
            sys.exit(1)
        threading.Thread(target=stream_httpd.serve_forever, daemon=True).start()
        ip_hint = _guess_lan_ip()
        print(
            f"MJPEG stream: http://{ip_hint}:{stream_port}/  "
            f"(use this URL on your Mac / phone on the same LAN)"
        )
        print(
            "  ffplay: ffplay -fflags nobuffer -framedrop "
            f"http://{ip_hint}:{stream_port}/"
        )
        print("  Or open the URL in a browser (Chrome, Safari).")
        print(
            "  Stream overlay: magenta + green share NN theta bearing (toward line); "
            "green length ~ |servo|/max. steer_sign does not rotate the rays. Red = axle.",
            flush=True,
        )

    print("Starting GPIO / motor controller…", flush=True)
    motors = MotorController(simulate=simulate)
    wpath = weights.resolve()
    print(f"SteerNet weights: {wpath}", flush=True)
    print("Warming up inference (loads .pt into RAM)…", flush=True)
    warmup_steer_net(wpath)
    if steer_only:
        print(
            f"Steer-only: drive PWM=0, steering ×{steer_test_mult:g} after --theta-gain, "
            f"clamp ±{SERVO_MAX_OFFSET}°. Logging every 10 frames.",
            flush=True,
        )
    print("Running — you should see lines below every ~0.3s. Ctrl+C to stop.\n", flush=True)

    frame_count = 0
    log_every = 10 if steer_only else 30
    try:
        while True:
            frame = picam2.capture_array("main")

            theta = predict_steering_learned(frame, wpath)

            offset = steer_sign * math.degrees(theta) * theta_gain
            if steer_only:
                offset *= steer_test_mult
            offset = max(-SERVO_MAX_OFFSET, min(SERVO_MAX_OFFSET, offset))

            if steer_only:
                speed = 0
            elif abs(offset) > 25:
                speed = SLOW_SPEED
            else:
                speed = NORMAL_SPEED

            motors.set_steering(offset)
            motors.set_drive_speed(int(speed))

            frame_count += 1
            if stream_q is not None:
                out = _overlay_stream_frame(
                    frame, theta, offset, int(speed), frame_count, steer_sign
                )
                try:
                    stream_q.put_nowait(out)
                except queue.Full:
                    try:
                        stream_q.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        stream_q.put_nowait(out)
                    except queue.Full:
                        pass
            if frame_count == 1 or frame_count % log_every == 0:
                print(
                    f"[{frame_count}] theta={math.degrees(theta):+6.2f}°  "
                    f"servo_off={offset:+.1f}°  spd={int(speed)}",
                    flush=True,
                )

            if frame_sleep > 0:
                time.sleep(frame_sleep)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        motors.stop()
        motors.cleanup()
        if stream_stop is not None:
            stream_stop.set()
        if stream_thread is not None:
            stream_thread.join(timeout=2.0)
        if stream_httpd is not None:
            threading.Thread(target=stream_httpd.shutdown, daemon=True).start()
            stream_httpd.server_close()
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
        help="Line RIGHT of center should give theta>0 (ray tilts right). If you see the opposite, try -1",
    )
    parser.add_argument(
        "--theta-gain",
        type=float,
        default=1.3,
        help="Multiply steering offset from |theta| in degrees (tune if turns too weak/strong)",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Do not touch GPIO (still captures camera on Pi)",
    )
    parser.add_argument(
        "--frame-sleep",
        type=float,
        default=0.0,
        help="Optional delay (seconds) after each iteration; default 0 for lowest latency",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="HTTP MJPEG on --stream-port (browser or ffplay); no ffmpeg",
    )
    parser.add_argument(
        "--stream-port",
        type=int,
        default=8888,
        help="HTTP port for --stream (default 8888)",
    )
    parser.add_argument(
        "--steer-only",
        action="store_true",
        help="Keep drive motor at 0; only move steering from the network (bench test)",
    )
    parser.add_argument(
        "--steer-test-mult",
        type=float,
        default=STEER_TEST_MULT_DEFAULT,
        help=f"Extra steering scale when using --steer-only (default {STEER_TEST_MULT_DEFAULT})",
    )
    parser.add_argument(
        "--no-pipeline-release",
        action="store_true",
        help="Do not stop PipeWire or kill other /dev/video* users before opening the camera",
    )
    args = parser.parse_args()
    if args.frame_sleep < 0:
        print("error: --frame-sleep must be >= 0", file=sys.stderr)
        sys.exit(1)
    if args.stream_port < 1 or args.stream_port > 65535:
        print("error: --stream-port must be 1-65535", file=sys.stderr)
        sys.exit(1)
    if args.steer_test_mult <= 0:
        print("error: --steer-test-mult must be > 0", file=sys.stderr)
        sys.exit(1)

    if not args.weights.is_file():
        print(f"error: weights not found: {args.weights.resolve()}", file=sys.stderr)
        sys.exit(1)

    run_live(
        args.weights,
        args.steer_sign,
        args.theta_gain,
        args.simulate,
        args.frame_sleep,
        args.stream,
        args.stream_port,
        args.steer_only,
        args.steer_test_mult,
        release_pipeline=not args.no_pipeline_release,
    )


if __name__ == "__main__":
    main()
