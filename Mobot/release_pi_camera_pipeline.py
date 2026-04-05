"""
Release the Pi camera / libcamera pipeline before opening Picamera2.

1) SIGTERM then SIGKILL any process (including other Python line followers) that
   holds /dev/video* — excludes the current PID.
2) Stop user PipeWire / WirePlumber / sockets and kill stray pipewire processes.
3) pkill rpicam-vid / rpicam-hello.

Mirrors the README and mobot_nn_line_follower camera-busy help.
"""

from __future__ import annotations

import glob
import os
import signal
import subprocess
import time


def _pids_holding_video_devices() -> set[int]:
    """PIDs with an open fd on /dev/video* (best effort; same-user processes)."""
    pids: set[int] = set()
    for dev in sorted(glob.glob("/dev/video*")):
        try:
            r = subprocess.run(
                ["lsof", "-t", dev],
                capture_output=True,
                text=True,
                timeout=20,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        for tok in r.stdout.split():
            if tok.isdigit():
                pids.add(int(tok))
    # sudo path: holders invisible to user lsof (needs NOPASSWD or prior auth)
    devs = sorted(glob.glob("/dev/video*"))
    if devs:
        try:
            r = subprocess.run(
                ["sudo", "-n", "lsof", "-t", *devs],
                capture_output=True,
                text=True,
                timeout=20,
            )
            if r.returncode == 0:
                for tok in r.stdout.split():
                    if tok.isdigit():
                        pids.add(int(tok))
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    return pids


def _kill_video_holders(*, exclude: set[int]) -> None:
    targets = _pids_holding_video_devices() - exclude
    if not targets:
        return
    print(
        f"Stopping prior camera users (holding /dev/video*): {sorted(targets)}",
        flush=True,
    )
    for pid in targets:
        if pid <= 1:
            continue
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    time.sleep(1.5)
    for pid in targets:
        if pid <= 1:
            continue
        try:
            os.kill(pid, 0)
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            pass
    time.sleep(0.5)


def release_pi_camera_pipeline(*, aggressive: bool = True) -> None:
    """Free the Pi camera: kill other video users, then stop PipeWire / rpicam."""
    me = {os.getpid()}
    _kill_video_holders(exclude=me)

    dn = subprocess.DEVNULL
    subprocess.run(
        [
            "systemctl",
            "--user",
            "stop",
            "wireplumber",
            "pipewire",
            "pipewire-pulse",
            "pipewire.socket",
            "pipewire-pulse.socket",
        ],
        stdout=dn,
        stderr=dn,
        timeout=60,
    )
    if aggressive:
        for sig, name in (("-9", "pipewire"), ("-9", "wireplumber"), ("-9", "pipewire-pulse")):
            subprocess.run(["pkill", sig, name], stdout=dn, stderr=dn)
    for name in ("rpicam-vid", "rpicam-hello"):
        subprocess.run(["pkill", name], stdout=dn, stderr=dn)
    time.sleep(2)
