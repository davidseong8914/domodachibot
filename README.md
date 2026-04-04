# domodachibot notes

## Pi basics

**Reset password**

```bash
sudo passwd pi
```

**Find IP addresses**

```bash
ip addr
```

**Quick camera test (preview on Pi HDMI / desktop)**

```bash
rpicam-hello -t 0
```

---

## Stream to Mac (low latency)

Use **raw H.264 over TCP** from the Pi and **ffplay on the Mac**. Avoid piping through VLC/RTSP if you care about delay—RTSP tends to add a lot of buffering.

### 1. On the Pi (SSH session)

Stop any old stream first, then:

```bash
pkill rpicam-vid 2>/dev/null; pkill vlc 2>/dev/null; sleep 1
rpicam-vid -t 0 -n --codec h264 --low-latency --inline --intra 15 --framerate 30 --listen -o tcp://0.0.0.0:8888
```

Leave this running. It waits for a client, then sends video.

### 2. On the Mac (local Terminal — not inside SSH)

`ffplay` must run **on the Mac**. If you run it in an SSH shell to the Pi, the window goes to the Pi’s HDMI, not your laptop.

Install FFmpeg once (includes `ffplay`):

```bash
brew install ffmpeg
```

Then connect (use the hostname or IP you use for `ssh`; `ping` it from the Mac first if unsure):

```bash
ffplay -fflags nobuffer -flags low_delay -framedrop -probesize 32 -analyzeduration 0 tcp://raspberrypi.local:8888
```

Example with a numeric IP: `tcp://192.168.1.50:8888`. Prefer LAN (`192.168.x.x`) when Mac and Pi are on the same Wi‑Fi; overlay VPNs (e.g. `172.26.x.x`) work but usually add latency.

**VLC on Mac:** Media → Open Network Stream → `tcp://PI_IP:8888` (same idea as `ffplay`).

### Why this feels better than RTSP

- **Before:** camera → H.264 → VLC (RTSP) → network → player (extra muxing/buffering).
- **Now:** camera → H.264 → TCP → `ffplay`, plus **shorter GOP** (`--intra 15`) so keyframes arrive more often.

---

## Record video (Pi)

Stop any other `rpicam-vid` using the camera (e.g. streaming), then record.

**720p (1280×720), until Ctrl+C**

```bash
rpicam-vid -n --codec h264 --width 1280 --height 720 --low-latency -t 0 -o ~/Desktop/recording.h264
```

**1080p (1920×1080), until Ctrl+C**

```bash
rpicam-vid -n --codec h264 --width 1920 --height 1080 --low-latency -t 0 -o ~/Desktop/recording.h264
```

`-t 0` means run until you press **Ctrl+C**.

**Wrap to MP4 without re-encoding**

```bash
ffmpeg -i ~/Desktop/recording.h264 -c copy ~/Desktop/recording.mp4
```

---

## Roboflow

Gate detection dataset on Roboflow — Andrew account.
