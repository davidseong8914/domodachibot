# domodachibot notes

Notes for Pi camera streaming/recording and for **line-following** data + SteerNet training on the Mac.

---

## Line following: dataset layout and workflow

### Layout

- **`test_*_.../`** — one folder per capture session.
- **`frames_5hz/`** — JPEGs named `frame_0001.jpg`, … (from `tools/extract_frames.py`).
- **`labels_5hz.jsonl`** — one JSON object per line: `image`, `target_px` or `theta_rad` (see `line_follow/angles.py`).
- **`labels_all_5hz.jsonl`** (repo root) — merged labels from all sessions (see below).

### End-to-end workflow

1. **Extract frames** from a video (default 0.2 Hz = one frame every 5 s):

   ```bash
   python3 tools/extract_frames.py path/to/video.mp4 -o test_x/frames_5hz
   ```

2. **Label** (click target pixel; bottom-center steering convention):

   ```bash
   python3 tools/label_steering.py --frames-dir test_x/frames_5hz --labels test_x/labels_5hz.jsonl
   ```

3. **Optional: remove frames** (disk only). If you delete JPEGs, fix labels next.

   ```bash
   python3 tools/prune_frames.py --dir test_x/frames_5hz --max-index 407    # drop index > 407
   python3 tools/prune_frames.py --dir test_x/frames_5hz --drop-multiples-of 3  # drop 3,6,9,…
   python3 tools/prune_frames.py -n --dir test_x/frames_5hz …   # dry-run
   ```

4. **Drop label rows** for missing files (rewrites the JSONL in place):

   ```bash
   python3 tools/sync_labels_to_disk.py --labels test_x/labels_5hz.jsonl
   ```

5. **Merge** sessions into one pool (later inputs override same `image` key). All clips are **mixed**; split happens at train time.

   ```bash
   python3 tools/merge_labels.py \
     test_1_0404_sunny/labels_5hz.jsonl \
     test_2_0404_sunny/labels_5hz.jsonl \
     test_3_0404_sunny/labels_5hz.jsonl \
     test_4_0404_sunny/labels_5hz.jsonl \
     -o labels_all_5hz.jsonl
   ```

6. **Train** SteerNet with a **random split** on the merged file (default in `train_steer`): frames from every session can land in train, val, or test. Use `--val-frac` / `--test-frac` (defaults **0.15** each) and `--seed` for a reproducible shuffle. `--augment` applies random appearance on **train** only.

   ```bash
   python3 -m line_follow.train_steer \
     --labels labels_all_5hz.jsonl \
     --split random \
     --val-frac 0.15 \
     --test-frac 0.15 \
     --seed 42 \
     --augment \
     --out line_follow/weights/steer.pt
   ```

   **Monitor training:** by default metrics print every 10 epochs. Use `--log-every 1` for every epoch, `--csv-log runs/metrics.csv` to plot curves, and `pip install tqdm` for an epoch progress bar:

   ```bash
   python3 -m line_follow.train_steer \
     --labels labels_all_5hz.jsonl \
     --split random --val-frac 0.15 --test-frac 0.15 --seed 42 \
     --augment --log-every 1 --csv-log runs/steer_metrics.csv \
     --out line_follow/weights/steer.pt
   ```

   **Clip split (optional):** hold out whole sessions by path substring, e.g. `--split clip --clip-val test_2_0404_sunny --clip-test test_3_0404_sunny`.

7. **Eval / inference:** `python3 -m line_follow.eval …` (see `--help`), learned path `line_follow/learned_predict.py`, classical baseline `line_follow/predict.py`.

### What `--augment` does (in training)

Implemented in `line_follow/label_dataset.py` when the train dataset is built with `augment=True`: HSV jitter, per-channel RGB gains, partial desaturation, small in-plane rotation, Gaussian blur, additive Gaussian noise. **Labels are not changed** (fine for mild rotation and photometric noise).

### Augmentation vs “how many images”

- The **dataset size is still the number of labeled images** (e.g. 500). Each **epoch** visits each training index **once** (subject to batching), same as without augmentation.
- Augmentation **does not** turn 500 rows into `500 × (number of aug types)` separate examples in one epoch. Each time a row is loaded, the **pixels** are randomly perturbed; over **many epochs** the model sees many different looks for the same label.
- To literally train **more gradient steps per epoch** on the same labels, you would add **oversampling** (e.g. a sampler that draws indices with replacement), which is separate from augmentation.

### One more capture session (copy-paste)

Use a **new folder name** each time (example: `test_5_0406_lane`). **Stop** `mobot_nn_line_follower` and any camera app before recording.

**1 — Pi: record driving video** (720p H.264 until Ctrl+C; tune path/name):

```bash
mkdir -p ~/Desktop/domodachibot/test_5_0406_lane
rpicam-vid -n --codec h264 --width 1280 --height 720 --low-latency -t 0 \
  -o ~/Desktop/domodachibot/test_5_0406_lane/drive.h264
# Ctrl+C when done
ffmpeg -y -i ~/Desktop/domodachibot/test_5_0406_lane/drive.h264 -c copy \
  ~/Desktop/domodachibot/test_5_0406_lane/drive.mp4
```

**2 — Copy the repo (with the new folder) to the Mac** (replace user/host/path):

```bash
rsync -avz pi@raspberrypi.local:~/Desktop/domodachibot/ ~/Projects/domodachibot/
# or scp just test_5_0406_lane/drive.mp4
```

**3 — Mac: extract frames** (default **0.2 Hz** = one frame every 5 s; add e.g. `--fps 0.5` for denser samples):

```bash
cd ~/Projects/domodachibot   # repo root on Mac
python3 tools/extract_frames.py test_5_0406_lane/drive.mp4 -o test_5_0406_lane/frames_5hz
```

**4 — Mac: label** (`n` / `p` / `s` / `q`; click target on line):

```bash
python3 tools/label_steering.py \
  --frames-dir test_5_0406_lane/frames_5hz \
  --labels test_5_0406_lane/labels_5hz.jsonl
```

**5 — Mac: merge into the pool** (put **existing** `labels_all_5hz.jsonl` first, then the **new** session; same `image` path twice → later file wins):

```bash
python3 tools/merge_labels.py \
  labels_all_5hz.jsonl \
  test_5_0406_lane/labels_5hz.jsonl \
  -o labels_all_5hz.jsonl
```

**6 — Mac: train** (copy `line_follow/weights/steer.pt` back to the Pi when happy):

```bash
python3 -m line_follow.train_steer \
  --labels labels_all_5hz.jsonl \
  --split random --val-frac 0.15 --test-frac 0.15 --seed 42 \
  --augment --out line_follow/weights/steer.pt
```

Optional: `python3 tools/sync_labels_to_disk.py --labels test_5_0406_lane/labels_5hz.jsonl` if you deleted frames after labeling.

---

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

### Line follow — SteerNet (`mobot_nn_line_follower.py`)

| Topic | Notes |
|--------|--------|
| **Primary runner** | Repo root **`mobot_nn_line_follower.py`** (PyTorch + Picamera2). Classical OpenCV followers live under **`Mobot/`**; they are separate and also pre-free the camera via **`Mobot/release_pi_camera_pipeline.py`**. |
| **Weights** | Train on Mac → copy **`line_follow/weights/steer.pt`** to the Pi. |
| **Drive speeds** | Edit **`NORMAL_SPEED` / `SLOW_SPEED`** at the top of `mobot_nn_line_follower.py` (not the Mobot CV scripts). |

**Run (from repo root on the Pi):**

```bash
cd ~/Desktop/domodachibot   # or your path
source venv/bin/activate
python3 -u mobot_nn_line_follower.py --weights line_follow/weights/steer.pt
```

First load can sit **30–90s** with no extra output while PyTorch imports; `-u` keeps logs line-buffered. Use **`--simulate`** only when you want **no GPIO** (camera still runs; **servo and motor commands are ignored**).

**Bench / SSH: steering only, drive motor off:**

```bash
python3 -u mobot_nn_line_follower.py --weights line_follow/weights/steer.pt --steer-only --steer-sign -1
# larger servo swings for debugging:
#   --steer-test-mult 3   (default 2)
#   --theta-gain 1.3      (default; raise if turns feel weak)
```

**Steering math (what the logs mean):**  
`servo_cmd` (clamped to **±40°** hardware offset around center **90°**) is computed as:

`offset = steer_sign × degrees(theta_net) × theta_gain`  
and with **`--steer-only`**, multiplied by **`--steer-test-mult`** before clamp.  
Convention: **`line_follow/angles.py`** — **θ > 0** points toward **+x** (right in the image). **`--steer-sign`** flips the **wheel command only**, not the NN/overlay angle.

**HTTP MJPEG stream** (no `rpicam-vid` at the same time):

```bash
python3 -u mobot_nn_line_follower.py --weights line_follow/weights/steer.pt --stream
# optional: --stream-port 8888
```

Viewer on Mac/phone: open the printed `http://…:8888/` or `ffplay -fflags nobuffer -framedrop http://PI:8888/`. Prefer LAN IP when possible; VPN IPs (e.g. `172.26.x.x`) work but often add latency.

**Stream overlay (diagnostics):** Gray vertical = image center. **Red** dot = bottom-center origin (`angles.py`). **Magenta** and **green** share the same bearing = **θ_net** in the image. **Green** length scales with **|servo_cmd| / SERVO_MAX_OFFSET** (~command magnitude), not a second angle. Overlay text is shortened to fit 640-wide MJPEG.

**If the servo does not move**

1. Confirm you did **not** pass **`--simulate`**.
2. **`--steer-only`** still drives the **servo**; only **drive PWM** is forced to 0.
3. Try **`--steer-test-mult 3`** (or higher) and/or raise **`--theta-gain`** so the command visibly hits the servo range.
4. **Wiring:** steering PWM is **GPIO 16 (BCM)** in `Mobot/mobot_line_follower_headless.py` (`MotorController`). Servo **power** must be adequate (**5 V**, shared ground with Pi); do not expect a bare signal wire alone to move the horn.
5. If GPIO permission errors appear, ensure user **`pi`** is in the **`gpio`** group (or run from an environment where RPi.GPIO can access pins).
6. Still stuck: verify the horn on a known-good servo tester or minimal PWM script; check for another process using the same pin.

### Camera “Device or resource busy”

Only one process can use the Pi camera. By default, **`mobot_nn_line_follower.py`** and Mobot camera scripts (`*line_follower*.py`, `camera_test.py`, `tune_params_live.py`) run **`Mobot/release_pi_camera_pipeline.py`**: **SIGTERM/SIGKILL** on other **`/dev/video*`** holders, then stop PipeWire / rpicam. Use **`--no-pipeline-release`** to skip that. If open still fails, inspect holders manually:

```bash
sudo fuser -v /dev/video0
sudo lsof /dev/video* 2>/dev/null
```

- **`python` with a PID:** usually another script or a stuck `mobot_nn_line_follower` — `ps -p PID -o args=` then `kill PID`, or `pkill -f mobot_nn_line_follower`.
- **`pipewire` / `wireplumber`:** the desktop stack keeps the camera nodes open; **`pipewire.socket`** can restart PipeWire after a plain `stop pipewire`. Use:

  ```bash
  systemctl --user stop wireplumber pipewire pipewire-pulse pipewire.socket pipewire-pulse.socket
  # run your line follower, then:
  systemctl --user start pipewire.socket pipewire pipewire-pulse.socket pipewire-pulse wireplumber
  ```

  Or run the robot **without the full desktop** (console / SSH-only) so nothing grabs the camera.

- **Still “Pipeline handler in use” after stopping PipeWire:** sockets can respawn it, or another user service holds the pipeline. Try:

  ```bash
  systemctl --user stop wireplumber pipewire pipewire-pulse pipewire.socket pipewire-pulse.socket 2>/dev/null
  pkill -9 pipewire; pkill -9 wireplumber; pkill -9 pipewire-pulse; sleep 2
  sudo lsof /dev/video0
  ```

  If nothing is listed but Picamera2 still fails, temporarily **mask** the socket, run your script, then unmask:

  ```bash
  systemctl --user mask pipewire.socket
  # run mobot_nn_line_follower …
  systemctl --user unmask pipewire.socket
  systemctl --user start pipewire.socket pipewire pipewire-pulse.socket pipewire-pulse wireplumber
  ```

  Optionally stop the desktop portal: `systemctl --user stop xdg-desktop-portal xdg-desktop-portal-gtk 2>/dev/null`. Last resort: **reboot** and start `mobot_nn_line_follower` before opening any camera app.

- **`pkill -f mobot_nn_line_follower` missed it:** **`motor_line_follower_vnc.py`** (VNC line follower) is a separate process — `pkill -f motor_line_follower_vnc` or `kill` its PID from `pgrep -af python`. Recheck `sudo lsof /dev/video*`; if **pipewire** still holds devices, run `systemctl --user stop wireplumber pipewire pipewire.socket`.

- Also stop `rpicam-vid` / `rpicam-hello` if you use them elsewhere.

---

## Stream to Mac (low latency)

Use **raw H.264 over TCP** from the Pi and **ffplay on the Mac**. Avoid piping through VLC/RTSP if you care about delay—RTSP tends to add a lot of buffering.

**Note:** If you are running **`mobot_nn_line_follower.py --stream`**, that uses **HTTP MJPEG** (see above), not this `rpicam-vid` TCP pipeline. Use `rpicam-vid` here only when the Python line follower is **not** using the camera.

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
