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
   python3 tools/label_steering.py --frames test_x/frames_5hz --labels test_x/labels_5hz.jsonl
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
