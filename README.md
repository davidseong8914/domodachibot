# domodachibot notes

Notes for Pi camera streaming/recording and for **line-following** data + SteerNet training on the Mac.

---

## Line following: dataset layout and workflow

### Layout

- **`test_*_.../`** — one folder per capture session.
- **`frames_6hz/`** — JPEGs named `frame_0001.jpg`, … from `tools/extract_frames.py --fps 6` (line-following dataset used for training here).
- **`labels_6hz.jsonl`** — one JSON object per line: `image`, `target_px`, `theta_rad` (see `line_follow/angles.py`).
- **`labels_all_6hz.jsonl`** (repo root) — merged labels from all sessions (later inputs override the same `image` key).

Other frame rates are fine if you name folders consistently; **`extract_frames.py`** defaults to sparse **`--fps 0.2`** (one frame every 5 s) when you want fewer JPEGs.

### End-to-end workflow

1. **Extract frames** at **6 fps** (example):

   ```bash
   python3 tools/extract_frames.py path/to/video.mp4 -o test_x/frames_6hz --fps 6
   ```

2. **Label** (click target pixel; bottom-center steering convention). Use the project venv if `cv2` is missing; the flag is **`--frames-dir`**:

   ```bash
   .venv/bin/python3 tools/label_steering.py --frames-dir test_x/frames_6hz --labels test_x/labels_6hz.jsonl
   ```

3. **Optional: remove frames** (disk only). If you delete JPEGs, fix labels next.

   ```bash
   python3 tools/prune_frames.py --dir test_x/frames_6hz --max-index 407    # drop index > 407
   python3 tools/prune_frames.py --dir test_x/frames_6hz --drop-multiples-of 3  # drop 3,6,9,…
   python3 tools/prune_frames.py -n --dir test_x/frames_6hz …   # dry-run
   ```

4. **Drop label rows** for missing files (rewrites the JSONL in place):

   ```bash
   python3 tools/sync_labels_to_disk.py --labels test_x/labels_6hz.jsonl
   ```

5. **Merge** sessions into one pool. All clips are **mixed**; split happens at train time.

   ```bash
   python3 tools/merge_labels.py \
     test_1_0404_sunny/labels_6hz.jsonl \
     test_2_0404_sunny/labels_6hz.jsonl \
     test_3_0404_sunny/labels_6hz.jsonl \
     test_4_0404_sunny/labels_6hz.jsonl \
     -o labels_all_6hz.jsonl
   ```

6. **Train** SteerNet with a **random split** on the merged file: `--val-frac` / `--test-frac` (defaults **0.15** each), `--seed`, and **`--augment`** for train-only photometric jitter. Defaults: **40 epochs**, **lr 1e-4**, **ImageNet-normalized** inputs, **MobileNet V3 Small** (see below).

   ```bash
   .venv/bin/python3 -m line_follow.train_steer \
     --labels labels_all_6hz.jsonl \
     --split random \
     --val-frac 0.15 \
     --test-frac 0.15 \
     --seed 42 \
     --augment \
     --log-every 1 \
     --csv-log runs/steer_metrics_6hz.csv \
     --out line_follow/weights/steer.pt
   ```

   **Clip split (optional):** hold out whole sessions by path substring, e.g. `--split clip --clip-val test_2_0404_sunny --clip-test test_3_0404_sunny`.

7. **Eval / inference:** `python3 -m line_follow.eval …` (see `--help`), learned path `line_follow/learned_predict.py`, classical baseline `line_follow/predict.py`. For an MP4 with NN steering ray overlaid, use **`tools/nn_overlay_video.py`** (documented below). **Live on the Pi** (camera + motors + optional browser preview): **`mobot_nn_line_follower.py`** — see [Pi: live NN line follow with SteerNet](#pi-live-nn-line-follow-with-steernet).

### What `--augment` does (in training)

Implemented in `line_follow/label_dataset.py` when the train dataset is built with `augment=True`: HSV jitter, per-channel RGB gains, partial desaturation, Gaussian blur, additive Gaussian noise. **In-plane rotation was removed** so augmented views stay aligned with steering labels when using the ImageNet-pretrained backbone. **Labels are not changed.**

### SteerNet architecture and input preprocessing

- **Backbone:** `torchvision.models.mobilenet_v3_small` with **`weights="DEFAULT"`** (ImageNet). The default classifier is replaced by a single **`Linear` → 2 outputs** `(sin θ, cos θ)` after the built-in global average pool (`line_follow/steer_net.py`).
- **Normalization:** Training and inference apply **ImageNet mean/std** on RGB in **[0, 1]** (`line_follow/imagenet_norm.py`, used from `label_dataset.py` and `learned_predict.py`). Checkpoints trained before this change are not compatible.
- **Training defaults** (`line_follow/train_steer.py`): **`--epochs` 40**, **`--lr` 1e-4** (AdamW), cosine schedule; override as needed.
- **Device:** **`--device auto`** (default) uses **CUDA → Apple MPS → CPU** (`line_follow/torch_device.py`). Inference uses the same auto order when `device` is omitted.
- **Dependencies:** `torchvision` (backbone + `normalize`), `tqdm` (epoch bar). See `requirements.txt`.

On Apple Silicon you should see **`device: mps`** at startup (or **`cuda`** on NVIDIA). Training saves the weights from the epoch with the lowest **validation MSE** (`best_val_mse` in the log).

### Training run results (`labels_all_6hz.jsonl`, MobileNet V3 Small)

| Item | Value |
|------|--------|
| Merged labeled frames | **1442** (`labels_all_6hz.jsonl`) |
| Split | random 15% val / 15% test, `seed=42` |
| Epochs | **40** (~**8 min** on MPS for this run; ~**12 s/epoch** with `tqdm`) |
| Best **val MSE** (checkpoint) | **0.015597** (epoch ~35) |
| Final epoch val MAE | ~**3.75°** |
| Hold-out **test MSE** | **0.012512** |
| Hold-out **test MAE** | **3.49°** |
| Metrics log | `runs/steer_metrics_6hz.csv` |
| Weights | `line_follow/weights/steer.pt` (+ `.meta.json`) |

*Interpretation:* train MSE keeps falling while val wobbles late in the cosine schedule; the saved weights follow **best val MSE**, not the last epoch.

### NN overlay on a video (Mac)

Run the same checkpoint on every frame and write an MP4 (cyan ray + `theta` text from bottom center). Input can be full-res; internally the model still resizes to **`img_h` / `img_w` from the checkpoint meta** (default **120×160**).

```bash
.venv/bin/python3 tools/nn_overlay_video.py "path/to/video.mp4" -o "path/to/out_overlay.mp4"
```

Optional temporal smoothing (same idea as `line_follow.eval --smooth`): `--smooth 0.3`.

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

## Pi: live NN line follow with SteerNet

On the Raspberry Pi, **`mobot_nn_line_follower.py`** runs **Picamera2** (main stream **640×480 BGR888**), predicts heading with **`predict_steering_learned`** (`line_follow/learned_predict.py` — **BGR → RGB**, resize to checkpoint **`img_w`×`img_h`**, **ImageNet normalize**, same **`SteerNet`** as training), and drives the **same GPIO motor/servo stack** as **`Mobot/mobot_line_follower_headless.py`**.

**Weights:** copy **`line_follow/weights/steer.pt`** from the Mac after training (default path in the script). Checkpoint **meta** sets inference resize (typical **160×120**).

### Python venv on the Pi

Use a venv under the repo (e.g. **`~/Desktop/domodachibot/venv`**). You need **`torch`**, **`torchvision`**, **OpenCV**, **Picamera2**, **NumPy**, etc.

Install **`torch` and `torchvision` from the same PyTorch CPU wheel index** so the aarch64 `+cpu` builds match (a plain **`pip install torchvision`** from PyPI alone can error with missing custom ops, e.g. `torchvision::nms`).

```bash
cd ~/Desktop/domodachibot
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# plus repo deps: opencv-python, picamera2, … (see requirements / what the Mobot scripts need)
```

If **`pip`** later upgrades **NumPy** to **2.x** while an old **SciPy** is installed, you may see a resolver warning; fix by upgrading SciPy or pinning NumPy if something breaks.

### Run (repo root)

Stop anything else using the camera (**`rpicam-vid`**, another Python script, etc.) first.

**Dry run (no GPIO — still uses the real camera):**

```bash
cd ~/Desktop/domodachibot
source venv/bin/activate
PYTHONUNBUFFERED=1 python3 mobot_nn_line_follower.py --simulate
```

**Drive the robot** (only when safe — removes `--simulate`):

```bash
PYTHONUNBUFFERED=1 python3 mobot_nn_line_follower.py
```

**Useful flags:** `--weights path/to/steer.pt`, **`--steer-sign -1`** if it steers the wrong way, **`--theta-gain`** to scale servo deflection vs. predicted angle.

`PYTHONUNBUFFERED=1` makes **`theta` / servo** log lines show up immediately over SSH (line-buffered prints).

### Browser preview (MJPEG + steering ray)

To see **the same live frames** the NN uses, with a **heading ray** from the image bottom center (same convention as **`tools/nn_overlay_video.py`** / **`line_follow/angles.draw_ray`**) and on-screen **theta / servo / speed** text:

```bash
PYTHONUNBUFFERED=1 python3 mobot_nn_line_follower.py --simulate --preview-port 8080
```

- **On the Pi desktop:** open **`http://127.0.0.1:8080/`** in Chromium.
- **From a laptop over SSH:** in one terminal, `ssh -L 8080:127.0.0.1:8080 pi@PI_IP`, then on the Pi run the command above; on the laptop open **`http://127.0.0.1:8080/`**.

The server binds **`0.0.0.0`**. Default **`--preview-port 0`** disables the HTTP server. **You cannot** run this and **`rpicam-vid`** at the same time on one camera.

---

## Stream to Mac (low latency)

Use **raw H.264 over TCP** from the Pi and **ffplay on the Mac**. Avoid piping through VLC/RTSP if you care about delay—RTSP tends to add a lot of buffering.

### 1. On the Pi (SSH session)

Stop any old stream first, and **release the camera** from other apps (e.g. **`mobot_nn_line_follower.py`**, Jupyter, another `rpicam-*` process), then:

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




# run this 
```
PYTHONUNBUFFERED=1 python3 mobot_nn_line_follower.py --simulate --preview-port 8080
```