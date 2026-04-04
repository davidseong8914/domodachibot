"""
Mobot Line Follower — Integrated Pipeline
==========================================

Runs on Raspberry Pi. Combines:
1. Live camera feed (Picamera2 + Pi Camera 3 Wide NoIR)
2. CV line detection (adaptive threshold + scan bands)
3. PD steering controller
4. Motor control via GPIO (drive motor + steering servo)

Hardware:
- Raspberry Pi (replaces ESP8266)
- Pi Camera 3 Wide NoIR (SC1226, IMX708, 120° FOV)
- DC motor for rear-wheel drive (via L298N H-bridge)
- Servo for steering
- No reaction wheel, no IMU — robot has two rear wheels for stability

Dependencies:
    pip install opencv-python numpy
    # Picamera2 should be pre-installed on Raspberry Pi OS
    # RPi.GPIO should be pre-installed on Raspberry Pi OS

Usage:
    python mobot_line_follower.py              # live camera mode
    python mobot_line_follower.py --test_dir ./photos  # offline tuning mode (no motors)
"""

from __future__ import annotations

import argparse
import time
import glob
import os
from dataclasses import dataclass

import cv2
import numpy as np


# =============================================================================
# CV PARAMETERS (tuned from offline testing)
# =============================================================================

ROI_BOTTOM_FRACTION = 0.45
SIDE_MARGIN_FRACTION = 0.15

BLUR_KERNEL = 9
BLOCK_SIZE = 201
THRESH_C = -4
MORPH_KERNEL = 5

# Larger closing kernel to bridge shadow gaps on the white line.
# Shadow falling on the line creates a dark band that splits the detected
# white region. A large closing (dilate then erode) fills that gap back in.
# Increase if shadow gaps are wider; decrease if it merges line with nearby noise.
CLOSE_KERNEL_SIZE = 15

BAND_HEIGHT = 20
BAND_FRACTIONS = (0.70, 0.45)  # near band, far band within ROI

MIN_WHITE_PIXELS_PER_COL = 3
MIN_SEGMENT_WIDTH = 12
CENTER_BIAS_WEIGHT = 2.0
MAX_SEGMENT_GAP = 10


# =============================================================================
# CAMERA PARAMETERS (tuned from camera_test.py)
# =============================================================================

LENS_POSITION = 3.0   # TUNE THIS with camera_test.py


# =============================================================================
# PD CONTROLLER PARAMETERS
# =============================================================================

KP = 0.1       # proportional gain: degrees of servo per pixel of error
KD = 0.0       # derivative gain: degrees of servo per pixel/frame of error change

# Servo limits
SERVO_CENTER = 90       # servo position for straight ahead (degrees)
SERVO_MAX_OFFSET = 40   # max steering deflection from center (degrees)

# Speed control
NORMAL_SPEED = 70       # barely crawling
SLOW_SPEED = 50
ERROR_SLOW_THRESHOLD = 80  # if |error| > this, slow down

# Lost line behavior
MAX_LOST_FRAMES = 15    # hold last steering for this many frames when line lost
LOST_SPEED = 50         # crawl speed when line is lost


# =============================================================================
# GPIO PIN CONFIGURATION (adjust to match your wiring)
# =============================================================================

# Drive motor (L298N)
PIN_DRIVE_EN = 17   # PWM-capable pin for speed control
PIN_DRIVE_IN1 = 27  # direction pin 1
PIN_DRIVE_IN2 = 22  # direction pin 2

# Steering servo
PIN_SERVO = 16      # PWM-capable pin for servo


# =============================================================================
# CV PIPELINE
# =============================================================================

@dataclass
class BandDetection:
    y_start: int
    x_center: int | None
    segment: tuple[int, int] | None
    occupancy: np.ndarray


def preprocess(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Convert to grayscale and blur.

    Why grayscale: white line vs gray concrete differs in brightness, not color.
    Why blur: concrete has tiny texture noise (specks, pores, dirt). Blur
    averages each pixel with its 9x9 neighbors. Noise specks (1-3px) get
    diluted to background level. The white line (30+ px wide) survives because
    its center pixels are surrounded by other bright pixels — averaging bright
    with bright stays bright.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (BLUR_KERNEL, BLUR_KERNEL), 0)


def segment_line(blurred_gray: np.ndarray) -> np.ndarray:
    """
    Adaptive threshold + morphological cleanup -> binary mask.

    Each pixel is compared to its local 201x201 neighborhood average.
    Pixels brighter than (local_avg + THRESH_C) become white.
    Large blockSize ensures the neighborhood always includes concrete,
    preventing the center-dropout problem on thick lines.

    Shadow fix: A shadow falling across the white line creates a dark gap
    in the mask, splitting the line into two white regions. A large closing
    kernel (dilate then erode) bridges this gap by expanding white regions
    until they connect across the shadow, then shrinking them back to
    approximately original size. The small opening kernel still removes
    tiny noise specks first.
    """
    thresh = cv2.adaptiveThreshold(
        blurred_gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        BLOCK_SIZE, THRESH_C,
    )

    # Small kernel for opening: removes tiny noise blobs (concrete texture)
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (MORPH_KERNEL, MORPH_KERNEL)
    )

    # Large kernel for closing: bridges shadow gaps on the white line
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (CLOSE_KERNEL_SIZE, CLOSE_KERNEL_SIZE)
    )

    # Step 1: Opening removes small false detections
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, open_kernel)

    # Step 2: Large closing bridges shadow gaps within the line
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_kernel)

    return cleaned


def contiguous_segments(binary_1d: np.ndarray) -> list[tuple[int, int]]:
    """Find runs of True values in a 1D boolean array."""
    segments = []
    start = None
    for i, value in enumerate(binary_1d):
        if value and start is None:
            start = i
        elif not value and start is not None:
            segments.append((start, i - 1))
            start = None
    if start is not None:
        segments.append((start, len(binary_1d) - 1))
    return segments


def merge_close_segments(segments: list[tuple[int, int]], max_gap: int = MAX_SEGMENT_GAP) -> list[tuple[int, int]]:
    """Merge segments separated by a small gap (paint skip, crack)."""
    if not segments:
        return []
    merged = [segments[0]]
    for start, end in segments[1:]:
        last_start, last_end = merged[-1]
        if start - last_end - 1 <= max_gap:
            merged[-1] = (last_start, end)
        else:
            merged.append((start, end))
    return merged


def choose_segment(segments: list[tuple[int, int]], reference_x: int) -> tuple[int, int] | None:
    """Pick the widest segment closest to the expected x-position."""
    valid = [s for s in segments if (s[1] - s[0] + 1) >= MIN_SEGMENT_WIDTH]
    if not valid:
        return None
    best, best_score = None, -np.inf
    for start, end in valid:
        width = end - start + 1
        center = 0.5 * (start + end)
        score = width - CENTER_BIAS_WEIGHT * abs(center - reference_x)
        if score > best_score:
            best_score = score
            best = (start, end)
    return best


def find_line_in_band(mask: np.ndarray, y_start: int, band_height: int, reference_x: int) -> BandDetection:
    """
    In a thin horizontal strip, find the white line's x-center.

    Steps: extract strip -> count white pixels per column -> threshold ->
    find contiguous runs -> merge close runs -> pick best segment ->
    weighted centroid within that segment.
    """
    strip = mask[y_start:y_start + band_height, :]
    col_counts = np.sum(strip > 0, axis=0)
    occupancy = col_counts >= MIN_WHITE_PIXELS_PER_COL

    segments = contiguous_segments(occupancy)
    segments = merge_close_segments(segments, MAX_SEGMENT_GAP)
    chosen = choose_segment(segments, reference_x)

    if chosen is None:
        return BandDetection(y_start=y_start, x_center=None, segment=None, occupancy=col_counts)

    start, end = chosen
    xs = np.arange(start, end + 1)
    weights = col_counts[start:end + 1].astype(np.float32)
    if np.sum(weights) > 0:
        x_center = int(np.round(np.sum(xs * weights) / np.sum(weights)))
    else:
        x_center = (start + end) // 2

    return BandDetection(y_start=y_start, x_center=x_center, segment=chosen, occupancy=col_counts)


def process_frame(frame_bgr: np.ndarray) -> dict:
    """Full CV pipeline: ROI crop -> threshold -> scan bands -> error."""
    h, w = frame_bgr.shape[:2]

    roi_y0 = int(h * (1.0 - ROI_BOTTOM_FRACTION))
    side_margin = int(w * SIDE_MARGIN_FRACTION)
    roi_x0, roi_x1 = side_margin, w - side_margin

    roi = frame_bgr[roi_y0:, roi_x0:roi_x1]
    roi_h, roi_w = roi.shape[:2]

    blurred = preprocess(roi)
    mask = segment_line(blurred)

    band_y_starts = [min(max(int(roi_h * frac), 0), roi_h - BAND_HEIGHT) for frac in BAND_FRACTIONS]
    image_center_x = roi_w // 2
    detections = []

    reference_x = image_center_x
    for i, y_start in enumerate(band_y_starts):
        det = find_line_in_band(mask, y_start, BAND_HEIGHT, reference_x)
        detections.append(det)
        if i == 0 and det.x_center is not None:
            reference_x = det.x_center

    near = detections[0]
    far = detections[1] if len(detections) > 1 else None

    error = None if near.x_center is None else near.x_center - image_center_x
    heading_trend = None
    if far and near.x_center is not None and far.x_center is not None:
        heading_trend = far.x_center - near.x_center

    return {
        "roi": roi, "mask": mask, "detections": detections,
        "image_center_x": image_center_x,
        "error": error, "heading_trend": heading_trend,
    }


# =============================================================================
# PD CONTROLLER
# =============================================================================

class LineFollowPD:
    """
    PD controller that converts pixel error into servo angle.

    error > 0 means line is to the RIGHT of image center -> steer right
    error < 0 means line is to the LEFT of image center -> steer left

    The derivative term is computed as the change in error between frames,
    which damps oscillation on curves.
    """

    def __init__(self, kp: float, kd: float):
        self.kp = kp
        self.kd = kd
        self.prev_error = 0.0
        self.last_valid_steering = 0.0
        self.lost_count = 0

    def compute(self, error: float | None) -> tuple[float, float]:
        """
        Args:
            error: pixel offset from center, or None if line lost

        Returns:
            (steering_angle, speed)
            steering_angle: degrees offset from center
            speed: motor PWM value (0-255)
        """
        if error is None:
            self.lost_count += 1
            if self.lost_count > MAX_LOST_FRAMES:
                return self.last_valid_steering, 0
            return self.last_valid_steering, LOST_SPEED

        self.lost_count = 0

        d_error = error - self.prev_error
        self.prev_error = error

        steering = self.kp * error + self.kd * d_error
        steering = max(-SERVO_MAX_OFFSET, min(SERVO_MAX_OFFSET, steering))
        self.last_valid_steering = steering

        if abs(error) > ERROR_SLOW_THRESHOLD:
            speed = SLOW_SPEED
        else:
            speed = NORMAL_SPEED

        return steering, speed


# =============================================================================
# MOTOR CONTROL (Raspberry Pi GPIO)
# =============================================================================

class MotorController:
    """
    Controls drive motor (L298N) and steering servo via RPi GPIO.

    For offline testing (no Pi hardware), set simulate=True and it
    just prints commands instead of driving GPIO.
    """

    def __init__(self, simulate: bool = False):
        self.simulate = simulate

        if not simulate:
            import RPi.GPIO as GPIO
            self.GPIO = GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)

            # Drive motor
            GPIO.setup(PIN_DRIVE_EN, GPIO.OUT)
            GPIO.setup(PIN_DRIVE_IN1, GPIO.OUT)
            GPIO.setup(PIN_DRIVE_IN2, GPIO.OUT)
            self.drive_pwm = GPIO.PWM(PIN_DRIVE_EN, 1000)  # 1kHz PWM
            self.drive_pwm.start(0)

            # Servo
            GPIO.setup(PIN_SERVO, GPIO.OUT)
            self.servo_pwm = GPIO.PWM(PIN_SERVO, 50)  # 50Hz for servo
            self.servo_pwm.start(0)

    def set_steering(self, angle_deg: float):
        """
        Set servo angle. 0 = straight ahead.
        Positive = right, negative = left.
        """
        servo_pos = SERVO_CENTER + angle_deg
        servo_pos = max(SERVO_CENTER - SERVO_MAX_OFFSET,
                        min(SERVO_CENTER + SERVO_MAX_OFFSET, servo_pos))

        if self.simulate:
            return

        # Convert angle to duty cycle (typical servo: 2-12% duty at 50Hz)
        duty = 2.0 + (servo_pos / 180.0) * 10.0
        self.servo_pwm.ChangeDutyCycle(duty)

    def set_drive_speed(self, speed: int):
        """
        Set drive motor speed.
        speed > 0: forward, speed == 0: stop, speed < 0: reverse
        """
        if self.simulate:
            return

        speed = max(-255, min(255, speed))

        if speed == 0:
            self.GPIO.output(PIN_DRIVE_IN1, self.GPIO.LOW)
            self.GPIO.output(PIN_DRIVE_IN2, self.GPIO.LOW)
            self.drive_pwm.ChangeDutyCycle(0)
        elif speed > 0:
            self.GPIO.output(PIN_DRIVE_IN1, self.GPIO.HIGH)
            self.GPIO.output(PIN_DRIVE_IN2, self.GPIO.LOW)
            self.drive_pwm.ChangeDutyCycle(speed / 255.0 * 100.0)
        else:
            self.GPIO.output(PIN_DRIVE_IN1, self.GPIO.LOW)
            self.GPIO.output(PIN_DRIVE_IN2, self.GPIO.HIGH)
            self.drive_pwm.ChangeDutyCycle(abs(speed) / 255.0 * 100.0)

    def stop(self):
        """Emergency stop: kill drive and center steering."""
        self.set_drive_speed(0)
        self.set_steering(0)

    def cleanup(self):
        """Release GPIO resources."""
        if not self.simulate:
            self.drive_pwm.stop()
            self.servo_pwm.stop()
            self.GPIO.cleanup()


# =============================================================================
# DEBUG VISUALIZATION
# =============================================================================

def visualize(results: dict, steering: float, speed: int) -> np.ndarray:
    """Draw debug overlay — same as tuning script plus PD output info."""
    roi = results["roi"].copy()
    mask_color = cv2.cvtColor(results["mask"], cv2.COLOR_GRAY2BGR)
    h, w = roi.shape[:2]
    cx = results["image_center_x"]

    cv2.line(roi, (cx, 0), (cx, h), (0, 255, 0), 2)
    cv2.line(mask_color, (cx, 0), (cx, h), (0, 255, 0), 2)

    band_colors = [(255, 140, 0), (0, 140, 255)]
    band_names = ["NEAR", "FAR"]

    for i, det in enumerate(results["detections"]):
        color = band_colors[i % len(band_colors)]
        label = band_names[i] if i < len(band_names) else f"B{i}"
        y0 = det.y_start

        cv2.rectangle(roi, (0, y0), (w, y0 + BAND_HEIGHT), color, 2)
        cv2.rectangle(mask_color, (0, y0), (w, y0 + BAND_HEIGHT), color, 2)

        if det.segment is not None:
            sx, ex = det.segment
            cy = y0 + BAND_HEIGHT // 2
            cv2.line(roi, (sx, cy), (ex, cy), (255, 0, 255), 2)

        if det.x_center is not None:
            pt = (det.x_center, y0 + BAND_HEIGHT // 2)
            cv2.circle(roi, pt, 7, (0, 0, 255), -1)
            cv2.circle(mask_color, pt, 7, (0, 0, 255), -1)

    if results["error"] is not None:
        info = f"Err:{results['error']:+d}px  Steer:{steering:+.1f}deg  Spd:{speed}"
    else:
        info = "LINE LOST"
    cv2.putText(roi, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return np.hstack([roi, mask_color])


# =============================================================================
# MAIN: LIVE CAMERA MODE
# =============================================================================

def run_live():
    """
    Main loop: camera -> CV -> PD -> motors.
    This replaces the entire ESP8266 firmware. Everything runs on the Pi.
    """
    from picamera2 import Picamera2
    from libcamera import controls as libcam_controls

    # --- Camera setup ---
    print("Initializing camera...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "BGR888"}
    )
    picam2.configure(config)
    picam2.start()

    # Step 1: Let auto-exposure and auto-WB settle
    print("Waiting for auto-exposure to settle...")
    time.sleep(2)

    # Step 2: Read what the camera auto-selected
    metadata = picam2.capture_metadata()
    auto_exposure = metadata["ExposureTime"]
    auto_gain = metadata["AnalogueGain"]
    auto_wb = metadata.get("ColourGains", (1.0, 1.0))
    print(f"Auto settings — Exposure: {auto_exposure}us, "
          f"Gain: {auto_gain:.2f}, WB: {auto_wb}")

    # Step 3: Lock everything so it doesn't drift during the run
    picam2.set_controls({
        # Lock focus
        "AfMode": libcam_controls.AfModeEnum.Manual,
        "LensPosition": LENS_POSITION,
        # Lock exposure and gain
        "AeEnable": False,
        "ExposureTime": auto_exposure,
        "AnalogueGain": auto_gain,
        # Lock white balance
        "AwbEnable": False,
        "ColourGains": auto_wb,
    })
    time.sleep(0.5)
    print(f"Camera locked — LensPosition: {LENS_POSITION}")

    # --- Controller + motors ---
    pd = LineFollowPD(kp=KP, kd=KD)
    motors = MotorController(simulate=True)

    print("\nLine follower running. Press Ctrl+C to stop.")
    print(f"PD gains: Kp={KP}, Kd={KD}")
    print(f"Speed: normal={NORMAL_SPEED}, slow={SLOW_SPEED}")

    frame_count = 0

    try:
        while True:
            # 1. Grab frame from camera
            frame = picam2.capture_array("main")

            # 2. Run CV pipeline -> get error in pixels
            results = process_frame(frame)

            # 3. PD controller -> steering angle + speed
            steering, speed = pd.compute(results["error"])

            # 4. Send commands to hardware
            motors.set_steering(steering)
            motors.set_drive_speed(int(speed))

            # 5. Print debug info every 10th frame to avoid flooding SSH
            frame_count += 1
            if frame_count % 10 == 0:
                if results["error"] is not None:
                    print(f"Err:{results['error']:+d}  Steer:{steering:+.1f}  Spd:{int(speed)}")
                else:
                    print("LINE LOST")

            # 6. Limit to ~30fps so the Pi isn't running at 100% CPU
            time.sleep(0.03)

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        motors.stop()
        motors.cleanup()
        picam2.stop()
        cv2.destroyAllWindows()


# =============================================================================
# MAIN: OFFLINE TEST MODE (for tuning on laptop, no motors)
# =============================================================================

def run_test(image_dir: str):
    """Browse photos and see what the PD controller would output. No motors."""
    image_paths = sorted(
        glob.glob(os.path.join(image_dir, "*.jpg"))
        + glob.glob(os.path.join(image_dir, "*.png"))
        + glob.glob(os.path.join(image_dir, "*.jpeg"))
    )
    if not image_paths:
        print(f"No images found in {image_dir}")
        return

    print(f"Found {len(image_paths)} images. Arrow keys to navigate, q to quit.")

    pd = LineFollowPD(kp=KP, kd=KD)
    idx = 0

    while True:
        img = cv2.imread(image_paths[idx])
        if img is None:
            idx = (idx + 1) % len(image_paths)
            continue

        img_small = cv2.resize(img, (640, 480))
        results = process_frame(img_small)
        steering, speed = pd.compute(results["error"])

        vis = visualize(results, steering, int(speed))
        footer = f"[{idx+1}/{len(image_paths)}] {os.path.basename(image_paths[idx])}"
        cv2.putText(vis, footer, (10, vis.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        cv2.imshow("Mobot Test", vis)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            break
        elif key in (83, ord('d')):
            idx = (idx + 1) % len(image_paths)
        elif key in (81, ord('a')):
            idx = (idx - 1) % len(image_paths)

    cv2.destroyAllWindows()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mobot line follower")
    parser.add_argument("--test_dir", type=str, default=None,
                        help="Image directory for offline testing (no motors)")
    args = parser.parse_args()

    if args.test_dir:
        run_test(args.test_dir)
    else:
        run_live()