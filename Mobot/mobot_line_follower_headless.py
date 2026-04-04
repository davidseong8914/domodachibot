"""
Mobot Line Follower — Headless Version (no display required)
=============================================================

Same as mobot_line_follower.py but with visualization disabled.
Run this when no monitor/VNC/X11 is available.

Usage:
    python mobot_line_follower_headless.py
    # Press Ctrl+C to stop
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
CLOSE_KERNEL_SIZE = 15

BAND_HEIGHT = 20
BAND_FRACTIONS = (0.70, 0.45)

MIN_WHITE_PIXELS_PER_COL = 3
MIN_SEGMENT_WIDTH = 12
CENTER_BIAS_WEIGHT = 2.0
MAX_SEGMENT_GAP = 10


# =============================================================================
# CAMERA PARAMETERS
# =============================================================================

LENS_POSITION = 3.0


# =============================================================================
# PD CONTROLLER PARAMETERS
# =============================================================================

KP = 0.2
KD = 0.0

SERVO_CENTER = 90
SERVO_MAX_OFFSET = 40

NORMAL_SPEED = 150
SLOW_SPEED = 100
ERROR_SLOW_THRESHOLD = 80

MAX_LOST_FRAMES = 15
LOST_SPEED = 50


# =============================================================================
# GPIO PIN CONFIGURATION
# =============================================================================

PIN_DRIVE_EN = 17
PIN_DRIVE_IN1 = 27
PIN_DRIVE_IN2 = 22
PIN_SERVO = 16


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
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (BLUR_KERNEL, BLUR_KERNEL), 0)


def segment_line(blurred_gray: np.ndarray) -> np.ndarray:
    thresh = cv2.adaptiveThreshold(
        blurred_gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        BLOCK_SIZE, THRESH_C,
    )
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (MORPH_KERNEL, MORPH_KERNEL)
    )
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (CLOSE_KERNEL_SIZE, CLOSE_KERNEL_SIZE)
    )
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, open_kernel)
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_kernel)
    return cleaned


def contiguous_segments(binary_1d: np.ndarray) -> list[tuple[int, int]]:
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
        "error": error, "heading_trend": heading_trend,
    }


# =============================================================================
# PD CONTROLLER
# =============================================================================

class LineFollowPD:
    def __init__(self, kp: float, kd: float):
        self.kp = kp
        self.kd = kd
        self.prev_error = 0.0
        self.last_valid_steering = 0.0
        self.lost_count = 0

    def compute(self, error: float | None) -> tuple[float, float]:
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
# MOTOR CONTROL
# =============================================================================

class MotorController:
    def __init__(self, simulate: bool = False):
        self.simulate = simulate

        if not simulate:
            import RPi.GPIO as GPIO
            self.GPIO = GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)

            GPIO.setup(PIN_DRIVE_EN, GPIO.OUT)
            GPIO.setup(PIN_DRIVE_IN1, GPIO.OUT)
            GPIO.setup(PIN_DRIVE_IN2, GPIO.OUT)
            self.drive_pwm = GPIO.PWM(PIN_DRIVE_EN, 1000)
            self.drive_pwm.start(0)

            GPIO.setup(PIN_SERVO, GPIO.OUT)
            self.servo_pwm = GPIO.PWM(PIN_SERVO, 50)
            self.servo_pwm.start(0)

    def set_steering(self, angle_deg: float):
        servo_pos = SERVO_CENTER + angle_deg
        servo_pos = max(SERVO_CENTER - SERVO_MAX_OFFSET,
                        min(SERVO_CENTER + SERVO_MAX_OFFSET, servo_pos))
        if self.simulate:
            return
        duty = 2.0 + (servo_pos / 180.0) * 10.0
        self.servo_pwm.ChangeDutyCycle(duty)

    def set_drive_speed(self, speed: int):
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
        self.set_drive_speed(0)
        self.set_steering(0)

    def cleanup(self):
        if not self.simulate:
            self.drive_pwm.stop()
            self.servo_pwm.stop()
            self.GPIO.cleanup()


# =============================================================================
# MAIN: HEADLESS LIVE MODE
# =============================================================================

def run_live():
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

    metadata = picam2.capture_metadata()
    auto_exposure = metadata["ExposureTime"]
    auto_gain = metadata["AnalogueGain"]
    auto_wb = metadata.get("ColourGains", (1.0, 1.0))
    print(f"Auto settings — Exposure: {auto_exposure}us, "
          f"Gain: {auto_gain:.2f}, WB: {auto_wb}")

    picam2.set_controls({
    "AfMode": libcam_controls.AfModeEnum.Manual,
    "LensPosition": LENS_POSITION,
    # Leave AeEnable and AwbEnable at default (auto-on)
    })
    time.sleep(0.5)
    print(f"Camera locked — LensPosition: {LENS_POSITION}")

    pd = LineFollowPD(kp=KP, kd=KD)
    motors = MotorController(simulate=False)

    print("\nLine follower running. Press Ctrl+C to stop.")
    print(f"PD gains: Kp={KP}, Kd={KD}")
    print(f"Speed: normal={NORMAL_SPEED}, slow={SLOW_SPEED}")

    frame_count = 0

    try:
        while True:
            frame = picam2.capture_array("main")
            results = process_frame(frame)
            steering, speed = pd.compute(results["error"])

            motors.set_steering(steering)
            motors.set_drive_speed(int(speed))

            # Print status every 30 frames (~1 per second)
            frame_count += 1
            if frame_count % 30 == 0:
                err = results["error"]
                err_str = f"{err:+d}px" if err is not None else "LOST"
                print(f"[{frame_count}] Err:{err_str}  Steer:{steering:+.1f}deg  Spd:{int(speed)}")

            time.sleep(0.03)

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        motors.stop()
        motors.cleanup()
        picam2.stop()
        print("Shutdown complete.")


if __name__ == "__main__":
    run_live()