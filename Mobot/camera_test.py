"""
Camera Test & Tuning Script
============================

Run this FIRST before the full line follower.
Purpose:
  1. Verify camera is connected and working
  2. Find the right LensPosition for your mount height
  3. Preview what the camera sees and save test frames
  4. Lock exposure/gain/white balance and verify stable output

Usage:
    python camera_test.py                    # live preview
    python camera_test.py --save_frames 10   # save 10 frames to disk

Controls (in live preview):
    q          : quit
    s          : save current frame as test_frame.jpg
    UP/DOWN    : increase/decrease LensPosition by 0.5
    LEFT/RIGHT : decrease/increase exposure time by 1000us
"""

import time
import argparse
from picamera2 import Picamera2
from libcamera import controls
import cv2
import numpy as np

from release_pi_camera_pipeline import release_pi_camera_pipeline


def main():
    parser = argparse.ArgumentParser(description="Camera test and tuning")
    parser.add_argument("--save_frames", type=int, default=0,
                        help="Save N frames to disk and exit (headless mode)")
    parser.add_argument(
        "--no-pipeline-release",
        action="store_true",
        help="Do not kill other /dev/video* users or stop PipeWire before opening the camera",
    )
    args = parser.parse_args()

    if not args.no_pipeline_release:
        print("Releasing camera pipeline (prior users + PipeWire / rpicam)…", flush=True)
        release_pi_camera_pipeline()

    # --- Initialize camera ---
    print("Initializing camera...")
    picam2 = Picamera2()

    config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "BGR888"}
    )
    picam2.configure(config)
    picam2.start()

    # Let auto-exposure and auto-focus settle
    print("Waiting for auto-exposure to settle...")
    time.sleep(2)

    # Read what auto-exposure chose
    metadata = picam2.capture_metadata()
    print(f"\nAuto-selected settings:")
    print(f"  ExposureTime:  {metadata.get('ExposureTime', '?')} us")
    print(f"  AnalogueGain:  {metadata.get('AnalogueGain', '?')}")
    print(f"  ColourGains:   {metadata.get('ColourGains', '?')}")
    print(f"  LensPosition:  {metadata.get('LensPosition', '?')}")
    print(f"  Lux:           {metadata.get('Lux', '?')}")

    # --- Headless mode: just save frames and exit ---
    if args.save_frames > 0:
        print(f"\nSaving {args.save_frames} frames...")
        for i in range(args.save_frames):
            frame = picam2.capture_array("main")
            filename = f"test_frame_{i:03d}.jpg"
            cv2.imwrite(filename, frame)
            print(f"  Saved {filename} ({frame.shape})")
            time.sleep(0.2)
        picam2.stop()
        print("Done. Transfer these to your laptop for inspection.")
        return

    # --- Interactive mode: live preview with tuning ---
    print("\nStarting live preview...")
    print("Controls:")
    print("  q           : quit")
    print("  s           : save current frame")
    print("  UP/DOWN     : adjust LensPosition (+/- 0.5)")
    print("  LEFT/RIGHT  : adjust ExposureTime (+/- 1000us)")
    print("  l           : lock exposure/gain/white balance at current values")
    print("  u           : unlock (re-enable auto exposure)")
    print()

    # Start with manual focus at a reasonable default
    lens_pos = 3.0  # ~33cm focal distance
    picam2.set_controls({
        "AfMode": controls.AfModeEnum.Manual,
        "LensPosition": lens_pos,
    })

    exposure_locked = False
    locked_exposure = None
    locked_gain = None
    locked_wb = None
    frame_count = 0

    try:
        while True:
            frame = picam2.capture_array("main")
            display = frame.copy()

            # Get current metadata
            metadata = picam2.capture_metadata()
            current_exposure = metadata.get("ExposureTime", 0)
            current_gain = metadata.get("AnalogueGain", 0)
            current_lens = metadata.get("LensPosition", 0)

            # Draw info overlay
            info_lines = [
                f"LensPos: {lens_pos:.1f} (actual: {current_lens:.2f})",
                f"Exposure: {current_exposure}us  Gain: {current_gain:.2f}",
                f"Locked: {'YES' if exposure_locked else 'NO'}",
                f"Frame: {frame_count}",
            ]
            for i, line in enumerate(info_lines):
                cv2.putText(display, line, (10, 25 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            # Draw crosshair at center
            h, w = display.shape[:2]
            cv2.line(display, (w // 2, 0), (w // 2, h), (0, 255, 0), 1)
            cv2.line(display, (0, h // 2), (w, h // 2), (0, 255, 0), 1)

            cv2.imshow("Camera Test", display)
            key = cv2.waitKey(1) & 0xFF
            frame_count += 1

            if key == ord('q'):
                break

            elif key == ord('s'):
                filename = f"test_frame_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Saved {filename}")

            elif key == 82 or key == ord('w'):  # UP arrow or 'w'
                lens_pos = min(lens_pos + 0.5, 15.0)
                picam2.set_controls({
                    "AfMode": controls.AfModeEnum.Manual,
                    "LensPosition": lens_pos,
                })
                print(f"LensPosition = {lens_pos:.1f}")

            elif key == 84 or key == ord('x'):  # DOWN arrow or 'x'
                lens_pos = max(lens_pos - 0.5, 0.0)
                picam2.set_controls({
                    "AfMode": controls.AfModeEnum.Manual,
                    "LensPosition": lens_pos,
                })
                print(f"LensPosition = {lens_pos:.1f}")

            elif key == ord('l'):
                # Lock exposure, gain, and white balance at current values
                metadata = picam2.capture_metadata()
                locked_exposure = metadata.get("ExposureTime", 10000)
                locked_gain = metadata.get("AnalogueGain", 1.0)
                locked_wb = metadata.get("ColourGains", (1.0, 1.0))

                picam2.set_controls({
                    "AeEnable": False,
                    "ExposureTime": locked_exposure,
                    "AnalogueGain": locked_gain,
                    "AwbEnable": False,
                    "ColourGains": locked_wb,
                })
                exposure_locked = True
                print(f"LOCKED — Exposure: {locked_exposure}us, "
                      f"Gain: {locked_gain:.2f}, WB: {locked_wb}")

            elif key == ord('u'):
                # Unlock — re-enable auto exposure and white balance
                picam2.set_controls({
                    "AeEnable": True,
                    "AwbEnable": True,
                })
                exposure_locked = False
                print("UNLOCKED — auto exposure/WB re-enabled")

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        picam2.stop()
        cv2.destroyAllWindows()

    print(f"\nFinal settings to copy into mobot_line_follower.py:")
    print(f"  LensPosition = {lens_pos:.1f}")
    if exposure_locked:
        print(f"  ExposureTime = {locked_exposure}")
        print(f"  AnalogueGain = {locked_gain:.2f}")
        print(f"  ColourGains  = {locked_wb}")


if __name__ == "__main__":
    main()