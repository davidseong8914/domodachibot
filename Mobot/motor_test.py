#!/usr/bin/env python3
"""
L298N DC Motor Driver ? Raspberry Pi 4
Pins:
  ENA ? GPIO 17 (PWM speed control)
  IN1 ? GPIO 27 (direction)
  IN2 ? GPIO 22 (direction)
"""

import RPi.GPIO as GPIO
import time

# ?? Pin definitions ??
EN = 17
IN1 = 27
IN2 = 22

# ?? Settings ??
SPEED = 75  # Constant duty cycle 0-100 %

# ?? Setup ??
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

GPIO.setup(EN, GPIO.OUT)
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)

pwm = GPIO.PWM(EN, 1000)  # 1 kHz PWM frequency
pwm.start(0)


def forward(speed: int = SPEED):
    """Spin motor forward at constant speed."""
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    pwm.ChangeDutyCycle(speed)
    print(f"Motor ? FORWARD at {speed}% duty cycle")


def backward(speed: int = SPEED):
    """Spin motor backward at constant speed."""
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    pwm.ChangeDutyCycle(speed)
    print(f"Motor ? BACKWARD at {speed}% duty cycle")


def stop():
    """Stop the motor."""
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    pwm.ChangeDutyCycle(0)
    print("Motor ? STOPPED")


# ?? Main ??
if __name__ == "__main__":
    try:
        forward(SPEED)
        time.sleep(5)       # run forward 5 seconds

        stop()
        time.sleep(1)

        backward(SPEED)
        time.sleep(5)       # run backward 5 seconds

        stop()

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        pwm.stop()
        GPIO.cleanup()
        print("GPIO cleaned up")
