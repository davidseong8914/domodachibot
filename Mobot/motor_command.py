#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L298N DC Motor (forward only) + Servo sinusoidal sweep ? Raspberry Pi 4

Motor pins:
  ENA -> GPIO 17 (PWM speed control)
  IN1 -> GPIO 27 (direction)
  IN2 -> GPIO 22 (direction)

Servo pin:
  SIG -> GPIO 16 (50 Hz PWM)
"""

import RPi.GPIO as GPIO
import time
import math

# -- Pin definitions --
EN    = 17
IN1   = 27
IN2   = 22
SERVO = 16

# -- Settings --
MOTOR_SPEED     = 75        # Constant duty cycle 0-100 %
SINE_PERIOD     = 2.0       # Servo sweep period (seconds)
SERVO_MIN_ANGLE = 0         # degrees
SERVO_MAX_ANGLE = 180       # degrees
DT              = 0.02      # Loop timestep (seconds)
RUN_TIME        = 10        # Total run time (seconds)


def angle_to_duty(angle):
    """
    Convert angle (0-180) to duty cycle for a standard servo.
    0   -> 2.5%  (0.5 ms pulse)
    180 -> 12.5% (2.5 ms pulse)
    """
    return 2.5 + (angle / 180.0) * 10.0


# -- Setup --
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

GPIO.setup(EN,    GPIO.OUT)
GPIO.setup(IN1,   GPIO.OUT)
GPIO.setup(IN2,   GPIO.OUT)
GPIO.setup(SERVO, GPIO.OUT)

motor_pwm = GPIO.PWM(EN, 1000)       # 1 kHz for motor
servo_pwm = GPIO.PWM(SERVO, 50)      # 50 Hz for servo

motor_pwm.start(0)
servo_pwm.start(angle_to_duty(90))   # Start at midpoint

# -- Drive motor forward --
GPIO.output(IN1, GPIO.HIGH)
GPIO.output(IN2, GPIO.LOW)
motor_pwm.ChangeDutyCycle(MOTOR_SPEED)
print("Motor -> FORWARD at {}%".format(MOTOR_SPEED))

# -- Main loop: sinusoidal servo sweep --
try:
    t0 = time.time()
    while True:
        t = time.time() - t0
        if t >= RUN_TIME:
            break

        mid   = (SERVO_MAX_ANGLE + SERVO_MIN_ANGLE) / 2.0
        amp   = (SERVO_MAX_ANGLE - SERVO_MIN_ANGLE) / 2.0
        angle = mid + amp * math.sin(2.0 * math.pi * t / SINE_PERIOD)

        servo_pwm.ChangeDutyCycle(angle_to_duty(angle))

        print("  t={:.2f}s | servo={:.1f} deg".format(t, angle), end="\r")
        time.sleep(DT)

    print()

except KeyboardInterrupt:
    print("\nInterrupted by user")

finally:
    # Stop motor
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    motor_pwm.ChangeDutyCycle(0)
    print("Motor -> STOPPED")

    # Center servo, then stop
    servo_pwm.ChangeDutyCycle(angle_to_duty(90))
    time.sleep(0.5)
    servo_pwm.ChangeDutyCycle(0)
    print("Servo -> centered & released")

    motor_pwm.stop()
    servo_pwm.stop()
    GPIO.cleanup()
    print("GPIO cleaned up")
