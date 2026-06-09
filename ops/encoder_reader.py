#!/usr/bin/env python3
"""
Bench-test payload quadrature encoders (no ROS).

Jetson BOARD pin numbers (40-pin header names in comments):
  Pitch (ENC1): A=33 GPIO13, B=32 GPIO07
  Roll  (ENC2): A=29 GPIO01, B=31 GPIO11

Run: python3 encoder_reader.py
Stop encoder_node first: pkill -f encoder_node
"""

import os

os.environ.setdefault('JETSON_MODEL_NAME', 'JETSON_ORIN_NANO')

import Jetson.GPIO as GPIO
import time

# Pitch = ENC1, Roll = ENC2 (BOARD mode, not BCM)
ENC1_A = 33   # Pitch A — GPIO13
ENC1_B = 32   # Pitch B — GPIO07
ENC2_A = 29   # Roll A  — GPIO01
ENC2_B = 31   # Roll B  — GPIO11

pitch_count = 0
roll_count = 0
last_pitch = 0
last_roll = 0

QUAD_TABLE = {
    (0b00, 0b01): +1,
    (0b01, 0b11): +1,
    (0b11, 0b10): +1,
    (0b10, 0b00): +1,
    (0b00, 0b10): -1,
    (0b10, 0b11): -1,
    (0b11, 0b01): -1,
    (0b01, 0b00): -1,
}


def _step(last_state, a, b):
    state = (a << 1) | b
    if state == last_state:
        return last_state, 0
    delta = QUAD_TABLE.get((last_state, state), 0)
    return state, delta


def pitch_callback(_channel):
    global pitch_count, last_pitch
    a = GPIO.input(ENC1_A)
    b = GPIO.input(ENC1_B)
    last_pitch, delta = _step(last_pitch, a, b)
    pitch_count += delta


def roll_callback(_channel):
    global roll_count, last_roll
    a = GPIO.input(ENC2_A)
    b = GPIO.input(ENC2_B)
    last_roll, delta = _step(last_roll, a, b)
    roll_count += delta


GPIO.setwarnings(False)
try:
    GPIO.cleanup()
except Exception:
    pass
GPIO.setmode(GPIO.BOARD)

for pin in (ENC1_A, ENC1_B, ENC2_A, ENC2_B):
    GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

last_pitch = (GPIO.input(ENC1_A) << 1) | GPIO.input(ENC1_B)
last_roll = (GPIO.input(ENC2_A) << 1) | GPIO.input(ENC2_B)

for pin in (ENC1_A, ENC1_B, ENC2_A, ENC2_B):
    cb = pitch_callback if pin in (ENC1_A, ENC1_B) else roll_callback
    GPIO.add_event_detect(pin, GPIO.BOTH, callback=cb, bouncetime=1)

print(
    'Encoder reader (BOARD) — pitch 33/32 roll 29/31 — Ctrl+C to stop',
    flush=True,
)

try:
    while True:
        print(f'Pitch: {pitch_count:+6d} | Roll: {roll_count:+6d}', end='\r')
        time.sleep(0.05)
except KeyboardInterrupt:
    print('\nStopping...')
finally:
    GPIO.cleanup()
