# Precision-stop software experiment

## Question

Can a measured-state stop controller replace the two-second same-direction ZVD
tail with a fast brake and small local correction around the position selected
by the operator?

## Test setup

- State: cart position, cart velocity, payload angle, payload angle rate.
- Input: cart velocity command through a 45 ms first-order actuator model.
- Nominal payload mode: 0.5 Hz with damping ratio 0.01.
- Stop target: cart position at the instant the stick reaches neutral.
- Controller: full-state LQR with command-speed and command-acceleration limits.
- Settle condition held for 0.5 s: position within 8 mm, velocity within
  3 mm/s, angle within 0.18 deg, and angle rate within 0.50 deg/s.

This is a linear software model, not a hardware validation.

## Nominal result at 60 mm/s and 150 mm/s^2

| Strategy | Peak forward travel | Near-zero velocity | Final position error | Residual angle RMS |
|---|---:|---:|---:|---:|
| Current ZVD tail | 62.20 mm | 2.05 s | 62.20 mm | 0.0049 deg |
| Immediate hard stop | 2.70 mm | 0.11 s | 2.70 mm | 0.6093 deg |
| Precision stop | 15.21 mm | 0.40 s | 0.00 mm | 0.0019 deg |

The precision stop meets all settle thresholds after 3.29 s. Unlike the ZVD
tail, most of that time is small corrective motion around the captured target,
not continued travel in the operator's previous direction.

## Speed and model-error checks

At 30, 60, and 100 mm/s with a 150 mm/s^2 command-acceleration limit, peak
precision-stop travel was 6.00, 15.21, and 37.18 mm respectively. The 100 mm/s
case exceeds the proposed 35 mm correction envelope. Raising the acceleration
limit to 200 mm/s^2 reduces that case to 28.80 mm.

With the controller fixed at 0.5 Hz and the simulated plant at 0.45 or 0.55 Hz,
the 60 mm/s stop remained below 17 mm peak travel, below 0.13 mm final error,
and below 0.012 deg residual angle RMS. A separate test starts with 1 deg of
payload angle and verifies recovery inside the 35 mm envelope.

## Decision and next gate

The concept passes the software gate but remains disabled from ROS/hardware.
The first live integration should use 60 mm/s, 150 mm/s^2, a 35 mm maximum
correction envelope, fresh encoder data, and an immediate zero-command fallback
on deadman release, stale payload state, mode change, or E-stop. Before enabling
motors, replay recorded encoder/gantry states through the controller and verify
axis polarity and command units.
