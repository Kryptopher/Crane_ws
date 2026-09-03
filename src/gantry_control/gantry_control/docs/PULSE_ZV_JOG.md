# Pulse / nonrobust ZV joystick jog

`pulse_zv_jog.py` provides exactly two operator-selectable modes:

- `PULSE`: direct proportional joystick velocity and an immediate zero command
  when LB or the stick is released.
- `ZV`: the configured fixed-speed operator pulse convolved with the ordinary
  two-impulse rope-mode ZV filter. There is no LQG, optimal endpoint, ZVD, or
  feedback correction.

Button 10 switches PULSE/ZV only while idle. Hold LB and move the left stick to
jog; RB requests E-stop. The ZV direction and speed are fixed from the first
impulse until its filtered stop completes. At zero damping the gains are 0.5
and 0.5, separated by `T=pi*sqrt(L/g)` (about 0.952 s for a 0.90 m rope).

The exact input-filter convolution is retained for a raw jog shorter than T.
That necessarily produces a delayed second half-pulse after release. For the
usual half/full/half/zero staircase, hold the jog longer than T before release.

With Mission Planner already running, use a second terminal:

```bash
cd /home/sanjay/crane_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run gantry_control pulse_zv_jog.py \
  --axis x --start-mode zv \
  --rope-length-m 0.90 --jog-speed-mm-s 100
```

The old `hybrid_zvd_lqg_jog.py` executable is a compatibility alias for the
same controller. Its previous LQG-only flags are accepted and ignored so saved
commands do not fail argument parsing.
