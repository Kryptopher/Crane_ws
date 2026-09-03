# Robust two-mode 2-D square input shaping

`robust_2d_square_player.py` commands an axis-aligned closed path in this exact
order:

1. `+X`
2. `+Y`
3. `-X`
4. `-Y`

The final point is the starting point. Each constant-velocity leg is convolved
with two three-impulse zero-vibration-derivative (ZVD) shapers: the rope-length
mode and a measured higher mode. Convolution gives nine positive impulses. The
same shaper is applied to both Cartesian components because the small-angle X
and Y dynamics share these modes. The next leg does not start until the
previous leg's last shaped stop impulse has finished, so commands never cut a
corner diagonally.

Nonzero-initial-condition (NZIC) shaping is enabled by default. Before `+X`,
the player continuously fits a bias plus a damped rope-mode sinusoid to the
last 2.5 seconds of encoder pitch and roll. This estimates angle and angular
rate without numerically differentiating quantized angle samples. The fit uses
Huber reweighting so isolated count/packet glitches do not set the estimated
phase. The estimate keeps updating while the cart is held at zero, and a
planned profile is discarded if a fresh estimate disagrees by more than the
validation tolerance.

Only the physical rope-mode initial state is actively canceled. The 5.06 Hz
mode remains a zero-excitation robustness constraint: its modal input gain has
not been identified well enough to interpret encoder angle as a trustworthy
nonzero structural-mode state. This avoids inventing an unmeasured high-mode
control gain while still preventing the maneuver from exciting that mode.

The NZIC optimizer independently chooses the nine acceleration weights and
nine deceleration weights for the first `+X` leg. It enforces all of the
following:

- all weights are nonnegative;
- acceleration and deceleration weights each sum to one;
- their first moments are equal, preserving exact commanded distance;
- zero nominal terminal angle/rate for the measured nonzero rope state;
- zero first frequency derivative at the rope mode and the 5.06 Hz mode;
- bounded residual across a +/-10% rope-frequency band and the measured
  4.94--5.15 Hz higher-mode band.

Consequently the corrected command still only speeds up, cruises, and slows
down. It cannot reverse or reaccelerate during a leg. If the current swing
phase cannot satisfy those constraints with positive velocity increments, the
cart stays at zero and the planner retries at the next measured phase.

Roll is deliberately not forecast open-loop across the entire `+X` move.
Instead, encoder state continues updating during the run. At the first corner,
the player holds zero, estimates roll again, and builds the `+Y` NZIC shaper
for its actual start time. An infeasible phase extends only this zero-velocity
dwell; it never produces a back-and-forth cart command. `-X` and `-Y` retain
the standard robust two-mode ZVD because their corresponding positive leg has
already canceled the initial sway.

For damping ratio `zeta` and damped natural frequency `omega_d`, each mode's
impulses occur at `0`, `T`, and `2T`, where `T = pi / omega_d`. With the default
zero-damping model their amplitudes are `[0.25, 0.50, 0.25]`. Convolving the two
ZVD sequences provides nominal zero residual and a zero derivative of residual
with respect to frequency at both modes.

Encoder-only offline identification from the six unshaped pulse archives in
`Final_Async_1000mm_v150_20260622/L090` found a rope mode near 0.519 Hz. The
first five same-session runs put the higher mode at 5.0518, 5.0536, 5.0666,
5.0736, and 5.0696 Hz, with a joint estimate of 5.0628 Hz. The weaker sixth run
estimated 4.9425 Hz. The data does **not** support an exact 6 Hz mode. The
default second-mode center is therefore 5.06 Hz, and its ZVD robustness covers
the observed 4.94--5.15 Hz range.

The raw constant-velocity duration must be at least the full convolved shaper
tail. This guarantees that command magnitude only rises during acceleration
and only falls during deceleration; startup and stopping impulses cannot
interleave. At 0.9 m, 5.06 Hz, and 400 mm/s, the tail is about 2.101 s and the
minimum leg is about 840 mm. A 1000 mm leg is valid; a 700 mm leg is rejected.

## Build and preview

```bash
cd /home/sanjay/crane_ws
colcon build --packages-select gantry_control
source install/setup.bash
ros2 run gantry_control robust_2d_square_player.py \
  --x-distance-mm 1000 --y-distance-mm 1000 \
  --speed-mm-s 400 --rope-length-m 0.90 \
  --second-mode-hz 5.06 --corner-dwell-s 1
```

Preview is the default and cannot move the crane. It prints all nine impulses,
every leg's raw and shaped stop times, minimum monotonic leg length, total
duration, and modeled residual at both modes.

## Run on hardware

Start the normal hardware/perception stack in terminal 1:

```bash
cd /home/sanjay/crane_ws
source install/setup.bash
ros2 launch gantry_control mission_planner.launch.py
```

Home the gantry and place it far enough from the positive X and Y limits for the
requested square. Then run the explicitly armed experiment in terminal 2:

```bash
cd /home/sanjay/crane_ws
source install/setup.bash
ros2 run gantry_control robust_2d_square_player.py --execute \
  --x-distance-mm 1000 --y-distance-mm 1000 \
  --speed-mm-s 400 --rope-length-m 0.90 \
  --second-mode-hz 5.06 --corner-dwell-s 1
```

The player requests `TRAJ` mode and enables the gantry only after it has fresh
`/gantry/state` and `/payload/pose_e_rel` samples. It verifies that the full
square fits within the configured workspace, streams at 100 Hz, aborts on stale
feedback, mode/enable/E-stop changes, workspace violations, or excessive path
error, and commands zero velocity on completion or interruption. Before moving,
the cart must remain below its measured speed limit for 0.5 seconds while the
NZIC state is estimated. The default per-axis fitted sway safety envelope is
1.5 degrees. A noisy fit, stale feedback, excessive sway, infeasible positive
impulses, or loss of higher-mode robustness prevents motion. Transient ROS
service discovery is retried for 15 seconds before an unavailable-service
abort. CSV logging is enabled by default under `./logs/robust_2d/`, including
the initial angle/rate estimates, fit errors, and correction count.

Do not run another process that publishes motion on `/traj_cmd` during this
experiment. Press `Ctrl+C` to command zero and stop the player.

Useful options:

- `--corner-dwell-s 1`: add a one-second zero-command pause after each corner.
- `--damping-ratio 0.02`: use measured rope-mode damping.
- `--timing-scale 1.02`: calibrate rope-mode timing.
- `--second-mode-hz 5.06`: set the higher-mode center; `0` restores one-mode
  shaping for an A/B comparison.
- `--second-mode-damping-ratio 0.02`: use measured higher-mode damping.
- `--second-mode-timing-scale 1.02`: calibrate higher-mode timing.
- `--settle-time-s 5`: extend residual-sway measurement after the final stop.
- `--start-angle-limit-deg 0.5`: require a quieter payload before starting.
- `--zero-ic-only`: disable NZIC planning and restore the measured-rest gate;
  `--start-angle-limit-deg` applies in this mode.
- `--ic-min-amplitude-deg 0.10`: below this fitted amplitude, keep the standard
  two-mode ZVD weights.
- `--ic-max-amplitude-deg 1.50`: refuse to start outside this per-axis NZIC
  safety envelope.
- `--ic-max-fit-rmse-deg 0.10`: reject a poor encoder sinusoid fit.
- `--ic-plan-validation-tolerance-deg 0.12`: tighten or loosen live forecast
  invalidation before each corrected leg.
- `--ic-y-plan-timeout-s 12`: maximum zero-velocity corner hold while seeking a
  feasible `+Y` phase.
- `--max-path-error-mm 50`: tighten the measured path-tracking abort guard.
- `--log-csv /path/to/run.csv`: choose an explicit output file.
- `--allow-missing-payload`: run without payload feedback; use only for sensor
  troubleshooting and only together with `--zero-ic-only`, because NZIC
  shaping cannot operate without measured encoder state.

The NZIC pole-cancellation and frequency-insensitivity constraints follow
Newman, Hong, and Vaughan, *The Design of Input Shapers Which Eliminate Nonzero
Initial Conditions*, and the overhead-crane generalization by Mohammed,
Alghanim, and Taheri, *A robust input shaper for trajectory control of overhead
cranes with non-zero initial states*.
