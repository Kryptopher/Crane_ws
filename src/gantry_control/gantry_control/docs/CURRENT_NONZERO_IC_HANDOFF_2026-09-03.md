# Adaptive Nonzero-IC Crane Experiment: Current Handoff

Last updated: 2026-09-03

## Critical operating rule

The human operator alone launches, enables, homes, jogs, stops, or otherwise
commands the gantry. Development agents may inspect logs, edit/build/test code,
and run offline analysis, but must never command the hardware. Preserve all
uncommitted work; this repository currently has a large dirty worktree.

## Objective

The experiment deliberately creates a known payload swing, identifies the
dominant pendulum motion online, predicts a positive payload peak, and starts a
forward-only shaped X move from that nonzero initial condition. The desired
outcome is accurate travel with little fundamental-mode residual swing.

## Current physical and software configuration

- X-axis experiment only.
- Payload angle comes from the Arduino payload encoder assembly.
- Nominal/current rope-to-payload-center length is approximately 35.5 in,
  or 0.9017 m.
- Payload and gantry telemetry normally run at approximately 100 Hz. Leave the
  Arduino and ROS encoder rates at 100 Hz for now.
- Recent trials start near X = 50 mm, request 1100 mm travel, and use
  750 mm/s maximum speed.
- The physical rail was reported extended from 1.15 m by 2.5 in, but
  `config/gantry.yaml` still contains `workspace_limit_m: 1.15`. Do not assume
  the software/controller limit was extended merely because the rail was.
- Current home/E-stop mapping in `config/gantry.yaml`:
  - Y/back home: node 0 input A, active low;
  - X/left home: node 1 input B, active high;
  - E-stop monitor: node 1 input A, active low/NC.
- The software-monitored E-stop was experimentally confirmed to stop motion,
  but it remains a secondary protection path rather than a safety-rated
  replacement for an external global-stop circuit.

Mission Planner defaults were made lightweight: camera tracking and the
separate initial-condition cart are opt-in. The payload encoder remains in use.

## Main implementation files

- `scripts/adaptive_paper_tdf_player.py`: experiment state machine, gates,
  controller-timed profile arming, logging, and automatic post-processing.
- `scripts/nonzero_ic_shaper.py`: free-swing estimator, two-impulse NZIC
  solver, robust three-impulse optimizer, and finite-amplitude correction.
- `scripts/nonzero_ic_exciter.py`: bounded return-to-anchor swing excitation.
- `scripts/plot_adaptive_paper_run.py`: automatic PNG/SVG/JSON outputs.
- `srv/ExecuteTimedProfile.srv`: atomic future-start profile sent to the
  controller.
- `src/gantry_controller.cpp`: controller-owned switch timing, workspace
  enforcement, feedback, homing, and E-stop handling.
- `payload_perception/encoder_serial_node.py`: Arduino parsing, encoder sample
  timestamping, relative position/velocity, and packet-age reporting.

## Data flow

```text
Arduino encoder packet
    -> encoder_serial_node (packet receive timestamp)
    -> /payload/pose_e_rel
    -> adaptive_paper_tdf_player
       -> bounded excitation
       -> 5 s free-swing fit
       -> future positive-peak prediction
       -> NZIC/robust schedule solve
       -> /gantry/execute_timed_profile
    -> gantry controller owns all velocity switch epochs
    -> residual logging and automatic plot/summary generation
```

The combined state sequence is:

```text
countdown -> excite -> id_hold -> wait_peak -> arm_profile
          -> armed_profile -> maneuver -> residual -> done
```

During excitation, the trolley makes bounded positive-axis excursions and
returns to its starting anchor. Once the requested swing amplitude converges,
the trolley holds zero for `tau = 5 s`. The maneuver does not begin until the
fit, telemetry, direction, workspace, command-speed, and timing gates pass.

## Measurement representation

The Arduino supplies encoder pitch/roll angles. The payload publisher also
computes the relative horizontal displacement used by the NZIC solver. For X:

```text
r_x = L sin(theta_x)
```

Angles are transported/logged in degrees where explicitly labelled; dynamics
and trigonometric calculations use radians. The shaper state is expressed in
millimetres and millimetres per second.

`/payload/pose_e_rel` now has 13 fields. The original first nine fields remain
unchanged, followed by raw relative velocities and encoder sample age:

```text
time, pitch_deg, roll_deg, x_rel_m, y_rel_m, z_rel_m,
vx_rel_m_s, vy_rel_m_s, vz_rel_m_s,
vx_rel_raw_m_s, vy_rel_raw_m_s, vz_rel_raw_m_s, sample_age_ms
```

The filtered velocity is an EMA and is deliberately not used to validate the
instantaneous peak. Its alpha of 0.35 at 100 Hz produces about 18.6 ms of
low-frequency group delay. At a roughly 285 mm, 3.25 rad/s swing, that delay
alone appears as approximately 17 mm/s of nonzero peak velocity and previously
made the peak diagnostic misleading. The nearest-event diagnostic now prefers
the raw finite-difference velocity.

## Timestamp repair already implemented

Previously, every publisher timer tick stamped the most recently received
Arduino values with the later timer time and could republish an unchanged
packet as a fresh measurement. This introduced roughly 10--25 ms of apparent
measurement lateness.

The encoder publisher now:

1. timestamps a sample with the serial packet's Jetson receive time;
2. publishes pose/relative state only once per new serial packet;
3. reports the timer pickup age as `sample_age_ms`; and
4. retains filtered and raw relative velocity separately.

The player calibrates the payload clock to its monotonic clock using the
minimum observed callback offset and logs the remaining callback queue delay.
The most recent live check showed about 97 Hz effective publication and one
sample age of 1.38 ms. In the timestamp-repair experiment, sample-age
p10/median/p90/max was approximately 0.96/4.92/7.46/9.08 ms.

Do not raise the ROS timer alone to 200 Hz. The Arduino remains approximately
100 Hz, so that would only poll each packet sooner; it would not add
measurements. The corrected source timestamp already preserves measurement
time.

## Current online frequency/state estimator

For irregularly sampled free-swing data, the current estimator searches
`omega` from 2.6 to 4.0 rad/s using 401 points and fits at each frequency

```text
x(t) = b0 + C cos(omega (t-t_ref)) + S sin(omega (t-t_ref)).
```

`b0`, `C`, and `S` are linear least-squares coefficients. The minimum-MSE grid
point is refined quadratically. Reported amplitude and normalized error are

```text
A     = sqrt(C^2 + S^2)
NRMSE = RMSE/A.
```

The final sample is the phase reference. The fitted sinusoid is projected to a
future positive peak in the direction of travel. Important: this production
NZIC estimator currently has neither a linear drift term nor exponential
damping. It is not identical to an earlier standalone example containing
`b0 + b1*t`.

The adaptive estimate around the present rope configuration is consistently
about 3.24--3.25 rad/s. This is the measured finite-amplitude fundamental, not
automatically the small-angle pole `sqrt(g/L)`. For a finite pendulum angle the
period grows with amplitude, so the measured frequency is expected to be
lower. The optional nonlinear correction uses the complete elliptic-integral
relation to map the measured finite-amplitude frequency to a small-angle
frequency before building the linear shaper.

## Nonzero-IC profiles

### Original two-impulse profile

For maximum velocity `V`, move time `tp = distance/V`, and solved quantities
`A0`, `A1`, and `Ts`, with `A0 + A1 = 1`:

```text
0 <= t < Ts       u = A0 V
Ts <= t < tp      u = V
tp <= t < tp+Ts   u = A1 V
t >= tp+Ts        u = 0.
```

The solver uses the projected initial displacement and payload velocity,
checks its terminal model residual, and rejects reverse or over-speed profiles.

### Robust profile

The robust option optimizes three nonnegative impulse weights over a configured
frequency band (recently +/-5%). Start weights add velocity and stop weights
remove it. Unit sums preserve monotonic velocity, and matching first moments
preserve exact modeled travel. The optimizer minimizes the worst normalized
residual over the band. Some optimized middle weights may be zero, so the
realized staircase need not visibly contain every possible level.

The entire knot sequence is sent once to `/gantry/execute_timed_profile` with a
future epoch. This removed the large and inconsistent per-switch ROS timer
jitter seen in early experiments. Controller scheduling is now sub-millisecond;
motor response remains event-dependent and is not proven to be one common pure
delay.

## Best and latest evidence

Best earlier robust scaled trial:

```text
nonzero_ic_robust5pct_scaled1p014_18deg_1100mm_v750_rep01
fitted omega                 3.243829 rad/s
shaper omega                 3.289243 rad/s (scale 1.014)
final-5-s residual p2p       0.2877 deg
final-5-s demeaned RMS       0.0909 deg
```

Pre-repair robust unscaled trial:

```text
nonzero_ic_robust5pct_unscaled_18deg_1100mm_v750_rep01
final-5-s residual p2p       1.9406 deg
final-5-s demeaned RMS       0.6709 deg
reported peak velocity       43.50 mm/s (filtered/timing-biased)
```

Post-timestamp-repair robust unscaled trial:

```text
nonzero_ic_robust_unscaled_timestampfix_18deg_1100mm_v750_rep01
travel                       1100.305 mm
fitted/shaper omega          3.246675 rad/s
robust band                  +/-5 percent
predicted peak state         284.917 mm, approximately 0 mm/s
measured nearest peak state  273.766 mm, 6.875 mm/s raw
measurement offset           -4.467 ms
arming queue delay           4.597 ms
final-5-s residual p2p       1.4590 deg
final-5-s demeaned RMS       0.5069 deg
```

The timestamp/raw-velocity repair therefore materially improved the unscaled
result and reduced the apparent peak velocity from tens of mm/s to 6.9 mm/s.
It did not yet match the best scaled robust result.

## New damping analysis

The latest schedule was armed from a 5 s window ending at payload time
41.3332 s and projected 1.6221 s forward to the selected peak. Over that same
window:

```text
undamped fit:
  omega             3.246695 rad/s
  amplitude          284.51 mm
  RMSE               2.40 mm
  predicted peak     284.51 mm

damped fit x=b0+exp(-lambda*t)(C cos(omega*t)+S sin(omega*t)):
  omega             3.246572 rad/s
  lambda             0.00585 1/s
  zeta=lambda/omega  0.00180
  end-window amp      280.47 mm
  RMSE                1.72 mm
  predicted peak      277.83 mm

measured near peak    273.77 mm
```

Thus a damping-aware envelope leaves the frequency essentially unchanged but
reduces peak-displacement prediction error from about 10.7 mm to about 4.1 mm.
This is currently the strongest identified estimator improvement.

## Why the prediction horizon was long

The robust optimization currently requires
`--timed-profile-lead-s >= 0.75`. On this computer, five offline runs of the
actual robust solve took only 120--126 ms. In the latest trial, the enforced
lead caused the nearby positive peak to be skipped and selected the following
peak 1.622 s ahead. A constant-amplitude model becomes less accurate over that
horizon.

Do not simply remove the guard without testing. A better implementation would
measure/budget optimizer plus ROS service time, use a conservative dynamic
lead (likely around 0.30--0.40 s), and retain the existing rule that the
controller must acknowledge the profile before its epoch. Combining a shorter
horizon with damping-aware state prediction is more principled than applying a
global empirical frequency scale.

## Current unresolved issues

1. **Constant-amplitude projection:** the zero-damping fit averages the
   decaying envelope and overpredicts a future peak.
2. **Overly conservative robust lead:** a 120 ms optimization is protected by
   a fixed 750 ms minimum, which can force prediction a full extra cycle ahead.
3. **Frequency interpretation:** the free-swing estimator returns the
   finite-amplitude fundamental, while the linear shaper expects a linearized
   model pole. Nonlinear correction is physically justified; a universal
   multiplicative scale is not yet justified across rope lengths/amplitudes.
4. **Residual scale advantage not fully explained:** scale 1.014 remains the
   best historical result. Timestamp repair explains part, but not all, of the
   difference. Differential motor transition dynamics and model mismatch
   remain possible; a single common actuator delay is not supported.
5. **Single-mode model:** measured high-frequency/second-mode content appears
   after acceleration changes. It should remain in plots, but it should be
   separated spectrally from the fundamental residual metric.
6. **Raw velocity noise:** raw finite differences are suitable for a
   nearest-peak diagnostic but contain large motion-transition spikes. Do not
   use them directly as an unfiltered feedback command.
7. **Open-loop final travel:** velocity integration can leave millimetre-scale
   position error. Any position trim must occur after shaping and be slow
   enough not to re-excite the payload.
8. **Workspace configuration mismatch:** physical rail extension and software
   controller limit must be reconciled explicitly before using additional
   stroke.

## Recommended next work

1. Add a damping-aware free-swing estimate with `lambda` and retain all
   existing amplitude, NRMSE, frequency-boundary, sample-count, freshness, and
   queue-delay gates.
2. Use damping for future state/envelope projection first. Do not automatically
   change the shaper dynamics to a damped plant in the same patch; that would
   confound two changes.
3. Add offline regression tests against the latest CSV and historical runs.
   Require improved held-out future-peak displacement without degrading phase.
4. Instrument total robust solve and service-acknowledgement time in every CSV.
5. Replace the fixed 0.75 s robust minimum with a measured conservative budget
   only after tests show the profile is always acknowledged safely.
6. Run one A/B pair at the same rope length, excitation, travel, speed, robust
   band, and frequency treatment: current estimator versus damping-aware state
   projection. Change nothing else.
7. Compare the fundamental-frequency residual over the final five seconds;
   also report total p2p and demeaned RMS so second-mode artifacts remain
   visible.
8. Only after state-prediction validation, repeat the frequency question using
   plain finite-amplitude omega versus nonlinear-corrected omega. Avoid the
   empirical 1.014 scale during that diagnostic.

## Verification status

After the timestamp repair, relevant unit tests passed (135 tests in the
combined focused run), and `gantry_control` plus `payload_perception` built
successfully. No hardware motion was initiated by the development agent.

Primary evidence files:

- `log/nonzero_ic_robust_unscaled_timestampfix_18deg_1100mm_v750_rep01.csv`
- `log/nonzero_ic_robust_unscaled_timestampfix_18deg_1100mm_v750_rep01_plot.png`
- `log/nonzero_ic_robust_unscaled_timestampfix_18deg_1100mm_v750_rep01_plot.svg`
- `log/nonzero_ic_robust_unscaled_timestampfix_18deg_1100mm_v750_rep01_summary.json`
- `log/nonzero_ic_robust5pct_scaled1p014_18deg_1100mm_v750_rep01.csv`
- `log/nonzero_ic_robust5pct_scaled1p014_18deg_1100mm_v750_rep01_summary.json`

Recent **nonrobust** comparison files:

- Nonlinear-corrected, scale 1.000:
  `log/nonzero_ic_video_18deg_1100mm_v750_rep03.csv`, with matching
  `_plot.png`, `_plot.svg`, and `_summary.json` files.
- Raw-fit frequency scaled by 1.014:
  `log/nonzero_ic_video_scaled1p014_18deg_1100mm_v750_rep04.csv`, with matching
  `_plot.png`, `_plot.svg`, and `_summary.json` files.

These two recent runs are both nonrobust, but they are not merely the same
frequency treatment with scale changed: Rep 03 used the nonlinear
finite-amplitude correction at scale 1.000 (3.24815 -> 3.26917 rad/s), while
Rep 04 applied scale 1.014 to the raw fitted frequency without nonlinear
correction (3.24596 -> 3.29141 rad/s). Preserve that distinction in any paper
comparison.
