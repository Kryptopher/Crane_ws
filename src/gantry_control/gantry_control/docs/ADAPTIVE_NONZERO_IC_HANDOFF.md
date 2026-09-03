# Adaptive Nonzero-Initial-Condition Crane Experiments: Handoff

Last updated: 2026-09-01

## 1. Current status

The X-axis experiment can now:

1. create a bounded payload swing automatically while returning the trolley to its starting anchor;
2. hold the trolley still for a configurable free-swing identification window `tau`;
3. fit the undamped swing frequency and instantaneous payload state;
4. predict the next swing peak in the requested travel direction;
5. atomically arm a controller-owned, precisely timed forward-only profile; and
6. log a 10 s residual window.

Three combined 5-degree excitation/nonzero-IC trials have been collected. Frequency identification is highly repeatable (`omega_n` approximately 3.239 rad/s, effective length approximately 0.935 m), but terminal swing cancellation is not yet satisfactory. The final-five-second residual is approximately 2.1--2.8 degrees peak-to-peak. This work should therefore still be treated as an experiment, not a production controller.

Important operational rule: the human operator starts all ROS launch files and motion trials. Automated development or analysis tools must not start, enable, home, jog, or command the gantry.

## 2. Hardware and coordinate assumptions

- Physical rope length: approximately 0.90 m to the payload center.
- Configured X workspace: 0 to 1150 mm, with a 5 mm software margin in these trials.
- Recent experiments start near X = 100 mm and request +800 mm travel.
- The payload swing state is payload position minus trolley position along the requested world axis.
- Positive initial swing means the payload is displaced in the direction of the requested move.
- The tested experimental velocity is 600 mm/s. A previously discussed 880 mm/s value is not validated here and should not be treated as an approved operating point.

Current SC4 input mapping in `config/gantry.yaml`:

| Function | Input | Polarity |
|---|---:|---|
| Back/Y home sensor | node 0 input A | active low |
| Left/X home sensor | node 1 input B | active high |
| Physical E-stop backup | node 1 input A | active low / normally closed |

The physical E-stop is monitored in the 100 Hz controller loop and calls an abrupt motor `NodeStop`, then latches the E-stop state. Releasing the switch does not resume motion; `/gantry/clear_estop` is required. This software-monitored input is a secondary path. The SC4 Global Stop or another external safety-rated circuit remains the primary safety path.

`/gantry/home_sensors` publishes 12 values in this order:

```text
[node0_InA, node0_InB, node1_InA, node1_InB,
 back_active, left_active,
 back_node, back_input_b, back_active_high,
 left_node, left_input_b, left_active_high]
```

## 3. Relevant source files

- `scripts/adaptive_paper_tdf_player.py`: ROS 2 experiment state machine, safety gates, command streaming, and CSV logging.
- `scripts/nonzero_ic_shaper.py`: free-swing estimator and closed-form nonzero-IC solver.
- `scripts/nonzero_ic_exciter.py`: bounded, positive-axis, return-to-anchor excitation.
- `scripts/plot_adaptive_paper_run.py`: automatic per-run PNG and JSON summary.
- `srv/ExecuteTimedProfile.srv`: atomic future-start staircase profile request.
- `test/test_nonzero_ic_shaper.py`: unit and state-machine tests.
- `src/gantry_controller.cpp`: TRAJ command handling, encoder state, homing, workspace checks, and physical E-stop monitor.
- `config/gantry.yaml`: home inputs, E-stop input, workspace, and controller parameters.
- `launch/mission_planner.launch.py`: lightweight stack options; camera trackers and IC cart are currently disabled by default.
- `ops/plot_bounded_nonzero_ic.py`: read-only plot generator for these trials.

The repository worktree contains many unrelated and uncommitted changes. Do not discard, reset, or overwrite them. The nonzero-IC work described here is also not committed as of this handoff.

## 4. Experiment state machine

The combined run follows:

```text
telemetry/countdown -> excite -> id_hold -> wait_peak -> arm_profile
                    -> armed_profile -> maneuver -> residual -> done
```

### Excite

The current bounded exciter is deliberately not the earlier rate-feedback/chattering controller. For each pendulum period it tracks a positive-only raised-cosine trolley reference:

```text
q_ref(t) = D/2 [1 - cos(omega t)]
v_ref(t) = D omega/2 sin(omega t)
```

The trolley starts at the anchor, moves only toward positive X, and returns to the anchor every cycle. Swing amplitude is estimated as half the encoder-angle peak-to-peak value over a complete cycle. If the target has not been reached, excursion `D` increases subject to:

- a response-scaled fine increment near `target - tolerance` (minimum 0.25 mm);
- the configured excursion increment as the maximum per-cycle step;
- maximum excursion/travel budget;
- maximum excitation velocity;
- slew-rate limit;
- workspace bounds;
- timeout; and
- abort angle.

When the measured cycle amplitude reaches `target - tolerance`, the controller stops adding energy and returns to the anchor. The ID timer begins only after the anchor-return condition is satisfied.

### ID hold

The trolley command is zero for `tau`. The current estimator in this path fits

```text
x(t) = b0 + C cos(omega t) + S sin(omega t)
```

for each of 401 candidate frequencies from 2.6 to 4.0 rad/s. At each frequency, `b0`, `C`, and `S` are obtained by linear least squares. The frequency with minimum mean squared residual is selected, followed by a local quadratic interpolation across the best grid cell. Reported quantities are:

```text
amplitude = sqrt(C^2 + S^2)
RMSE      = sqrt(mean((x - x_hat)^2))
NRMSE     = RMSE / amplitude
```

This exact adaptive nonzero-IC estimator does **not** include the `b1*t` drift term that appeared in a separate standalone zero-zeta example. The distinction must be preserved when reproducing or publishing the method.

The last sample is the sinusoid's time reference. After the grid fit, the fitted state is projected forward by the measured solver/sample age so that the captured initial state is not stale merely because fitting took several milliseconds.

The payload publisher time is mapped to the player's monotonic clock using the
minimum observed value of `callback_receive_time - payload_time`. Each
observation contains that fixed clock offset plus nonnegative ROS/executor
queueing, so the minimum is the least-delayed online clock calibration. A
payload timestamp reset restarts the calibration.

### Wait for peak

By default, motion is phase locked. The fitted sinusoid predicts the next
positive peak in the direction-of-travel coordinate at least 0.25 s in the
future. For negative-axis travel, this correctly selects the negative
world-axis peak. The state is projected to that exact future epoch and must
satisfy:

```text
signed swing > 0
abs(fitted payload velocity) <= 10 mm/s
```

The frequency fit remains adaptive while waiting. The schedule is sent once
through `/gantry/execute_timed_profile`. While the service is pending and until
the accepted epoch, the trolley remains commanded to zero. The legacy
timer-streamed peak gate remains available with
`--no-controller-timed-profile` for comparison only.

Before arming, the latest sample must also satisfy
`payload_queue_delay <= 5 ms` by default. A fresh callback is not necessarily a
fresh measurement when the executor is draining queued messages; this gate
holds zero and retries instead of shifting the predicted peak. Configure the
gate with `--nonzero-ic-max-payload-queue-delay-ms`.

### Maneuver

Let `V` be `vmax`, `tp = distance/V`, and let the solver return `A0`, `A1`, and `Ts`, with `A0 + A1 = 1`. The velocity profile is:

```text
0 <= t < Ts       : u = A0 V
Ts <= t < tp      : u = V
tp <= t < tp+Ts   : u = A1 V
t >= tp+Ts        : u = 0
```

The solver implements the closed form derived from the original MATLAB
equations, checks the terminal position and velocity residual numerically, and
rejects singular or out-of-interval solutions. The controller buffers the four
velocity knots atomically, maps the requested ROS epoch to its steady clock,
and writes to the motor API only at the start and the three switches. It does
not depend on four independently delivered ROS stream messages.

## 5. Safety and validity gates

Current nonzero-IC defaults or explicit test values are:

| Gate | Value used |
|---|---:|
| ID duration `tau` | 5.0 s |
| Frequency search | 2.6 to 4.0 rad/s |
| Frequency grid | 401 points plus quadratic refinement |
| Minimum fit samples | 100 |
| Minimum fit-window duration | max(0.5 s, 0.9 `tau`) |
| Minimum fitted amplitude | 10 mm |
| Maximum fit NRMSE | 0.05 |
| Reject frequency at search boundary | yes, within one grid step |
| Start direction | positive swing in travel direction |
| Peak velocity tolerance | +/-10 mm/s |
| Forward-only gains | `0 <= A0,A1 <= 1` |
| Base velocity | 600 mm/s |
| Absolute command-speed bound | 600 mm/s |
| Requested travel | 800 mm |
| Max-travel guard | 850 mm |
| Workspace | 0 to 1150 mm |
| Workspace margin | 5 mm |
| Residual collection | 10 s |
| Gantry-state freshness | 0.25 s |
| Encoder diagnostic freshness | 0.25 s |
| Encoder health required | yes |
| Controller-timed start lead | at least 0.25 s |

Before starting, the solved switch-event positions are transformed into world
coordinates and must remain within `[workspace_min + margin,
workspace_max - margin]`. During excitation, arming, maneuver, and residual
collection, the player now aborts on stale gantry state, stale/unhealthy
encoder diagnostics, actual workspace-margin violation, or actual directional
travel above `max-travel`. The gantry controller independently enforces its
encoder-based workspace and the hardware E-stop monitor. The encoder publisher
also stops publishing payload pose when the Arduino serial stream is stale, so
a frozen sample cannot acquire a fresh ROS timestamp.

For the bounded excitation used in Reps 1--3:

| Parameter | Value |
|---|---:|
| Target angle | 5 degrees |
| Angle tolerance | 1 degree |
| Excitation speed | 40 mm/s |
| Initial excursion | 10 mm |
| Excursion increment | 5 mm/cycle |
| Maximum excursion | 40 mm |
| Position gain | 2 1/s |
| Return speed | 20 mm/s |
| Return tolerance | 1 mm |
| Slew limit | 200 mm/s^2 |
| Timeout | 20 s |
| Stand-clear countdown | 5 s |
| Abort angle | 12 degrees |

## 6. Results to date

All three trials used an 800 mm move at 600 mm/s after bounded 5-degree
excitation. They predate the controller-timed profile implementation and are
the baseline for the next comparison.

| Metric | Rep 1 | Rep 2 | Rep 3 |
|---|---:|---:|---:|
| Fitted `omega_n` (rad/s) | 3.239858 | 3.238117 | 3.238145 |
| Fitted half-period (s) | 0.969670 | 0.970191 | 0.970183 |
| Fitted amplitude (mm) | 81.091 | 112.443 | 107.951 |
| Fit RMSE (mm) | 0.603 | 0.802 | 0.847 |
| Fit NRMSE | 0.00744 | 0.00713 | 0.00785 |
| Locked initial swing (mm) | 81.056 | 112.432 | 107.930 |
| Locked payload velocity (mm/s) | -7.764 | 5.231 | 6.901 |
| `A0` | 0.403683 | 0.361902 | 0.369168 |
| `A1` | 0.596317 | 0.638098 | 0.630832 |
| `Ts` (s) | 1.083251 | 1.124808 | 1.120502 |
| Actual travel (mm) | 796.372 | 803.605 | 809.713 |
| Travel error (mm) | -3.628 | +3.605 | +9.713 |
| Residual 10 s p2p (deg) | 4.903 | 5.332 | 5.119 |
| Residual 10 s RMS (deg) | 0.784 | 0.986 | 1.043 |
| Final 5 s p2p (deg) | 2.120 | 2.674 | 2.831 |

Rep 3's fitted frequency corresponds to an effective simple-pendulum length of approximately 0.935 m (`g/omega_n^2`), close to the measured 0.90 m rope-to-payload-center estimate.

For Rep 3, the locked command levels were approximately:

```text
A0*V = 221.5 mm/s
V    = 600.0 mm/s
A1*V = 378.5 mm/s
0
```

The modeled maneuver duration was approximately `tp + Ts = 1.3333 + 1.1205 = 2.4538 s`.

### Rep 3 logging verification

The earlier CSV logger skipped the `wait_peak` state. That omission made plots connect the last ID point directly to the maneuver and created a misleading straight line/white gap. The state now logs at zero command before each peak-lock attempt.

Rep 3 verifies the repair:

- 30 `wait_peak` rows over 0.410 s;
- median sample spacing 9.91 ms;
- one observed maximum spacing of 79.7 ms;
- commanded and measured trolley velocity both zero;
- fixed trolley position at 99.541 mm; and
- payload angle evolves from 1.56 degrees to approximately 6.76 degrees during the wait.

The angle trace is now visible rather than an invented straight connector. There is still visible high-frequency/jagged encoder behavior near the maneuver transitions, so the angle measurement path and timestamp alignment remain investigation items.

The later 10-degree Rep 1 review isolated the remaining pre-motion kink.  The
frequency-grid calculation blocked the single-threaded ROS executor for about
76 ms; one stale payload sample was repeated and queued samples were then
plotted at their delayed timer-callback times.  This was a recording artifact:
the measured slope across the gap was about 30.0 deg/s versus 30.6 deg/s from
the fitted sinusoid.  The implementation now:

- evaluates the frequency grid as batched least squares;
- runs the nonzero-IC fit on a worker so subscriptions and logging continue;
- plots payload angle against its aligned measurement timestamp; and
- removes repeated timer rows carrying the same payload sample timestamp.

The high-frequency response after trolley motion starts or stops is retained;
it is treated as measured second-mode dynamics, not as this timestamp artifact.

## 7. Latency measurements already collected

Two 600 mm/s pulse trials instrumented publisher, controller, motor write, and feedback timing.

Representative pre-delay Rep 2 values:

| Segment | Start event | Stop event |
|---|---:|---:|
| Publisher to controller receive | 1.70 ms | 0.41 ms |
| Controller receive to apply | 3.99 ms | 7.77 ms |
| Motor API write duration | 2.51 ms | 2.72 ms |
| Publisher to motor-write completion | 8.20 ms | 10.91 ms |
| Write completion to Teknic measured-velocity median crossing | 8.87 ms | 14.59 ms |
| Write completion to raw position-derivative median crossing | 13.39 ms | 19.17 ms |
| Write completion to filtered-position-velocity median crossing | 33.06 ms | 37.44 ms |

These results show real transport/application latency, but they do not justify
adding one fixed command pre-delay. Review of the three combined trials showed
that the controller-applied switch-interval errors predict the travel error:
approximately -3.69, +3.72, and +9.80 mm predicted versus -3.63, +3.60, and
+9.71 mm measured. In Rep 3 the three applied interval errors were about
+7.54, -2.68, and +25.73 ms. Propagating the measured trolley trajectory
through the simple pendulum model reproduced the late residual closely
(approximately 49.6 mm predicted versus 44.5 mm measured, correlation 0.974).
This is the reason for moving switch ownership into the controller rather than
adding a guessed scalar delay.

## 8. Known limitations and open questions

1. **Residual cancellation was incomplete with streamed switching.** The
   mathematical terminal residual is near machine zero, but the three baseline
   runs retained roughly 2.1--2.8 degrees p2p in the final five seconds. The
   controller-timed implementation still needs hardware validation.
2. **Travel is open-loop velocity integrated.** Rep 3 overshot the requested travel by 9.7 mm. There is no final position correction in this profile.
3. **Angle and displacement signals differ.** Identification and shaper state use relative displacement in millimetres; validation plots use the encoder axis angle in degrees. Their calibration, signs, timing, and filtering must be checked together.
4. **The model is undamped and single-mode.** Rope/payload geometry, trolley dynamics, command filtering, structural modes, and damping are omitted.
5. **Peak locking uses the fitted state.** It can satisfy the mathematical +/-10 mm/s gate even when raw finite-difference encoder velocity is noisy.
6. **Historical logs can contain an approximately 80 ms `wait_peak` gap.** The
   fit is now performed off the ROS callback thread and angle plots use the
   payload measurement timestamp. New runs should be checked to confirm that
   pre-motion telemetry remains continuous on the Jetson.
7. **Do not shorten the configured rope merely to accept a large initial condition.** Adaptive frequency ID should describe the actual plant. Rope length is a model prior/fallback, not a tuning knob for making unsafe gains pass.
8. **Do not increase speed yet.** First establish repeatability, sensor timing, and residual improvement at 600 mm/s.

## 9. Recommended next work

1. Run one conservative 5-degree/600 mm/s validation with controller-timed
   switching and inspect the log before repeating.
2. Confirm the log contains controller acceptance, the predicted peak epoch,
   and switch timing without the earlier per-switch ROS delivery jitter.
3. Compare fitted state at the scheduled epoch against raw encoder angle and a
   locally fitted encoder sinusoid at the same timestamp.
4. Separate residual into fundamental frequency and high-frequency encoder components.
5. Test a small model-based delay sweep offline or in simulation before any hardware compensation.
6. Add a final low-speed position correction only after the swing-shaping segment, ensuring it does not re-excite the payload.
7. Once repeatable, compare against pulse and fixed-frequency ZV baselines using identical excitation amplitude, travel, velocity, and residual metric.

## 10. Operator commands

### Lightweight Mission Planner stack

This keeps the payload encoder but disables the joystick, camera trackers, and separate IC cart:

```bash
cd /home/sanjay/crane_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch gantry_control mission_planner.launch.py \
  start_encoder:=true \
  start_joy:=false \
  start_tracker:=false \
  start_phase1_tracker:=false \
  start_ic_cart:=false
```

### Combined bounded-excitation/nonzero-IC trial template

Only the human operator should run this after homing, confirming the E-stop, checking the workspace, and standing clear. Increment the log filename for every trial.

The player now always saves a CSV. If `--log-csv` is omitted, it creates a
timestamped file under `~/crane_ws/log/adaptive_runs`. By default it also saves
`<csv_stem>_plot.png` and `<csv_stem>_summary.json` after completion or
shutdown. Use `--no-auto-plot` only when automatic post-processing is not
wanted.

Two default-off experiment controls keep shaper-frequency sensitivity separate
from actuator timing compensation:

- `--nonzero-ic-shaper-frequency-scale 0.99|1.00|1.01` multiplies only the
  frequency passed to the closed-form shaper. The fitted frequency, fitted
  state, and future positive-peak prediction are unchanged.
- `--nonzero-ic-shaper-omega-rad-s 3.25` instead supplies an absolute shaper
  frequency. It is mutually exclusive with a non-unit frequency scale. The
  adaptive sinusoid fit remains active only to estimate the live initial state
  and future positive peak; it does not select the schedule frequency.
- `--timed-profile-actuation-lead-ms N` advances the complete controller-owned
  knot schedule by `N` milliseconds relative to that fitted physical peak. It
  is the equivalent-pure-delay actuator model: the command is issued early so
  the physical velocity response, rather than the motor API write, aligns with
  the payload peak. It requires `--controller-timed-profile` and defaults to
  zero. The 2026-09-02 700 mm/s timing audit measured 17.5--19.3 ms of
  area-equivalent delay, so use an explicit 18 ms nominal value for the next
  controlled validation rather than changing the global default.
- `--nonzero-ic-max-payload-queue-delay-ms 5` rejects a queued payload sample
  at the arming instant. The 2026-09-02 scale-1.014 trial exposed a 38.0 ms
  arming offset even though callback freshness passed; the uncorrected profile
  consequently began well after the intended peak.

The CSV and automatic JSON summary record the fitted frequency, scaled shaper
frequency, scale, actuator lead, calibrated payload-clock offset, live queue
delay, and frozen arming queue delay. They also keep the sinusoid-predicted peak
state separate from the nearest filtered encoder measurement at command start
and at the intended physical peak. The direct velocity diagnostic is encoder
relative velocity plus measured trolley velocity; its sample-time offset is
logged because the 100 Hz stream cannot normally sample the event exactly.
Change only one control at a time and compare the final-five-second residual
fundamental against the measured pre-maneuver amplitude.

```bash
cd /home/sanjay/crane_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 run gantry_control adaptive_paper_tdf_player.py \
  --profile nonzero-ic \
  --axis x \
  --target-distance-mm 800 \
  --vmax-mm-s 600 \
  --nonzero-ic-max-command-speed-mm-s 600 \
  --max-travel-mm 850 \
  --workspace-min-mm 0 \
  --workspace-max-mm 1150 \
  --workspace-margin-mm 5 \
  --tau 5.0 \
  --robust-rope-length-m 0.9 \
  --nonzero-ic-adaptive-frequency \
  --nonzero-ic-max-fit-nrmse 0.05 \
  --nonzero-ic-start-at-peak \
  --nonzero-ic-peak-velocity-tolerance-mm-s 10 \
  --nonzero-ic-max-payload-queue-delay-ms 5 \
  --controller-timed-profile \
  --timed-profile-lead-s 0.25 \
  --nonzero-ic-shaper-frequency-scale 1.00 \
  --timed-profile-actuation-lead-ms 18 \
  --excite \
  --excite-target-angle-deg 5 \
  --excite-angle-tolerance-deg 1 \
  --excite-speed-mm-s 40 \
  --excite-initial-excursion-mm 10 \
  --excite-excursion-step-mm 5 \
  --excite-travel-budget-mm 40 \
  --excite-position-kp-s 2 \
  --excite-return-speed-mm-s 20 \
  --excite-return-tolerance-mm 1 \
  --excite-slew-mm-s2 200 \
  --excite-timeout-s 20 \
  --excite-standclear-s 5 \
  --excite-abort-angle-deg 12 \
  --residual-window 10 \
  --log-csv /home/sanjay/crane_ws/log/bounded_nonzero_ic_5deg_rep04.csv
```

### Regenerate the comparison plot (read-only with respect to hardware)

```bash
cd /home/sanjay/crane_ws
python3 ops/plot_bounded_nonzero_ic.py
```

Output:

```text
/home/sanjay/crane_ws/log/bounded_nonzero_ic_5deg_comparison.png
```

### Build and test after code changes

```bash
cd /home/sanjay/crane_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select gantry_control payload_perception --symlink-install
source install/setup.bash
python3 -m pytest -q src/gantry_control/gantry_control/test
```

At the time of the frequency-scale/actuator-lead update, `gantry_control`
built successfully and its suite contained 100 passing tests.

## 11. Evidence files

- `log/bounded_nonzero_ic_5deg_rep01.csv`
- `log/bounded_nonzero_ic_5deg_rep02.csv`
- `log/bounded_nonzero_ic_5deg_rep03.csv`
- `log/bounded_nonzero_ic_5deg_rep04.csv`
- `log/bounded_nonzero_ic_5deg_rep04_plot.png`
- `log/bounded_nonzero_ic_5deg_rep04_summary.json`
- `log/bounded_nonzero_ic_5deg_comparison.png`
- `log/traj_latency_600mms_rep01.csv`
- `log/traj_latency_600mms_predelay_rep02.csv`
