# Forward-only minimum-time stop for hybrid joystick jog

## Behavior

`hybrid_zvd_optimal_jog.py` keeps the PULSE/HYBRID joystick interface and the
ZVD-shaped start. In HYBRID mode it continuously estimates
`[cart position, cart velocity, payload angle, payload angle rate]` and solves a
background **stop-if-released-now** problem while the stick remains held.

At release, the node immediately plays the freshest validated sequence. The
cart's final position is free: it may travel a short distance in the existing
direction while stopping, but it may never reverse. The commanded cart velocity
on each axis is constrained to the incoming direction or zero. Measurements
during playback are used only to abort model divergence and verify rest; they do
not alter the open-loop sequence.

The legacy executable `hybrid_zvd_lqg_jog.py` remains installed for command-line
compatibility, but it runs this controller and contains no LQR/LQG endpoint
feedback.

## Optimal-control formulation

The digital nonzero-initial-condition problem is a linear program. For each
candidate sampled horizon it minimizes forward stopping distance subject to:

```text
x_s[k+1] = A_s x_s[k] + B_s u[k]
u[k] has the incoming sign (or is zero)
vector command-speed and command-acceleration limits
forward workspace and maximum stopping-distance limits
terminal cart velocity and payload sway-energy bounds
u[N-1] approximately zero
```

Final cart position is not constrained to the release coordinate. This removes
the cause of the former repeated return corrections.

The planner first tests the maximum horizon, then uses a quasi-convex bisection
over horizon length to find the shortest feasible sampled stop. At that horizon,
the LP's secondary objective minimizes forward travel. With the nominal 0.90 m
rope and 100 mm/s jog, representative simulated stops take approximately
0.8--1.0 s and 35--45 mm of forward travel, depending on the measured swing
phase.

Nine exact-ZOH models combine nominal and +/-10% rope length with nominal and
+/-15% actuator time constant. The nominal terminal sway bound is 0.12 degrees;
off-nominal models use the configured robust multiplier. An independent
post-solve validator re-simulates every scenario and rejects command reversal,
cart reversal, vector speed/slew violations, workspace overrun, or excessive
terminal velocity/sway.

## Continuous plan refresh

The Kalman observer updates at each payload measurement. A single latest-wins
worker repeatedly:

1. Forecasts the measured state to the expected optimizer completion time.
2. Solves the forward-only minimum-time stop from that forecast state.
3. Caches the plan with its state and intended execution time.

At release the cache must satisfy age, cart-velocity, angle, and angular-rate
agreement limits. A valid optimal plan takes priority. If the plan is missing
or stale, the controller falls back to a deterministic ZV release: for zero
damping this commands half the release velocity for one switch interval and
then zero. This preserves the expected shaped-stop behavior instead of issuing
an immediate hard stop. The fallback is rejected if its projected travel would
violate the configured workspace or forward-distance limit.

After either stop, the node remains armed in `idle`, so the operator may start
another jog. A terminal-verification miss also returns to `idle`. Hardware or
telemetry safety faults still terminate the node. If an optimal sequence
diverges from its measured motion, the controller transfers to the same
workspace-checked ZV release tail rather than issuing an abrupt zero.

Important defaults are:

| Parameter | Default | Meaning |
|---|---:|---|
| `--optimal-min-horizon-s` | 0.30 s | Shortest horizon tested |
| `--optimal-horizon-s` | 2.80 s | Longest permitted stop |
| `--optimal-horizon-resolution-s` | 0.04 s | Minimum-time search resolution |
| `--optimal-plan-forecast-s` | 0.12 s | State forecast to expected LP completion |
| `--optimal-replan-period-s` | 0.10 s | Requested stop-plan refresh period |
| `--optimal-max-plan-age-s` | 0.50 s | Maximum release/cache time mismatch |
| `--max-regulate-accel-mm-s2` | 1000 | Vector command slew limit |
| `--max-forward-stop-distance-mm` | 250 | Hard forward travel cap |
| `--quiescent-angle-deg` | 0.12 deg | Nominal terminal/verification sway |
| `--quiescent-cart-vel-mm-s` | 2.0 | Nominal terminal/verification speed |

Tune the identified rope length, damping, and actuator time constant before
loosening terminal tolerances. If no forward-only plan is feasible, increase
the maximum horizon or forward distance before changing the residual-sway bound.

## Build and staged validation

```bash
colcon build --packages-select gantry_control
source install/setup.bash
ros2 run gantry_control hybrid_zvd_optimal_jog.py \
  --axis x --start-mode hybrid \
  --log-csv /tmp/hybrid_optimal_jog.csv
```

Validate one axis at low speed before using the 100 mm/s default. The
ROS-independent tests are:

```bash
PYTHONPATH=src/gantry_control/gantry_control/scripts \
  python3 -m pytest -q \
  src/gantry_control/gantry_control/test/test_optimal_stop_planner.py
```

## Research basis

- Dhanda, Vaughan, and Singhose, “Optimal Input Shaping Filters for Non-Zero
  Initial States,” ACC 2009. The optimal solution is bang-bang and its digital
  approximation is a quasi-convex linear program:
  <https://doi.org/10.1109/ACC.2009.5160293>
- Stein and Singh, “Minimum Time Control of a Gantry Crane System with Rate
  Constraints,” 2023. Velocity-limited zero-residual profiles are
  bang-off-bang, with robustness obtained through terminal sensitivity:
  <https://arxiv.org/abs/2301.08716>
- Giacomelli et al., “Model Predictive Control for operator-in-the-loop overhead
  cranes,” ETFA 2018. The operator's release time is unknowable, motivating
  online state prediction and constrained velocity control:
  <https://doi.org/10.1109/ETFA.2018.8502591>
- Mohammed, Alghanim, and Andani, “A robust input shaper for trajectory control
  of overhead cranes with non-zero initial states,” 2021:
  <https://doi.org/10.1007/s40435-020-00631-0>

## Limitation

The executed tail remains open loop. Continuous replanning makes the release
state current, and the scenario bank adds model tolerance, but a disturbance
after release cannot be rejected without feedback. Runtime measurements can
abort a divergent plan; they cannot preserve zero residual sway after an abort.
