# Adaptive Input Shaping Paper Test Plan

## Objective
Build a repeatable data package for comparing crane payload residual swing under:

- Pulse baseline
- IS1 adaptive TDF
- IS2 paper closed-form TDF
- Robust model-based ZVD

All tests use the gimbal encoder payload estimate, normally `/payload/pose_e_rel`.

## Test Matrix
Recommended first paper matrix:

- Axis: `x`
- Target: `750 mm`
- Speed: `150 mm/s`
- Rope lengths: `1.00 m`, `1.20 m`, `1.40 m`
- Repeats: `3` minimum, `5` preferred for final paper figures
- Residual window: `3 s`
- Payload and box geometry fixed

## Method Definitions
Pulse commands `vmax` until `tf = distance / vmax`, then stops.

IS1 starts at `K*vmax`, estimates the system online, then uses the paper TDF velocity schedule with switches:

```text
T, 2T, tf, tf+T, tf+2T
```

IS2 starts at `K*vmax`, estimates until `tau`, solves the paper closed-form equations, then switches:

```text
tau+T, tau+2T, tf, tf+tau+T, tf+tau+2T
```

Robust ZVD is model-based from rope length:

```text
omega_n = sqrt(g / L)
T = pi / omega_n
Kz = exp(-zeta*pi/sqrt(1-zeta^2))
[A0, A1, A2] = [1, 2Kz, Kz^2] / (1 + 2Kz + Kz^2)
```

It uses the same switch locations as IS1. With `zeta=0`, amplitudes are `[0.25, 0.50, 0.25]`.

## Required Outputs
Each campaign folder should contain:

- `manifest.json` and `manifest.csv`
- `RUN_COMMANDS.sh`
- `TEST_PLAN.md`
- `raw/*.csv`
- `terminal/*.log`
- `tables/summary_metrics.csv`
- `tables/summary_by_condition.csv`
- `plots/*.png`
- `SUMMARY.md`
- `<session>.zip`

## Primary Metrics
- Residual swing p2p over the residual window
- Residual swing RMS over the residual window
- Residual max absolute swing
- Cart travel error
- First valid ID time
- Schedule lock time
- Estimated `T`, damping, `omega_n`, and `condB`
- Abort/no-estimate rate

## Procedure
1. Home the gantry.
2. Verify `/payload/pose_e_rel` is publishing from the gimbal encoder.
3. Set physical rope length for the block.
4. Reset payload origin at the start position.
5. Let payload settle.
6. Run each trial in the campaign sequence.
7. Return to the same start position between trials.
8. Do not touch the payload during the residual window.
9. Add anomalies to `operator_notes.md`.

## Fairness Rules
Compare methods only when rope length, target distance, speed, payload mass, start pose, and repeat count match. If `K` differs between IS1 and IS2, label the comparison as exploratory, not fair.

## Paper Figures
Use:

- Overlay of command velocity, cart travel, payload swing, and residual swing
- Residual RMS/p2p bar charts by method and rope length
- Travel error by method and rope length
- Schedule `T` and ID time by method
- Residual RMS boxplot across repeats

