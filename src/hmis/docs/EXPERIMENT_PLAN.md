# HMIS staged experiment plan

## Metrics

Record these for both ordinary `JOG` and HMIS:

- Peak payload displacement relative to the cart.
- Residual RMS displacement over two periods after the shaped command settles.
- Time to complete an operator task.
- Cart path length and final position error.
- Human command versus shaped-command lag.
- Number of deadman releases, timeouts, saturations, and E-stops.

Use the same operator task, speed limit, rope length, payload, and starting
position for paired runs.

## Gate 1 — software only

1. Run the unit tests.
2. Run `simulate_hmis` at the nominal frequency.
3. Sweep the simulated true frequency by at least +/-10% while leaving HMIS at
   the nominal value.
4. Verify that deadman/reset tests leave no delayed command in the history.

Exit criterion: no command exceeds configured speed, all safety tests pass,
and nominal residual RMS is lower than direct manual input.

## Gate 2 — motors disabled

1. Launch the real joystick, gantry state publisher, and HMIS.
2. Inspect `/hmis/human_cmd`, `/hmis/shaped_cmd`, and `/hmis/status`.
3. Exercise circles, reversals, slow centering, deadman release, joystick
   disconnect, wrong mode, stale state, and RB E-stop.

Exit criterion: command directions and units are correct, and every safety
case produces immediate zero.

## Gate 3 — unloaded cart

Start at 20 mm/s and a conservative acceleration limit. Confirm workspace
clamping and operator controls before attaching the payload. HMIS should never
be the only layer enforcing workspace limits; the gantry controller remains
authoritative.

## Gate 4 — payload, single axis

Use X only, a verified frequency, and short travel near workspace center. Run
at least ten paired JOG/HMIS trials. Center the stick while holding LB for the
normal HMIS stop; separately test deadman release as the safety-stop case.

Exit criterion: repeatable residual-sway reduction without unacceptable task
delay or position drift.

## Gate 5 — 2-D operator task

Add Y and diagonal moves, then repeat paired trials across rope lengths and
payload masses. Freeze v0.1 parameters during each run. Only after these data
are stable should closed-loop residual damping or online frequency adaptation
be enabled.
