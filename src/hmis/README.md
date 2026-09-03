# HMIS — Human-Machine Input Shaping

HMIS gives the operator continuous control of the gantry trajectory while a
real-time input shaper reduces payload sway. It does **not** play a predefined
CSV or trajectory profile.

The live signal path is:

```text
/joy -> deadzone + speed/acceleration limits -> ZVD convolution -> /traj_cmd STREAM
                                                        |-> /hmis/shaped_cmd
                    operator intent -> /hmis/human_cmd
```

Unlike the existing `ZV_JOG` mode, HMIS does not freeze the first stick
direction or reduce the stick to an on/off event. Every 2-D change in stick
magnitude and direction becomes part of the shaped command.

## Safety behavior

- HMIS never homes, changes the gantry mode, or enables motors automatically.
- It only sends motion when `/gantry/state` is fresh, homed, enabled, in
  `TRAJ`, and not E-stopped.
- LB (button 4) is the deadman. Releasing it immediately publishes zero and
  clears delayed commands.
- RB (button 5) immediately publishes zero and calls `/gantry/estop`.
- Joystick or gantry-state timeout immediately publishes zero.
- Keep LB held and smoothly center the stick for a shaped low-sway stop.
  Releasing LB is a safety stop, so it intentionally truncates the shaper and
  may leave some residual sway.

Input shaping is causal: a ZVD shaper has a visible response horizon of one
full oscillation period. With the default 0.5 Hz mode, its impulses occur at
0, 1, and 2 seconds. The first impulse responds immediately at reduced gain;
the remaining command authority arrives later.

## Build and test

```bash
cd ~/crane_ws
PYTHONPATH=src/hmis pytest -q src/hmis/test
colcon build --packages-select hmis
source install/setup.bash
```

Run the software-only benchmark:

```bash
ros2 run hmis simulate_hmis
ros2 run hmis simulate_hmis --csv /tmp/hmis_sim.csv
ros2 run hmis simulate_precision_stop
ros2 run hmis simulate_precision_stop --csv /tmp/hmis_precision_stop.csv
```

`simulate_precision_stop` starts at the instant the operator centers the stick
while the cart is moving. It compares the current ZVD tail, an immediate hard
stop, and an experimental full-state LQR precision stop. The precision-stop
controller is currently simulation-only and cannot command the gantry. Results
and hardware-entry criteria are recorded in `docs/PRECISION_STOP_TEST.md`.

## Rig bring-up

Start the normal gantry stack first. In a second terminal:

```bash
source ~/crane_ws/install/setup.bash
ros2 launch hmis hmis.launch.py
```

Then explicitly home and wait until `/gantry/state` reports `homed: true`:

```bash
ros2 service call /gantry/set_mode gantry_control/srv/SetMode "{mode: 'HOME'}"
ros2 topic echo /gantry/state --once
```

After homing has completed, select real-time trajectory mode and enable:

```bash
ros2 service call /gantry/set_mode gantry_control/srv/SetMode "{mode: 'TRAJ'}"
ros2 service call /gantry/enable std_srvs/srv/Trigger "{}"
```

Hold LB and use the left stick. Monitor the unshaped and shaped commands with:

```bash
ros2 topic echo /hmis/status
ros2 topic echo /hmis/human_cmd
ros2 topic echo /hmis/shaped_cmd
```

The diagnostic `Twist` topics use standard SI units (m/s). HMIS converts to the
mm/s expected by `TrajCmd` only at the controller interface.

Before moving hardware, replace `natural_frequency_hz` in `config/hmis.yaml`
with the identified payload sway frequency. See `docs/CONTROL_DESIGN.md` and
`docs/EXPERIMENT_PLAN.md` for the model, tradeoffs, and staged validation.
