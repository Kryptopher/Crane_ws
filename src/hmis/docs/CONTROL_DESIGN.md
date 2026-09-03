# HMIS control design

## Objective

Let the human specify the path and speed in real time, subject to normal
workspace and velocity limits, while reducing excitation of the suspended
payload's dominant pendulum mode.

The operator command is a 2-D velocity signal `u_h(t)`. HMIS applies the same
causal impulse shaper independently to X and Y:

```text
u_s(t) = sum(A_i * u_h(t - t_i))
```

For a payload natural frequency `f_n` and damping ratio `zeta`:

```text
omega_n = 2*pi*f_n
omega_d = omega_n*sqrt(1-zeta^2)
T_d     = pi/omega_d
K       = exp(-pi*zeta/sqrt(1-zeta^2))
```

The robust ZVD shaper is:

```text
times      = [0, T_d, 2*T_d]
amplitudes = [1, 2K, K^2] / (1 + K)^2
```

The amplitudes sum to one, so steady stick input eventually produces the
requested speed. ZVD is the default because it is less sensitive than ZV to a
small frequency-model error.

## What v0.1 does

- Shapes the complete, continuously changing stick signal, including diagonal
  motion and reversals.
- Bounds stick speed and vector acceleration before shaping.
- Uses `/traj_cmd` command 6 (`STREAM`) already supported by `gantry_control`.
- Exposes human and shaped command topics for time-aligned experiment logging.
- Fails closed on stale joystick/state data, deadman release, wrong mode,
  disabled motors, incomplete homing, or E-stop.

## Intent versus sway tradeoff

No causal input shaper can both cancel a known flexible mode and reproduce all
human input with zero delay. HMIS responds immediately with the first impulse,
then completes the requested command over the shaper horizon. ZVD uses a full
mode period; ZV uses half a mode period but is less robust.

The safety override is deliberately outside the shaper. Deadman release clears
all delayed motion immediately. Normal low-sway stopping is performed by
centering the stick while continuing to hold LB until `/hmis/shaped_cmd` is
zero.

## Relationship to the current controller

`ZV_JOG` in `gantry_controller.cpp` is a useful proof of concept, but it is a
state machine for one start and one stop. At activation it quantizes the input
to the dominant axis and stores only its sign. Subsequent magnitude and
direction changes are not represented. HMIS replaces that event-level behavior
with convolution of the entire live signal and stays separate from the
hardware controller.

## Next control layers

The first experimental layer is a precision-stop controller. At stick neutral,
it captures the current cart position and controls the state
`[cart position error, cart velocity, payload angle, payload angle rate]` back
to zero. This replaces the same-direction ZVD stop tail with bounded braking
and local corrective motion. It remains simulation-only until its excursion,
sensor-freshness, and fallback behavior pass the staged test plan.

After that validation, measured residual damping from `/payload/pose_e_rel` can
be integrated into the live HMIS state machine behind an explicit enable
parameter. Frequency adaptation should update model parameters only while the
controller is inactive; changing them during active motion can create a
command discontinuity.
