import math

import pytest

from nonzero_ic_exciter import (
    BoundedCycleExciter,
    BoundedExciteConfig,
    ExciteCommand,
    ExciteConfig,
    NonzeroIcExciter,
)

GRAVITY = 9.80665
ROPE_LENGTH_M = 0.9
OMEGA = math.sqrt(GRAVITY / ROPE_LENGTH_M)


def _run(
    config: ExciteConfig,
    *,
    angle0_deg: float,
    rate0_deg_s: float = 0.0,
    rig_sign: float = 1.0,
    dt: float = 0.01,
    t_max: float = 45.0,
):
    """Drive an exact undamped cart-pendulum with the controller.

    State is ``[payload_position, payload_velocity, trolley_position]`` in mm;
    within each constant trolley-velocity step the payload-minus-trolley pair
    is an SHO, matching the propagation used by ``test_nonzero_ic_shaper``.
    ``rig_sign = -1`` emulates hardware whose angle sign opposes cart motion.
    """
    length_mm = ROPE_LENGTH_M * 1000.0
    payload_pos = length_mm * math.radians(angle0_deg)
    payload_vel = length_mm * math.radians(rate0_deg_s)
    trolley_pos = 0.0

    controller = NonzeroIcExciter(config)
    history = []
    t = 0.0
    last: ExciteCommand | None = None
    while t < t_max:
        angle_deg = math.degrees((payload_pos - trolley_pos) / length_mm)
        cmd = controller.update(t, angle_deg, cart_offset_mm=trolley_pos)
        last = cmd
        history.append((t, angle_deg, cmd))
        if cmd.abort_reason is not None or cmd.converged:
            break

        trolley_vel = rig_sign * cmd.velocity_mm_s
        rel_p = payload_pos - trolley_pos
        rel_v = payload_vel - trolley_vel
        phase = OMEGA * dt
        rp = rel_p * math.cos(phase) + rel_v * math.sin(phase) / OMEGA
        rv = -OMEGA * rel_p * math.sin(phase) + rel_v * math.cos(phase)
        trolley_pos += trolley_vel * dt
        payload_pos = trolley_pos + rp
        payload_vel = trolley_vel + rv
        t += dt

    return controller, last, history


def _base_config(**overrides) -> ExciteConfig:
    params = dict(
        target_angle_deg=21.0,
        omega_rad_s=OMEGA,
        tolerance_deg=1.5,
        band_deg=5.0,
        speed_mm_s=150.0,
        timeout_s=40.0,
        abort_angle_deg=32.0,
    )
    params.update(overrides)
    return ExciteConfig(**params)


def _run_bounded(
    config: BoundedExciteConfig,
    *,
    angle_noise_deg: float = 0.0,
    dt: float = 0.01,
    t_max: float = 35.0,
):
    """Run the bounded exciter against the same exact pendulum model."""
    length_mm = ROPE_LENGTH_M * 1000.0
    payload_pos = 0.0
    payload_vel = 0.0
    trolley_pos = 0.0
    controller = BoundedCycleExciter(config)
    history = []
    t = 0.0
    last = None
    sample_index = 0
    while t < t_max:
        angle_deg = math.degrees((payload_pos - trolley_pos) / length_mm)
        if angle_noise_deg:
            angle_deg += angle_noise_deg if sample_index % 2 else -angle_noise_deg
        last = controller.update(t, angle_deg, cart_offset_mm=trolley_pos)
        history.append((t, angle_deg, trolley_pos, last))
        if last.abort_reason is not None or last.converged:
            break

        trolley_vel = last.velocity_mm_s
        rel_p = payload_pos - trolley_pos
        rel_v = payload_vel - trolley_vel
        phase = OMEGA * dt
        rp = rel_p * math.cos(phase) + rel_v * math.sin(phase) / OMEGA
        rv = -OMEGA * rel_p * math.sin(phase) + rel_v * math.cos(phase)
        trolley_pos += trolley_vel * dt
        payload_pos = trolley_pos + rp
        payload_vel = trolley_vel + rv
        t += dt
        sample_index += 1
    return controller, last, history


def _bounded_config(**overrides) -> BoundedExciteConfig:
    params = dict(
        target_angle_deg=12.0,
        omega_rad_s=OMEGA,
        tolerance_deg=1.5,
        speed_mm_s=100.0,
        initial_excursion_mm=15.0,
        excursion_step_mm=10.0,
        max_excursion_mm=100.0,
        position_kp_s=2.0,
        return_speed_mm_s=30.0,
        return_tolerance_mm=1.0,
        settle_cycles=1.0,
        timeout_s=30.0,
        abort_angle_deg=22.0,
        slew_mm_s2=600.0,
    )
    params.update(overrides)
    return BoundedExciteConfig(**params)


def test_bounded_exciter_converges_and_returns_to_anchor():
    config = _bounded_config()
    _, last, history = _run_bounded(config)

    assert last is not None and last.converged
    assert last.abort_reason is None
    assert abs(history[-1][2]) <= config.return_tolerance_mm
    assert last.amplitude_est_deg >= config.target_angle_deg - config.tolerance_deg


def test_bounded_exciter_corrects_small_return_overshoot_past_anchor():
    config = _bounded_config(return_tolerance_mm=1.0)
    controller = BoundedCycleExciter(config)
    controller.update(0.0, 0.0, cart_offset_mm=0.0)
    controller._returning = True
    controller._phase = 'return'
    controller._u_prev = 0.0

    command = controller.update(0.01, 0.0, cart_offset_mm=-1.08)

    assert command.abort_reason is None
    assert command.velocity_mm_s > 0.0


def test_bounded_exciter_never_commands_negative_position_excursion():
    config = _bounded_config()
    _, _, history = _run_bounded(config)
    positions = [row[2] for row in history]

    assert min(positions) >= -0.25
    assert max(positions) <= config.max_excursion_mm + 0.25


def test_bounded_exciter_respects_speed_and_uses_slow_cycle_reversals():
    config = _bounded_config()
    _, _, history = _run_bounded(config)
    commands = [row[3].velocity_mm_s for row in history]
    times = [row[0] for row in history]

    assert max(abs(value) for value in commands) <= config.speed_mm_s + 1.0e-6
    reversals = []
    last_sign = 0
    for t, value in zip(times, commands):
        sign = 0 if abs(value) < 2.0 else (1 if value > 0.0 else -1)
        if sign and last_sign and sign != last_sign:
            reversals.append(t)
        if sign:
            last_sign = sign
    assert len(reversals) >= 2
    assert all(
        later - earlier >= 0.35 * math.pi / config.omega_rad_s
        for earlier, later in zip(reversals, reversals[1:])
    )


def test_bounded_exciter_encoder_noise_does_not_create_command_chatter():
    config = _bounded_config(target_angle_deg=14.0, abort_angle_deg=25.0)
    _, _, history = _run_bounded(config, angle_noise_deg=0.25)
    commands = [row[3].velocity_mm_s for row in history]
    signs = []
    for value in commands:
        sign = 0 if abs(value) < 2.0 else (1 if value > 0.0 else -1)
        if sign and (not signs or sign != signs[-1]):
            signs.append(sign)

    # Several physical half-cycle reversals are expected; sample-to-sample
    # alternation from differentiated encoder noise is not.
    assert len(signs) < 20


def test_swings_up_from_near_rest_to_target_without_large_overshoot():
    config = _base_config()
    _, last, history = _run(config, angle0_deg=1.5)

    assert last is not None and last.converged, 'controller never reported convergence'
    assert last.abort_reason is None
    assert abs(last.amplitude_est_deg - config.target_angle_deg) <= 2.0 * config.tolerance_deg

    peak_angle = max(abs(angle) for _, angle, _ in history)
    assert peak_angle <= config.target_angle_deg + config.band_deg + 2.0


def test_hands_off_at_a_swing_extremum():
    _, last, _ = _run(_base_config(), angle0_deg=2.0)

    assert last is not None and last.converged
    # Near-zero rate at hand-off means the payload energy is all potential, so
    # stopping the trolley injects a minimal transient into the ID window.
    assert abs(last.angle_rate_deg_s) <= _base_config().peak_velocity_deg_s


def test_removes_energy_when_started_above_target():
    config = _base_config(target_angle_deg=14.0, abort_angle_deg=40.0)
    _, last, history = _run(config, angle0_deg=27.0)

    assert last is not None and last.converged
    assert last.amplitude_est_deg == pytest.approx(
        config.target_angle_deg, abs=2.0 * config.tolerance_deg)
    # It must have actively pulled the envelope down, not merely coasted.
    assert history[0][2].amplitude_est_deg > config.target_angle_deg + 5.0


def test_auto_calibrates_a_reversed_drive_sign():
    config = _base_config(drive_sign=1.0, auto_calibrate_sign=True)
    controller, last, _ = _run(config, angle0_deg=1.5, rig_sign=-1.0)

    assert last is not None and last.converged
    assert controller.drive_sign == -1.0


def test_wrong_sign_without_auto_calibration_times_out_cleanly():
    config = _base_config(drive_sign=1.0, auto_calibrate_sign=False, timeout_s=8.0)
    _, last, _ = _run(config, angle0_deg=1.5, rig_sign=-1.0)

    assert last is not None
    assert last.converged is False
    assert last.abort_reason is not None
    assert 'converge' in last.abort_reason


def test_command_never_exceeds_configured_speed():
    config = _base_config()
    _, _, history = _run(config, angle0_deg=1.5)

    assert history, 'no control ticks were recorded'
    assert all(abs(cmd.velocity_mm_s) <= config.speed_mm_s + 1.0e-6 for _, _, cmd in history)


def test_coasts_when_the_cart_leaves_its_travel_budget():
    config = _base_config(travel_budget_mm=120.0)
    controller = NonzeroIcExciter(config)
    controller.update(0.0, 5.0, cart_offset_mm=0.0)
    controller.update(0.01, 6.0, cart_offset_mm=0.0)
    outside = controller.update(0.02, 7.0, cart_offset_mm=200.0)

    assert outside.phase == 'coast'
    # The command is slewed toward zero rather than snapped, but it is strictly
    # decaying and never a fresh drive.
    assert abs(outside.velocity_mm_s) < config.speed_mm_s


def test_holds_last_command_through_a_non_finite_sample():
    config = _base_config()
    controller = NonzeroIcExciter(config)
    controller.update(0.0, 5.0)
    good = controller.update(0.01, 6.0)
    stale = controller.update(0.02, float('nan'))

    assert stale.phase == 'stale'
    assert stale.velocity_mm_s == pytest.approx(good.velocity_mm_s)


@pytest.mark.parametrize(
    'override',
    [
        {'target_angle_deg': 0.0},
        {'omega_rad_s': -1.0},
        {'band_deg': 0.1, 'tolerance_deg': 1.0},
        {'abort_angle_deg': 10.0, 'target_angle_deg': 21.0},
        {'drive_sign': 0.0},
        {'speed_mm_s': float('nan')},
    ],
)
def test_rejects_invalid_configuration(override):
    params = dict(target_angle_deg=21.0, omega_rad_s=OMEGA)
    params.update(override)
    with pytest.raises(ValueError):
        ExciteConfig(**params)
