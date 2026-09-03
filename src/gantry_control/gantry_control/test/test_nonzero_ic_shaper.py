import math
import sys
import time
from types import MethodType, SimpleNamespace

import pytest
from std_msgs.msg import Float64MultiArray, MultiArrayDimension

from adaptive_paper_tdf_player import (
    AdaptivePaperTdfPlayer,
    GantryStateSample,
    TimedPayloadObservation,
    actuator_compensated_profile_start,
    check_args,
    closer_timed_payload_observation,
    payload_time_at_wall,
    payload_wall_at_time,
    parse_args,
    scaled_nonzero_ic_frequency,
    select_nonzero_ic_shaper_frequency,
    update_payload_clock_calibration,
)
from nonzero_ic_exciter import ExciteCommand
from nonzero_ic_shaper import (
    FreeSwingFrequencyEstimate,
    correct_finite_amplitude_frequency,
    estimate_free_swing_frequency,
    solve_nonzero_ic_shaper,
    solve_robust_nonzero_ic_shaper,
)


class _RecordingLogger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(message)

    def warn(self, message):
        self.messages.append(message)

    def error(self, message):
        self.messages.append(message)


def test_frequency_scale_does_not_modify_the_fitted_frequency_value():
    fitted_omega = 3.2242

    assert scaled_nonzero_ic_frequency(fitted_omega, 0.99) == pytest.approx(
        3.191958)
    assert scaled_nonzero_ic_frequency(fitted_omega, 1.00) == pytest.approx(
        fitted_omega)
    assert scaled_nonzero_ic_frequency(fitted_omega, 1.01) == pytest.approx(
        3.256442)
    assert fitted_omega == pytest.approx(3.2242)


def test_fixed_shaper_frequency_overrides_scale_without_changing_fit():
    fitted_omega = 3.2242

    selected = select_nonzero_ic_shaper_frequency(
        fitted_omega_rad_s=fitted_omega,
        scale=1.0,
        fixed_omega_rad_s=3.25,
    )

    assert selected == pytest.approx(3.25)
    assert fitted_omega == pytest.approx(3.2242)


def test_finite_amplitude_correction_is_identity_at_zero_amplitude():
    corrected = correct_finite_amplitude_frequency(
        finite_amplitude_omega_rad_s=3.27,
        amplitude_mm=0.0,
        rope_length_m=0.9017,
    )

    assert corrected.amplitude_angle_rad == 0.0
    assert corrected.correction_factor == 1.0
    assert corrected.small_angle_omega_rad_s == pytest.approx(3.27)


@pytest.mark.parametrize(
    ('amplitude_deg', 'finite_omega', 'expected_small_angle_omega'),
    [
        (7.027629, 3.267464110759542, 3.2705390586721155),
        (11.888572, 3.2616268558019414, 3.270425208444327),
        (17.685630, 3.2514111915934283, 3.270879435106458),
    ],
)
def test_finite_amplitude_correction_matches_measured_rope_sweep(
    amplitude_deg,
    finite_omega,
    expected_small_angle_omega,
):
    rope_length_m = 0.9017
    amplitude_mm = 1000.0 * rope_length_m * math.sin(
        math.radians(amplitude_deg))

    corrected = correct_finite_amplitude_frequency(
        finite_amplitude_omega_rad_s=finite_omega,
        amplitude_mm=amplitude_mm,
        rope_length_m=rope_length_m,
    )

    assert math.degrees(corrected.amplitude_angle_rad) == pytest.approx(
        amplitude_deg)
    assert corrected.small_angle_omega_rad_s == pytest.approx(
        expected_small_angle_omega, abs=1.0e-12)


@pytest.mark.parametrize(
    ('omega', 'amplitude_mm', 'rope_length_m'),
    [
        (0.0, 100.0, 0.9017),
        (3.27, float('nan'), 0.9017),
        (3.27, 100.0, 0.0),
        (3.27, 901.7, 0.9017),
    ],
)
def test_finite_amplitude_correction_rejects_invalid_inputs(
    omega,
    amplitude_mm,
    rope_length_m,
):
    with pytest.raises(ValueError):
        correct_finite_amplitude_frequency(
            finite_amplitude_omega_rad_s=omega,
            amplitude_mm=amplitude_mm,
            rope_length_m=rope_length_m,
        )


def test_actuation_lead_advances_profile_relative_to_fitted_peak():
    fitted_peak_wall = 100.500

    assert actuator_compensated_profile_start(
        fitted_peak_wall, 0.0) == pytest.approx(100.500)
    assert actuator_compensated_profile_start(
        fitted_peak_wall, 25.0) == pytest.approx(100.475)


def test_payload_clock_calibration_separates_clock_offset_and_queue_delay():
    offset, delay, reset = update_payload_clock_calibration(
        None, None, payload_time_s=20.0, receive_wall_s=100.0)
    assert offset == pytest.approx(80.0)
    assert delay == pytest.approx(0.0)
    assert reset is False

    offset, delay, reset = update_payload_clock_calibration(
        offset, 20.0, payload_time_s=20.010, receive_wall_s=100.012)
    assert offset == pytest.approx(80.0)
    assert delay == pytest.approx(0.002)
    assert reset is False

    # A cleaner observation lowers the calibrated offset; later excess is
    # measured relative to that floor rather than hidden as sample freshness.
    offset, delay, _ = update_payload_clock_calibration(
        offset, 20.010, payload_time_s=20.020, receive_wall_s=100.019)
    assert offset == pytest.approx(79.999)
    assert delay == pytest.approx(0.0)
    offset, delay, _ = update_payload_clock_calibration(
        offset, 20.020, payload_time_s=20.030, receive_wall_s=100.033)
    assert delay == pytest.approx(0.004)


def test_payload_clock_calibration_restarts_after_source_clock_reset():
    offset, delay, reset = update_payload_clock_calibration(
        80.0, 20.0, payload_time_s=0.2, receive_wall_s=101.0)

    assert offset == pytest.approx(100.8)
    assert delay == pytest.approx(0.0)
    assert reset is True


def test_payload_clock_mapping_is_invertible():
    assert payload_time_at_wall(100.25, 80.0) == pytest.approx(20.25)
    assert payload_wall_at_time(20.25, 80.0) == pytest.approx(100.25)


def test_nearest_measured_peak_observation_is_kept_separate_from_prediction():
    first = closer_timed_payload_observation(
        None,
        sample_time_s=10.008,
        target_time_s=10.0,
        swing_mm=250.0,
        world_payload_velocity_mm_s=-21.0,
    )
    closer = closer_timed_payload_observation(
        first,
        sample_time_s=9.997,
        target_time_s=10.0,
        swing_mm=252.0,
        world_payload_velocity_mm_s=4.0,
    )
    farther = closer_timed_payload_observation(
        closer,
        sample_time_s=10.006,
        target_time_s=10.0,
        swing_mm=251.0,
        world_payload_velocity_mm_s=-16.0,
    )

    assert isinstance(first, TimedPayloadObservation)
    assert closer.offset_ms == pytest.approx(-3.0)
    assert closer.world_payload_velocity_mm_s == pytest.approx(4.0)
    assert farther is closer


def test_payload_callback_diagnostic_combines_relative_and_cart_velocity():
    player = SimpleNamespace(
        args=SimpleNamespace(axis='x'),
        latest_payload_time=20.004,
        latest_swing_mm=240.0,
        latest_payload_rel_vx_mm_s=-35.0,
        latest_payload_rel_vy_mm_s=0.0,
        timed_profile_start_payload_time=20.000,
        timed_profile_peak_payload_time=20.020,
        nonzero_ic_measured_command_start=None,
        nonzero_ic_measured_peak=None,
    )
    cart = GantryStateSample(
        rx_wall=100.0,
        stamp=100.0,
        x=0.1,
        y=0.0,
        vx=0.012,
        vy=0.0,
    )

    AdaptivePaperTdfPlayer._capture_timed_payload_observations(player, cart)

    assert player.nonzero_ic_measured_command_start.offset_ms == pytest.approx(4.0)
    assert (
        player.nonzero_ic_measured_command_start.world_payload_velocity_mm_s
        == pytest.approx(-23.0)
    )
    assert player.nonzero_ic_measured_peak.offset_ms == pytest.approx(-16.0)


def test_payload_callback_diagnostic_prefers_unfiltered_relative_velocity():
    player = SimpleNamespace(
        args=SimpleNamespace(axis='x'),
        latest_payload_time=20.0,
        latest_swing_mm=240.0,
        latest_payload_rel_vx_mm_s=55.0,
        latest_payload_rel_vy_mm_s=0.0,
        latest_payload_rel_raw_vx_mm_s=3.0,
        latest_payload_rel_raw_vy_mm_s=0.0,
        timed_profile_start_payload_time=20.0,
        timed_profile_peak_payload_time=None,
        nonzero_ic_measured_command_start=None,
        nonzero_ic_measured_peak=None,
    )
    cart = GantryStateSample(
        rx_wall=100.0,
        stamp=100.0,
        x=0.1,
        y=0.0,
        vx=0.002,
        vy=0.0,
    )

    AdaptivePaperTdfPlayer._capture_timed_payload_observations(player, cart)

    assert (
        player.nonzero_ic_measured_command_start.world_payload_velocity_mm_s
        == pytest.approx(5.0)
    )


def test_free_swing_fit_recovers_frequency_and_terminal_state():
    omega = math.sqrt(9.80665 / 0.9)
    amplitude_mm = 72.0
    phase_rad = 0.63
    offset_mm = -3.5
    samples = []
    for index in range(501):
        time_s = index * 0.01
        deterministic_noise_mm = 0.35 * math.sin(17.0 * time_s)
        position_mm = (
            offset_mm
            + amplitude_mm * math.cos(omega * time_s + phase_rad)
            + deterministic_noise_mm
        )
        samples.append((time_s, position_mm))

    estimate = estimate_free_swing_frequency(
        samples,
        omega_min_rad_s=2.6,
        omega_max_rad_s=4.0,
        grid_count=401,
    )
    fitted_position, fitted_velocity = estimate.oscillation_state_at(5.0)
    expected_position = amplitude_mm * math.cos(omega * 5.0 + phase_rad)
    expected_velocity = -amplitude_mm * omega * math.sin(
        omega * 5.0 + phase_rad
    )

    assert estimate.omega_n_rad_s == pytest.approx(omega, abs=2.0e-4)
    assert estimate.amplitude_mm == pytest.approx(amplitude_mm, abs=0.1)
    assert estimate.offset_mm == pytest.approx(offset_mm, abs=0.1)
    assert estimate.normalized_rmse < 0.01
    assert fitted_position == pytest.approx(expected_position, abs=0.1)
    assert fitted_velocity == pytest.approx(expected_velocity, abs=0.5)


def test_free_swing_fit_rejects_an_insufficient_window():
    with pytest.raises(ValueError, match='at least 6'):
        estimate_free_swing_frequency(
            [(0.0, 1.0), (0.1, 2.0)],
            omega_min_rad_s=2.5,
            omega_max_rad_s=4.2,
        )


def test_matches_supplied_matlab_example():
    shaper = solve_nonzero_ic_shaper(
        initial_swing_mm=100.0,
        initial_payload_velocity_mm_s=20.0,
        maximum_speed_mm_s=240.0,
        omega_n_rad_s=math.pi,
        move_duration_s=600.0 / 240.0,
    )

    assert shaper.A0 == pytest.approx(0.2300448389348808, abs=1.0e-12)
    assert shaper.A1 == pytest.approx(0.7699551610651192, abs=1.0e-12)
    assert shaper.is_forward_only
    assert shaper.switch_time_s == pytest.approx(
        1.7069813556924764, abs=1.0e-12
    )
    assert shaper.terminal_position_residual_mm == pytest.approx(0.0, abs=1.0e-11)
    assert shaper.terminal_velocity_residual_mm_s == pytest.approx(0.0, abs=1.0e-11)


def test_command_has_expected_four_intervals_and_exact_displacement():
    vmax = 240.0
    shaper = solve_nonzero_ic_shaper(
        initial_swing_mm=100.0,
        initial_payload_velocity_mm_s=20.0,
        maximum_speed_mm_s=vmax,
        omega_n_rad_s=math.pi,
        move_duration_s=2.5,
    )

    assert shaper.gain_at(0.5 * shaper.switch_time_s) == pytest.approx(shaper.A0)
    assert shaper.gain_at(
        0.5 * (shaper.switch_time_s + shaper.move_duration_s)
    ) == pytest.approx(1.0)
    assert shaper.gain_at(
        shaper.move_duration_s + 0.5 * shaper.switch_time_s
    ) == pytest.approx(shaper.A1)
    assert shaper.gain_at(shaper.duration_s) == 0.0

    displacement_mm = vmax * (
        shaper.A0 * shaper.switch_time_s
        + (shaper.move_duration_s - shaper.switch_time_s)
        + shaper.A1 * shaper.switch_time_s
    )
    assert displacement_mm == pytest.approx(600.0, abs=1.0e-10)


def test_robust_nonzero_ic_schedule_is_forward_only_exact_and_improves_band():
    vmax = 750.0
    move_duration = 1100.0 / vmax
    shaper = solve_robust_nonzero_ic_shaper(
        initial_swing_mm=287.293,
        initial_payload_velocity_mm_s=0.0,
        maximum_speed_mm_s=vmax,
        omega_n_rad_s=3.271088,
        move_duration_s=move_duration,
        frequency_band_fraction=0.05,
    )

    assert sum(shaper.start_amplitudes) == pytest.approx(1.0, abs=1e-10)
    assert sum(shaper.stop_amplitudes) == pytest.approx(1.0, abs=1e-10)
    assert sum(
        amplitude * delay
        for amplitude, delay in zip(
            shaper.start_amplitudes, shaper.impulse_times_s
        )
    ) == pytest.approx(sum(
        amplitude * delay
        for amplitude, delay in zip(
            shaper.stop_amplitudes, shaper.impulse_times_s
        )
    ), abs=1e-10)
    assert shaper.terminal_position_residual_mm == pytest.approx(0.0, abs=1e-7)
    assert shaper.terminal_velocity_residual_mm_s == pytest.approx(0.0, abs=1e-7)
    assert (
        shaper.worst_case_residual_fraction
        < shaper.baseline_worst_case_residual_fraction
    )

    events = shaper.gain_events()
    assert events[0][0] == pytest.approx(0.0)
    assert events[-1] == pytest.approx((shaper.duration_s, 0.0))
    assert all(0.0 <= gain <= 1.0 for _, gain in events)
    displacement = 0.0
    prior_time = 0.0
    prior_gain = 0.0
    for event_time, gain in events:
        displacement += vmax * prior_gain * (event_time - prior_time)
        prior_time = event_time
        prior_gain = gain
    assert displacement == pytest.approx(1100.0, abs=1e-7)


@pytest.mark.parametrize('band', [0.0, 0.25, float('nan')])
def test_robust_nonzero_ic_rejects_invalid_frequency_band(band):
    with pytest.raises(ValueError, match='band fraction'):
        solve_robust_nonzero_ic_shaper(
            initial_swing_mm=287.293,
            initial_payload_velocity_mm_s=0.0,
            maximum_speed_mm_s=750.0,
            omega_n_rad_s=3.271088,
            move_duration_s=1100.0 / 750.0,
            frequency_band_fraction=band,
        )


def test_piecewise_crane_dynamics_finish_at_rest_over_the_trolley():
    omega = math.pi
    vmax = 240.0
    shaper = solve_nonzero_ic_shaper(
        initial_swing_mm=100.0,
        initial_payload_velocity_mm_s=20.0,
        maximum_speed_mm_s=vmax,
        omega_n_rad_s=omega,
        move_duration_s=2.5,
    )

    # Exact propagation of x_ddot=-omega^2*(x-q), q_dot=u for each
    # constant-velocity interval.  Payload position/velocity and trolley
    # position remain continuous when the commanded velocity switches.
    payload_position = 100.0
    payload_velocity = 20.0
    trolley_position = 0.0
    segments = (
        (shaper.switch_time_s, shaper.A0 * vmax),
        (shaper.move_duration_s - shaper.switch_time_s, vmax),
        (shaper.switch_time_s, shaper.A1 * vmax),
    )
    for duration, trolley_velocity in segments:
        relative_position = payload_position - trolley_position
        relative_velocity = payload_velocity - trolley_velocity
        phase = omega * duration
        next_relative_position = (
            relative_position * math.cos(phase)
            + relative_velocity * math.sin(phase) / omega
        )
        next_relative_velocity = (
            -omega * relative_position * math.sin(phase)
            + relative_velocity * math.cos(phase)
        )
        trolley_position += trolley_velocity * duration
        payload_position = trolley_position + next_relative_position
        payload_velocity = trolley_velocity + next_relative_velocity

    assert trolley_position == pytest.approx(600.0, abs=1.0e-10)
    assert payload_position == pytest.approx(trolley_position, abs=1.0e-10)
    assert payload_velocity == pytest.approx(0.0, abs=1.0e-10)


def test_zero_initial_state_reduces_to_standard_zv():
    shaper = solve_nonzero_ic_shaper(
        initial_swing_mm=0.0,
        initial_payload_velocity_mm_s=0.0,
        maximum_speed_mm_s=240.0,
        omega_n_rad_s=math.pi,
        move_duration_s=2.5,
    )

    assert (shaper.A0, shaper.A1) == pytest.approx((0.5, 0.5))
    assert shaper.switch_time_s == pytest.approx(1.0)
    assert shaper.terminal_position_residual_mm == pytest.approx(0.0, abs=1.0e-12)
    assert shaper.terminal_velocity_residual_mm_s == pytest.approx(0.0, abs=1.0e-12)


def test_signed_coefficients_are_kept_when_within_command_speed_bound():
    shaper = solve_nonzero_ic_shaper(
        initial_swing_mm=-150.0,
        initial_payload_velocity_mm_s=0.0,
        maximum_speed_mm_s=240.0,
        omega_n_rad_s=math.pi,
        move_duration_s=2.5,
        maximum_absolute_gain=400.0 / 240.0,
    )

    assert shaper.A0 < 0.0
    assert shaper.A1 > 1.0
    assert not shaper.is_forward_only
    assert max(abs(shaper.A0), abs(shaper.A1)) * 240.0 < 400.0
    assert shaper.terminal_position_residual_mm == pytest.approx(0.0, abs=1.0e-11)
    assert shaper.terminal_velocity_residual_mm_s == pytest.approx(0.0, abs=1.0e-11)


def test_signed_solution_is_rejected_only_when_hardware_bound_is_exceeded():
    with pytest.raises(ValueError, match='command-speed bound'):
        solve_nonzero_ic_shaper(
            initial_swing_mm=150.0,
            initial_payload_velocity_mm_s=0.0,
            maximum_speed_mm_s=240.0,
            omega_n_rad_s=math.pi,
            move_duration_s=2.5,
            maximum_absolute_gain=400.0 / 240.0,
        )


def test_negative_world_move_uses_direction_transformed_initial_state():
    positive = solve_nonzero_ic_shaper(
        initial_swing_mm=100.0,
        initial_payload_velocity_mm_s=20.0,
        maximum_speed_mm_s=240.0,
        omega_n_rad_s=math.pi,
        move_duration_s=2.5,
    )
    negative_world = solve_nonzero_ic_shaper(
        initial_swing_mm=(-1.0) * (-100.0),
        initial_payload_velocity_mm_s=(-1.0) * (-20.0),
        maximum_speed_mm_s=240.0,
        omega_n_rad_s=math.pi,
        move_duration_s=2.5,
    )

    assert negative_world == positive


def test_rejects_a_switch_after_the_unshaped_move():
    with pytest.raises(ValueError, match='switch time'):
        solve_nonzero_ic_shaper(
            initial_swing_mm=100.0,
            initial_payload_velocity_mm_s=20.0,
            maximum_speed_mm_s=240.0,
            omega_n_rad_s=math.pi,
            move_duration_s=1.0,
        )


@pytest.mark.parametrize(
    'override',
    [
        {'maximum_speed_mm_s': 0.0},
        {'omega_n_rad_s': -1.0},
        {'move_duration_s': float('nan')},
    ],
)
def test_rejects_invalid_inputs(override):
    arguments = {
        'initial_swing_mm': 100.0,
        'initial_payload_velocity_mm_s': 20.0,
        'maximum_speed_mm_s': 240.0,
        'omega_n_rad_s': math.pi,
        'move_duration_s': 2.5,
    }
    arguments.update(override)
    with pytest.raises(ValueError):
        solve_nonzero_ic_shaper(**arguments)


def test_nonzero_profile_uses_state_capture_time_not_pre_fit_timer_time():
    class Logger:
        def info(self, _message):
            pass

    player = SimpleNamespace(
        latest_payload_wall=10.0,
        args=SimpleNamespace(payload_fresh_timeout=0.25, excite=False),
        phase='wait_peak',
        wall_start=None,
        latest_cart_q_mm=50.0,
        final_zero_wall=1.0,
        residual_samples=[(0.0, 1.0)],
        get_logger=lambda: Logger(),
    )
    player._configure_nonzero_ic_schedule = lambda: 10.085

    started = AdaptivePaperTdfPlayer._maybe_begin_nonzero_ic_profile(
        player,
        now=10.0,
    )

    assert started
    assert player.wall_start == pytest.approx(10.085)
    assert player.start_cart_q_mm == pytest.approx(50.0)
    assert player.residual_samples == []


@pytest.mark.parametrize(
    ('direction', 'expected_wall'),
    ((1.0, 102.0), (-1.0, 101.0)),
)
def test_controller_timed_start_selects_peak_in_direction_of_travel(
    direction,
    expected_wall,
):
    estimate = FreeSwingFrequencyEstimate(
        omega_n_rad_s=math.pi,
        offset_mm=0.0,
        cosine_coefficient_mm=100.0,
        sine_coefficient_mm=0.0,
        amplitude_mm=100.0,
        rmse_mm=0.1,
        normalized_rmse=0.001,
        reference_time_s=0.0,
        window_duration_s=5.0,
        sample_count=501,
    )
    player = SimpleNamespace(
        latest_payload_time=0.0,
        latest_payload_wall=100.0,
        payload_clock_offset_s=100.0,
        direction=direction,
    )

    start_wall = AdaptivePaperTdfPlayer._next_positive_peak_wall(
        player,
        estimate,
        earliest_wall=100.25,
    )

    assert start_wall == pytest.approx(expected_wall)


def test_controller_profile_lead_does_not_change_fitted_peak_state():
    calls = {}

    class Logger:
        def info(self, _message):
            pass

    player = SimpleNamespace(
        latest_payload_wall=10.0,
        latest_payload_queue_delay_ms=0.5,
        args=SimpleNamespace(
            payload_fresh_timeout=0.25,
            excite=False,
            controller_timed_profile=True,
            timed_profile_lead_s=0.25,
            timed_profile_actuation_lead_ms=25.0,
            nonzero_ic_max_payload_queue_delay_ms=5.0,
        ),
        phase='wait_peak',
        get_logger=lambda: Logger(),
    )
    estimate = object()
    player._update_nonzero_ic_frequency_estimate = lambda: estimate

    def next_peak(received_estimate, earliest_wall):
        calls['next_peak'] = (received_estimate, earliest_wall)
        return 12.0

    player._next_positive_peak_wall = next_peak
    player._configure_nonzero_ic_schedule = lambda state_capture_wall_override: (
        calls.update(state_capture=state_capture_wall_override)
    )
    player._arm_controller_timed_profile = lambda start_wall, fitted_peak_wall: (
        calls.update(start=start_wall, peak=fitted_peak_wall)
    )

    started = AdaptivePaperTdfPlayer._maybe_begin_nonzero_ic_profile(
        player,
        now=10.0,
    )

    assert started is False
    assert calls['next_peak'][0] is estimate
    # Keep the configured service lead after subtracting the actuator advance.
    assert calls['next_peak'][1] == pytest.approx(10.275)
    # The state remains projected to the physical peak, while the controller
    # schedule is issued 25 ms earlier.
    assert calls['state_capture'] == pytest.approx(12.0)
    assert calls['peak'] == pytest.approx(12.0)
    assert calls['start'] == pytest.approx(11.975)


def test_controller_profile_waits_for_a_nonqueued_payload_sample():
    player = SimpleNamespace(
        latest_payload_wall=10.0,
        latest_payload_queue_delay_ms=38.0,
        args=SimpleNamespace(
            payload_fresh_timeout=0.25,
            excite=False,
            controller_timed_profile=True,
            nonzero_ic_max_payload_queue_delay_ms=5.0,
        ),
        phase='wait_peak',
        wall_start=None,
        _last_nonzero_ic_wait_log_wall=-math.inf,
        _last_nonzero_ic_wait_reason='',
        get_logger=lambda: _RecordingLogger(),
    )
    player._configure_nonzero_ic_schedule = lambda **_kwargs: pytest.fail(
        'a queued payload sample must not arm the profile')

    started = AdaptivePaperTdfPlayer._maybe_begin_nonzero_ic_profile(
        player,
        now=10.01,
    )

    assert started is False
    assert 'callback queue' in player._last_nonzero_ic_wait_reason


def test_nonzero_profile_publishes_first_command_in_locking_callback():
    published = []
    logged = []
    player = SimpleNamespace(
        done=False,
        start_requested=True,
        motion_started=True,
        _last_stream_wall=None,
        wall_start=None,
        run_start_wall=time.monotonic() - 5.0,
        phase='wait_peak',
        aborted=False,
        final_zero_wall=None,
        schedule=object(),
        args=SimpleNamespace(
            profile='nonzero-ic',
            axis='x',
            residual_window=10.0,
            print_period=0.25,
            excite=False,
        ),
        _last_id_status_print=0.0,
        _runtime_safety_reason=lambda _now: None,
    )
    player._publish_stream = lambda vx, vy: published.append((vx, vy))

    def begin(_now):
        player.wall_start = time.monotonic()
        return True

    player._maybe_begin_nonzero_ic_profile = begin
    player._axis_velocity_for_time = lambda _move_t: 358.0
    player._log_row = lambda move_t, vx, vy: logged.append((move_t, vx, vy))
    player._end_time = lambda: 2.0
    player._is_colleague_profile = lambda: False

    AdaptivePaperTdfPlayer._stream_timer_cb(player)

    assert published[0] == (0.0, 0.0)
    assert published[1] == (358.0, 0.0)
    assert len(logged) == 2
    assert -5.01 < logged[0][0] < -4.99
    assert logged[0][1:] == (0.0, 0.0)
    assert 0.0 <= logged[1][0] < 0.01


def test_encoder_diagnostics_are_decoded_from_publisher_labels():
    fields = (
        'time', 'arduino_ms', 'pitch_raw', 'roll_raw',
        'pitch_count', 'roll_count',
        'imu1_gx_dps', 'imu1_gy_dps', 'imu1_gz_dps',
        'imu2_gx_dps', 'imu2_gy_dps', 'imu2_gz_dps',
        'packet_age_ms', 'packet_seen', 'serial_lines', 'parse_errors',
        'stale', 'wrap_events',
    )
    msg = Float64MultiArray()
    dim = MultiArrayDimension()
    dim.label = ','.join(fields)
    dim.size = len(fields)
    dim.stride = len(fields)
    msg.layout.dim.append(dim)
    msg.data = [
        4.0, 4000.0, 10.0, 20.0, 30.0, 40.0,
        1.1, 1.2, 1.3, 2.1, 2.2, 2.3,
        7.5, 1.0, 1234.0, 2.0, 0.0, 3.0,
    ]
    player = SimpleNamespace(
        latest_enc_diag_wall=None,
        latest_enc_time=None,
        latest_enc_arduino_ms=None,
        latest_enc_pitch_raw=None,
        latest_enc_roll_raw=None,
        latest_enc_pitch_count=None,
        latest_enc_roll_count=None,
        latest_enc_imu1_ax=None,
        latest_enc_imu1_ay=None,
        latest_enc_imu1_az=None,
        latest_enc_imu1_gx=None,
        latest_enc_imu1_gy=None,
        latest_enc_imu1_gz=None,
        latest_enc_imu2_ax=None,
        latest_enc_imu2_ay=None,
        latest_enc_imu2_az=None,
        latest_enc_imu2_gx=None,
        latest_enc_imu2_gy=None,
        latest_enc_imu2_gz=None,
        latest_enc_packet_age_ms=None,
        latest_enc_packet_seen=None,
        latest_enc_serial_lines=None,
        latest_enc_parse_errors=None,
        latest_enc_stale=None,
    )

    AdaptivePaperTdfPlayer._encoder_diag_cb(player, msg)

    assert player.latest_enc_imu1_gx == pytest.approx(1.1)
    assert player.latest_enc_imu2_gz == pytest.approx(2.3)
    assert player.latest_enc_packet_age_ms == pytest.approx(7.5)
    assert player.latest_enc_serial_lines == pytest.approx(1234.0)
    assert player.latest_enc_parse_errors == pytest.approx(2.0)
    assert player.latest_enc_stale == pytest.approx(0.0)


def test_runtime_safety_rejects_stale_gantry_and_excess_actual_travel():
    now = 100.0
    args = SimpleNamespace(
        gantry_fresh_timeout=0.25,
        require_encoder_health=False,
        axis='x',
        workspace_min_mm=0.0,
        workspace_max_mm=1150.0,
        workspace_margin_mm=5.0,
        max_travel_mm=850.0,
    )
    state = SimpleNamespace(rx_wall=99.0, x=0.5, y=0.0)
    player = SimpleNamespace(
        args=args,
        latest_gantry_state_sample=state,
        phase='maneuver',
        start_cart_q_mm=100.0,
        direction=1.0,
    )
    player._encoder_health_ready = lambda _now: True

    reason = AdaptivePaperTdfPlayer._runtime_safety_reason(player, now)
    assert 'gantry state is stale' in reason

    state.rx_wall = now
    state.x = 0.951
    reason = AdaptivePaperTdfPlayer._runtime_safety_reason(player, now)
    assert 'actual travel' in reason


def test_runtime_safety_rejects_unhealthy_encoder_and_workspace_margin():
    now = 100.0
    args = SimpleNamespace(
        gantry_fresh_timeout=0.25,
        require_encoder_health=True,
        axis='x',
        workspace_min_mm=0.0,
        workspace_max_mm=1150.0,
        workspace_margin_mm=5.0,
        max_travel_mm=850.0,
    )
    state = SimpleNamespace(rx_wall=now, x=0.5, y=0.0)
    player = SimpleNamespace(
        args=args,
        latest_gantry_state_sample=state,
        phase='id_hold',
        start_cart_q_mm=None,
        direction=1.0,
    )
    player._encoder_health_ready = lambda _now: False
    reason = AdaptivePaperTdfPlayer._runtime_safety_reason(player, now)
    assert 'encoder diagnostics' in reason

    player._encoder_health_ready = lambda _now: True
    state.x = 0.002
    reason = AdaptivePaperTdfPlayer._runtime_safety_reason(player, now)
    assert 'workspace margin' in reason


def test_excite_phase_hands_off_to_id_hold_on_convergence():
    published = []

    class FakeExciter:
        def update(self, now, angle_deg, cart_offset_mm):
            return ExciteCommand(
                velocity_mm_s=0.0,
                amplitude_est_deg=21.0,
                peak_est_deg=21.0,
                angle_rate_deg_s=0.3,
                phase='converged',
                drive_sign=1.0,
                converged=True,
            )

    player = SimpleNamespace(
        args=SimpleNamespace(
            axis='x',
            payload_fresh_timeout=0.25,
            excite_travel_budget_mm=200.0,
            excite_target_angle_deg=21.0,
            tau=5.0,
        ),
        latest_enc_wall=100.45,
        latest_cart_q_mm=500.0,
        excite_start_cart_q_mm=490.0,
        excite_angle_bias_deg=0.0,
        exciter=FakeExciter(),
        _latest_excite_cmd=None,
        phase='excite',
        id_hold_start_wall=None,
        run_start_wall=99.0,
        _last_excite_wait_log_wall=0.0,
        get_logger=lambda: _RecordingLogger(),
        _excite_axis_angle_deg=lambda: 21.0,
        _cart_within_workspace=lambda: True,
        _publish_stream=lambda vx, vy: published.append((vx, vy)),
        _log_row=lambda move_t, vx, vy: None,
    )

    AdaptivePaperTdfPlayer._run_excite_phase(player, now=100.5)

    assert player.phase == 'id_hold'
    assert player.id_hold_start_wall == pytest.approx(100.5)
    assert published == [(0.0, 0.0)]
    assert player._latest_excite_cmd.converged


def test_excite_phase_drives_the_axis_until_converged():
    published = []

    class FakeExciter:
        def update(self, now, angle_deg, cart_offset_mm):
            return ExciteCommand(
                velocity_mm_s=133.0,
                amplitude_est_deg=8.0,
                peak_est_deg=8.5,
                angle_rate_deg_s=40.0,
                phase='servo',
                drive_sign=1.0,
            )

    player = SimpleNamespace(
        args=SimpleNamespace(
            axis='x',
            payload_fresh_timeout=0.25,
            excite_travel_budget_mm=200.0,
            excite_target_angle_deg=21.0,
            tau=5.0,
        ),
        latest_enc_wall=100.45,
        latest_cart_q_mm=500.0,
        excite_start_cart_q_mm=490.0,
        excite_angle_bias_deg=0.0,
        exciter=FakeExciter(),
        _latest_excite_cmd=None,
        phase='excite',
        id_hold_start_wall=None,
        run_start_wall=99.0,
        _last_excite_wait_log_wall=0.0,
        get_logger=lambda: _RecordingLogger(),
        _excite_axis_angle_deg=lambda: 12.0,
        _cart_within_workspace=lambda: True,
        _publish_stream=lambda vx, vy: published.append((vx, vy)),
        _log_row=lambda move_t, vx, vy: None,
    )

    AdaptivePaperTdfPlayer._run_excite_phase(player, now=100.5)

    assert player.phase == 'excite'
    assert published == [(133.0, 0.0)]


def test_excite_phase_aborts_when_the_cart_leaves_the_workspace():
    player = SimpleNamespace(
        args=SimpleNamespace(axis='x', payload_fresh_timeout=0.25),
        latest_enc_wall=100.45,
        exciter=SimpleNamespace(),
        phase='excite',
        wall_start=None,
        _last_excite_wait_log_wall=0.0,
        aborted=False,
        get_logger=lambda: _RecordingLogger(),
        _excite_axis_angle_deg=lambda: 5.0,
        _cart_within_workspace=lambda: False,
        _publish_stream=lambda vx, vy: None,
    )

    def abort(move_t, reason=None):
        player.aborted = True
        player.abort_reason = reason

    player._abort_without_estimate = abort
    player._abort_excite = MethodType(AdaptivePaperTdfPlayer._abort_excite, player)

    AdaptivePaperTdfPlayer._run_excite_phase(player, now=100.5)

    assert player.aborted
    assert 'workspace' in player.abort_reason
    # wall_start is anchored so the main loop can collect residual and finish.
    assert player.wall_start == pytest.approx(100.5)
    assert player.phase == 'residual'


def test_id_hold_phase_advances_to_wait_peak_after_tau():
    published = []
    logged = []
    now_ref = time.monotonic()
    player = SimpleNamespace(
        done=False,
        start_requested=True,
        motion_started=True,
        _last_stream_wall=None,
        wall_start=None,
        phase='id_hold',
        aborted=False,
        id_hold_start_wall=now_ref - 6.0,
        run_start_wall=now_ref - 6.0,
        args=SimpleNamespace(profile='nonzero-ic', tau=5.0, excite=True),
        get_logger=lambda: _RecordingLogger(),
        _publish_stream=lambda vx, vy: published.append((vx, vy)),
        _log_row=lambda move_t, vx, vy: logged.append(move_t),
        _runtime_safety_reason=lambda _now: None,
    )

    AdaptivePaperTdfPlayer._stream_timer_cb(player)

    assert player.phase == 'wait_peak'
    assert published == [(0.0, 0.0)]
    assert logged and logged[0] < 0.0


def _excite_argv(*extra):
    return [
        'adaptive_paper_tdf_player.py',
        '--profile', 'nonzero-ic',
        '--excite',
        '--axis', 'x',
        '--target-distance-mm', '800',
        '--vmax-mm-s', '600',
        '--max-travel-mm', '850',
        '--nonzero-ic-max-command-speed-mm-s', '600',
        '--nonzero-ic-adaptive-frequency',
        *extra,
    ]


def test_check_args_accepts_a_valid_excite_invocation(monkeypatch):
    monkeypatch.setattr(sys, 'argv', _excite_argv('--excite-target-angle-deg', '21'))
    assert check_args(parse_args()) is True


def test_check_args_accepts_independent_frequency_and_timing_controls(monkeypatch):
    monkeypatch.setattr(
        sys,
        'argv',
        _excite_argv(
            '--nonzero-ic-shaper-frequency-scale', '1.01',
            '--timed-profile-actuation-lead-ms', '25',
        ),
    )
    args = parse_args()

    assert check_args(args) is True
    assert args.nonzero_ic_shaper_frequency_scale == pytest.approx(1.01)
    assert args.timed_profile_actuation_lead_ms == pytest.approx(25.0)
    assert args.nonzero_ic_max_payload_queue_delay_ms == pytest.approx(5.0)


def test_check_args_accepts_opt_in_finite_amplitude_correction(monkeypatch):
    monkeypatch.setattr(
        sys,
        'argv',
        _excite_argv('--nonzero-ic-finite-amplitude-correction'),
    )
    args = parse_args()

    assert check_args(args) is True
    assert args.nonzero_ic_finite_amplitude_correction is True


def test_finite_amplitude_correction_defaults_off(monkeypatch):
    monkeypatch.setattr(sys, 'argv', _excite_argv())
    args = parse_args()

    assert check_args(args) is True
    assert args.nonzero_ic_finite_amplitude_correction is False


def test_check_args_accepts_opt_in_robust_nonzero_ic(monkeypatch):
    monkeypatch.setattr(
        sys,
        'argv',
        _excite_argv(
            '--nonzero-ic-robust',
            '--nonzero-ic-robust-band-fraction', '0.05',
            '--timed-profile-lead-s', '1.0',
        ),
    )
    args = parse_args()

    assert check_args(args) is True
    assert args.nonzero_ic_robust is True
    assert args.nonzero_ic_robust_band_fraction == pytest.approx(0.05)


def test_check_args_rejects_short_controller_lead_for_robust_nonzero_ic(
    monkeypatch,
):
    monkeypatch.setattr(
        sys,
        'argv',
        _excite_argv(
            '--nonzero-ic-robust',
            '--timed-profile-lead-s', '0.25',
        ),
    )
    assert check_args(parse_args()) is False


@pytest.mark.parametrize('band', ['0', '0.25', 'nan'])
def test_check_args_rejects_invalid_robust_nonzero_ic_band(monkeypatch, band):
    monkeypatch.setattr(
        sys,
        'argv',
        _excite_argv(
            '--nonzero-ic-robust',
            '--nonzero-ic-robust-band-fraction', band,
        ),
    )
    assert check_args(parse_args()) is False


def test_check_args_accepts_fixed_shaper_frequency_with_adaptive_state_fit(
    monkeypatch,
):
    monkeypatch.setattr(
        sys,
        'argv',
        _excite_argv('--nonzero-ic-shaper-omega-rad-s', '3.25'),
    )
    args = parse_args()

    assert check_args(args) is True
    assert args.nonzero_ic_adaptive_frequency is True
    assert args.nonzero_ic_shaper_omega_rad_s == pytest.approx(3.25)


@pytest.mark.parametrize(
    'argv',
    [
        [
            'adaptive_paper_tdf_player.py',
            '--profile', 'pulse',
            '--nonzero-ic-finite-amplitude-correction',
        ],
        _excite_argv(
            '--nonzero-ic-finite-amplitude-correction',
            '--no-nonzero-ic-adaptive-frequency',
        ),
        _excite_argv(
            '--nonzero-ic-finite-amplitude-correction',
            '--nonzero-ic-shaper-omega-rad-s', '3.27',
        ),
    ],
)
def test_check_args_rejects_incompatible_finite_amplitude_correction(
    monkeypatch,
    argv,
):
    monkeypatch.setattr(sys, 'argv', argv)
    assert check_args(parse_args()) is False


@pytest.mark.parametrize(
    'extra',
    [
        ('--nonzero-ic-shaper-frequency-scale', '0.79'),
        ('--nonzero-ic-shaper-frequency-scale', '1.21'),
        ('--timed-profile-actuation-lead-ms', '-1'),
        ('--timed-profile-actuation-lead-ms', '101'),
        ('--nonzero-ic-max-payload-queue-delay-ms', '-1'),
        ('--nonzero-ic-max-payload-queue-delay-ms', '101'),
        ('--no-controller-timed-profile',
         '--timed-profile-actuation-lead-ms', '25'),
        ('--nonzero-ic-shaper-omega-rad-s', '-1'),
        ('--nonzero-ic-shaper-omega-rad-s', '4.1'),
        ('--nonzero-ic-shaper-omega-rad-s', '3.25',
         '--nonzero-ic-shaper-frequency-scale', '1.01'),
    ],
)
def test_check_args_rejects_bad_frequency_or_timing_controls(
    monkeypatch,
    extra,
):
    monkeypatch.setattr(sys, 'argv', _excite_argv(*extra))
    assert check_args(parse_args()) is False


@pytest.mark.parametrize(
    'extra',
    [
        ('--excite-target-angle-deg', '35', '--excite-abort-angle-deg', '30'),
        ('--excite-speed-mm-s', '900'),
        ('--excite-angle-band-deg', '0.2', '--excite-angle-tolerance-deg', '1.0'),
    ],
)
def test_check_args_rejects_bad_excite_parameters(monkeypatch, extra):
    monkeypatch.setattr(sys, 'argv', _excite_argv(*extra))
    assert check_args(parse_args()) is False


def test_check_args_rejects_excite_outside_nonzero_ic(monkeypatch):
    monkeypatch.setattr(
        sys, 'argv',
        ['adaptive_paper_tdf_player.py', '--profile', 'pulse', '--excite'],
    )
    assert check_args(parse_args()) is False
