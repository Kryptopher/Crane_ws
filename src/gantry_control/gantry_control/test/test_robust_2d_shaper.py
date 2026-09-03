import math

import pytest

from robust_2d_shaper import (
    G,
    Robust2dSquareProfile,
    SwayState,
    estimate_sway_state,
    propagate_sway_state,
)


def make_profile(**overrides):
    arguments = {
        'x_distance_mm': 400.0,
        'y_distance_mm': 350.0,
        'speed_mm_s': 150.0,
        'rope_length_m': 0.90,
        'damping_ratio': 0.0,
        'timing_scale': 1.0,
        'corner_dwell_s': 0.0,
    }
    arguments.update(overrides)
    return Robust2dSquareProfile(**arguments)


def test_standard_zvd_model_and_nominal_zero_residual():
    profile = make_profile()
    expected_spacing = math.pi / math.sqrt(G / 0.90)

    assert profile.impulse_spacing_s == pytest.approx(expected_spacing)
    assert profile.amplitudes == pytest.approx((0.25, 0.50, 0.25))
    assert sum(profile.amplitudes) == pytest.approx(1.0)
    assert profile.residual_fraction() < 1.0e-12


def test_damped_zvd_uses_damped_period_and_cancels_nominal_mode():
    profile = make_profile(damping_ratio=0.03)
    omega_n = math.sqrt(G / 0.90)
    expected_spacing = math.pi / (omega_n * math.sqrt(1.0 - 0.03**2))

    assert profile.impulse_spacing_s == pytest.approx(expected_spacing)
    assert profile.residual_fraction() < 1.0e-12


def test_each_leg_finishes_before_next_axis_starts():
    profile = make_profile(corner_dwell_s=0.12)
    expected_names = ('+X', '+Y', '-X', '-Y')
    assert tuple(leg.name for leg in profile.legs) == expected_names

    for current, following in zip(profile.legs, profile.legs[1:]):
        assert following.start_s == pytest.approx(
            current.shaped_stop_s + profile.corner_dwell_s
        )
        midpoint = 0.5 * (current.shaped_stop_s + following.start_s)
        assert profile.command_at(midpoint) == pytest.approx((0.0, 0.0))


def test_command_is_always_axis_aligned_and_follows_requested_order():
    profile = make_profile()
    events = profile.event_times_s()
    nonzero_phases = []
    previous_phase = None
    for left, right in zip(events, events[1:]):
        if right - left < 1.0e-12:
            continue
        midpoint = 0.5 * (left + right)
        vx, vy = profile.command_at(midpoint)
        assert abs(vx) < 1.0e-12 or abs(vy) < 1.0e-12
        assert math.hypot(vx, vy) <= profile.speed_mm_s + 1.0e-12
        if math.hypot(vx, vy) > 1.0e-12:
            phase = profile.phase_at(midpoint)
            if phase != previous_phase:
                nonzero_phases.append(phase)
                previous_phase = phase

    assert nonzero_phases == ['+X', '+Y', '-X', '-Y']


def test_exact_displacement_hits_all_four_corners_and_returns_home():
    profile = make_profile()

    for leg, waypoint in zip(profile.legs, profile.waypoints_mm[1:]):
        assert profile.displacement_at(leg.shaped_stop_s) == pytest.approx(
            waypoint, abs=1.0e-10
        )
    assert profile.displacement_at(profile.duration_s) == pytest.approx(
        (0.0, 0.0), abs=1.0e-10
    )
    assert profile.command_at(profile.duration_s) == (0.0, 0.0)


def test_event_integral_matches_closed_form_displacement():
    profile = make_profile(corner_dwell_s=0.07)
    integrated_x = 0.0
    integrated_y = 0.0
    events = profile.event_times_s()
    for left, right in zip(events, events[1:]):
        vx, vy = profile.command_at(0.5 * (left + right))
        integrated_x += vx * (right - left)
        integrated_y += vy * (right - left)

    assert (integrated_x, integrated_y) == pytest.approx(
        profile.displacement_at(profile.duration_s), abs=1.0e-10
    )


def test_zvd_retains_low_residual_with_ten_percent_frequency_error():
    profile = make_profile()

    assert profile.residual_fraction(0.90) < 0.025
    assert profile.residual_fraction(1.10) < 0.025


def test_two_mode_zvd_convolution_cancels_both_modes():
    profile = make_profile(
        x_distance_mm=1000.0,
        y_distance_mm=1000.0,
        speed_mm_s=400.0,
        second_mode_frequency_hz=5.06,
    )
    second_spacing = 1.0 / (2.0 * 5.06)

    assert len(profile.modes) == 2
    assert len(profile.impulse_times_s) == 9
    assert len(profile.amplitudes) == 9
    assert tuple(sorted(profile.impulse_times_s)) == profile.impulse_times_s
    assert sum(profile.amplitudes) == pytest.approx(1.0)
    assert profile.shaper_tail_s == pytest.approx(
        2.0 * profile.impulse_spacing_s + 2.0 * second_spacing
    )
    assert profile.residual_fraction() < 1.0e-12
    assert profile.second_mode_residual_fraction() < 1.0e-12


def test_two_mode_design_is_robust_over_measured_second_mode_band():
    profile = make_profile(
        x_distance_mm=1000.0,
        y_distance_mm=1000.0,
        speed_mm_s=400.0,
        second_mode_frequency_hz=5.06,
    )

    assert profile.second_mode_residual_fraction(4.94 / 5.06) < 0.025
    assert profile.second_mode_residual_fraction(5.15 / 5.06) < 0.025


def test_two_mode_speed_rises_then_falls_without_reacceleration():
    profile = make_profile(
        x_distance_mm=1000.0,
        y_distance_mm=1000.0,
        speed_mm_s=400.0,
        second_mode_frequency_hz=5.06,
    )
    leg = profile.legs[0]
    events = sorted({
        leg.start_s + delay
        for delay in profile.impulse_times_s
    } | {
        leg.raw_stop_s + delay
        for delay in profile.impulse_times_s
    })
    speeds = [
        profile.command_at(0.5 * (left + right))[0]
        for left, right in zip(events, events[1:])
    ]
    peak_index = speeds.index(max(speeds))

    assert speeds[:peak_index + 1] == sorted(speeds[:peak_index + 1])
    assert speeds[peak_index:] == sorted(speeds[peak_index:], reverse=True)
    assert max(speeds) == pytest.approx(profile.speed_mm_s)


def test_robust_harmonic_estimator_recovers_nonzero_state_bias_and_outlier():
    profile = make_profile(damping_ratio=0.02)
    mode = profile.modes[0]
    reference_s = 10.0
    expected = SwayState(math.radians(0.4), math.radians(0.8))
    bias = math.radians(0.13)
    samples = []
    for index in range(251):
        stamp = reference_s - 2.5 + 0.01 * index
        state = propagate_sway_state(expected, mode, stamp - reference_s)
        angle = bias + state.angle_rad
        if index == 100:
            angle += math.radians(2.0)
        samples.append((stamp, angle))

    estimate = estimate_sway_state(
        samples,
        mode=mode,
        reference_time_s=reference_s,
        window_s=2.5,
        minimum_samples=120,
        minimum_span_s=1.5,
    )

    assert estimate.state.angle_rad == pytest.approx(expected.angle_rad, abs=1e-6)
    assert estimate.state.angular_rate_rad_s == pytest.approx(
        expected.angular_rate_rad_s, abs=2e-6
    )
    assert estimate.bias_rad == pytest.approx(bias, abs=1e-6)
    assert math.degrees(estimate.rmse_rad) < 0.01


def test_nzic_corrections_preserve_distance_monotonicity_and_both_modes():
    profile = make_profile(
        x_distance_mm=1000.0,
        y_distance_mm=1000.0,
        speed_mm_s=400.0,
        second_mode_frequency_hz=5.06,
        corner_dwell_s=1.0,
    )
    state = SwayState(math.radians(0.5), 0.0)
    corrected = profile.with_leg_nonzero_initial_condition(
        leg_index=0,
        initial_state=state,
        initial_state_time_s=0.0,
        minimum_correction_amplitude_rad=math.radians(0.1),
    )
    corrected = corrected.with_leg_nonzero_initial_condition(
        leg_index=1,
        initial_state=state,
        initial_state_time_s=corrected.legs[1].start_s,
        minimum_correction_amplitude_rad=math.radians(0.1),
    )

    assert tuple(item.leg_name for item in corrected.ic_corrections) == ('+X', '+Y')
    for correction in corrected.ic_corrections:
        assert correction.nominal_residual_rad < math.radians(1e-8)
        assert (
            correction.primary_band_residual_rad
            < correction.uncorrected_primary_band_residual_rad
        )
        assert correction.second_mode_band_residual_fraction <= 0.03
        assert min(correction.start_amplitudes) >= 0.0
        assert min(correction.stop_amplitudes) >= 0.0
        assert sum(correction.start_amplitudes) == pytest.approx(1.0)
        assert sum(correction.stop_amplitudes) == pytest.approx(1.0)
        assert sum(
            amplitude * delay
            for amplitude, delay in zip(
                correction.start_amplitudes, corrected.impulse_times_s
            )
        ) == pytest.approx(sum(
            amplitude * delay
            for amplitude, delay in zip(
                correction.stop_amplitudes, corrected.impulse_times_s
            )
        ))

    for leg in corrected.legs[:2]:
        assert list(np_cumulative(leg.start_amplitudes)) == sorted(
            np_cumulative(leg.start_amplitudes)
        )
        assert list(np_cumulative(leg.stop_amplitudes)) == sorted(
            np_cumulative(leg.stop_amplitudes)
        )
    assert corrected.displacement_at(corrected.duration_s) == pytest.approx(
        (0.0, 0.0), abs=3e-9
    )
    assert corrected.command_at(corrected.duration_s) == (0.0, 0.0)


def np_cumulative(values):
    total = 0.0
    result = []
    for value in values:
        total += value
        result.append(total)
    return result


def test_runtime_y_retiming_adds_only_zero_velocity_dwell():
    profile = make_profile(
        x_distance_mm=1000.0,
        y_distance_mm=1000.0,
        speed_mm_s=400.0,
        second_mode_frequency_hz=5.06,
        corner_dwell_s=1.0,
    )
    old_y_start = profile.legs[1].start_s
    delayed = profile.with_retimed_leg_start(1, old_y_start + 0.6)

    assert delayed.legs[1].start_s == pytest.approx(old_y_start + 0.6)
    assert delayed.duration_s == pytest.approx(profile.duration_s + 0.6)
    assert delayed.command_at(old_y_start + 0.3) == pytest.approx((0.0, 0.0))
    assert delayed.displacement_at(delayed.duration_s) == pytest.approx(
        (0.0, 0.0), abs=1e-9
    )


def test_profile_rejects_leg_shorter_than_shaper_tail():
    with pytest.raises(ValueError, match='monotonic'):
        make_profile(
            x_distance_mm=700.0,
            y_distance_mm=700.0,
            speed_mm_s=400.0,
            second_mode_frequency_hz=5.06,
        )


@pytest.mark.parametrize(
    'override',
    [
        {'x_distance_mm': 0.0},
        {'speed_mm_s': -1.0},
        {'rope_length_m': float('nan')},
        {'damping_ratio': 1.0},
        {'second_mode_frequency_hz': -5.0},
        {'second_mode_frequency_hz': 5.2, 'second_mode_damping_ratio': 1.0},
        {'corner_dwell_s': -0.1},
    ],
)
def test_invalid_physical_configurations_are_rejected(override):
    with pytest.raises(ValueError):
        make_profile(**override)
