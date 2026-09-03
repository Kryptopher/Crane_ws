from hmis.simulation import run_simulation
from hmis.precision_stop_simulation import run_precision_stop_comparison


def test_nominal_zvd_reduces_residual_sway():
    _, metrics = run_simulation(dt=0.005)
    assert metrics['hmis_residual_rms_mm'] < 0.1 * metrics['direct_residual_rms_mm']


def test_precision_stop_reduces_zvd_post_neutral_travel():
    _, metrics = run_precision_stop_comparison(dt_s=0.01)
    precision = metrics['precision_stop']
    zvd = metrics['zvd_tail']
    assert precision.peak_forward_excursion_mm < 0.4 * zvd.peak_forward_excursion_mm
    assert precision.final_position_error_mm < 1.0
    assert precision.precision_settle_time_s is not None


def test_precision_stop_removes_hard_stop_residual_sway():
    _, metrics = run_precision_stop_comparison(dt_s=0.01)
    precision = metrics['precision_stop']
    hard_stop = metrics['hard_stop']
    assert precision.residual_angle_rms_deg < 0.1 * hard_stop.residual_angle_rms_deg
    assert precision.peak_command_speed_mm_s <= 60.0 + 1.0e-6
    assert precision.peak_command_acceleration_mm_s2 <= 150.0 + 1.0e-6


def test_precision_stop_handles_small_sway_at_neutral():
    _, metrics = run_precision_stop_comparison(
        dt_s=0.01,
        initial_angle_deg=1.0,
        initial_angle_rate_deg_s=0.0,
    )
    precision = metrics['precision_stop']
    assert precision.peak_absolute_excursion_mm < 35.0
    assert precision.final_position_error_mm < 1.0
    assert precision.residual_angle_rms_deg < 0.02


def test_precision_stop_tolerates_ten_percent_frequency_error_after_shaped_cruise():
    for plant_frequency_hz in (0.45, 0.55):
        _, metrics = run_precision_stop_comparison(
            dt_s=0.01,
            mode_frequency_hz=plant_frequency_hz,
            controller_frequency_hz=0.5,
        )
        precision = metrics['precision_stop']
        assert precision.peak_absolute_excursion_mm < 25.0
        assert precision.final_position_error_mm < 1.0
        assert precision.residual_angle_rms_deg < 0.03


def test_precision_stop_at_low_jog_speed_stays_inside_correction_envelope():
    _, metrics = run_precision_stop_comparison(
        dt_s=0.01,
        cruise_speed_m_s=0.1,
        max_command_acceleration_m_s2=0.2,
    )
    precision = metrics['precision_stop']
    assert precision.peak_absolute_excursion_mm < 35.0
    assert precision.final_position_error_mm < 1.0
    assert precision.peak_command_acceleration_mm_s2 <= 200.0 + 1.0e-6
