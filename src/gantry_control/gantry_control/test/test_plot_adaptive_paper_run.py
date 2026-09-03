import numpy as np
import pandas as pd
import pytest

from plot_adaptive_paper_run import (
    make_summary,
    payload_measurement_series,
    residual_metrics,
    swing_demean_offset,
)


def test_payload_measurement_time_removes_stale_arm_duplicate():
    run_time = np.arange(0.0, 0.60, 0.01)
    sample_time = 100.0 + run_time
    angle = 9.0 * np.sin(3.2 * run_time)
    age_ms = np.full_like(run_time, 2.0)

    # Reproduce the arming trace: a late timer row repeats the last payload
    # measurement, then a queued measurement is logged later than its stamp.
    run_time = np.insert(run_time, 50, 0.57)
    sample_time = np.insert(sample_time, 50, sample_time[49])
    angle = np.insert(angle, 50, angle[49])
    age_ms = np.insert(age_ms, 50, 80.0)
    run_time[51] = 0.58

    data = pd.DataFrame({
        'run_time_sec': run_time,
        'payload_sample_time_s': sample_time,
        'payload_wall_age_ms': age_ms,
        'swing_axis_angle_deg': angle,
    })
    plotted_time, plotted_angle = payload_measurement_series(
        data, 'swing_axis_angle_deg')

    assert len(plotted_time) == len(data) - 1
    assert not plotted_time.duplicated().any()
    # The queued sample belongs at its measurement time, 0.50 s, rather than
    # at the delayed 0.58 s timer callback.
    queued = plotted_time.index[plotted_angle == angle[51]]
    assert len(queued) == 1
    assert plotted_time.loc[queued[0]] == pytest.approx(0.50, abs=1.0e-12)


def test_summary_records_frequency_scale_and_actuation_lead(tmp_path):
    data = pd.DataFrame({
        'traveled_mm': [999.8],
        'omega_n_rad_s': [3.2564],
        'nonzero_ic_fitted_omega_rad_s': [3.2242],
        'nonzero_ic_small_angle_omega_rad_s': [3.2435],
        'nonzero_ic_fitted_amplitude_angle_deg': [17.7],
        'nonzero_ic_frequency_correction_factor': [1.00599],
        'nonzero_ic_shaper_omega_rad_s': [3.2564],
        'nonzero_ic_shaper_frequency_scale': [1.01],
        'nonzero_ic_shaper_frequency_mode': ['scaled-fit'],
        'nonzero_ic_configured_shaper_omega_rad_s': [0.0],
        'timed_profile_actuation_lead_ms': [25.0],
        'payload_calibrated_clock_offset_s': [80.0],
        'payload_queue_delay_ms': [1.5],
        'timed_profile_payload_clock_offset_s': [80.0],
        'timed_profile_arm_payload_queue_delay_ms': [1.5],
        'nonzero_ic_fit_nrmse': [0.008],
        'nonzero_ic_initial_swing_mm': [260.0],
        'nonzero_ic_initial_payload_velocity_mm_s': [0.0],
        'nonzero_ic_predicted_peak_swing_mm': [260.0],
        'nonzero_ic_predicted_peak_payload_velocity_mm_s': [0.0],
        'nonzero_ic_predicted_command_start_swing_mm': [259.6],
        'nonzero_ic_predicted_command_start_payload_velocity_mm_s': [48.0],
        'nonzero_ic_measured_command_start_swing_mm': [259.4],
        'nonzero_ic_measured_command_start_payload_velocity_mm_s': [46.0],
        'nonzero_ic_measured_command_start_sample_offset_ms': [-2.0],
        'nonzero_ic_measured_peak_swing_mm': [260.2],
        'nonzero_ic_measured_peak_payload_velocity_mm_s': [-3.0],
        'nonzero_ic_measured_peak_sample_offset_ms': [1.0],
        'A0': [0.1],
        'A1': [0.9],
        'schedule_T_sec': [1.2],
        'schedule_source': ['nonzero_ic_adaptive_free_swing'],
        'swing_axis_angle_deg': [1.0],
        'phase': ['residual'],
        'run_time_sec': [10.0],
        'payload_sample_time_s': [100.0],
        'payload_wall_age_ms': [2.0],
    })

    summary = make_summary(data, tmp_path / 'run.csv')

    assert summary['fitted_omega_n_rad_s'] == pytest.approx(3.2242)
    assert summary['small_angle_omega_n_rad_s'] == pytest.approx(3.2435)
    assert summary['fitted_amplitude_angle_deg'] == pytest.approx(17.7)
    assert summary['frequency_correction_factor'] == pytest.approx(1.00599)
    assert summary['shaper_omega_n_rad_s'] == pytest.approx(3.2564)
    assert summary['shaper_frequency_scale'] == pytest.approx(1.01)
    assert summary['shaper_frequency_mode'] == 'scaled-fit'
    assert summary['configured_shaper_omega_rad_s'] == pytest.approx(0.0)
    assert summary['actuation_lead_ms'] == pytest.approx(25.0)
    assert summary['payload_calibrated_clock_offset_s'] == pytest.approx(80.0)
    assert summary['payload_queue_delay_ms'] == pytest.approx(1.5)
    assert summary['timed_profile_payload_clock_offset_s'] == pytest.approx(80.0)
    assert summary['timed_profile_arm_payload_queue_delay_ms'] == pytest.approx(1.5)
    assert summary['predicted_peak_payload_velocity_mm_s'] == pytest.approx(0.0)
    assert summary['predicted_command_start_payload_velocity_mm_s'] == pytest.approx(48.0)
    assert summary['measured_command_start_payload_velocity_mm_s'] == pytest.approx(46.0)
    assert summary['measured_peak_payload_velocity_mm_s'] == pytest.approx(-3.0)
    assert summary['measured_peak_sample_offset_ms'] == pytest.approx(1.0)


def test_swing_demean_offset_prefers_residual_measurements():
    data = pd.DataFrame({
        'phase': ['excite', 'residual', 'residual', 'residual'],
        'run_time_sec': [0.00, 0.01, 0.02, 0.03],
        'payload_sample_time_s': [10.00, 10.01, 10.02, 10.03],
        'payload_wall_age_ms': [1.0, 1.0, 1.0, 1.0],
        'swing_axis_angle_deg': [20.0, 1.0, 2.0, 3.0],
    })

    assert swing_demean_offset(
        data, 'swing_axis_angle_deg') == pytest.approx(2.0)


def test_residual_metrics_report_demeaned_rms():
    data = pd.DataFrame({
        'phase': ['residual'] * 4,
        'run_time_sec': [0.0, 1.0, 2.0, 3.0],
        'swing_axis_angle_deg': [1.0, 3.0, 1.0, 3.0],
    })

    metrics = residual_metrics(data, 'swing_axis_angle_deg')

    assert metrics['residual_10s_mean_deg'] == pytest.approx(2.0)
    assert metrics['residual_10s_rms_deg'] == pytest.approx(np.sqrt(5.0))
    assert metrics['residual_10s_rms_demeaned_deg'] == pytest.approx(1.0)
