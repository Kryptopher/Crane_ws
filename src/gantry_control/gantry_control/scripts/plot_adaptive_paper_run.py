#!/usr/bin/env python3
"""Create a per-run plot and machine-readable summary from a player CSV."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PHASE_COLORS = {
    'excite': '#dbeafe',
    'id_hold': '#fef3c7',
    'wait_peak': '#fee2e2',
    'arm_profile': '#ffedd5',
    'armed_profile': '#e0e7ff',
    'maneuver': '#dcfce7',
    'residual': '#f3e8ff',
}


def finite_series(data: pd.DataFrame, name: str) -> pd.Series:
    if name not in data:
        return pd.Series(np.nan, index=data.index, dtype=float)
    return pd.to_numeric(data[name], errors='coerce')


def last_finite(data: pd.DataFrame, name: str) -> float | None:
    values = finite_series(data, name).dropna()
    return None if values.empty else float(values.iloc[-1])


def last_text(data: pd.DataFrame, name: str) -> str:
    if name not in data:
        return ''
    values = data[name].dropna().astype(str)
    values = values[values != '']
    return '' if values.empty else values.iloc[-1]


def angle_column(data: pd.DataFrame) -> str | None:
    for name in ('swing_axis_angle_deg', 'swing_pitch_deg', 'swing_roll_deg'):
        if name in data and finite_series(data, name).notna().any():
            return name
    return None


def payload_aligned_time(data: pd.DataFrame) -> pd.Series:
    """Map payload measurement stamps onto the run-time clock.

    The player timer and the payload subscription share one executor.  If a
    calculation briefly delays that executor, a payload sample can be logged
    later than it was measured.  Plotting that sample at the timer callback
    time compresses the delayed measurements and creates a false vertical
    discontinuity.  The clocks have a constant offset, so a robust offset from
    fresh samples restores the measurement time without changing any data.
    """
    run_time = finite_series(data, 'run_time_sec')
    sample_time = finite_series(data, 'payload_sample_time_s')
    valid = run_time.notna() & sample_time.notna()
    if not valid.any():
        return run_time

    offset_candidates = valid.copy()
    if 'payload_wall_age_ms' in data:
        age_ms = finite_series(data, 'payload_wall_age_ms')
        fresh = valid & age_ms.notna() & age_ms.le(20.0)
        if int(fresh.sum()) >= 3:
            offset_candidates = fresh
    offset = float((run_time[offset_candidates] - sample_time[offset_candidates]).median())
    aligned = sample_time + offset
    return aligned.where(sample_time.notna(), run_time)


def payload_measurement_series(
    data: pd.DataFrame,
    name: str,
) -> tuple[pd.Series, pd.Series]:
    """Return one point per payload measurement on its aligned timebase."""
    frame = pd.DataFrame({
        'time': payload_aligned_time(data),
        'sample_time': finite_series(data, 'payload_sample_time_s'),
        'value': finite_series(data, name),
    })
    frame = frame[frame['time'].notna() & frame['value'].notna()].copy()
    if frame.empty:
        return frame['time'], frame['value']

    # Timer rows may repeat the most recent payload value.  Keep the first
    # observation of each sensor timestamp; later copies only create a false
    # horizontal hold followed by a near-vertical jump.
    stamped = frame['sample_time'].notna()
    duplicate = stamped & frame.duplicated('sample_time', keep='first')
    frame = frame[~duplicate].sort_values('time', kind='stable')
    frame.loc[frame['time'].diff().gt(0.10), 'value'] = np.nan
    return frame['time'], frame['value']


def swing_demean_offset(data: pd.DataFrame, angle_name: str | None) -> float:
    """Estimate the static encoder-angle bias without modifying raw data.

    Prefer the complete residual window because the trolley is stationary and
    that window normally contains several oscillation cycles.  Fall back to
    all payload measurements for incomplete/aborted logs.
    """
    if angle_name is None:
        return 0.0
    basis = data
    if 'phase' in data:
        residual = data[data['phase'].astype(str) == 'residual']
        if not residual.empty:
            basis = residual
    _, angle = payload_measurement_series(basis, angle_name)
    finite = pd.to_numeric(angle, errors='coerce').dropna()
    return 0.0 if finite.empty else float(finite.mean())


def shade_phases(axis, data: pd.DataFrame) -> None:
    if 'phase' not in data or 'run_time_sec' not in data:
        return
    run_time = finite_series(data, 'run_time_sec')
    ranges: list[tuple[str, float, float]] = []
    for phase, color in PHASE_COLORS.items():
        mask = data['phase'].astype(str) == phase
        values = run_time[mask].dropna()
        if values.empty:
            continue
        start = float(values.min())
        end = float(values.max())
        ranges.append((phase, start, end))
        axis.axvspan(start, end, color=color, alpha=0.45, linewidth=0)

    compact = [item for item in ranges if item[0] in {
        'wait_peak', 'arm_profile', 'armed_profile'}]
    for phase, start, end in ranges:
        if phase in {'wait_peak', 'arm_profile', 'armed_profile'}:
            continue
        axis.text(
            0.5 * (start + end), 0.985, phase.replace('_', ' '),
            transform=axis.get_xaxis_transform(), ha='center', va='top',
            fontsize=8)
    if compact:
        start = min(item[1] for item in compact)
        end = max(item[2] for item in compact)
        axis.text(
            0.5 * (start + end), 0.985, 'peak + arm',
            transform=axis.get_xaxis_transform(), ha='center', va='top',
            fontsize=8, rotation=90 if end - start < 0.8 else 0)


def residual_metrics(data: pd.DataFrame, angle_name: str | None) -> dict:
    result = {
        'residual_10s_p2p_deg': None,
        'residual_10s_rms_deg': None,
        'residual_10s_mean_deg': None,
        'residual_10s_rms_demeaned_deg': None,
        'residual_final_5s_p2p_deg': None,
        'residual_final_5s_rms_deg': None,
        'residual_final_5s_mean_deg': None,
        'residual_final_5s_rms_demeaned_deg': None,
    }
    if angle_name is None or 'phase' not in data:
        return result
    residual = data[data['phase'].astype(str) == 'residual'].copy()
    if residual.empty:
        return result
    t = finite_series(residual, 'run_time_sec')
    angle = finite_series(residual, angle_name)
    valid = t.notna() & angle.notna()
    if not valid.any():
        return result
    t = t[valid]
    angle = angle[valid]
    relative_time = t - float(t.iloc[0])
    for duration, prefix in ((10.0, 'residual_10s'), (5.0, 'residual_final_5s')):
        window = angle[relative_time >= max(0.0, float(relative_time.max()) - duration)]
        if window.empty:
            continue
        mean = float(window.mean())
        centered = window - mean
        result[f'{prefix}_p2p_deg'] = float(window.max() - window.min())
        result[f'{prefix}_rms_deg'] = float(np.sqrt(np.mean(np.square(window))))
        result[f'{prefix}_mean_deg'] = mean
        result[f'{prefix}_rms_demeaned_deg'] = float(
            np.sqrt(np.mean(np.square(centered))))
    return result


def make_summary(data: pd.DataFrame, csv_path: Path) -> dict:
    angle_name = angle_column(data)
    summary = {
        'csv': str(csv_path),
        'samples': int(len(data)),
        'schedule_source': '',
        'travel_mm': last_finite(data, 'traveled_mm'),
        'omega_n_rad_s': last_finite(data, 'omega_n_rad_s'),
        'fitted_omega_n_rad_s': last_finite(
            data, 'nonzero_ic_fitted_omega_rad_s'),
        'small_angle_omega_n_rad_s': last_finite(
            data, 'nonzero_ic_small_angle_omega_rad_s'),
        'fitted_amplitude_angle_deg': last_finite(
            data, 'nonzero_ic_fitted_amplitude_angle_deg'),
        'frequency_correction_factor': last_finite(
            data, 'nonzero_ic_frequency_correction_factor'),
        'shaper_omega_n_rad_s': last_finite(
            data, 'nonzero_ic_shaper_omega_rad_s'),
        'shaper_frequency_scale': last_finite(
            data, 'nonzero_ic_shaper_frequency_scale'),
        'configured_shaper_omega_rad_s': last_finite(
            data, 'nonzero_ic_configured_shaper_omega_rad_s'),
        'robust_enabled': bool(
            last_finite(data, 'nonzero_ic_robust_enabled') or False),
        'robust_band_fraction': last_finite(
            data, 'nonzero_ic_robust_band_fraction'),
        'robust_impulse_times_s': last_text(
            data, 'nonzero_ic_robust_impulse_times_s'),
        'robust_start_amplitudes': last_text(
            data, 'nonzero_ic_robust_start_amplitudes'),
        'robust_stop_amplitudes': last_text(
            data, 'nonzero_ic_robust_stop_amplitudes'),
        'robust_worst_residual_fraction': last_finite(
            data, 'nonzero_ic_robust_worst_residual_fraction'),
        'two_impulse_worst_residual_fraction': last_finite(
            data, 'nonzero_ic_two_impulse_worst_residual_fraction'),
        'robust_optimizer_iterations': last_finite(
            data, 'nonzero_ic_robust_optimizer_iterations'),
        'actuation_lead_ms': last_finite(
            data, 'timed_profile_actuation_lead_ms'),
        'payload_calibrated_clock_offset_s': last_finite(
            data, 'payload_calibrated_clock_offset_s'),
        'payload_queue_delay_ms': last_finite(
            data, 'payload_queue_delay_ms'),
        'timed_profile_payload_clock_offset_s': last_finite(
            data, 'timed_profile_payload_clock_offset_s'),
        'timed_profile_arm_payload_queue_delay_ms': last_finite(
            data, 'timed_profile_arm_payload_queue_delay_ms'),
        'fit_nrmse': last_finite(data, 'nonzero_ic_fit_nrmse'),
        'initial_swing_mm': last_finite(data, 'nonzero_ic_initial_swing_mm'),
        'initial_payload_velocity_mm_s': last_finite(
            data, 'nonzero_ic_initial_payload_velocity_mm_s'),
        'predicted_peak_swing_mm': last_finite(
            data, 'nonzero_ic_predicted_peak_swing_mm'),
        'predicted_peak_payload_velocity_mm_s': last_finite(
            data, 'nonzero_ic_predicted_peak_payload_velocity_mm_s'),
        'predicted_command_start_swing_mm': last_finite(
            data, 'nonzero_ic_predicted_command_start_swing_mm'),
        'predicted_command_start_payload_velocity_mm_s': last_finite(
            data, 'nonzero_ic_predicted_command_start_payload_velocity_mm_s'),
        'measured_command_start_swing_mm': last_finite(
            data, 'nonzero_ic_measured_command_start_swing_mm'),
        'measured_command_start_payload_velocity_mm_s': last_finite(
            data, 'nonzero_ic_measured_command_start_payload_velocity_mm_s'),
        'measured_command_start_sample_offset_ms': last_finite(
            data, 'nonzero_ic_measured_command_start_sample_offset_ms'),
        'measured_peak_swing_mm': last_finite(
            data, 'nonzero_ic_measured_peak_swing_mm'),
        'measured_peak_payload_velocity_mm_s': last_finite(
            data, 'nonzero_ic_measured_peak_payload_velocity_mm_s'),
        'measured_peak_sample_offset_ms': last_finite(
            data, 'nonzero_ic_measured_peak_sample_offset_ms'),
        'A0': last_finite(data, 'A0'),
        'A1': last_finite(data, 'A1'),
        'switch_time_s': last_finite(data, 'schedule_T_sec'),
        'angle_source': angle_name or '',
        'swing_plot_demean_offset_deg': swing_demean_offset(data, angle_name),
    }
    if 'schedule_source' in data:
        sources = data['schedule_source'].dropna().astype(str)
        sources = sources[sources != '']
        if not sources.empty:
            summary['schedule_source'] = sources.iloc[-1]
    if 'nonzero_ic_shaper_frequency_mode' in data:
        modes = data['nonzero_ic_shaper_frequency_mode'].dropna().astype(str)
        modes = modes[modes != '']
        if not modes.empty:
            summary['shaper_frequency_mode'] = modes.iloc[-1]
    summary.update(residual_metrics(data, angle_name))
    return summary


def plot_run(data: pd.DataFrame, output: Path) -> None:
    run_time = finite_series(data, 'run_time_sec')
    angle_name = angle_column(data)
    angle_offset_deg = swing_demean_offset(data, angle_name)
    fig, axes = plt.subplots(4, 1, figsize=(12, 13), constrained_layout=True)
    fig.suptitle(output.stem.replace('_plot', '').replace('_', ' '), fontsize=14)

    axes[0].plot(run_time, finite_series(data, 'cart_q_mm'), color='#1d4ed8', lw=1.4)
    axes[0].set_ylabel('Trolley position (mm)')

    axes[1].step(
        run_time, finite_series(data, 'cmd_vx_mm_s'), where='post',
        color='#dc2626', lw=1.2, label='Commanded velocity')
    axes[1].plot(
        run_time, finite_series(data, 'cart_vx_mm_s'),
        color='#1d4ed8', lw=1.1, label='Measured velocity')
    axes[1].set_ylabel('X velocity (mm/s)')
    axes[1].legend(loc='upper left')

    if angle_name is not None:
        angle_time, angle = payload_measurement_series(data, angle_name)
        axes[2].plot(
            angle_time, angle - angle_offset_deg,
            color='#7c3aed', lw=1.2)
    axes[2].set_ylabel('Payload swing, demeaned (deg)')

    residual = (
        data[data['phase'].astype(str) == 'residual'].copy()
        if 'phase' in data else data.iloc[0:0].copy())
    if not residual.empty and angle_name is not None:
        residual_t, residual_angle = payload_measurement_series(
            residual, angle_name)
        residual_t = residual_t - float(residual_t.dropna().iloc[0])
        axes[3].plot(
            residual_t, residual_angle - angle_offset_deg,
            color='#7c3aed', lw=1.3)
    axes[3].set_ylabel('Residual swing, demeaned (deg)')
    axes[3].set_xlabel('Time (s)')

    for axis in axes[:3]:
        shade_phases(axis, data)
        axis.set_xlabel('Run time (s)')
    for axis in axes:
        axis.axhline(0.0, color='black', lw=0.7, alpha=0.4)
        axis.grid(True, alpha=0.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches='tight')
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv_path')
    parser.add_argument('--output', default='')
    parser.add_argument('--summary', default='')
    args = parser.parse_args()

    csv_path = Path(args.csv_path).expanduser().resolve()
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        print(f'Cannot plot missing or empty CSV: {csv_path}')
        return 2
    output = (
        Path(args.output).expanduser().resolve()
        if args.output else csv_path.with_name(csv_path.stem + '_plot.png'))
    summary_path = (
        Path(args.summary).expanduser().resolve()
        if args.summary else csv_path.with_name(csv_path.stem + '_summary.json'))

    data = pd.read_csv(csv_path, low_memory=False)
    if data.empty:
        print(f'Cannot plot CSV without rows: {csv_path}')
        return 2
    plot_run(data, output)
    svg_output = output if output.suffix.lower() == '.svg' else output.with_suffix('.svg')
    if svg_output != output:
        plot_run(data, svg_output)
    summary = make_summary(data, csv_path)
    summary['plot'] = str(output)
    summary['plot_svg'] = str(svg_output)
    summary_path.write_text(json.dumps(summary, indent=2) + '\n')
    print(f'Plot saved: {output}')
    if svg_output != output:
        print(f'SVG saved: {svg_output}')
    print(f'Summary saved: {summary_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
