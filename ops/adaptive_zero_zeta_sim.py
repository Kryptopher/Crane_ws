#!/usr/bin/env python3
"""ROS-free zero-zeta system ID and adaptive input-shaper schedule.

This is the simulation-portable form of the final encoder-based
``zero-zeta-ls`` method used by ``adaptive_paper_tdf_player.py``.

The identifier searches the shaper half-period ``Tp`` and fits, for every
candidate,

    y(t) = c0 + c1*t + a*cos(omega*t) + b*sin(omega*t)
    omega = pi / Tp

The candidate with minimum ``RMSE / hypot(a, b)`` is selected.  Damping is
fixed to zero.  The identified ``Tp`` is a modal half-period; it is NOT the
same quantity as the closed-form AIS schedule delay ``T``.

Minimal simulation use::

    from adaptive_zero_zeta_sim import identify_zero_zeta, solve_ais_schedule

    estimate = identify_zero_zeta(time_s, relative_swing_mm)
    schedule = solve_ais_schedule(estimate.omega_n_rad_s, K=0.20, tau_s=3.7)
    gain = schedule.velocity_gain(simulation_time_s, pulse_duration_s)
    commanded_velocity = maximum_velocity * gain

Only NumPy is required.  No ROS packages or crane hardware are used.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ZeroZetaEstimate:
    """Best undamped single-mode least-squares estimate."""

    shaper_half_period_s: float
    omega_n_rad_s: float
    frequency_hz: float
    zeta: float
    offset: float
    drift_per_s: float
    cosine_coefficient: float
    sine_coefficient: float
    amplitude: float
    phase_rad: float
    rmse: float
    normalized_rmse: float
    peak_to_peak: float
    condition: float
    sample_count: int
    window_duration_s: float

    def oscillation_state_at(self, elapsed_from_first_sample_s: float) -> tuple[float, float]:
        """Return fitted oscillatory displacement and velocity (drift excluded)."""
        phase = self.omega_n_rad_s * float(elapsed_from_first_sample_s)
        cosine = math.cos(phase)
        sine = math.sin(phase)
        displacement = (
            self.cosine_coefficient * cosine
            + self.sine_coefficient * sine
        )
        velocity = self.omega_n_rad_s * (
            -self.cosine_coefficient * sine
            + self.sine_coefficient * cosine
        )
        return displacement, velocity


@dataclass(frozen=True)
class AisSchedule:
    """Undamped three-impulse adaptive shaper used by the final experiment."""

    omega_n_rad_s: float
    modal_half_period_s: float
    K: float
    A: float
    A2: float
    tau_s: float
    schedule_delay_s: float

    @property
    def amplitudes(self) -> tuple[float, float, float]:
        return self.K, self.A, self.A2

    @property
    def impulse_times_s(self) -> tuple[float, float, float]:
        return 0.0, self.tau_s + self.schedule_delay_s, self.tau_s + 2.0 * self.schedule_delay_s

    def residual_phasor(self) -> complex:
        """Return the nominal undamped modal residual; it should be near zero."""
        return sum(
            amplitude * np.exp(-1j * self.omega_n_rad_s * delay)
            for amplitude, delay in zip(self.amplitudes, self.impulse_times_s)
        )

    def velocity_gain(self, time_s: float, pulse_duration_s: float) -> float:
        """Return the shaped rectangular-pulse velocity gain at ``time_s``."""
        if pulse_duration_s <= self.tau_s + 2.0 * self.schedule_delay_s:
            raise ValueError(
                'pulse_duration_s must exceed tau + 2*T so the full-speed '
                'middle interval exists'
            )
        t = float(time_s)
        first_switch = self.tau_s + self.schedule_delay_s
        second_switch = self.tau_s + 2.0 * self.schedule_delay_s
        if t < 0.0:
            return 0.0
        if t < first_switch:
            return self.K
        if t < second_switch:
            return self.K + self.A
        if t < pulse_duration_s:
            return 1.0
        if t < pulse_duration_s + first_switch:
            return 1.0 - self.K
        if t < pulse_duration_s + second_switch:
            return self.A2
        return 0.0


def identify_zero_zeta(
    times_s: Iterable[float],
    relative_swing: Iterable[float],
    *,
    half_period_min_s: float = 0.45,
    half_period_max_s: float = 1.45,
    grid_count: int = 240,
    minimum_samples: int = 80,
    minimum_peak_to_peak: float = 4.0,
    minimum_amplitude: float = 1.0,
    maximum_normalized_rmse: float = 1.5,
) -> ZeroZetaEstimate:
    """Run the exact final-experiment zero-zeta grid regression.

    Signal units are arbitrary but must be consistent.  The default amplitude
    gates are the final experiment's millimetre settings.  For metre-valued or
    normalized simulation signals, scale the two amplitude gates accordingly.
    """
    times = np.asarray(tuple(times_s), dtype=float)
    swing = np.asarray(tuple(relative_swing), dtype=float)
    if times.ndim != 1 or swing.ndim != 1 or len(times) != len(swing):
        raise ValueError('times_s and relative_swing must be equal-length 1-D arrays')
    finite = np.isfinite(times) & np.isfinite(swing)
    times = times[finite]
    swing = swing[finite]
    if len(times) < minimum_samples:
        raise ValueError(f'need at least {minimum_samples} finite samples; got {len(times)}')
    order = np.argsort(times)
    times = times[order]
    swing = swing[order]
    if np.any(np.diff(times) <= 0.0):
        raise ValueError('sample times must be strictly increasing')
    if not 0.0 < half_period_min_s < half_period_max_s:
        raise ValueError('half-period bounds must be positive and ordered')
    if grid_count < 5:
        raise ValueError('grid_count must be at least 5')

    peak_to_peak = float(np.max(swing) - np.min(swing))
    if peak_to_peak < minimum_peak_to_peak:
        raise ValueError(
            f'swing peak-to-peak {peak_to_peak:.6g} is below '
            f'{minimum_peak_to_peak:.6g}'
        )

    # This is intentionally identical to AdaptivePaperTdfPlayer's final
    # vectorized zero-zeta fitter.
    centered_time = times - float(times[0])
    half_periods = np.linspace(
        half_period_min_s,
        half_period_max_s,
        max(5, int(grid_count)),
    )
    omegas = math.pi / half_periods
    phase = omegas[:, None] * centered_time[None, :]
    design = np.empty((len(half_periods), len(centered_time), 4), dtype=float)
    design[:, :, 0] = 1.0
    design[:, :, 1] = centered_time
    design[:, :, 2] = np.cos(phase)
    design[:, :, 3] = np.sin(phase)

    try:
        xtx = np.einsum('gni,gnj->gij', design, design, optimize=True)
        xty = np.einsum('gni,n->gi', design, swing, optimize=True)
        coefficients = np.linalg.solve(xtx, xty)
        prediction = np.einsum('gni,gi->gn', design, coefficients, optimize=True)
        rmse = np.sqrt(np.mean((swing[None, :] - prediction) ** 2, axis=1))
        amplitude = np.hypot(coefficients[:, 2], coefficients[:, 3])
        normalized_rmse = rmse / np.maximum(amplitude, 1.0e-9)
        # cond(X) = sqrt(cond(X.T @ X)), matching the online implementation.
        condition = np.sqrt(np.linalg.cond(xtx))
    except (FloatingPointError, np.linalg.LinAlgError, ValueError) as exc:
        raise ValueError(f'zero-zeta least-squares grid failed: {exc}') from exc

    valid = (
        np.isfinite(normalized_rmse)
        & np.isfinite(rmse)
        & np.isfinite(amplitude)
        & np.isfinite(condition)
        & (amplitude >= minimum_amplitude)
        & (normalized_rmse <= maximum_normalized_rmse)
    )
    if not np.any(valid):
        raise ValueError('no zero-zeta frequency candidate passed the fit gates')
    valid_indices = np.flatnonzero(valid)
    best = int(valid_indices[np.argmin(normalized_rmse[valid])])
    coef = coefficients[best]
    omega = float(omegas[best])
    return ZeroZetaEstimate(
        shaper_half_period_s=float(half_periods[best]),
        omega_n_rad_s=omega,
        frequency_hz=omega / (2.0 * math.pi),
        zeta=0.0,
        offset=float(coef[0]),
        drift_per_s=float(coef[1]),
        cosine_coefficient=float(coef[2]),
        sine_coefficient=float(coef[3]),
        amplitude=float(amplitude[best]),
        phase_rad=math.atan2(-float(coef[3]), float(coef[2])),
        rmse=float(rmse[best]),
        normalized_rmse=float(normalized_rmse[best]),
        peak_to_peak=peak_to_peak,
        condition=float(condition[best]),
        sample_count=int(len(times)),
        window_duration_s=float(times[-1] - times[0]),
    )


def solve_ais_schedule(
    omega_n_rad_s: float,
    *,
    K: float = 0.20,
    tau_s: float = 3.7,
) -> AisSchedule:
    """Solve the final paper-style undamped AIS amplitudes and switch delay.

    This is the closed-form cubic/root calculation used by
    ``_colleague_paper_closed_form_amplitudes``.  The shortest positive,
    forward-only solution is returned.
    """
    omega = float(omega_n_rad_s)
    if not math.isfinite(omega) or omega <= 0.0:
        raise ValueError('omega_n_rad_s must be finite and positive')
    if not math.isfinite(K) or not 0.0 < K < 1.0:
        raise ValueError('K must be in (0, 1)')
    if not math.isfinite(tau_s) or tau_s < 0.0:
        raise ValueError('tau_s must be finite and nonnegative')

    phi = omega * tau_s
    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)
    if abs(sin_phi) < 1.0e-9:
        raise ValueError('AIS closed form is singular for this omega*tau phase')
    polynomial = np.trim_zeros(np.array([
        -K * sin_phi,
        K - 1.0 + 3.0 * K * cos_phi,
        3.0 * K * sin_phi,
        K - K * cos_phi - 1.0,
    ]), trim='f')

    candidates: list[tuple[float, float]] = []
    for root in np.roots(polynomial):
        root = complex(root)
        if abs(root.imag) > 1.0e-7:
            continue
        base_delay = 2.0 * math.atan(float(root.real)) / omega
        for branch in range(-5, 6):
            delay = base_delay + 2.0 * math.pi * branch / omega
            if delay <= 0.0:
                continue
            denominator = (
                math.cos(omega * (tau_s + delay))
                - math.cos(omega * (tau_s + 2.0 * delay))
            )
            if abs(denominator) < 1.0e-9:
                continue
            middle_amplitude = -(
                K - math.cos(omega * (tau_s + 2.0 * delay)) * (K - 1.0)
            ) / denominator
            third_amplitude = 1.0 - K - middle_amplitude
            if (
                math.isfinite(middle_amplitude)
                and 0.0 < middle_amplitude < 1.0 - K
                and third_amplitude >= 0.0
            ):
                candidates.append((middle_amplitude, delay))
    if not candidates:
        raise ValueError('no positive forward-only AIS schedule exists')
    middle_amplitude, delay = min(candidates, key=lambda item: item[1])
    schedule = AisSchedule(
        omega_n_rad_s=omega,
        modal_half_period_s=math.pi / omega,
        K=K,
        A=middle_amplitude,
        A2=1.0 - K - middle_amplitude,
        tau_s=tau_s,
        schedule_delay_s=delay,
    )
    if abs(schedule.residual_phasor()) > 1.0e-6:
        raise ValueError('AIS schedule failed its nominal residual check')
    return schedule


def _load_csv(path: Path, time_column: str, signal_column: str) -> tuple[list[float], list[float]]:
    times: list[float] = []
    signal: list[float] = []
    with path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            try:
                times.append(float(row[time_column]))
                signal.append(float(row[signal_column]))
            except (KeyError, TypeError, ValueError):
                continue
    return times, signal


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv_path', type=Path)
    parser.add_argument('--time-column', default='time_s')
    parser.add_argument('--signal-column', default='swing_mm')
    parser.add_argument('--half-period-min-s', type=float, default=0.45)
    parser.add_argument('--half-period-max-s', type=float, default=1.45)
    parser.add_argument('--grid-count', type=int, default=240)
    parser.add_argument('--minimum-samples', type=int, default=80)
    parser.add_argument('--minimum-p2p', type=float, default=4.0)
    parser.add_argument('--minimum-amplitude', type=float, default=1.0)
    parser.add_argument('--maximum-nrmse', type=float, default=1.5)
    parser.add_argument('--K', type=float, default=0.20)
    parser.add_argument('--tau-s', type=float, default=3.7)
    parser.add_argument(
        '--pulse-duration-s', type=float, default=0.0,
        help='If positive, also verify that the shaped pulse is feasible.',
    )
    args = parser.parse_args()
    times, signal = _load_csv(args.csv_path, args.time_column, args.signal_column)
    estimate = identify_zero_zeta(
        times,
        signal,
        half_period_min_s=args.half_period_min_s,
        half_period_max_s=args.half_period_max_s,
        grid_count=args.grid_count,
        minimum_samples=args.minimum_samples,
        minimum_peak_to_peak=args.minimum_p2p,
        minimum_amplitude=args.minimum_amplitude,
        maximum_normalized_rmse=args.maximum_nrmse,
    )
    schedule = solve_ais_schedule(
        estimate.omega_n_rad_s,
        K=args.K,
        tau_s=args.tau_s,
    )
    if args.pulse_duration_s > 0.0:
        schedule.velocity_gain(0.0, args.pulse_duration_s)
    print(json.dumps({
        'system_id': asdict(estimate),
        'ais_schedule': {
            **asdict(schedule),
            'amplitudes': schedule.amplitudes,
            'impulse_times_s': schedule.impulse_times_s,
            'nominal_residual_magnitude': abs(schedule.residual_phasor()),
        },
    }, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
