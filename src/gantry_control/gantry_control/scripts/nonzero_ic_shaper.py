#!/usr/bin/env python3
"""Closed-form input shaper for an undamped crane with a nonzero initial state.

The state is ``[payload position, payload velocity, trolley position]``.  The
coordinate origin is translated to the trolley position at motion start, so
``initial_swing_mm`` is payload position minus trolley position.  Inputs are
expressed in the positive direction of the requested move.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class NonzeroIcShaper:
    """Two-impulse velocity shaper and its verified terminal residual."""

    A0: float
    A1: float
    switch_time_s: float
    move_duration_s: float
    omega_n_rad_s: float
    initial_swing_mm: float
    initial_payload_velocity_mm_s: float
    terminal_position_residual_mm: float
    terminal_velocity_residual_mm_s: float

    @property
    def duration_s(self) -> float:
        return self.move_duration_s + self.switch_time_s

    @property
    def is_forward_only(self) -> bool:
        """Whether every commanded velocity is between zero and ``vmax``."""
        return 0.0 <= self.A0 <= 1.0 and 0.0 <= self.A1 <= 1.0

    def gain_at(self, time_s: float) -> float:
        """Return the commanded velocity as a fraction of maximum speed."""
        if not math.isfinite(time_s) or time_s < 0.0:
            return 0.0
        if time_s < self.switch_time_s:
            return self.A0
        if time_s < self.move_duration_s:
            return self.A0 + self.A1
        if time_s < self.duration_s:
            return self.A1
        return 0.0


@dataclass(frozen=True)
class RobustNonzeroIcShaper:
    """Positive-weight NZIC profile optimized over a frequency band.

    ``start_amplitudes`` are positive velocity increments applied at
    ``impulse_times_s``.  ``stop_amplitudes`` are positive decrements applied
    at the same delays after ``move_duration_s``.  Unit sums make the command
    rise monotonically from zero to ``vmax`` and return monotonically to zero;
    equal first moments preserve the requested travel exactly.
    """

    impulse_times_s: tuple[float, float, float]
    start_amplitudes: tuple[float, float, float]
    stop_amplitudes: tuple[float, float, float]
    move_duration_s: float
    omega_n_rad_s: float
    initial_swing_mm: float
    initial_payload_velocity_mm_s: float
    terminal_position_residual_mm: float
    terminal_velocity_residual_mm_s: float
    frequency_band_fraction: float
    worst_case_residual_fraction: float
    baseline_worst_case_residual_fraction: float
    optimizer_iterations: int

    @property
    def duration_s(self) -> float:
        return self.move_duration_s + self.impulse_times_s[-1]

    @property
    def tail_s(self) -> float:
        return self.impulse_times_s[-1]

    def gain_events(self) -> tuple[tuple[float, float], ...]:
        """Return merged ``(time, velocity/vmax)`` staircase events."""
        changes: list[tuple[float, float]] = []
        for delay_s, amplitude in zip(
            self.impulse_times_s, self.start_amplitudes
        ):
            changes.append((delay_s, amplitude))
        for delay_s, amplitude in zip(
            self.impulse_times_s, self.stop_amplitudes
        ):
            changes.append((self.move_duration_s + delay_s, -amplitude))
        changes.sort(key=lambda item: item[0])

        events: list[tuple[float, float]] = []
        gain = 0.0
        index = 0
        while index < len(changes):
            event_time = changes[index][0]
            while (
                index < len(changes)
                and math.isclose(
                    changes[index][0], event_time,
                    rel_tol=0.0, abs_tol=1.0e-10,
                )
            ):
                gain += changes[index][1]
                index += 1
            if abs(gain) < 1.0e-12:
                gain = 0.0
            elif abs(gain - 1.0) < 1.0e-12:
                gain = 1.0
            if (
                events
                and math.isclose(
                    events[-1][1], gain, rel_tol=0.0, abs_tol=1.0e-12
                )
            ):
                continue
            events.append((event_time, gain))
        return tuple(events)

    def gain_at(self, time_s: float) -> float:
        if not math.isfinite(time_s) or time_s < 0.0:
            return 0.0
        gain = 0.0
        for event_time, event_gain in self.gain_events():
            if time_s < event_time:
                break
            gain = event_gain
        return gain


@dataclass(frozen=True)
class FreeSwingFrequencyEstimate:
    """Zero-damping sinusoid fitted to a stationary-trolley swing window."""

    omega_n_rad_s: float
    offset_mm: float
    cosine_coefficient_mm: float
    sine_coefficient_mm: float
    amplitude_mm: float
    rmse_mm: float
    normalized_rmse: float
    reference_time_s: float
    window_duration_s: float
    sample_count: int

    def oscillation_state_at(self, time_s: float) -> tuple[float, float]:
        """Return fitted displacement and velocity, excluding sensor offset."""
        phase = self.omega_n_rad_s * (float(time_s) - self.reference_time_s)
        cosine = math.cos(phase)
        sine = math.sin(phase)
        position_mm = (
            self.cosine_coefficient_mm * cosine
            + self.sine_coefficient_mm * sine
        )
        velocity_mm_s = self.omega_n_rad_s * (
            -self.cosine_coefficient_mm * sine
            + self.sine_coefficient_mm * cosine
        )
        return position_mm, velocity_mm_s


@dataclass(frozen=True)
class FiniteAmplitudeFrequencyCorrection:
    """Map a measured finite-amplitude period to the pendulum's linear pole."""

    finite_amplitude_omega_rad_s: float
    small_angle_omega_rad_s: float
    amplitude_angle_rad: float
    correction_factor: float


def correct_finite_amplitude_frequency(
    finite_amplitude_omega_rad_s: float,
    amplitude_mm: float,
    rope_length_m: float,
) -> FiniteAmplitudeFrequencyCorrection:
    """Return the exact undamped simple-pendulum small-angle frequency.

    The free-swing fit observes the finite-amplitude angular frequency

        omega_f = pi * omega_0 / (2*K(sin(theta_0/2)^2)),

    while the linear nonzero-IC model requires ``omega_0``.  ``amplitude_mm``
    is the fitted horizontal payload displacement amplitude, so
    ``theta_0 = asin(amplitude / rope_length)``.  The complete elliptic
    integral is evaluated through its arithmetic-geometric-mean identity,
    keeping this realtime helper independent of an iterative ODE fit.
    """
    omega_f = float(finite_amplitude_omega_rad_s)
    amplitude_m = abs(float(amplitude_mm)) * 1.0e-3
    length_m = float(rope_length_m)
    if not math.isfinite(omega_f) or omega_f <= 0.0:
        raise ValueError('finite-amplitude frequency must be positive and finite')
    if not math.isfinite(amplitude_m):
        raise ValueError('finite-amplitude displacement must be finite')
    if not math.isfinite(length_m) or length_m <= 0.0:
        raise ValueError('rope length must be positive and finite')
    amplitude_ratio = amplitude_m / length_m
    if amplitude_ratio >= 1.0:
        raise ValueError(
            'fitted swing amplitude must be smaller than the rope length')

    theta_0 = math.asin(amplitude_ratio)
    elliptic_parameter = math.sin(0.5 * theta_0) ** 2
    arithmetic = 1.0
    geometric = math.sqrt(1.0 - elliptic_parameter)
    for _ in range(32):
        next_arithmetic = 0.5 * (arithmetic + geometric)
        next_geometric = math.sqrt(arithmetic * geometric)
        arithmetic, geometric = next_arithmetic, next_geometric
        if abs(arithmetic - geometric) <= 1.0e-15 * arithmetic:
            break
    correction_factor = 1.0 / arithmetic
    return FiniteAmplitudeFrequencyCorrection(
        finite_amplitude_omega_rad_s=omega_f,
        small_angle_omega_rad_s=omega_f * correction_factor,
        amplitude_angle_rad=theta_0,
        correction_factor=correction_factor,
    )


def estimate_free_swing_frequency(
    samples: Iterable[tuple[float, float]],
    *,
    omega_min_rad_s: float,
    omega_max_rad_s: float,
    grid_count: int = 401,
) -> FreeSwingFrequencyEstimate:
    """Fit ``offset + C*cos(omega*t) + S*sin(omega*t)`` by grid search.

    Times may be irregular.  The final sample is used as the phase reference,
    which makes the fitted state at capture numerically well conditioned.
    """
    if (
        not math.isfinite(omega_min_rad_s)
        or not math.isfinite(omega_max_rad_s)
        or omega_min_rad_s <= 0.0
        or omega_max_rad_s <= omega_min_rad_s
    ):
        raise ValueError('frequency search bounds must satisfy 0 < min < max')
    if grid_count < 5:
        raise ValueError('frequency grid_count must be at least 5')

    finite_samples = [
        (float(time_s), float(position_mm))
        for time_s, position_mm in samples
        if math.isfinite(time_s) and math.isfinite(position_mm)
    ]
    if len(finite_samples) < 6:
        raise ValueError('free-swing fit needs at least 6 finite samples')
    finite_samples.sort(key=lambda item: item[0])
    times = np.asarray([item[0] for item in finite_samples], dtype=float)
    positions = np.asarray([item[1] for item in finite_samples], dtype=float)
    if times[-1] - times[0] <= 0.0:
        raise ValueError('free-swing fit window must have positive duration')
    relative_times = times - times[-1]

    frequencies = np.linspace(
        omega_min_rad_s, omega_max_rad_s, int(grid_count), dtype=float
    )

    def fit_at(omega: float) -> tuple[np.ndarray, float]:
        design = np.column_stack((
            np.ones_like(relative_times),
            np.cos(omega * relative_times),
            np.sin(omega * relative_times),
        ))
        coefficients, _, _, _ = np.linalg.lstsq(design, positions, rcond=None)
        residual = positions - design @ coefficients
        return coefficients, float(np.mean(residual * residual))

    # Evaluate the complete frequency grid as one batch of 3x3 normal
    # equations.  The earlier per-frequency ``lstsq`` loop occupied the
    # single-threaded ROS executor for roughly 65--75 ms with a 5 s/100 Hz
    # window.  During that pause payload samples were queued or dropped, which
    # made an otherwise continuous pre-motion sinusoid look discontinuous in
    # the run log.  This batched form is algebraically equivalent and leaves
    # only the final refined candidate for the more robust ``lstsq`` call.
    phases = frequencies[:, None] * relative_times[None, :]
    cosine = np.cos(phases)
    sine = np.sin(phases)
    sample_count = float(len(relative_times))

    normal = np.empty((len(frequencies), 3, 3), dtype=float)
    normal[:, 0, 0] = sample_count
    normal[:, 0, 1] = normal[:, 1, 0] = np.sum(cosine, axis=1)
    normal[:, 0, 2] = normal[:, 2, 0] = np.sum(sine, axis=1)
    normal[:, 1, 1] = np.sum(cosine * cosine, axis=1)
    normal[:, 1, 2] = normal[:, 2, 1] = np.sum(cosine * sine, axis=1)
    normal[:, 2, 2] = np.sum(sine * sine, axis=1)

    rhs = np.empty((len(frequencies), 3), dtype=float)
    rhs[:, 0] = np.sum(positions)
    rhs[:, 1] = cosine @ positions
    rhs[:, 2] = sine @ positions
    coefficients_grid = np.linalg.solve(normal, rhs)

    # SSE = y'y - 2*b'X'y + b'X'Xb.  Avoid constructing a prediction matrix
    # for every candidate while retaining the same least-squares objective.
    position_energy = float(positions @ positions)
    errors = (
        position_energy
        - 2.0 * np.einsum('gi,gi->g', coefficients_grid, rhs)
        + np.einsum(
            'gi,gij,gj->g', coefficients_grid, normal, coefficients_grid,
            optimize=True,
        )
    ) / sample_count
    errors = np.maximum(errors, 0.0)
    best_index = int(np.argmin(errors))
    best_omega = float(frequencies[best_index])

    # Quadratic interpolation provides sub-grid frequency precision without a
    # dependency on an iterative optimizer.
    if 0 < best_index < len(frequencies) - 1:
        left_error = float(errors[best_index - 1])
        center_error = float(errors[best_index])
        right_error = float(errors[best_index + 1])
        curvature = left_error - 2.0 * center_error + right_error
        if curvature > 1.0e-20:
            grid_step = float(frequencies[1] - frequencies[0])
            offset_steps = 0.5 * (left_error - right_error) / curvature
            offset_steps = max(-1.0, min(1.0, offset_steps))
            best_omega += offset_steps * grid_step

    coefficients, mean_square_error = fit_at(best_omega)
    offset_mm, cosine_mm, sine_mm = (float(value) for value in coefficients)
    amplitude_mm = math.hypot(cosine_mm, sine_mm)
    rmse_mm = math.sqrt(max(mean_square_error, 0.0))
    normalized_rmse = rmse_mm / max(amplitude_mm, 1.0e-12)
    return FreeSwingFrequencyEstimate(
        omega_n_rad_s=best_omega,
        offset_mm=offset_mm,
        cosine_coefficient_mm=cosine_mm,
        sine_coefficient_mm=sine_mm,
        amplitude_mm=amplitude_mm,
        rmse_mm=rmse_mm,
        normalized_rmse=normalized_rmse,
        reference_time_s=float(times[-1]),
        window_duration_s=float(times[-1] - times[0]),
        sample_count=len(times),
    )


def _terminal_residuals(
    *,
    initial_swing_mm: float,
    initial_payload_velocity_mm_s: float,
    maximum_speed_mm_s: float,
    omega_n_rad_s: float,
    move_duration_s: float,
    switch_time_s: float,
    A0: float,
    A1: float,
) -> tuple[float, float]:
    """Evaluate the two terminal equations used by the closed-form design."""
    x0 = initial_swing_mm
    v0 = initial_payload_velocity_mm_s
    vmax = maximum_speed_mm_s
    omega = omega_n_rad_s
    tp = move_duration_s
    ts = switch_time_s

    # The source equations are written as velocity and omega*position
    # residuals.  Divide the latter by omega to return millimetres.
    velocity_residual = (
        -v0
        + vmax * A0
        + vmax * A1 * math.cos(omega * ts)
        - vmax * A0 * math.cos(omega * tp)
        - vmax * A1 * math.cos(omega * (ts + tp))
    )
    omega_position_residual = (
        -omega * x0
        + vmax * (
            -A1 * math.sin(omega * ts)
            + A0 * math.sin(omega * tp)
            + A1 * math.sin(omega * (ts + tp))
        )
    )
    return omega_position_residual / omega, velocity_residual


def solve_nonzero_ic_shaper(
    *,
    initial_swing_mm: float,
    initial_payload_velocity_mm_s: float,
    maximum_speed_mm_s: float,
    omega_n_rad_s: float,
    move_duration_s: float,
    maximum_absolute_gain: float | None = None,
    residual_tolerance: float = 1.0e-8,
) -> NonzeroIcShaper:
    """Solve the two-impulse nonzero-initial-condition shaper.

    This is the closed-form solution in the supplied MATLAB example.  The
    returned command is ``A0*vmax`` until ``Ts``, ``vmax`` until ``tp``, and
    ``A1*vmax`` until ``tp+Ts``.  The supplied equations can legitimately
    return a negative coefficient or one greater than unity.  Set
    ``maximum_absolute_gain`` to impose a hardware command-speed bound without
    changing that closed-form solution.
    """
    values = {
        'initial_swing_mm': initial_swing_mm,
        'initial_payload_velocity_mm_s': initial_payload_velocity_mm_s,
        'maximum_speed_mm_s': maximum_speed_mm_s,
        'omega_n_rad_s': omega_n_rad_s,
        'move_duration_s': move_duration_s,
        'residual_tolerance': residual_tolerance,
    }
    for name, value in values.items():
        if not math.isfinite(value):
            raise ValueError(f'{name} must be finite')
    if maximum_speed_mm_s <= 0.0:
        raise ValueError('maximum_speed_mm_s must be positive')
    if omega_n_rad_s <= 0.0:
        raise ValueError('omega_n_rad_s must be positive')
    if move_duration_s <= 0.0:
        raise ValueError('move_duration_s must be positive')
    if residual_tolerance <= 0.0:
        raise ValueError('residual_tolerance must be positive')
    if maximum_absolute_gain is not None and (
        not math.isfinite(maximum_absolute_gain)
        or maximum_absolute_gain < 1.0
    ):
        raise ValueError('maximum_absolute_gain must be finite and at least 1')

    x0 = float(initial_swing_mm)
    v0 = float(initial_payload_velocity_mm_s)
    vmax = float(maximum_speed_mm_s)
    omega = float(omega_n_rad_s)
    tp = float(move_duration_s)

    # The zero-state limit of the supplied formula is 0/0.  Its continuous
    # physical solution is the ordinary undamped ZV shaper.
    normalized_ic = math.hypot(omega * x0 / vmax, v0 / vmax)
    if normalized_ic <= 1.0e-12:
        switch_time_s = math.pi / omega
        A0 = 0.5
        A1 = 0.5
    else:
        phase = omega * tp
        sin_phase = math.sin(phase)
        cos_phase = math.cos(phase)
        alpha_numerator = (
            -2.0 * vmax
            + v0
            + 2.0 * vmax * cos_phase
            - v0 * cos_phase
            + omega * x0 * sin_phase
        )
        alpha_denominator = (
            omega * x0
            - v0 * sin_phase
            - omega * x0 * cos_phase
        )

        # This is algebraically atan2(2*alpha/(alpha^2+1),
        # (1-alpha^2)/(alpha^2+1)), but remains well-conditioned when the
        # denominator of alpha approaches zero.
        scale = math.hypot(alpha_numerator, alpha_denominator)
        if scale <= 1.0e-14 * max(vmax, abs(omega * x0), abs(v0), 1.0):
            raise ValueError('nonzero-IC phase equation is singular for this move')
        numerator_scaled = alpha_numerator / scale
        denominator_scaled = alpha_denominator / scale
        switch_phase = math.atan2(
            2.0 * numerator_scaled * denominator_scaled,
            denominator_scaled**2 - numerator_scaled**2,
        )
        if switch_phase <= 1.0e-12:
            switch_phase += 2.0 * math.pi
        switch_time_s = switch_phase / omega

        amplitude_denominator = (
            -math.sin(switch_phase)
            - sin_phase
            + sin_phase * math.cos(switch_phase)
            + math.sin(switch_phase) * cos_phase
        )
        if abs(amplitude_denominator) <= 1.0e-12:
            raise ValueError('nonzero-IC amplitude equation is singular for this move')
        A1 = (omega * x0 / vmax - sin_phase) / amplitude_denominator
        A0 = 1.0 - A1

    time_tolerance_s = 1.0e-10 * max(1.0, tp)
    if switch_time_s <= 0.0 or switch_time_s > tp + time_tolerance_s:
        raise ValueError(
            f'nonzero-IC switch time Ts={switch_time_s:.6f}s is outside '
            f'the monotonic interval (0, tp={tp:.6f}s]'
        )
    switch_time_s = min(switch_time_s, tp)

    if not (math.isfinite(A0) and math.isfinite(A1)):
        raise ValueError('nonzero-IC solution produced non-finite amplitudes')
    if maximum_absolute_gain is not None and max(
        abs(A0), abs(A1), 1.0
    ) > maximum_absolute_gain + 1.0e-10:
        raise ValueError(
            f'nonzero-IC solution exceeds the command-speed bound: '
            f'A0={A0:.6f}, A1={A1:.6f}, '
            f'maximum absolute gain={maximum_absolute_gain:.6f}'
        )

    position_residual, velocity_residual = _terminal_residuals(
        initial_swing_mm=x0,
        initial_payload_velocity_mm_s=v0,
        maximum_speed_mm_s=vmax,
        omega_n_rad_s=omega,
        move_duration_s=tp,
        switch_time_s=switch_time_s,
        A0=A0,
        A1=A1,
    )
    residual_scale = max(abs(x0), abs(v0) / omega, vmax / omega, 1.0)
    normalized_residual = max(
        abs(position_residual), abs(velocity_residual) / omega
    ) / residual_scale
    if normalized_residual > residual_tolerance:
        raise ValueError(
            'closed-form nonzero-IC solution failed terminal verification: '
            f'position={position_residual:.6g}mm, '
            f'velocity={velocity_residual:.6g}mm/s'
        )

    return NonzeroIcShaper(
        A0=A0,
        A1=A1,
        switch_time_s=switch_time_s,
        move_duration_s=tp,
        omega_n_rad_s=omega,
        initial_swing_mm=x0,
        initial_payload_velocity_mm_s=v0,
        terminal_position_residual_mm=position_residual,
        terminal_velocity_residual_mm_s=velocity_residual,
    )


def _robust_terminal_state(
    *,
    initial_swing_mm: float,
    initial_payload_velocity_mm_s: float,
    maximum_speed_mm_s: float,
    omega_rad_s: float,
    move_duration_s: float,
    impulse_times_s: np.ndarray,
    start_amplitudes: np.ndarray,
    stop_amplitudes: np.ndarray,
) -> np.ndarray:
    """Return terminal ``[position, velocity/omega]`` for one staircase."""

    final_time_s = move_duration_s + float(impulse_times_s[-1])

    def transition(elapsed_s: float) -> np.ndarray:
        phase = omega_rad_s * elapsed_s
        cosine = math.cos(phase)
        sine = math.sin(phase)
        return np.asarray(((cosine, sine), (-sine, cosine)), dtype=float)

    state = transition(final_time_s) @ np.asarray((
        initial_swing_mm,
        initial_payload_velocity_mm_s / omega_rad_s,
    ))
    velocity_jump = maximum_speed_mm_s / omega_rad_s
    for delay_s, amplitude in zip(impulse_times_s, start_amplitudes):
        state += transition(final_time_s - float(delay_s)) @ np.asarray((
            0.0, -velocity_jump * float(amplitude),
        ))
    for delay_s, amplitude in zip(impulse_times_s, stop_amplitudes):
        state += transition(
            final_time_s - move_duration_s - float(delay_s)
        ) @ np.asarray((0.0, velocity_jump * float(amplitude)))
    return state


def solve_robust_nonzero_ic_shaper(
    *,
    initial_swing_mm: float,
    initial_payload_velocity_mm_s: float,
    maximum_speed_mm_s: float,
    omega_n_rad_s: float,
    move_duration_s: float,
    frequency_band_fraction: float = 0.05,
    band_sample_count: int = 21,
    residual_tolerance: float = 1.0e-7,
) -> RobustNonzeroIcShaper:
    """Optimize a forward-only NZIC staircase over a frequency band.

    The ordinary two-impulse closed form supplies an exactly feasible seed.
    One intermediate start/stop opportunity and separate positive start and
    stop weights add freedom to reduce off-nominal residual.  The optimizer
    enforces unity start/stop sums, equal first moments (exact travel), and
    exact nominal terminal position and velocity.  It then minimizes a smooth
    approximation of the worst residual across the requested frequency band.

    This is a specified-insensitivity robust NZIC profile, not the existing
    zero-initial-condition ZVD baseline.
    """

    values = {
        'initial_swing_mm': initial_swing_mm,
        'initial_payload_velocity_mm_s': initial_payload_velocity_mm_s,
        'maximum_speed_mm_s': maximum_speed_mm_s,
        'omega_n_rad_s': omega_n_rad_s,
        'move_duration_s': move_duration_s,
        'residual_tolerance': residual_tolerance,
    }
    if any(not math.isfinite(float(value)) for value in values.values()):
        raise ValueError('robust nonzero-IC inputs must be finite')
    if maximum_speed_mm_s <= 0.0 or omega_n_rad_s <= 0.0:
        raise ValueError('robust nonzero-IC speed and frequency must be positive')
    if move_duration_s <= 0.0:
        raise ValueError('robust nonzero-IC move duration must be positive')
    if (
        not math.isfinite(frequency_band_fraction)
        or not 0.0 < frequency_band_fraction < 0.25
    ):
        raise ValueError('robust frequency band fraction must be in (0, 0.25)')
    if band_sample_count < 5:
        raise ValueError('robust frequency band needs at least 5 samples')
    if residual_tolerance <= 0.0:
        raise ValueError('robust residual tolerance must be positive')

    baseline = solve_nonzero_ic_shaper(
        initial_swing_mm=initial_swing_mm,
        initial_payload_velocity_mm_s=initial_payload_velocity_mm_s,
        maximum_speed_mm_s=maximum_speed_mm_s,
        omega_n_rad_s=omega_n_rad_s,
        move_duration_s=move_duration_s,
        maximum_absolute_gain=1.0,
        residual_tolerance=residual_tolerance,
    )
    initial_amplitude_mm = math.hypot(
        initial_swing_mm,
        initial_payload_velocity_mm_s / omega_n_rad_s,
    )
    if initial_amplitude_mm < 1.0e-6:
        raise ValueError(
            'robust nonzero-IC optimization needs a measurable initial state')

    # Variables are start weights p[0:3], stop weights q[0:3], and the two
    # nonzero impulse times.  The seed exactly reproduces the closed form.
    initial_variables = np.asarray((
        baseline.A0, 0.0, baseline.A1,
        baseline.A0, 0.0, baseline.A1,
        0.5 * baseline.switch_time_s, baseline.switch_time_s,
    ), dtype=float)
    sample_scales = np.linspace(
        1.0 - frequency_band_fraction,
        1.0 + frequency_band_fraction,
        int(band_sample_count),
    )

    def unpack(variables: np.ndarray):
        return (
            variables[:3],
            variables[3:6],
            np.asarray((0.0, variables[6], variables[7])),
        )

    def normalized_profile_terminals(
        times: np.ndarray,
        start: np.ndarray,
        stop: np.ndarray,
        frequency_scales: np.ndarray,
    ) -> np.ndarray:
        scales = np.asarray(frequency_scales, dtype=float)
        omegas = omega_n_rad_s * scales
        final_time_s = move_duration_s + times[-1]
        initial_phase = omegas * final_time_s
        position = (
            initial_swing_mm * np.cos(initial_phase)
            + initial_payload_velocity_mm_s / omegas * np.sin(initial_phase)
        )
        velocity_over_omega = (
            -initial_swing_mm * np.sin(initial_phase)
            + initial_payload_velocity_mm_s / omegas * np.cos(initial_phase)
        )
        jump = maximum_speed_mm_s / omegas
        for delay_s, amplitude in zip(times, start):
            phase = omegas * (final_time_s - delay_s)
            position -= jump * amplitude * np.sin(phase)
            velocity_over_omega -= jump * amplitude * np.cos(phase)
        for delay_s, amplitude in zip(times, stop):
            phase = omegas * (final_time_s - move_duration_s - delay_s)
            position += jump * amplitude * np.sin(phase)
            velocity_over_omega += jump * amplitude * np.cos(phase)
        return np.column_stack((position, velocity_over_omega)) / initial_amplitude_mm

    def normalized_terminals(
        variables: np.ndarray, frequency_scales: np.ndarray
    ) -> np.ndarray:
        start, stop, times = unpack(variables)
        return normalized_profile_terminals(
            times, start, stop, frequency_scales)

    def normalized_terminal(
        variables: np.ndarray, frequency_scale: float
    ) -> np.ndarray:
        return normalized_terminals(
            variables, np.asarray((frequency_scale,))
        )[0]

    def equality_constraints(variables: np.ndarray) -> np.ndarray:
        start, stop, times = unpack(variables)
        nominal = normalized_terminal(variables, 1.0)
        return np.asarray((
            float(np.sum(start) - 1.0),
            float(np.sum(stop) - 1.0),
            float(np.dot(start - stop, times) / move_duration_s),
            float(nominal[0]),
            float(nominal[1]),
        ))

    def band_objective(variables: np.ndarray) -> float:
        residuals = normalized_terminals(variables, sample_scales)
        residual_norms = np.linalg.norm(residuals, axis=1)
        # An eighth norm is smooth for SLSQP but emphasizes the edge/worst
        # residual much more strongly than an ordinary least-squares mean.
        return float(np.sum((residual_norms**2 + 1.0e-16) ** 4) ** 0.125)

    minimum_time_gap_s = 1.0e-4
    constraints = [
        {'type': 'eq', 'fun': equality_constraints},
        {
            'type': 'ineq',
            'fun': lambda variables: variables[6] - minimum_time_gap_s,
        },
        {
            'type': 'ineq',
            'fun': lambda variables: (
                variables[7] - variables[6] - minimum_time_gap_s
            ),
        },
        {
            'type': 'ineq',
            'fun': lambda variables: move_duration_s - variables[7],
        },
    ]
    result = minimize(
        band_objective,
        initial_variables,
        method='SLSQP',
        bounds=[(0.0, 1.0)] * 6 + [
            (minimum_time_gap_s, move_duration_s),
            (2.0 * minimum_time_gap_s, move_duration_s),
        ],
        constraints=constraints,
        options={'ftol': 1.0e-12, 'maxiter': 1000, 'disp': False},
    )
    equality_error = float(np.max(np.abs(equality_constraints(result.x))))
    if not result.success or equality_error > residual_tolerance:
        raise ValueError(
            'no forward-only robust nonzero-IC schedule was found: '
            f'{result.message}; equality error={equality_error:.3e}')

    start_raw, stop_raw, times_raw = unpack(result.x)
    start = np.asarray([
        0.0 if abs(value) < 1.0e-10 else float(value)
        for value in start_raw
    ])
    stop = np.asarray([
        0.0 if abs(value) < 1.0e-10 else float(value)
        for value in stop_raw
    ])
    times = np.asarray(times_raw, dtype=float)
    terminal = _robust_terminal_state(
        initial_swing_mm=initial_swing_mm,
        initial_payload_velocity_mm_s=initial_payload_velocity_mm_s,
        maximum_speed_mm_s=maximum_speed_mm_s,
        omega_rad_s=omega_n_rad_s,
        move_duration_s=move_duration_s,
        impulse_times_s=times,
        start_amplitudes=start,
        stop_amplitudes=stop,
    )
    terminal_fraction = max(
        abs(float(terminal[0])), abs(float(terminal[1]))
    ) / initial_amplitude_mm
    if terminal_fraction > residual_tolerance:
        raise ValueError(
            'robust nonzero-IC schedule failed nominal verification: '
            f'normalized residual={terminal_fraction:.3e}')

    dense_scales = np.linspace(
        1.0 - frequency_band_fraction,
        1.0 + frequency_band_fraction,
        101,
    )

    def worst_case(
        impulse_times: np.ndarray,
        start_weights: np.ndarray,
        stop_weights: np.ndarray,
    ) -> float:
        return float(np.max(np.linalg.norm(
            normalized_profile_terminals(
                impulse_times, start_weights, stop_weights, dense_scales
            ),
            axis=1,
        )))

    robust_worst = worst_case(times, start, stop)
    baseline_times = np.asarray((0.0, baseline.switch_time_s))
    baseline_weights = np.asarray((baseline.A0, baseline.A1))
    baseline_worst = worst_case(
        baseline_times, baseline_weights, baseline_weights)
    if robust_worst >= baseline_worst - 1.0e-6:
        raise ValueError(
            'robust nonzero-IC optimization did not improve the frequency band: '
            f'robust={robust_worst:.6f}, baseline={baseline_worst:.6f}')

    robust = RobustNonzeroIcShaper(
        impulse_times_s=tuple(float(value) for value in times),
        start_amplitudes=tuple(float(value) for value in start),
        stop_amplitudes=tuple(float(value) for value in stop),
        move_duration_s=float(move_duration_s),
        omega_n_rad_s=float(omega_n_rad_s),
        initial_swing_mm=float(initial_swing_mm),
        initial_payload_velocity_mm_s=float(initial_payload_velocity_mm_s),
        terminal_position_residual_mm=float(terminal[0]),
        terminal_velocity_residual_mm_s=float(terminal[1] * omega_n_rad_s),
        frequency_band_fraction=float(frequency_band_fraction),
        worst_case_residual_fraction=robust_worst,
        baseline_worst_case_residual_fraction=baseline_worst,
        optimizer_iterations=int(getattr(result, 'nit', 0)),
    )
    gains = tuple(gain for _, gain in robust.gain_events())
    if min(gains) < -1.0e-8 or max(gains) > 1.0 + 1.0e-8:
        raise ValueError('robust nonzero-IC command is not forward-only')
    return robust
