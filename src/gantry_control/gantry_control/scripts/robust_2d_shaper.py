#!/usr/bin/env python3
"""Pure robust 2-D input-shaping primitives for the square player.

Besides the conventional zero-initial-condition multi-mode ZVD profile, this
module implements a nonzero-initial-condition (NZIC) shaper.  The NZIC design
changes the *weights* of the existing acceleration and deceleration impulses,
not their times.  Positivity, unit-sum, and first-moment constraints guarantee
that a corrected leg still accelerates monotonically to the requested speed,
decelerates monotonically to zero, and travels exactly the requested distance.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import math
from typing import Iterable

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize


G = 9.80665


@dataclass(frozen=True)
class ShapedLeg:
    """One unshaped constant-velocity leg and its shaped timing envelope."""

    name: str
    start_s: float
    raw_duration_s: float
    vx_mm_s: float
    vy_mm_s: float
    shaper_tail_s: float
    start_amplitudes: tuple[float, ...]
    stop_amplitudes: tuple[float, ...]

    @property
    def raw_stop_s(self) -> float:
        return self.start_s + self.raw_duration_s

    @property
    def shaped_stop_s(self) -> float:
        return self.raw_stop_s + self.shaper_tail_s


@dataclass(frozen=True)
class ZvdMode:
    """One modal ZVD factor before convolution with the other modes."""

    name: str
    natural_frequency_hz: float
    damping_ratio: float
    timing_scale: float
    impulse_spacing_s: float
    amplitudes: tuple[float, float, float]


@dataclass(frozen=True)
class SwayState:
    """Small-angle oscillator state in radians and radians/second."""

    angle_rad: float
    angular_rate_rad_s: float

    def amplitude_rad(self, mode: ZvdMode) -> float:
        """Return the damped sinusoid amplitude represented by this state."""
        omega_n = 2.0 * math.pi * mode.natural_frequency_hz
        omega_d = omega_n * math.sqrt(1.0 - mode.damping_ratio**2)
        quadrature = (
            self.angular_rate_rad_s
            + mode.damping_ratio * omega_n * self.angle_rad
        ) / omega_d
        return math.hypot(self.angle_rad, quadrature)


@dataclass(frozen=True)
class SwayEstimate:
    """Robust harmonic-regression estimate at ``reference_time_s``."""

    state: SwayState
    bias_rad: float
    rmse_rad: float
    sample_count: int
    span_s: float
    reference_time_s: float


@dataclass(frozen=True)
class NzicCorrection:
    """Diagnostics for one corrected Cartesian leg."""

    leg_name: str
    initial_state: SwayState
    initial_amplitude_rad: float
    start_amplitudes: tuple[float, ...]
    stop_amplitudes: tuple[float, ...]
    nominal_residual_rad: float
    primary_band_residual_rad: float
    uncorrected_primary_band_residual_rad: float
    second_mode_band_residual_fraction: float
    equality_error: float
    iterations: int


def _make_zvd_mode(
    *,
    name: str,
    natural_frequency_hz: float,
    damping_ratio: float,
    timing_scale: float,
) -> ZvdMode:
    if not math.isfinite(natural_frequency_hz) or natural_frequency_hz <= 0.0:
        raise ValueError(f'{name} frequency must be finite and positive')
    if not math.isfinite(damping_ratio) or not 0.0 <= damping_ratio < 0.99:
        raise ValueError(f'{name} damping ratio must be in [0, 0.99)')
    if not math.isfinite(timing_scale) or timing_scale <= 0.0:
        raise ValueError(f'{name} timing scale must be finite and positive')

    damped_factor = math.sqrt(1.0 - damping_ratio**2)
    omega_d_rad_s = 2.0 * math.pi * natural_frequency_hz * damped_factor
    spacing_s = timing_scale * math.pi / omega_d_rad_s
    decay = math.exp(-math.pi * damping_ratio / max(damped_factor, 1.0e-12))
    denominator = (1.0 + decay) ** 2
    amplitudes = (
        1.0 / denominator,
        2.0 * decay / denominator,
        decay**2 / denominator,
    )
    return ZvdMode(
        name=name,
        natural_frequency_hz=float(natural_frequency_hz),
        damping_ratio=float(damping_ratio),
        timing_scale=float(timing_scale),
        impulse_spacing_s=spacing_s,
        amplitudes=amplitudes,
    )


def _convolve_modes(
    modes: Iterable[ZvdMode],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    impulses = [(0.0, 1.0)]
    for mode in modes:
        impulses = [
            (
                time_s + impulse_index * mode.impulse_spacing_s,
                amplitude * mode.amplitudes[impulse_index],
            )
            for time_s, amplitude in impulses
            for impulse_index in range(3)
        ]
    impulses.sort(key=lambda item: item[0])
    return (
        tuple(item[0] for item in impulses),
        tuple(item[1] for item in impulses),
    )


def oscillator_transition(
    mode: ZvdMode,
    elapsed_s: float,
    *,
    natural_frequency_scale: float = 1.0,
) -> np.ndarray:
    """Return the exact 2x2 transition matrix of a damped oscillator."""
    if not math.isfinite(elapsed_s):
        raise ValueError('elapsed_s must be finite')
    if not math.isfinite(natural_frequency_scale) or natural_frequency_scale <= 0.0:
        raise ValueError('natural_frequency_scale must be finite and positive')
    omega_n = (
        2.0 * math.pi * mode.natural_frequency_hz * natural_frequency_scale
    )
    zeta = mode.damping_ratio
    omega_d = omega_n * math.sqrt(1.0 - zeta**2)
    decay = math.exp(-zeta * omega_n * elapsed_s)
    sine = math.sin(omega_d * elapsed_s)
    cosine = math.cos(omega_d * elapsed_s)
    zeta_term = zeta * omega_n / omega_d
    return decay * np.array([
        [cosine + zeta_term * sine, sine / omega_d],
        [-(omega_n**2) * sine / omega_d, cosine - zeta_term * sine],
    ])


def propagate_sway_state(
    state: SwayState,
    mode: ZvdMode,
    elapsed_s: float,
) -> SwayState:
    """Propagate an unforced sway state forward by ``elapsed_s``."""
    vector = oscillator_transition(mode, elapsed_s) @ np.array([
        state.angle_rad,
        state.angular_rate_rad_s,
    ])
    return SwayState(float(vector[0]), float(vector[1]))


def estimate_sway_state(
    samples: Iterable[tuple[float, float]],
    *,
    mode: ZvdMode,
    reference_time_s: float,
    window_s: float = 2.5,
    minimum_samples: int = 120,
    minimum_span_s: float = 1.5,
) -> SwayEstimate:
    """Estimate current angle/rate using bias-aware robust harmonic regression.

    ``samples`` contains ``(time_s, angle_rad)`` pairs.  A constant encoder
    offset and a damped sinusoid at the configured rope frequency are fitted
    together.  Four Huber reweighting passes prevent an occasional encoder
    packet or count glitch from dominating the phase estimate.
    """
    if not math.isfinite(reference_time_s):
        raise ValueError('reference_time_s must be finite')
    if not math.isfinite(window_s) or window_s <= 0.0:
        raise ValueError('window_s must be finite and positive')
    if minimum_samples < 3:
        raise ValueError('minimum_samples must be at least 3')
    if not math.isfinite(minimum_span_s) or minimum_span_s <= 0.0:
        raise ValueError('minimum_span_s must be finite and positive')

    selected = [
        (float(stamp), float(angle))
        for stamp, angle in samples
        if (
            math.isfinite(stamp)
            and math.isfinite(angle)
            and reference_time_s - window_s <= stamp <= reference_time_s + 1.0e-9
        )
    ]
    if len(selected) < minimum_samples:
        raise ValueError(
            f'need at least {minimum_samples} payload samples; have {len(selected)}'
        )
    selected.sort(key=lambda item: item[0])
    span_s = selected[-1][0] - selected[0][0]
    if span_s < minimum_span_s:
        raise ValueError(
            f'payload history spans {span_s:.3f}s; need {minimum_span_s:.3f}s'
        )

    relative_time = np.asarray(
        [item[0] - reference_time_s for item in selected], dtype=float
    )
    observations = np.asarray([item[1] for item in selected], dtype=float)
    omega_n = 2.0 * math.pi * mode.natural_frequency_hz
    omega_d = omega_n * math.sqrt(1.0 - mode.damping_ratio**2)
    envelope = np.exp(-mode.damping_ratio * omega_n * relative_time)
    design = np.column_stack((
        np.ones_like(relative_time),
        envelope * np.cos(omega_d * relative_time),
        envelope * np.sin(omega_d * relative_time),
    ))
    weights = np.ones(len(observations), dtype=float)
    coefficients = np.zeros(3, dtype=float)
    for _ in range(4):
        weighted_design = design * np.sqrt(weights)[:, None]
        weighted_observations = observations * np.sqrt(weights)
        coefficients, _, _, _ = np.linalg.lstsq(
            weighted_design, weighted_observations, rcond=None
        )
        residual = observations - design @ coefficients
        median = float(np.median(residual))
        robust_sigma = 1.4826 * float(np.median(np.abs(residual - median)))
        # Roughly one magnetic-encoder count at the current configuration is
        # far below this floor.  The floor avoids unstable weights on an
        # almost perfectly sinusoidal data set.
        huber_limit = 1.5 * max(robust_sigma, math.radians(0.002))
        weights = np.minimum(
            1.0,
            huber_limit / np.maximum(np.abs(residual - median), 1.0e-15),
        )

    residual = observations - design @ coefficients
    bias, in_phase, quadrature = (float(value) for value in coefficients)
    angular_rate = (
        -mode.damping_ratio * omega_n * in_phase + omega_d * quadrature
    )
    return SwayEstimate(
        state=SwayState(in_phase, angular_rate),
        bias_rad=bias,
        rmse_rad=float(math.sqrt(
            np.sum(weights * residual**2) / max(np.sum(weights), 1.0)
        )),
        sample_count=len(selected),
        span_s=float(span_s),
        reference_time_s=float(reference_time_s),
    )


class Robust2dSquareProfile:
    """Multi-mode ZVD-shaped ``+X, +Y, -X, -Y`` velocity profile.

    A single vector shaper is applied component-wise.  Legs never overlap: the
    next raw leg starts only after the previous leg's final stop impulse.  This
    makes the commanded path axis-aligned and returns its velocity integral to
    the starting point exactly.
    """

    def __init__(
        self,
        *,
        x_distance_mm: float,
        y_distance_mm: float,
        speed_mm_s: float,
        rope_length_m: float,
        damping_ratio: float = 0.0,
        timing_scale: float = 1.0,
        second_mode_frequency_hz: float | None = None,
        second_mode_damping_ratio: float = 0.0,
        second_mode_timing_scale: float = 1.0,
        corner_dwell_s: float = 0.0,
        gravity_m_s2: float = G,
    ) -> None:
        positive = {
            'x_distance_mm': x_distance_mm,
            'y_distance_mm': y_distance_mm,
            'speed_mm_s': speed_mm_s,
            'rope_length_m': rope_length_m,
            'timing_scale': timing_scale,
            'gravity_m_s2': gravity_m_s2,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f'{name} must be finite and positive')
        if not math.isfinite(corner_dwell_s) or corner_dwell_s < 0.0:
            raise ValueError('corner_dwell_s must be finite and nonnegative')

        self.x_distance_mm = float(x_distance_mm)
        self.y_distance_mm = float(y_distance_mm)
        self.speed_mm_s = float(speed_mm_s)
        self.rope_length_m = float(rope_length_m)
        self.damping_ratio = float(damping_ratio)
        self.timing_scale = float(timing_scale)
        self.second_mode_frequency_hz = (
            None
            if second_mode_frequency_hz is None
            else float(second_mode_frequency_hz)
        )
        self.second_mode_damping_ratio = float(second_mode_damping_ratio)
        self.second_mode_timing_scale = float(second_mode_timing_scale)
        self.corner_dwell_s = float(corner_dwell_s)
        self.gravity_m_s2 = float(gravity_m_s2)

        self.omega_n_rad_s = math.sqrt(self.gravity_m_s2 / self.rope_length_m)
        primary_mode = _make_zvd_mode(
            name='rope mode',
            natural_frequency_hz=self.omega_n_rad_s / (2.0 * math.pi),
            damping_ratio=self.damping_ratio,
            timing_scale=self.timing_scale,
        )
        modes = [primary_mode]
        if self.second_mode_frequency_hz is not None:
            modes.append(_make_zvd_mode(
                name='second mode',
                natural_frequency_hz=self.second_mode_frequency_hz,
                damping_ratio=self.second_mode_damping_ratio,
                timing_scale=self.second_mode_timing_scale,
            ))
        self.modes = tuple(modes)
        self.omega_d_rad_s = (
            2.0 * math.pi * primary_mode.natural_frequency_hz
            * math.sqrt(1.0 - primary_mode.damping_ratio**2)
        )
        # Backwards-compatible name for the rope-mode half period.
        self.impulse_spacing_s = primary_mode.impulse_spacing_s
        self.impulse_times_s, self.amplitudes = _convolve_modes(self.modes)
        self.ic_corrections: tuple[NzicCorrection, ...] = ()

        x_duration = self.x_distance_mm / self.speed_mm_s
        y_duration = self.y_distance_mm / self.speed_mm_s
        shortest_duration = min(x_duration, y_duration)
        if shortest_duration + 1.0e-12 < self.shaper_tail_s:
            minimum_distance = self.speed_mm_s * self.shaper_tail_s
            raise ValueError(
                'raw leg duration must be at least the multi-mode shaper tail '
                f'({self.shaper_tail_s:.4f}s) to keep acceleration and '
                'deceleration monotonic; increase each distance to at least '
                f'{minimum_distance:.1f}mm or reduce speed'
            )
        specifications = (
            ('+X', x_duration, self.speed_mm_s, 0.0),
            ('+Y', y_duration, 0.0, self.speed_mm_s),
            ('-X', x_duration, -self.speed_mm_s, 0.0),
            ('-Y', y_duration, 0.0, -self.speed_mm_s),
        )
        start_s = 0.0
        legs: list[ShapedLeg] = []
        for index, (name, duration, vx, vy) in enumerate(specifications):
            leg = ShapedLeg(
                name=name,
                start_s=start_s,
                raw_duration_s=duration,
                vx_mm_s=vx,
                vy_mm_s=vy,
                shaper_tail_s=self.shaper_tail_s,
                start_amplitudes=self.amplitudes,
                stop_amplitudes=self.amplitudes,
            )
            legs.append(leg)
            if index + 1 < len(specifications):
                start_s = leg.shaped_stop_s + self.corner_dwell_s
        self.legs = tuple(legs)
        self.duration_s = self.legs[-1].shaped_stop_s

    @property
    def shaper_tail_s(self) -> float:
        return self.impulse_times_s[-1]

    @property
    def minimum_monotonic_leg_distance_mm(self) -> float:
        return self.speed_mm_s * self.shaper_tail_s

    @property
    def waypoints_mm(self) -> tuple[tuple[float, float], ...]:
        return (
            (0.0, 0.0),
            (self.x_distance_mm, 0.0),
            (self.x_distance_mm, self.y_distance_mm),
            (0.0, self.y_distance_mm),
            (0.0, 0.0),
        )

    def command_at(self, time_s: float) -> tuple[float, float]:
        """Return the shaped Cartesian velocity at ``time_s`` in mm/s."""
        if not math.isfinite(time_s) or time_s < 0.0 or time_s >= self.duration_s:
            return 0.0, 0.0
        vx = 0.0
        vy = 0.0
        for leg in self.legs:
            for delay_s, start_amplitude, stop_amplitude in zip(
                self.impulse_times_s,
                leg.start_amplitudes,
                leg.stop_amplitudes,
            ):
                delayed_start = leg.start_s + delay_s
                delayed_stop = leg.raw_stop_s + delay_s
                if delayed_start <= time_s:
                    vx += start_amplitude * leg.vx_mm_s
                    vy += start_amplitude * leg.vy_mm_s
                if delayed_stop <= time_s:
                    vx -= stop_amplitude * leg.vx_mm_s
                    vy -= stop_amplitude * leg.vy_mm_s
        return vx, vy

    def displacement_at(self, time_s: float) -> tuple[float, float]:
        """Return the exact integral of the shaped command in millimetres."""
        if not math.isfinite(time_s) or time_s <= 0.0:
            return 0.0, 0.0
        clamped_time = min(time_s, self.duration_s)
        x_mm = 0.0
        y_mm = 0.0
        for leg in self.legs:
            for delay_s, start_amplitude, stop_amplitude in zip(
                self.impulse_times_s,
                leg.start_amplitudes,
                leg.stop_amplitudes,
            ):
                delayed_start = leg.start_s + delay_s
                delayed_stop = leg.raw_stop_s + delay_s
                start_age_s = max(clamped_time - delayed_start, 0.0)
                stop_age_s = max(clamped_time - delayed_stop, 0.0)
                weighted_time_s = (
                    start_amplitude * start_age_s
                    - stop_amplitude * stop_age_s
                )
                x_mm += leg.vx_mm_s * weighted_time_s
                y_mm += leg.vy_mm_s * weighted_time_s
        return x_mm, y_mm

    def phase_at(self, time_s: float) -> str:
        if time_s < 0.0:
            return 'not_started'
        for index, leg in enumerate(self.legs):
            if leg.start_s <= time_s < leg.shaped_stop_s:
                return leg.name
            if index + 1 < len(self.legs):
                next_leg = self.legs[index + 1]
                if leg.shaped_stop_s <= time_s < next_leg.start_s:
                    return f'corner_after_{leg.name}'
        return 'complete'

    def event_times_s(self) -> tuple[float, ...]:
        """Return every command discontinuity, useful for exact validation."""
        events = {0.0, self.duration_s}
        for leg in self.legs:
            for delay_s in self.impulse_times_s:
                delayed_start = leg.start_s + delay_s
                events.add(delayed_start)
                events.add(delayed_start + leg.raw_duration_s)
        return tuple(sorted(events))

    def _terminal_state_map(
        self,
        *,
        leg: ShapedLeg,
        mode: ZvdMode,
        initial_state: SwayState,
        initial_state_time_s: float = 0.0,
        natural_frequency_scale: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``C, d`` such that normalized final state is ``C q + d``.

        ``q`` concatenates the leg's start and stop amplitude vectors.  The
        ``initial_state_time_s`` uses the complete square's time base.  Runtime
        replanning can therefore estimate +Y at its actual delayed start,
        without a long open-loop phase forecast from square time zero.
        """
        omega_n = (
            2.0 * math.pi * mode.natural_frequency_hz
            * natural_frequency_scale
        )
        omega_d = omega_n * math.sqrt(1.0 - mode.damping_ratio**2)
        final_time_s = leg.shaped_stop_s
        component_speed_m_s = 1.0e-3 * (
            leg.vx_mm_s if abs(leg.vx_mm_s) > 0.0 else leg.vy_mm_s
        )
        velocity_jump_state = np.array([0.0, -1.0 / self.rope_length_m])
        normalization = np.diag([1.0, 1.0 / omega_d])

        start_columns = [
            normalization
            @ oscillator_transition(
                mode,
                final_time_s - (leg.start_s + delay_s),
                natural_frequency_scale=natural_frequency_scale,
            )
            @ velocity_jump_state
            * component_speed_m_s
            for delay_s in self.impulse_times_s
        ]
        stop_columns = [
            normalization
            @ oscillator_transition(
                mode,
                final_time_s - (leg.raw_stop_s + delay_s),
                natural_frequency_scale=natural_frequency_scale,
            )
            @ velocity_jump_state
            * (-component_speed_m_s)
            for delay_s in self.impulse_times_s
        ]
        initial_vector = np.array([
            initial_state.angle_rad,
            initial_state.angular_rate_rad_s,
        ])
        constant = (
            normalization
            @ oscillator_transition(
                mode,
                final_time_s - initial_state_time_s,
                natural_frequency_scale=natural_frequency_scale,
            )
            @ initial_vector
        )
        return np.column_stack((*start_columns, *stop_columns)), constant

    def _solve_nzic_leg(
        self,
        *,
        leg: ShapedLeg,
        initial_state: SwayState,
        initial_state_time_s: float,
        primary_band_fraction: float,
        second_mode_band_fraction: float,
        maximum_second_mode_residual_fraction: float,
    ) -> NzicCorrection:
        count = len(self.amplitudes)
        base = np.asarray((*self.amplitudes, *self.amplitudes), dtype=float)
        times = np.asarray(self.impulse_times_s, dtype=float)
        primary_mode = self.modes[0]

        equality_blocks: list[np.ndarray] = [
            np.concatenate((np.ones(count), np.zeros(count))),
            np.concatenate((np.zeros(count), np.ones(count))),
            np.concatenate((times, -times)),
        ]
        equality_targets: list[np.ndarray] = [
            np.asarray([1.0]),
            np.asarray([1.0]),
            np.asarray([0.0]),
        ]

        # Exact nominal cancellation plus zero first frequency derivative is
        # the ZVD-style robustness condition for the nonzero state.  A centered
        # finite difference avoids a long, error-prone symbolic derivative and
        # is deterministic at this very small problem size.
        mode_initial_states = [initial_state] + [
            SwayState(0.0, 0.0) for _ in self.modes[1:]
        ]
        derivative_step_fraction = 0.005
        for mode, mode_initial_state in zip(self.modes, mode_initial_states):
            center_matrix, center_constant = self._terminal_state_map(
                leg=leg,
                mode=mode,
                initial_state=mode_initial_state,
                initial_state_time_s=initial_state_time_s,
            )
            plus_matrix, plus_constant = self._terminal_state_map(
                leg=leg,
                mode=mode,
                initial_state=mode_initial_state,
                initial_state_time_s=initial_state_time_s,
                natural_frequency_scale=1.0 + derivative_step_fraction,
            )
            minus_matrix, minus_constant = self._terminal_state_map(
                leg=leg,
                mode=mode,
                initial_state=mode_initial_state,
                initial_state_time_s=initial_state_time_s,
                natural_frequency_scale=1.0 - derivative_step_fraction,
            )
            frequency_step_hz = (
                derivative_step_fraction * mode.natural_frequency_hz
            )
            equality_blocks.extend((
                center_matrix,
                (plus_matrix - minus_matrix) / (2.0 * frequency_step_hz),
            ))
            equality_targets.extend((
                -center_constant,
                -(plus_constant - minus_constant) / (2.0 * frequency_step_hz),
            ))

        equality_matrix = np.vstack(equality_blocks)
        equality_target = np.concatenate(equality_targets)

        initial_amplitude = initial_state.amplitude_rad(primary_mode)
        primary_normalization = max(initial_amplitude, math.radians(0.05))
        objective_terms: list[tuple[np.ndarray, np.ndarray, float]] = []
        for scale in np.linspace(
            1.0 - primary_band_fraction,
            1.0 + primary_band_fraction,
            9,
        ):
            if abs(scale - 1.0) < 1.0e-12:
                continue
            matrix, constant = self._terminal_state_map(
                leg=leg,
                mode=primary_mode,
                initial_state=initial_state,
                initial_state_time_s=initial_state_time_s,
                natural_frequency_scale=float(scale),
            )
            objective_terms.append((
                matrix / primary_normalization,
                constant / primary_normalization,
                1.0,
            ))

        # The measured higher-mode band is much narrower than the deliberately
        # conservative +/-10% rope band.  Strong weighting preserves the
        # original two-mode shaper's measured-band rejection while the primary
        # mode terms cancel the observed initial sway.
        for mode in self.modes[1:]:
            omega_d = (
                2.0 * math.pi * mode.natural_frequency_hz
                * math.sqrt(1.0 - mode.damping_ratio**2)
            )
            speed_m_s = 1.0e-3 * math.hypot(
                leg.vx_mm_s, leg.vy_mm_s
            )
            input_normalization = speed_m_s / (
                self.rope_length_m * omega_d
            )
            for scale in np.linspace(
                1.0 - second_mode_band_fraction,
                1.0 + second_mode_band_fraction,
                7,
            ):
                if abs(scale - 1.0) < 1.0e-12:
                    continue
                matrix, constant = self._terminal_state_map(
                    leg=leg,
                    mode=mode,
                    initial_state=SwayState(0.0, 0.0),
                    initial_state_time_s=initial_state_time_s,
                    natural_frequency_scale=float(scale),
                )
                objective_terms.append((
                    matrix / input_normalization,
                    constant / input_normalization,
                    1000.0,
                ))

        hessian = np.zeros((2 * count, 2 * count), dtype=float)
        gradient = np.zeros(2 * count, dtype=float)
        for matrix, constant, weight in objective_terms:
            hessian += weight * matrix.T @ matrix
            gradient += weight * matrix.T @ constant
        regularization = 1.0e-6

        def objective(amplitudes: np.ndarray) -> float:
            deviation = amplitudes - base
            return float(
                0.5 * amplitudes @ hessian @ amplitudes
                + gradient @ amplitudes
                + 0.5 * regularization * deviation @ deviation
            )

        def objective_jacobian(amplitudes: np.ndarray) -> np.ndarray:
            return (
                hessian @ amplitudes
                + gradient
                + regularization * (amplitudes - base)
            )

        result = minimize(
            objective,
            base,
            jac=objective_jacobian,
            method='SLSQP',
            bounds=Bounds(np.zeros(2 * count), np.ones(2 * count)),
            constraints=[LinearConstraint(
                equality_matrix, equality_target, equality_target
            )],
            options={'ftol': 1.0e-12, 'maxiter': 2000, 'disp': False},
        )
        equality_error = float(np.max(np.abs(
            equality_matrix @ result.x - equality_target
        )))
        if not result.success or equality_error > 1.0e-7:
            raise ValueError(
                'no monotonic robust NZIC solution exists for the current '
                f'{leg.name} sway phase ({result.message}, '
                f'equality error={equality_error:.2e})'
            )
        if float(np.min(result.x)) < -1.0e-9:
            raise ValueError('NZIC optimizer returned a negative velocity increment')

        start_amplitudes = tuple(float(max(value, 0.0)) for value in result.x[:count])
        stop_amplitudes = tuple(float(max(value, 0.0)) for value in result.x[count:])
        optimized = np.asarray((*start_amplitudes, *stop_amplitudes))

        center_matrix, center_constant = self._terminal_state_map(
            leg=leg,
            mode=primary_mode,
            initial_state=initial_state,
            initial_state_time_s=initial_state_time_s,
        )
        nominal_residual = float(np.linalg.norm(
            center_matrix @ optimized + center_constant
        ))
        primary_band_residual = 0.0
        uncorrected_band_residual = 0.0
        for scale in np.linspace(
            1.0 - primary_band_fraction,
            1.0 + primary_band_fraction,
            41,
        ):
            matrix, constant = self._terminal_state_map(
                leg=leg,
                mode=primary_mode,
                initial_state=initial_state,
                initial_state_time_s=initial_state_time_s,
                natural_frequency_scale=float(scale),
            )
            primary_band_residual = max(
                primary_band_residual,
                float(np.linalg.norm(matrix @ optimized + constant)),
            )
            uncorrected_band_residual = max(
                uncorrected_band_residual,
                float(np.linalg.norm(matrix @ base + constant)),
            )

        second_mode_band_residual = 0.0
        for mode in self.modes[1:]:
            omega_d = (
                2.0 * math.pi * mode.natural_frequency_hz
                * math.sqrt(1.0 - mode.damping_ratio**2)
            )
            speed_m_s = 1.0e-3 * math.hypot(
                leg.vx_mm_s, leg.vy_mm_s
            )
            input_normalization = speed_m_s / (
                self.rope_length_m * omega_d
            )
            for scale in np.linspace(
                1.0 - second_mode_band_fraction,
                1.0 + second_mode_band_fraction,
                41,
            ):
                matrix, constant = self._terminal_state_map(
                    leg=leg,
                    mode=mode,
                    initial_state=SwayState(0.0, 0.0),
                    initial_state_time_s=initial_state_time_s,
                    natural_frequency_scale=float(scale),
                )
                second_mode_band_residual = max(
                    second_mode_band_residual,
                    float(np.linalg.norm(matrix @ optimized + constant))
                    / input_normalization,
                )
        if (
            len(self.modes) > 1
            and second_mode_band_residual
            > maximum_second_mode_residual_fraction + 1.0e-9
        ):
            raise ValueError(
                'current sway phase cannot preserve higher-mode robustness: '
                f'predicted band residual={second_mode_band_residual:.4f} > '
                f'{maximum_second_mode_residual_fraction:.4f}'
            )
        if primary_band_residual > uncorrected_band_residual + 1.0e-9:
            raise ValueError(
                'NZIC solution would increase worst-case rope-mode residual'
            )

        return NzicCorrection(
            leg_name=leg.name,
            initial_state=initial_state,
            initial_amplitude_rad=initial_amplitude,
            start_amplitudes=start_amplitudes,
            stop_amplitudes=stop_amplitudes,
            nominal_residual_rad=nominal_residual,
            primary_band_residual_rad=primary_band_residual,
            uncorrected_primary_band_residual_rad=uncorrected_band_residual,
            second_mode_band_residual_fraction=second_mode_band_residual,
            equality_error=equality_error,
            iterations=int(getattr(result, 'nit', 0)),
        )

    def with_nonzero_initial_conditions(
        self,
        *,
        pitch_state: SwayState,
        roll_state: SwayState,
        minimum_correction_amplitude_rad: float = math.radians(0.02),
        primary_band_fraction: float = 0.10,
        second_mode_band_fraction: float = 0.025,
        maximum_second_mode_residual_fraction: float = 0.03,
    ) -> 'Robust2dSquareProfile':
        """Return a profile corrected for the measured 2-D initial sway.

        Pitch is canceled by the first +X leg and roll by the first +Y leg.
        The roll state is supplied at square time zero; the terminal equations
        include its free evolution before +Y, so no nominal phase shortcut is
        used.
        """
        finite_values = (
            pitch_state.angle_rad,
            pitch_state.angular_rate_rad_s,
            roll_state.angle_rad,
            roll_state.angular_rate_rad_s,
            minimum_correction_amplitude_rad,
            primary_band_fraction,
            second_mode_band_fraction,
            maximum_second_mode_residual_fraction,
        )
        if any(not math.isfinite(value) for value in finite_values):
            raise ValueError('NZIC states and limits must be finite')
        if minimum_correction_amplitude_rad < 0.0:
            raise ValueError('minimum correction amplitude must be nonnegative')
        if not 0.0 < primary_band_fraction < 0.5:
            raise ValueError('primary band fraction must be in (0, 0.5)')
        if not 0.0 < second_mode_band_fraction < 0.5:
            raise ValueError('second-mode band fraction must be in (0, 0.5)')
        if maximum_second_mode_residual_fraction <= 0.0:
            raise ValueError('maximum second-mode residual must be positive')

        corrected_legs = list(self.legs)
        corrections: list[NzicCorrection] = []
        for leg_index, initial_state in ((0, pitch_state), (1, roll_state)):
            if initial_state.amplitude_rad(self.modes[0]) < minimum_correction_amplitude_rad:
                continue
            correction = self._solve_nzic_leg(
                leg=corrected_legs[leg_index],
                initial_state=initial_state,
                initial_state_time_s=0.0,
                primary_band_fraction=primary_band_fraction,
                second_mode_band_fraction=second_mode_band_fraction,
                maximum_second_mode_residual_fraction=(
                    maximum_second_mode_residual_fraction
                ),
            )
            corrected_legs[leg_index] = replace(
                corrected_legs[leg_index],
                start_amplitudes=correction.start_amplitudes,
                stop_amplitudes=correction.stop_amplitudes,
            )
            corrections.append(correction)

        corrected = copy.copy(self)
        corrected.legs = tuple(corrected_legs)
        corrected.ic_corrections = tuple(corrections)
        return corrected

    def with_retimed_leg_start(
        self,
        leg_index: int,
        new_start_s: float,
    ) -> 'Robust2dSquareProfile':
        """Delay one leg and every following leg without changing commands."""
        if not 0 <= leg_index < len(self.legs):
            raise ValueError('leg index is out of range')
        if not math.isfinite(new_start_s):
            raise ValueError('new leg start must be finite')
        old_start_s = self.legs[leg_index].start_s
        if new_start_s + 1.0e-12 < old_start_s:
            raise ValueError('runtime retiming may delay a leg, not advance it')
        if leg_index > 0 and new_start_s + 1.0e-12 < self.legs[leg_index - 1].shaped_stop_s:
            raise ValueError('retimed leg would overlap the previous leg')
        shift_s = new_start_s - old_start_s
        if shift_s <= 1.0e-12:
            return self
        shifted_legs = [
            replace(leg, start_s=leg.start_s + shift_s)
            if index >= leg_index else leg
            for index, leg in enumerate(self.legs)
        ]
        shifted = copy.copy(self)
        shifted.legs = tuple(shifted_legs)
        shifted.duration_s = shifted.legs[-1].shaped_stop_s
        return shifted

    def with_leg_nonzero_initial_condition(
        self,
        *,
        leg_index: int,
        initial_state: SwayState,
        initial_state_time_s: float,
        minimum_correction_amplitude_rad: float = math.radians(0.02),
        primary_band_fraction: float = 0.10,
        second_mode_band_fraction: float = 0.025,
        maximum_second_mode_residual_fraction: float = 0.03,
    ) -> 'Robust2dSquareProfile':
        """Correct one leg using a state estimated on the profile time base."""
        if not 0 <= leg_index < len(self.legs):
            raise ValueError('leg index is out of range')
        leg = self.legs[leg_index]
        if not math.isfinite(initial_state_time_s):
            raise ValueError('initial state time must be finite')
        if initial_state_time_s > leg.start_s + 1.0e-9:
            raise ValueError('initial state cannot be later than the corrected leg start')
        if minimum_correction_amplitude_rad < 0.0:
            raise ValueError('minimum correction amplitude must be nonnegative')

        corrected_legs = list(self.legs)
        retained_corrections = [
            correction
            for correction in self.ic_corrections
            if correction.leg_name != leg.name
        ]
        if initial_state.amplitude_rad(self.modes[0]) < minimum_correction_amplitude_rad:
            corrected_legs[leg_index] = replace(
                leg,
                start_amplitudes=self.amplitudes,
                stop_amplitudes=self.amplitudes,
            )
        else:
            correction = self._solve_nzic_leg(
                leg=leg,
                initial_state=initial_state,
                initial_state_time_s=initial_state_time_s,
                primary_band_fraction=primary_band_fraction,
                second_mode_band_fraction=second_mode_band_fraction,
                maximum_second_mode_residual_fraction=(
                    maximum_second_mode_residual_fraction
                ),
            )
            corrected_legs[leg_index] = replace(
                leg,
                start_amplitudes=correction.start_amplitudes,
                stop_amplitudes=correction.stop_amplitudes,
            )
            retained_corrections.append(correction)

        corrected = copy.copy(self)
        corrected.legs = tuple(corrected_legs)
        corrected.ic_corrections = tuple(retained_corrections)
        return corrected

    def _residual_fraction_for_mode(
        self,
        mode: ZvdMode,
        natural_frequency_scale: float,
    ) -> float:
        """Normalized residual vibration from the impulse sequence.

        The result is zero at the modeled frequency for an ideal plant and is
        useful for quantifying robustness to rope-length/frequency error.
        """
        if (
            not math.isfinite(natural_frequency_scale)
            or natural_frequency_scale <= 0.0
        ):
            raise ValueError('natural_frequency_scale must be finite and positive')
        omega_n = (
            2.0 * math.pi * mode.natural_frequency_hz
            * natural_frequency_scale
        )
        omega_d = omega_n * math.sqrt(1.0 - mode.damping_ratio**2)
        final_time = self.shaper_tail_s
        real = 0.0
        imag = 0.0
        for delay_s, amplitude in zip(self.impulse_times_s, self.amplitudes):
            age = final_time - delay_s
            magnitude = amplitude * math.exp(-mode.damping_ratio * omega_n * age)
            phase = omega_d * age
            real += magnitude * math.cos(phase)
            imag += magnitude * math.sin(phase)
        return math.hypot(real, imag)

    def residual_fraction(self, natural_frequency_scale: float = 1.0) -> float:
        """Residual fraction for the rope mode."""
        return self._residual_fraction_for_mode(
            self.modes[0], natural_frequency_scale
        )

    def second_mode_residual_fraction(
        self,
        natural_frequency_scale: float = 1.0,
    ) -> float:
        """Residual fraction for the second mode, when configured."""
        if len(self.modes) < 2:
            raise ValueError('no second mode is configured')
        return self._residual_fraction_for_mode(
            self.modes[1], natural_frequency_scale
        )

    def iter_summary_rows(self) -> Iterable[tuple[str, float, float, float, float]]:
        """Yield name, start, raw stop, shaped stop, signed distance."""
        for leg in self.legs:
            distance = math.hypot(
                leg.vx_mm_s * leg.raw_duration_s,
                leg.vy_mm_s * leg.raw_duration_s,
            )
            sign = 1.0 if leg.name.startswith('+') else -1.0
            yield (
                leg.name,
                leg.start_s,
                leg.raw_stop_s,
                leg.shaped_stop_s,
                sign * distance,
            )
