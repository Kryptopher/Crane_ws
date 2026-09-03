#!/usr/bin/env python3
"""
Open-loop optimal stopping for a velocity-controlled crane.

The planner is deliberately ROS-independent.  At joystick release it accepts
the measured state ``[cart position error, cart velocity, payload angle,
payload angle rate]`` for one or two axes and solves one convex quadratic
program.  The module also provides the forward-only minimum-time formulation
used by joystick mode.  Its final position is free, it searches the shortest
feasible horizon, and it forbids command polarity reversal.  Returned velocity
sequences are intended to be played without feedback; measurements after
release are safety/validation signals only.

The optimization uses exact zero-order-hold cart/pendulum dynamics and a bank
of rope-length/actuator-lag models.  Terminal rest constraints are imposed for
every model in the bank, while vector command, vector command-slew, correction
envelope, and workspace constraints are imposed along the nominal trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Mapping, Sequence

import numpy as np
from scipy import sparse
from scipy.linalg import expm
from scipy.optimize import linprog

try:
    import osqp
except ImportError:  # Keep model/observer utilities importable for diagnostics.
    osqp = None


G = 9.80665
STATE_SIZE = 4


@dataclass(frozen=True)
class CraneModel:
    rope_length_m: float
    damping_ratio: float
    actuator_time_constant_s: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.rope_length_m) or self.rope_length_m <= 0.0:
            raise ValueError('rope_length_m must be positive and finite')
        if (
            not math.isfinite(self.actuator_time_constant_s)
            or self.actuator_time_constant_s <= 0.0
        ):
            raise ValueError('actuator_time_constant_s must be positive and finite')
        if not math.isfinite(self.damping_ratio) or not 0.0 <= self.damping_ratio < 1.0:
            raise ValueError('damping_ratio must be in [0, 1)')


def discrete_crane_model(model: CraneModel, dt_s: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the exact ZOH model for cart position/velocity and payload angle/rate.

    The input is cart velocity command.  The cart velocity follows the command
    through a first-order lag and the small-angle pendulum is base excited by
    cart acceleration.
    """
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError('dt_s must be positive and finite')
    length = model.rope_length_m
    tau = model.actuator_time_constant_s
    omega = math.sqrt(G / length)
    ac = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, -1.0 / tau, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 1.0 / (length * tau), -omega * omega,
         -2.0 * model.damping_ratio * omega],
    ])
    bc = np.array([[0.0], [1.0 / tau], [0.0], [-1.0 / (length * tau)]])
    augmented = np.zeros((STATE_SIZE + 1, STATE_SIZE + 1))
    augmented[:STATE_SIZE, :STATE_SIZE] = ac
    augmented[:STATE_SIZE, STATE_SIZE:] = bc
    discrete = expm(augmented * dt_s)
    return discrete[:STATE_SIZE, :STATE_SIZE], discrete[:STATE_SIZE, STATE_SIZE:]


def prediction_matrices(
    a: np.ndarray,
    b: np.ndarray,
    step_count: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return affine maps ``x[k] = F[k] x[0] + G[k] u`` for k=0..N."""
    if a.shape != (STATE_SIZE, STATE_SIZE) or b.shape != (STATE_SIZE, 1):
        raise ValueError('unexpected state-space dimensions')
    if step_count <= 0:
        raise ValueError('step_count must be positive')
    f_sequence = [np.eye(STATE_SIZE)]
    g_sequence = [np.zeros((STATE_SIZE, step_count))]
    for k in range(1, step_count + 1):
        f_next = a @ f_sequence[-1]
        g_next = a @ g_sequence[-1]
        g_next[:, k - 1] = b[:, 0]
        f_sequence.append(f_next)
        g_sequence.append(g_next)
    return f_sequence, g_sequence


@dataclass(frozen=True)
class OptimalStopConfig:
    sample_period_s: float = 0.02
    control_knot_period_s: float = 0.10
    horizon_s: float = 4.5
    command_lead_time_s: float = 0.75
    rope_length_m: float = 0.90
    damping_ratio: float = 0.02
    actuator_time_constant_s: float = 0.045
    rope_length_uncertainty_fraction: float = 0.10
    actuator_uncertainty_fraction: float = 0.15
    uncertainty_samples: int = 3
    max_command_speed_m_s: float = 0.025
    max_command_acceleration_m_s2: float = 0.100
    max_position_excursion_m: float = 0.080
    terminal_position_tolerance_m: float = 0.010
    terminal_velocity_tolerance_m_s: float = 0.002
    terminal_angle_tolerance_rad: float = math.radians(0.12)
    terminal_angle_rate_tolerance_rad_s: float = math.radians(0.50)
    robust_terminal_tolerance_multiplier: float = 4.0
    final_command_tolerance_m_s: float = 0.001
    vector_constraint_facets: int = 16
    running_cost_stride: int = 5
    path_constraint_stride: int = 5
    terminal_position_weight: float = 8.0
    terminal_velocity_weight: float = 5.0
    terminal_angle_weight: float = 35.0
    terminal_angle_rate_weight: float = 15.0
    running_position_weight: float = 0.01
    running_velocity_weight: float = 0.01
    running_angle_weight: float = 0.08
    running_angle_rate_weight: float = 0.02
    effort_weight: float = 0.004
    slew_weight: float = 0.020
    solver_absolute_tolerance: float = 1.0e-3
    solver_relative_tolerance: float = 1.0e-3
    solver_max_iterations: int = 30000
    solver_time_limit_s: float = 0.40

    def __post_init__(self) -> None:
        positive = {
            'sample_period_s': self.sample_period_s,
            'control_knot_period_s': self.control_knot_period_s,
            'horizon_s': self.horizon_s,
            'command_lead_time_s': self.command_lead_time_s,
            'rope_length_m': self.rope_length_m,
            'actuator_time_constant_s': self.actuator_time_constant_s,
            'max_command_speed_m_s': self.max_command_speed_m_s,
            'max_command_acceleration_m_s2': self.max_command_acceleration_m_s2,
            'max_position_excursion_m': self.max_position_excursion_m,
            'terminal_position_tolerance_m': self.terminal_position_tolerance_m,
            'terminal_velocity_tolerance_m_s': self.terminal_velocity_tolerance_m_s,
            'terminal_angle_tolerance_rad': self.terminal_angle_tolerance_rad,
            'terminal_angle_rate_tolerance_rad_s': self.terminal_angle_rate_tolerance_rad_s,
            'robust_terminal_tolerance_multiplier': self.robust_terminal_tolerance_multiplier,
            'final_command_tolerance_m_s': self.final_command_tolerance_m_s,
            'solver_absolute_tolerance': self.solver_absolute_tolerance,
            'solver_relative_tolerance': self.solver_relative_tolerance,
            'solver_time_limit_s': self.solver_time_limit_s,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f'{name} must be positive and finite')
        fractions = {
            'rope_length_uncertainty_fraction': self.rope_length_uncertainty_fraction,
            'actuator_uncertainty_fraction': self.actuator_uncertainty_fraction,
        }
        for name, value in fractions.items():
            if not math.isfinite(value) or not 0.0 <= value < 1.0:
                raise ValueError(f'{name} must be in [0, 1)')
        if not math.isfinite(self.damping_ratio) or not 0.0 <= self.damping_ratio < 1.0:
            raise ValueError('damping_ratio must be in [0, 1)')
        if self.uncertainty_samples not in (1, 3):
            raise ValueError('uncertainty_samples must be 1 or 3')
        if self.vector_constraint_facets < 4 or self.vector_constraint_facets % 4:
            raise ValueError('vector_constraint_facets must be a multiple of four and >= 4')
        if (
            self.running_cost_stride <= 0
            or self.path_constraint_stride <= 0
            or self.solver_max_iterations <= 0
        ):
            raise ValueError('stride and solver iteration count must be positive')
        if self.step_count < 2:
            raise ValueError('horizon must contain at least two samples')
        if self.control_knot_period_s < self.sample_period_s:
            raise ValueError('control_knot_period_s must be >= sample_period_s')
        if self.command_lead_time_s >= self.horizon_s - self.control_knot_period_s:
            raise ValueError('command_lead_time_s leaves no usable control horizon')

    @property
    def step_count(self) -> int:
        return int(math.ceil(self.horizon_s / self.sample_period_s))


@dataclass(frozen=True)
class ModelScenario:
    label: str
    model: CraneModel
    a: np.ndarray
    b: np.ndarray
    f: list[np.ndarray]
    g: list[np.ndarray]


@dataclass(frozen=True)
class OpenLoopStopPlan:
    axes: tuple[str, ...]
    commands_m_s: np.ndarray
    predicted_states: Mapping[str, np.ndarray]
    solve_time_s: float
    solver_status: str
    objective: float
    iterations: int
    sample_period_s: float
    command_lead_time_s: float

    @property
    def duration_s(self) -> float:
        return float(self.commands_m_s.shape[0]) * self.sample_period_s


class OpenLoopPlanningError(RuntimeError):
    pass


@dataclass(frozen=True)
class MinimumTimeStopConfig:
    """Configuration for the forward-only, free-endpoint stopping LP."""

    sample_period_s: float = 0.02
    minimum_horizon_s: float = 0.30
    maximum_horizon_s: float = 2.80
    horizon_resolution_s: float = 0.04
    rope_length_m: float = 0.90
    damping_ratio: float = 0.02
    actuator_time_constant_s: float = 0.045
    rope_length_uncertainty_fraction: float = 0.10
    actuator_uncertainty_fraction: float = 0.15
    uncertainty_samples: int = 3
    max_command_speed_m_s: float = 0.100
    max_command_acceleration_m_s2: float = 1.000
    terminal_velocity_tolerance_m_s: float = 0.002
    terminal_angle_tolerance_rad: float = math.radians(0.12)
    robust_terminal_tolerance_multiplier: float = 2.0
    final_command_tolerance_m_s: float = 0.0005
    vector_constraint_facets: int = 16
    residual_constraint_facets: int = 16
    solver_time_limit_s: float = 0.08

    def __post_init__(self) -> None:
        positive = {
            'sample_period_s': self.sample_period_s,
            'minimum_horizon_s': self.minimum_horizon_s,
            'maximum_horizon_s': self.maximum_horizon_s,
            'horizon_resolution_s': self.horizon_resolution_s,
            'rope_length_m': self.rope_length_m,
            'actuator_time_constant_s': self.actuator_time_constant_s,
            'max_command_speed_m_s': self.max_command_speed_m_s,
            'max_command_acceleration_m_s2': self.max_command_acceleration_m_s2,
            'terminal_velocity_tolerance_m_s': self.terminal_velocity_tolerance_m_s,
            'terminal_angle_tolerance_rad': self.terminal_angle_tolerance_rad,
            'robust_terminal_tolerance_multiplier': (
                self.robust_terminal_tolerance_multiplier
            ),
            'final_command_tolerance_m_s': self.final_command_tolerance_m_s,
            'solver_time_limit_s': self.solver_time_limit_s,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f'{name} must be positive and finite')
        if self.maximum_horizon_s <= self.minimum_horizon_s:
            raise ValueError('maximum_horizon_s must exceed minimum_horizon_s')
        for name, value in (
            ('rope_length_uncertainty_fraction', self.rope_length_uncertainty_fraction),
            ('actuator_uncertainty_fraction', self.actuator_uncertainty_fraction),
        ):
            if not math.isfinite(value) or not 0.0 <= value < 1.0:
                raise ValueError(f'{name} must be in [0, 1)')
        if not math.isfinite(self.damping_ratio) or not 0.0 <= self.damping_ratio < 1.0:
            raise ValueError('damping_ratio must be in [0, 1)')
        if self.uncertainty_samples not in (1, 3):
            raise ValueError('uncertainty_samples must be 1 or 3')
        for name, facets in (
            ('vector_constraint_facets', self.vector_constraint_facets),
            ('residual_constraint_facets', self.residual_constraint_facets),
        ):
            if facets < 4 or facets % 4:
                raise ValueError(f'{name} must be a multiple of four and >= 4')

    @property
    def maximum_step_count(self) -> int:
        return int(math.floor(self.maximum_horizon_s / self.sample_period_s))


@dataclass(frozen=True)
class MinimumTimeStopPlan:
    axes: tuple[str, ...]
    commands_m_s: np.ndarray
    predicted_states: Mapping[str, np.ndarray]
    direction_signs: np.ndarray
    initial_states: np.ndarray
    solve_time_s: float
    solver_status: str
    objective: float
    iterations: int
    sample_period_s: float

    @property
    def duration_s(self) -> float:
        return float(self.commands_m_s.shape[0]) * self.sample_period_s

    @property
    def terminal_displacement_m(self) -> np.ndarray:
        nominal = next(iter(self.predicted_states.values()))
        return nominal[-1, :, 0] - nominal[0, :, 0]


class ForwardOnlyMinimumTimeStopPlanner:
    """
    Find the shortest robust stop that never reverses the trolley.

    Final cart position is deliberately free.  Commands are constrained to the
    incoming direction (or zero), so the positive first-order velocity model
    cannot reverse.  A bounded-horizon feasibility LP is quasi-convex in time;
    bisection finds the shortest feasible sampled horizon and a final LP then
    minimizes forward stopping distance at that horizon.
    """

    def __init__(self, config: MinimumTimeStopConfig = MinimumTimeStopConfig()):
        self.config = config
        self.scenarios = self._build_scenarios()
        self.nominal_scenario = min(
            self.scenarios,
            key=lambda item: (
                abs(item.model.rope_length_m - config.rope_length_m)
                + abs(
                    item.model.actuator_time_constant_s
                    - config.actuator_time_constant_s
                )
            ),
        )

    def _uncertainty_factors(self, fraction: float) -> tuple[float, ...]:
        if self.config.uncertainty_samples == 1 or fraction == 0.0:
            return (1.0,)
        return (1.0 - fraction, 1.0, 1.0 + fraction)

    def _build_scenarios(self) -> tuple[ModelScenario, ...]:
        scenarios = []
        for length_factor in self._uncertainty_factors(
            self.config.rope_length_uncertainty_fraction
        ):
            for tau_factor in self._uncertainty_factors(
                self.config.actuator_uncertainty_fraction
            ):
                model = CraneModel(
                    rope_length_m=self.config.rope_length_m * length_factor,
                    damping_ratio=self.config.damping_ratio,
                    actuator_time_constant_s=(
                        self.config.actuator_time_constant_s * tau_factor
                    ),
                )
                a, b = discrete_crane_model(model, self.config.sample_period_s)
                f, g = prediction_matrices(a, b, self.config.maximum_step_count)
                scenarios.append(ModelScenario(
                    label=(
                        f'L={model.rope_length_m:.4f},'
                        f'tau={model.actuator_time_constant_s:.4f}'
                    ),
                    model=model,
                    a=a,
                    b=b,
                    f=f,
                    g=g,
                ))
        return tuple(scenarios)

    @staticmethod
    def _validate_inputs(
        states: Mapping[str, Sequence[float]],
        direction_signs: Mapping[str, float],
        initial_commands_m_s: Mapping[str, float],
        forward_workspace_m: Mapping[str, float] | None,
    ) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        axes, raw_states = RobustOpenLoopStopPlanner._validated_states(states)
        expected = set(axes)
        if set(direction_signs) != expected or set(initial_commands_m_s) != expected:
            raise ValueError('directions and initial commands must contain every axis')
        directions = np.array([float(direction_signs[axis]) for axis in axes])
        if not np.all(np.isin(directions, (-1.0, 1.0))):
            raise ValueError('every forward direction must be -1 or +1')
        initial_commands = np.array([
            float(initial_commands_m_s[axis]) for axis in axes
        ])
        if not np.all(np.isfinite(initial_commands)):
            raise ValueError('initial commands must be finite')
        if forward_workspace_m is None:
            workspace = np.full(len(axes), np.inf)
        else:
            if set(forward_workspace_m) != expected:
                raise ValueError('forward workspace must contain every axis')
            workspace = np.array([float(forward_workspace_m[axis]) for axis in axes])
            if np.any(np.isnan(workspace)) or np.any(workspace <= 0.0):
                raise ValueError('forward workspace must be positive')
        signed_states = raw_states * directions[:, None]
        signed_commands = initial_commands * directions
        return axes, signed_states, directions, signed_commands, workspace

    def _sampled_horizons(self) -> tuple[int, int, int]:
        dt = self.config.sample_period_s
        stride = max(1, int(round(self.config.horizon_resolution_s / dt)))
        minimum = max(2, int(math.ceil(self.config.minimum_horizon_s / dt)))
        maximum = self.config.maximum_step_count
        minimum = int(math.ceil(minimum / stride) * stride)
        maximum = int(math.floor(maximum / stride) * stride)
        return minimum, maximum, stride

    @staticmethod
    def _state_map(
        scenario: ModelScenario,
        step_count: int,
        axis_index: int,
        axis_count: int,
        command_speed: float,
    ) -> np.ndarray:
        matrix = np.zeros((STATE_SIZE, step_count * axis_count))
        matrix[:, axis_index::axis_count] = (
            command_speed * scenario.g[step_count][:, :step_count]
        )
        return matrix

    def _solve_horizon(
        self,
        x0: np.ndarray,
        directions: np.ndarray,
        initial_signed_commands: np.ndarray,
        workspace: np.ndarray,
        step_count: int,
    ):
        axis_count = len(directions)
        variable_count = step_count * axis_count
        speed = self.config.max_command_speed_m_s
        initial_z = initial_signed_commands / speed
        if np.any(initial_z < -1.0e-3) or np.linalg.norm(initial_z) > 1.001:
            raise ValueError('initial command is inconsistent with the forward direction')
        initial_z = np.clip(initial_z, 0.0, 1.0)

        a_rows = []
        b_rows = []

        def upper(row, bound) -> None:
            a_rows.append(np.asarray(row, dtype=float))
            b_rows.append(float(bound))

        vector_angles = (
            np.arange(self.config.vector_constraint_facets) + 0.5
        ) * 2.0 * math.pi / self.config.vector_constraint_facets
        vector_normals = np.column_stack((
            np.cos(vector_angles), np.sin(vector_angles)
        )) if axis_count == 2 else np.array([[1.0], [-1.0]])
        polygon_radius = 1.0 if axis_count == 1 else math.cos(
            math.pi / self.config.vector_constraint_facets
        )

        for k in range(step_count):
            for normal in vector_normals:
                row = np.zeros(variable_count)
                for axis_index, coefficient in enumerate(normal):
                    row[k * axis_count + axis_index] = (
                        coefficient * directions[axis_index]
                    )
                upper(row, polygon_radius)

        normalized_slew = (
            self.config.max_command_acceleration_m_s2
            * self.config.sample_period_s / speed
        )
        if axis_count == 2:
            normalized_slew *= polygon_radius
        for k in range(step_count):
            for normal in vector_normals:
                row = np.zeros(variable_count)
                previous_projection = 0.0
                for axis_index, coefficient in enumerate(normal):
                    signed_coefficient = coefficient * directions[axis_index]
                    row[k * axis_count + axis_index] = signed_coefficient
                    if k:
                        row[(k - 1) * axis_count + axis_index] = -signed_coefficient
                    else:
                        previous_projection += coefficient * (
                            directions[axis_index] * initial_z[axis_index]
                        )
                upper(row, normalized_slew + previous_projection)

        residual_angles = (
            np.arange(self.config.residual_constraint_facets) + 0.5
        ) * 2.0 * math.pi / self.config.residual_constraint_facets
        residual_normals = np.column_stack((
            np.cos(residual_angles), np.sin(residual_angles)
        ))
        residual_radius = math.cos(
            math.pi / self.config.residual_constraint_facets
        )

        nominal_position_maps = []
        for scenario in self.scenarios:
            robust_multiplier = (
                1.0 if scenario.label == self.nominal_scenario.label
                else self.config.robust_terminal_tolerance_multiplier
            )
            omega = math.sqrt(G / scenario.model.rope_length_m)
            for axis_index in range(axis_count):
                state_map = self._state_map(
                    scenario, step_count, axis_index, axis_count, speed
                )
                free = scenario.f[step_count] @ x0[axis_index]

                upper(
                    state_map[1],
                    robust_multiplier * self.config.terminal_velocity_tolerance_m_s
                    - free[1],
                )
                upper(-state_map[1], free[1] + 1.0e-7)

                angle_scale = self.config.terminal_angle_tolerance_rad
                rate_scale = omega * angle_scale
                for normal in residual_normals:
                    row = (
                        normal[0] * state_map[2] / angle_scale
                        + normal[1] * state_map[3] / rate_scale
                    )
                    offset = (
                        normal[0] * free[2] / angle_scale
                        + normal[1] * free[3] / rate_scale
                    )
                    upper(row, robust_multiplier * residual_radius - offset)

                if scenario.label == self.nominal_scenario.label:
                    nominal_position_maps.append((axis_index, state_map[0], free[0]))

        for axis_index, position_map, free_position in nominal_position_maps:
            if math.isfinite(workspace[axis_index]):
                upper(position_map, workspace[axis_index] - free_position)
            upper(-position_map, free_position + 1.0e-7)

        final_normalized = self.config.final_command_tolerance_m_s / speed
        for axis_index in range(axis_count):
            row = np.zeros(variable_count)
            row[(step_count - 1) * axis_count + axis_index] = 1.0
            upper(row, final_normalized)

        objective = np.full(variable_count, 1.0e-6)
        for _, position_map, _ in nominal_position_maps:
            objective += position_map / max(1, axis_count)

        started = time.monotonic()
        result = linprog(
            objective,
            A_ub=np.asarray(a_rows),
            b_ub=np.asarray(b_rows),
            bounds=(0.0, 1.0),
            method='highs',
            options={
                'presolve': True,
                'time_limit': self.config.solver_time_limit_s,
            },
        )
        elapsed = time.monotonic() - started
        return result, elapsed

    def plan(
        self,
        states: Mapping[str, Sequence[float]],
        *,
        direction_signs: Mapping[str, float],
        initial_commands_m_s: Mapping[str, float],
        forward_workspace_m: Mapping[str, float] | None = None,
    ) -> MinimumTimeStopPlan:
        axes, x0, directions, initial_commands, workspace = self._validate_inputs(
            states,
            direction_signs,
            initial_commands_m_s,
            forward_workspace_m,
        )
        if np.any(x0[:, 1] < -0.010):
            raise OpenLoopPlanningError(
                'measured cart velocity opposes the requested forward stop direction'
            )

        minimum, maximum, stride = self._sampled_horizons()
        total_solve_time = 0.0
        result, elapsed = self._solve_horizon(
            x0, directions, initial_commands, workspace, maximum
        )
        total_solve_time += elapsed
        if not result.success:
            raise OpenLoopPlanningError(
                f'no forward-only stop is feasible within '
                f'{maximum * self.config.sample_period_s:.2f}s: {result.message}'
            )

        best_steps = maximum
        best_result = result
        low_index = 0
        high_index = (maximum - minimum) // stride
        while low_index <= high_index:
            middle_index = (low_index + high_index) // 2
            candidate_steps = minimum + middle_index * stride
            candidate, elapsed = self._solve_horizon(
                x0, directions, initial_commands, workspace, candidate_steps
            )
            total_solve_time += elapsed
            if candidate.success:
                best_steps = candidate_steps
                best_result = candidate
                high_index = middle_index - 1
            else:
                low_index = middle_index + 1

        # Bisection assumes the expected quasi-convex feasibility boundary.
        # Verify the immediately shorter grid point so solver tolerances cannot
        # make us report a non-minimum sampled horizon.
        previous_steps = best_steps - stride
        if previous_steps >= minimum:
            previous, elapsed = self._solve_horizon(
                x0, directions, initial_commands, workspace, previous_steps
            )
            total_solve_time += elapsed
            if previous.success:
                best_steps = previous_steps
                best_result = previous

        signed_commands = (
            self.config.max_command_speed_m_s
            * np.asarray(best_result.x).reshape(best_steps, len(axes))
        )
        commands = signed_commands * directions[None, :]
        predicted = self._predict_all(x0, signed_commands)
        self._validate_solution(
            signed_commands,
            predicted,
            initial_commands,
            workspace,
        )
        return MinimumTimeStopPlan(
            axes=axes,
            commands_m_s=commands,
            predicted_states=predicted,
            direction_signs=directions,
            initial_states=x0,
            solve_time_s=total_solve_time,
            solver_status=str(best_result.message),
            objective=float(best_result.fun),
            iterations=int(best_result.nit),
            sample_period_s=self.config.sample_period_s,
        )

    def _predict_all(
        self,
        x0: np.ndarray,
        signed_commands: np.ndarray,
    ) -> dict[str, np.ndarray]:
        predicted = {}
        for scenario in self.scenarios:
            states = np.empty((len(signed_commands) + 1, len(x0), STATE_SIZE))
            states[0] = x0
            for k, command in enumerate(signed_commands):
                states[k + 1] = (
                    states[k] @ scenario.a.T
                    + command[:, None] * scenario.b[:, 0][None, :]
                )
            predicted[scenario.label] = states
        return predicted

    def _validate_solution(
        self,
        signed_commands: np.ndarray,
        predicted: Mapping[str, np.ndarray],
        initial_signed_commands: np.ndarray,
        workspace: np.ndarray,
    ) -> None:
        margin = 1.015
        if np.min(signed_commands) < -2.0e-6:
            raise OpenLoopPlanningError('forward-only command validation failed')
        if np.max(np.linalg.norm(signed_commands, axis=1)) > (
            margin * self.config.max_command_speed_m_s
        ):
            raise OpenLoopPlanningError('command-speed validation failed')
        previous = np.vstack((initial_signed_commands, signed_commands[:-1]))
        slew = np.linalg.norm(signed_commands - previous, axis=1)
        slew /= self.config.sample_period_s
        if np.max(slew) > margin * self.config.max_command_acceleration_m_s2:
            raise OpenLoopPlanningError('command-acceleration validation failed')
        if np.max(np.abs(signed_commands[-1])) > (
            margin * self.config.final_command_tolerance_m_s
        ):
            raise OpenLoopPlanningError('final-command validation failed')

        for scenario in self.scenarios:
            states = predicted[scenario.label]
            if np.min(states[:, :, 1]) < -2.0e-5:
                raise OpenLoopPlanningError('predicted cart reversal validation failed')
            multiplier = (
                1.0 if scenario.label == self.nominal_scenario.label
                else self.config.robust_terminal_tolerance_multiplier
            )
            terminal = states[-1]
            if np.max(terminal[:, 1]) > (
                margin * multiplier * self.config.terminal_velocity_tolerance_m_s
            ):
                raise OpenLoopPlanningError('terminal cart-velocity validation failed')
            omega = math.sqrt(G / scenario.model.rope_length_m)
            residual = np.sqrt(
                terminal[:, 2] ** 2 + (terminal[:, 3] / omega) ** 2
            )
            if np.max(residual) > (
                margin * multiplier * self.config.terminal_angle_tolerance_rad
            ):
                raise OpenLoopPlanningError('terminal residual-sway validation failed')

        nominal = predicted[self.nominal_scenario.label]
        displacement = nominal[-1, :, 0] - nominal[0, :, 0]
        if np.any(displacement < -2.0e-5) or np.any(displacement > workspace + 2.0e-5):
            raise OpenLoopPlanningError('forward workspace validation failed')


class RobustOpenLoopStopPlanner:
    """Build and solve a scenario-robust, constrained open-loop stop QP."""

    def __init__(self, config: OptimalStopConfig = OptimalStopConfig()):
        if osqp is None:
            raise OpenLoopPlanningError(
                'OSQP is unavailable; install the Python osqp package before hardware use'
            )
        self.config = config
        self.scenarios = self._build_scenarios()
        self.nominal_scenario = min(
            self.scenarios,
            key=lambda item: (
                abs(item.model.rope_length_m - config.rope_length_m)
                + abs(item.model.actuator_time_constant_s - config.actuator_time_constant_s)
            ),
        )

    def _uncertainty_factors(self, fraction: float) -> tuple[float, ...]:
        if self.config.uncertainty_samples == 1 or fraction == 0.0:
            return (1.0,)
        return (1.0 - fraction, 1.0, 1.0 + fraction)

    def _build_scenarios(self) -> tuple[ModelScenario, ...]:
        scenarios = []
        for length_factor in self._uncertainty_factors(
            self.config.rope_length_uncertainty_fraction
        ):
            for tau_factor in self._uncertainty_factors(
                self.config.actuator_uncertainty_fraction
            ):
                model = CraneModel(
                    rope_length_m=self.config.rope_length_m * length_factor,
                    damping_ratio=self.config.damping_ratio,
                    actuator_time_constant_s=(
                        self.config.actuator_time_constant_s * tau_factor
                    ),
                )
                a, b = discrete_crane_model(model, self.config.sample_period_s)
                f, g = prediction_matrices(a, b, self.config.step_count)
                scenarios.append(ModelScenario(
                    label=f'L={model.rope_length_m:.4f},tau={model.actuator_time_constant_s:.4f}',
                    model=model,
                    a=a,
                    b=b,
                    f=f,
                    g=g,
                ))
        return tuple(scenarios)

    @staticmethod
    def _validated_states(
        states: Mapping[str, Sequence[float]],
    ) -> tuple[tuple[str, ...], np.ndarray]:
        axes = tuple(states.keys())
        if not axes or len(axes) > 2 or len(set(axes)) != len(axes):
            raise ValueError('states must contain one or two uniquely named axes')
        matrix = np.empty((len(axes), STATE_SIZE))
        for axis_index, axis in enumerate(axes):
            values = np.asarray(states[axis], dtype=float)
            if values.shape != (STATE_SIZE,) or not np.all(np.isfinite(values)):
                raise ValueError(f'state for {axis!r} must contain four finite values')
            matrix[axis_index] = values
        return axes, matrix

    @staticmethod
    def _axis_state_map(
        g: np.ndarray,
        axis_index: int,
        axis_count: int,
        variable_count: int,
        command_scale: float,
    ) -> np.ndarray:
        result = np.zeros((STATE_SIZE, variable_count))
        result[:, axis_index::axis_count] = command_scale * g
        return result

    @staticmethod
    def _add_quadratic_state_cost(
        hessian: np.ndarray,
        linear: np.ndarray,
        state_map: np.ndarray,
        free_state: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        hessian += state_map.T @ weights @ state_map
        linear += state_map.T @ weights @ free_state

    def _polygon_normals(self, axis_count: int) -> np.ndarray:
        if axis_count == 1:
            return np.array([[1.0], [-1.0]])
        facets = self.config.vector_constraint_facets
        angles = (np.arange(facets) + 0.5) * 2.0 * math.pi / facets
        return np.column_stack((np.cos(angles), np.sin(angles)))

    def _control_basis(self) -> np.ndarray:
        """
        Return a piecewise-linear command basis on the model sample grid.

        Optimizing smooth knots instead of every 20 ms command makes the
        release-time QP small without coarsening the pendulum prediction.
        """
        steps = self.config.step_count
        command_times = np.arange(steps) * self.config.sample_period_s
        duration = float(command_times[-1])
        knot_count = int(math.ceil(duration / self.config.control_knot_period_s)) + 1
        knot_times = np.linspace(0.0, duration, knot_count)
        basis = np.zeros((steps, knot_count))
        for row, sample_time in enumerate(command_times):
            if sample_time >= knot_times[-1]:
                basis[row, -1] = 1.0
                continue
            right = int(np.searchsorted(knot_times, sample_time, side='right'))
            left = max(0, right - 1)
            span = knot_times[right] - knot_times[left]
            alpha = 0.0 if span <= 0.0 else (sample_time - knot_times[left]) / span
            basis[row, left] = 1.0 - alpha
            basis[row, right] = alpha
        return basis

    def plan(
        self,
        states: Mapping[str, Sequence[float]],
        *,
        workspace_error_bounds_m: Mapping[str, tuple[float, float]] | None = None,
        initial_planned_command_m_s: Mapping[str, float] | None = None,
        allowed_command_directions: Mapping[str, float] | None = None,
    ) -> OpenLoopStopPlan:
        """
        Solve once for a complete stop sequence.

        ``initial_planned_command_m_s`` is normally zero.  The jog-to-stop
        handoff deliberately cuts the old cruise command immediately; the
        first-order actuator state in ``states`` models the physical decay.
        Subsequent samples obey the configured command-slew constraint.

        When ``allowed_command_directions`` is supplied, each value must be
        -1, 0, or +1.  A nonzero value restricts that axis to commands with the
        specified sign; zero holds that axis at zero.  The jog controller uses
        the direction opposite the incoming jog.  A first-order velocity
        actuator driven with one command polarity can cross zero at most once,
        which makes repeated cart reversals impossible in the planning model.
        """
        axes, x0 = self._validated_states(states)
        axis_count = len(axes)
        steps = self.config.step_count
        control_basis = self._control_basis()
        knot_count = control_basis.shape[1]
        variable_count = knot_count * axis_count
        speed = self.config.max_command_speed_m_s
        initial = np.array([
            0.0 if initial_planned_command_m_s is None
            else float(initial_planned_command_m_s.get(axis, 0.0))
            for axis in axes
        ])
        if not np.all(np.isfinite(initial)) or np.linalg.norm(initial) > speed * 1.001:
            raise ValueError('initial planned command exceeds the command-speed limit')
        initial_z = initial / speed

        command_directions = None
        if allowed_command_directions is not None:
            if set(allowed_command_directions) != set(axes):
                raise ValueError('allowed command directions must contain every planned axis')
            command_directions = np.array([
                float(allowed_command_directions[axis]) for axis in axes
            ])
            if (
                not np.all(np.isfinite(command_directions))
                or np.any(~np.isin(command_directions, (-1.0, 0.0, 1.0)))
            ):
                raise ValueError('allowed command directions must be -1, 0, or +1')

        if workspace_error_bounds_m is None:
            workspace_bounds = np.tile(
                [-self.config.max_position_excursion_m,
                 self.config.max_position_excursion_m],
                (axis_count, 1),
            )
        else:
            workspace_bounds = np.array([
                workspace_error_bounds_m[axis] for axis in axes
            ], dtype=float)
            if workspace_bounds.shape != (axis_count, 2):
                raise ValueError('workspace bounds must contain one pair per axis')
            workspace_bounds[:, 0] = np.maximum(
                workspace_bounds[:, 0], -self.config.max_position_excursion_m
            )
            workspace_bounds[:, 1] = np.minimum(
                workspace_bounds[:, 1], self.config.max_position_excursion_m
            )
            if not np.all(np.isfinite(workspace_bounds)) or np.any(
                workspace_bounds[:, 0] >= workspace_bounds[:, 1]
            ):
                raise ValueError('workspace leaves no feasible correction envelope')

        hessian = np.eye(variable_count) * 1.0e-9
        linear = np.zeros(variable_count)
        command_map = sparse.lil_matrix((steps * axis_count, variable_count))
        for k in range(steps):
            for axis_index in range(axis_count):
                for knot_index, coefficient in enumerate(control_basis[k]):
                    if coefficient:
                        command_map[k * axis_count + axis_index,
                                    knot_index * axis_count + axis_index] = coefficient
        command_map = command_map.tocsc()
        terminal_weights = np.diag([
            self.config.terminal_position_weight
            / self.config.terminal_position_tolerance_m**2,
            self.config.terminal_velocity_weight
            / self.config.terminal_velocity_tolerance_m_s**2,
            self.config.terminal_angle_weight
            / self.config.terminal_angle_tolerance_rad**2,
            self.config.terminal_angle_rate_weight
            / self.config.terminal_angle_rate_tolerance_rad_s**2,
        ]) / len(self.scenarios)
        running_scales = np.array([
            self.config.max_position_excursion_m,
            self.config.max_command_speed_m_s,
            math.radians(1.0),
            math.radians(3.0),
        ])
        running_weights = np.diag(np.array([
            self.config.running_position_weight,
            self.config.running_velocity_weight,
            self.config.running_angle_weight,
            self.config.running_angle_rate_weight,
        ]) / running_scales**2)

        state_maps: dict[tuple[str, int, int], np.ndarray] = {}
        free_states: dict[tuple[str, int, int], np.ndarray] = {}
        for scenario in self.scenarios:
            for axis_index in range(axis_count):
                key = (scenario.label, axis_index, steps)
                state_map = self._axis_state_map(
                    scenario.g[steps] @ control_basis,
                    axis_index, axis_count, variable_count, speed
                )
                free_state = scenario.f[steps] @ x0[axis_index]
                state_maps[key] = state_map
                free_states[key] = free_state
                self._add_quadratic_state_cost(
                    hessian, linear, state_map, free_state, terminal_weights
                )

        nominal = self.nominal_scenario
        for k in range(self.config.running_cost_stride, steps,
                       self.config.running_cost_stride):
            for axis_index in range(axis_count):
                state_map = self._axis_state_map(
                    nominal.g[k] @ control_basis,
                    axis_index, axis_count, variable_count, speed
                )
                free_state = nominal.f[k] @ x0[axis_index]
                self._add_quadratic_state_cost(
                    hessian, linear, state_map, free_state,
                    running_weights * self.config.sample_period_s,
                )

        hessian += (
            self.config.effort_weight
            * (command_map.T @ command_map).toarray() / steps
        )
        difference_samples = sparse.lil_matrix(
            (steps * axis_count, steps * axis_count)
        )
        difference_offset = np.zeros(steps * axis_count)
        for k in range(steps):
            for axis_index in range(axis_count):
                row = k * axis_count + axis_index
                difference_samples[row, row] = 1.0
                if k:
                    difference_samples[row, row - axis_count] = -1.0
                else:
                    difference_offset[row] = initial_z[axis_index]
        difference = difference_samples.tocsc() @ command_map
        hessian += self.config.slew_weight * (difference.T @ difference).toarray()
        linear -= self.config.slew_weight * np.asarray(
            difference.T @ difference_offset
        ).reshape(-1)

        constraint_rows: list[sparse.csc_matrix] = []
        lower_bounds: list[np.ndarray] = []
        upper_bounds: list[np.ndarray] = []

        def add_constraints(matrix, lower, upper) -> None:
            matrix = sparse.csc_matrix(matrix)
            lower = np.broadcast_to(np.asarray(lower, dtype=float), (matrix.shape[0],)).copy()
            upper = np.broadcast_to(np.asarray(upper, dtype=float), (matrix.shape[0],)).copy()
            constraint_rows.append(matrix)
            lower_bounds.append(lower)
            upper_bounds.append(upper)

        normals = self._polygon_normals(axis_count)
        polygon_radius = 1.0 if axis_count == 1 else math.cos(
            math.pi / self.config.vector_constraint_facets
        )
        command_rows = sparse.lil_matrix((steps * len(normals), variable_count))
        row = 0
        for k in range(steps):
            for normal in normals:
                for axis_index, coefficient in enumerate(normal):
                    for knot_index, basis_coefficient in enumerate(control_basis[k]):
                        if basis_coefficient:
                            command_rows[
                                row, knot_index * axis_count + axis_index
                            ] = coefficient * basis_coefficient
                row += 1
        add_constraints(command_rows, -np.inf, polygon_radius)

        # Enforce a single-return topology for joystick stops.  The QP used to
        # alternate command polarity several times to reduce its quadratic
        # terminal cost, which was mathematically valid but physically looked
        # like repeated back-and-forth jogging.  This is a hard feasibility
        # constraint so cost tuning can never reintroduce that behavior.
        if command_directions is not None:
            polarity_rows = sparse.lil_matrix(
                (steps * axis_count, variable_count)
            )
            polarity_lower = np.zeros(steps * axis_count)
            polarity_upper = np.full(steps * axis_count, np.inf)
            for k in range(steps):
                for axis_index, direction in enumerate(command_directions):
                    polarity_row = k * axis_count + axis_index
                    if direction == 0.0:
                        direction = 1.0
                        polarity_upper[polarity_row] = 0.0
                    for knot_index, basis_coefficient in enumerate(control_basis[k]):
                        if basis_coefficient:
                            polarity_rows[
                                polarity_row, knot_index * axis_count + axis_index
                            ] = direction * basis_coefficient
            add_constraints(polarity_rows, polarity_lower, polarity_upper)

        lead_steps = min(
            steps,
            int(math.ceil(self.config.command_lead_time_s / self.config.sample_period_s)),
        )
        add_constraints(
            command_map[:lead_steps * axis_count],
            np.zeros(lead_steps * axis_count),
            np.zeros(lead_steps * axis_count),
        )

        slew_rows = sparse.lil_matrix((steps * len(normals), variable_count))
        slew_upper = np.empty(steps * len(normals))
        normalized_slew = (
            self.config.max_command_acceleration_m_s2
            * self.config.sample_period_s / speed
        )
        if axis_count == 2:
            normalized_slew *= polygon_radius
        row = 0
        for k in range(steps):
            for normal in normals:
                for axis_index, coefficient in enumerate(normal):
                    for knot_index in range(knot_count):
                        basis_difference = control_basis[k, knot_index]
                        if k:
                            basis_difference -= control_basis[k - 1, knot_index]
                        if basis_difference:
                            slew_rows[
                                row, knot_index * axis_count + axis_index
                            ] = coefficient * basis_difference
                slew_upper[row] = normalized_slew + (
                    float(normal @ initial_z) if k == 0 else 0.0
                )
                row += 1
        add_constraints(slew_rows, -np.inf, slew_upper)

        # Nominal path position constraints: an inscribed vector envelope plus
        # exact per-axis bounds derived from the physical workspace.
        position_polygon_rows = []
        position_polygon_lower = []
        position_polygon_upper = []
        position_axis_rows = []
        position_axis_lower = []
        position_axis_upper = []
        path_steps = list(range(
            self.config.path_constraint_stride,
            steps + 1,
            self.config.path_constraint_stride,
        ))
        if not path_steps or path_steps[-1] != steps:
            path_steps.append(steps)
        for k in path_steps:
            maps = []
            free = []
            for axis_index in range(axis_count):
                state_map = self._axis_state_map(
                    nominal.g[k] @ control_basis,
                    axis_index, axis_count, variable_count, speed
                )
                maps.append(state_map[0])
                free.append(float((nominal.f[k] @ x0[axis_index])[0]))
                position_axis_rows.append(state_map[0])
                position_axis_lower.append(workspace_bounds[axis_index, 0] - free[-1])
                position_axis_upper.append(workspace_bounds[axis_index, 1] - free[-1])
            map_matrix = np.vstack(maps)
            free_vector = np.asarray(free)
            for normal in normals:
                position_polygon_rows.append(normal @ map_matrix)
                position_polygon_lower.append(-np.inf)
                position_polygon_upper.append(
                    self.config.max_position_excursion_m * polygon_radius
                    - float(normal @ free_vector)
                )
        add_constraints(
            np.asarray(position_polygon_rows),
            np.asarray(position_polygon_lower),
            np.asarray(position_polygon_upper),
        )
        add_constraints(
            np.asarray(position_axis_rows),
            np.asarray(position_axis_lower),
            np.asarray(position_axis_upper),
        )

        tolerances = np.array([
            self.config.terminal_position_tolerance_m,
            self.config.terminal_velocity_tolerance_m_s,
            self.config.terminal_angle_tolerance_rad,
            self.config.terminal_angle_rate_tolerance_rad_s,
        ])
        for scenario in self.scenarios:
            # Off-nominal scenarios shape the objective (specified-insensitivity
            # style) but do not receive hard boxes.  A common tight box across
            # distinct frequencies can make a low-authority stop infeasible.
            # The independent validator below still rejects excessive robust
            # residuals before a command can be returned.
            if scenario.label != self.nominal_scenario.label:
                continue
            for axis_index in range(axis_count):
                key = (scenario.label, axis_index, steps)
                matrix = state_maps[key] / tolerances[:, None]
                offset = free_states[key] / tolerances
                add_constraints(matrix, -1.0 - offset, 1.0 - offset)

        final_rows = sparse.lil_matrix((axis_count, variable_count))
        for axis_index in range(axis_count):
            for knot_index, coefficient in enumerate(control_basis[-1]):
                if coefficient:
                    final_rows[
                        axis_index, knot_index * axis_count + axis_index
                    ] = coefficient
        final_normalized = self.config.final_command_tolerance_m_s / speed
        add_constraints(final_rows, -final_normalized, final_normalized)

        p_matrix = sparse.triu(sparse.csc_matrix(2.0 * hessian), format='csc')
        q_vector = 2.0 * linear
        a_matrix = sparse.vstack(constraint_rows, format='csc')
        lower = np.concatenate(lower_bounds)
        upper = np.concatenate(upper_bounds)

        solver = osqp.OSQP()
        started = time.monotonic()
        solver.setup(
            P=p_matrix,
            q=q_vector,
            A=a_matrix,
            l=lower,
            u=upper,
            verbose=False,
            eps_abs=self.config.solver_absolute_tolerance,
            eps_rel=self.config.solver_relative_tolerance,
            max_iter=self.config.solver_max_iterations,
            polishing=True,
            scaled_termination=True,
            check_termination=25,
            time_limit=self.config.solver_time_limit_s,
        )
        try:
            result = solver.solve(raise_error=False)
        except TypeError:  # OSQP < 1.0 did not expose this keyword.
            result = solver.solve()
        solve_time = time.monotonic() - started
        status = str(result.info.status)
        if result.x is None or result.info.status_val not in (1, 2):
            raise OpenLoopPlanningError(
                f'open-loop stop QP failed: {status}; '
                f'primal_residual={result.info.prim_res:.3g}'
            )
        commands = speed * np.asarray(command_map @ result.x).reshape(steps, axis_count)
        predicted = self._predict_all(x0, commands)
        self._validate_solution(
            commands,
            predicted,
            workspace_bounds,
            initial,
            command_directions,
        )
        return OpenLoopStopPlan(
            axes=axes,
            commands_m_s=commands,
            predicted_states=predicted,
            solve_time_s=solve_time,
            solver_status=status,
            objective=float(result.info.obj_val),
            iterations=int(result.info.iter),
            sample_period_s=self.config.sample_period_s,
            command_lead_time_s=self.config.command_lead_time_s,
        )

    def _predict_all(
        self,
        x0: np.ndarray,
        commands: np.ndarray,
    ) -> dict[str, np.ndarray]:
        predicted = {}
        for scenario in self.scenarios:
            states = np.empty((commands.shape[0] + 1, commands.shape[1], STATE_SIZE))
            states[0] = x0
            for k in range(commands.shape[0]):
                for axis_index in range(commands.shape[1]):
                    states[k + 1, axis_index] = (
                        scenario.a @ states[k, axis_index]
                        + scenario.b[:, 0] * commands[k, axis_index]
                    )
            predicted[scenario.label] = states
        return predicted

    def _validate_solution(
        self,
        commands: np.ndarray,
        predicted: Mapping[str, np.ndarray],
        workspace_bounds: np.ndarray,
        initial_command: np.ndarray,
        command_directions: np.ndarray | None,
    ) -> None:
        """Reject numerically inaccurate output before it can reach hardware."""
        relative_margin = 1.01
        if not np.all(np.isfinite(commands)):
            raise OpenLoopPlanningError('solver returned a non-finite command')
        magnitudes = np.linalg.norm(commands, axis=1)
        if np.max(magnitudes) > self.config.max_command_speed_m_s * relative_margin:
            raise OpenLoopPlanningError('post-solve command-speed validation failed')
        prior = np.vstack((initial_command, commands[:-1]))
        slew = np.linalg.norm(commands - prior, axis=1) / self.config.sample_period_s
        if np.max(slew) > self.config.max_command_acceleration_m_s2 * relative_margin:
            raise OpenLoopPlanningError('post-solve command-slew validation failed')
        if np.max(np.abs(commands[-1])) > self.config.final_command_tolerance_m_s * 1.05:
            raise OpenLoopPlanningError('post-solve final-command validation failed')
        if command_directions is not None:
            signed_commands = commands * command_directions[None, :]
            for axis_index, direction in enumerate(command_directions):
                if direction == 0.0:
                    if np.max(np.abs(commands[:, axis_index])) > 2.0e-5:
                        raise OpenLoopPlanningError(
                            'post-solve zero-axis command-direction validation failed'
                        )
                elif np.min(signed_commands[:, axis_index]) < -2.0e-5:
                    raise OpenLoopPlanningError(
                        'post-solve single-return command-direction validation failed'
                    )
        lead_steps = int(math.ceil(
            self.config.command_lead_time_s / self.config.sample_period_s
        ))
        if np.max(np.abs(commands[:lead_steps])) > 2.0e-5:
            raise OpenLoopPlanningError('post-solve planning-lead validation failed')

        nominal_states = predicted[self.nominal_scenario.label]
        position_magnitude = np.linalg.norm(nominal_states[:, :, 0], axis=1)
        if np.max(position_magnitude) > self.config.max_position_excursion_m * relative_margin:
            raise OpenLoopPlanningError('post-solve correction-envelope validation failed')
        for axis_index in range(commands.shape[1]):
            positions = nominal_states[:, axis_index, 0]
            if (
                np.min(positions) < workspace_bounds[axis_index, 0] - 1.0e-5
                or np.max(positions) > workspace_bounds[axis_index, 1] + 1.0e-5
            ):
                raise OpenLoopPlanningError('post-solve workspace validation failed')

        nominal_tolerances = np.array([
            self.config.terminal_position_tolerance_m,
            self.config.terminal_velocity_tolerance_m_s,
            self.config.terminal_angle_tolerance_rad,
            self.config.terminal_angle_rate_tolerance_rad_s,
        ])
        for label, states in predicted.items():
            multiplier = (
                1.0 if label == self.nominal_scenario.label
                else self.config.robust_terminal_tolerance_multiplier
            )
            tolerances = nominal_tolerances * multiplier * 1.02
            if np.any(np.abs(states[-1]) > tolerances):
                residual = np.max(np.abs(states[-1]) / tolerances)
                raise OpenLoopPlanningError(
                    f'post-solve robust terminal validation failed for {label} ({residual:.3f}x)'
                )


class CraneStateObserver:
    """Small continuous-running Kalman observer used to estimate angle rate."""

    def __init__(
        self,
        model: CraneModel,
        *,
        angle_measurement_std_rad: float = math.radians(0.06),
    ):
        self.model = model
        self.angle_measurement_std_rad = float(angle_measurement_std_rad)
        if (
            not math.isfinite(self.angle_measurement_std_rad)
            or self.angle_measurement_std_rad <= 0
        ):
            raise ValueError('angle_measurement_std_rad must be positive and finite')
        self.state: np.ndarray | None = None
        self.covariance = np.diag([1.0e-6, 2.0e-3, 2.0e-5, 4.0e-3])
        self.last_time_s: float | None = None
        self.last_command_m_s = 0.0
        self.c = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ])
        self.measurement_covariance = np.diag([
            (0.0005) ** 2,
            (0.0060) ** 2,
            self.angle_measurement_std_rad**2,
        ])

    def reset(self) -> None:
        self.state = None
        self.covariance = np.diag([1.0e-6, 2.0e-3, 2.0e-5, 4.0e-3])
        self.last_time_s = None
        self.last_command_m_s = 0.0

    def update(
        self,
        time_s: float,
        cart_position_m: float,
        cart_velocity_m_s: float,
        payload_angle_rad: float,
        applied_command_m_s: float,
    ) -> np.ndarray:
        values = np.array([
            time_s, cart_position_m, cart_velocity_m_s,
            payload_angle_rad, applied_command_m_s,
        ], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError('observer inputs must be finite')
        measurement = np.array([cart_position_m, cart_velocity_m_s, payload_angle_rad])
        if self.state is None or self.last_time_s is None or time_s <= self.last_time_s:
            self.state = np.array([cart_position_m, cart_velocity_m_s, payload_angle_rad, 0.0])
            self.last_time_s = time_s
            self.last_command_m_s = applied_command_m_s
            return self.state.copy()

        dt = min(0.10, max(1.0e-4, time_s - self.last_time_s))
        a, b = discrete_crane_model(self.model, dt)
        predicted_state = a @ self.state + b[:, 0] * self.last_command_m_s
        # Continuous disturbance intensities, scaled by elapsed time.  Cart
        # acceleration and angle acceleration receive most of the model slack.
        process_covariance = np.diag([1.0e-10, 8.0e-4, 2.0e-9, 8.0e-3]) * dt
        predicted_covariance = a @ self.covariance @ a.T + process_covariance
        innovation_covariance = (
            self.c @ predicted_covariance @ self.c.T + self.measurement_covariance
        )
        gain = np.linalg.solve(
            innovation_covariance,
            self.c @ predicted_covariance,
        ).T
        innovation = measurement - self.c @ predicted_state
        self.state = predicted_state + gain @ innovation
        identity_update = np.eye(STATE_SIZE) - gain @ self.c
        # Joseph form preserves positive semidefiniteness under round-off.
        self.covariance = (
            identity_update @ predicted_covariance @ identity_update.T
            + gain @ self.measurement_covariance @ gain.T
        )
        self.last_time_s = time_s
        self.last_command_m_s = applied_command_m_s
        return self.state.copy()
