import math

import numpy as np

from optimal_stop_planner import (
    G,
    CraneModel,
    CraneStateObserver,
    ForwardOnlyMinimumTimeStopPlanner,
    MinimumTimeStopConfig,
    OptimalStopConfig,
    RobustOpenLoopStopPlanner,
    discrete_crane_model,
)


def representative_release_states():
    return {
        'x': [0.0, 0.100, math.radians(0.20), math.radians(1.0)],
        'y': [0.0, 0.000, math.radians(-0.10), math.radians(0.5)],
    }


def test_exact_discrete_model_has_semigroup_property():
    model = CraneModel(0.90, 0.02, 0.045)
    a_half, b_half = discrete_crane_model(model, 0.01)
    a_full, b_full = discrete_crane_model(model, 0.02)
    assert np.allclose(a_half @ a_half, a_full, atol=1.0e-12)
    assert np.allclose(a_half @ b_half + b_half, b_full, atol=1.0e-12)


def test_robust_plan_obeys_limits_and_terminal_boxes():
    config = OptimalStopConfig(
        horizon_s=5.0,
        command_lead_time_s=1.0,
        solver_time_limit_s=1.0,
    )
    planner = RobustOpenLoopStopPlanner(config)
    plan = planner.plan(representative_release_states())

    lead_steps = math.ceil(config.command_lead_time_s / config.sample_period_s)
    assert np.max(np.abs(plan.commands_m_s[:lead_steps])) <= 2.0e-5
    assert np.max(np.linalg.norm(plan.commands_m_s, axis=1)) <= (
        1.01 * config.max_command_speed_m_s
    )
    previous = np.vstack((np.zeros(2), plan.commands_m_s[:-1]))
    slew = np.linalg.norm(plan.commands_m_s - previous, axis=1) / config.sample_period_s
    assert np.max(slew) <= 1.01 * config.max_command_acceleration_m_s2

    nominal_tolerances = np.array([
        config.terminal_position_tolerance_m,
        config.terminal_velocity_tolerance_m_s,
        config.terminal_angle_tolerance_rad,
        config.terminal_angle_rate_tolerance_rad_s,
    ])
    for scenario in planner.scenarios:
        multiplier = (
            1.0 if scenario.label == planner.nominal_scenario.label
            else config.robust_terminal_tolerance_multiplier
        )
        terminal = plan.predicted_states[scenario.label][-1]
        assert np.all(np.abs(terminal) <= nominal_tolerances * multiplier * 1.02)


def test_robust_plan_reduces_worst_case_sway_energy_by_ninety_percent():
    config = OptimalStopConfig(
        horizon_s=5.0,
        command_lead_time_s=1.0,
        solver_time_limit_s=1.0,
    )
    planner = RobustOpenLoopStopPlanner(config)
    release = representative_release_states()
    initial = np.array(list(release.values()))
    plan = planner.plan(release)

    ratios = []
    for scenario in planner.scenarios:
        hard_stop = initial.copy()
        for _ in range(config.step_count):
            for axis_index in range(hard_stop.shape[0]):
                hard_stop[axis_index] = scenario.a @ hard_stop[axis_index]
        optimized = plan.predicted_states[scenario.label][-1]
        omega = math.sqrt(G / scenario.model.rope_length_m)

        def sway_energy(states):
            return float(np.sum(states[:, 2] ** 2 + (states[:, 3] / omega) ** 2))

        ratios.append(sway_energy(optimized) / sway_energy(hard_stop))
    assert max(ratios) < 0.10


def test_minimum_time_stop_is_forward_only_and_reaches_robust_rest():
    config = MinimumTimeStopConfig(
        maximum_horizon_s=3.0,
        solver_time_limit_s=0.5,
    )
    planner = ForwardOnlyMinimumTimeStopPlanner(config)
    release = {
        'x': [0.0, 0.100, math.radians(0.04), math.radians(0.15)],
    }
    plan = planner.plan(
        release,
        direction_signs={'x': 1.0},
        initial_commands_m_s={'x': 0.100},
        forward_workspace_m={'x': 0.300},
    )

    assert config.minimum_horizon_s <= plan.duration_s <= config.maximum_horizon_s
    assert plan.duration_s < 1.20
    assert np.min(plan.commands_m_s[:, 0]) >= -2.0e-6
    for scenario in planner.scenarios:
        states = plan.predicted_states[scenario.label][:, 0]
        assert np.min(states[:, 1]) >= -2.0e-5
        assert np.min(np.diff(states[:, 0])) >= -2.0e-5
        multiplier = (
            1.0 if scenario.label == planner.nominal_scenario.label
            else config.robust_terminal_tolerance_multiplier
        )
        omega = math.sqrt(G / scenario.model.rope_length_m)
        residual = math.hypot(states[-1, 2], states[-1, 3] / omega)
        assert states[-1, 1] <= (
            1.02 * multiplier * config.terminal_velocity_tolerance_m_s
        )
        assert residual <= (
            1.02 * multiplier * config.terminal_angle_tolerance_rad
        )
    nominal = plan.predicted_states[planner.nominal_scenario.label][:, 0]
    assert 0.010 < nominal[-1, 0] < 0.100


def test_minimum_time_stop_handles_negative_world_direction_without_reversal():
    config = MinimumTimeStopConfig(
        maximum_horizon_s=3.0,
        solver_time_limit_s=0.5,
    )
    planner = ForwardOnlyMinimumTimeStopPlanner(config)
    plan = planner.plan(
        {'x': [0.0, -0.100, math.radians(-0.03), math.radians(-0.1)]},
        direction_signs={'x': -1.0},
        initial_commands_m_s={'x': -0.100},
        forward_workspace_m={'x': 0.300},
    )

    assert np.max(plan.commands_m_s[:, 0]) <= 2.0e-6
    signed_states = plan.predicted_states[planner.nominal_scenario.label][:, 0]
    assert np.min(signed_states[:, 1]) >= -2.0e-5


def test_minimum_time_stop_rejects_insufficient_forward_workspace():
    config = MinimumTimeStopConfig(
        maximum_horizon_s=2.0,
        solver_time_limit_s=0.5,
    )
    planner = ForwardOnlyMinimumTimeStopPlanner(config)

    try:
        planner.plan(
            {'x': [0.0, 0.100, 0.0, 0.0]},
            direction_signs={'x': 1.0},
            initial_commands_m_s={'x': 0.100},
            forward_workspace_m={'x': 0.005},
        )
    except RuntimeError as exc:
        assert 'no forward-only stop is feasible' in str(exc)
    else:
        raise AssertionError('planner accepted an impossible 5 mm stopping space')


def test_minimum_time_diagonal_stop_obeys_vector_limit_and_axis_signs():
    config = MinimumTimeStopConfig(
        maximum_horizon_s=3.0,
        solver_time_limit_s=0.5,
    )
    planner = ForwardOnlyMinimumTimeStopPlanner(config)
    component = 0.100 / math.sqrt(2.0)
    plan = planner.plan(
        {
            'x': [0.0, component, math.radians(0.03), math.radians(0.1)],
            'y': [0.0, -component, math.radians(-0.02), math.radians(-0.1)],
        },
        direction_signs={'x': 1.0, 'y': -1.0},
        initial_commands_m_s={'x': component, 'y': -component},
        forward_workspace_m={'x': 0.250, 'y': 0.250},
    )

    assert np.min(plan.commands_m_s[:, 0]) >= -2.0e-6
    assert np.max(plan.commands_m_s[:, 1]) <= 2.0e-6
    assert np.max(np.linalg.norm(plan.commands_m_s, axis=1)) <= (
        1.02 * config.max_command_speed_m_s
    )


def test_observer_recovers_angle_rate_during_driven_motion():
    model = CraneModel(0.90, 0.02, 0.045)
    dt = 0.01
    a, b = discrete_crane_model(model, dt)
    observer = CraneStateObserver(model)
    state = np.array([0.0, 0.0, math.radians(0.8), math.radians(-1.2)])
    command = 0.04
    observer.update(0.0, state[0], state[1], state[2], command)

    for k in range(1, 501):
        state = a @ state + b[:, 0] * command
        if k == 120:
            command = -0.02
        elif k == 260:
            command = 0.0
        estimate = observer.update(
            k * dt, state[0], state[1], state[2], command
        )
    assert abs(estimate[3] - state[3]) < math.radians(0.08)
