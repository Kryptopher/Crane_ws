"""Model-based precision stopping for a velocity-controlled gantry crane."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from scipy.linalg import expm, solve_discrete_are


@dataclass(frozen=True)
class PrecisionStopConfig:
    """Parameters for one Cartesian axis of the precision-stop controller."""

    sample_period_s: float = 0.01
    rope_length_m: float = 0.90
    damping_ratio: float = 0.02
    actuator_time_constant_s: float = 0.045
    max_command_speed_m_s: float = 0.060
    max_command_acceleration_m_s2: float = 0.150
    q_cart_position: float = 500.0
    q_cart_velocity: float = 0.5
    q_payload_angle: float = 300.0
    q_payload_angle_rate: float = 20.0
    r_command: float = 8.0

    def __post_init__(self) -> None:
        positive = {
            'sample_period_s': self.sample_period_s,
            'rope_length_m': self.rope_length_m,
            'actuator_time_constant_s': self.actuator_time_constant_s,
            'max_command_speed_m_s': self.max_command_speed_m_s,
            'max_command_acceleration_m_s2': self.max_command_acceleration_m_s2,
            'q_cart_position': self.q_cart_position,
            'q_cart_velocity': self.q_cart_velocity,
            'q_payload_angle': self.q_payload_angle,
            'q_payload_angle_rate': self.q_payload_angle_rate,
            'r_command': self.r_command,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f'{name} must be positive and finite')
        if not math.isfinite(self.damping_ratio) or not 0.0 <= self.damping_ratio < 1.0:
            raise ValueError('damping_ratio must be in [0, 1)')


def discrete_cart_pendulum_model(
    sample_period_s: float,
    rope_length_m: float,
    damping_ratio: float,
    actuator_time_constant_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact zero-order-hold matrices for ``[p, v, theta, theta_dot]``.

    The controller input is cart velocity command. Cart velocity follows that
    command through a first-order actuator model, matching the existing
    payload stabilizer's state definition.
    """
    omega = math.sqrt(9.80665 / rope_length_m)
    tau = actuator_time_constant_s
    continuous_a = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, -1.0 / tau, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 1.0 / (rope_length_m * tau), -omega * omega,
         -2.0 * damping_ratio * omega],
    ])
    continuous_b = np.array([
        [0.0],
        [1.0 / tau],
        [0.0],
        [-1.0 / (rope_length_m * tau)],
    ])
    augmented = np.zeros((5, 5))
    augmented[:4, :4] = continuous_a
    augmented[:4, 4:] = continuous_b
    discrete = expm(augmented * sample_period_s)
    return discrete[:4, :4], discrete[:4, 4:]


class PrecisionStopController:
    """Full-state LQR stop controller with command speed and slew limits.

    The stop target is the physical cart position captured when the operator
    centers the stick. The controller is deliberately ROS-independent so the
    stop law can be validated before it is allowed to command hardware.
    """

    def __init__(self, config: PrecisionStopConfig = PrecisionStopConfig()):
        self.config = config
        self.a, self.b = discrete_cart_pendulum_model(
            config.sample_period_s,
            config.rope_length_m,
            config.damping_ratio,
            config.actuator_time_constant_s,
        )
        q = np.diag([
            config.q_cart_position,
            config.q_cart_velocity,
            config.q_payload_angle,
            config.q_payload_angle_rate,
        ])
        r = np.array([[config.r_command]])
        p = solve_discrete_are(self.a, self.b, q, r)
        self.gain = np.linalg.solve(r + self.b.T @ p @ self.b, self.b.T @ p @ self.a)
        self.target_position_m = 0.0
        self.previous_command_m_s = 0.0
        self.active = False

    def start(self, target_position_m: float, previous_command_m_s: float) -> None:
        """Capture the operator-selected position and enter precision-stop mode."""
        if not math.isfinite(target_position_m) or not math.isfinite(previous_command_m_s):
            raise ValueError('start state must be finite')
        self.target_position_m = float(target_position_m)
        self.previous_command_m_s = float(previous_command_m_s)
        self.active = True

    def stop(self) -> None:
        self.previous_command_m_s = 0.0
        self.active = False

    def update(self, state: Sequence[float], dt_s: float | None = None) -> float:
        """Return the next bounded velocity command in m/s.

        ``state`` is absolute ``[cart_position, cart_velocity, payload_angle,
        payload_angle_rate]`` in SI units.
        """
        if not self.active:
            return 0.0
        if len(state) != 4 or not all(math.isfinite(float(value)) for value in state):
            raise ValueError('state must contain four finite values')
        dt = self.config.sample_period_s if dt_s is None else float(dt_s)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError('dt_s must be positive and finite')

        error_state = np.array([
            float(state[0]) - self.target_position_m,
            float(state[1]),
            float(state[2]),
            float(state[3]),
        ])
        requested = float((-self.gain @ error_state)[0])
        speed = self.config.max_command_speed_m_s
        requested = max(-speed, min(speed, requested))

        max_delta = self.config.max_command_acceleration_m_s2 * dt
        command = max(
            self.previous_command_m_s - max_delta,
            min(self.previous_command_m_s + max_delta, requested),
        )
        self.previous_command_m_s = command
        return command

    @staticmethod
    def is_settled(
        state: Sequence[float],
        target_position_m: float,
        position_tolerance_m: float = 0.008,
        velocity_tolerance_m_s: float = 0.003,
        angle_tolerance_rad: float = math.radians(0.18),
        angle_rate_tolerance_rad_s: float = math.radians(0.50),
    ) -> bool:
        return (
            abs(float(state[0]) - target_position_m) <= position_tolerance_m
            and abs(float(state[1])) <= velocity_tolerance_m_s
            and abs(float(state[2])) <= angle_tolerance_rad
            and abs(float(state[3])) <= angle_rate_tolerance_rad_s
        )
