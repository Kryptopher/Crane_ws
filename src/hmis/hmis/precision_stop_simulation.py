"""Compare the ZVD tail, a hard stop, and model-based precision stopping."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from .precision_stop import (
    PrecisionStopConfig,
    PrecisionStopController,
    discrete_cart_pendulum_model,
)
from .shaper import ImpulseShaper


@dataclass(frozen=True)
class StopMetrics:
    peak_forward_excursion_mm: float
    peak_absolute_excursion_mm: float
    final_position_error_mm: float
    first_near_zero_velocity_s: float | None
    precision_settle_time_s: float | None
    peak_payload_angle_deg: float
    residual_angle_rms_deg: float
    peak_command_speed_mm_s: float
    peak_command_acceleration_mm_s2: float


class LinearCranePlant:
    """Exact discrete linear plant used only by the software experiment."""

    def __init__(
        self,
        dt_s: float,
        rope_length_m: float,
        damping_ratio: float,
        actuator_time_constant_s: float,
        initial_velocity_m_s: float,
        initial_angle_rad: float = 0.0,
        initial_angle_rate_rad_s: float = 0.0,
    ):
        self.a, self.b = discrete_cart_pendulum_model(
            dt_s,
            rope_length_m,
            damping_ratio,
            actuator_time_constant_s,
        )
        self.state = np.array([
            0.0,
            initial_velocity_m_s,
            initial_angle_rad,
            initial_angle_rate_rad_s,
        ])

    def step(self, command_m_s: float) -> np.ndarray:
        self.state = self.a @ self.state + self.b[:, 0] * command_m_s
        return self.state.copy()


def run_precision_stop_comparison(
    cruise_speed_m_s: float = 0.060,
    mode_frequency_hz: float = 0.5,
    controller_frequency_hz: float | None = None,
    damping_ratio: float = 0.01,
    actuator_time_constant_s: float = 0.045,
    max_command_acceleration_m_s2: float = 0.150,
    dt_s: float = 0.01,
    duration_s: float = 8.0,
    initial_angle_deg: float = 0.0,
    initial_angle_rate_deg_s: float = 0.0,
) -> tuple[list[dict[str, float | str]], dict[str, StopMetrics]]:
    """Run three controllers from the instant the operator requests stop."""
    rope_length_m = 9.80665 / (2.0 * math.pi * mode_frequency_hz) ** 2
    model_frequency_hz = mode_frequency_hz if controller_frequency_hz is None else (
        controller_frequency_hz
    )
    model_rope_length_m = 9.80665 / (2.0 * math.pi * model_frequency_hz) ** 2
    shaper = ImpulseShaper.zvd(mode_frequency_hz, damping_ratio)
    precision_config = PrecisionStopConfig(
        sample_period_s=dt_s,
        rope_length_m=model_rope_length_m,
        damping_ratio=damping_ratio,
        actuator_time_constant_s=actuator_time_constant_s,
        max_command_speed_m_s=abs(cruise_speed_m_s),
        max_command_acceleration_m_s2=max_command_acceleration_m_s2,
    )
    precision = PrecisionStopController(precision_config)
    precision.start(target_position_m=0.0, previous_command_m_s=cruise_speed_m_s)

    initial_angle = math.radians(initial_angle_deg)
    initial_rate = math.radians(initial_angle_rate_deg_s)
    plants = {
        name: LinearCranePlant(
            dt_s,
            rope_length_m,
            damping_ratio,
            actuator_time_constant_s,
            cruise_speed_m_s,
            initial_angle,
            initial_rate,
        )
        for name in ('zvd_tail', 'hard_stop', 'precision_stop')
    }
    records: list[dict[str, float | str]] = []
    previous_commands = {name: cruise_speed_m_s for name in plants}
    steps = int(duration_s / dt_s) + 1

    for index in range(steps):
        now = index * dt_s
        zvd_command = cruise_speed_m_s * sum(
            amplitude
            for delay, amplitude in zip(shaper.delays_s, shaper.amplitudes)
            if delay > now + 1.0e-12
        )
        commands = {
            'zvd_tail': zvd_command,
            'hard_stop': 0.0,
            'precision_stop': precision.update(plants['precision_stop'].state, dt_s),
        }
        for name, plant in plants.items():
            command = commands[name]
            state = plant.step(command)
            command_acceleration = (command - previous_commands[name]) / dt_s
            previous_commands[name] = command
            records.append({
                'strategy': name,
                'time_s': now,
                'cart_position_mm': 1000.0 * state[0],
                'cart_velocity_mm_s': 1000.0 * state[1],
                'payload_angle_deg': math.degrees(state[2]),
                'payload_angle_rate_deg_s': math.degrees(state[3]),
                'command_velocity_mm_s': 1000.0 * command,
                'command_acceleration_mm_s2': 1000.0 * command_acceleration,
            })

    metrics = {
        name: _calculate_metrics(records, name, dt_s)
        for name in plants
    }
    return records, metrics


def _calculate_metrics(
    records: list[dict[str, float | str]],
    strategy: str,
    dt_s: float,
) -> StopMetrics:
    samples = [record for record in records if record['strategy'] == strategy]
    residual_samples = samples[-max(1, int(1.0 / dt_s)):]
    first_near_zero = next(
        (
            float(sample['time_s'])
            for sample in samples
            if abs(float(sample['cart_velocity_mm_s'])) <= 5.0
        ),
        None,
    )

    hold_samples = max(1, int(0.5 / dt_s))
    settle_time = None
    settled_run = 0
    for sample in samples:
        state = (
            float(sample['cart_position_mm']) / 1000.0,
            float(sample['cart_velocity_mm_s']) / 1000.0,
            math.radians(float(sample['payload_angle_deg'])),
            math.radians(float(sample['payload_angle_rate_deg_s'])),
        )
        if PrecisionStopController.is_settled(state, 0.0):
            settled_run += 1
            if settled_run >= hold_samples:
                settle_time = float(sample['time_s']) - (hold_samples - 1) * dt_s
                break
        else:
            settled_run = 0

    accelerations = [abs(float(sample['command_acceleration_mm_s2'])) for sample in samples]
    return StopMetrics(
        peak_forward_excursion_mm=max(float(sample['cart_position_mm']) for sample in samples),
        peak_absolute_excursion_mm=max(
            abs(float(sample['cart_position_mm'])) for sample in samples
        ),
        final_position_error_mm=abs(float(samples[-1]['cart_position_mm'])),
        first_near_zero_velocity_s=first_near_zero,
        precision_settle_time_s=settle_time,
        peak_payload_angle_deg=max(abs(float(sample['payload_angle_deg'])) for sample in samples),
        residual_angle_rms_deg=math.sqrt(sum(
            float(sample['payload_angle_deg']) ** 2 for sample in residual_samples
        ) / len(residual_samples)),
        peak_command_speed_mm_s=max(
            abs(float(sample['command_velocity_mm_s'])) for sample in samples
        ),
        peak_command_acceleration_mm_s2=max(accelerations),
    )


def _format_optional(value: float | None) -> str:
    return '--' if value is None else f'{value:.2f} s'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--speed-mm-s', type=float, default=60.0)
    parser.add_argument('--max-accel-mm-s2', type=float, default=150.0)
    parser.add_argument('--frequency-hz', type=float, default=0.5)
    parser.add_argument(
        '--controller-frequency-hz', type=float,
        help='controller model frequency; default matches the simulated plant',
    )
    parser.add_argument('--damping-ratio', type=float, default=0.01)
    parser.add_argument('--initial-angle-deg', type=float, default=0.0)
    parser.add_argument('--initial-angle-rate-deg-s', type=float, default=0.0)
    parser.add_argument('--csv', type=Path)
    args = parser.parse_args()
    records, metrics = run_precision_stop_comparison(
        cruise_speed_m_s=args.speed_mm_s / 1000.0,
        mode_frequency_hz=args.frequency_hz,
        controller_frequency_hz=args.controller_frequency_hz,
        damping_ratio=args.damping_ratio,
        max_command_acceleration_m_s2=args.max_accel_mm_s2 / 1000.0,
        initial_angle_deg=args.initial_angle_deg,
        initial_angle_rate_deg_s=args.initial_angle_rate_deg_s,
    )

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
        print(f'wrote {args.csv}')

    print('strategy        peak +travel   final error   near-zero v   settled   residual angle')
    for name, result in metrics.items():
        print(
            f'{name:15s} {result.peak_forward_excursion_mm:8.2f} mm '
            f'{result.final_position_error_mm:10.2f} mm '
            f'{_format_optional(result.first_near_zero_velocity_s):>12s} '
            f'{_format_optional(result.precision_settle_time_s):>12s} '
            f'{result.residual_angle_rms_deg:10.4f} deg')


if __name__ == '__main__':
    main()
