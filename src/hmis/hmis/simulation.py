"""Small simulation comparing direct and HMIS commands on a pendulum model."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from .shaper import CausalInputShaper, ImpulseShaper


def human_command(time_s: float) -> tuple[float, float]:
    """An irregular 2-D operator input used only as a repeatable benchmark."""
    if 0.5 <= time_s < 1.7:
        return 70.0, 0.0
    if 1.7 <= time_s < 2.8:
        return 35.0, 55.0
    if 2.8 <= time_s < 3.6:
        return -25.0, 55.0
    if 3.6 <= time_s < 4.4:
        return -30.0, -20.0
    return 0.0, 0.0


class PendulumAxis:
    def __init__(self, frequency_hz: float, damping_ratio: float):
        self.omega = 2.0 * math.pi * frequency_hz
        self.damping_ratio = damping_ratio
        self.rope_length_m = 9.80665 / (self.omega * self.omega)
        self.angle_rad = 0.0
        self.angular_velocity_rad_s = 0.0
        self.last_cart_velocity_m_s = 0.0

    def step(self, cart_velocity_mm_s: float, dt: float) -> float:
        cart_velocity_m_s = cart_velocity_mm_s / 1000.0
        cart_acceleration = (cart_velocity_m_s - self.last_cart_velocity_m_s) / dt
        angular_acceleration = (
            -2.0 * self.damping_ratio * self.omega * self.angular_velocity_rad_s
            - self.omega * self.omega * self.angle_rad
            - cart_acceleration / self.rope_length_m
        )
        self.angular_velocity_rad_s += angular_acceleration * dt
        self.angle_rad += self.angular_velocity_rad_s * dt
        self.last_cart_velocity_m_s = cart_velocity_m_s
        return 1000.0 * self.rope_length_m * self.angle_rad


def run_simulation(
    frequency_hz: float = 0.5,
    damping_ratio: float = 0.01,
    dt: float = 0.002,
    duration_s: float = 10.0,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    shaper = CausalInputShaper(ImpulseShaper.zvd(frequency_hz, damping_ratio))
    direct_plants = (PendulumAxis(frequency_hz, damping_ratio),
                     PendulumAxis(frequency_hz, damping_ratio))
    shaped_plants = (PendulumAxis(frequency_hz, damping_ratio),
                     PendulumAxis(frequency_hz, damping_ratio))
    records: list[dict[str, float]] = []
    residual_start_s = 7.0
    steps = int(duration_s / dt) + 1

    for index in range(steps):
        now = index * dt
        human = human_command(now)
        shaped = shaper.shape(now, human)
        direct_swing = tuple(
            plant.step(command, dt) for plant, command in zip(direct_plants, human)
        )
        shaped_swing = tuple(
            plant.step(command, dt) for plant, command in zip(shaped_plants, shaped)
        )
        records.append({
            'time_s': now,
            'human_vx_mm_s': human[0],
            'human_vy_mm_s': human[1],
            'shaped_vx_mm_s': shaped[0],
            'shaped_vy_mm_s': shaped[1],
            'direct_swing_x_mm': direct_swing[0],
            'direct_swing_y_mm': direct_swing[1],
            'hmis_swing_x_mm': shaped_swing[0],
            'hmis_swing_y_mm': shaped_swing[1],
        })

    residual = [record for record in records if record['time_s'] >= residual_start_s]
    direct_rms = _vector_rms(residual, 'direct_swing_x_mm', 'direct_swing_y_mm')
    hmis_rms = _vector_rms(residual, 'hmis_swing_x_mm', 'hmis_swing_y_mm')
    reduction = 100.0 * (1.0 - hmis_rms / direct_rms) if direct_rms > 0.0 else 0.0
    return records, {
        'direct_residual_rms_mm': direct_rms,
        'hmis_residual_rms_mm': hmis_rms,
        'residual_reduction_percent': reduction,
    }


def _vector_rms(records: list[dict[str, float]], x_key: str, y_key: str) -> float:
    if not records:
        return 0.0
    energy = sum(record[x_key] ** 2 + record[y_key] ** 2 for record in records)
    return math.sqrt(energy / len(records))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frequency-hz', type=float, default=0.5)
    parser.add_argument('--damping-ratio', type=float, default=0.01)
    parser.add_argument('--csv', type=Path, help='optional output CSV')
    args = parser.parse_args()
    records, metrics = run_simulation(args.frequency_hz, args.damping_ratio)

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
        print(f'wrote {args.csv}')

    print(f"direct residual RMS: {metrics['direct_residual_rms_mm']:.3f} mm")
    print(f"HMIS residual RMS:   {metrics['hmis_residual_rms_mm']:.3f} mm")
    print(f"reduction:           {metrics['residual_reduction_percent']:.1f}%")


if __name__ == '__main__':
    main()
