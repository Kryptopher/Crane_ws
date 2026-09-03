"""ROS-independent input-shaping primitives used by HMIS."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Deque, Sequence, Tuple


Vector2 = Tuple[float, float]


@dataclass(frozen=True)
class ImpulseShaper:
    """A causal impulse sequence whose amplitudes have unity DC gain."""

    delays_s: Tuple[float, ...]
    amplitudes: Tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.delays_s or len(self.delays_s) != len(self.amplitudes):
            raise ValueError('delays and amplitudes must be non-empty and equal length')
        if any(not math.isfinite(value) for value in self.delays_s + self.amplitudes):
            raise ValueError('shaper values must be finite')
        if self.delays_s[0] != 0.0 or any(delay < 0.0 for delay in self.delays_s):
            raise ValueError('shaper must be causal and start at zero delay')
        if any(b <= a for a, b in zip(self.delays_s, self.delays_s[1:])):
            raise ValueError('shaper delays must be strictly increasing')
        if not math.isclose(sum(self.amplitudes), 1.0, abs_tol=1e-9):
            raise ValueError('shaper amplitudes must sum to one')

    @property
    def horizon_s(self) -> float:
        return self.delays_s[-1]

    @classmethod
    def zv(cls, natural_frequency_hz: float, damping_ratio: float = 0.0) -> 'ImpulseShaper':
        """Construct a two-impulse zero-vibration (ZV) shaper."""
        spacing, decay = _mode_terms(natural_frequency_hz, damping_ratio)
        denominator = 1.0 + decay
        return cls(
            delays_s=(0.0, spacing),
            amplitudes=(1.0 / denominator, decay / denominator),
        )

    @classmethod
    def zvd(cls, natural_frequency_hz: float, damping_ratio: float = 0.0) -> 'ImpulseShaper':
        """Construct a robust three-impulse ZVD shaper."""
        spacing, decay = _mode_terms(natural_frequency_hz, damping_ratio)
        denominator = (1.0 + decay) ** 2
        return cls(
            delays_s=(0.0, spacing, 2.0 * spacing),
            amplitudes=(1.0 / denominator, 2.0 * decay / denominator,
                        decay * decay / denominator),
        )


def _mode_terms(natural_frequency_hz: float, damping_ratio: float) -> Tuple[float, float]:
    if not math.isfinite(natural_frequency_hz) or natural_frequency_hz <= 0.0:
        raise ValueError('natural_frequency_hz must be positive and finite')
    if not math.isfinite(damping_ratio) or not 0.0 <= damping_ratio < 1.0:
        raise ValueError('damping_ratio must be in [0, 1)')
    root = math.sqrt(1.0 - damping_ratio * damping_ratio)
    omega_n = 2.0 * math.pi * natural_frequency_hz
    spacing = math.pi / (omega_n * root)
    decay = math.exp(-math.pi * damping_ratio / root)
    return spacing, decay


class CausalInputShaper:
    """Apply an impulse shaper to arbitrary timestamped two-axis commands.

    Input samples are treated as zero-order-held values. Before the first
    sample, the input is defined as zero. The caller supplies a monotonic time
    so this class is deterministic and easy to test without ROS.
    """

    def __init__(self, shaper: ImpulseShaper):
        self.shaper = shaper
        self._history: Deque[Tuple[float, Vector2]] = deque()
        self._last_time_s: float | None = None

    def reset(self) -> None:
        self._history.clear()
        self._last_time_s = None

    def shape(self, time_s: float, command: Sequence[float]) -> Vector2:
        if not math.isfinite(time_s):
            raise ValueError('time_s must be finite')
        if len(command) != 2:
            raise ValueError('command must contain exactly two axes')
        vector = (float(command[0]), float(command[1]))
        if not all(math.isfinite(value) for value in vector):
            raise ValueError('command must be finite')
        if self._last_time_s is not None and time_s < self._last_time_s:
            raise ValueError('timestamps must be monotonic')

        if self._history and time_s == self._history[-1][0]:
            self._history[-1] = (time_s, vector)
        else:
            self._history.append((time_s, vector))
        self._last_time_s = time_s

        shaped_x = 0.0
        shaped_y = 0.0
        for delay_s, amplitude in zip(self.shaper.delays_s, self.shaper.amplitudes):
            delayed = self._sample_at(time_s - delay_s)
            shaped_x += amplitude * delayed[0]
            shaped_y += amplitude * delayed[1]

        self._prune(time_s - self.shaper.horizon_s)
        return shaped_x, shaped_y

    def _sample_at(self, target_time_s: float) -> Vector2:
        for sample_time_s, value in reversed(self._history):
            if sample_time_s <= target_time_s + 1e-12:
                return value
        return 0.0, 0.0

    def _prune(self, oldest_needed_s: float) -> None:
        # Preserve the latest sample at or before the oldest delayed query.
        while len(self._history) > 1 and self._history[1][0] <= oldest_needed_s:
            self._history.popleft()


def apply_deadzone(value: float, deadzone: float) -> float:
    """Remove and rescale a symmetric joystick deadzone."""
    if not 0.0 <= deadzone < 1.0:
        raise ValueError('deadzone must be in [0, 1)')
    value = max(-1.0, min(1.0, float(value)))
    if abs(value) <= deadzone:
        return 0.0
    return math.copysign((abs(value) - deadzone) / (1.0 - deadzone), value)


class RateLimiter2D:
    """Vector-magnitude speed and acceleration limiter."""

    def __init__(self, max_speed: float, max_acceleration: float):
        if max_speed <= 0.0 or max_acceleration <= 0.0:
            raise ValueError('speed and acceleration limits must be positive')
        self.max_speed = float(max_speed)
        self.max_acceleration = float(max_acceleration)
        self._value: Vector2 = (0.0, 0.0)
        self._time_s: float | None = None

    def reset(self, time_s: float | None = None) -> None:
        self._value = (0.0, 0.0)
        self._time_s = time_s

    def limit(self, time_s: float, requested: Sequence[float]) -> Vector2:
        target = _limit_norm((float(requested[0]), float(requested[1])), self.max_speed)
        if self._time_s is None:
            self._time_s = time_s
            self._value = target
            return target
        dt = max(0.0, time_s - self._time_s)
        delta = (target[0] - self._value[0], target[1] - self._value[1])
        delta = _limit_norm(delta, self.max_acceleration * dt)
        self._value = (self._value[0] + delta[0], self._value[1] + delta[1])
        self._time_s = time_s
        return self._value


def _limit_norm(vector: Vector2, limit: float) -> Vector2:
    magnitude = math.hypot(*vector)
    if magnitude <= limit or magnitude == 0.0:
        return vector
    scale = limit / magnitude
    return vector[0] * scale, vector[1] * scale
