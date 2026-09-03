#!/usr/bin/env python3
"""Pure timing model for a nonrobust two-impulse ZV joystick filter."""

from __future__ import annotations

from dataclasses import dataclass
import math


GRAVITY_M_S2 = 9.80665


@dataclass(frozen=True)
class NonrobustZvJogShaper:
    """Convolve a finite operator velocity pulse with a two-impulse ZV shaper."""

    impulse_spacing_s: float
    first_weight: float
    second_weight: float

    def gain_at(
        self,
        elapsed_s: float,
        release_elapsed_s: float | None,
    ) -> float:
        """Return the filtered command gain after the operator starts a jog.

        ``release_elapsed_s=None`` means the operator is still holding the jog.
        The expression is the exact convolution of the raw rectangular jog
        command with impulses ``[A0 at 0, A1 at T]``.  It therefore also handles
        a release before the delayed impulse arrives.
        """
        elapsed = float(elapsed_s)
        if not math.isfinite(elapsed) or elapsed < 0.0:
            return 0.0
        release = None if release_elapsed_s is None else float(release_elapsed_s)
        if release is not None and (not math.isfinite(release) or release < 0.0):
            raise ValueError('release_elapsed_s must be finite and nonnegative')

        immediate_active = release is None or elapsed < release
        delayed_active = elapsed >= self.impulse_spacing_s and (
            release is None or elapsed < release + self.impulse_spacing_s
        )
        return (
            self.first_weight * float(immediate_active)
            + self.second_weight * float(delayed_active)
        )

    def is_complete(self, elapsed_s: float, release_elapsed_s: float | None) -> bool:
        """Whether all filtered motion caused by the released jog has ended."""
        if release_elapsed_s is None:
            return False
        return float(elapsed_s) >= (
            float(release_elapsed_s) + self.impulse_spacing_s
        )


def make_nonrobust_zv_jog_shaper(
    *,
    rope_length_m: float,
    damping_ratio: float = 0.0,
    impulse_spacing_s: float = 0.0,
    gravity_m_s2: float = GRAVITY_M_S2,
) -> NonrobustZvJogShaper:
    """Build the ordinary two-impulse ZV filter for the rope mode."""
    values = {
        'rope_length_m': rope_length_m,
        'damping_ratio': damping_ratio,
        'impulse_spacing_s': impulse_spacing_s,
        'gravity_m_s2': gravity_m_s2,
    }
    if any(not math.isfinite(float(value)) for value in values.values()):
        raise ValueError('ZV shaper arguments must be finite')
    if rope_length_m <= 0.0 or gravity_m_s2 <= 0.0:
        raise ValueError('rope length and gravity must be positive')
    if not 0.0 <= damping_ratio < 1.0:
        raise ValueError('damping ratio must be in [0, 1)')
    if impulse_spacing_s < 0.0:
        raise ValueError('impulse spacing must be nonnegative')

    root = math.sqrt(max(1.0 - damping_ratio * damping_ratio, 1.0e-12))
    spacing = (
        float(impulse_spacing_s)
        if impulse_spacing_s > 0.0
        else math.pi / (math.sqrt(gravity_m_s2 / rope_length_m) * root)
    )
    decay = math.exp(-math.pi * damping_ratio / root)
    denominator = 1.0 + decay
    return NonrobustZvJogShaper(
        impulse_spacing_s=spacing,
        first_weight=1.0 / denominator,
        second_weight=decay / denominator,
    )
