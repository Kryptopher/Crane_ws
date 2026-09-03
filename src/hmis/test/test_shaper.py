import cmath
import math

import pytest

from hmis.shaper import CausalInputShaper, ImpulseShaper, RateLimiter2D, apply_deadzone


def test_zvd_has_unity_gain_and_cancels_undamped_mode():
    frequency_hz = 0.5
    shaper = ImpulseShaper.zvd(frequency_hz, 0.0)
    response = sum(
        amplitude * cmath.exp(-1j * 2.0 * math.pi * frequency_hz * delay)
        for delay, amplitude in zip(shaper.delays_s, shaper.amplitudes)
    )
    assert sum(shaper.amplitudes) == pytest.approx(1.0)
    assert abs(response) < 1e-12


def test_zvd_shapes_arbitrary_two_axis_sequence():
    core = CausalInputShaper(ImpulseShaper.zvd(0.5, 0.0))
    assert core.shape(0.0, (8.0, 4.0)) == pytest.approx((2.0, 1.0))
    assert core.shape(0.5, (-4.0, 12.0)) == pytest.approx((-1.0, 3.0))
    # At one impulse spacing: 0.25 * current + 0.50 * the t=0 command.
    assert core.shape(1.0, (0.0, -8.0)) == pytest.approx((4.0, 0.0))
    # At the full horizon all three delayed, independently varying inputs contribute.
    assert core.shape(2.0, (0.0, 0.0)) == pytest.approx((2.0, -3.0))


def test_reset_discards_delayed_motion():
    core = CausalInputShaper(ImpulseShaper.zvd(0.5, 0.0))
    core.shape(0.0, (40.0, 0.0))
    core.reset()
    assert core.shape(0.2, (0.0, 0.0)) == (0.0, 0.0)


def test_timestamp_must_be_monotonic():
    core = CausalInputShaper(ImpulseShaper.zv(0.5, 0.0))
    core.shape(1.0, (0.0, 0.0))
    with pytest.raises(ValueError, match='monotonic'):
        core.shape(0.9, (0.0, 0.0))


def test_deadzone_is_rescaled():
    assert apply_deadzone(0.05, 0.08) == 0.0
    assert apply_deadzone(-0.08, 0.08) == 0.0
    assert apply_deadzone(1.0, 0.08) == pytest.approx(1.0)
    assert apply_deadzone(-0.54, 0.08) == pytest.approx(-0.5)


def test_rate_limiter_bounds_vector_acceleration_and_speed():
    limiter = RateLimiter2D(max_speed=10.0, max_acceleration=5.0)
    limiter.reset(0.0)
    assert limiter.limit(1.0, (30.0, 40.0)) == pytest.approx((3.0, 4.0))
    assert limiter.limit(2.0, (30.0, 40.0)) == pytest.approx((6.0, 8.0))
    assert math.hypot(*limiter.limit(3.0, (30.0, 40.0))) == pytest.approx(10.0)
