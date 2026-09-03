import math

import pytest

from zv_jog_shaper import make_nonrobust_zv_jog_shaper


def test_zero_damping_uses_two_equal_impulses_for_point_nine_meter_rope():
    shaper = make_nonrobust_zv_jog_shaper(rope_length_m=0.9)

    assert shaper.impulse_spacing_s == pytest.approx(
        math.pi / math.sqrt(9.80665 / 0.9)
    )
    assert (shaper.first_weight, shaper.second_weight) == pytest.approx((0.5, 0.5))


def test_long_jog_has_half_full_half_zero_sequence():
    shaper = make_nonrobust_zv_jog_shaper(
        rope_length_m=0.9,
        impulse_spacing_s=1.0,
    )
    release_s = 3.0

    assert shaper.gain_at(0.5, release_s) == pytest.approx(0.5)
    assert shaper.gain_at(1.5, release_s) == pytest.approx(1.0)
    assert shaper.gain_at(3.5, release_s) == pytest.approx(0.5)
    assert shaper.gain_at(4.0, release_s) == pytest.approx(0.0)
    assert shaper.is_complete(4.0, release_s)


def test_short_operator_pulse_is_convolved_exactly():
    shaper = make_nonrobust_zv_jog_shaper(
        rope_length_m=0.9,
        impulse_spacing_s=1.0,
    )
    release_s = 0.4

    assert shaper.gain_at(0.2, release_s) == pytest.approx(0.5)
    assert shaper.gain_at(0.6, release_s) == pytest.approx(0.0)
    assert shaper.gain_at(1.2, release_s) == pytest.approx(0.5)
    assert shaper.gain_at(1.4, release_s) == pytest.approx(0.0)
    assert shaper.is_complete(1.4, release_s)


def test_damped_weights_are_normalized():
    shaper = make_nonrobust_zv_jog_shaper(
        rope_length_m=0.9,
        damping_ratio=0.03,
    )

    assert shaper.first_weight + shaper.second_weight == pytest.approx(1.0)
    assert shaper.first_weight > shaper.second_weight


@pytest.mark.parametrize(
    'arguments',
    [
        {'rope_length_m': 0.0},
        {'rope_length_m': 0.9, 'damping_ratio': 1.0},
        {'rope_length_m': 0.9, 'impulse_spacing_s': -1.0},
    ],
)
def test_invalid_configuration_is_rejected(arguments):
    with pytest.raises(ValueError):
        make_nonrobust_zv_jog_shaper(**arguments)
