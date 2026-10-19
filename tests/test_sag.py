import numpy as np
import pytest

from ika import sag
from ika.body import (LEFT_HIP, LEFT_SHOULDER, LEFT_WRIST, NOSE, RIGHT_HIP,
                      RIGHT_SHOULDER, RIGHT_WRIST, Body)


def _body(nose_y=0.2, wrist_y=0.3):
    image = np.full((33, 3), 0.5)
    image[NOSE] = (0.5, nose_y, 0)
    image[LEFT_SHOULDER] = (0.6, 0.35, 0)
    image[RIGHT_SHOULDER] = (0.4, 0.35, 0)
    image[LEFT_HIP] = (0.58, 0.75, 0)
    image[RIGHT_HIP] = (0.42, 0.75, 0)
    image[LEFT_WRIST] = (0.6, wrist_y, 0)
    image[RIGHT_WRIST] = (0.4, wrist_y, 0)
    return Body(image, np.zeros((33, 3)), np.ones(33))


def _heights(reference, bodies):
    track = sag.GuardTrack(reference=reference)
    for i, body in enumerate(bodies):
        track.update(body, i / 30.0)
    return track.snapshot()[1]["left"]


def test_the_guard_is_measured_from_the_shoulders_so_a_head_movement_is_not_a_drop():
    # Forty frames of a steady guard, then the head dips (a slip, a bob)
    # while the hands stay where they were relative to the shoulders.
    steady = [_body() for _ in range(40)]
    bob = [_body(nose_y=0.3) for _ in range(10)]
    shoulders = _heights("shoulders", steady + bob)
    nose = _heights("nose", steady + bob)
    assert abs(shoulders[-1] - shoulders[39]) < 1e-9
    assert nose[-1] > nose[39] + 0.1        # the old reference reads the bob as the hand rising


def test_a_hand_that_drops_reads_lower_from_either_reference():
    frames = [_body() for _ in range(40)] + [_body(wrist_y=0.5) for _ in range(5)]
    for reference in sag.REFERENCES:
        h = _heights(reference, frames)
        assert h[-1] < h[39] - 0.3


def test_unknown_reference_and_rule_are_refused():
    with pytest.raises(ValueError):
        sag.GuardTrack(reference="ear")
    with pytest.raises(ValueError):
        sag.read_occasions([], rule="vibes")


def test_effect_is_the_share_of_pairs_where_the_setup_sits_lower():
    assert sag.effect([0, 1, 2], [3, 4, 5]) == 1.0
    assert sag.effect([3, 4, 5], [0, 1, 2]) == 0.0
    assert sag.effect([1, 2], [1, 2]) == 0.5
    assert sag.effect([1], [1]) == 0.5               # a tie counts half
    assert sag.effect([0, 2], [1]) == 0.5
    assert sag.effect([], [1]) == 0.5


def _occasions(rng, inside_mean, rest_mean, sd, n):
    occ = []
    for _ in range(n):
        occ.append(sag.Occasion(("punch_right",), "left", rng.normal(inside_mean, sd)))
        occ.append(sag.Occasion(("punch_left",), "left", rng.normal(rest_mean, sd)))
        occ.append(sag.Occasion(("punch_right",), "right", rng.normal(rest_mean, sd)))
        occ.append(sag.Occasion(("punch_left",), "right", rng.normal(rest_mean, sd)))
    return occ


def test_a_tiny_difference_is_not_a_habit_however_sure_the_test_is():
    occ = _occasions(np.random.default_rng(0), -0.12, -0.10, 0.19, 6000)
    inside = [o.height for o in occ if o.context == ("punch_right",) and o.side == "left"]
    rest = [o.height for o in occ if o.context != ("punch_right",) and o.side == "left"]
    assert sag.lower_than(inside, rest) < 1e-6
    assert sag.effect(inside, rest) < sag.MIN_EFFECT
    assert not sag.read_occasions(occ, rule="effect")


def test_a_shallow_consistent_sag_the_lift_bar_missed_is_named_by_its_size():
    # Both setups mostly sit below the "down" line, so counting hands below
    # it gives a lift under 1.35; the heights still differ by half a
    # standard deviation, consistently.
    occ = _occasions(np.random.default_rng(1), -0.40, -0.30, 0.19, 300)
    found = sag.read_occasions(occ, rule="effect")
    assert [(f.context, f.then) for f in found] == [(("punch_right",), "guard_down_left")]
    assert found[0].lift < sag.MIN_LIFT
    assert not sag.read_occasions(occ, rule="lift")


def test_the_shipped_rule_is_the_effect_size():
    assert sag.RULE == "effect" and sag.REFERENCE == "shoulders"
    occ = _occasions(np.random.default_rng(1), -0.40, -0.30, 0.19, 300)
    assert sag.read_occasions(occ) == sag.read_occasions(occ, rule="effect")
