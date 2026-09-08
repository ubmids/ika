"""Relational features, which is where a strike is suspected to live.

Reading one body gave 82% on gross posture and 43% on strikes, and the
hypothesis is that a punch is not a shape but a relationship: one person's
wrist travelling toward another person's body. These tests check the features
actually measure that, using pairs whose geometry is imposed by hand.
"""

import numpy as np
import pytest

from ika.interaction import (FEATURE_DIM, INDEX, NAMES, Pair, features,
                             pair_from_frame)


def pair_at(gap, a_left=(0.0, 0.0), a_right=(0.0, 0.0), at=0.0, scale=0.2):
    """Two people `gap` torso-lengths apart, with A's wrists placed by offset
    from A's own centre."""
    a = np.array([0.4, 0.5])
    b = np.array([0.4 + gap * scale, 0.5])
    return Pair(
        at=at, a_centre=a, b_centre=b,
        a_wrists=np.stack([a + np.array(a_left) * scale,
                           a + np.array(a_right) * scale]),
        b_wrists=np.stack([b, b]),
        a_scale=scale, b_scale=scale,
    )


def f(window, name):
    return features(window)[INDEX[name]]


def test_feature_vector_has_the_advertised_shape():
    window = [pair_at(3.0, at=0.0), pair_at(3.0, at=1 / 30)]
    v = features(window)
    assert v.shape == (FEATURE_DIM,) == (len(NAMES),)
    assert np.isfinite(v).all()


def test_two_frames_are_the_minimum():
    with pytest.raises(ValueError, match="at least two"):
        features([pair_at(3.0)])


def test_closing_the_distance_reads_as_closing():
    approach = [pair_at(4.0 - i * 0.3, at=i / 30) for i in range(8)]
    retreat = [pair_at(1.9 + i * 0.3, at=i / 30) for i in range(8)]
    assert f(approach, "gap_delta") < -1.0
    assert f(retreat, "gap_delta") > 1.0
    assert f(approach, "closing_peak") > 0
    assert f(retreat, "closing_peak") < f(approach, "closing_peak")


def test_a_punch_is_a_wrist_travelling_toward_the_other_person():
    """The single most important feature here. A jab from A's left hand should
    show up as that specific wrist closing on B, and no other."""
    still = [pair_at(3.0, a_left=(0.2, 0.0), at=i / 30) for i in range(8)]
    jab = [pair_at(3.0, a_left=(0.2 + i * 0.35, 0.0), at=i / 30) for i in range(8)]

    assert f(jab, "a_left_reach_rate") > f(still, "a_left_reach_rate") + 1.0
    assert f(jab, "a_left_reach_min") < f(still, "a_left_reach_min") - 1.0
    # and it must not smear onto the other three wrists
    for other in ("a_right_reach_rate", "b_left_reach_rate", "b_right_reach_rate"):
        assert abs(f(jab, other) - f(still, other)) < 0.5, other


def test_which_hand_threw_it_is_distinguishable():
    """A strike is sided, and collapsing that would merge a jab with a cross."""
    left = [pair_at(3.0, a_left=(i * 0.35, 0.0), at=i / 30) for i in range(8)]
    right = [pair_at(3.0, a_right=(i * 0.35, 0.0), at=i / 30) for i in range(8)]
    assert f(left, "a_left_reach_rate") > f(left, "a_right_reach_rate") + 1.0
    assert f(right, "a_right_reach_rate") > f(right, "a_left_reach_rate") + 1.0


def test_who_threw_it_is_distinguishable():
    """Ordered by screen position rather than track id, because track ids are
    assigned by arrival order and carry no directional meaning."""
    window = [pair_at(3.0, a_left=(i * 0.35, 0.0), at=i / 30) for i in range(8)]
    assert f(window, "a_left_reach_rate") > f(window, "b_left_reach_rate") + 1.0


def test_a_clinch_shows_as_overlap():
    apart = [pair_at(4.0, at=i / 30) for i in range(8)]
    close = [pair_at(0.4, at=i / 30) for i in range(8)]
    assert f(apart, "overlap_share") == 0.0
    assert f(close, "overlap_share") == 1.0


def test_distance_is_scaled_so_it_survives_the_pair_moving_away():
    """The same punch at two camera distances must read the same, or every
    number becomes a function of where the pair happened to stand."""
    near = [pair_at(3.0, a_left=(i * 0.35, 0.0), at=i / 30, scale=0.30)
            for i in range(8)]
    far = [pair_at(3.0, a_left=(i * 0.35, 0.0), at=i / 30, scale=0.10)
           for i in range(8)]
    assert abs(f(near, "a_left_reach_rate") - f(far, "a_left_reach_rate")) < 0.5
    assert abs(f(near, "gap_end") - f(far, "gap_end")) < 0.2


def test_a_lone_body_is_an_absence_not_a_degraded_pair():
    """Filling the gap with a guess would invent a relationship."""
    from ika.body import Body, N_LANDMARKS

    def body(x):
        marks = np.zeros((N_LANDMARKS, 3), dtype=np.float32)
        marks[:, 0], marks[:, 1] = x, 0.5
        return Body(image=marks, world=marks.copy(),
                    visibility=np.ones(N_LANDMARKS, dtype=np.float32))

    assert pair_from_frame(0.0, []) is None
    assert pair_from_frame(0.0, [body(0.3)]) is None
    assert pair_from_frame(0.0, [body(0.3), body(0.7)]) is not None


def test_pairs_are_ordered_left_to_right_on_screen():
    from ika.body import Body, N_LANDMARKS
    from ika.body import LEFT_HIP, LEFT_SHOULDER, RIGHT_HIP, RIGHT_SHOULDER

    def body(x):
        marks = np.zeros((N_LANDMARKS, 3), dtype=np.float32)
        marks[:, 0], marks[:, 1] = x, 0.5
        marks[LEFT_HIP] = (x, 0.6, 0); marks[RIGHT_HIP] = (x, 0.6, 0)
        marks[LEFT_SHOULDER] = (x, 0.4, 0); marks[RIGHT_SHOULDER] = (x, 0.4, 0)
        return Body(image=marks, world=marks.copy(),
                    visibility=np.ones(N_LANDMARKS, dtype=np.float32))

    for order in ([body(0.2), body(0.8)], [body(0.8), body(0.2)]):
        pair = pair_from_frame(0.0, order)
        assert pair.a_centre[0] < pair.b_centre[0], "a must be the left person"
