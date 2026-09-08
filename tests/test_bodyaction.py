"""Body movement: the features, the actions, and what occlusion really costs.

The most important test here is the one about occlusion, because the first two
attempts at simulating it were both wrong in ways that made the results a lie.
Version one shortened the spine when crouching, which `posture` normalises
against, so every limb read as proportionally longer and handed the classifier
a free crouch cue. Version two only lowered the visibility flag while still
passing the true leg coordinates, so the model happily read knees it was told
it could not see. Both scored ~86% on an action a webcam physically cannot
observe.
"""

import numpy as np
import pytest

from ika import bodyaction, bodysynth, posture
from ika.body import (LEFT_ANKLE, LEFT_KNEE, N_LANDMARKS, RIGHT_ANKLE,
                      RIGHT_KNEE)
from ika.bodymotion import (FEATURE_DIM, BodyWindow, features, index_of)


def rows_for(name, count=25, **kw):
    return np.stack([features(bodyaction.make(name, seed=s, **kw)) for s in range(count)])


# --- features -------------------------------------------------------------

def test_feature_vector_has_the_advertised_shape():
    f = features(bodyaction.make("jab_left", seed=0))
    assert f.shape == (FEATURE_DIM,) and np.isfinite(f).all()


def test_two_frames_are_the_minimum():
    window = bodyaction.make("jab_left", seed=0)
    with pytest.raises(ValueError, match="at least two"):
        features(window[:1])


def test_nothing_encodes_the_window_length():
    """Duration and frame count were features in the hand lane, the model
    leaned on them, and any other window length then read as out of
    distribution and came back confidently wrong."""
    window = bodyaction.make("cross_right", seed=3)
    short = features(window[: max(2, len(window) - 2)])
    full = features(window)
    assert not any(abs(v - len(window)) < 0.01 for v in full)
    assert np.sign(short[index_of("right_elbow_angle")]) == \
        np.sign(full[index_of("right_elbow_angle")])


def test_a_punch_is_the_punching_arm_straightening():
    """And only that arm, which is what makes the two strikes separable."""
    guard, jab, cross = rows_for("idle"), rows_for("jab_left"), rows_for("cross_right")
    left = index_of("left_elbow_angle", "peak")
    right = index_of("right_elbow_angle", "peak")

    assert jab[:, left].mean() > guard[:, left].mean() * 5
    assert jab[:, left].mean() > jab[:, right].mean() * 5
    assert cross[:, right].mean() > cross[:, left].mean() * 5


def test_dropping_and_raising_the_guard_differ_by_sign():
    down = rows_for("drop_guard")[:, index_of("right_guard_height")].mean()
    up = rows_for("raise_guard")[:, index_of("right_guard_height")].mean()
    assert down < -0.5 < 0.5 < up


def test_a_slip_moves_the_head_and_little_else():
    slip = rows_for("slip_left")
    assert abs(slip[:, index_of("head_offset")].mean()) > 0.1
    assert slip[:, index_of("left_elbow_angle", "peak")].mean() < 2.0


def test_idle_is_quiet_on_everything():
    idle = rows_for("idle")
    for name in ("left_elbow_angle", "right_elbow_angle"):
        assert idle[:, index_of(name, "peak")].mean() < 2.0
    for name in ("right_guard_height", "head_offset", "left_knee_angle"):
        assert abs(idle[:, index_of(name)].mean()) < 0.15


# --- occlusion, and what it costs ----------------------------------------

def test_occlusion_corrupts_coordinates_not_just_confidence():
    """A landmarker does not withhold an unseen joint, it extrapolates a
    plausible wrong one. Simulating occlusion as a flag alone let the model
    read knees it was told were invisible."""
    visible = bodyaction.make("crouch", seed=1, occlude_legs=False)
    hidden = bodyaction.make("crouch", seed=1, occlude_legs=True)
    canon_v = visible[-1].features[:99].reshape(33, 3)
    canon_h = hidden[-1].features[:99].reshape(33, 3)
    # Legs must actually differ.
    assert np.abs(canon_v[LEFT_KNEE] - canon_h[LEFT_KNEE]).max() > 0.05
    # And confidence must reflect it.
    assert hidden[-1].visibility[LEFT_ANKLE] < 0.2
    assert hidden[-1].visibility[RIGHT_KNEE] < 0.2


def test_the_knee_signal_vanishes_when_the_legs_do():
    """The measurement that makes the occlusion honest: with legs hidden,
    crouch and idle are indistinguishable on knee angle."""
    knee = index_of("left_knee_angle")
    for occlude, expect_signal in ((False, True), (True, False)):
        idle = rows_for("idle", occlude_legs=occlude)[:, knee]
        crouch = rows_for("crouch", occlude_legs=occlude)[:, knee]
        separation = abs(crouch.mean() - idle.mean()) / (idle.std() + 1e-9)
        if expect_signal:
            assert separation > 10, f"visible legs gave only {separation:.1f} sigma"
        else:
            assert separation < 3, f"hidden legs still gave {separation:.1f} sigma"


def test_upper_body_reads_survive_occlusion():
    """The other half of the finding: punches and guard are unaffected."""
    left = index_of("left_elbow_angle", "peak")
    hidden = rows_for("jab_left", occlude_legs=True)[:, left].mean()
    visible = rows_for("jab_left", occlude_legs=False)[:, left].mean()
    assert abs(hidden - visible) / visible < 0.25


# --- feints and windowing -------------------------------------------------

def test_a_feint_lies_about_its_opening():
    window = bodyaction.feint("jab_left", "cross_right", switch=0.5, seed=0)
    right = index_of("right_elbow_angle", "peak")
    left = index_of("left_elbow_angle", "peak")
    early = features(window[: max(2, int(len(window) * 0.4))])
    whole = features(window)
    assert early[left] > early[right], "should look like a jab early"
    assert whole[right] > whole[left], "and end up a cross"


def test_a_feint_is_labelled_by_what_it_becomes():
    windows, labels, classes = bodyaction.dataset(per_class=5, feint_rate=1.0, seed=0)
    assert len(labels) == 5 * len(classes) + int(5 * len(classes))


def test_the_window_forgets_old_frames():
    """It runs for as long as the camera does, so history has to be a ring.

    Timestamps have to advance for this to mean anything: an earlier version
    pushed the same frames repeatedly with their original times, so window
    time never moved and nothing was ever evicted.
    """
    from ika.bodymotion import Frame

    window = BodyWindow(seconds=0.4)
    source = bodyaction.make("idle", seed=0)
    for i in range(300):
        f = source[i % len(source)]
        window.push(Frame(i / 30.0, f.features, f.wrists, f.scale, f.visibility))
    assert len(window) <= int(0.4 * 30) + 2


def test_the_window_answers_while_the_movement_is_happening():
    """The hand lane set this bar at 60% of the window, which meant it refused
    to look at a swipe until the swipe was over."""
    window = BodyWindow(seconds=0.6)
    frames = bodyaction.make("jab_left", seed=0)
    ready = 0
    for frame in frames:
        window.push(frame)
        if window.ready:
            ready += 1
    assert ready >= len(frames) // 2, f"ready for only {ready} of {len(frames)} frames"


def test_index_of_refuses_an_unknown_quantity():
    with pytest.raises(KeyError):
        index_of("not_a_read")
