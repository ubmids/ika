"""Motion features and the rolling window.

The central claim here is that the dynamic lane needs the very thing the
static lane throws away. These tests pin that down: translation must matter,
and scale must not.
"""

import numpy as np
import pytest

from ika import features, synth, trajectory
from ika.motion import (MOTION_DIM, NET_X, NET_Y, PEAK_SPEED, PINCH_MIN,
                        POSE_DRIFT, STRAIGHTNESS, MotionWindow, Sample)
from ika.motion import features as motion_features
from ika.schema import DYNAMIC_GESTURES

FPS = 30.0


def window_of(name, seed=0):
    return trajectory.make(name, fps=FPS, seed=seed)


def test_feature_vector_has_the_advertised_shape():
    v = motion_features(window_of("swipe_left"))
    assert v.shape == (MOTION_DIM,) and np.isfinite(v).all()


def test_translation_is_kept_not_discarded():
    """The opposite of the static lane, and the whole point of this module.
    A swipe *is* translation, so erasing it would erase the gesture."""
    left = motion_features(window_of("swipe_left"))
    right = motion_features(window_of("swipe_right"))
    assert left[NET_X] < -0.5 and right[NET_X] > 0.5


def test_scale_is_discarded():
    """Distances are in hand widths, so the same swipe close to the camera and
    far from it must read the same."""
    near = trajectory.make("swipe_right", seed=3)
    far = [
        Sample(s.at, s.position * 0.5 + 0.25, s.span * 0.5, s.pinch, s.pose)
        for s in near
    ]
    a, b = motion_features(near), motion_features(far)
    assert abs(a[NET_X] - b[NET_X]) < 0.25 * abs(a[NET_X]), "net displacement should be scale free"


@pytest.mark.parametrize("name", ["swipe_left", "swipe_right", "swipe_up", "swipe_down"])
def test_swipes_are_straight_and_idling_is_not(name):
    """Straightness is what separates a deliberate stroke from a fidget that
    happens to end up somewhere else."""
    assert motion_features(window_of(name))[STRAIGHTNESS] > 0.9
    assert motion_features(window_of("none"))[STRAIGHTNESS] < 0.6


def test_a_drag_is_slower_than_a_swipe_along_the_same_path():
    """They trace similar shapes, so speed and pose have to do the separating."""
    swipe = motion_features(window_of("swipe_right", seed=5))
    drag = motion_features(window_of("pinch_drag", seed=5))
    assert swipe[PEAK_SPEED] > drag[PEAK_SPEED] * 1.8, "peak speed should separate them"
    assert drag[PINCH_MIN] < 0.3 < swipe[PINCH_MIN], "and the pinch should too"


def test_a_snap_barely_moves_but_changes_shape():
    """The mirror image of a swipe, which is why pose drift is a feature."""
    snap = motion_features(window_of("snap"))
    swipe = motion_features(window_of("swipe_left"))
    assert abs(snap[NET_X]) < 0.3 and abs(snap[NET_Y]) < 0.3
    assert snap[POSE_DRIFT] > swipe[POSE_DRIFT] * 3, "pose drift should be much larger"


def test_the_window_forgets_old_frames():
    """It runs indefinitely, so history has to be bounded by time."""
    w = MotionWindow(seconds=0.5)
    pose = features.extract(synth.pose("open_palm"))
    for i in range(200):
        w.push(Sample(i / FPS, np.array([0.5, 0.5]), 0.16, 1.0, pose))
    assert len(w) <= int(0.5 * FPS) + 2


def test_the_window_says_when_it_cannot_answer():
    """Better to report nothing than to classify two frames of history.

    The bar is deliberately low. It used to be 60% of the window length, which
    meant 13 frames of history for a gesture only 14 frames long, so the window
    would not look at a swipe until it was over.
    """
    w = MotionWindow(seconds=0.6)
    pose = features.extract(synth.pose("open_palm"))
    assert w.features() is None
    for i in range(3):
        w.push(Sample(i / FPS, np.array([0.5, 0.5]), 0.16, 1.0, pose))
    assert not w.ready and w.features() is None
    for i in range(3, 30):
        w.push(Sample(i / FPS, np.array([0.5, 0.5]), 0.16, 1.0, pose))
    assert w.ready and w.features() is not None


def test_clearing_the_window_resets_it():
    w = MotionWindow(seconds=0.6)
    pose = features.extract(synth.pose("fist"))
    for i in range(30):
        w.push(Sample(i / FPS, np.array([0.5, 0.5]), 0.16, 1.0, pose))
    w.clear()
    assert len(w) == 0 and w.features() is None


@pytest.mark.parametrize("name", DYNAMIC_GESTURES)
def test_every_dynamic_gesture_can_be_generated(name):
    window = trajectory.make(name, fps=FPS, seed=1)
    assert len(window) >= 6
    assert np.isfinite(motion_features(window)).all()


def test_the_features_hold_no_window_artifacts():
    """Window duration and frame count were features once. Because every
    training window was the same length, the model leaned on them, and a
    shorter window at inference read as out-of-distribution and came back
    "none" with total confidence. The vector must describe the gesture only."""
    long_window = trajectory.make("swipe_left", seed=8)
    clipped = long_window[: max(8, len(long_window) - 2)]
    a, b = motion_features(long_window), motion_features(clipped)
    assert len(a) == MOTION_DIM
    # Nothing in here may encode how many frames arrived.
    assert not any(abs(v - len(long_window)) < 0.01 for v in a)
    assert np.sign(a[NET_X]) == np.sign(b[NET_X])


def test_a_sliding_window_labels_fragments_honestly():
    """A window holding the tail of a swipe is closer to nothing than to a
    swipe, and labelling it otherwise teaches the model to fire on fragments."""
    kept = sum(
        trajectory.make_in_context("swipe_left", fps=FPS, seed=s)[1] for s in range(60)
    )
    assert 0 < kept < 60, f"{kept}/60 kept: the overlap rule is doing nothing"
