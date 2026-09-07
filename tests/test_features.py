"""The invariance claims in `features`, checked rather than assumed.

If these pass, a gesture means the same thing whether your hand is near or far,
left or right of frame, or turned over. If they fail, the model will only ever
work sitting exactly where you sat while recording, and no amount of training
data fixes that.
"""

import numpy as np
import pytest

from ika import features, synth
from ika.schema import THUMB_TIP, INDEX_TIP

CANON = slice(0, 63)
BASIS = slice(63, 72)


def test_feature_vector_has_the_advertised_shape():
    f = features.extract(synth.pose("point"))
    assert f.shape == (features.FEATURE_DIM,)
    assert f.dtype == np.float32
    assert np.isfinite(f).all()


@pytest.mark.parametrize("name", sorted(synth.POSES))
def test_moving_the_hand_changes_nothing(name):
    """Translation invariance. The wrist is the origin, so where the hand sits
    in frame is not information the model should ever see."""
    hand = synth.pose(name)
    moved = synth.place(hand, translate=(3.5, -2.0, 0.8))
    assert np.abs(features.extract(hand) - features.extract(moved)).max() < 1e-5


@pytest.mark.parametrize("scale", [0.25, 0.5, 2.0, 7.0])
def test_hand_size_changes_nothing(scale):
    """Scale invariance, which is what makes leaning toward the camera safe."""
    hand = synth.pose("peace")
    resized = synth.place(hand, scale=scale)
    assert np.abs(features.extract(hand) - features.extract(resized)).max() < 1e-5


def test_rotating_the_hand_preserves_its_shape():
    """The canonical block is expressed in the palm's own frame, so turning the
    whole hand must leave it untouched."""
    hand = synth.pose("point")
    turned = synth.place(hand, rotate=synth.rotation(yaw=0.9, pitch=-0.4, roll=1.2))
    a, b = features.extract(hand), features.extract(turned)
    assert np.abs(a[CANON] - b[CANON]).max() < 1e-5


def test_rotating_the_hand_does_change_its_orientation_block():
    """The other half of the same design: orientation is kept, not discarded,
    so gestures that differ only by which way up they are stay separable."""
    hand = synth.pose("thumbs_up")
    turned = synth.place(hand, rotate=synth.rotation(pitch=np.pi))
    a, b = features.extract(hand), features.extract(turned)
    assert np.abs(a[BASIS] - b[BASIS]).max() > 0.5


def test_thumbs_up_and_thumbs_down_are_distinguishable():
    """The gesture that would be destroyed by full rotation invariance. Same
    hand shape, opposite orientation, and the features must tell them apart."""
    up = synth.pose("thumbs_up")
    down = synth.place(up, rotate=synth.rotation(roll=np.pi))
    a, b = features.extract(up), features.extract(down)
    assert np.abs(a[CANON] - b[CANON]).max() < 1e-5, "shape should be identical"
    assert np.linalg.norm(a - b) > 1.0, "yet the vectors must differ overall"


def test_a_mirrored_left_hand_matches_the_right():
    """Halves the data needed: one model covers both hands."""
    right = synth.pose("point")
    left = right.copy()
    left[:, 0] *= -1.0
    assert np.abs(
        features.extract(right, is_left=False) - features.extract(left, is_left=True)
    ).max() < 1e-5


@pytest.mark.parametrize("name", sorted(synth.POSES))
def test_poses_are_separable_from_each_other(name):
    """Every pose must sit further from all others than from a jittered copy of
    itself, or no classifier can succeed regardless of architecture."""
    own = features.extract(synth.pose(name))
    jittered = features.extract(synth.place(synth.pose(name), noise=0.01, seed=1))
    self_distance = np.linalg.norm(own - jittered)

    others = [
        np.linalg.norm(own - features.extract(synth.pose(other)))
        for other in synth.POSES
        if other != name
    ]
    assert self_distance < min(others), (
        f"{name}: jitter moved it {self_distance:.3f}, "
        f"nearest other pose is {min(others):.3f} away"
    )


def test_small_jitter_moves_features_only_slightly():
    """Landmark estimates are noisy in real video, so the mapping must be
    stable rather than merely correct."""
    hand = synth.pose("open_palm")
    base = features.extract(hand)
    for seed in range(5):
        noisy = features.extract(synth.place(hand, noise=0.005, seed=seed))
        assert np.abs(base - noisy).max() < 0.35


def test_pinch_distance_is_small_only_when_pinching():
    assert features.pinch_distance(synth.pose("pinch")) < 0.2
    for name in ("fist", "open_palm", "point", "peace"):
        assert features.pinch_distance(synth.pose(name)) > 0.5


def test_degenerate_input_does_not_explode():
    """A collapsed detection should give finite numbers, not NaN poisoning the
    model and every downstream average."""
    assert np.isfinite(features.extract(np.zeros((21, 3)))).all()
    assert np.isfinite(features.extract(np.ones((21, 3)))).all()
