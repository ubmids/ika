"""Body features: the invariances, and whether the named quantities are honest.

The second half matters more than it sounds. It is easy to write a function
called `guard_height` and never notice it returns something else, and every
downstream claim about reading a fighter would inherit the mistake.
"""

import numpy as np
import pytest

from ika import bodysynth, posture
from ika.body import (LEFT_ELBOW, LEFT_WRIST, N_LANDMARKS, RIGHT_ELBOW,
                      RIGHT_WRIST)

CANON = slice(0, 99)
BASIS = slice(99, 108)
VIS = slice(108, 108 + N_LANDMARKS)


def d(features, name):
    return features[posture.DERIVED[name]]


def test_feature_vector_has_the_advertised_shape():
    f = posture.extract(bodysynth.posture("guard"))
    assert f.shape == (posture.FEATURE_DIM,)
    assert f.dtype == np.float32 and np.isfinite(f).all()


def test_wrong_landmark_count_is_refused():
    with pytest.raises(ValueError, match="expected 33"):
        posture.extract(np.zeros((21, 3)))


# --- invariance -----------------------------------------------------------

@pytest.mark.parametrize("name", sorted(bodysynth.POSTURES))
def test_where_the_body_stands_changes_nothing(name):
    body = bodysynth.posture(name)
    moved = bodysynth.place(body, translate=(4.0, -2.5, 1.5))
    assert np.abs(posture.extract(body) - posture.extract(moved)).max() < 1e-4


@pytest.mark.parametrize("scale", [0.3, 0.6, 2.0, 5.0])
def test_how_far_away_they_are_changes_nothing(scale):
    body = bodysynth.posture("jab_left")
    assert np.abs(
        posture.extract(body) - posture.extract(bodysynth.place(body, scale=scale))
    ).max() < 1e-4


def test_which_way_they_face_does_not_change_their_posture():
    """The canonical block is in the torso's own frame, so turning the whole
    body must leave it untouched. A fighter side-on is still in a guard."""
    body = bodysynth.posture("guard")
    turned = bodysynth.place(body, rotate=bodysynth.rotation(yaw=1.1, pitch=0.3))
    a, b = posture.extract(body), posture.extract(turned)
    assert np.abs(a[CANON] - b[CANON]).max() < 1e-4


def test_but_orientation_is_still_recorded():
    """Kept rather than discarded, for the same reason thumbs up and thumbs
    down had to stay separable in the hand lane."""
    body = bodysynth.posture("guard")
    turned = bodysynth.place(body, rotate=bodysynth.rotation(yaw=1.1))
    a, b = posture.extract(body), posture.extract(turned)
    assert np.abs(a[BASIS] - b[BASIS]).max() > 0.3


def test_small_jitter_moves_features_only_slightly():
    body = bodysynth.posture("guard")
    base = posture.extract(body)
    for seed in range(5):
        noisy = posture.extract(bodysynth.place(body, noise=0.01, seed=seed))
        assert np.abs(base - noisy).max() < 0.4


# --- do the named quantities mean what they say? -------------------------

def test_a_thrown_punch_straightens_that_arm_and_only_that_arm():
    guard = posture.extract(bodysynth.posture("guard"))
    jab = posture.extract(bodysynth.posture("jab_left"))
    cross = posture.extract(bodysynth.posture("cross_right"))

    # Cosine at the elbow: folded is positive, straight is negative.
    assert d(jab, "left_elbow_angle") < d(guard, "left_elbow_angle") - 0.5
    assert abs(d(jab, "right_elbow_angle") - d(guard, "right_elbow_angle")) < 0.2
    assert d(cross, "right_elbow_angle") < d(guard, "right_elbow_angle") - 0.5
    assert abs(d(cross, "left_elbow_angle") - d(guard, "left_elbow_angle")) < 0.2


def test_dropping_the_guard_shows_up_as_guard_height():
    """The single most useful read in the domain, and the thing FRIDAY spotted."""
    up = posture.extract(bodysynth.posture("guard"))
    down = posture.extract(bodysynth.posture("guard_down"))
    for side in ("left_guard_height", "right_guard_height"):
        assert d(down, side) < d(up, side) - 0.5


def test_guard_height_survives_a_leaning_fighter():
    """Measured along the spine, not the world vertical, so a hand is still
    "high" on someone slipping sideways."""
    upright = posture.extract(bodysynth.body_pose(guard=1.0))
    leaning = posture.extract(bodysynth.body_pose(guard=1.0, lean=0.5))
    assert abs(d(upright, "left_guard_height") - d(leaning, "left_guard_height")) < 0.25


def test_stance_width_tracks_the_feet():
    narrow = posture.extract(bodysynth.body_pose(stance=0.6))
    wide = posture.extract(bodysynth.body_pose(stance=2.0))
    assert d(wide, "stance_width") > d(narrow, "stance_width") * 1.5


def test_twist_separates_shoulders_from_hips():
    """The gap between the two is torque, which is where power comes from."""
    square = posture.extract(bodysynth.body_pose(twist=0.0))
    turned = posture.extract(bodysynth.body_pose(twist=0.6))
    assert abs(d(turned, "shoulder_twist")) > abs(d(square, "shoulder_twist")) + 0.3
    assert abs(d(turned, "hip_twist")) < 0.2, "hips should stay square"


def test_slipping_moves_the_head_off_the_centre_line():
    straight = posture.extract(bodysynth.posture("guard"))
    slipped = posture.extract(bodysynth.posture("slip_left"))
    assert abs(d(slipped, "head_offset")) > abs(d(straight, "head_offset")) + 0.1


def test_crouching_bends_the_knees():
    tall = posture.extract(bodysynth.body_pose(crouch=0.0))
    low = posture.extract(bodysynth.body_pose(crouch=1.0))
    assert d(low, "left_knee_angle") > d(tall, "left_knee_angle") + 0.1


def test_every_named_quantity_declares_its_dependencies():
    """So a caller can ask whether a specific read is trustworthy this frame."""
    assert set(posture.DEPENDS_ON) == set(posture.DERIVED)
    for name, joints in posture.DEPENDS_ON.items():
        assert joints and all(0 <= j < N_LANDMARKS for j in joints), name


# --- visibility, which has no hand-lane equivalent -----------------------

def test_visibility_travels_with_the_features():
    """Pose landmarkers guess at what they cannot see and return the guess.
    Passing confidence through lets a model discount it."""
    body = bodysynth.posture("guard")
    visibility = np.ones(N_LANDMARKS, dtype=np.float32)
    visibility[LEFT_WRIST] = 0.05
    f = posture.extract(body, visibility)
    assert f[VIS][LEFT_WRIST] == pytest.approx(0.05)
    assert f[VIS][RIGHT_WRIST] == pytest.approx(1.0)


def test_hidden_joints_are_not_zeroed():
    """Zeroing would place an unseen wrist at the hip centre, which is a
    specific wrong posture rather than an absent one."""
    body = bodysynth.posture("guard")
    visibility = np.ones(N_LANDMARKS, dtype=np.float32)
    visibility[LEFT_WRIST] = 0.0
    seen = posture.extract(body)
    unseen = posture.extract(body, visibility)
    assert np.allclose(seen[CANON], unseen[CANON]), "coordinates must be untouched"


def test_confidence_is_the_weakest_link():
    """A read is only as good as the least visible joint it depends on."""
    visibility = np.ones(N_LANDMARKS, dtype=np.float32)
    visibility[LEFT_ELBOW] = 0.2
    assert posture.confidence(visibility, posture.DEPENDS_ON["left_elbow_angle"]) == \
        pytest.approx(0.2)
    assert posture.confidence(visibility, posture.DEPENDS_ON["right_elbow_angle"]) == \
        pytest.approx(1.0)


def test_leg_reads_are_gated_by_visibility_not_assumed():
    """Measured on 400 real photographs at webcam framing: head, shoulders,
    elbows, wrists and hips were visible in 100% of detected bodies, knees in
    4%, ankles in 0%. So anything footwork-related has to be checked before it
    is believed, and the four reads that depend on legs are exactly the ones
    that go unusable."""
    from ika.body import LEFT_ANKLE, LEFT_KNEE, RIGHT_ANKLE, RIGHT_KNEE

    leg_reads = {"left_knee_angle", "right_knee_angle", "stance_width", "weight_shift"}
    for name in leg_reads:
        joints = set(posture.DEPENDS_ON[name])
        assert joints & {LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE}, name

    # Upper-body reads must not depend on legs, or they would be dragged down
    # with them at this framing.
    for name in set(posture.DERIVED) - leg_reads:
        joints = set(posture.DEPENDS_ON[name])
        assert not (joints & {LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE}), name

    # A body seen from the waist up should still yield trustworthy upper reads.
    visibility = np.ones(N_LANDMARKS, dtype=np.float32)
    for joint in (LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE):
        visibility[joint] = 0.05
    assert posture.confidence(visibility, posture.DEPENDS_ON["right_guard_height"]) > 0.5
    assert posture.confidence(visibility, posture.DEPENDS_ON["stance_width"]) < 0.5
