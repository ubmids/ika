"""The pose lane. Shipped before it had any tests, which is what this fixes.

The property that matters most here is the one that has no equivalent in the
hand lane: pose landmarkers always return all 33 points, guessing at the ones
they cannot see. A limb behind a torso comes back as a confident-looking
number that is fiction, so anything reading a specific joint has to consult
`visibility` first, and these tests pin that contract down.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from ika import body
from ika.hands import ModelMissing

IMAGE = Path("/private/tmp/claude-501/-Users-subomi/73442120-54a1-4fc0-8623-460baf461692"
             "/scratchpad/hands.jpg")
MODEL = body.MODELS / "pose_landmarker_lite.task"

needs_model = pytest.mark.skipif(not MODEL.exists(), reason="pose model not downloaded")
needs_image = pytest.mark.skipif(not IMAGE.exists(), reason="no test photograph")


def test_the_landmark_names_are_complete_and_unique():
    """Named so nothing downstream indexes a magic number, which is exactly
    the fragility that bit the motion features."""
    named = {
        v for k, v in vars(body).items()
        if isinstance(v, int) and k.isupper() and k != "N_LANDMARKS"
    }
    assert body.N_LANDMARKS == 33
    assert named == set(range(33)), sorted(set(range(33)) - named)


def test_the_skeleton_only_connects_real_landmarks():
    for a, b in body.LIMBS:
        assert 0 <= a < body.N_LANDMARKS and 0 <= b < body.N_LANDMARKS
        assert a != b


def test_the_torso_is_the_rigid_reference():
    """Shoulders and hips do not move relative to each other when a limb
    does, which is why a canonical frame is built from them, the same reason
    the hand lane uses the knuckle row."""
    assert set(body.TORSO) == {
        body.LEFT_SHOULDER, body.RIGHT_SHOULDER, body.LEFT_HIP, body.RIGHT_HIP
    }


def test_extremities_are_sided():
    """Which hand threw the punch is the whole point, so these cannot be
    collapsed into an unsided set."""
    assert body.LEFT_WRIST != body.RIGHT_WRIST
    assert body.LEFT_ANKLE != body.RIGHT_ANKLE
    assert len(set(body.EXTREMITIES)) == 4


def test_visibility_gates_a_landmark():
    """The contract that has no hand-lane equivalent."""
    marks = np.zeros((33, 3), dtype=np.float32)
    visibility = np.zeros(33, dtype=np.float32)
    visibility[body.LEFT_WRIST] = 0.9
    visibility[body.RIGHT_WRIST] = 0.1
    subject = body.Body(image=marks, world=marks, visibility=visibility)
    assert subject.seen(body.LEFT_WRIST)
    assert not subject.seen(body.RIGHT_WRIST)
    assert not subject.seen(body.LEFT_ANKLE), "unseen points must not read as visible"


def test_a_missing_model_names_the_fix():
    """A fresh clone has no weights, and a bare FileNotFoundError is a puzzle."""
    with pytest.raises(ModelMissing) as caught:
        body.PoseTracker(model_path="/nowhere/pose.task")
    message = str(caught.value)
    assert "curl" in message and "pose_landmarker" in message


def test_every_variant_has_a_download_url():
    for variant in ("lite", "full"):
        assert body.POSE_URLS[variant].startswith("https://")
        assert variant in body.POSE_URLS[variant]


@needs_model
@needs_image
def test_it_finds_a_real_person_and_returns_both_coordinate_systems():
    rgb = cv2.cvtColor(cv2.imread(str(IMAGE)), cv2.COLOR_BGR2RGB)
    with body.PoseTracker(variant="lite") as tracker:
        bodies = tracker(rgb, 0)
    assert bodies, "no body found in a photograph of a person"
    subject = bodies[0]
    assert subject.image.shape == (33, 3) and subject.world.shape == (33, 3)
    assert subject.visibility.shape == (33,)
    # Image coordinates are frame-normalised; world coordinates are metric and
    # hip-centred, so they must not be the same numbers.
    assert 0.0 <= subject.image[:, :2].min() and subject.image[:, :2].max() <= 1.2
    assert not np.allclose(subject.image, subject.world)


@needs_model
@needs_image
def test_some_landmarks_are_honestly_reported_as_unseen():
    """On a close crop most of the legs are not in shot. If everything came
    back visible, the visibility field would be worthless."""
    rgb = cv2.cvtColor(cv2.imread(str(IMAGE)), cv2.COLOR_BGR2RGB)
    with body.PoseTracker(variant="lite") as tracker:
        subject = tracker(rgb, 0)[0]
    seen = int((subject.visibility >= 0.5).sum())
    assert 5 < seen < 33, f"{seen}/33 visible, which is suspiciously all or nothing"


@needs_model
@needs_image
def test_it_is_fast_enough_to_leave_budget_for_everything_else():
    """The whole point of the pose lane being viable: 263ms is the entire
    latency budget, so seeing must be cheap."""
    import time

    rgb = cv2.cvtColor(cv2.imread(str(IMAGE)), cv2.COLOR_BGR2RGB)
    frame = cv2.resize(rgb, (640, 480))
    with body.PoseTracker(variant="lite") as tracker:
        tracker(frame, 0)
        start = time.time()
        for i in range(30):
            tracker(frame, 100 + i * 30)
        per_frame = (time.time() - start) / 30
    assert per_frame < 0.033, f"{per_frame*1000:.0f} ms/frame is too slow for 30 fps"


@needs_model
def test_the_tracker_closes_cleanly():
    tracker = body.PoseTracker(variant="lite")
    tracker.close()
