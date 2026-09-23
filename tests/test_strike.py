"""The stand-back punch reader, and the bench it was measured on.

The reader's accuracy is a measurement on real footage (`scripts/train_strike.py`),
not something a unit test can assert. What is pinned here is the machinery
that measurement depends on: that the features are causal, that the shipped
weights load and are used, and that the scoring cannot flatter a detector.
"""

import numpy as np
import pytest

from ika import bodysynth
from ika.shadow import Consensus, Labels, Punch, consensus, match, score
from ika.strike import DIM, WEIGHTS, ExcursionReader, StrikeFeatures, StrikeReader


class FakeBody:
    def __init__(self, marks):
        self.image = marks.copy()
        self.world = marks.copy()
        self.visibility = np.ones(33, dtype=np.float32)


def body(right_reach=0.0, guard=1.0):
    marks = bodysynth.body_pose(guard=guard).astype(np.float32)
    image = np.zeros((33, 3), dtype=np.float32)
    image[:, 0] = 0.5 + marks[:, 0] * 0.2
    image[:, 1] = 0.5 - marks[:, 1] * 0.2
    b = FakeBody(image)
    b.world = marks.copy()
    # A straight thrown at the camera: the right wrist travels forward and out.
    b.image[16, 0] += 0.08 * right_reach
    b.image[16, 1] += 0.02 * right_reach
    b.world[16, 2] -= 0.4 * right_reach
    return b


def session(punch_at=(2.0,), seconds=3.0, fps=30.0):
    frames = []
    for i in range(int(seconds * fps)):
        at = i / fps
        reach = 0.0
        for p in punch_at:
            phase = (at - p) / 0.15
            if -1.0 <= phase <= 1.0:
                reach = max(reach, 1.0 - abs(phase))
        frames.append((at, body(right_reach=reach)))
    return frames


def test_features_are_causal():
    """The live loop and a replay must agree, so no feature may see the future.
    Changing frames after t cannot change the vector at t."""
    a, b = StrikeFeatures(), StrikeFeatures()
    frames = session(punch_at=(2.0,))
    other = session(punch_at=(2.0, 2.5))
    out_a = out_b = None
    for (at, fa), (_, fb) in zip(frames, other):
        ra, rb = a.update(fa, at), b.update(fb, at)
        if at <= 2.3 and ra is not None:
            out_a, out_b = ra["right"], rb["right"]
    np.testing.assert_array_equal(out_a, out_b)
    assert out_a.shape == (DIM,)


def test_a_body_the_landmarker_was_guessing_at_is_not_read():
    f = StrikeFeatures()
    guessed = body()
    guessed.visibility[16] = 0.1
    for i in range(20):
        assert f.update(guessed, i / 30) is None


def test_the_excursion_rule_calls_a_straight_and_not_a_still_guard():
    r = ExcursionReader()
    calls = [s for at, b in session(punch_at=(2.0,)) for s in r.observe(b, at)]
    assert [s.side for s in calls] == ["right"]
    assert 1.8 <= calls[0].at <= 2.1
    still = ExcursionReader()
    assert not [s for at, b in session(punch_at=()) for s in still.observe(b, at)]


@pytest.mark.skipif(not WEIGHTS.exists(), reason="strike weights not shipped")
def test_the_shipped_weights_load_and_score_both_arms():
    r = StrikeReader()
    assert 0.0 < r.model.threshold < 1.0
    for at, b in session(punch_at=()):
        r.observe(b, at)
    rows = r.features.update(body(), 3.1)
    scores = r.model(np.stack([rows["left"], rows["right"]]))
    assert scores.shape == (2,) and np.all((scores >= 0) & (scores <= 1))


# --- the bench cannot flatter a detector ----------------------------------

def P(at, side="right"):
    return Punch(at, side)


def test_matching_is_one_to_one():
    """Firing three times on one punch is one hit and two false punches."""
    pairs, missed, extra = match([P(1.0)], [P(0.95), P(1.0), P(1.05)])
    assert len(pairs) == 1 and not missed and len(extra) == 2


def test_labellers_agree_on_a_punch_even_when_they_argue_about_the_arm():
    a = Labels([P(1.0, "left"), P(2.0)], [(0.0, 3.0)], [(0.0, 3.0)])
    b = Labels([P(1.05, "right"), P(2.0), P(2.6)], [(0.0, 3.0)], [(0.0, 3.0)])
    c = consensus(a, b)
    assert [p.side for p in c.agreed] == ["?", "right"]
    assert [p.at for p in c.disputed] == [2.6]


def test_a_detection_on_a_disputed_label_is_neither_credited_nor_charged():
    truth = Consensus(agreed=[P(1.0)], disputed=[P(2.0)], windows=[(0.0, 60.0)],
                      agreement=1.0, agreement_unsided=1.0)
    s = score([P(1.0), P(2.0), P(3.0)], truth)
    assert (s.hits, s.missed, s.false) == (1, 0, 1)


def test_detections_outside_the_labelled_windows_do_not_count():
    truth = Consensus(agreed=[P(1.0)], disputed=[], windows=[(0.0, 3.0)],
                      agreement=1.0, agreement_unsided=1.0)
    s = score([P(1.0), P(10.0)], truth)
    assert s.false == 0
