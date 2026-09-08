"""The pose gate, whose whole power is the rarity of the pose it requires.

The swipe lane fired 3 times a minute at an idle hand while catching half of
real swipes, and that is not a training problem: a fast straight hand movement
is what reaching for a cup looks like, so displacement cannot separate intent
from traffic however good the model gets. The fix is interaction design.

Measured, and the number that matters is how often an idle hand already
satisfies the gate:

    gate pose          share of idle   idle firings/min   swipes caught
    none                         n/a               3.00             50%
    open_palm + point            35%               1.17             42%
    pinch                       3.3%               0.00             39%
    thumbs_up                   0.3%               0.00             33%
"""

import pytest

from ika.control import SWIPE_GATE, SWIPE_GATE_SHARE
from ika.gate import PoseGate


def hold(gate, pose, frames, start=0.0, fps=30.0):
    for i in range(frames):
        gate.push(pose, start + i / fps)
    return gate


def test_a_relaxed_hand_does_not_open_the_gate():
    """The property the whole mechanism exists for."""
    assert not hold(PoseGate(), "rest", 18).open()


def test_the_gate_pose_opens_it():
    assert hold(PoseGate(), SWIPE_GATE[0], 18).open()


def test_a_partial_hold_is_enough():
    """The pose classifier is briefly uncertain at the start and end of any
    movement, and demanding a perfect run rejected genuine swipes in favour of
    nothing at all."""
    gate = PoseGate(share=0.5)
    hold(gate, SWIPE_GATE[0], 10)
    hold(gate, "rest", 8, start=10 / 30.0)
    assert gate.held == pytest.approx(10 / 18, abs=0.01)
    assert gate.open()


def test_a_brief_touch_of_the_pose_is_not_enough():
    gate = PoseGate(share=0.5)
    hold(gate, SWIPE_GATE[0], 3)
    hold(gate, "rest", 15, start=3 / 30.0)
    assert not gate.open()


def test_the_gate_forgets_old_frames():
    """It runs for as long as the camera does, so the record is bounded by
    time. Otherwise a pose held once would open the gate forever."""
    gate = PoseGate(seconds=0.4)
    hold(gate, SWIPE_GATE[0], 12)
    assert gate.open()
    hold(gate, "rest", 14, start=0.5)
    assert not gate.open(), "the old pose should have aged out"


def test_an_empty_gate_is_shut():
    """Not open by default, because the default has to be the safe one."""
    assert not PoseGate().open()
    assert PoseGate().held == 0.0


def test_clearing_shuts_it():
    gate = hold(PoseGate(), SWIPE_GATE[0], 18)
    assert gate.open()
    gate.clear()
    assert not gate.open()


def test_the_gate_pose_is_a_rare_one():
    """The gate is only as good as the pose is unusual. open_palm and point
    together are 35% of what an idle hand reads as, and gating on them only
    halved the idle firings. This guards against someone picking a common
    pose again for ergonomic reasons."""
    assert "open_palm" not in SWIPE_GATE, "35% of idle frames; not a gate"
    assert "point" not in SWIPE_GATE, "18% of idle frames; not a gate"
    assert "rest" not in SWIPE_GATE, "the definition of idle"
    assert SWIPE_GATE, "there must be a gate at all"


def test_the_share_leaves_room_for_classifier_wobble():
    assert 0.0 < SWIPE_GATE_SHARE < 1.0
