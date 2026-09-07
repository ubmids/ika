"""The state machine, which is where hand control is actually won.

These tests describe the difference between a demo and something you would
leave running: not "does it recognise a fist" but "does it stay quiet while I
reach for my coffee".
"""

import numpy as np
import pytest

from owo.machine import GestureMachine

CLASSES = ["rest", "open_palm", "fist", "point", "peace"]


def onehot(name: str, confidence: float = 1.0) -> np.ndarray:
    p = np.full(len(CLASSES), (1.0 - confidence) / (len(CLASSES) - 1))
    p[CLASSES.index(name)] = confidence
    return p


def feed(machine, name, frames, start=0.0, step=1 / 30, confidence=1.0):
    """Hold a gesture for a number of frames, collecting intents."""
    out = []
    for i in range(frames):
        out += machine.update(onehot(name, confidence), start + i * step)
    return out


def engaged_machine(**kw):
    m = GestureMachine(CLASSES, **kw)
    feed(m, "open_palm", 10)
    assert m.engaged
    feed(m, "rest", 10, start=1.0)   # settle back to neutral
    return m


def test_nothing_fires_before_you_engage():
    """The whole point of engagement. Hands in frame must be inert."""
    m = GestureMachine(CLASSES)
    assert feed(m, "fist", 60) == []
    assert feed(m, "point", 60, start=2.0) == []
    assert not m.engaged


def test_the_engage_gesture_arms_it():
    m = GestureMachine(CLASSES)
    intents = feed(m, "open_palm", 10)
    assert [i.kind for i in intents] == ["engaged"]
    assert m.engaged


def test_a_held_gesture_fires_once_not_thirty_times():
    """Hysteresis. Holding a fist for two seconds means one fist, not sixty."""
    m = engaged_machine()
    fired = [i for i in feed(m, "fist", 60, start=2.0) if i.kind == "fired"]
    assert len(fired) == 1
    assert fired[0].gesture == "fist"


def test_returning_to_rest_lets_the_same_gesture_fire_again():
    m = engaged_machine()
    first = [i for i in feed(m, "fist", 20, start=2.0) if i.kind == "fired"]
    again = [i for i in feed(m, "fist", 20, start=3.0) if i.kind == "fired"]
    assert len(first) == 1 and len(again) == 0, "spent until neutral is seen"

    feed(m, "rest", 10, start=4.0)
    third = [i for i in feed(m, "fist", 20, start=5.0) if i.kind == "fired"]
    assert len(third) == 1


def test_a_brief_flicker_does_not_fire():
    """Passing through a shape on the way somewhere is not a gesture."""
    m = engaged_machine(dwell_frames=6)
    out = []
    for i in range(40):
        # two frames of fist, then back to rest, over and over
        name = "fist" if i % 8 < 2 else "rest"
        out += m.update(onehot(name), 2.0 + i / 30)
    assert [i for i in out if i.kind == "fired"] == []


def test_low_confidence_never_fires():
    """An uncertain model must be treated as silent, not as a vote."""
    m = engaged_machine(threshold=0.8)
    assert [i for i in feed(m, "fist", 60, start=2.0, confidence=0.5) if i.kind == "fired"] == []


def test_a_vanished_hand_clears_the_streak():
    """Losing the hand mid-gesture must not leave a primed trigger behind."""
    m = engaged_machine(dwell_frames=6)
    feed(m, "fist", 4, start=2.0)          # part way to firing
    m.update(None, 2.2)                    # hand leaves frame
    fired = [i for i in feed(m, "fist", 4, start=2.3) if i.kind == "fired"]
    assert fired == [], "streak should have restarted from zero"


def test_it_disengages_when_your_hands_go_quiet():
    """So you do not have to remember to turn it off."""
    m = engaged_machine(disengage_after=1.0)
    out = []
    for i in range(120):
        out += m.update(None, 10.0 + i / 30)
    assert [i.kind for i in out] == ["disengaged"]
    assert not m.engaged


def test_cooldown_spaces_out_different_gestures():
    m = engaged_machine(cooldown=1.0, dwell_frames=3)
    fired = [i for i in feed(m, "fist", 6, start=2.0) if i.kind == "fired"]
    assert len(fired) == 1
    feed(m, "rest", 4, start=2.3)
    soon = [i for i in feed(m, "point", 6, start=2.5) if i.kind == "fired"]
    assert soon == [], "inside the cooldown"
    feed(m, "rest", 4, start=3.4)
    later = [i for i in feed(m, "point", 6, start=3.6) if i.kind == "fired"]
    assert len(later) == 1


def test_progress_climbs_toward_firing_then_resets():
    """What the on-screen meter shows, so the user can learn the timing."""
    m = engaged_machine(dwell_frames=10)
    seen, fired = [], []
    for i in range(40):
        out = m.update(onehot("fist"), 2.0 + i / 30)
        if out:
            fired += out
            break
        seen.append(m.progress)
    assert seen == sorted(seen), "the meter must only ever climb"
    assert 0 < max(seen) < 1.0
    assert [i.kind for i in fired] == ["fired"]
    assert m.progress == 0.0, "resets after firing"


def test_latency_from_gesture_to_action_is_acceptable():
    """Smoothing and dwell both cost time, and the total is what the hand
    feels. Measured rather than assumed, because this is the number that
    decides whether the thing feels responsive or laggy."""
    for dwell, smoothing, budget in [(5, 0.6, 12), (5, 0.0, 6), (8, 0.6, 16)]:
        m = engaged_machine(dwell_frames=dwell, smoothing=smoothing)
        for frame in range(40):
            if m.update(onehot("fist"), 2.0 + frame / 30):
                assert frame + 1 <= budget, (
                    f"dwell={dwell} smoothing={smoothing}: took {frame + 1} frames"
                )
                break
        else:
            raise AssertionError(f"dwell={dwell} smoothing={smoothing}: never fired")


def test_neutral_poses_never_fire():
    m = engaged_machine()
    assert [i for i in feed(m, "rest", 90, start=2.0) if i.kind == "fired"] == []


def test_reset_disarms_everything():
    m = engaged_machine()
    m.reset()
    assert not m.engaged and m.smoothed is None and m.progress == 0.0


def test_smoothing_rejects_a_single_wrong_frame():
    """One bad frame in a confident stream should not move the answer."""
    m = engaged_machine(dwell_frames=4, smoothing=0.7)
    out = []
    for i in range(12):
        name = "peace" if i == 5 else "fist"   # one impostor frame
        out += m.update(onehot(name), 2.0 + i / 30)
    fired = [i for i in out if i.kind == "fired"]
    assert len(fired) == 1 and fired[0].gesture == "fist"
