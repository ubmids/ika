"""The drill loop, which is the product the measurements pointed at.

It exists because of one realisation: the two events a drill coach needs are
geometric, not classified. A punch is a hand closing on the lens; a dropped
guard is a wrist below its own baseline. Neither needs a model trained on
anyone's hands, so the loop closes without recording a labelled dataset in
laptop framing, which does not exist publicly.
"""

import numpy as np
import pytest

from ika import bodysynth, synth
from ika.drill import (CALIBRATION_SECONDS, GUARD_DROP, GUARD_RESTORE,
                       SIMULTANEOUS_SECONDS, DrillReader, Event, read_habits)

FPS = 30.0


class FakeHand:
    def __init__(self, marks, is_left=False):
        self.image = marks
        self.world = marks
        self.is_left = is_left


class FakeBody:
    def __init__(self, marks, visibility=None):
        self.image = marks
        self.world = marks
        self.visibility = (np.ones(33, dtype=np.float32) if visibility is None
                           else visibility)


def hand(grow=1.0, is_left=False):
    marks = synth.place(synth.pose("open_palm"), scale=grow)
    image = np.zeros((21, 3), dtype=np.float32)
    image[:, 0] = 0.5 + marks[:, 0] * 0.08
    image[:, 1] = 0.5 - marks[:, 1] * 0.08
    return FakeHand(image, is_left)


def body(guard=1.0, visibility=None):
    marks = bodysynth.body_pose(guard=guard)
    image = np.zeros((33, 3), dtype=np.float32)
    image[:, 0] = 0.5 + marks[:, 0] * 0.08
    image[:, 1] = 0.55 - marks[:, 1] * 0.08
    made = FakeBody(image, visibility)
    made.world = marks.astype(np.float32)
    return made


class Session:
    """Drives a reader through a scripted session on a synthetic clock."""

    def __init__(self, **kw):
        self.reader = DrillReader(**kw)
        self.at = 0.0

    def rest(self, seconds, guard=1.0):
        out = []
        for _ in range(int(seconds * FPS)):
            out += self.reader.observe([hand()], body(guard), self.at)
            self.at += 1 / FPS
        return out

    def punch(self, is_left=False, rate=3.5):
        out = []
        for i in range(8):
            out += self.reader.observe(
                [hand(float(np.exp(rate * i / FPS)), is_left)], body(1.0), self.at)
            self.at += 1 / FPS
        for i in range(6):
            out += self.reader.observe(
                [hand(float(np.exp(rate * (7 - i) / FPS)), is_left)], body(1.0), self.at)
            self.at += 1 / FPS
        return out

    def drop_guard(self):
        out = []
        for i in range(10):
            out += self.reader.observe([hand()], body(1.0 - i / 9), self.at)
            self.at += 1 / FPS
        for i in range(10):
            out += self.reader.observe([hand()], body(i / 9), self.at)
            self.at += 1 / FPS
        return out


# --- calibration ----------------------------------------------------------

def test_it_learns_this_persons_resting_guard_rather_than_assuming_one():
    """A resting guard of +0.14 and a dropped one of -0.84 were measured on
    synthetic bodies. Real people are built differently and sit at different
    distances, so the baseline has to be learned."""
    s = Session()
    s.rest(CALIBRATION_SECONDS + 0.5)
    assert s.reader.calibrated
    assert s.reader.baseline.samples > 10
    assert abs(s.reader.baseline.guard_left - s.reader.baseline.guard_right) < 0.1


def test_nothing_is_reported_during_calibration():
    """Otherwise the first seconds produce events against a baseline that does
    not exist yet."""
    s = Session()
    assert s.rest(CALIBRATION_SECONDS - 0.5) == []
    assert not s.reader.calibrated


# --- punches --------------------------------------------------------------

def test_a_hand_closing_on_the_lens_is_a_punch():
    s = Session()
    s.rest(CALIBRATION_SECONDS + 0.3)
    names = [e.name for e in s.punch()]
    assert "punch_right" in names


def test_which_hand_threw_it_is_recorded():
    s = Session()
    s.rest(CALIBRATION_SECONDS + 0.3)
    assert "punch_left" in [e.name for e in s.punch(is_left=True)]


def test_one_punch_is_reported_once_not_every_frame():
    """The closing signal holds for the whole strike, so without a cooldown a
    single punch fires on eight consecutive frames."""
    s = Session()
    s.rest(CALIBRATION_SECONDS + 0.3)
    fired = [e for e in s.punch() if e.name.startswith("punch")]
    assert len(fired) == 1, [e.detail for e in fired]


def test_a_still_hand_is_not_a_punch():
    s = Session()
    s.rest(CALIBRATION_SECONDS + 0.3)
    assert [e for e in s.rest(2.0) if e.name.startswith("punch")] == []


def test_no_punch_is_reported_without_a_body_in_view():
    """Measured: a lean-in with no body visible is indistinguishable from a
    strike, crossover at 1.25/s. Reporting a punch there would tell someone
    they threw one when they leaned, and they would rightly stop trusting it."""
    s = Session()
    s.rest(CALIBRATION_SECONDS + 0.3)
    out = []
    for i in range(8):
        out += s.reader.observe([hand(float(np.exp(3.5 * i / FPS)))], None, s.at)
        s.at += 1 / FPS
    assert [e for e in out if e.name.startswith("punch")] == []


# --- guard ----------------------------------------------------------------

def test_a_dropped_guard_is_detected_and_recovers():
    s = Session()
    s.rest(CALIBRATION_SECONDS + 0.3)
    names = [e.name for e in s.drop_guard()]
    assert any(n.startswith("guard_down") for n in names)
    assert any(n.startswith("guard_up") for n in names)


def test_the_guard_has_hysteresis():
    """One threshold makes the state chatter at the boundary, filling the habit
    stream with alternating guard events that mean nothing."""
    assert GUARD_RESTORE < GUARD_DROP


def test_a_guard_read_off_an_invented_wrist_is_refused():
    """A pose model does not withhold a joint it cannot see, it extrapolates
    one, so a guard height off that is a number with nothing behind it."""
    s = Session()
    s.rest(CALIBRATION_SECONDS + 0.3)
    blind = np.ones(33, dtype=np.float32)
    from ika.body import LEFT_WRIST, RIGHT_WRIST
    blind[LEFT_WRIST] = blind[RIGHT_WRIST] = 0.05
    out = []
    for i in range(20):
        out += s.reader.observe([hand()], body(0.0, visibility=blind), s.at)
        s.at += 1 / FPS
    assert [e for e in out if e.name.startswith("guard")] == []


# --- the flaw the first end-to-end run exposed ---------------------------

def test_both_hands_dropping_is_one_lapse_not_two():
    """The first run reported "after guard_down_left: guard_down_right, 100%
    of the time, 6.7x" as a discovery. It was not a discovery, it was the order
    the two events happened to be emitted in. Sequential mining over
    simultaneous events invents dependencies, and four such phantoms appeared
    alongside the one real habit."""
    s = Session()
    s.rest(CALIBRATION_SECONDS + 0.3)
    s.drop_guard()
    stream = s.reader.stream()
    assert "guard_down_both" in stream
    assert "guard_down_left" not in stream and "guard_down_right" not in stream


def test_one_side_dropping_alone_keeps_its_side():
    """Merging must not erase a genuinely one-sided lapse, which is exactly the
    thing a coach would want told."""
    merged = DrillReader()
    e = Event("guard_down_left", 10.0, 1.0)
    kept = merged._append(e)
    assert kept is not None and kept.name == "guard_down_left"


def test_events_far_apart_are_not_merged():
    r = DrillReader()
    r._append(Event("guard_down_left", 10.0, 1.0))
    late = r._append(Event("guard_down_right", 10.0 + SIMULTANEOUS_SECONDS + 0.1, 1.0))
    assert late is not None and late.name == "guard_down_right"


# --- the whole point ------------------------------------------------------

def test_it_finds_a_habit_that_was_planted():
    """The FRIDAY read, end to end, with no trained action classifier. Plant
    "drops the guard after two punches, 75% of the time" and see if it comes
    back out."""
    s = Session()
    s.rest(CALIBRATION_SECONDS + 1.0)
    rng = np.random.default_rng(0)
    for _ in range(40):
        s.punch(); s.rest(0.4)
        s.punch(); s.rest(0.4)
        if rng.random() < 0.75:
            s.drop_guard()
        else:
            s.rest(0.7)
        s.rest(0.6)

    found = read_habits(s.reader)
    hits = [
        f for f in found
        if f.then == "guard_down_both" and "punch_right" in f.context
    ]
    assert hits, [f.describe() for f in found]
    best = max(hits, key=lambda f: f.probability)
    assert 0.5 < best.probability < 0.95, best.describe()
    assert best.lift > 2.0


def test_a_subject_with_no_habit_yields_no_such_finding():
    """The other half. Drop the guard at random rather than after punches, and
    the punch-then-drop habit must not appear."""
    s = Session()
    s.rest(CALIBRATION_SECONDS + 1.0)
    rng = np.random.default_rng(1)
    for _ in range(40):
        for _ in range(rng.integers(1, 4)):
            s.punch(); s.rest(0.4)
        if rng.random() < 0.35:
            s.drop_guard()
        s.rest(0.6)

    found = read_habits(s.reader)
    strong = [
        f for f in found
        if f.then == "guard_down_both" and f.context == ("punch_right", "punch_right")
        and f.probability > 0.6
    ]
    assert not strong, [f.describe() for f in strong]


def test_the_stream_is_bounded():
    """It runs for as long as the camera does."""
    from ika.drill import STREAM_LIMIT

    r = DrillReader()
    for i in range(STREAM_LIMIT + 500):
        r._append(Event(f"punch_right", i * 1.0, 1.0))
    assert len(r.events) == STREAM_LIMIT


def test_reset_clears_everything_including_the_baseline():
    s = Session()
    s.rest(CALIBRATION_SECONDS + 0.5)
    s.punch()
    s.reader.reset()
    assert not s.reader.calibrated
    assert s.reader.stream() == []
    assert s.reader.baseline.samples == 0
