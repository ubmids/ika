"""Calling the habit out loud, before it happens."""

from dataclasses import dataclass

from ika.cue import Caller, measure
from ika.drill import Event
from ika.profile import Profile, rate_in


@dataclass(frozen=True)
class Habit:
    context: tuple
    then: str
    probability: float = 0.7


def test_it_calls_the_lapse_when_the_setup_happens():
    c = Caller([Habit(("punch_right", "punch_right"), "guard_down_both")])
    assert c.update(["punch_right"], 1.0) is None
    cue = c.update(["punch_right", "punch_right"], 1.3)
    assert cue is not None and cue.words == "hands up"


def test_it_only_calls_lapses():
    """ "After a jab you throw a cross" may be true; shouting it is useless."""
    c = Caller([Habit(("punch_left",), "punch_right")])
    assert c.armed == 0
    assert c.update(["punch_left"], 1.0) is None


def test_it_does_not_nag():
    c = Caller([Habit(("punch_right",), "guard_down_right")], min_gap=2.0)
    assert c.update(["punch_right"], 1.0)
    assert c.update(["punch_right", "punch_right"], 1.5) is None
    assert c.update(["punch_right"] * 3, 3.1)


def test_the_more_specific_setup_wins():
    c = Caller([Habit(("punch_right",), "guard_down_right", 0.4),
                Habit(("punch_left", "punch_right"), "guard_down_both", 0.8)])
    cue = c.update(["punch_left", "punch_right"], 1.0)
    assert cue.then == "guard_down_both"


def test_measure_counts_a_cue_right_only_if_the_lapse_follows_in_time():
    c = Caller([Habit(("punch_right",), "guard_down_right")], min_gap=0.0)
    c.update(["punch_right"], 1.0)
    c.update(["punch_right"], 5.0)
    events = [Event("guard_down_right", 1.4, 1.0), Event("guard_down_right", 9.0, 1.0)]
    m = measure(c.cues, events, window=1.5)
    assert (m["cues"], m["right"]) == (2, 1)
    assert abs(m["median_lead"] - 0.4) < 1e-9


# --- history --------------------------------------------------------------

def test_a_habit_rate_is_counted_from_the_stream_including_zeros():
    stream = ["punch_right", "guard_down_right", "punch_right", "punch_left"]
    assert rate_in(stream, ("punch_right",), "guard_down_right") == (1, 2)
    assert rate_in(["punch_left"], ("punch_right",), "guard_down_right") == (0, 0)


def test_history_shows_a_habit_fading_and_survives_the_file(tmp_path):
    habit = Habit(("punch_right",), "guard_down_right")
    p = Profile(subject="t")
    p.record_session(1.0, 60.0, {}, [(habit, 10, 10)])
    p.record_session(2.0, 60.0, {}, [(habit, 0, 10)])
    path = tmp_path / "t.json"
    p.save(path)
    (key, series), = Profile.load(path).history()
    assert key == "punch_right -> guard_down_right"
    assert [(h, s) for _, h, s in series] == [(10, 10), (0, 10)]
