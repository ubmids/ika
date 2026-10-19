"""`ika history` as text: separated by mode, a sparkline per habit, and labels
that come from the profile's own decision rather than a new statistic."""

import time

from ika import cli
from ika.history import EMPTY, describe, labels, render, spark
from ika.profile import DEFAULT_HALF_LIFE, Profile
from ika.tell.habits import Finding

DAY = 86400.0
GUARD = (("punch_right",), "guard_down_left")


def _finding(context, then, support=24, hits=14, base=0.12):
    return Finding(context, then, support, hits, hits / support, base, 1e-4)


def _drill_sessions(p, at_list, context=GUARD[0], then=GUARD[1], hits=14, support=24):
    for at in at_list:
        p.add([_finding(context, then, support, hits)], at=at)
        p.record_session(at, 180.0, {"punch_left": 40, "punch_right": 50},
                         [(_finding(context, then), hits, support)])


def test_a_spark_per_session_and_a_dot_where_the_setup_never_came_up():
    assert spark([(0, 0, 10), (0, 10, 10), (0, 0, 0), (0, 5, 10)]) == "▁█·▅"


def test_habits_are_described_in_words():
    assert describe("punch_right -> guard_down_left") == "after right -> left hand down"
    assert describe("punch_right then punch_left -> punch_right") == "after right, left -> right"
    assert describe("coach_1-2 -> guard_down_right") == "called 1-2 -> right hand down"
    assert describe("coach_1-2 -> guard_down_both") == "called 1-2 -> hands down"


def test_a_habit_that_stands_now_is_holding():
    now = time.time()
    p = Profile("t")
    _drill_sessions(p, [now - 2 * DAY, now - DAY])
    assert p.habits(now=now)
    assert labels(p, now)["punch_right -> guard_down_left"] == "holding"


def test_a_habit_that_stood_and_decayed_away_is_fading():
    now = time.time()
    p = Profile("t")
    _drill_sessions(p, [now - 2 * DAY, now - DAY])
    later = now + 8 * DEFAULT_HALF_LIFE
    assert not p.habits(now=later)
    assert labels(p, later)["punch_right -> guard_down_left"] == "fading"


def test_one_session_is_new_and_more_without_enough_is_not_yet():
    now = time.time()
    p = Profile("t")
    _drill_sessions(p, [now - DAY])
    assert labels(p, now)["punch_right -> guard_down_left"] == "new"
    # Two sessions, each too thin to pass the pooled support floor.
    q = Profile("t")
    _drill_sessions(q, [now - 2 * DAY, now - DAY], hits=3, support=4)
    assert labels(q, now)["punch_right -> guard_down_left"] == "not yet"


def test_a_coach_habit_needs_a_session_after_the_one_that_proposed_it():
    now = time.time()
    p = Profile("t")
    for days in (2, 1):
        p.add([_finding(("coach_1-2",), "guard_down_right", 20, 16, 0.1)], at=now - days * DAY)
    assert labels(p, now)["coach_1-2 -> guard_down_right"] == "new"


def test_coach_comes_first_and_the_two_modes_never_share_a_section():
    now = time.time()
    p = Profile("me")
    _drill_sessions(p, [now - 3 * DAY, now - 2 * DAY])
    coach = _finding(("coach_1-2",), "guard_down_right")
    p.record_session(now - DAY, 180.0, {"calls": 47, "answered": 40}, [(coach, 9, 15)],
                     mode="coach")
    text = "\n".join(render(p, "me", now=now))
    coach_at = text.index("ika called the combos")
    drill_at = text.index("ika read what you threw")
    assert coach_at < drill_at
    assert "called 1-2 -> right hand down" in text[coach_at:drill_at]
    assert "after right -> left hand down" in text[drill_at:]
    assert "47 calls" in text and "40 answered" in text and "90 punches" in text
    assert "3 session(s), 1 coach and 2 drill" in text
    # A drill habit's sparkline has one mark per drill session, not per session.
    row = next(line for line in text.splitlines() if "after right -> left hand down" in line)
    assert "▅▅ " in row and "58%  14/24" in row


def test_the_dates_of_each_session_are_shown():
    now = time.time()
    p = Profile("me")
    _drill_sessions(p, [now - DAY])
    text = "\n".join(render(p, "me", now=now))
    assert time.strftime("%Y-%m-%d", time.localtime(now - DAY)) in text


def test_an_empty_record_says_what_to_run(tmp_path, capsys):
    assert cli.main(["history", "--profile", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "Nothing on record for me yet" in out
    assert "ika coach" in out and "ika drill" in out
    assert out.index("ika coach") < out.index("ika drill")
    assert render(Profile("me"), "me") == EMPTY.format(subject="me").split("\n")
