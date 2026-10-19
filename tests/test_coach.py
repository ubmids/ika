"""Coach mode: ika calls the combos, so the setup is known, not read.

What these hold: the schedule gives the statistics enough of each combo and
leaves room for the guard read between calls; a call counts as answered only
when something is thrown in its window; the guard is read after the response
and not on top of it; the mining is the drill's own calibrated test; and coach
habits stay apart from drill habits in the profile.
"""

from collections import Counter

import numpy as np
import pytest

from ika import coach, sag
from ika.profile import Profile
from ika.tell.habits import Finding

FPS = 30.0


class Track(sag.GuardTrack):
    """Guard heights given directly, rather than read off a body."""

    def __init__(self, t, h):
        super().__init__()
        self._snap = (np.asarray(t, dtype=np.float64),
                      {s: np.asarray(v, dtype=np.float64) for s, v in h.items()})

    def snapshot(self):
        return self._snap


def flat(seconds, level=0.0):
    t = np.arange(0.0, seconds, 1.0 / FPS)
    return t, {"left": np.full(len(t), level), "right": np.full(len(t), level)}


def answered_session(calls, drop_after=None, rng=None, depth=-0.6, noise=0.05):
    """A person who throws every call on time. The right hand sits `depth`
    below the guard after `drop_after`, and at the guard after anything else."""
    rng = rng or np.random.default_rng(0)
    seconds = calls[-1].at + coach.gap(calls[-1].combo) + 2.0
    t, h = flat(seconds)
    punches = []
    for call in calls:
        at = call.at + 0.7
        for side in call.combo.sides():
            punches.append((at, side))
            at += 0.3
        if drop_after is not None and call.combo.name == drop_after:
            end = punches[-1][0]
            a, b = np.searchsorted(t, end + 0.2), np.searchsorted(t, end + 1.4)
            h["right"][a:b] = depth
    for side in h:
        h[side] = h[side] + rng.normal(0.0, noise, len(t))
    return Track(t, h), punches


# --- combos and numbering -------------------------------------------------

def test_numbers_are_standard_boxing_numbers():
    combo = coach.Combo.parse("1-2-3-4-5-6")
    assert combo.words == "one, two, three, four, five, six"
    # Odd is the lead hand, even the rear; orthodox leads with the left.
    assert combo.sides() == ("left", "right") * 3
    assert coach.Combo.parse("1-2").sides("southpaw") == ("right", "left")


def test_a_combo_outside_one_to_six_is_refused():
    with pytest.raises(ValueError):
        coach.Combo.parse("1-7")
    with pytest.raises(ValueError):
        coach.Combo.parse("jab")


def test_every_catalogue_combo_is_one_to_four_punches():
    for name in coach.CATALOGUE:
        assert 1 <= len(coach.Combo.parse(name)) <= 4


# --- the schedule ---------------------------------------------------------

def test_a_seed_repeats_a_schedule_exactly():
    combos = coach.choose(seed=4)
    assert coach.schedule(combos, seed=9) == coach.schedule(combos, seed=9)
    assert coach.schedule(combos, seed=9) != coach.schedule(combos, seed=10)


def test_every_combo_is_called_the_same_number_of_times_to_within_one():
    for seed in range(20):
        combos = coach.choose(seed=seed)
        counts = Counter(c.combo for c in coach.schedule(combos, seed=seed))
        assert set(counts) == set(combos)
        assert max(counts.values()) - min(counts.values()) <= 1


def test_one_default_round_calls_each_combo_often_enough_to_be_tested():
    """The rank test needs sag.MIN_SUPPORT answered calls of a combo. One
    three-minute round of three combos calls each 13 to 17 times, so the
    misses the punch reader makes still leave a combo testable."""
    for seed in range(50):
        combos = coach.choose(seed=seed)
        counts = Counter(c.combo for c in coach.schedule(combos, 180.0, seed=seed))
        assert min(counts.values()) >= 13, counts


def test_calls_are_dealt_evenly_not_in_blocks():
    """A guard that sags as someone tires would make whichever combo came
    late look like a habit if calls came in blocks."""
    combos = coach.choose(seed=1)
    calls = coach.schedule(combos, 600.0, seed=1)
    halves = [Counter(c.combo for c in calls[:len(calls) // 2]),
              Counter(c.combo for c in calls[len(calls) // 2:])]
    for combo in combos:
        assert abs(halves[0][combo] - halves[1][combo]) <= 2


def test_the_same_combo_is_never_called_twice_in_a_row():
    for seed in range(20):
        calls = coach.schedule(coach.choose(seed=seed), seed=seed)
        assert all(a.combo != b.combo for a, b in zip(calls, calls[1:]))


def test_the_guard_window_ends_before_the_next_call_can_be_answered():
    """The gap after a call has to hold the latest response and then the
    whole guard window after it, or one combo's read lands on the next one's
    punches."""
    for name in coach.CATALOGUE:
        combo = coach.Combo.parse(name)
        _lo, hi = coach.window(0.0, combo)
        assert hi + sag.WINDOW[1] < coach.gap(combo) + coach.REACTION


def test_calls_fit_inside_the_round():
    calls = coach.schedule(coach.choose(seed=2), 180.0, seed=2, start=10.0)
    assert calls[0].at == 10.0
    last = calls[-1]
    assert last.at + coach.gap(last.combo) <= 190.0


# --- choosing what to drill -------------------------------------------------

def test_known_habits_are_drilled_first_and_the_rest_rotate():
    picked = coach.choose(seed=0, known=["2-3-2"])
    assert picked[0].name == "2-3-2"
    assert len(picked) == coach.PER_SESSION
    assert len(set(picked)) == len(picked)
    others = {tuple(c.name for c in coach.choose(seed=s)) for s in range(10)}
    assert len(others) > 5


def test_a_hunch_from_one_session_is_called_again_but_never_fills_the_session():
    record = Profile("x")
    proposed = [Finding(("coach_1-2",), "guard_down_right", 15, 9, 0.6, 0.1, 0.01),
                Finding(("coach_3-2",), "guard_down_left", 15, 9, 0.6, 0.1, 0.01),
                Finding(("coach_2",), "guard_down_left", 15, 9, 0.6, 0.1, 0.01)]
    record.add(proposed, at=1000.0)
    picked = [c.name for c in coach.plan(record, now=2000.0, seed=0)]
    assert len(picked) == coach.PER_SESSION
    hunches = {"1-2", "3-2", "2"}
    assert len(hunches & set(picked)) == coach.PER_SESSION - 1


def test_drill_habits_do_not_steer_the_coach_plan():
    record = Profile("x")
    record.add([Finding(("punch_right",), "guard_down_right", 20, 15, 0.75, 0.1, 1e-6)],
               at=1000.0)
    a = [c.name for c in coach.plan(record, now=2000.0, seed=3)]
    b = [c.name for c in coach.choose(seed=3)]
    assert a == b


# --- alignment --------------------------------------------------------------

def test_a_call_with_nothing_thrown_is_unanswered_and_not_read():
    calls = coach.schedule([coach.Combo.parse("1-2")], 20.0)
    t, h = flat(30.0)
    reps = coach.align(calls, Track(t, h), [])
    assert reps and not any(r.answered for r in reps)
    assert coach.occasions(reps) == []


def test_a_punch_outside_the_window_does_not_answer_the_call():
    combo = coach.Combo.parse("1-2")
    call = coach.Call(5.0, combo)
    t, h = flat(15.0)
    _lo, hi = coach.window(5.0, combo)
    early = coach.align([call], Track(t, h), [(5.05, "left")])
    late = coach.align([call], Track(t, h), [(hi + 0.1, "left")])
    on_time = coach.align([call], Track(t, h), [(5.7, "left")])
    assert not early[0].answered and not late[0].answered
    assert on_time[0].answered and on_time[0].punches == 1


def test_one_detected_punch_is_enough_to_answer_a_long_combo():
    """Which punch, or how many, is not asked: at 47% recall a four-punch
    combo is caught whole one time in twenty."""
    call = coach.Call(5.0, coach.Combo.parse("1-2-3-2"))
    t, h = flat(15.0)
    rep, = coach.align([call], Track(t, h), [(6.0, "right")])
    assert rep.answered


def test_the_response_ends_no_earlier_than_this_persons_usual_pace():
    """A missed last punch must not end the response early, or the guard
    window sits on top of that punch."""
    combo = coach.Combo.parse("1-2-3-2")
    calls = [coach.Call(5.0 + 5 * i, combo) for i in range(8)]
    t, h = flat(60.0)
    # This person's last punch lands 2.1 s after the call; in the last rep
    # only the first punch was caught.
    punches = [(c.at + 2.1, "right") for c in calls[:-1]] + [(calls[-1].at + 0.8, "left")]
    reps = coach.align(calls, Track(t, h), punches)
    assert reps[-1].end == pytest.approx(calls[-1].at + 2.1, abs=0.01)


def test_before_the_pace_is_learned_the_expected_pace_stands():
    combo = coach.Combo.parse("1-2-3")
    call = coach.Call(5.0, combo)
    t, h = flat(15.0)
    rep, = coach.align([call], Track(t, h), [(5.6, "left")])
    assert rep.end == pytest.approx(coach.expected_end(5.0, combo))


def test_the_guard_is_read_after_the_response_not_during_it():
    """The window sits after the end of the response: a hand that is low only
    while punching reads as in guard."""
    combo = coach.Combo.parse("1-2")
    call = coach.Call(5.0, combo)
    t, h = flat(15.0)
    during = (t > 5.5) & (t < 6.1)
    h["right"][during] = -0.8
    rep, = coach.align([call], Track(t, h), [(5.7, "left"), (6.0, "right")])
    assert rep.heights["right"] == pytest.approx(0.0, abs=1e-6)


def test_frames_where_that_arm_is_punching_are_left_out():
    """sag.GuardTrack.after does this for the drill, and coach mode reuses it:
    a stray punch by the right arm inside the guard window is not a sag."""
    combo = coach.Combo.parse("1-2")
    call = coach.Call(5.0, combo)
    t, h = flat(15.0)
    stray = 6.0 + 0.8
    near = np.abs(t - stray) <= 0.15
    h["right"][near] = -0.8
    punches = [(5.7, "left"), (6.0, "right"), (stray, "right")]
    rep, = coach.align([call], Track(t, h), punches)
    assert rep.heights["right"] == pytest.approx(0.0, abs=1e-6)


# --- mining -----------------------------------------------------------------

def test_a_combo_that_leaves_the_right_hand_down_is_found():
    combos = [coach.Combo.parse(n) for n in ("1-2", "1-2-3", "2-3-2")]
    calls = coach.schedule(combos, 180.0, seed=0)
    track, punches = answered_session(calls, drop_after="2-3-2")
    found = coach.read(coach.align(calls, track, punches))
    assert found
    best = found[0]
    assert best.context == ("coach_2-3-2",)
    assert best.then == "guard_down_right"
    assert "after 2-3-2: right hand down" in coach.describe(best)


def test_no_habit_is_found_on_a_fighter_without_one():
    combos = [coach.Combo.parse(n) for n in ("1-2", "1-2-3", "2-3-2")]
    # The read's false rate is the rank test's level, about 5%: the size bar
    # (sag.MIN_EFFECT) does not bind at one round's sample sizes, since any
    # difference the corrected test is sure of is already that big. So this
    # is held to 10% over enough sessions to tell 5% from 15%.
    false = 0
    for seed in range(100):
        calls = coach.schedule(combos, 180.0, seed=seed)
        track, punches = answered_session(calls, rng=np.random.default_rng(seed), noise=0.3)
        false += bool(coach.read(coach.align(calls, track, punches)))
    assert false <= 10


def test_the_test_is_the_drills_own():
    """Same rank test, same correction, same support floor: a combo answered
    fewer than sag.MIN_SUPPORT times is not tested at all."""
    combos = [coach.Combo.parse(n) for n in ("1-2", "1-2-3", "2-3-2")]
    calls = coach.schedule(combos, 180.0, seed=0)
    track, punches = answered_session(calls, drop_after="2-3-2")
    few = [c for c in calls if c.combo.name != "2-3-2"] + \
          [c for c in calls if c.combo.name == "2-3-2"][:sag.MIN_SUPPORT - 1]
    few.sort(key=lambda c: c.at)
    assert coach.read(coach.align(few, track, punches)) == []
    reps = coach.align(calls, track, punches)
    assert coach.read(reps) == sag.read_occasions(coach.occasions(reps))


def test_the_session_rate_counts_answered_reps_of_that_combo_and_side():
    combos = [coach.Combo.parse(n) for n in ("1-2", "2-3-2")]
    calls = coach.schedule(combos, 60.0, seed=0)
    track, punches = answered_session(calls, drop_after="2-3-2")
    reps = coach.align(calls, track, punches)
    hits, support = coach.rate(reps, ("coach_2-3-2",), "guard_down_right")
    assert support == sum(c.combo.name == "2-3-2" for c in calls)
    assert hits == support
    assert coach.rate(reps, ("coach_1-2",), "guard_down_right")[0] == 0


# --- the live call ----------------------------------------------------------

def test_a_known_habit_adds_a_reminder_to_its_call_only():
    habits = [Finding(("coach_1-2",), "guard_down_right", 20, 15, 0.75, 0.1, 1e-6),
              Finding(("punch_right",), "guard_down_left", 20, 15, 0.75, 0.1, 1e-6)]
    remind = coach.reminders(habits)
    assert remind == {"1-2": "right hand up"}
    calls = coach.with_reminders(
        coach.schedule([coach.Combo.parse("1-2"), coach.Combo.parse("3-2")], 30.0), remind)
    for call in calls:
        if call.combo.name == "1-2":
            assert call.words == "one, two... right hand up"
        else:
            assert call.words == "three, two"


def test_both_hands_on_one_combo_say_hands_up():
    habits = [Finding(("coach_1-2",), "guard_down_right", 20, 15, 0.75, 0.1, 1e-6),
              Finding(("coach_1-2",), "guard_down_left", 20, 15, 0.75, 0.1, 1e-6)]
    assert coach.reminders(habits) == {"1-2": "hands up"}


# --- what a session puts on the profile ---------------------------------------

def test_a_follow_up_that_came_to_nothing_is_recorded_against_the_hunch():
    """The drill can only record what looked strong. Coach mode knows how often
    it called each combo, so a hunch called again and not seen again is
    recorded with its plain counts."""
    combos = [coach.Combo.parse(n) for n in ("1-2", "1-2-3", "2-3-2")]
    calls = coach.schedule(combos, 180.0, seed=0)
    track, punches = answered_session(calls)
    reps = coach.align(calls, track, punches)
    record = Profile("x")
    record.add([Finding(("coach_1-2",), "guard_down_right", 15, 9, 0.6, 0.1, 0.01)], at=1.0)
    added = coach.evidence(reps, record)
    follow, = added
    assert follow.context == ("coach_1-2",) and follow.then == "guard_down_right"
    assert follow.hits == 0
    assert follow.support == sum(c.combo.name == "1-2" for c in calls)
    # Nothing is recorded for a combo with no hunch on it and nothing to propose.
    assert coach.evidence(reps, Profile("y")) == []


def test_the_session_that_proposed_a_coach_habit_does_not_confirm_it():
    """Picked for looking strong, so it is not counted. Two more sessions are."""
    strong = Finding(("coach_1-2",), "guard_down_right", 20, 16, 0.8, 0.1, 1e-6)
    record = Profile("x")
    day = 86400.0
    record.add([strong], at=1 * day)
    record.add([strong], at=2 * day)
    assert record.habits(now=2 * day) == []
    record.add([strong], at=3 * day)
    standing, = record.habits(now=3 * day)
    assert standing.sessions == 2
    assert standing.support == pytest.approx(40, abs=2)


def test_drill_habits_still_count_every_session():
    strong = Finding(("punch_right",), "guard_down_right", 20, 16, 0.8, 0.1, 1e-6)
    record = Profile("x")
    record.add([strong], at=86400.0)
    record.add([strong], at=2 * 86400.0)
    assert len(record.habits(now=2 * 86400.0)) == 1
