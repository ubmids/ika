"""A habit record across sessions: does it accumulate, and does it let go?

The second question is the one that decides whether this is a product. Telling
a fighter about a habit is how the habit gets coached out, so every finding
this thing stores is on a clock. A record that keeps quoting last month's read
is not a stale feature, it is a confident lie that walks its owner into a
counter.

Ground truth is imposed by hand throughout: either a planted `Habit` whose
answer is known before anything runs, or `Observation` counts written out so
the arithmetic can be checked against `binomial_tail` directly.
"""

import json

import numpy as np
import pytest

from ika.profile import (
    DEFAULT_FDR,
    DEFAULT_HALF_LIFE,
    FORGETTABLE_AFTER_HALF_LIVES,
    MIN_EFFECTIVE_SESSIONS,
    MIN_SESSIONS,
    SECONDS_PER_DAY,
    Observation,
    Profile,
    propose,
)
from ika.tell import Fighter, Habit, mine
from ika.tell.actions import names
from ika.tell.habits import binomial_tail

DAY = SECONDS_PER_DAY
PLANTED = Habit(("jab", "jab"), "drop_guard", 0.70)

# Weak on purpose: 22% against a 9% base rate, which is a real tendency that a
# single short session usually cannot prove. Used for the pooling tests.
WEAK = Habit(("low_kick",), "step_back", 0.22)

# A week instead of the two-week default, so the decay tests run over a
# readable number of days rather than a couple of months.
WEEK = 7.0 * DAY


def session(habits, length=1200, seed=0):
    """One session of proposals from a fighter with `habits` planted."""
    return propose(names(Fighter(habits=habits, seed=seed).sequence(length)))


def standing(profile, now, half_life=DEFAULT_HALF_LIFE,
             context=("jab", "jab"), then="drop_guard"):
    """The one habit under test, or None if the record no longer claims it."""
    found = [h for h in profile.habits(now, half_life)
             if h.context == context and h.then == then]
    return found[0] if found else None


# --- the arithmetic of decay ---------------------------------------------

def test_an_observation_halves_in_weight_every_half_life():
    observation = Observation(("jab",), "cross", 10, 20, at=0.0, base=0.2)
    assert observation.weight(0.0, WEEK) == pytest.approx(1.0)
    assert observation.weight(WEEK, WEEK) == pytest.approx(0.5)
    assert observation.weight(3 * WEEK, WEEK) == pytest.approx(0.125)


def test_an_observation_from_the_future_is_not_worth_more_than_one_from_now():
    """Clock skew between a recording box and a laptop should not manufacture
    weight above 1.0, which is what a negative age would do."""
    observation = Observation(("jab",), "cross", 10, 20, at=100.0, base=0.2)
    assert observation.weight(0.0, WEEK) == pytest.approx(1.0)


def test_a_large_older_sample_outweighs_a_tiny_fresh_one():
    """18 of 20 last week is better evidence than 2 of 2 today, and pooling
    weighted counts is what gets that ordering right. Averaging the two rates
    would report 95%, splitting the difference with a sample of two."""
    profile = Profile("f", [
        Observation(("jab",), "drop_guard", 18, 20, at=0.0, base=0.06),
        Observation(("jab",), "drop_guard", 2, 2, at=WEEK, base=0.06),
    ])
    habit = standing(profile, WEEK, WEEK, context=("jab",))
    assert habit is not None
    # The week-old session still carries 14 of the roughly 16 effective trials.
    assert habit.probability == pytest.approx(0.912, abs=0.01)
    assert habit.probability < 0.93, "a sample of two moved the answer too far"


# --- accumulating across sessions ----------------------------------------

def test_evidence_accumulates_across_sessions():
    profile = Profile("f")
    support = []
    for k in range(5):
        profile.add(session((PLANTED,), seed=10 + k), at=k * DAY)
        habit = standing(profile, k * DAY, WEEK)
        if habit:
            support.append(habit.support)
    assert profile.sessions() == 5
    assert len(support) >= 3
    assert support == sorted(support), support
    assert support[-1] > support[0] * 1.5


def test_sessions_counts_sessions_and_not_findings():
    """Two findings from one session are one session of evidence. Counting
    findings would let a single noisy stream look like a history."""
    profile = Profile("f")
    profile.add(session((PLANTED, WEAK), seed=1), at=0.0)
    assert len(profile.observations) >= 2
    assert profile.sessions() == 1


def test_pooling_lets_a_weak_but_consistent_tendency_become_significant():
    """Hand-built so the statistics are checkable: 5 of 20 against a 12% base
    rate is 2.1x, but on one session's evidence it is not significant. The
    same rate repeated five times is, and nothing about the tendency changed,
    only how much of it was seen."""
    assert binomial_tail(5, 20, 0.12) > DEFAULT_FDR       # 0.083, no claim
    assert binomial_tail(25, 100, 0.12) < DEFAULT_FDR     # 0.00026, a claim

    thin = Profile("f", [Observation(("jab",), "low_kick", 5, 20, 0.0, 0.12)])
    assert thin.habits(0.0, WEEK) == []

    pooled = Profile("f", [
        Observation(("jab",), "low_kick", 5, 20, at=k * DAY, base=0.12)
        for k in range(5)
    ])
    habit = standing(pooled, 4 * DAY, WEEK, context=("jab",), then="low_kick")
    assert habit is not None
    assert habit.p_value < DEFAULT_FDR
    assert habit.support > 20 and habit.sessions == 5
    assert habit.probability == pytest.approx(0.25, abs=0.02)


def test_a_tendency_no_single_session_can_prove_ends_up_on_the_record():
    """The same point end to end, on streams rather than by hand. In 700
    action sessions the 22% tendency clears the miner's own bar in about 40%
    of sessions, so a session-at-a-time product would report it as an
    on-again-off-again read. Pooled, every subject's record has it."""
    proven = 0
    sessions_that_proved_it = 0
    total_sessions = 0
    for subject in range(6):
        profile = Profile(f"f{subject}")
        for k in range(8):
            stream = names(Fighter(habits=(WEAK,), seed=600 + subject * 20 + k).sequence(700))
            total_sessions += 1
            sessions_that_proved_it += any(
                f.context == ("low_kick",) and f.then == "step_back" for f in mine(stream)
            )
            profile.add(propose(stream), at=k * DAY)
        proven += standing(profile, 7 * DAY, WEEK,
                           context=("low_kick",), then="step_back") is not None
    assert sessions_that_proved_it / total_sessions < 0.6, "sessions were not thin"
    assert proven >= 5, f"pooling found it for only {proven} of 6 subjects"


# --- not inventing anything ----------------------------------------------

def test_a_brand_new_subject_gets_no_habits_from_one_session():
    """The miner reports a habit that does not exist in 15% of fighters that
    have none, and a first session is exactly when a record has no way to tell
    the difference. Requiring a second session is the guard: an invented
    finding comes from the noise of one stream, and the odds of the same
    context and outcome recurring are small."""
    invented = 0
    for seed in range(30):
        profile = Profile("new")
        profile.add(session((), length=3000, seed=1000 + seed), at=0.0)
        invented += bool(profile.habits(0.0, WEEK))
    assert invented == 0, f"{invented} of 30 new subjects were handed a read"


def test_one_session_of_a_real_habit_is_also_withheld():
    """The cost of the rule above, stated rather than hidden: an 11.5x habit
    is not quoted until a second session confirms it. Paying one session of
    delay to not invent reads on 15% of new subjects is the right trade for
    something a fighter acts on."""
    profile = Profile("f")
    profile.add(session((PLANTED,), seed=1), at=0.0)
    assert standing(profile, 0.0, WEEK) is None
    assert profile.observations, "the evidence is kept, it is just not quoted"

    profile.add(session((PLANTED,), seed=2), at=DAY)
    habit = standing(profile, DAY, WEEK)
    assert habit is not None and habit.lift > 5
    assert habit.sessions == MIN_SESSIONS


def test_pooling_does_not_invent_habits_on_a_fighter_with_none():
    """Pooling could easily be worse than one session rather than better,
    because a session only reports its extremes: pool five sessions of
    cherry-picked tails and the binomial test cannot tell it is looking at a
    selected sample. Measured over 20 null subjects of five sessions each,
    with proposals capped at a 0.2 false-discovery rate."""
    invented = []
    for subject in range(20):
        profile = Profile("null")
        for k in range(5):
            profile.add(session((), seed=9000 + subject * 10 + k), at=k * DAY)
        invented.append(len(profile.habits(4 * DAY, WEEK)))
    rate = float(np.mean(np.array(invented) > 0))
    assert rate < 0.15, f"pooling is worse than one stream: {invented}"


# --- decay, which is the point of the module -----------------------------

def test_a_habit_that_stops_happening_fades_away():
    """The whole reason this module exists. Five sessions establish the habit
    at 11.5x, then the fighter is coached out of it and the sessions stop
    containing it. Measured: with a one week half life the read is dropped
    after 11 absent daily sessions, and at the two week default after 23.

    Note what does the dropping. The pooled p-value never rises: decay shrinks
    hits and support together, so the *rate* stays at 70% and stays
    astronomically unlikely. Only the standing session weight falls, which is
    why the gate has to be on how much un-decayed evidence is left and not on
    the p-value alone."""
    for seed in (0, 1):
        profile = Profile("f")
        for k in range(5):
            profile.add(session((PLANTED,), seed=400 + seed * 10 + k), at=k * DAY)
        established = standing(profile, 4 * DAY, WEEK)
        assert established is not None and established.lift > 5

        absent = 0
        while standing(profile, (4 + absent) * DAY, WEEK) is not None:
            absent += 1
            assert absent < 30, "a coached-out habit is still being quoted"
            profile.add(session((), seed=8000 + seed * 40 + absent), at=(4 + absent) * DAY)
        assert 9 <= absent <= 13, f"faded after {absent} absent sessions"


def test_decay_erodes_the_p_value_before_the_habit_drops_out():
    """A moderate habit does lose significance on the statistics alone, which
    the 11.5x one above does not. Same 5-of-20 evidence, read at increasing
    ages against the two week default: p climbs from 0.0003 to 0.02 as the
    effective sample shrinks, and by six weeks there is not enough left to
    test at all."""
    profile = Profile("f", [
        Observation(("jab",), "low_kick", 5, 20, at=k * DAY, base=0.12)
        for k in range(5)
    ])
    seen = []
    for day in (4, 10, 20):
        habit = standing(profile, day * DAY, DEFAULT_HALF_LIFE,
                         context=("jab",), then="low_kick")
        assert habit is not None
        seen.append((habit.support, habit.p_value))
    supports = [s for s, _ in seen]
    p_values = [p for _, p in seen]
    assert supports == sorted(supports, reverse=True), supports
    assert p_values == sorted(p_values), p_values
    assert p_values[0] < 0.001 and p_values[-1] > 0.01
    assert standing(profile, 42 * DAY, DEFAULT_HALF_LIFE,
                    context=("jab",), then="low_kick") is None


def test_a_long_history_fades_more_slowly_than_a_thin_one():
    """The asymmetry that makes decay usable rather than annoying. Twenty
    sessions of evidence should survive a quiet fortnight; two sessions should
    not, because two sessions were never much of a claim."""
    def days_until_forgotten(count):
        profile = Profile("f", [
            Observation(("jab",), "drop_guard", 14, 20, at=k * DAY, base=0.06)
            for k in range(count)
        ])
        last = (count - 1) * DAY
        day = 0
        while standing(profile, last + day * DAY, WEEK, context=("jab",)) is not None:
            day += 1
            assert day < 60
        return day

    thin = days_until_forgotten(2)
    long_history = days_until_forgotten(20)
    assert thin < long_history, (thin, long_history)
    assert long_history > 3 * thin


def test_a_shorter_half_life_forgets_sooner():
    """The half life is the product's one dial for how fast advice goes stale,
    so it has to actually move the answer."""
    profile = Profile("f", [
        Observation(("jab",), "drop_guard", 14, 20, at=k * DAY, base=0.06)
        for k in range(5)
    ])
    now = 4 * DAY + 20 * DAY
    assert standing(profile, now, 2.0 * DAY, context=("jab",)) is None
    assert standing(profile, now, 30.0 * DAY, context=("jab",)) is not None


def test_a_habit_that_comes_back_is_quoted_again():
    """Fading is not deleting. The evidence stays on the record, so a tendency
    that reappears is backed by everything that came before it instead of
    starting from nothing.

    It still takes two fresh sessions rather than one, and that is the same
    rule as for a new subject: after a month away, one session of evidence is
    exactly as thin as a first session, whatever the file remembers."""
    profile = Profile("f")
    for k in range(5):
        profile.add(session((PLANTED,), seed=440 + k), at=k * DAY)
    stale = 30 * DAY
    assert standing(profile, stale, WEEK) is None

    profile.add(session((PLANTED,), seed=460), at=stale)
    assert standing(profile, stale, WEEK) is None, "one session revived it"

    profile.add(session((PLANTED,), seed=461), at=stale + DAY)
    revived = standing(profile, stale + DAY, WEEK)
    assert revived is not None
    assert revived.sessions == 7, "the older sessions still count for something"


# --- the record on disk ---------------------------------------------------

def test_a_saved_profile_loads_back_exactly(tmp_path):
    profile = Profile("kolawole")
    for k in range(3):
        profile.add(session((PLANTED,), seed=20 + k), at=k * DAY + 0.5)
    path = tmp_path / "kolawole.json"
    profile.save(path)
    loaded = Profile.load(path)
    assert loaded.subject == profile.subject
    assert loaded.observations == profile.observations
    assert loaded.habits(2 * DAY, WEEK) == profile.habits(2 * DAY, WEEK)


def test_the_saved_record_is_json_someone_can_read_and_edit(tmp_path):
    """JSON and not pickle because this is a record about a person that they
    may want to read, correct, or delete a line from, and because unpickling
    is a promise to run whatever the file says."""
    profile = Profile("kolawole", [
        Observation(("jab", "jab"), "drop_guard", 14, 20, at=DAY, base=0.06),
        Observation(("low_kick",), "step_back", 6, 30, at=DAY, base=0.09),
    ])
    path = tmp_path / "kolawole.json"
    profile.save(path)

    payload = json.loads(path.read_text())
    assert payload["subject"] == "kolawole"
    assert payload["observations"][0]["context"] == ["jab", "jab"]
    assert payload["observations"][0]["hits"] == 14

    # Deleting a row by hand is a supported way to retract a finding.
    payload["observations"] = payload["observations"][:1]
    path.write_text(json.dumps(payload))
    assert len(Profile.load(path).observations) == 1


def test_an_empty_profile_round_trips_and_claims_nothing(tmp_path):
    path = tmp_path / "empty.json"
    Profile("nobody").save(path)
    loaded = Profile.load(path)
    assert loaded.observations == [] and loaded.sessions() == 0
    assert loaded.habits(0.0) == []


# --- bounding the file ----------------------------------------------------

def test_forgetting_before_a_date_drops_exactly_that():
    profile = Profile("f", [
        Observation(("jab",), "drop_guard", 14, 20, at=k * DAY, base=0.06)
        for k in range(6)
    ])
    profile.forget_before(3 * DAY)
    assert [o.at for o in profile.observations] == [3 * DAY, 4 * DAY, 5 * DAY]


def test_pruning_old_observations_does_not_change_the_verdict():
    """A record that only grows is a file that eventually stops being
    openable. Past six half lives an observation carries under 2% of its
    weight, so dropping the tail must not change which habits stand.

    It does not change them exactly. Measured on a 30 session record, the 23
    dropped sessions were together still worth about one trial, and the pooled
    count moved from 65/82 to 64/81. That is the honest size of the error this
    bound accepts: a trial, not a verdict."""
    profile = Profile("f")
    for k in range(30):
        profile.add(session((PLANTED,), seed=470 + k), at=k * WEEK)
    now = 29 * WEEK
    before = profile.habits(now, WEEK)
    kept = len(profile.observations)

    profile.prune(now, WEEK)
    after = profile.habits(now, WEEK)
    assert len(profile.observations) < kept, "nothing was dropped"
    assert [(h.context, h.then) for h in after] == [(h.context, h.then) for h in before]
    for old_habit, new_habit in zip(before, after):
        assert abs(new_habit.support - old_habit.support) <= 1
        assert new_habit.probability == pytest.approx(old_habit.probability, abs=0.01)
        assert new_habit.p_value < DEFAULT_FDR
    assert all(now - o.at <= FORGETTABLE_AFTER_HALF_LIVES * WEEK
               for o in profile.observations)


# --- reporting ------------------------------------------------------------

def test_habits_come_back_ranked_by_the_size_of_the_tell():
    profile = Profile("f")
    for k in range(4):
        profile.add(session((PLANTED, WEAK), seed=480 + k), at=k * DAY)
    found = profile.habits(3 * DAY, WEEK)
    assert len(found) >= 2
    assert [h.lift for h in found] == sorted([h.lift for h in found], reverse=True)
    assert found[0].context == ("jab", "jab") and found[0].then == "drop_guard"


def test_a_nonsense_half_life_is_refused_rather_than_guessed_at():
    """Units are the likely bug here, seconds against days, and a record that
    quietly picked a half life would hide it behind plausible output."""
    profile = Profile("f", [Observation(("jab",), "cross", 10, 20, 0.0, 0.2)])
    with pytest.raises(ValueError):
        profile.habits(0.0, half_life=0.0)


def test_a_standing_habit_describes_its_own_evidence_and_age():
    profile = Profile("f", [
        Observation(("jab", "jab"), "drop_guard", 14, 20, at=k * DAY, base=0.06)
        for k in range(3)
    ])
    habit = standing(profile, 2 * DAY, WEEK)
    assert habit is not None
    text = habit.describe(now=2 * DAY)
    assert "drop_guard" in text and "sessions" in text and "days ago" in text
    assert habit.effective_sessions <= habit.sessions
    assert habit.effective_sessions >= MIN_EFFECTIVE_SESSIONS
    assert habit.age(2 * DAY) == pytest.approx(0.0)
