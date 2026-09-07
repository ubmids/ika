"""Habit mining: does it find planted habits, and does it invent any?

The second question matters more. A miner that reports tells a fighter does
not have is worse than no miner, because someone will act on it.
"""

import numpy as np
import pytest

from ika.tell import Fighter, Habit, lead_times, mine, summarise
from ika.tell.actions import names
from ika.tell.habits import benjamini_hochberg, binomial_tail

PLANTED = Habit(("jab", "jab"), "drop_guard", 0.70)


def stream_of(habits, length=2000, seed=0):
    return names(Fighter(habits=habits, seed=seed).sequence(length))


# --- the generator itself has to be trustworthy first ---------------------

def test_a_planted_habit_is_actually_in_the_stream():
    """If the fixture does not contain the habit, nothing downstream means
    anything. Same lesson as the synthetic pinch that never pinched."""
    stream = stream_of((PLANTED,), 4000, seed=3)
    after = [stream[i + 2] for i in range(len(stream) - 2)
             if stream[i] == "jab" and stream[i + 1] == "jab"]
    rate = after.count("drop_guard") / len(after)
    assert 0.6 < rate < 0.85, rate


def test_a_fighter_with_no_habits_has_no_dependence():
    stream = stream_of((), 4000, seed=4)
    after = [stream[i + 2] for i in range(len(stream) - 2)
             if stream[i] == "jab" and stream[i + 1] == "jab"]
    overall = stream.count("drop_guard") / len(stream)
    assert abs(after.count("drop_guard") / len(after) - overall) < 0.06


def test_actions_have_plausible_timing():
    stream = Fighter(habits=(PLANTED,), seed=5).sequence(200)
    durations = np.array([a.duration for a in stream])
    assert 0.10 < durations.min() and durations.max() < 0.55
    assert all(b.start >= a.end for a, b in zip(stream, stream[1:])), "actions overlap"


# --- the statistics -------------------------------------------------------

def test_binomial_tail_matches_hand_computed_values():
    assert binomial_tail(0, 10, 0.5) == pytest.approx(1.0)
    assert binomial_tail(10, 10, 0.5) == pytest.approx(0.5 ** 10)
    assert binomial_tail(6, 10, 0.5) == pytest.approx(0.376953125, rel=1e-9)


def test_binomial_tail_survives_large_counts():
    """Computed in log space because the naive form overflows well before the
    stream lengths this is used on."""
    value = binomial_tail(400, 3000, 0.1)
    assert 0.0 <= value <= 1.0 and np.isfinite(value)
    assert binomial_tail(2000, 3000, 0.1) == pytest.approx(0.0, abs=1e-12)


def test_benjamini_hochberg_keeps_the_obvious_and_drops_the_noise():
    p_values = [1e-9, 1e-8, 0.4, 0.6, 0.9]
    keep = benjamini_hochberg(p_values, fdr=0.05)
    assert keep[:2] == [True, True] and not any(keep[2:])


def test_benjamini_hochberg_rejects_everything_when_nothing_is_real():
    """Uniform p-values are what pure noise looks like."""
    rng = np.random.default_rng(0)
    keep = benjamini_hochberg(list(rng.uniform(size=500)), fdr=0.05)
    assert sum(keep) <= 2, f"{sum(keep)} of 500 uniform p-values survived"


# --- mining ---------------------------------------------------------------

def test_it_finds_the_planted_habit_at_the_right_strength():
    found = mine(stream_of((PLANTED,), 3000, seed=1))
    match = [f for f in found if f.context == ("jab", "jab") and f.then == "drop_guard"]
    assert match, [f.describe() for f in found]
    assert 0.55 < match[0].probability < 0.85
    assert match[0].lift > 5


def test_one_planted_habit_yields_one_finding():
    """A single habit generates a family of overlapping candidates: the same
    habit with irrelevant prefixes, and diluted shorter versions. Reporting
    the family as separate reads would badly mislead whoever acts on it."""
    for seed in (1, 2, 3, 4, 5):
        found = mine(stream_of((PLANTED,), 3000, seed=seed))
        assert len(found) <= 2, [f.describe() for f in found]


def test_it_finds_both_when_two_are_planted():
    habits = (PLANTED, Habit(("low_kick",), "step_back", 0.45))
    found = mine(stream_of(habits, 3000, seed=2))
    contexts = {(f.context, f.then) for f in found}
    assert (("jab", "jab"), "drop_guard") in contexts
    assert (("low_kick",), "step_back") in contexts


def test_it_mostly_invents_nothing_on_a_fighter_with_no_habits():
    """The honest headline. Measured over 30 null fighters: a false find turns
    up in well under a quarter of them."""
    invented = [len(mine(stream_of((), 3000, seed=1000 + s))) for s in range(30)]
    assert np.mean(np.array(invented) > 0) < 0.25, invented
    assert np.mean(invented) < 0.5


def test_a_short_stream_produces_nothing_rather_than_guesses():
    assert mine(stream_of((PLANTED,), 20, seed=0)) == []


def test_small_but_significant_tendencies_are_not_reported():
    """A tendency 1.05x above chance is not a tell even when p is tiny, and at
    long stream lengths plenty of those cross significance."""
    found = mine(stream_of((Habit(("jab",), "cross", 0.14),), 6000, seed=8))
    assert all(f.lift >= 1.35 for f in found)


# --- lead time ------------------------------------------------------------

def test_lead_time_shrinks_by_exactly_the_detection_cost():
    stream = Fighter(habits=(PLANTED,), seed=11).sequence(600)
    found = mine(names(stream))
    free = summarise(lead_times(stream, found, detection_cost=0.0))
    costly = summarise(lead_times(stream, found, detection_cost=0.20))
    assert free["warnings"] == costly["warnings"]
    assert free["median_lead"] - costly["median_lead"] == pytest.approx(0.20, abs=1e-6)


def test_predictions_beat_chance_by_a_wide_margin():
    train = Fighter(habits=(PLANTED,), seed=7).sequence(2000)
    found = mine(names(train))
    unseen = Fighter(habits=(PLANTED,), seed=99).sequence(1200)
    result = summarise(lead_times(unseen, found, detection_cost=0.0))
    assert result["warnings"] > 20
    # Base rate of drop_guard is about 6%; the warnings must do far better.
    assert result["precision"] > 0.4, result


def test_a_slow_pipeline_makes_correct_predictions_useless():
    """The finding this experiment exists to produce. Gaps between actions are
    around 260ms, so a 400ms recognition delay lands every warning too late
    however good the prediction was."""
    train = Fighter(habits=(PLANTED,), seed=7).sequence(2000)
    found = mine(names(train))
    unseen = Fighter(habits=(PLANTED,), seed=99).sequence(1200)
    fast = summarise(lead_times(unseen, found, detection_cost=0.05))
    slow = summarise(lead_times(unseen, found, detection_cost=0.40))
    assert fast["precision"] == pytest.approx(slow["precision"]), "same predictions"
    assert fast["useful"] > 0.4 and slow["useful"] < 0.15


def test_summarise_handles_an_empty_run():
    assert summarise([])["warnings"] == 0
