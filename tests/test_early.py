"""Committing to a call before the movement finishes.

The interesting property is not accuracy. It is that accuracy and latency
trade against each other, and that the trade has an interior optimum: waiting
for certainty costs more time than the certainty is worth.
"""

import numpy as np
import pytest
import torch

from ika import trajectory
from ika.model import GestureNet
from ika.motion import NET_X
from ika.motion import features as motion_features
from ika.sequence import _stratified
from ika.tell.early import MIN_FRAMES, prefix, sweep, training_set, watch


@pytest.fixture(scope="module")
def trained():
    x, y, classes = training_set(per_class=120, feint_rate=0.35, seed=0)
    tr, va = _stratified(y, 0.2, 0)
    torch.manual_seed(0)
    model = GestureNet(x.shape[1], classes, hidden=(96, 48), dropout=0.25)
    model.fit_standardiser(x[tr])
    xt, yt = torch.tensor(x[tr]), torch.tensor(y[tr])
    xv, yv = torch.tensor(x[va]), torch.tensor(y[va])
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss = torch.nn.CrossEntropyLoss()
    best, state = 0.0, None
    for _ in range(120):
        model.train()
        order = torch.randperm(len(xt))
        for s in range(0, len(order), 64):
            b = order[s : s + 64]
            if len(b) < 2:
                continue
            opt.zero_grad(); loss(model(xt[b]), yt[b]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            a = float((model(xv).argmax(1) == yv).float().mean())
        if a > best:
            best, state = a, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    model.eval()
    assert best > 0.75, best
    return model, classes


def test_a_feint_actually_deceives():
    """If the opening does not lie, nothing about early commitment is hard.
    A left-then-right feint must read as leftward while it is still lying."""
    window = trajectory.feint("swipe_left", "swipe_right", switch=0.45, seed=0)
    early = motion_features(window[: max(MIN_FRAMES, int(len(window) * 0.4))])
    whole = motion_features(window)
    assert early[NET_X] < 0, "should look leftward early"
    assert whole[NET_X] > 0, "and end up rightward"


def test_a_feint_is_labelled_by_what_it_becomes():
    """What a fighter has to get right is the real move, not the fake."""
    window = trajectory.feint("swipe_left", "swipe_right", seed=1)
    assert motion_features(window)[NET_X] > 0


def test_prefix_refuses_to_judge_too_little():
    window = trajectory.make("swipe_left", seed=0)
    assert prefix(window, 0.01) is None
    assert len(prefix(window, 1.0)) == len(window)


def test_training_set_contains_feints():
    plain, _, _ = training_set(per_class=40, feint_rate=0.0, seed=0)
    with_feints, _, _ = training_set(per_class=40, feint_rate=0.5, seed=0)
    assert len(with_feints) > len(plain) * 1.2


def test_watch_reports_when_it_committed(trained):
    model, _ = trained
    window = trajectory.make("swipe_right", seed=5)
    result = watch(model, window, "swipe_right", threshold=0.5)
    assert result is not None
    assert MIN_FRAMES <= result.frames <= len(window)
    assert 0 < result.fraction <= 1.0
    assert result.latency == pytest.approx(window[result.frames - 1].at - window[0].at)


def test_a_higher_bar_means_a_later_call(trained):
    """The whole mechanism: certainty is bought with time."""
    model, classes = trained
    rows = sweep(model, classes, thresholds=(0.5, 0.95), per_class=25, feint_rate=0.35)
    assert rows[1]["median_latency"] > rows[0]["median_latency"]
    assert rows[1]["median_fraction"] > rows[0]["median_fraction"]


def test_feints_cost_accuracy_at_a_low_bar(trained):
    """Committing early against an opponent who lies is a real risk, and the
    measurement has to show it rather than average it away."""
    model, classes = trained
    honest = sweep(model, classes, thresholds=(0.5,), per_class=30, feint_rate=0.0)[0]
    deceptive = sweep(model, classes, thresholds=(0.5,), per_class=30, feint_rate=0.5)[0]
    assert honest["accuracy"] > deceptive["accuracy"] + 0.1


def test_waiting_for_certainty_is_not_free(trained):
    """At a high bar it is accurate but slow, and slow is what kills it."""
    model, classes = trained
    rows = sweep(model, classes, thresholds=(0.5, 0.99), per_class=25, feint_rate=0.35)
    assert rows[1]["accuracy"] > rows[0]["accuracy"]
    assert rows[1]["median_latency"] >= rows[0]["median_latency"] * 1.5


def test_recognition_errors_degrade_the_warnings():
    """Composition check: a misread action corrupts the context, so the habit
    lookup misses or matches the wrong one, exactly as it would live."""
    from ika.tell import Fighter, Habit, lead_times, mine, summarise
    from ika.tell.actions import VOCABULARY, names

    habits = (Habit(("jab", "jab"), "drop_guard", 0.70),)
    found = mine(names(Fighter(habits=habits, seed=7).sequence(2000)))
    test = Fighter(habits=habits, seed=99).sequence(1200)

    perfect = summarise(lead_times(test, found, 0.0, recognition=1.0,
                                   vocabulary=VOCABULARY, seed=0))
    sloppy = summarise(lead_times(test, found, 0.0, recognition=0.6,
                                  vocabulary=VOCABULARY, seed=0))
    assert perfect["precision"] > sloppy["precision"] + 0.1
