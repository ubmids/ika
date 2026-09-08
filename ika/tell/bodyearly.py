"""Early commitment on bodies, and whether an occluded camera can still read.

The hand lane established the rule: commit before the movement ends, because
the latency cost of certainty is worse than the accuracy cost of speed. This
asks the same question of a body, and adds one the hand lane never faced.

A webcam cannot see feet. Measured over 400 real photographs, knees were
visible in 4% and ankles in none, so the synthetic bodies here are generated
with their legs marked unseen by default. That is not pessimism, it is the
camera people actually have, and training on fully visible bodies would teach
a model to rely on information it will never receive.

Which makes one action a genuine test of the whole approach. `crouch` is
distinguishable only by knee angle. If the pipeline still calls it, something
is wrong with the experiment. If it cannot, that is an honest limit worth
knowing before anyone builds on it.
"""

from __future__ import annotations

import numpy as np

from .. import bodyaction
from ..bodymotion import features as body_features
from ..model import GestureNet
from .early import FRACTIONS, MIN_FRAMES, Commitment, prefix, watch


def training_set(
    per_class: int = 250,
    classes: list[str] | None = None,
    fractions: tuple[float, ...] = FRACTIONS,
    feint_rate: float = 0.3,
    occlude_legs: bool = True,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Body movement prefixes and labels, at every prefix length."""
    classes = classes or list(bodyaction.ACTIONS)
    rng = np.random.default_rng(seed)
    vectors, labels = [], []

    def add(window, label):
        for fraction in fractions:
            part = prefix(window, fraction)
            if part is not None and len(part) >= 2:
                vectors.append(body_features(part))
                labels.append(label)

    for label, name in enumerate(classes):
        for _ in range(per_class):
            add(bodyaction.make(name, seed=int(rng.integers(1 << 30)),
                                occlude_legs=occlude_legs), label)

    for _ in range(int(per_class * len(classes) * feint_rate)):
        looks_like, becomes = bodyaction.FEINTS[rng.integers(len(bodyaction.FEINTS))]
        add(
            bodyaction.feint(looks_like, becomes,
                             switch=float(rng.uniform(0.3, 0.6)),
                             seed=int(rng.integers(1 << 30)),
                             occlude_legs=occlude_legs),
            classes.index(becomes),
        )
    return np.stack(vectors), np.array(labels, dtype=np.int64), classes


def sweep(
    model: GestureNet,
    classes: list[str],
    thresholds: tuple[float, ...] = (0.5, 0.7, 0.85, 0.95, 0.99),
    per_class: int = 40,
    feint_rate: float = 0.3,
    occlude_legs: bool = True,
    seed: int = 99,
) -> list[dict]:
    """The accuracy-versus-latency curve for bodies, plus a per-action breakdown.

    The breakdown is the point: an overall figure would hide that one action is
    invisible to this camera, and that is exactly the finding worth having.
    """
    rng = np.random.default_rng(seed)
    trials = [
        (name, bodyaction.make(name, seed=int(rng.integers(1 << 30)),
                               occlude_legs=occlude_legs))
        for name in classes
        for _ in range(per_class)
    ]
    for _ in range(int(per_class * len(classes) * feint_rate)):
        looks_like, becomes = bodyaction.FEINTS[rng.integers(len(bodyaction.FEINTS))]
        trials.append((becomes, bodyaction.feint(
            looks_like, becomes, switch=float(rng.uniform(0.3, 0.6)),
            seed=int(rng.integers(1 << 30)), occlude_legs=occlude_legs)))

    rows = []
    for threshold in thresholds:
        calls = [
            (name, watch(model, window, name, threshold, feature_fn=body_features))
            for name, window in trials
        ]
        made = [(n, c) for n, c in calls if c is not None]
        if not made:
            rows.append({"threshold": threshold, "committed": 0.0, "accuracy": 0.0,
                         "median_latency": 0.0, "per_action": {}})
            continue
        correct = [c for _, c in made if c.correct]
        per_action: dict[str, float] = {}
        for name in classes:
            own = [c for n, c in made if n == name]
            per_action[name] = (
                sum(c.correct for c in own) / len(own) if own else float("nan")
            )
        rows.append({
            "threshold": threshold,
            "committed": len(made) / len(trials),
            "accuracy": len(correct) / len(made),
            "median_latency": float(np.median([c.latency for _, c in made])),
            "median_fraction": float(np.median([c.fraction for _, c in made])),
            "per_action": per_action,
        })
    return rows
