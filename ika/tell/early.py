"""Committing to a call before the movement finishes.

`lead.py` found the constraint: the median gap between one action ending and
the next beginning is 263 ms, and confirming a gesture takes 250 ms, so a
warning that waits for the movement to complete arrives with nothing to spare.

The way out is not a faster model. It is refusing to wait. A punch is
recognisable from its opening, long before it lands, so the classifier is
trained on *prefixes*: the first 20%, 30%, half of an action, each labelled
with what the action turned out to be. At inference it watches a movement
unfold and commits the moment it is confident enough.

That trade is the whole product decision, and it has two sides that pull
against each other. Commit early and you are guessing; commit late and the
answer is useless. Neither accuracy nor latency alone says anything, so what
this module produces is the curve between them, and `ika early` prints it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..model import GestureNet
from ..motion import MOTION_DIM, Sample
from ..motion import features as motion_features
from .. import trajectory

# Below this the features are noise: two or three samples cannot establish a
# direction, let alone a speed.
MIN_FRAMES = 4

# Prefix lengths to train on. Dense at the short end, because that is where
# the decision actually gets made and where the model needs the most help.
FRACTIONS = (0.25, 0.35, 0.45, 0.55, 0.7, 0.85, 1.0)


def prefix(window: list[Sample], fraction: float) -> list[Sample] | None:
    """The first `fraction` of a movement, or None if too little to judge."""
    count = int(round(len(window) * fraction))
    return window[:count] if count >= MIN_FRAMES else None


def training_set(
    per_class: int = 400,
    classes: list[str] | None = None,
    fractions: tuple[float, ...] = FRACTIONS,
    feint_rate: float = 0.35,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Features and labels over prefixes of every length.

    One model across all prefix lengths rather than one per length: at
    inference we do not know how far through a movement we are, so a model
    that had to be told would be useless.

    Feints are in the training set, labelled by what they become. Without them
    the model learns that the opening of a movement always tells the truth,
    which is both false and dangerous: trained on honest movements only it
    reported 99% accuracy from four frames and had no notion that an opening
    could lie. With them it learns to stay uncertain while a movement is
    genuinely ambiguous, which is what makes a confidence threshold mean
    anything at all.
    """
    classes = classes or list(trajectory.DYNAMIC_GESTURES)
    rng = np.random.default_rng(seed)
    vectors, labels = [], []

    def add(window, label):
        for fraction in fractions:
            part = prefix(window, fraction)
            if part is not None:
                vectors.append(motion_features(part))
                labels.append(label)

    for label, name in enumerate(classes):
        for _ in range(per_class):
            add(trajectory.make(name, seed=int(rng.integers(1 << 30))), label)

    for _ in range(int(per_class * len(classes) * feint_rate)):
        looks_like, becomes = trajectory.FEINTS[rng.integers(len(trajectory.FEINTS))]
        add(
            trajectory.feint(looks_like, becomes, switch=float(rng.uniform(0.3, 0.6)),
                             seed=int(rng.integers(1 << 30))),
            classes.index(becomes),
        )

    return np.stack(vectors), np.array(labels, dtype=np.int64), classes


@dataclass(frozen=True)
class Commitment:
    """One decision: what was called, when, and whether it was right."""

    predicted: str
    actual: str
    confidence: float
    frames: int
    fraction: float       # how much of the movement had been seen
    latency: float        # seconds from the movement starting

    @property
    def correct(self) -> bool:
        return self.predicted == self.actual


def watch(
    model: GestureNet,
    window: list[Sample],
    actual: str,
    threshold: float,
    fps: float = 30.0,
) -> Commitment | None:
    """Replay a movement frame by frame and commit as soon as we dare.

    Returns None if confidence never reached the threshold, which is a real
    outcome rather than a failure: staying silent about a movement you cannot
    read is correct behaviour, and the alternative is a system that guesses.
    """
    for count in range(MIN_FRAMES, len(window) + 1):
        part = window[:count]
        index, confidence, _ = model.predict(motion_features(part))
        if confidence >= threshold:
            return Commitment(
                predicted=model.classes[index],
                actual=actual,
                confidence=confidence,
                frames=count,
                fraction=count / len(window),
                latency=part[-1].at - window[0].at,
            )
    return None


def sweep(
    model: GestureNet,
    classes: list[str],
    thresholds: tuple[float, ...] = (0.5, 0.7, 0.85, 0.95, 0.99),
    per_class: int = 60,
    feint_rate: float = 0.35,
    seed: int = 99,
    fps: float = 30.0,
) -> list[dict]:
    """The accuracy-versus-latency curve, on movements never trained on.

    Feints appear in the test set at the same rate as in training, so accuracy
    here is accuracy in the presence of deception rather than against an
    opponent who never lies.
    """
    rng = np.random.default_rng(seed)
    trials = [
        (name, trajectory.make(name, seed=int(rng.integers(1 << 30))))
        for name in classes
        for _ in range(per_class)
    ]
    for _ in range(int(per_class * len(classes) * feint_rate)):
        looks_like, becomes = trajectory.FEINTS[rng.integers(len(trajectory.FEINTS))]
        trials.append((becomes, trajectory.feint(
            looks_like, becomes, switch=float(rng.uniform(0.3, 0.6)),
            seed=int(rng.integers(1 << 30)))))

    rows = []
    for threshold in thresholds:
        commitments = [
            c for c in (watch(model, w, name, threshold, fps) for name, w in trials)
            if c is not None
        ]
        if not commitments:
            rows.append({"threshold": threshold, "committed": 0.0, "accuracy": 0.0,
                         "median_latency": 0.0, "median_fraction": 0.0})
            continue
        correct = [c for c in commitments if c.correct]
        rows.append({
            "threshold": threshold,
            "committed": len(commitments) / len(trials),
            "accuracy": len(correct) / len(commitments),
            "median_latency": float(np.median([c.latency for c in commitments])),
            "median_fraction": float(np.median([c.fraction for c in commitments])),
            "median_frames": float(np.median([c.frames for c in commitments])),
        })
    return rows
