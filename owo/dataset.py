"""Recorded gestures on disk, and gestures invented from nothing.

One decision here matters more than the rest: what gets stored is **raw
landmarks**, not feature vectors. Features are a guess about what the model
needs, and that guess will change. If the dataset held features, every
improvement to `features.py` would throw away all the recording work. Storing
landmarks means the features are re-derived on load and the data outlives the
guess.

`synthetic` exists so the whole training pipeline can be built and proved
before anyone records anything real. By the time a hand is recorded, the code
that learns from it has already been shown to work.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import features, synth


@dataclass
class Dataset:
    """Landmarks with labels. Features are derived, never stored."""

    landmarks: np.ndarray   # (n, 21, 3)
    is_left: np.ndarray     # (n,) bool
    labels: np.ndarray      # (n,) int
    classes: list[str]

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def features(self) -> np.ndarray:
        """(n, FEATURE_DIM), recomputed from landmarks every time."""
        return np.stack(
            [features.extract(m, bool(left)) for m, left in zip(self.landmarks, self.is_left)]
        )

    def counts(self) -> dict[str, int]:
        return {
            name: int((self.labels == i).sum()) for i, name in enumerate(self.classes)
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            landmarks=self.landmarks,
            is_left=self.is_left,
            labels=self.labels,
            classes=np.array(self.classes, dtype=object),
        )

    @classmethod
    def load(cls, path: str | Path) -> Dataset:
        blob = np.load(path, allow_pickle=True)
        return cls(
            landmarks=blob["landmarks"],
            is_left=blob["is_left"],
            labels=blob["labels"],
            classes=[str(c) for c in blob["classes"]],
        )

    def split(self, fraction: float = 0.2, seed: int = 0) -> tuple[Dataset, Dataset]:
        """Stratified train/validation split.

        Stratified rather than random because a random split of a small
        recording can easily leave a class out of validation entirely, and then
        the reported accuracy is measuring something other than what it claims.
        """
        rng = np.random.default_rng(seed)
        train_idx, val_idx = [], []
        for label in range(len(self.classes)):
            idx = np.flatnonzero(self.labels == label)
            rng.shuffle(idx)
            cut = max(1, int(round(len(idx) * fraction))) if len(idx) > 1 else 0
            val_idx.extend(idx[:cut])
            train_idx.extend(idx[cut:])
        return self.subset(np.array(train_idx, int)), self.subset(np.array(val_idx, int))

    def subset(self, index: np.ndarray) -> Dataset:
        return Dataset(
            landmarks=self.landmarks[index],
            is_left=self.is_left[index],
            labels=self.labels[index],
            classes=list(self.classes),
        )

    @classmethod
    def concat(cls, parts: list[Dataset]) -> Dataset:
        if not parts:
            raise ValueError("nothing to concatenate")
        classes = parts[0].classes
        for part in parts[1:]:
            if part.classes != classes:
                raise ValueError("cannot concatenate datasets with different classes")
        return cls(
            landmarks=np.concatenate([p.landmarks for p in parts]),
            is_left=np.concatenate([p.is_left for p in parts]),
            labels=np.concatenate([p.labels for p in parts]),
            classes=list(classes),
        )


def synthetic(
    per_class: int = 400,
    classes: list[str] | None = None,
    noise: float = 0.012,
    seed: int = 0,
    curl_jitter: float = 0.12,
    dropout: float = 0.0,
) -> Dataset:
    """A dataset of hands that never existed, for proving the pipeline.

    Every sample is a named pose put through a random rotation, translation,
    scale and jitter, plus a wobble on the finger curls so the class is a
    cloud rather than a point. The point is not realism, it is having a
    labelled problem that is genuinely learnable but not trivial.

    A caution about what results on this data mean. At default settings a small
    MLP reaches 100% here, and that number says the pipeline is wired up
    correctly, nothing more. Each class is a tight cloud around a pose someone
    designed to be distinctive, whereas real hands vary by person, get
    partially occluded, and arrive through a landmark estimator that is
    sometimes simply wrong. Turn up `noise`, `curl_jitter` and `dropout` to get
    a number that is actually informative.
    """
    classes = classes or sorted(synth.POSES)
    rng = np.random.default_rng(seed)

    marks, lefts, labels = [], [], []
    for label, name in enumerate(classes):
        base = synth.POSES[name]
        for _ in range(per_class):
            curl = {k: float(np.clip(v + rng.normal(0, curl_jitter), 0, 1))
                    for k, v in base.get("curl", {}).items()}
            hand = synth.synthetic_hand(
                curl=curl,
                spread=float(max(0.0, base.get("spread", 0.0) + rng.normal(0, 0.08))),
                pinch=float(np.clip(base.get("pinch", 0.0) + rng.normal(0, 0.05), 0, 1)),
            )
            turn = base.get("rotate")
            if turn is not None:
                hand = synth.place(hand, rotate=synth.rotation(*turn))
            is_left = bool(rng.random() < 0.5)
            if is_left:
                hand = hand.copy()
                hand[:, 0] *= -1.0
            placed = synth.place(
                hand,
                # Deliberately bounded. thumbs_up and thumbs_down are the same
                # shape a half turn apart, so rotating freely here would merge
                # two classes and inflate the accuracy of a model that had in
                # fact learned to ignore orientation.
                rotate=synth.rotation(*(rng.uniform(-0.45, 0.45, 3))),
                translate=tuple(rng.uniform(-2, 2, 3)),
                scale=float(rng.uniform(0.5, 2.0)),
                noise=noise,
                seed=int(rng.integers(1 << 30)),
            )
            if dropout > 0.0:
                # Landmarks the estimator lost: it does not omit them, it
                # guesses, and the guess can be far from the hand. Simulated by
                # displacing a few points badly rather than zeroing them.
                lost = rng.random(placed.shape[0]) < dropout
                if lost.any():
                    span = float(np.linalg.norm(placed.max(0) - placed.min(0)))
                    placed = placed.copy()
                    placed[lost] += rng.normal(0, 0.18 * span, (int(lost.sum()), 3))
            marks.append(placed)
            lefts.append(is_left)
            labels.append(label)

    return Dataset(
        landmarks=np.array(marks, dtype=np.float32),
        is_left=np.array(lefts, dtype=bool),
        labels=np.array(labels, dtype=np.int64),
        classes=list(classes),
    )
