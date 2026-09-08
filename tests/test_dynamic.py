"""The shipped dynamic classifier, and the reason it is the shipped one."""

import numpy as np

from ika import trajectory
from ika.dynamic import train_motion
from ika.motion import MOTION_DIM


def test_it_learns_the_movements():
    vectors, labels, classes, _ = trajectory.realistic_dataset(per_class=80, seed=0)
    assert vectors.shape[1] == MOTION_DIM
    _, report = train_motion(vectors, labels, classes, epochs=120)
    assert report["balanced"] > 0.75, report["balanced"]


def test_it_is_judged_on_balanced_recall_not_plain_accuracy():
    """The honest sliding-window dataset is dominated by `none`, which is
    correct for a live system, so plain accuracy would flatter a model that
    predicted nothing forever."""
    vectors, labels, classes, _ = trajectory.realistic_dataset(per_class=80, seed=0)
    _, report = train_motion(vectors, labels, classes, epochs=120)
    assert "balanced" in report and "accuracy" in report
    counts = np.bincount(labels, minlength=len(classes))
    assert counts.max() > 2 * np.median(counts), "test assumes a skewed dataset"


def test_the_idle_class_is_the_one_that_matters():
    """Every point of idle recall below 100% is a firing nobody asked for, and
    it is the weakest class, so the report has to expose it."""
    vectors, labels, classes, _ = trajectory.realistic_dataset(per_class=80, seed=0)
    _, report = train_motion(vectors, labels, classes, epochs=120)
    assert "none" in report["recall"]


def test_it_is_smaller_than_the_sequence_model_it_replaced():
    """A data-efficiency sweep found the GRU never beat these 14 hand-picked
    features and lost when data was scarce, so the small model ships."""
    import torch

    from ika.sequence import SequenceNet

    vectors, labels, classes, _ = trajectory.realistic_dataset(per_class=40, seed=0)
    model, _ = train_motion(vectors, labels, classes, epochs=20)
    mlp = sum(p.numel() for p in model.parameters())
    gru = sum(p.numel() for p in SequenceNet(classes).parameters())
    assert mlp < gru
