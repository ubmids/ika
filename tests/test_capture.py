"""The three camera-driven surfaces: stream, recorder, window app.

None can be run without a camera, but each holds pure logic that was
untested, and the recorder in particular has a rule worth pinning: it stores
only the most confident hand, because capturing both would label whatever your
other hand happens to be doing as the gesture you are performing.
"""

import inspect

import numpy as np
import pytest

from ika import app, record, stream
from ika.dataset import Dataset


# --- stream ---------------------------------------------------------------

def test_stream_is_a_dry_run_by_default():
    assert inspect.signature(stream.run_stream).parameters["live_control"].default is False


def test_stream_detects_loosely_on_purpose():
    """Missing the hand entirely is a far worse failure than tracking it
    loosely, since everything downstream can filter a noisy landmark but
    nothing can recover an absent one."""
    source = inspect.getsource(stream.run_stream)
    assert "detection_confidence=0.3" in source


def test_stream_can_be_time_bounded():
    """So it can be run from a pipe without running forever."""
    assert "seconds" in inspect.signature(stream.run_stream).parameters


def test_stream_flushes_so_a_reader_sees_it_live():
    assert "flush=True" in inspect.getsource(stream._say)


# --- recorder -------------------------------------------------------------

def test_the_recorder_captures_one_hand_only():
    """Capturing both would label the other hand's idle pose as the gesture,
    which is a quiet way to poison a dataset."""
    source = inspect.getsource(record.run_recorder)
    assert "max_hands=1" in source
    assert "max(hands, key=lambda h: h.score)" in source


def test_the_recorder_nags_you_to_move():
    """The variation captured is the variation the model will tolerate."""
    assert len(record.PROMPTS) >= 4
    joined = " ".join(record.PROMPTS).lower()
    assert "rotate" in joined and "closer" in joined


def test_the_recorder_captures_over_seconds_not_one_frame():
    assert inspect.signature(record.run_recorder).parameters["seconds"].default >= 2.0


def test_saving_writes_a_loadable_dataset(tmp_path):
    """The save path is the one part that can be exercised without a camera,
    and it is the part that would lose an afternoon of recording."""
    rng = np.random.default_rng(0)
    marks = [rng.random((21, 3)).astype(np.float32) for _ in range(30)]
    lefts = [bool(i % 2) for i in range(30)]
    labels = [i % 3 for i in range(30)]
    classes = ["rest", "fist", "point"]

    path = tmp_path / "session.npz"
    record._save(path, marks, lefts, labels, classes)
    assert path.exists()

    loaded = Dataset.load(path)
    assert len(loaded) == 30
    assert loaded.classes == classes
    assert loaded.landmarks.shape == (30, 21, 3)
    assert loaded.features.shape[0] == 30
    assert np.isfinite(loaded.features).all()
    assert loaded.counts() == {"rest": 10, "fist": 10, "point": 10}


def test_a_saved_session_can_be_split_and_trained(tmp_path):
    """End to end for the one thing only a person can produce."""
    from ika import synth
    from ika.train import train

    marks, lefts, labels = [], [], []
    classes = ["fist", "point", "open_palm"]
    rng = np.random.default_rng(1)
    for label, name in enumerate(classes):
        for _ in range(40):
            hand = synth.place(synth.pose(name), noise=0.01,
                               seed=int(rng.integers(1 << 30)))
            marks.append(hand.astype(np.float32))
            lefts.append(False)
            labels.append(label)

    path = tmp_path / "s.npz"
    record._save(path, marks, lefts, labels, classes)
    _, report = train(Dataset.load(path), epochs=40, verbose=False)
    assert report["accuracy"] > 0.8, report["accuracy"]


# --- window app -----------------------------------------------------------

def test_the_window_app_is_a_dry_run_by_default():
    assert inspect.signature(app.run_live).parameters["live_control"].default is False


def test_the_pinch_has_hysteresis():
    """A single threshold sits exactly where landmark noise flickers, and a
    flickering click is worse than no click."""
    assert app.PINCH_CLOSE < app.PINCH_OPEN
    assert app.PINCH_OPEN - app.PINCH_CLOSE > 0.1


def test_the_dynamic_lane_is_off_in_the_window_app():
    assert inspect.signature(app.run_live).parameters["dynamic_checkpoint"].default is None
