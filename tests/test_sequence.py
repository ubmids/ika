"""The sequence model and its input encoding."""

import numpy as np

from ika import trajectory
from ika.sequence import FRAME_DIM, STEPS, SequenceNet, sequence_tensor, train_sequence


def test_windows_of_any_length_become_the_same_shape():
    """Gestures have genuinely different durations; the model needs one shape.
    Resampled rather than padded, so trailing zeros are never mistaken for
    part of the movement."""
    for name in ("snap", "pinch_drag", "swipe_left"):
        window = trajectory.make(name, seed=2)
        tensor = sequence_tensor(window)
        assert tensor.shape == (STEPS, FRAME_DIM)
        assert np.isfinite(tensor).all()


def test_position_is_relative_to_the_start_of_the_window():
    """Absolute frame position is not a gesture, so the first step is zero."""
    tensor = sequence_tensor(trajectory.make("swipe_right", seed=4))
    assert abs(tensor[0, 0]) < 1e-6 and abs(tensor[0, 1]) < 1e-6
    assert tensor[-1, 0] > 0.5, "and a rightward swipe should end to the right"


def test_it_learns_the_synthetic_gestures():
    """Learns the movements well above chance, which for 7 classes is 14%.

    The bar sits at 75% rather than 90% because the `none` class was
    deliberately enriched: idle now wanders with momentum through six poses
    instead of drifting gently through two. That made the problem much harder
    and much more honest, and it dropped this from ~100% to the mid-80s.
    """
    _, labels, classes, windows = trajectory.realistic_dataset(per_class=60, seed=0)
    _, report = train_sequence(windows, labels, classes, epochs=80, seed=0)
    assert report["accuracy"] > 0.75, report["accuracy"]


def test_a_checkpoint_round_trips_exactly(tmp_path):
    _, labels, classes, windows = trajectory.realistic_dataset(per_class=30, seed=0)
    model, _ = train_sequence(windows, labels, classes, epochs=30, seed=0)
    path = tmp_path / "dynamic.pt"
    model.save(path)
    reloaded = SequenceNet.load(path)
    tensor = sequence_tensor(windows[0])
    a, b = model.predict(tensor), reloaded.predict(tensor)
    assert a[0] == b[0] and np.allclose(a[2], b[2], atol=1e-6)


def test_inference_is_cheap_enough_for_the_capture_loop():
    """It runs every frame beside landmark detection, so it has to be nearly
    free or it costs frames."""
    import time

    _, labels, classes, windows = trajectory.realistic_dataset(per_class=20, seed=0)
    model, _ = train_sequence(windows, labels, classes, epochs=10, seed=0)
    tensor = sequence_tensor(windows[0])
    model.predict(tensor)
    start = time.time()
    for _ in range(100):
        model.predict(tensor)
    per_call = (time.time() - start) / 100
    assert per_call < 0.005, f"{per_call*1000:.2f} ms per window is too slow"
