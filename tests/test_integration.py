"""The whole stack, driven by invented hands.

Each layer has its own tests. This one checks they are wired together in the
right order and orientation, which is the failure the unit tests cannot see:
every part correct, the assembly wrong.

No camera, no recording, no checkpoint on disk. A model is trained inline on
synthetic data, then fed a scripted sequence of poses as though they arrived
from a webcam.
"""

import numpy as np
import pytest

from ika import control, features, synth
from ika.dataset import synthetic
from ika.machine import GestureMachine
from ika.train import train


@pytest.fixture(scope="module")
def trained():
    data = synthetic(per_class=120, seed=1)
    model, report = train(data, epochs=60, verbose=False)
    assert report["accuracy"] > 0.9, f"fixture model is too weak: {report['accuracy']:.1%}"
    return model


def frames_of(name: str, count: int, seed: int = 0):
    """A pose held for `count` frames, jittered as a real hand would be."""
    rng = np.random.default_rng(seed)
    hand = synth.pose(name)
    for _ in range(count):
        placed = synth.place(
            hand,
            rotate=synth.rotation(*rng.uniform(-0.15, 0.15, 3)),
            translate=tuple(rng.uniform(-0.5, 0.5, 3)),
            scale=float(rng.uniform(0.85, 1.15)),
            noise=0.008,
            seed=int(rng.integers(1 << 30)),
        )
        yield features.extract(placed)


def test_the_model_recognises_its_own_poses(trained):
    """The join between synth, features and model, in the trivial direction."""
    for name in synth.POSES:
        votes = []
        for vector in frames_of(name, 12, seed=3):
            best, _, _ = trained.predict(vector)
            votes.append(trained.classes[best])
        winner = max(set(votes), key=votes.count)
        assert winner == name, f"{name} was mostly read as {winner}"


def test_a_scripted_session_produces_the_right_actions(trained):
    """The real thing: engage, perform two gestures, and check what happened."""
    machine = GestureMachine(trained.classes, dwell_frames=4, smoothing=0.5, cooldown=0.2)
    controller = control.Controller()          # dry run
    performed, now = [], 0.0

    def play(name, count):
        nonlocal now
        for vector in frames_of(name, count, seed=7):
            _, _, probabilities = trained.predict(vector)
            for intent in machine.update(probabilities, now):
                if intent.kind == "fired" and control.perform(intent.gesture, controller):
                    performed.append(intent.gesture)
            now += 1 / 30

    play("open_palm", 12)                      # arm it
    assert machine.engaged, "never engaged"

    play("rest", 8)
    play("fist", 12)                           # play / pause
    play("rest", 8)
    play("peace", 12)                          # next desktop
    play("rest", 8)

    assert performed == ["fist", "peace"], performed
    assert controller.log == ["tap media_play_pause", "hotkey ctrl+right"]


def test_a_hand_that_never_settles_does_nothing(trained):
    """Waving about is not a command. This is the property that makes it safe
    to leave running while you use your hands for something else."""
    machine = GestureMachine(trained.classes, dwell_frames=5, smoothing=0.6)
    controller = control.Controller()
    now, fired = 0.0, []

    for i in range(150):
        name = ["fist", "point", "peace", "open_palm", "l_shape"][i % 5]
        vector = next(frames_of(name, 1, seed=i))
        _, _, probabilities = trained.predict(vector)
        for intent in machine.update(probabilities, now):
            if intent.kind == "fired":
                fired.append(intent.gesture)
                control.perform(intent.gesture, controller)
        now += 1 / 30

    assert fired == [], f"a flailing hand fired {fired}"


def test_nothing_fires_while_disengaged(trained):
    """End to end version of the engagement guarantee."""
    machine = GestureMachine(trained.classes, dwell_frames=4)
    controller = control.Controller()
    now = 0.0
    for name in ("fist", "peace", "thumbs_up", "l_shape"):
        for vector in frames_of(name, 20, seed=11):
            _, _, probabilities = trained.predict(vector)
            for intent in machine.update(probabilities, now):
                if intent.kind == "fired":
                    control.perform(intent.gesture, controller)
            now += 1 / 30
    assert controller.log == []


def test_a_checkpoint_survives_a_round_trip(tmp_path, trained):
    """Saving and loading must not change a single prediction, or the model you
    ship is not the model you measured."""
    from ika.model import GestureNet

    path = tmp_path / "model.pt"
    trained.save(path)
    reloaded = GestureNet.load(path)

    assert reloaded.classes == trained.classes
    for vector in frames_of("point", 20, seed=5):
        a = trained.predict(vector)
        b = reloaded.predict(vector)
        assert a[0] == b[0]
        assert np.allclose(a[2], b[2], atol=1e-6)
