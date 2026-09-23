"""The drill session as run from the terminal, with the camera faked."""

import numpy as np

from ika import drill_app
from test_drill import body


class Clock:
    """Wall time that advances one camera frame per read."""

    def __init__(self):
        self.now = 1_000_000.0

    def time(self):
        return self.now


class Camera:
    """A camera that gives `frames` frames and then the user presses Ctrl-C."""

    def __init__(self, clock, frames):
        self.clock, self.left = clock, frames

    def isOpened(self):
        return True

    def get(self, prop):
        return {3: 640.0, 4: 480.0}.get(prop, 0.0)

    def read(self):
        if self.left == 0:
            raise KeyboardInterrupt
        self.left -= 1
        self.clock.now += 1 / 30
        return True, np.zeros((480, 640, 3), dtype=np.uint8)

    def release(self):
        pass


class Pose:
    def __init__(self, *_a, **_k):
        pass

    def __enter__(self):
        return lambda _rgb, _stamp: [body()]

    def __exit__(self, *_exc):
        return False


def _run(tmp_path, monkeypatch, frames):
    clock = Clock()
    monkeypatch.setattr(drill_app.time, "time", clock.time)
    monkeypatch.setattr(drill_app, "_open", lambda _src: (Camera(clock, frames), False))
    monkeypatch.setattr(drill_app, "PoseTracker", Pose)
    return drill_app.run_drill(camera=0, profile=tmp_path / "profiles",
                               voice=False, quiet=True)


def test_ctrl_c_ends_a_camera_round_and_still_saves_it(tmp_path, monkeypatch, capsys):
    result = _run(tmp_path, monkeypatch, frames=30 * 8)
    out = capsys.readouterr().out
    assert "round over" in out
    assert "round: " in out
    assert result["seconds"] > 7
    assert (tmp_path / "profiles" / "me.json").exists()


def test_ctrl_c_before_calibration_saves_nothing(tmp_path, monkeypatch, capsys):
    _run(tmp_path, monkeypatch, frames=30)
    assert "round over" in capsys.readouterr().out
    assert not (tmp_path / "profiles" / "me.json").exists()


def test_an_empty_frame_is_explained_not_silent(tmp_path, monkeypatch, capsys):
    clock = Clock()
    monkeypatch.setattr(drill_app.time, "time", clock.time)
    monkeypatch.setattr(drill_app, "_open", lambda _src: (Camera(clock, 30 * 10), False))

    class Nobody(Pose):
        def __enter__(self):
            return lambda _rgb, _stamp: []

    monkeypatch.setattr(drill_app, "PoseTracker", Nobody)
    drill_app.run_drill(camera=0, profile=tmp_path / "profiles", voice=False, quiet=True)
    out = capsys.readouterr().out
    assert "no one in view" in out
    assert "never calibrated" in out
