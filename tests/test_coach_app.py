"""Coach mode as run from the terminal, with the camera faked and nothing spoken."""

import json
import time

from ika import cli, coach_app, drill_app
from ika.profile import Profile
from ika.tell.habits import Finding
from test_drill_app import Camera, Clock, Pose


def _run(tmp_path, monkeypatch, frames, **kw):
    clock = Clock()
    monkeypatch.setattr(coach_app.time, "time", clock.time)
    monkeypatch.setattr(coach_app, "_open", lambda _src: (Camera(clock, frames), False))
    monkeypatch.setattr(coach_app, "PoseTracker", Pose)
    kw.setdefault("seconds", 12.0)
    return coach_app.run_coach(camera=0, profile=tmp_path / "profiles", voice=False,
                               quiet=True, seed=1, **kw)


def _known_habit(tmp_path, combo="1-2"):
    """A profile where the right hand dropping after the call `combo` stands."""
    record = Profile("me")
    now = time.time()
    for days in (2, 1):
        record.add([Finding((f"coach_{combo}",), "guard_down_right", 20, 16, 0.8, 0.1, 1e-6)],
                   at=now - days * 86400)
    store = tmp_path / "profiles" / "me.json"
    store.parent.mkdir(parents=True)
    record.save(store)
    return store


def test_a_round_calls_combos_and_saves_a_coach_session(tmp_path, monkeypatch, capsys):
    result = _run(tmp_path, monkeypatch, frames=30 * 25)
    out = capsys.readouterr().out
    assert "[box]" in out and "[time]" in out
    assert ">> " in out
    # Nobody threw anything: every call is unanswered, and said so.
    assert result["reps"] and not any(r.answered for r in result["reps"])
    assert "with nothing thrown in time" in out
    record = json.loads((tmp_path / "profiles" / "me.json").read_text())
    row, = record["log"]
    assert row["mode"] == "coach"
    assert row["counts"]["calls"] == len(result["reps"])
    assert row["counts"]["answered"] == 0


def test_the_round_stops_on_its_own_after_the_last_call(tmp_path, monkeypatch, capsys):
    result = _run(tmp_path, monkeypatch, frames=30 * 60)
    out = capsys.readouterr().out
    assert "round over" not in out           # it ended, not Ctrl-C
    assert result["seconds"] < 25


def test_a_known_habit_is_reminded_in_its_call(tmp_path, monkeypatch, capsys):
    _known_habit(tmp_path)
    _run(tmp_path, monkeypatch, frames=30 * 25, combos=["1-2", "3-2"])
    out = capsys.readouterr().out
    assert ">> ONE, TWO... RIGHT HAND UP" in out
    assert ">> THREE, TWO\n" in out


def test_known_habits_are_drilled_without_being_asked(tmp_path, monkeypatch, capsys):
    _known_habit(tmp_path, combo="2-3-2")
    result = _run(tmp_path, monkeypatch, frames=30 * 25)
    assert "2-3-2" in result["combos"]


def test_ctrl_c_before_calibration_saves_nothing(tmp_path, monkeypatch, capsys):
    _run(tmp_path, monkeypatch, frames=30)
    assert "round over" in capsys.readouterr().out
    assert not (tmp_path / "profiles" / "me.json").exists()


def test_the_drill_does_not_watch_for_coach_habits(tmp_path, monkeypatch, capsys):
    """A coach habit is keyed on a call, which never happens in a drill."""
    _known_habit(tmp_path)
    clock = Clock()
    monkeypatch.setattr(drill_app.time, "time", clock.time)
    monkeypatch.setattr(drill_app, "_open", lambda _src: (Camera(clock, 30 * 8), False))
    monkeypatch.setattr(drill_app, "PoseTracker", Pose)
    drill_app.run_drill(camera=0, profile=tmp_path / "profiles", voice=False, quiet=True)
    out = capsys.readouterr().out
    assert "watching for" not in out
    row = json.loads((tmp_path / "profiles" / "me.json").read_text())["log"][-1]
    assert row["mode"] == "drill"
    assert not any(key.startswith("coach_") for key in row["habits"])


def test_history_shows_coach_and_drill_sessions_apart(tmp_path, monkeypatch, capsys):
    store = _known_habit(tmp_path)
    record = Profile.load(store)
    now = time.time()
    record.record_session(now - 3600, 180.0, {"punch_left": 30, "guard_down_left": 2})
    record.log[-1].pop("mode")                 # a row written before coach mode
    habit = Finding(("coach_1-2",), "guard_down_right", 0, 0, 0.0, 0.0, 1.0)
    record.record_session(now, 180.0, {"calls": 47, "answered": 40}, [(habit, 9, 15)],
                          mode="coach")
    record.save(store)
    assert cli.main(["history", "--profile", str(tmp_path / "profiles")]) == 0
    out = capsys.readouterr().out
    assert "drill    30 punches" in out
    assert "coach    47 calls" in out and "40 answered" in out
    assert "after a called combo (coach):" in out
    assert "called 1-2 -> right hand down" in out
    assert "60%" in out
    assert "the call 1-2" in out and "ends with a reminder" in out


def test_coach_is_a_command_with_drills_options():
    args = cli.build_parser().parse_args(["coach", "round.mp4", "--silent", "--quiet",
                                          "--rounds", "2", "--seed", "4",
                                          "--combos", "1-2,2-3-2"])
    assert args.func is cli._coach
    assert args.source == "round.mp4" and args.silent and args.quiet
    assert args.rounds == 2 and args.seed == 4 and args.seconds == 180.0
