"""The command line surface: 400 lines that nothing was checking.

Parsing and preflight only. The commands themselves open cameras or train
models, but a broken flag, a default resolving against the wrong directory, or
a missing-model message that never prints are all silent failures that reach
the user before anything else does.
"""

from pathlib import Path

import pytest

from ika import cli


def parse(*argv):
    return cli.build_parser().parse_args(list(argv))


def test_every_command_is_reachable():
    expected = {"record", "train", "live", "train-dynamic", "compare",
                "tell", "early", "bindings"}
    parser = cli.build_parser()
    actions = [a for a in parser._actions if hasattr(a, "choices") and a.choices]
    available = set(actions[0].choices)
    assert expected <= available, expected - available


def test_a_command_is_required():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([])


def test_defaults_resolve_against_the_repo_not_the_cwd():
    """`ika live` used to work only from inside the project, because the
    checkpoint default was a relative path."""
    args = parse("live")
    assert Path(args.checkpoint).is_absolute()
    assert Path(args.checkpoint).parent.name == "checkpoints"
    assert Path(parse("record").out).is_absolute()
    assert Path(parse("compare").root).is_absolute()


def test_live_is_a_dry_run_unless_asked_otherwise():
    """The most important default in the project."""
    assert parse("live").live is False
    assert parse("live", "--live").live is True


def test_live_tracks_both_hands_by_default():
    assert parse("live").hands == 2
    assert parse("live", "--hands", "1").hands == 1


def test_the_dynamic_lane_is_off_unless_requested():
    """It fires about three times a minute at an idle hand."""
    assert parse("live").dynamic is None
    assert parse("live", "--dynamic").dynamic is not None


def test_the_terminal_is_the_default_surface():
    assert parse("live").window is False
    assert parse("live", "--window").window is True
    assert parse("live", "--stream").stream is True


def test_preflight_names_the_fix_when_the_model_is_absent(monkeypatch, tmp_path):
    """Checked before curses takes the screen, because the teardown wipes
    anything printed from inside the capture loop."""
    from ika.hands import ModelMissing

    monkeypatch.setattr("ika.hands.MODEL", tmp_path / "absent.task")
    with pytest.raises(ModelMissing) as caught:
        cli._preflight()
    assert "curl" in str(caught.value)


def test_preflight_passes_when_the_model_is_present():
    from ika.hands import MODEL

    if MODEL.exists():
        cli._preflight()


def test_bindings_runs_and_describes_everything(capsys):
    assert cli.main(["bindings"]) == 0
    printed = capsys.readouterr().out
    from ika import control

    for name in control.ACTIONS:
        assert name in printed


def test_a_missing_model_exits_cleanly_rather_than_traceback(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("ika.hands.MODEL", tmp_path / "absent.task")
    assert cli.main(["live"]) == 1
    assert "curl" in capsys.readouterr().out


def test_train_accepts_several_sessions():
    args = parse("train", "a.npz", "b.npz")
    assert args.data == ["a.npz", "b.npz"]


def test_tell_and_early_need_no_camera_or_model():
    """They run on simulation, so they must not be gated behind preflight."""
    assert parse("tell").null is False
    assert parse("early").feints == 0.35
