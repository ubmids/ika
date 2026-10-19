"""The full-screen round view, rendered to text without a terminal.

`live.render` is pure, so every layout question is checked here as text: it
fills the terminal exactly, never runs past the right edge, and keeps the
clock and the call readable from a narrow 80x24 up to a wide window.
"""

import io
import math

import pytest

from ika import coach_app, drill_app, live
from ika.live import LiveState, big, clock_text, figure, guard_meter, render, ticks
from test_drill import body
from test_drill_app import Camera, Clock, Pose

SIZES = [(40, 12), (60, 20), (80, 24), (100, 30), (120, 40), (200, 60)]


def _state(**kw):
    base = dict(mode="coach", phase="box", round_no=1, rounds=3, remaining=161.0,
                clock=50.0, big_text="1-2-3", words="one, two, three... right hand up",
                fresh=True, guard={"left": -0.05, "right": -0.31},
                punches=[(45.0, "left"), (46.0, "right"), (49.5, "right")],
                punch_count=37, body=body(), notes=["calibrated.", "so far: nothing"],
                fps=30.0)
    base.update(kw)
    return LiveState(**base)


@pytest.mark.parametrize("cols,rows", SIZES)
def test_the_screen_fills_the_terminal_and_never_runs_past_it(cols, rows):
    lines = render(_state(), cols, rows)
    assert len(lines) == rows
    assert all(len(line) <= cols for line in lines)


def test_a_tiny_terminal_still_shows_the_clock_in_plain_text():
    assert "2:41" in "\n".join(render(_state(remaining=161.0), 40, 12))


@pytest.mark.parametrize("cols,rows", SIZES[1:])
def test_the_clock_is_drawn_large_from_60_by_20_up(cols, rows):
    text = "\n".join(render(_state(remaining=161.0), cols, rows))
    sizes = [s for s in ("b3", "b2", "b1", "half") if all(line in text for line in big("2:41", s))]
    assert sizes, "the round clock is not in large type"


def test_the_clock_grows_with_the_terminal():
    def size(cols, rows):
        text = "\n".join(render(_state(), cols, rows))
        return next(s for s in ("b3", "b2", "b1", "half")
                    if all(line in text for line in big("2:41", s)))
    assert live.SIZES.index(size(200, 60)) > live.SIZES.index(size(80, 24))


def test_the_call_is_large_and_in_words_at_80_by_24():
    text = "\n".join(render(_state(), 80, 24))
    assert all(line in text for line in big("1-2-3", "half"))
    assert "ONE, TWO, THREE... RIGHT HAND UP" in text


def test_a_call_too_long_for_large_type_steps_down_rather_than_overflowing():
    state = _state(big_text="right hand up and keep it there please")
    lines = render(state, 80, 24)
    assert all(len(line) <= 80 for line in lines)


def test_the_header_says_the_mode_the_round_and_the_phase():
    top = render(_state(phase="rest", round_no=2, rounds=3), 80, 24)[0]
    assert "ika coach" in top and "round 2 of 3" in top and "REST" in top
    assert "HOLD YOUR GUARD" in render(_state(phase="calibrating"), 80, 24)[0]


def test_a_narrow_terminal_drops_the_figure_and_keeps_the_reads():
    text = "\n".join(render(_state(), 60, 20))
    assert "┌" not in text
    assert "left guard" in text and "right guard" in text and "37 thrown" in text


def test_no_one_in_view_is_said_in_the_figure_box():
    assert "no one in view" in "\n".join(render(_state(body=None), 80, 24))


def test_the_figure_draws_the_pose_and_nothing_without_one():
    assert any(ch != " " for row in figure(body(), 20, 10) for ch in row)
    assert all(row.strip() == "" for row in figure(None, 20, 10))


def test_unseen_joints_are_not_drawn():
    hidden = body(visibility=[0.0] * 33)
    assert all(row.strip() == "" for row in figure(hidden, 20, 10))


def test_clock_text():
    assert clock_text(161.0) == "2:41"
    assert clock_text(0.2) == "0:01"         # counts down to zero, not past it
    assert clock_text(-3.0) == "0:00"
    assert clock_text(None) == "-:--"
    assert clock_text(math.nan) == "-:--"


def test_the_guard_meter_reads_down_below_the_down_line():
    assert guard_meter(0.0, 20)[1] == "up"
    assert guard_meter(-0.19, 20)[1] == "up"
    assert guard_meter(-0.25, 20)[1] == "DOWN"
    bar, word = guard_meter(math.nan, 20)
    assert word == "-" and "█" not in bar
    # A lower hand fills less of the bar.
    assert guard_meter(-0.4, 20)[0].count("█") < guard_meter(0.0, 20)[0].count("█")


def test_punch_ticks_put_the_newest_on_the_right_and_forget_old_ones():
    strip = ticks([(10.0, "left"), (-50.0, "left"), (10.0, "right")], "left", 10.0, 20)
    assert len(strip) == 20 and strip[-1] == "█" and strip.count("█") == 1


def test_large_type_shapes():
    assert big("1", "plain") == ["1"]
    assert len(big("12", "half")) == 3
    assert len(big("12", "b1")) == 5 and len(big("12", "b2")) == 10


def test_colour_is_only_escape_codes_around_the_same_text():
    plain = render(_state(), 80, 24)
    coloured = render(_state(), 80, 24, color=True)
    import re
    strip = [re.sub(r"\x1b\[[0-9;]*m", "", line) for line in coloured]
    assert strip == plain


class _Tty(io.StringIO):
    def isatty(self):
        return True


def test_the_live_screen_is_only_for_a_terminal():
    assert live.wanted(False, False, False, False, out=_Tty())
    assert not live.wanted(False, False, False, False, out=io.StringIO())
    assert not live.wanted(True, False, False, False, out=_Tty())      # --quiet
    assert not live.wanted(False, True, False, False, out=_Tty())      # --plain
    assert not live.wanted(False, False, True, True, out=_Tty())       # replay --silent
    assert live.wanted(False, False, True, False, out=_Tty())          # replay, watched


def test_the_screen_hands_the_terminal_back():
    out = _Tty()
    with live.Screen(out=out) as screen:
        assert screen.draw(_state(), force=True)
    text = out.getvalue()
    assert text.startswith("\x1b[?1049h") and text.endswith("\x1b[?1049l")
    assert "\x1b[?25h" in text            # the cursor comes back


def _live_drill(tmp_path, monkeypatch, frames, voice=False, seconds=None):
    clock = Clock()
    drawn, rings = [], []
    monkeypatch.setattr(drill_app.time, "time", clock.time)
    monkeypatch.setattr(drill_app, "_open", lambda _src: (Camera(clock, frames), False))
    monkeypatch.setattr(drill_app, "PoseTracker", Pose)
    monkeypatch.setattr(drill_app, "bell", rings.append)
    monkeypatch.setattr(live.Screen, "draw",
                        lambda self, state, force=False: drawn.append(
                            (state.phase, render(state, 80, 24))))
    monkeypatch.setattr(live.Screen, "__enter__", lambda self: self)
    monkeypatch.setattr(live.Screen, "__exit__", lambda self, *a: False)
    result = drill_app.run_drill(camera=0, profile=tmp_path / "profiles", voice=voice,
                                 live=True, seconds=seconds)
    return result, drawn, rings


def test_a_live_drill_draws_every_frame_and_still_prints_the_summary(
        tmp_path, monkeypatch, capsys):
    result, drawn, rings = _live_drill(tmp_path, monkeypatch, frames=30 * 8)
    out = capsys.readouterr().out
    assert result["seconds"] > 7
    assert len(drawn) == 30 * 8
    assert drawn[0][0] == "calibrating" and drawn[-1][0] == "box"
    assert "round over" in out and "punches" in out and "saved to" in out
    # The per-event transcript is replaced by the screen, not printed under it.
    assert "calibrated. resting guard" not in out
    assert rings == []                     # --silent rounds make no sound


def test_the_bell_rings_once_to_start_and_three_times_at_the_end(tmp_path, monkeypatch):
    _result, _drawn, rings = _live_drill(tmp_path, monkeypatch, frames=30 * 10,
                                         voice=True, seconds=8.0)
    assert rings == [1, 3]


def test_a_live_coach_round_shows_the_call_and_the_round(tmp_path, monkeypatch, capsys):
    clock = Clock()
    drawn, rings = [], []
    monkeypatch.setattr(coach_app.time, "time", clock.time)
    monkeypatch.setattr(coach_app, "_open", lambda _src: (Camera(clock, 30 * 30), False))
    monkeypatch.setattr(coach_app, "PoseTracker", Pose)
    monkeypatch.setattr(coach_app, "bell", rings.append)
    monkeypatch.setattr(live.Screen, "draw",
                        lambda self, state, force=False: drawn.append(
                            (state.phase, state.big_text, render(state, 80, 24))))
    monkeypatch.setattr(live.Screen, "__enter__", lambda self: self)
    monkeypatch.setattr(live.Screen, "__exit__", lambda self, *a: False)
    coach_app.run_coach(camera=0, profile=tmp_path / "profiles", voice=False, live=True,
                        seconds=12.0, seed=1)
    phases = {phase for phase, _text, _lines in drawn}
    assert {"calibrating", "box"} <= phases
    called = [text for phase, text, _lines in drawn if phase == "box" and text]
    assert called and all("-" in t or t.isdigit() for t in called)
    out = capsys.readouterr().out
    assert ">> " not in out                # calls are on screen, not in the transcript
    assert "calls" in out                  # the session summary still prints
    assert rings == []
