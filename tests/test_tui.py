"""The terminal drawing, tested without a terminal.

curses needs a tty, but everything that could actually be wrong here is pure:
the braille rasteriser and the bars. The parts that need a real terminal are
layout calls that fail loudly the moment you run it.
"""

import numpy as np
import pytest

from ika.canvas import Braille
from ika.tui import bar, draw_hands, meter


class FakeHand:
    """Just enough of a Hand for the drawing code."""

    def __init__(self, marks):
        self.image = marks


def test_braille_cell_geometry():
    """Each character is an independent 2x4 dot grid, which is the entire
    reason this is legible at terminal resolution."""
    c = Braille(10, 3)
    assert (c.width, c.height) == (20, 12)


def test_setting_dots_produces_braille_characters():
    c = Braille(4, 1)
    c.set(0, 0)
    row = c.text_rows()[0]
    assert row[0] == "⠁"          # dot 1 only
    c.set(1, 3)
    assert c.text_rows()[0][0] == "⢁"   # dots 1 and 8


def test_dots_outside_the_canvas_are_dropped_not_crashed():
    """The hand goes off-frame constantly."""
    c = Braille(4, 2)
    for x, y in [(-5, 0), (0, -5), (999, 0), (0, 999), (-1, -1)]:
        c.set(x, y)
    assert all(ch == " " for row in c.text_rows() for ch in row)


def test_lines_are_continuous():
    """Bones drawn with gaps look like noise rather than a hand."""
    c = Braille(20, 5)
    c.line(0, 0, 39, 19)
    filled = sum(ch != " " for row in c.text_rows() for ch in row)
    assert filled >= 18, f"only {filled} cells touched by a diagonal"


def test_clear_empties_the_canvas():
    c = Braille(6, 2)
    c.line(0, 0, 11, 7)
    c.clear()
    assert all(ch == " " for row in c.text_rows() for ch in row)


def test_bars_have_fixed_width_and_are_ordered():
    for width in (10, 24):
        assert len(bar(0.0, width)) == width
        assert len(bar(1.0, width)) == width
        assert len(bar(0.5, width)) == width
    assert bar(1.0, 10).strip() == "█" * 10
    assert bar(0.0, 10).strip() == ""


def test_a_small_value_still_shows_something():
    """Sub-character blocks matter: 3% confidence rendering as an empty bar
    looks identical to no reading at all."""
    assert bar(0.03, 10).strip() != ""


def test_bars_clamp_instead_of_overflowing():
    assert len(bar(-1.0, 12)) == 12
    assert len(bar(5.0, 12)) == 12


def test_meter_is_fixed_width():
    assert len(meter(0.0, 16)) == len(meter(1.0, 16)) == 16
    assert meter(1.0, 8) == "▓" * 8
    assert meter(0.0, 8) == "░" * 8


def test_drawing_two_hands_marks_more_than_one():
    """The thing that was wrong: only one hand was ever tracked."""
    rng = np.random.default_rng(0)
    left = FakeHand(np.clip(rng.random((21, 3)) * 0.4 + 0.05, 0, 1).astype(np.float32))
    right = FakeHand(np.clip(rng.random((21, 3)) * 0.4 + 0.55, 0, 1).astype(np.float32))

    one = Braille(40, 8)
    draw_hands(one, [left])
    two = Braille(40, 8)
    draw_hands(two, [left, right])

    count = lambda c: sum(ch != " " for row in c.text_rows() for ch in row)
    assert count(two) > count(one) * 1.3


def test_drawing_no_hands_is_blank_and_safe():
    c = Braille(20, 4)
    draw_hands(c, [])
    assert all(ch == " " for row in c.text_rows() for ch in row)


def test_the_drawing_is_mirrored():
    """Mirrored so moving your hand right moves the drawing right. Getting
    this backwards makes the whole thing feel broken."""
    marks = np.tile(np.array([[0.15, 0.5, 0.0]], dtype=np.float32), (21, 1))
    c = Braille(40, 6)
    draw_hands(c, [FakeHand(marks)])
    rows = c.text_rows()
    columns = [i for row in rows for i, ch in enumerate(row) if ch != " "]
    assert min(columns) > 20, "a hand at x=0.15 should draw on the right"
