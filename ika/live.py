"""The round as you see it from two metres away.

`ika drill` and `ika coach` print a transcript, which is the right thing for a
pipe, a script or a replay, and the wrong thing for someone boxing at the
other end of the room: a scrolling line of text cannot be read mid-combo. This
is the screen for that person. A big round clock, the call in large letters,
a small drawing of the pose so you can see you are in shot, a meter per hand
for the guard, and a strip of the punches it caught.

Everything that decides what goes on screen is pure: `render` takes a
`LiveState` and a terminal size and returns the lines. `Screen` is the only
part that touches the terminal, and it is never built unless stdout is a
terminal, so scripts, pipes and tests keep the plain transcript.

The guard meter is the shoulder-relative read the habit statistics use
(`sag.GuardTrack`): height above the shoulders in torso lengths, against your
own recent guard. Zero is your guard, and below `sag.DOWN` counts as down.
It needs about a second of frames before it has a guard to compare with, so
it reads "-" until then.
"""

from __future__ import annotations

import math
import shutil
import sys
from dataclasses import dataclass, field

from .canvas import Braille

# --- a 3x5 pixel font, for the clock and the call ------------------------

_FONT = {
    "0": ("###", "#.#", "#.#", "#.#", "###"),
    "1": (".#.", "##.", ".#.", ".#.", "###"),
    "2": ("###", "..#", "###", "#..", "###"),
    "3": ("###", "..#", "###", "..#", "###"),
    "4": ("#.#", "#.#", "###", "..#", "..#"),
    "5": ("###", "#..", "###", "..#", "###"),
    "6": ("###", "#..", "###", "#.#", "###"),
    "7": ("###", "..#", "..#", ".#.", ".#."),
    "8": ("###", "#.#", "###", "#.#", "###"),
    "9": ("###", "#.#", "###", "..#", "###"),
    "A": (".#.", "#.#", "###", "#.#", "#.#"),
    "B": ("##.", "#.#", "##.", "#.#", "##."),
    "C": (".##", "#..", "#..", "#..", ".##"),
    "D": ("##.", "#.#", "#.#", "#.#", "##."),
    "E": ("###", "#..", "##.", "#..", "###"),
    "F": ("###", "#..", "##.", "#..", "#.."),
    "G": (".##", "#..", "#.#", "#.#", ".##"),
    "H": ("#.#", "#.#", "###", "#.#", "#.#"),
    "I": ("###", ".#.", ".#.", ".#.", "###"),
    "J": ("..#", "..#", "..#", "#.#", ".#."),
    "K": ("#.#", "#.#", "##.", "#.#", "#.#"),
    "L": ("#..", "#..", "#..", "#..", "###"),
    "M": ("#.#", "###", "###", "#.#", "#.#"),
    "N": ("##.", "#.#", "#.#", "#.#", "#.#"),
    "O": (".#.", "#.#", "#.#", "#.#", ".#."),
    "P": ("##.", "#.#", "##.", "#..", "#.."),
    "Q": (".#.", "#.#", "#.#", "##.", ".##"),
    "R": ("##.", "#.#", "##.", "#.#", "#.#"),
    "S": (".##", "#..", ".#.", "..#", "##."),
    "T": ("###", ".#.", ".#.", ".#.", ".#."),
    "U": ("#.#", "#.#", "#.#", "#.#", "###"),
    "V": ("#.#", "#.#", "#.#", "#.#", ".#."),
    "W": ("#.#", "#.#", "###", "###", "#.#"),
    "X": ("#.#", "#.#", ".#.", "#.#", "#.#"),
    "Y": ("#.#", "#.#", ".#.", ".#.", ".#."),
    "Z": ("###", "..#", ".#.", "#..", "###"),
    ":": (".", "#", ".", "#", "."),
    "-": ("...", "...", "###", "...", "..."),
    ".": (".", ".", ".", ".", "#"),
    ",": (".", ".", ".", "#", "#"),
    " ": ("..", "..", "..", "..", ".."),
}


def _pixels(text: str) -> list[str]:
    """The text as five rows of '#' and '.', one blank column between glyphs."""
    rows = ["", "", "", "", ""]
    glyphs = [_FONT.get(ch, _FONT[" "]) for ch in text.upper()]
    for i, glyph in enumerate(glyphs):
        for r in range(5):
            rows[r] += glyph[r] + ("." if i < len(glyphs) - 1 else "")
    return rows


# Sizes, smallest first. "half" draws two pixel rows per text row with half
# blocks, so a pixel is one column by half a row, which is about square in a
# terminal cell. "bN" draws a pixel as 2N columns by N rows of full blocks.
SIZES = ("plain", "half", "b1", "b2", "b3")


def big(text: str, size: str) -> list[str]:
    """`text` in large letters. Unknown characters are drawn as spaces."""
    if size == "plain":
        return [text.upper()]
    rows = _pixels(text)
    if size == "half":
        rows = rows + ["." * len(rows[0])]
        out = []
        for top, bottom in zip(rows[0::2], rows[1::2]):
            out.append("".join(
                "█" if a == "#" and b == "#" else "▀" if a == "#" else
                "▄" if b == "#" else " " for a, b in zip(top, bottom)))
        return out
    n = int(size[1:])
    out = []
    for row in rows:
        line = "".join(("█" if p == "#" else " ") * (2 * n) for p in row)
        out += [line] * n
    return out


def big_size(text: str, size: str) -> tuple[int, int]:
    """(columns, rows) that `big(text, size)` takes."""
    lines = big(text, size)
    return max((len(line) for line in lines), default=0), len(lines)


# --- what the screen shows ------------------------------------------------

@dataclass
class LiveState:
    """Everything on screen, as plain values. The apps fill it in each frame."""

    mode: str = "drill"                 # "drill" or "coach"
    clock: float = 0.0
    phase: str = "calibrating"          # calibrating, ready, box, rest, time
    round_no: int = 0
    rounds: int = 0
    remaining: float | None = None      # seconds left in this phase, if known
    elapsed: float = 0.0                # seconds boxed so far, when nothing is counting down
    big_text: str = ""                  # the call (coach) or the cue (drill), in large letters
    words: str = ""                     # the same in words, under it
    fresh: bool = False                 # the big text was just said
    said_at: float = -1e9               # when it was said, on the session clock
    guard: dict = field(default_factory=lambda: {"left": float("nan"),
                                                 "right": float("nan")})
    punches: list = field(default_factory=list)   # (time, "left" | "right")
    punch_count: int = 0
    body: object = None                 # a pose with .image and .visibility, or None
    aspect: float = 4 / 3
    notes: list = field(default_factory=list)
    fps: float = 0.0


def clock_text(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "-:--"
    seconds = max(0, int(math.ceil(seconds - 1e-9)))
    return f"{seconds // 60}:{seconds % 60:02d}"


# Guard meter range, in torso lengths against your own guard. sag.DOWN is
# 0.2, so the down mark sits halfway along: plenty of room to see a hand
# sliding toward it, and a little above zero for a guard held high.
GUARD_RANGE = (-0.6, 0.2)


def guard_meter(height: float, width: int, down: float = 0.2) -> tuple[str, str]:
    """A bar for one hand's guard and its label: "up", "DOWN" or "-"."""
    lo, hi = GUARD_RANGE
    mark = int(round((-down - lo) / (hi - lo) * (width - 1)))
    if height is None or not math.isfinite(height):
        cells = ["·"] * width
        cells[mark] = "|"
        return "".join(cells), "-"
    fill = int(round(max(0.0, min(1.0, (height - lo) / (hi - lo))) * width))
    cells = ["█" if i < fill else "░" for i in range(width)]
    cells[mark] = "|" if mark >= fill else "┃"
    return "".join(cells), ("up" if height >= -down else "DOWN")


TICK_SECONDS = 12.0


def ticks(punches, side: str, now: float, width: int,
          seconds: float = TICK_SECONDS) -> str:
    """The last `seconds` of one hand's punches, newest at the right."""
    cells = ["·"] * width
    for at, s in punches:
        age = now - at
        if s != side or age < 0 or age > seconds:
            continue
        col = width - 1 - int(age / seconds * (width - 1))
        cells[max(0, min(width - 1, col))] = "█"
    return "".join(cells)


def figure(body, columns: int, rows: int, aspect: float = 4 / 3,
           threshold: float = 0.5) -> list[str]:
    """A stick figure of the pose, fitted to the box, mirrored like a mirror.

    Fitted to the joints that were actually seen rather than the whole camera
    frame, so the figure stays large however far back you stand. Unseen
    joints are left out, as in the gesture app: a pose model invents a limb
    it cannot see, and drawing the guess would look like evidence.
    """
    from .body import LIMBS, NOSE, LEFT_SHOULDER, RIGHT_SHOULDER

    canvas = Braille(columns, rows)
    if body is None:
        return canvas.text_rows()
    image = [(float(p[0]) * aspect, float(p[1])) for p in body.image]
    seen = [float(v) >= threshold for v in body.visibility]
    joints = {j for a, b in LIMBS for j in (a, b) if seen[a] and seen[b]}
    if seen[NOSE]:
        joints.add(NOSE)
    if not joints:
        return canvas.text_rows()
    xs = [image[j][0] for j in joints]
    ys = [image[j][1] for j in joints]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)
    margin = 3
    scale = min((canvas.width - 2 * margin) / max(max(xs) - min(xs), 1e-6),
                (canvas.height - 2 * margin) / max(max(ys) - min(ys), 1e-6))
    cx = (max(xs) + min(xs)) / 2
    cy = (max(ys) + min(ys)) / 2

    def at(j):
        x, y = image[j]
        # Mirrored, so raising your left hand raises the hand on the left.
        return (int(round(canvas.width / 2 - (x - cx) * scale)),
                int(round(canvas.height / 2 + (y - cy) * scale)))

    for a, b in LIMBS:
        if seen[a] and seen[b] and a != NOSE and b != NOSE:
            canvas.line(*at(a), *at(b))
    if seen[NOSE]:
        if seen[LEFT_SHOULDER] and seen[RIGHT_SHOULDER]:
            width = abs(image[LEFT_SHOULDER][0] - image[RIGHT_SHOULDER][0]) * scale
            radius = max(1, int(width * 0.22))
        else:
            radius = max(1, int(span * scale * 0.06))
        x, y = at(NOSE)
        for k in range(24):
            angle = 2 * math.pi * k / 24
            canvas.set(int(round(x + radius * math.cos(angle))),
                       int(round(y + radius * math.sin(angle))))
    return canvas.text_rows()


# --- layout ---------------------------------------------------------------

_STYLES = {"bold": "1", "dim": "2", "green": "1;32", "yellow": "1;33", "red": "1;31",
           "cyan": "36", "reverse": "7", "badge_box": "1;30;42", "badge_rest": "1;30;43",
           "badge_time": "1;37;41", "badge_wait": "1;30;47"}


class Grid:
    """A fixed rows-by-columns page of text, with a style per cell."""

    def __init__(self, cols: int, rows: int):
        self.cols, self.rows = cols, rows
        self.chars = [[" "] * cols for _ in range(rows)]
        self.style = [[""] * cols for _ in range(rows)]

    def put(self, row: int, col: int, text: str, style: str = "") -> None:
        if not 0 <= row < self.rows:
            return
        for i, ch in enumerate(text):
            c = col + i
            if 0 <= c < self.cols:
                self.chars[row][c] = ch
                self.style[row][c] = style

    def lines(self, color: bool = False) -> list[str]:
        if not color:
            return ["".join(row) for row in self.chars]
        out = []
        for chars, styles in zip(self.chars, self.style):
            line, current = [], ""
            for ch, st in zip(chars, styles):
                if st != current:
                    line.append("\x1b[0m" + (f"\x1b[{_STYLES[st]}m" if st else ""))
                    current = st
                line.append(ch)
            if current:
                line.append("\x1b[0m")
            out.append("".join(line))
        return out


_PHASE = {"calibrating": ("HOLD YOUR GUARD", "badge_wait"),
          "ready": ("GET READY", "badge_wait"),
          "box": ("BOX", "badge_box"),
          "rest": ("REST", "badge_rest"),
          "time": ("TIME", "badge_time")}

MIN_COLS, MIN_ROWS = 40, 12


def render(state: LiveState, cols: int, rows: int, color: bool = False) -> list[str]:
    """The whole screen, as exactly `rows` lines of at most `cols` columns."""
    cols, rows = max(MIN_COLS, cols), max(MIN_ROWS, rows)
    g = Grid(cols, rows)

    # Header: what this is, the round, and the phase as a badge you can read
    # out of the corner of your eye.
    badge, badge_style = _PHASE.get(state.phase, (state.phase.upper(), "badge_wait"))
    g.put(0, 1, f"ika {state.mode}", "bold")
    if state.rounds > 1 and state.round_no:
        g.put(0, 12, f"round {state.round_no} of {state.rounds}", "bold")
    badge = f" {badge} "
    g.put(0, cols - len(badge) - 1, badge, badge_style)

    # A figure on the left when there is room for one beside a readable clock.
    show_figure = cols >= 70 and rows >= 18
    fig_cols = max(16, min(40, cols // 4)) if show_figure else 0
    fig_rows = rows - 2 if show_figure else 0     # box from row 1 to the row above the footer
    left = fig_cols + 4 if show_figure else 1
    width = cols - left - 1

    clock = clock_text(state.remaining if state.remaining is not None else state.elapsed)
    big_text = state.big_text.strip()

    # Fixed rows under the big type: words, two guard meters, two tick rows,
    # notes, and the gaps between groups.
    note_rows = 2 if rows < 30 else 3
    fixed = 2 + 1 + 1 + 1 + 2 + 1 + 2 + note_rows
    layout = None
    for clock_size, call_size in (("b3", "b2"), ("b2", "b1"), ("b2", "half"),
                                  ("b1", "half"), ("b1", "plain"), ("half", "plain"),
                                  ("plain", "plain")):
        cw, ch = big_size("0:00", clock_size)
        kw, kh = big_size(big_text, call_size) if big_text else (0, 1)
        if cw <= width and rows - 2 >= fixed + ch + kh:
            layout = (clock_size, call_size)
            break
    clock_size, call_size = layout or ("plain", "plain")
    # A long call that will not fit at this size steps down on its own.
    while big_text and call_size != "plain" and big_size(big_text, call_size)[0] > width:
        call_size = SIZES[SIZES.index(call_size) - 1]

    row = 2
    clock_style = ("red" if state.phase == "box" and state.remaining is not None
                   and state.remaining <= 10 else "bold")
    for line in big(clock, clock_size):
        g.put(row, left + max(0, (width - len(line)) // 2), line, clock_style)
        row += 1
    row += 1

    if big_text:
        for line in big(big_text, call_size):
            g.put(row, left + max(0, (width - len(line)) // 2), line,
                  "yellow" if state.fresh else "bold")
            row += 1
    else:
        row += 1
    words = state.words.upper()
    if len(words) > width:
        words = words[:max(0, width - 1)] + "…"
    g.put(row, left + max(0, (width - len(words)) // 2), words,
          "yellow" if state.fresh else "")
    row += 2

    # Guard meters: a long bar with the "down" line marked on it.
    label_w = 12
    bar_w = max(10, min(48, width - label_w - 11))
    for side in ("left", "right"):
        bar, word = guard_meter(state.guard.get(side, float("nan")), bar_w)
        g.put(row, left, f"{side} guard".ljust(label_w), "bold")
        g.put(row, left + label_w, bar, "red" if word == "DOWN" else "green" if word == "up" else "dim")
        g.put(row, left + label_w + bar_w + 1, word, "red" if word == "DOWN" else "")
        row += 1
    row += 1

    for side in ("left", "right"):
        g.put(row, left, f"{side} hand".ljust(label_w), "dim")
        g.put(row, left + label_w, ticks(state.punches, side, state.clock, bar_w), "cyan")
        if side == "right":
            g.put(row, left + label_w + bar_w + 1, f"{state.punch_count} thrown", "dim")
        row += 1

    # Notes: the newest few things the session said, under everything else.
    notes = [n for n in state.notes if n][-note_rows:]
    start = rows - 1 - note_rows
    for i, note in enumerate(notes):
        text = note if len(note) <= width else note[:width - 1] + "…"
        g.put(start + i, left, text, "dim" if i < len(notes) - 1 else "")

    if show_figure:
        g.put(1, 1, "┌" + "─" * fig_cols + "┐", "dim")
        drawing = figure(state.body, fig_cols, fig_rows - 2, state.aspect)
        for i, line in enumerate(drawing):
            g.put(2 + i, 1, "│", "dim")
            g.put(2 + i, 2, line, "cyan")
            g.put(2 + i, 2 + fig_cols, "│", "dim")
        g.put(2 + len(drawing), 1, "└" + "─" * fig_cols + "┘", "dim")
        if state.body is None:
            msg = "no one in view"
            g.put(2 + len(drawing) // 2, 2 + max(0, (fig_cols - len(msg)) // 2), msg, "dim")

    footer = "Ctrl-C ends the round"
    if state.fps:
        footer += f"    {state.fps:.0f} fps"
    g.put(rows - 1, 1, footer, "dim")
    return g.lines(color)


# --- the terminal ---------------------------------------------------------

class Screen:
    """Draws `render` into the terminal's alternate screen, and puts it back.

    Plain ANSI rather than curses: the round has to hand the terminal back
    with the transcript summary printed under it, and curses owns the
    terminal until it exits. Redraws are capped at `fps`, which is plenty for
    a clock and a meter and costs the pose loop almost nothing.
    """

    def __init__(self, out=None, fps: float = 12.0):
        self.out = out or sys.stdout
        self.every = 1.0 / fps
        self._last = -1e9
        self.color = True

    def __enter__(self):
        self.out.write("\x1b[?1049h\x1b[?25l\x1b[H\x1b[2J")
        self.out.flush()
        return self

    def __exit__(self, *_exc):
        self.out.write("\x1b[0m\x1b[?25h\x1b[?1049l")
        self.out.flush()
        return False

    def draw(self, state: LiveState, force: bool = False) -> bool:
        import time

        now = time.monotonic()
        if not force and now - self._last < self.every:
            return False
        self._last = now
        size = shutil.get_terminal_size((80, 24))
        lines = render(state, size.columns, size.lines, color=self.color)
        self.out.write("\x1b[H" + "\n".join(line + "\x1b[K" for line in lines))
        self.out.flush()
        return True


def wanted(quiet: bool, plain: bool, replay: bool, silent: bool, out=None) -> bool:
    """Whether a round gets the live screen or the plain transcript.

    Plain whenever anything might be reading the output: a pipe or file,
    `--quiet`, `--plain`, or a file replay with `--silent`, which is how every
    measured number in the README was produced.
    """
    out = out or sys.stdout
    try:
        tty = out.isatty()
    except (AttributeError, ValueError):
        tty = False
    return tty and not quiet and not plain and not (replay and silent)
