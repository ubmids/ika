"""The terminal app.

Same pipeline as the windowed version, drawn in text. The hand goes on a
braille canvas, which gives four times the vertical resolution of plain
characters and is enough to see your own fingers move, and everything else is
bars and numbers.

Both hands are tracked here. The windowed version asked the landmarker for one,
which was a simplification on my part rather than a limit of anything: each
hand gets its own classifier pass and its own state machine, so their
hysteresis and dwell timers cannot interfere. Engaging with either hand engages
the system, and the cursor follows whichever hand is actually pointing.
"""

from __future__ import annotations

import curses
import math
import time
from pathlib import Path

import cv2
import numpy as np

from . import control, features
from .canvas import Braille
from .hands import HandTracker
from .live_commit import EarlyCommitter
from .machine import GestureMachine
from .model import GestureNet
from .pointer import CursorSmoother, map_to_screen, screen_size
from .schema import (INDEX_DIP, INDEX_MCP, INDEX_PIP, INDEX_TIP, MIDDLE_DIP,
                     MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP, PINKY_DIP, PINKY_MCP,
                     PINKY_PIP, PINKY_TIP, RING_DIP, RING_MCP, RING_PIP,
                     RING_TIP, THUMB_CMC, THUMB_IP, THUMB_MCP, THUMB_TIP,
                     TIPS, WRIST)

PINCH_CLOSE = 0.38
PINCH_OPEN = 0.55

BAR_FULL = "█"
BAR_PARTS = " ▏▎▍▌▋▊▉"


def bar(value: float, width: int) -> str:
    """A bar with sub-character resolution, so small values are still visible."""
    value = max(0.0, min(1.0, float(value)))
    exact = value * width
    whole = int(exact)
    remainder = exact - whole
    out = BAR_FULL * whole
    if whole < width:
        out += BAR_PARTS[int(remainder * (len(BAR_PARTS) - 1))]
    return out.ljust(width)[:width]


def meter(value: float, width: int) -> str:
    filled = int(max(0.0, min(1.0, value)) * width)
    return ("▓" * filled + "░" * (width - filled))[:width]


PALM_LAYER, BONE_LAYER, TIP_LAYER = 1, 2, 3

# Knuckle row plus the wrist, in order around the palm, so it fills as a
# convex shape rather than a bow tie.
_PALM_OUTLINE = (WRIST, PINKY_MCP, RING_MCP, MIDDLE_MCP, INDEX_MCP, THUMB_CMC)
_CHAINS = (
    (THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP),
    (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
)


class HandRenderer:
    """Draws hands, and remembers them between frames so they stop shaking.

    Two things separate this from the stick figure it replaces.

    **Smoothing.** Landmark estimates wobble a dot or two every frame. On a
    still hand that reads as static, and it was most of why the old drawing
    looked scratchy. An exponential average over positions costs nothing and
    is the single biggest improvement here. It is presentation only: the
    classifier still sees the raw landmarks, because smoothing its input would
    add lag to every decision.

    **Solidity.** A filled palm with tapered fingers reads as a hand; lines
    between joints read as a bundle of sticks. Widths scale with how large the
    hand appears, so it looks right near the camera and far from it.
    """

    def __init__(self, smoothing: float = 0.55):
        self.smoothing = smoothing
        self._previous: dict[str, np.ndarray] = {}

    def forget(self, label: str) -> None:
        self._previous.pop(label, None)

    def draw(self, canvas: Braille, hands) -> None:
        canvas.clear()
        live = set()
        for hand in hands:
            label = getattr(hand, "label", "?")
            live.add(label)
            marks = np.asarray(hand.image, dtype=np.float64)
            previous = self._previous.get(label)
            if previous is not None and previous.shape == marks.shape:
                k = self.smoothing
                marks = k * previous + (1.0 - k) * marks
            self._previous[label] = marks
            self._draw_one(canvas, marks)
        for label in list(self._previous):
            if label not in live:
                self.forget(label)

    def _draw_one(self, canvas: Braille, marks: np.ndarray) -> None:
        # Mirrored, so moving your hand right moves the drawing right.
        points = [
            ((1.0 - x) * (canvas.width - 1), y * (canvas.height - 1))
            for x, y, _ in marks
        ]

        span = math.hypot(
            points[MIDDLE_MCP][0] - points[WRIST][0],
            points[MIDDLE_MCP][1] - points[WRIST][1],
        )
        # Line art, not silhouette. At anatomically correct width a finger is
        # ~19% of palm length, which means adjacent fingers *touch* at the
        # knuckles, and in a dot matrix with no outlines they merge into one
        # solid mitten. Drawing them at about half that width leaves a visible
        # gap, and the hand reads far better as a drawing than as a shape.
        base = max(0.6, span * 0.045)    # finger half-width at the knuckle
        tip = max(0.4, span * 0.025)     # and at the fingertip

        # The palm is outlined for the same reason: filled, it swallows the
        # fingers that grow out of it.
        for i in range(len(_PALM_OUTLINE)):
            a = points[_PALM_OUTLINE[i]]
            b = points[_PALM_OUTLINE[(i + 1) % len(_PALM_OUTLINE)]]
            canvas.stroke(*a, *b, 0.7, 0.7, layer=PALM_LAYER)

        for chain in _CHAINS:
            for i in range(len(chain) - 1):
                a, b = chain[i], chain[i + 1]
                # Taper along the whole finger, not per segment, so the joints
                # do not step in width.
                start = base + (tip - base) * (i / (len(chain) - 1))
                end = base + (tip - base) * ((i + 1) / (len(chain) - 1))
                canvas.stroke(*points[a], *points[b], start, end, layer=BONE_LAYER)

        for index in TIPS:
            canvas.dot(int(points[index][0]), int(points[index][1]),
                       max(1, int(round(tip))), layer=TIP_LAYER)


_renderer = HandRenderer()


def draw_hands(canvas: Braille, hands) -> None:
    """Module-level entry point, kept for the tests and simple callers."""
    _renderer.draw(canvas, hands)


class HandState:
    """Per-hand decision state, so two hands never share a timer.

    Two layers, deliberately, because arming and acting want opposite things.

    `machine` owns engagement only. Arming the system should be slow and
    deliberate, so the dwell timer's 250 ms is a feature there: you do not want
    your machine to go live because a hand passed through an open palm.

    `committer` owns the actual calls, and it commits on evidence rather than
    waiting for a pose to be held. Measured on identical streams, 7 frames
    against 11, 233 ms against 367 ms.

    That split is safe here for a reason worth stating. Committing early gets
    fooled by a movement that lies about its opening, and against a fighter who
    feints that is a real cost. Someone gesturing at their own computer is not
    feinting at themselves, so the accuracy penalty that makes this a trade in
    the `tell` lane very nearly vanishes in this one.
    """

    def __init__(self, classes: list[str], threshold: float, dwell: int,
                 smoothing: float, commit: bool = True):
        self.machine = GestureMachine(
            classes, threshold=threshold, dwell_frames=dwell, smoothing=smoothing
        )
        self.committer = (
            EarlyCommitter(classes, threshold=threshold) if commit else None
        )
        self.gesture = "-"
        self.confidence = 0.0
        self.probabilities: np.ndarray | None = None
        self.pinched = False

    @property
    def progress(self) -> float:
        """What the on-screen meter shows, from whichever layer is deciding."""
        if self.committer is not None and self.machine.engaged:
            return self.committer.progress
        return self.machine.progress

    @property
    def candidate(self) -> str | None:
        if self.committer is not None and self.machine.engaged:
            return self.committer.watching and self.gesture or None
        return self.machine.candidate


def run_terminal(
    checkpoint: str | Path = "checkpoints/static.pt",
    camera: int = 0,
    width: int = 480,
    live_control: bool = False,
    threshold: float = 0.80,
    dwell: int = 5,
    smoothing: float = 0.6,
    max_hands: int = 2,
    pointer: bool = True,
    commit: bool = True,
) -> None:
    model = GestureNet.load(checkpoint)
    curses.wrapper(
        _loop, model, camera, width, live_control, threshold, dwell, smoothing,
        max_hands, pointer, commit,
    )


def _loop(stdscr, model, camera, width, live_control, threshold, dwell, smoothing,
          max_hands, pointer, commit=True) -> None:
    curses.curs_set(0)
    stdscr.nodelay(True)
    if curses.has_colors():
        curses.start_color()
        curses.use_default_colors()
        for i, colour in enumerate(
            (curses.COLOR_GREEN, curses.COLOR_YELLOW, curses.COLOR_CYAN, curses.COLOR_RED), 1
        ):
            curses.init_pair(i, colour, -1)
    GREEN, YELLOW, CYAN, RED = (curses.color_pair(i) for i in (1, 2, 3, 4))
    DIM = curses.A_DIM

    capture = cv2.VideoCapture(camera)
    if not capture.isOpened():
        stdscr.addstr(1, 2, f"cannot open camera {camera}.")
        stdscr.addstr(2, 2, "macOS needs camera permission for your terminal:")
        stdscr.addstr(3, 2, "System Settings > Privacy & Security > Camera")
        stdscr.addstr(5, 2, "press any key")
        stdscr.nodelay(False)
        stdscr.getch()
        return

    controller = control.Controller(live=live_control)
    cursor = CursorSmoother()
    screen_w, screen_h = screen_size()
    states: dict[str, HandState] = {}
    fired_log: list[str] = []

    started = time.time()
    frames, fps = 0, 0.0
    canvas: Braille | None = None
    canvas_size = (0, 0)
    renderer = HandRenderer()

    with HandTracker(max_hands=max_hands) as tracker:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            scale = width / bgr.shape[1]
            bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            now = time.time()
            hands = tracker(rgb, int((now - started) * 1000))

            seen = set()
            for hand in hands:
                key = hand.label
                seen.add(key)
                state = states.setdefault(
                    key, HandState(model.classes, threshold, dwell, smoothing, commit)
                )
                vector = features.extract(hand.world, hand.is_left)
                best, state.confidence, state.probabilities = model.predict(vector)
                state.gesture = model.classes[best]

                for intent in state.machine.update(state.probabilities, now):
                    if intent.kind == "fired":
                        # In commit mode the dwell machine is kept only for
                        # engagement, so its own firings are ignored rather
                        # than doubling every action.
                        if state.committer is not None:
                            continue
                        if control.perform(intent.gesture, controller):
                            fired_log.append(
                                f"{key[:1]}  {intent.gesture} -> "
                                f"{control.ACTIONS[intent.gesture].describe}"
                            )
                    elif intent.kind == "engaged":
                        fired_log.append(f"{key[:1]}  engaged")
                    elif intent.kind == "disengaged":
                        fired_log.append(f"{key[:1]}  disengaged")
                        cursor.reset()
                        if state.committer is not None:
                            state.committer.reset()

                if state.committer is not None and state.machine.engaged:
                    call = state.committer.update(state.probabilities, now)
                    if call is not None and control.perform(call.label, controller):
                        fired_log.append(
                            f"{key[:1]}  {call.label} -> "
                            f"{control.ACTIONS[call.label].describe}"
                            f"  ({call.frames}f)"
                        )
                del fired_log[:-4]

            # A hand that left the frame must not leave a primed trigger behind.
            for key, state in states.items():
                if key not in seen:
                    state.machine.update(None, now)
                    if state.committer is not None:
                        state.committer.update(None, now)
                    state.gesture, state.confidence = "-", 0.0
                    state.probabilities = None
                    if state.pinched:
                        state.pinched = False
                        controller.press_mouse(False)

            engaged = any(s.machine.engaged for s in states.values())

            # Continuous control follows whichever hand is actually pointing.
            cursor_at = None
            if engaged and pointer:
                for hand in hands:
                    state = states.get(hand.label)
                    if state is None:
                        continue
                    distance = features.pinch_distance(hand.world)
                    if state.gesture in control.CONTINUOUS or state.pinched:
                        nx, ny = hand.image[INDEX_TIP, 0], hand.image[INDEX_TIP, 1]
                        sx, sy = map_to_screen(nx, ny, screen_w, screen_h, mirrored=True)
                        cursor_at = cursor(sx, sy)
                        controller.move_cursor(*cursor_at)
                    if not state.pinched and distance < PINCH_CLOSE:
                        state.pinched = True
                        controller.press_mouse(True)
                    elif state.pinched and distance > PINCH_OPEN:
                        state.pinched = False
                        controller.press_mouse(False)
                    break

            frames += 1
            if frames % 15 == 0:
                fps = frames / (now - started)

            height, term_w = stdscr.getmaxyx()
            box_w, box_h = _box_size(
                term_w, height, bgr.shape[1] / bgr.shape[0],
                _text_rows(states, fired_log),
            )
            if canvas is None or canvas_size != (box_w, box_h):
                canvas = Braille(box_w - 2, box_h - 2)
                canvas_size = (box_w, box_h)
            renderer.draw(canvas, hands)

            stdscr.erase()
            _render(stdscr, canvas, box_w, box_h, hands, states, engaged, fps,
                    live_control, cursor_at, fired_log, controller, model,
                    (GREEN, YELLOW, CYAN, RED, DIM))
            stdscr.refresh()

            key = stdscr.getch()
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                for state in states.values():
                    state.machine.reset()
                cursor.reset()
                fired_log.append("reset")

    for state in states.values():
        if state.pinched:
            controller.press_mouse(False)
    capture.release()


def _box_size(term_w: int, term_h: int, aspect: float, text_rows: int) -> tuple[int, int]:
    """Fill the terminal, without stretching the hand.

    Two things to respect. The readouts below need their rows, so the picture
    gets what is left. And a braille cell is 2 dots wide by 4 tall, so a square
    block of characters is a *tall thin* pixel grid: drawing a 4:3 camera frame
    into it without correcting would squash the hand vertically by half.

    Solving `(cols * 2) / (rows * 4) == aspect` gives `cols = rows * 2 *
    aspect`, so the picture grows until it runs out of either width or height.
    Earlier this was capped at 78 by 14 regardless of terminal size, which
    wasted most of a large window.
    """
    available_rows = max(4, term_h - text_rows)
    available_cols = max(18, term_w - 4)

    rows = available_rows
    cols = int(rows * 2 * aspect)
    if cols > available_cols:                 # width-limited instead
        cols = available_cols
        rows = max(4, int(cols / (2 * aspect)))
    return cols + 2, rows + 2


def _text_rows(states: dict, fired_log: list) -> int:
    """How many rows the readouts below the picture will need."""
    per_hand = 4                      # gesture, dwell, two runners-up
    hands = max(1, len(states))
    return (
        1          # title
        + 2        # box borders
        + 2        # engaged line and a gap
        + hands * (per_hand + 1)
        + 1        # cursor
        + min(3, max(1, len(fired_log)))
        + 2        # footer and a gap
    )


def _render(stdscr, canvas, box_w, box_h, hands, states, engaged, fps,
            live_control, cursor_at, fired_log, controller, model, colours) -> None:
    GREEN, YELLOW, CYAN, RED, DIM = colours
    height, term_w = stdscr.getmaxyx()

    def put(row, col, text, attr=0):
        if 0 <= row < height and col < term_w:
            stdscr.addnstr(row, col, text, term_w - col - 1, attr)

    mode = "LIVE" if live_control else "dry run"
    put(0, 2, "ika", curses.A_BOLD)
    put(0, 6, "bare hands", DIM)
    right = f"{len(hands)} hand{'s' if len(hands) != 1 else ''}   {fps:.0f} fps   {mode}"
    put(0, max(18, box_w + 2 - len(right)), right, RED if live_control else DIM)

    put(1, 2, "┌" + "─" * (box_w - 2) + "┐", DIM)
    layer_attr = {1: CYAN | DIM, 2: CYAN, 3: GREEN | curses.A_BOLD}
    rows = canvas.runs()
    for i, row in enumerate(rows):
        put(2 + i, 2, "│", DIM)
        for column, text, layer in row:
            put(2 + i, 3 + column, text, layer_attr.get(layer, CYAN))
        put(2 + i, 2 + box_w - 1, "│", DIM)
    bottom = 2 + len(rows)
    put(bottom, 2, "└" + "─" * (box_w - 2) + "┘", DIM)

    row = bottom + 1
    if engaged:
        put(row, 2, "ENGAGED", GREEN | curses.A_BOLD)
    else:
        put(row, 2, "hold an open palm to engage", YELLOW)
    row += 2

    if not states:
        put(row, 2, "no hand in frame", DIM)
        row += 2
    for label in sorted(states):
        state = states[label]
        if state.probabilities is None:
            put(row, 2, f"{label:<6} away", DIM)
            row += 1
            continue
        marks = "  pinched" if state.pinched else ""
        wide = max(24, min(60, term_w - 34))
        put(row, 2, f"{label:<6}", curses.A_BOLD)
        put(row, 9, f"{state.gesture:<12}", GREEN if state.machine.engaged else 0)
        put(row, 22, f"{state.confidence:>4.0%} ")
        put(row, 28, bar(state.confidence, wide), CYAN)
        put(row, 29 + wide, marks, YELLOW)
        row += 1
        if state.candidate:
            put(row, 9, f"{state.candidate:<12}", DIM)
            put(row, 28, meter(state.progress, wide), YELLOW)
            row += 1
        # runners-up, so a misread is legible instead of mysterious
        order = np.argsort(state.probabilities)[::-1][1:3]
        for index in order:
            if state.probabilities[index] < 0.02:
                continue
            put(row, 9, f"{model.classes[index]:<12}", DIM)
            put(row, 28, bar(float(state.probabilities[index]), wide), DIM)
            put(row, 22, f"{state.probabilities[index]:>4.0%} ", DIM)
            row += 1
        row += 1

    if cursor_at and row < height - 2:
        put(row, 2, f"cursor  {cursor_at[0]}, {cursor_at[1]}", DIM)
        row += 1

    # Only as many log lines as fit above the footer, newest last.
    spare = max(0, (height - 2) - row)
    if spare and fired_log:
        for entry in fired_log[-spare:]:
            put(row, 2, entry, GREEN)
            row += 1

    put(height - 1, 2, "q quit    r reset", DIM)
