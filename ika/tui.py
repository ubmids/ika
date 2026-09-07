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
import time
from pathlib import Path

import cv2
import numpy as np

from . import control, features
from .canvas import Braille
from .hands import HandTracker
from .machine import GestureMachine
from .model import GestureNet
from .pointer import CursorSmoother, map_to_screen, screen_size
from .schema import CONNECTIONS, INDEX_TIP, TIPS

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


def draw_hands(canvas: Braille, hands) -> None:
    """Both hands, mirrored so the drawing moves the way you do."""
    canvas.clear()
    for hand in hands:
        points = [
            (
                int((1.0 - x) * (canvas.width - 1)),
                int(y * (canvas.height - 1)),
            )
            for x, y, _ in hand.image
        ]
        for a, b in CONNECTIONS:
            canvas.line(*points[a], *points[b])
        for index in TIPS:
            canvas.dot(*points[index], size=2)


class HandState:
    """Per-hand classifier state, so two hands never share a dwell timer."""

    def __init__(self, classes: list[str], threshold: float, dwell: int, smoothing: float):
        self.machine = GestureMachine(
            classes, threshold=threshold, dwell_frames=dwell, smoothing=smoothing
        )
        self.gesture = "-"
        self.confidence = 0.0
        self.probabilities: np.ndarray | None = None
        self.pinched = False


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
) -> None:
    model = GestureNet.load(checkpoint)
    curses.wrapper(
        _loop, model, camera, width, live_control, threshold, dwell, smoothing,
        max_hands, pointer,
    )


def _loop(stdscr, model, camera, width, live_control, threshold, dwell, smoothing,
          max_hands, pointer) -> None:
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
                    key, HandState(model.classes, threshold, dwell, smoothing)
                )
                vector = features.extract(hand.world, hand.is_left)
                best, state.confidence, state.probabilities = model.predict(vector)
                state.gesture = model.classes[best]

                for intent in state.machine.update(state.probabilities, now):
                    if intent.kind == "fired":
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
                del fired_log[:-4]

            # A hand that left the frame must not leave a primed trigger behind.
            for key, state in states.items():
                if key not in seen:
                    state.machine.update(None, now)
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
            box_w = max(20, min(term_w - 4, 78))
            # Budget the text first and give the picture what is left. Sizing
            # the box by a fixed guess pushed the readouts off the bottom of a
            # 24-row terminal, hiding the half that is actually useful.
            box_h = max(4, min(height - _text_rows(states, fired_log), 14))
            if canvas is None or canvas_size != (box_w, box_h):
                canvas = Braille(box_w - 2, box_h - 2)
                canvas_size = (box_w, box_h)
            draw_hands(canvas, hands)

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
    for i, line in enumerate(canvas.text_rows()):
        put(2 + i, 2, "│", DIM)
        put(2 + i, 3, line, CYAN)
        put(2 + i, 2 + box_w - 1, "│", DIM)
    bottom = 2 + len(canvas.text_rows())
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
        put(row, 2, f"{label:<6}", curses.A_BOLD)
        put(row, 9, f"{state.gesture:<12}", GREEN if state.machine.engaged else 0)
        put(row, 22, f"{state.confidence:>4.0%} ")
        put(row, 28, bar(state.confidence, 24), CYAN)
        put(row, 53, marks, YELLOW)
        row += 1
        if state.machine.candidate:
            put(row, 9, f"{state.machine.candidate:<12}", DIM)
            put(row, 28, meter(state.machine.progress, 24), YELLOW)
            row += 1
        # runners-up, so a misread is legible instead of mysterious
        order = np.argsort(state.probabilities)[::-1][1:3]
        for index in order:
            if state.probabilities[index] < 0.02:
                continue
            put(row, 9, f"{model.classes[index]:<12}", DIM)
            put(row, 28, bar(float(state.probabilities[index]), 24), DIM)
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
