"""The live loop: hands in, computer driven.

Two control paths, because they want opposite things.

Discrete gestures (play, desktop, volume) go through `GestureMachine`, which
deliberately makes you hold a pose so that passing shapes are ignored. Waiting
is a feature there.

Continuous control does not go through it at all. A cursor must follow your
finger this frame, and a pinch has to feel like a physical click, so those are
read directly with hysteresis. Routing them through a dwell timer would make
pointing feel like wading.

The pinch is measured geometrically rather than classified, using two
thresholds: close below one, open above a higher one. A single threshold sits
right at the boundary where landmark noise flickers, and a flickering click is
worse than no click at all.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from . import control, draw, features
from .hands import HandTracker
from .machine import GestureMachine
from .model import GestureNet
from .pointer import CursorSmoother, map_to_screen, screen_size
from .schema import INDEX_TIP

PINCH_CLOSE = 0.38    # below this, the fingers are together
PINCH_OPEN = 0.55     # above this, they are apart. The gap is the hysteresis.


def run_live(
    checkpoint: str | Path = "checkpoints/static.pt",
    camera: int = 0,
    width: int = 640,
    live_control: bool = False,
    threshold: float = 0.80,
    dwell: int = 5,
    smoothing: float = 0.6,
    pointer: bool = True,
) -> None:
    """Run it. Actions are printed, not performed, unless live_control is set."""
    model = GestureNet.load(checkpoint)
    machine = GestureMachine(
        model.classes, threshold=threshold, dwell_frames=dwell, smoothing=smoothing
    )
    controller = control.Controller(live=live_control)
    cursor = CursorSmoother()
    screen_w, screen_h = screen_size()

    capture = cv2.VideoCapture(camera)
    if not capture.isOpened():
        raise RuntimeError(
            f"cannot open camera {camera}. macOS needs camera permission for your "
            "terminal: System Settings > Privacy & Security > Camera."
        )

    pinched = False
    started = time.time()
    frames, fps = 0, 0.0

    print("\n  bindings")
    for line in control.describe_bindings():
        print(line)
    print(
        "\n  " + ("DRIVING THE MACHINE" if live_control else "dry run: actions are printed only")
        + "\n  hold open_palm to engage, q to quit\n"
    )

    with HandTracker(max_hands=1) as tracker:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            scale = width / bgr.shape[1]
            bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            bgr = cv2.flip(bgr, 1)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            height = bgr.shape[0]

            now = time.time()
            hands = tracker(rgb, int((now - started) * 1000))
            hand = max(hands, key=lambda h: h.score) if hands else None

            probabilities = None
            gesture, confidence = "-", 0.0
            if hand is not None:
                draw.hand(bgr, hand.image)
                vector = features.extract(hand.world, hand.is_left)
                best, confidence, probabilities = model.predict(vector)
                gesture = model.classes[best]

            for intent in machine.update(probabilities, now):
                if intent.kind == "fired":
                    if control.perform(intent.gesture, controller):
                        print(f"  {intent.gesture}  ->  {control.ACTIONS[intent.gesture].describe}")
                elif intent.kind == "engaged":
                    print("  engaged")
                elif intent.kind == "disengaged":
                    print("  disengaged")
                    cursor.reset()

            # Continuous control, outside the state machine entirely.
            if machine.engaged and hand is not None and pointer:
                distance = features.pinch_distance(hand.world)
                if gesture in control.CONTINUOUS or pinched:
                    nx, ny = hand.image[INDEX_TIP, 0], hand.image[INDEX_TIP, 1]
                    # The frame is already mirrored for display, so no second flip.
                    sx, sy = map_to_screen(nx, ny, screen_w, screen_h, mirrored=False)
                    controller.move_cursor(*cursor(sx, sy))

                if not pinched and distance < PINCH_CLOSE:
                    pinched = True
                    controller.press_mouse(True)
                elif pinched and distance > PINCH_OPEN:
                    pinched = False
                    controller.press_mouse(False)
            elif pinched:
                pinched = False
                controller.press_mouse(False)

            frames += 1
            if frames % 15 == 0:
                fps = frames / (now - started)

            _hud(bgr, model, probabilities, machine, gesture, confidence, fps,
                 live_control, pinched, controller, width, height)
            cv2.imshow("ika", bgr)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                machine.reset()
                cursor.reset()

    if pinched:
        controller.press_mouse(False)
    capture.release()
    cv2.destroyAllWindows()


def _hud(bgr, model, probabilities, machine, gesture, confidence, fps,
         live_control, pinched, controller, width, height) -> None:
    state = "ENGAGED" if machine.engaged else "hold open_palm to engage"
    mode = "LIVE" if live_control else "dry run"
    draw.text(
        bgr,
        [
            f"{state}   {mode}   {fps:.0f} fps",
            f"{gesture} {confidence:.0%}" + ("   pinched" if pinched else ""),
        ],
    )

    # The dwell meter. Without it a gesture that is nearly recognised looks
    # identical to one being ignored, and there is no way to learn the timing.
    if machine.candidate:
        draw.text(bgr, [f"{machine.candidate}"], origin=(10, 64), scale=0.45)
        draw.bar(bgr, machine.progress, (10, 72), width=140)

    # Top three classes, so a misfire is legible rather than mysterious.
    if probabilities is not None:
        order = np.argsort(probabilities)[::-1][:3]
        for row, index in enumerate(order):
            y = height - 70 + row * 20
            draw.text(bgr, [f"{model.classes[index]:<12}"], origin=(width - 190, y + 8), scale=0.42)
            draw.bar(bgr, float(probabilities[index]), (width - 90, y), width=80, height=7)

    for row, entry in enumerate(controller.log[-3:]):
        draw.text(bgr, [entry], origin=(10, height - 60 + row * 16), scale=0.4)
