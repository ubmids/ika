"""A plain-text mode that prints events instead of taking over the screen.

The curses app needs to own a real terminal, which makes it impossible to run
from anywhere that is not one. This mode exists so the thing can be driven and
watched from a pipe, a log, or a conversation: no cursor addressing, no
redraws, just a line whenever something changes.

It also makes a better record. The full-screen version shows you the present
moment and forgets it; this leaves a transcript you can read afterwards to see
exactly what the classifier thought, which is far more useful when a gesture
did not fire and you want to know why.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

from . import control, features
from .hands import HandTracker
from .machine import GestureMachine
from .model import GestureNet
from .pointer import CursorSmoother, map_to_screen, screen_size
from .schema import INDEX_TIP

PINCH_CLOSE = 0.38
PINCH_OPEN = 0.55


def _say(text: str) -> None:
    """Unbuffered, so a reader watching the pipe sees it immediately."""
    print(text, flush=True)


def run_stream(
    checkpoint: str | Path = "checkpoints/static.pt",
    camera: int = 0,
    width: int = 480,
    live_control: bool = False,
    threshold: float = 0.80,
    dwell: int = 5,
    smoothing: float = 0.6,
    max_hands: int = 2,
    seconds: float | None = None,
    heartbeat: float = 1.0,
    pointer: bool = True,
) -> None:
    """Run headless, printing what happens. `seconds` bounds the session."""
    model = GestureNet.load(checkpoint)
    capture = cv2.VideoCapture(camera)
    if not capture.isOpened():
        _say(
            f"cannot open camera {camera}. macOS needs camera permission for "
            "your terminal: System Settings > Privacy & Security > Camera."
        )
        return

    controller = control.Controller(live=live_control)
    cursor = CursorSmoother()
    screen_w, screen_h = screen_size()

    machines: dict[str, GestureMachine] = {}
    pinched: dict[str, bool] = {}
    started = time.time()
    frames, last_beat, with_hand, two_hands = 0, 0.0, 0, 0
    seen_gestures: dict[str, int] = {}
    brightness: list[float] = []

    _say(f"ika: {'DRIVING THE MACHINE' if live_control else 'dry run'}, "
         f"tracking up to {max_hands} hand(s), threshold {threshold}, dwell {dwell}")
    _say("wave at the camera. hold an open palm to engage.")

    # A low detection threshold on purpose. Missing the hand entirely is a
    # far worse failure than tracking it loosely, since everything downstream
    # can filter a noisy landmark but nothing can recover an absent one.
    with HandTracker(max_hands=max_hands, detection_confidence=0.3,
                     tracking_confidence=0.3) as tracker:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            now = time.time()
            if seconds is not None and now - started > seconds:
                break

            scale = width / bgr.shape[1]
            bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            hands = tracker(rgb, int((now - started) * 1000))
            frames += 1
            if hands:
                with_hand += 1
            if len(hands) >= 2:
                two_hands += 1
            if frames % 15 == 0:
                brightness.append(float(rgb.mean()) / 255.0)

            status: list[str] = []
            seen = set()
            for hand in hands:
                label = hand.label
                seen.add(label)
                machine = machines.setdefault(
                    label,
                    GestureMachine(model.classes, threshold=threshold,
                                   dwell_frames=dwell, smoothing=smoothing),
                )
                vector = features.extract(hand.world, hand.is_left)
                best, confidence, probabilities = model.predict(vector)
                gesture = model.classes[best]
                seen_gestures[gesture] = seen_gestures.get(gesture, 0) + 1
                distance = features.pinch_distance(hand.world)

                runner = int(np.argsort(probabilities)[-2])
                status.append(
                    f"{label:<5} {gesture:<11} {confidence:>4.0%}"
                    f"  (then {model.classes[runner]} {probabilities[runner]:.0%})"
                    f"  pinch {distance:.2f}"
                    + (f"  dwell {machine.progress:.0%}" if machine.candidate else "")
                )

                for intent in machine.update(probabilities, now):
                    stamp = time.strftime("%H:%M:%S")
                    if intent.kind == "fired":
                        if control.perform(intent.gesture, controller):
                            _say(f"{stamp}  {label}: {intent.gesture} -> "
                                 f"{control.ACTIONS[intent.gesture].describe}")
                        else:
                            _say(f"{stamp}  {label}: {intent.gesture} "
                                 f"(nothing bound to it)")
                    else:
                        _say(f"{stamp}  {label}: {intent.kind}")

                if machine.engaged and pointer:
                    if gesture in control.CONTINUOUS or pinched.get(label):
                        nx, ny = hand.image[INDEX_TIP, 0], hand.image[INDEX_TIP, 1]
                        controller.move_cursor(
                            *cursor(*map_to_screen(nx, ny, screen_w, screen_h))
                        )
                    if not pinched.get(label) and distance < PINCH_CLOSE:
                        pinched[label] = True
                        controller.press_mouse(True)
                        _say(f"{time.strftime('%H:%M:%S')}  {label}: pinch closed")
                    elif pinched.get(label) and distance > PINCH_OPEN:
                        pinched[label] = False
                        controller.press_mouse(False)
                        _say(f"{time.strftime('%H:%M:%S')}  {label}: pinch opened")

            for label, machine in machines.items():
                if label not in seen:
                    machine.update(None, now)
                    if pinched.get(label):
                        pinched[label] = False
                        controller.press_mouse(False)

            if now - last_beat >= heartbeat:
                last_beat = now
                fps = frames / max(1e-6, now - started)
                engaged = [k for k, m in machines.items() if m.engaged]
                head = (f"{time.strftime('%H:%M:%S')}  {len(hands)} hand(s)  "
                        f"{fps:.0f} fps" + (f"  ENGAGED {','.join(engaged)}" if engaged else ""))
                _say(head if not status else head + "\n    " + "\n    ".join(status))

    capture.release()
    for label, held in pinched.items():
        if held:
            controller.press_mouse(False)

    elapsed = time.time() - started
    _say(f"\nstopped after {elapsed:.0f}s, {frames} frames, "
         f"{frames / max(1e-6, elapsed):.0f} fps")
    _say(f"a hand was visible in {with_hand}/{frames} frames "
         f"({100 * with_hand / max(1, frames):.0f}%), "
         f"two hands in {100 * two_hands / max(1, frames):.0f}%")
    if brightness:
        mean = sum(brightness) / len(brightness)
        note = ("  <- very dark, the landmarker will struggle" if mean < 0.18
                else "  <- very bright" if mean > 0.85 else "")
        _say(f"mean frame brightness {mean:.2f}{note}")
    if seen_gestures:
        total = sum(seen_gestures.values())
        ranked = sorted(seen_gestures.items(), key=lambda kv: -kv[1])
        _say("what it thought it saw, by frame:")
        for name, count in ranked:
            _say(f"    {name:<13} {100 * count / total:>4.0f}%  ({count})")
    else:
        _say("no hand was ever classified. if you were in frame, the detector "
             "is the problem, not the classifier.")
