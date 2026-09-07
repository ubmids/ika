"""Collecting your own gestures.

The variation you capture here is the variation the model will tolerate. Record
every sample with your hand in one spot at one angle and the classifier learns
that spot and that angle, and `features` invariance can only carry it so far
past the data it actually saw. So the recorder actively nags you to move: it
captures over several seconds rather than in a single frame, and tells you to
rotate, lean in and lean back while it does.

Only the highest-confidence hand is stored per frame. Capturing both would
label whatever your other hand happens to be doing as the gesture you are
performing, which is a quiet way to poison a dataset.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from . import draw
from .dataset import Dataset
from .hands import HandTracker
from .schema import STATIC_GESTURES

PROMPTS = [
    "rotate your wrist slowly",
    "move closer, then further away",
    "drift around the frame",
    "tilt your hand left and right",
    "try it with your other hand",
]


def run_recorder(
    out: str | Path = "data/session.npz",
    gestures: list[str] | None = None,
    seconds: float = 4.0,
    camera: int = 0,
    width: int = 640,
) -> None:
    """Interactive capture. Space records the current gesture, s saves."""
    gestures = gestures or list(STATIC_GESTURES)
    out = Path(out)

    marks: list[np.ndarray] = []
    lefts: list[bool] = []
    labels: list[int] = []
    batches: list[int] = []          # sizes, so the last one can be undone

    capture = cv2.VideoCapture(camera)
    if not capture.isOpened():
        raise RuntimeError(
            f"cannot open camera {camera}. macOS needs camera permission for your "
            "terminal: System Settings > Privacy & Security > Camera."
        )

    index = 0
    recording_until = 0.0
    countdown_until = 0.0
    started = time.time()

    with HandTracker(max_hands=1) as tracker:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            scale = width / bgr.shape[1]
            bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            bgr = cv2.flip(bgr, 1)   # mirror, so moving right looks like right
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            hands = tracker(rgb, int((time.time() - started) * 1000))
            best = max(hands, key=lambda h: h.score) if hands else None
            if best is not None:
                draw.hand(bgr, best.image)

            now = time.time()
            target = gestures[index]
            counts = {g: labels.count(i) for i, g in enumerate(gestures)}

            state = "idle"
            if now < countdown_until:
                state = "countdown"
            elif now < recording_until:
                state = "recording"
                if best is not None:
                    marks.append(best.world)
                    lefts.append(best.is_left)
                    labels.append(index)
                    batches[-1] += 1

            if state == "countdown":
                remaining = countdown_until - now
                draw.text(bgr, [f"{remaining:.0f}"], origin=(width // 2 - 10, 90), scale=2.0)
                banner = f"get ready: {target}"
            elif state == "recording":
                fraction = 1.0 - (recording_until - now) / seconds
                draw.bar(bgr, fraction, (10, bgr.shape[0] - 24), width=width - 20, height=10)
                prompt = PROMPTS[int(fraction * len(PROMPTS)) % len(PROMPTS)]
                banner = f"recording {target}   {prompt}"
            else:
                banner = f"next: {target}   space to record"

            draw.text(
                bgr,
                [
                    banner,
                    f"{target}: {counts.get(target, 0)} samples    total {len(labels)}",
                    "space record   tab next   z undo   s save   q quit",
                ],
            )
            cv2.imshow("ika recorder", bgr)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" ") and state == "idle":
                countdown_until = now + 2.0
                recording_until = countdown_until + seconds
                batches.append(0)
            elif key == 9:  # tab
                index = (index + 1) % len(gestures)
            elif key == ord("z") and batches:
                size = batches.pop()
                if size:
                    del marks[-size:], lefts[-size:], labels[-size:]
            elif key == ord("s") and labels:
                _save(out, marks, lefts, labels, gestures)

    capture.release()
    cv2.destroyAllWindows()

    if labels:
        _save(out, marks, lefts, labels, gestures)
    else:
        print("  nothing recorded")


def _save(out: Path, marks, lefts, labels, gestures) -> None:
    Dataset(
        landmarks=np.array(marks, dtype=np.float32),
        is_left=np.array(lefts, dtype=bool),
        labels=np.array(labels, dtype=np.int64),
        classes=list(gestures),
    ).save(out)
    counts = {g: labels.count(i) for i, g in enumerate(gestures)}
    print(f"  saved {len(labels)} samples to {out}")
    for name, n in counts.items():
        flag = "" if n >= 100 else "   <- thin, record more"
        print(f"    {name:<14} {n:>5}{flag}")
