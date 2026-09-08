"""The drill session as a runnable thing: camera in, habits out.

Streaming rather than full-screen, so it works from a pipe and leaves a
transcript. The full-screen app would show the present moment and forget it,
and a session's value is the record.

Habits accumulate into a `Profile` on disk, because a tendency seen twenty
times across five sessions is far better evidence than the same tendency seen
four times in one, and until now that evidence died with the process.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from .body import PoseTracker
from .drill import DrillReader, read_habits
from .hands import HandTracker
from .profile import Profile


def _say(text: str) -> None:
    print(text, flush=True)


def run_drill(
    camera: int = 0,
    width: int = 480,
    calibration: float = 3.0,
    profile: str | Path = "data/profiles",
    subject: str = "me",
    seconds: float | None = None,
    report_every: float = 20.0,
) -> None:
    capture = cv2.VideoCapture(camera)
    if not capture.isOpened():
        _say(f"  cannot open camera {camera}. macOS needs camera permission for "
             "your terminal: System Settings > Privacy & Security > Camera.")
        return

    store = Path(profile) / f"{subject}.json"
    record = Profile.load(store) if store.exists() else Profile(subject=subject)
    if record.observations:
        _say(f"  loaded {record.sessions()} earlier session(s) for {subject}")

    reader = DrillReader(calibration=calibration)
    started = time.time()
    frames, last_report = 0, started
    announced = False

    with HandTracker(max_hands=2, detection_confidence=0.3) as hands, \
         PoseTracker(variant="lite", max_bodies=1, detection_confidence=0.4) as pose:
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
            stamp = int((now - started) * 1000)

            seen_hands = hands(rgb, stamp)
            bodies = pose(rgb, stamp)
            frames += 1

            for event in reader.observe(seen_hands, bodies[0] if bodies else None,
                                        now - started):
                _say(f"  {time.strftime('%H:%M:%S')}  {event.name}"
                     + (f"   {event.detail}" if event.detail else ""))

            if reader.calibrated and not announced:
                announced = True
                _say(f"  calibrated. resting guard "
                     f"L {reader.baseline.guard_left:+.2f} "
                     f"R {reader.baseline.guard_right:+.2f}. go.")

            if now - last_report >= report_every and reader.calibrated:
                last_report = now
                _report(reader)

    capture.release()
    elapsed = time.time() - started
    _say(f"\n  {elapsed:.0f}s, {frames} frames, {frames/max(elapsed,1e-6):.0f} fps")
    _say(f"  events: {reader.counts() or 'none'}")

    found = read_habits(reader)
    _report(reader, found)
    if found:
        record.add(found, at=time.time())
        record.save(store)
        _say(f"\n  saved to {store} ({record.sessions()} session(s))")
        standing = record.habits(now=time.time())
        if standing:
            _say("\n  standing habits, across every session:")
            for habit in standing[:6]:
                _say(f"    {habit.describe()}")


def _report(reader: DrillReader, found=None) -> None:
    found = read_habits(reader) if found is None else found
    if not found:
        _say("  no habit yet. Nothing significant beyond chance.")
        return
    _say("  this session:")
    for habit in found[:6]:
        _say(f"    {habit.describe()}")
