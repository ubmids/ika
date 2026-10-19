"""The drill session as a runnable thing: camera in, habits out.

Streaming rather than full-screen, so it works from a pipe and leaves a
transcript. The full-screen app would show the present moment and forget it,
and a session's value is the record.

Habits accumulate into a `Profile` on disk, because a tendency seen twenty
times across five sessions is far better evidence than the same tendency seen
four times in one, and until now that evidence died with the process.

The source can be a video file instead of a camera. That is not only a
convenience: it is how the whole loop was checked against real people without
anyone standing in front of a lens, and a file replays on its own clock, so a
recorded round gives the same answer every time.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2

from .body import PoseTracker
from .coach import is_coach
from .cue import Caller, bell, measure
from .drill import DrillReader, read_habits
from .hands import HandTracker
from .profile import Profile, propose
from .sag import current_combo


class _NativeLogsToFile:
    """Send the process's native stderr to a file for the session.

    MediaPipe's C++ side logs telemetry failures and delegate notices straight
    to file descriptor 2, a few a minute, into the middle of the transcript.
    Python-level settings do not reach it. Redirecting the descriptor does,
    and it is restored before any exception propagates, so a real traceback
    still reaches the terminal.
    """

    def __init__(self, path: Path):
        self.path = path

    def __enter__(self):
        import os
        import sys

        sys.stderr.flush()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "ab")
        self._saved = os.dup(2)
        os.dup2(self._file.fileno(), 2)
        return self

    def __exit__(self, *_exc):
        import os
        import sys

        sys.stderr.flush()
        os.dup2(self._saved, 2)
        os.close(self._saved)
        self._file.close()
        return False


def pose_variant(framing: str) -> str:
    """Which pose model a framing reads with.

    Standing back, `full`: on the labelled footage it caught 5 to 9 points
    more punches than `lite` at the same false rate, and at 21 ms a frame it
    still leaves room for 30 fps. Close in, `lite`, because `full` finds
    nothing on a crop that close, measured when the body lane was built.
    """
    return "full" if framing == "stand" else "lite"


def _say(text: str) -> None:
    print(text, flush=True)


def _open(source):
    """A camera index or a path. Returns (capture, is_file)."""
    if isinstance(source, str) and not source.isdigit():
        if not Path(source).exists():
            return None, True
        return cv2.VideoCapture(source), True
    return cv2.VideoCapture(int(source)), False


def run_drill(
    camera=0,
    width: int = 640,
    calibration: float = 3.0,
    profile: str | Path = "data/profiles",
    subject: str = "me",
    seconds: float | None = None,
    report_every: float = 20.0,
    framing: str = "stand",
    voice: bool = True,
    save: bool = True,
    quiet: bool = False,
    live: bool = False,
    ring: bool = True,
) -> dict:
    """Run one session. Returns what happened, for scripts and tests.

    `live` draws the round on a full screen (`ika.live`) instead of printing
    a transcript; the summary is printed either way, after the screen closes.
    `ring` sounds a bell when the round starts and when a timed round ends,
    and only when the voice is on: `--silent` and a file replay are quiet.
    """
    capture, is_file = _open(camera)
    if capture is None or not capture.isOpened():
        if is_file:
            _say(f"  cannot open {camera}")
        else:
            _say(f"  cannot open camera {camera}. macOS needs camera permission for "
                 "your terminal: System Settings > Privacy & Security > Camera.")
        return {}

    store = Path(profile) / f"{subject}.json"
    record = Profile.load(store) if store.exists() else Profile(subject=subject)
    # Coach habits are keyed on a called combo, which a drill never has.
    standing = [h for h in record.habits(now=time.time()) if not is_coach(h.context)]
    if record.log or record.observations:
        _say(f"  {record.sessions() or len(record.log)} earlier session(s) for {subject}")
    for habit in standing[:3]:
        _say(f"    watching for: {habit.describe()}")

    frame_w = capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 640
    frame_h = capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480
    reader = DrillReader(calibration=calibration, framing=framing,
                         aspect=frame_w / frame_h)
    caller = Caller(standing, voice=voice and not is_file)
    ring = ring and voice and not is_file
    view, state = _live_view(live, "drill", frame_w / frame_h, standing)
    note = (lambda text: _note(state, text)) if view is not None else _say
    started = time.time()
    frames, last_report, clock = 0, 0.0, 0.0
    announced = False
    boxing_from = 0.0
    nudged = -1e9

    from contextlib import nullcontext

    hand_tracker = (HandTracker(max_hands=2, detection_confidence=0.3)
                    if framing == "close" else nullcontext(None))
    with _NativeLogsToFile(Path(profile).parent / "logs" / "mediapipe.log"), \
         hand_tracker as hands, \
         PoseTracker(variant=pose_variant(framing), max_bodies=1,
                     detection_confidence=0.4) as pose:
        # A camera round ends when you press Ctrl-C. That is the normal way
        # out, not an abort, so the round is still summarised and saved.
        try:
            with view if view is not None else nullcontext():
                while True:
                    ok, bgr = capture.read()
                    if not ok:
                        break
                    clock = (capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 if is_file
                             else time.time() - started)
                    if seconds is not None and clock > seconds:
                        if ring and reader.calibrated:
                            bell(3)
                        break

                    # Shrunk to at most `width`, never enlarged. 640 because that is the
                    # width the punch reader was measured at; at 480 the pose loses
                    # enough detail that a replay of the same clip read differently.
                    if bgr.shape[1] > width:
                        scale = width / bgr.shape[1]
                        bgr = cv2.resize(bgr, None, fx=scale, fy=scale,
                                         interpolation=cv2.INTER_AREA)
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    stamp = int(clock * 1000)

                    # The hand landmarker only feeds the close reader; standing back
                    # it finds a hand in a third of frames and nothing reads it.
                    seen_hands = hands(rgb, stamp) if framing == "close" else []
                    bodies = pose(rgb, stamp)
                    frames += 1

                    for event in reader.observe(seen_hands, bodies[0] if bodies else None, clock):
                        if view is not None:
                            _show_event(state, event)
                        elif not quiet:
                            _say(f"  {clock:6.1f}s  {event.name}"
                                 + (f"   {event.detail}" if event.detail else ""))
                        cue = caller.update(current_combo(reader.punches()), clock)
                        if cue and view is not None:
                            state.big_text, state.words = cue.words, _setup_words(cue.context)
                            state.said_at = clock
                        elif cue and not quiet:
                            _say(f"  {clock:6.1f}s  >> {cue.words.upper()}   "
                                 f"(after {' then '.join(cue.context)}, {cue.probability:.0%})")

                    # Silence while it waits looks like it is working. If the guard
                    # has not been learned well past the calibration time, say why.
                    if (not reader.calibrated and clock > calibration + 4.0
                            and clock - nudged >= 8.0):
                        nudged = clock
                        note("  still calibrating: "
                             + ("I can see you, but hold your guard still with both hands in shot."
                                if bodies else
                                "no one in view. Step back until head to hips are in shot."))

                    if reader.calibrated and not announced:
                        announced = True
                        boxing_from = clock
                        note(f"  calibrated. resting guard "
                             f"L {reader.baseline.guard_left:+.2f} "
                             f"R {reader.baseline.guard_right:+.2f}. go.")
                        if ring:
                            bell(1)

                    if clock - last_report >= report_every and reader.calibrated:
                        last_report = clock
                        found = read_habits(reader)
                        # This session's own habits join the watch list as they appear.
                        caller.arm(list(standing) + list(found))
                        if view is not None:
                            for habit in found[:2]:
                                note(f"  habit: {habit.describe()}")
                        elif not quiet:
                            _report(found)

                    if view is not None:
                        state.clock = clock
                        if reader.calibrated:
                            state.phase = "box"
                            state.remaining = (seconds - clock) if seconds is not None else None
                            state.elapsed = clock - boxing_from
                        else:
                            state.remaining = max(0.0, calibration - clock)
                        _show_frame(state, reader, bodies, clock,
                                    frames / max(time.time() - started, 1e-6))
                        view.draw(state)
        except KeyboardInterrupt:
            _say("\n  round over.")

    capture.release()
    elapsed = time.time() - started
    if not reader.calibrated:
        _say("\n  never calibrated, so nothing was read or saved. Stand back until the\n"
             "  camera sees you from head to hips, then hold your guard still.")
        return {"events": [], "found": [], "cues": [], "stats": {}, "seconds": clock}
    found = read_habits(reader)
    summary(reader, found, clock, frames / max(elapsed, 1e-6))
    stats = measure(caller.cues, list(reader.events))
    if caller.cues:
        _say(f"  called it {stats['cues']} times; the lapse followed "
             f"{stats['right']} ({stats['precision']:.0%}), "
             f"{stats['median_lead']:.1f}s before the hand went down")

    if save and reader.calibrated:
        now = time.time()
        from . import sag

        record.add(propose(reader.habit_stream()), at=now)
        # Guard habits are proposed at a tighter rate than punch patterns.
        # Simulated at the measured sensor noise over eight six-minute
        # sessions: proposing at 0.2 found a strong habit in 82% of fighters
        # but told 10% of fighters with none that they had one; 0.1 found it
        # in 65% and misled 2%. A false read about your guard is the costlier
        # mistake, so 0.1.
        record.add(sag.read(reader.guard_track, reader.punches(), fdr=sag.PROPOSAL_FDR), at=now)
        from .tell.habits import Finding

        known = {(tuple(h.context), h.then): h for h in list(standing) + list(found)}
        for context, then in record.tracked():
            if is_coach(context):
                continue
            known.setdefault((context, then), Finding(context, then, 0, 0, 0.0, 0.0, 1.0))
        record.record_session(now, clock, reader.counts(),
                              [(h, *reader.rate(h)) for h in known.values()])
        store.parent.mkdir(parents=True, exist_ok=True)
        record.save(store)
        _say(f"\n  saved to {store}. `ika history --subject {subject}` shows the trend.")
    return {"events": list(reader.events), "found": found, "cues": caller.cues,
            "stats": stats, "seconds": clock}


def summary(reader: DrillReader, found, seconds: float, fps: float) -> None:
    counts = reader.counts()
    punches = sum(v for k, v in counts.items() if k.startswith("punch_"))
    drops = sum(v for k, v in counts.items() if k.startswith("guard_down_"))
    minutes = max(seconds / 60.0, 1e-6)
    _say(f"\n  round: {seconds:.0f}s at {fps:.0f} fps")
    _say(f"  {punches} punches ({punches / minutes:.0f} a minute), "
         f"guard dropped {drops} times")
    _report(found)


def _report(found) -> None:
    if not found:
        _say("  no habit yet. Nothing significant beyond chance.")
        return
    _say("  this session:")
    for habit in found[:6]:
        _say(f"    {habit.describe()}")


# --- the live screen --------------------------------------------------------
# Shared with coach mode. Kept here, beside the loop that feeds them, rather
# than in `live`, which only draws and knows nothing about readers.

# How long a call or cue stays bright, then on screen at all.
FRESH_SECONDS = 1.5
SHOWN_SECONDS = 5.0


def _live_view(live: bool, mode: str, aspect: float, standing=()):
    from .live import LiveState, Screen

    state = LiveState(mode=mode, aspect=aspect)
    for habit in list(standing)[:2]:
        _note(state, f"watching for: {habit.describe()}")
    return (Screen() if live else None), state


def _note(state, text: str) -> None:
    """Add a line under the screen, unless it was one of the last few."""
    text = text.strip()
    if text and text not in state.notes[-4:]:
        state.notes.append(text)
    del state.notes[:-8]


def _setup_words(context) -> str:
    """A drill setup in words: ("punch_right", "punch_left") -> "after right, left"."""
    return "after " + ", ".join(c.rsplit("_", 1)[-1] for c in context)


def _show_event(state, event) -> None:
    if event.name in ("punch_left", "punch_right"):
        state.punches.append((event.at, event.name.rsplit("_", 1)[1]))
        state.punch_count += 1
        from .live import TICK_SECONDS
        while state.punches and event.at - state.punches[0][0] > TICK_SECONDS:
            state.punches.pop(0)


def _show_frame(state, reader, bodies, clock: float, fps: float) -> None:
    """The per-frame parts of the screen: the pose, the guard, the call's age."""
    import math

    state.body = bodies[0] if bodies else None
    state.fps = fps
    # The guard meter averages the last few frames of the same shoulder
    # relative read the habit statistics use, so it moves like a hand and
    # does not flicker like a landmark.
    recent = list(reader.guard_track.frames)[-8:]
    for side in ("left", "right"):
        values = [f.height[side] for f in recent if math.isfinite(f.height[side])]
        state.guard[side] = sum(values) / len(values) if values else float("nan")
    age = clock - state.said_at
    state.fresh = age < FRESH_SECONDS
    if state.mode == "drill" and age > SHOWN_SECONDS:
        state.big_text = state.words = ""
