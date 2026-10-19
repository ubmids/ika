"""Coach mode as a runnable thing: ika calls combos, you throw them.

The same plumbing as the drill (`drill_app`): the camera or a recorded file,
the same pose model, the same calibration on a held guard, the same punch
reader and guard track, the same profile on disk. What changes is who
decides the setup. Here the calls are made on a schedule (`coach.schedule`),
each one is checked for a response, and the guard read is taken after it.

A replayed file runs the schedule on the file's own clock. The person in the
file never heard those calls, so a replay is a test of the plumbing and of the
null, not of anyone's habits; it is how this was run without a camera.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2

from . import coach, sag
from .body import PoseTracker
from .cue import _say as speak
from .cue import _speaker
from .drill import DrillReader
from .drill_app import _NativeLogsToFile, _open, _say, pose_variant
from .profile import Profile


class _Voice:
    """Speaks a line, or only prints it. Never both silent and unprinted."""

    def __init__(self, on: bool):
        self.on = on and _speaker() is not None

    def __call__(self, words: str) -> None:
        if self.on:
            speak(words)


def run_coach(
    camera=0,
    width: int = 640,
    calibration: float = 3.0,
    profile: str | Path = "data/profiles",
    subject: str = "me",
    seconds: float = 180.0,
    rounds: int = 1,
    rest: float = 60.0,
    combos: list[str] | None = None,
    seed: int | None = None,
    voice: bool = True,
    save: bool = True,
    quiet: bool = False,
) -> dict:
    """Run one coach session of `rounds` rounds. Returns what happened."""
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
    now = time.time()
    standing = [h for h in record.habits(now=now) if coach.is_coach(h.context)]
    seed = int(now) % 100_000 if seed is None else int(seed)
    chosen = ([coach.Combo.parse(c) for c in combos] if combos
              else coach.plan(record, now, seed=seed))
    remind = coach.reminders(standing)
    coached = [row for row in record.log if row.get("mode") == "coach"]
    if coached:
        _say(f"  {len(coached)} earlier coach session(s) for {subject}")
    for habit in standing[:3]:
        _say(f"    known: {habit.describe()}")
    _say(f"  this session: {', '.join(c.name for c in chosen)}  (seed {seed})")
    per = coach.reps_per_combo(chosen, seconds)
    _say(f"  {rounds} round(s) of {seconds:.0f}s, about {per:.0f} calls of each combo a round")

    frame_w = capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 640
    frame_h = capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480
    reader = DrillReader(calibration=calibration, framing="stand",
                         aspect=frame_w / frame_h)
    say = _Voice(voice and not is_file)
    started = time.time()
    clock, frames = 0.0, 0
    calls: list[coach.Call] = []     # the whole session's schedule, once it starts
    made: list[coach.Call] = []      # the calls actually said
    marks: list[tuple[float, str]] = []
    nudged = -1e9
    begun = None

    with _NativeLogsToFile(Path(profile).parent / "logs" / "mediapipe.log"), \
         PoseTracker(variant=pose_variant("stand"), max_bodies=1,
                     detection_confidence=0.4) as pose:
        try:
            while True:
                ok, bgr = capture.read()
                if not ok:
                    break
                clock = (capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 if is_file
                         else time.time() - started)
                if bgr.shape[1] > width:
                    scale = width / bgr.shape[1]
                    bgr = cv2.resize(bgr, None, fx=scale, fy=scale,
                                     interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                bodies = pose(rgb, int(clock * 1000))
                frames += 1
                for event in reader.observe([], bodies[0] if bodies else None, clock):
                    if not quiet and event.name.startswith("punch_"):
                        _say(f"  {clock:6.1f}s  {event.name}")

                if not reader.calibrated:
                    if clock > calibration + 4.0 and clock - nudged >= 8.0:
                        nudged = clock
                        _say("  still calibrating: "
                             + ("I can see you, but hold your guard still with both "
                                "hands in shot." if bodies else
                                "no one in view. Step back until head to hips are in shot."))
                    continue

                if begun is None:
                    begun = clock
                    calls, marks = _session(chosen, seconds, rounds, rest, seed,
                                            clock + coach.LEAD_IN)
                    calls = coach.with_reminders(calls, remind)
                    _say(f"  calibrated. resting guard L {reader.baseline.guard_left:+.2f} "
                         f"R {reader.baseline.guard_right:+.2f}.")

                while marks and marks[0][0] <= clock:
                    _at, words = marks.pop(0)
                    _say(f"  {clock:6.1f}s  [{words}]")
                    say(words)
                    if words.endswith("rest"):
                        # Between rounds, what this session has shown so far
                        # joins what history knows, so the next round's calls
                        # for a combo that is already a habit carry the reminder.
                        so_far = coach.align(made, reader.guard_track, reader.punches())
                        found = coach.read(so_far)
                        for f in found[:3]:
                            _say(f"    so far: {coach.describe(f)}")
                        remind = coach.reminders(list(standing) + list(found))
                        calls = made + coach.with_reminders(calls[len(made):], remind)
                while len(made) < len(calls) and calls[len(made)].at <= clock:
                    call = calls[len(made)]
                    made.append(call)
                    _say(f"  {clock:6.1f}s  >> {call.words.upper()}")
                    say(call.words)
                if begun is not None and not marks and len(made) == len(calls):
                    # The last call's guard window has to be seen before stopping.
                    last = calls[-1] if calls else None
                    if last is None or clock > last.at + coach.gap(last.combo):
                        break
        except KeyboardInterrupt:
            _say("\n  round over.")

    capture.release()
    elapsed = time.time() - started
    if not reader.calibrated:
        _say("\n  never calibrated, so nothing was read or saved. Stand back until the\n"
             "  camera sees you from head to hips, then hold your guard still.")
        return {"reps": [], "found": [], "calls": [], "seconds": clock}

    # A call is only judged once its guard window has been seen in full.
    judged = [c for c in made if c.at + coach.gap(c.combo) <= clock + 1e-6]
    reps = coach.align(judged, reader.guard_track, reader.punches())
    found = coach.read(reps)
    report(reps, found, clock, frames / max(elapsed, 1e-6))

    if save and reps:
        stamp = time.time()
        record.add(coach.read(reps, fdr=sag.PROPOSAL_FDR), at=stamp)
        known = {(tuple(h.context), h.then) for h in list(standing) + list(found)}
        known |= {(c, t) for c, t in record.tracked() if coach.is_coach(c)}
        from .tell.habits import Finding

        rates = [(Finding(c, t, 0, 0, 0.0, 0.0, 1.0), *coach.rate(reps, c, t))
                 for c, t in sorted(known)]
        counts = dict(reader.counts())
        counts["calls"] = len(reps)
        counts["answered"] = sum(r.answered for r in reps)
        record.record_session(stamp, clock, counts, rates, mode="coach")
        store.parent.mkdir(parents=True, exist_ok=True)
        record.save(store)
        _say(f"\n  saved to {store}. `ika history --subject {subject}` shows the trend.")
    return {"reps": reps, "found": found, "calls": made, "seconds": clock,
            "combos": [c.name for c in chosen]}


def _session(combos, seconds, rounds, rest, seed, start):
    """Every call of every round, and the spoken marks around them."""
    calls, marks = [], []
    for r in range(rounds):
        begin = start + r * (seconds + rest)
        marks.append((begin - coach.LEAD_IN, f"round {r + 1}, box" if rounds > 1 else "box"))
        round_calls = coach.schedule(combos, seconds, seed=seed * 101 + r, start=begin)
        calls += round_calls
        end = (round_calls[-1].at + coach.gap(round_calls[-1].combo)) if round_calls else begin
        marks.append((end, "time" if r == rounds - 1 else "time. rest"))
    return calls, sorted(marks)


def report(reps, found, seconds: float, fps: float) -> None:
    answered = sum(r.answered for r in reps)
    read = sum(r.read for r in reps)
    _say(f"\n  session: {seconds:.0f}s at {fps:.0f} fps")
    _say(f"  {len(reps)} calls, {answered} answered ({answered / max(len(reps), 1):.0%}), "
         f"{len(reps) - answered} with nothing thrown in time, {read} with a guard read")
    by = {}
    for r in reps:
        by.setdefault(r.call.combo.name, []).append(r)
    for name, mine in sorted(by.items()):
        occ = coach.occasions(mine)
        cells = []
        for side in sag.WRISTS:
            h = [o.height for o in occ if o.side == side]
            cells.append(f"{side} {sum(h) / len(h):+.2f}" if h else f"{side}    -")
        _say(f"    {name:9s} called {len(mine):3d}, answered "
             f"{sum(r.answered for r in mine):3d}   guard after: {'  '.join(cells)}")
    if not found:
        _say("  no habit yet. No called combo leaves a hand lower than the others, "
             "beyond chance.")
        return
    _say("  this session:")
    for f in found[:6]:
        _say(f"    {coach.describe(f)}")
