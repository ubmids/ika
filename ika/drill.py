"""The drill loop: watch someone at their laptop, name what they did, find the habit.

This is the product the measurements point at, and it exists because of one
realisation. Everything upstream needed a trained action classifier, and
training one needed labelled footage in laptop framing, which does not exist
publicly and would have to be recorded. But the two events a drill coach
actually needs are **geometric**, not classified:

    a punch is a hand closing on the lens          -> ika/approach.py
    a dropped guard is a wrist below its baseline  -> posture guard_height

Neither needs a model trained on anyone's hands. So the loop closes today.

The events feed `tell/habits.py`, which is where the whole idea lands: not
"you threw a jab" but "you drop your right hand after two punches, 71% of the
time". That is the read, and it is the thing a person cannot see themselves.

Two things are deliberately not hardcoded.

**Guard thresholds are calibrated, not assumed.** A resting guard height of
+0.14 and a dropped one of -0.84 were measured on synthetic bodies, and real
people are built differently and sit at different distances. So the first few
seconds establish this person's own baseline and everything is relative to it.

**Punches require the body in view.** Measured: a lean-in with no body visible
is indistinguishable from a strike, crossover at 1.25/s. The identical hand
sequence reads `closing` without a body and `lean` with one. So a punch is
only ever reported when there is a torso to compare against.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from . import posture
from .approach import ApproachTracker

# Seconds of quiet watching before any event is reported, used to learn this
# person's resting guard height and hand size.
CALIBRATION_SECONDS = 3.0

# A guard counts as dropped once it falls this far below the baseline, in torso
# lengths, and is restored once it comes back within the smaller figure. Two
# thresholds because one makes the state chatter at the boundary, which would
# fill the habit stream with alternating guard events that mean nothing.
GUARD_DROP = 0.28
GUARD_RESTORE = 0.14

# After calibration the resting guard keeps being learned, as a high
# percentile of the last ADAPT_SECONDS of guard heights. Between punches a
# fighter's hands are mostly up, so the 75th percentile sits on their guard
# even when they drop it a quarter of the time. Adapting at all is what makes
# calibration survive a real session: replayed over real footage, a baseline
# fixed in the first three seconds caught 2 of 54 labelled drops, because
# those three seconds were not a guard and nothing ever corrected them. It
# also follows someone who steps closer or further, which changes nothing
# about their guard and everything about where their wrists sit in the image.
ADAPT_SECONDS = 20.0
ADAPT_PERCENTILE = 75.0

# How long a guard has to stay down before it counts as dropped. Without this
# every uppercut and body shot is also a dropped guard, since the wrist goes
# below its resting height on the way out: replayed over a real workout, the
# instantaneous rule reported 105 guard drops in three minutes of punching.
# A dropped guard is a hand that stays down, which is what a coach means.
GUARD_HOLD = 0.4

# Minimum gap between two reported punches. A punch takes about 140 ms, and
# without this the closing signal fires on consecutive frames of one strike.
PUNCH_COOLDOWN = 0.30

# How long an event stream is kept in memory before the oldest go. Long enough
# to mine a session, bounded so a loop left running does not grow for ever.
STREAM_LIMIT = 4000

# Two events this close together are one thing happening, not a sequence.
#
# This is not a tidiness measure. Both hands usually drop together, so
# guard_down_left and guard_down_right land on adjacent frames, and the habit
# miner then reports "after guard_down_left: guard_down_right, 100% of the
# time, 6.7x" as a discovery. It is not a discovery, it is the order the two
# events happened to be emitted in, and a first end-to-end run produced four
# such phantoms alongside the one real habit. Sequential mining over
# simultaneous events invents dependencies.
SIMULTANEOUS_SECONDS = 0.15

# Events that only end a state some earlier event opened. They are recorded,
# and they are kept out of what the habit miner sees.
#
# The first run on a clean subject reported "after guard_down_both:
# guard_up_both, 100% of the time, 7.8x" as a habit. It is not one. A guard
# that is down can only come up, so the miner was rediscovering the state
# machine, and "after guard_up_both: punch_right" followed for the same reason
# from the other side. Mining lapses and strikes alone asks the only question
# a coach cares about, which is what comes before a lapse.
RESTORES = ("guard_up_",)

# What the sequence miner reads. Punches only, since the replay on real
# footage showed the dropped-guard event is too unreliable to mine: labellers
# agreed on half the drops, the detector caught 30% of those, and a simulation
# at those error rates named a planted habit 22% of the time in 24 minutes.
# Guard habits are read by `sag`, from the continuous guard height, instead.
MINED = ("punch_",)


@dataclass(frozen=True)
class Event:
    """Something the subject did, at a time, with how sure we are."""

    name: str
    at: float
    confidence: float
    detail: str = ""


@dataclass
class Baseline:
    """This person's resting posture, learned rather than assumed."""

    guard_left: float = 0.0
    guard_right: float = 0.0
    samples: int = 0
    ready: bool = False
    recent: deque = field(default_factory=deque)

    def track(self, at: float, left: float, right: float) -> None:
        """Keep learning after calibration. See ADAPT_SECONDS."""
        self.recent.append((at, left, right))
        while self.recent and at - self.recent[0][0] > ADAPT_SECONDS:
            self.recent.popleft()
        # Until a few seconds have been seen, the calibration mean stands.
        if len(self.recent) >= 60:
            values = np.asarray([(l, r) for _, l, r in self.recent])
            self.guard_left, self.guard_right = np.percentile(
                values, ADAPT_PERCENTILE, axis=0).tolist()

    def observe(self, left: float, right: float) -> None:
        # Running mean, so calibration costs no storage and cannot be skewed
        # by whichever frame happened to be last.
        self.samples += 1
        k = 1.0 / self.samples
        self.guard_left += (left - self.guard_left) * k
        self.guard_right += (right - self.guard_right) * k


@dataclass
class DrillReader:
    """Turns a live stream of hands and bodies into named events.

    Holds one `ApproachTracker` per hand, because two hands closing on the
    camera are two different punches and sharing a tracker would average them
    into one meaningless signal.
    """

    calibration: float = CALIBRATION_SECONDS
    drop: float = GUARD_DROP
    restore: float = GUARD_RESTORE
    cooldown: float = PUNCH_COOLDOWN
    hold: float = GUARD_HOLD
    # "stand" reads punches from the pose, for shadowboxing a couple of metres
    # back; "close" reads a palm closing on the lens, at arm's length. See
    # `strike.py` for why the close reader fails at shadowboxing distance.
    framing: str = "stand"
    aspect: float = 1.0

    baseline: Baseline = field(default_factory=Baseline, init=False)
    events: deque[Event] = field(default_factory=lambda: deque(maxlen=STREAM_LIMIT),
                                 init=False)
    _approach: dict[str, ApproachTracker] = field(default_factory=dict, init=False)
    _guard_down: dict[str, bool] = field(default_factory=dict, init=False)
    _last_punch: dict[str, float] = field(default_factory=dict, init=False)
    _started: float | None = field(default=None, init=False)
    _below_since: dict[str, float] = field(default_factory=dict, init=False)
    _strike: object = field(default=None, init=False)
    guard_track: object = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.framing not in ("stand", "close"):
            raise ValueError(f"framing is 'stand' or 'close', not {self.framing!r}")
        if self.framing == "stand":
            from .strike import StrikeReader
            self._strike = StrikeReader(aspect=self.aspect)
        from .sag import GuardTrack
        self.guard_track = GuardTrack(aspect=self.aspect)

    @property
    def calibrated(self) -> bool:
        return self.baseline.ready

    def reset(self) -> None:
        self.baseline = Baseline()
        self.events.clear()
        self._approach.clear()
        self._guard_down.clear()
        self._last_punch.clear()
        self._below_since.clear()
        self._started = None
        if self._strike is not None:
            self._strike.reset()
        self.guard_track.reset()

    def observe(self, hands, body, at: float) -> list[Event]:
        """One frame in, any events it produced out.

        `body` may be None, in which case punches are not reported at all
        rather than reported unreliably. That is not caution for its own sake:
        without a torso to compare against, a hard lean forward reads as a
        strike, and a coach told they punched when they leaned would rightly
        stop trusting the thing.
        """
        if self._started is None:
            self._started = at

        found: list[Event] = []
        self.guard_track.update(body, at)

        guard = self._guard_heights(body)
        if guard is not None:
            left, right = guard
            if at - self._started < self.calibration:
                self.baseline.observe(left, right)
                return []
            if not self.baseline.ready:
                self.baseline.ready = True
            self.baseline.track(at, left, right)
            found += self._guard_events(left, right, at)

        found += self._punch_events(hands, body, at)
        emitted = [self._append(event) for event in found]
        return [event for event in emitted if event is not None]

    def _append(self, event: Event) -> Event | None:
        """Add an event, merging it with its opposite side if that just fired.

        Merging happens in the stream rather than within one frame, because the
        two hands do not cross the threshold on the same frame: they land one
        or two apart, so a same-frame merge would never fire. When it does
        merge, the already-stored event is replaced and None is returned, since
        one lapse should be reported once.
        """
        stem, _, side = event.name.rpartition("_")
        if side in ("left", "right") and self.events:
            previous = self.events[-1]
            other = "right" if side == "left" else "left"
            simultaneous = (previous.name == f"{stem}_{other}"
                            and event.at - previous.at <= SIMULTANEOUS_SECONDS)
            if simultaneous and stem == "punch":
                # Nobody throws both hands at once. Two arms firing together is
                # one punch and the other arm swinging with the body's turn,
                # which is the commonest false punch on real footage, so the
                # stronger call stands and the other is dropped. Merging them
                # into a "punch_both" put an event in the stream that no one
                # ever throws.
                if event.confidence > previous.confidence:
                    self.events[-1] = event
                    return event
                return None
            if simultaneous:
                merged = Event(f"{stem}_both", previous.at,
                               min(previous.confidence, event.confidence),
                               "both sides")
                self.events[-1] = merged
                return None
        self.events.append(event)
        return event

    def _guard_heights(self, body) -> tuple[float, float] | None:
        """Both guard heights, or None if the landmarker was guessing.

        Gated on visibility because a pose model does not withhold a joint it
        cannot see, it extrapolates one, and a guard read off an invented wrist
        is a number with nothing behind it.
        """
        if body is None:
            return None
        features = posture.extract(body.world, body.visibility)
        checks = [
            posture.confidence(body.visibility, posture.DEPENDS_ON[key])
            for key in ("left_guard_height", "right_guard_height")
        ]
        if min(checks) < 0.5:
            return None
        return (float(features[posture.DERIVED["left_guard_height"]]),
                float(features[posture.DERIVED["right_guard_height"]]))

    def _guard_events(self, left: float, right: float, at: float) -> list[Event]:
        out = []
        for side, value, rest in (("left", left, self.baseline.guard_left),
                                  ("right", right, self.baseline.guard_right)):
            fallen = rest - value
            was_down = self._guard_down.get(side, False)
            if fallen > self.drop:
                self._below_since.setdefault(side, at)
            else:
                self._below_since.pop(side, None)
            held = at - self._below_since.get(side, at)
            if not was_down and fallen > self.drop and held >= self.hold:
                self._guard_down[side] = True
                # Stamped when the hand went down, not when the hold was
                # satisfied, so the stream keeps the real order of events.
                out.append(Event(f"guard_down_{side}", self._below_since[side],
                                 min(1.0, fallen / self.drop), f"{fallen:.2f} below rest"))
            elif was_down and fallen < self.restore:
                self._guard_down[side] = False
                out.append(Event(f"guard_up_{side}", at, 1.0, ""))
        return out

    def _punch_events(self, hands, body, at: float) -> list[Event]:
        if body is None:
            # Deliberate: see observe(). A lean is not a punch.
            return []
        if self._strike is not None:
            return [Event(f"punch_{s.side}", s.at, s.score, "")
                    for s in self._strike.observe(body, at)]
        out = []
        for hand in hands or []:
            side = "left" if getattr(hand, "is_left", False) else "right"
            tracker = self._approach.setdefault(side, ApproachTracker())
            verdict = tracker.update(hand, at, body=body)
            if not verdict.closing:
                continue
            if at - self._last_punch.get(side, -1e9) < self.cooldown:
                continue
            self._last_punch[side] = at
            out.append(Event(f"punch_{side}", at, verdict.confidence,
                             f"{verdict.rate:.1f}/s"))
        return out

    def stream(self) -> list[str]:
        """Event names in order, which is what the habit miner consumes."""
        return [event.name for event in self.events]

    def habit_stream(self) -> list[str]:
        """The stream the sequence miner reads: the punches, in order."""
        return [event.name for event in self.events if event.name.startswith(MINED)]

    def punches(self) -> list[tuple[float, str]]:
        return [(event.at, event.name[len("punch_"):]) for event in self.events
                if event.name.startswith("punch_")]

    def rate(self, habit) -> tuple[int, int]:
        """How often a known habit happened this session, as (hits, support).

        Counted the way the habit was found: a punch pattern from the stream, a
        guard habit from where the hand sat after each occasion of its setup.
        A session where the setup came up and the habit did not is a zero, not
        a silence, which is what lets `ika history` show a habit fading.
        """
        from .profile import rate_in
        from . import sag

        if habit.then.startswith("punch_"):
            return rate_in(self.habit_stream(), tuple(habit.context), habit.then)
        side = habit.then.rsplit("_", 1)[1]
        mine = [o for o in sag.occasions(self.guard_track, self.punches())
                if o.context == tuple(habit.context) and o.side == side]
        return sum(o.down for o in mine), len(mine)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for event in self.events:
            out[event.name] = out.get(event.name, 0) + 1
        return out


def read_habits(reader: DrillReader, fdr: float = 0.05):
    """Every habit the session supports: punch patterns and guard habits.

    Kept as a function rather than a method so the reader stays a sensor and
    the statistics stay where the tests for them already live. Both kinds come
    back as `tell.habits.Finding`, held to the same false-discovery rate.
    """
    from . import sag
    from .tell.habits import mine

    found = mine(reader.habit_stream(), fdr=fdr) + sag.read(reader.guard_track,
                                                           reader.punches(), fdr=fdr)
    return sorted(found, key=lambda f: -f.lift)
