"""Where a movement starts and where it ends, in a stream nobody cut up first.

Every result in this project so far was measured on pre-segmented input.
`bodyaction.make` returns one action per list, `trajectory.make` returns one
gesture per list, and `machine.py`, `tell/early.py` and `live_commit.py` were
all scored on windows whose boundaries somebody had already drawn. Live there
is nobody to draw them. The camera delivers one unbroken stream, and finding
the edges of a movement inside it is the missing piece.

The budget is the one `tell/lead.py` measured. The median gap between one
action ending and the next starting is 263 ms, confirming a movement the way
`machine.py` does costs 250 ms, and committing early costs 100 ms. So a
segmenter that will not say anything about a movement until it is over has
spent the whole budget before the answer exists. That shapes the design: the
open segment is readable from the frame it opens (`current`), and closing it is
a separate, later report (`update`). Nothing downstream has to wait for the
close.

What the mechanisms are for, and which failure each one prevents:

**Two thresholds, not one.** Enter on `active`, leave on `quiet`. With a single
threshold, activity sitting near the line crosses it on noise several times per
second and the stream comes back as dozens of one-frame segments. The gap
between the two is the whole point, exactly as `app.py` reads a pinch and
`machine.py` clears its dwell.

**A minimum duration.** Noise produces brief spikes that clear `active` for a
frame or two. A jab runs 140 ms to 240 ms (`bodyaction.ACTIONS`), so anything
shorter than `min_seconds` is thrown away rather than reported as a movement
that nobody made.

**A bridge across dips.** A punch is not monotone. Velocity passes through
close to zero at full extension, before the retraction, and a segmenter that
closed on the first quiet frame would report two half-punches instead of one
punch. So leaving requires the stream to stay quiet for `bridge` seconds. The
cost is honest and bounded: the close is reported `bridge` late, which is why
`end` is backdated to the last frame that was actually moving rather than set
to the frame that noticed.

**No smoother.** `live_commit.py` already measured what an exponential smoother
costs: it lags by construction. The bridge does the job a smoother would do at
the boundary without adding delay at the onset, which is the edge that has to
be cheap.

**Bounded memory.** This runs as long as the camera does. Nothing here grows
with the length of the stream: the open segment is kept as running aggregates
rather than as frames, the pre-roll used to backdate a start is a fixed-length
deque, and only the last `RECENT_SEGMENTS` closed segments are retained. A
plain list of every segment would be fine for a test and a leak for a session.

**Energy is an integral, not a sum.** `energy` is activity multiplied by
elapsed time, so it means the same thing at 30 fps and at 60 fps. A running
total of per-frame activity would instead be a frame counter wearing a
different name, which is the exact mistake `motion.MOTION_DIM` documents: the
last time window length leaked in as a feature, the model learned to lean on it
and then failed on any other window length.

Duration is deliberately not capped. A cap would make the length of a movement
a property of the segmenter rather than of the movement, which is the same
mistake in a new place. Someone holding a movement for a minute gets a segment
a minute long, and memory does not care because only aggregates are kept.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, fields

import numpy as np

# Enter a movement here, leave it here. Both are in the 0 to 1 units that
# `activity_of` produces. The distance between them is the hysteresis, and it
# is wide because the failure it prevents is not subtle: measured on a burst
# with noisy edges, one threshold at 0.25 emits 11 segments where these two
# emit 1.
ACTIVE_ACTIVITY = 0.35
QUIET_ACTIVITY = 0.15

# Shorter than this and it was not a movement. Set below the fastest real
# action in `bodyaction.ACTIONS` (a 140 ms jab) and above the two or three
# frames a noise spike lasts at 30 fps.
MIN_MOVEMENT_SECONDS = 0.08

# How long the stream has to stay quiet before a movement is called finished.
# 100 ms covers the mid-flight dip of a punch (about two frames at 30 fps at
# full extension) and is the same 100 ms that early commitment costs in
# `live_commit.py`, so a close still lands inside the 263 ms gap.
DIP_BRIDGE_SECONDS = 0.10

# How far back a start can be backdated. A movement is entered on `active`,
# but it began earlier, when activity first left the baseline, so the frames
# between `quiet` and `active` are held here and folded in on entry. Bounded
# for two reasons: memory is fixed, and a slow drift that spends two seconds
# creeping over `quiet` must not backdate an onset two seconds into the past.
PREROLL_FRAMES = 8

# How many closed segments are kept for inspection. Recent, not all, because
# a camera left running for a day would otherwise accumulate every segment of
# the day in a list nobody reads.
RECENT_SEGMENTS = 32

# A gap longer than this is a break in the stream, not a slow frame: a dropped
# tracker, a stalled pipeline, a paused camera. Frames from before such a gap
# describe a different movement, the same judgement `live_commit.py` makes when
# the subject leaves the frame, so the open segment is closed rather than
# stretched across the hole.
STALE_GAP_SECONDS = 0.5

# The span `activity_of` wants between its two feature vectors. Differencing
# adjacent frames divides by 33 ms and so multiplies per-frame landmark noise
# by 30: measured on `bodyaction`, an idle body reads 3.6 and a jab reads 4.0,
# which is not a signal. Over a 100 ms span the same idle reads 0.7 and the
# same jab reads 4.6. The noise does not grow with the span but the movement
# does, and 100 ms is short enough to stay well inside the 263 ms gap.
ACTIVITY_SPAN_SECONDS = 0.1

# The rate of change at which `activity_of` reads 0.5. Calibrated so both lanes
# land in the same range: over a 100 ms span, an idle body measures about 0.7
# and a jab about 4.6, an idling hand about 0.5 and a swipe about 2.6.
HALF_ACTIVITY_RATE = 2.5

# Smallest usable time step. Same guard as `bodymotion` and `motion` use, so a
# repeated timestamp divides by this instead of by zero.
MIN_DT = 1e-4


@dataclass(frozen=True)
class Segment:
    """One movement, bounded.

    `end` is None while the movement is still underway. That is the whole
    reason this is readable live: a consumer gets the start, the peak so far
    and the energy so far without waiting for the close.

    The fields stop here on purpose. `frames` and the duration implied by
    `start` and `end` are the only counts of time in the object, and nothing
    derived from them is passed on as a description of the movement.
    `motion.MOTION_DIM` records what happened the last time window length
    reached a model: every training window was the same length, the model
    learned to lean on it, and a shorter window at inference read as
    out-of-distribution and came back "none" with total confidence.
    """

    start: float          # seconds, backdated to when activity left the baseline
    end: float | None     # None while still underway
    peak: float           # seconds, when activity was highest
    frames: int           # samples inside the movement
    energy: float         # integral of activity over time, in activity-seconds

    @property
    def duration(self) -> float | None:
        """Seconds from start to end, or None while it is still running."""
        return None if self.end is None else self.end - self.start

    @property
    def milliseconds(self) -> float | None:
        """Duration in the unit `lead.py` states the 263 ms gap in."""
        d = self.duration
        return None if d is None else d * 1000.0


SEGMENT_FIELDS = tuple(f.name for f in fields(Segment))


class Segmenter:
    """Movement boundaries from a stream of activity values.

    Feed it one activity value per frame with `update`. It returns a `Segment`
    on the frame a movement is judged to have finished, and None otherwise.
    While a movement is underway, `current` is the open segment, and it is
    usable immediately: waiting for `update` to return would cost the whole
    263 ms budget.

    Activity is any scalar where bigger means more is happening.
    `activity_of` builds one from movement features, but a stream that already
    has a motion energy can be fed straight in. The thresholds are in the same
    units as whatever is fed in, so a differently scaled input needs
    differently scaled thresholds.
    """

    def __init__(self, quiet: float = QUIET_ACTIVITY,
                 active: float = ACTIVE_ACTIVITY,
                 min_seconds: float = MIN_MOVEMENT_SECONDS,
                 bridge: float = DIP_BRIDGE_SECONDS):
        if not active > quiet:
            # Equal thresholds are a single threshold, which is the failure
            # this class exists to avoid, so it is refused at construction
            # rather than discovered in a stream of one-frame segments.
            raise ValueError(f"active ({active}) must exceed quiet ({quiet})")
        if min_seconds < 0 or bridge < 0:
            raise ValueError("min_seconds and bridge cannot be negative")

        self.quiet = float(quiet)
        self.active = float(active)
        self.min_seconds = float(min_seconds)
        self.bridge = float(bridge)

        # Bounded on purpose. See RECENT_SEGMENTS and PREROLL_FRAMES.
        self.recent: deque[Segment] = deque(maxlen=RECENT_SEGMENTS)
        self._preroll: deque[tuple[float, float, float]] = deque(maxlen=PREROLL_FRAMES)

        self.rejected = 0        # how many segments were dropped as too short
        self._last_at: float | None = None
        self._clear_open()

    # ------------------------------------------------------------------ state

    def reset(self) -> None:
        """Forget everything, including the history. A new stream, not a gap."""
        self.recent.clear()
        self.rejected = 0
        self._forget()

    def _forget(self) -> None:
        """Drop the movement in progress and the pre-roll behind it.

        Used at a break in the stream as well as at a reset, because activity
        from before a dropout describes a different movement. `live_commit.py`
        clears its evidence for the same reason: no half-built segment may
        survive a hole in the input.
        """
        self._preroll.clear()
        self._last_at = None
        self._clear_open()

    def _clear_open(self) -> None:
        self._start: float | None = None
        self._peak_at = 0.0
        self._peak = 0.0
        self._frames = 0
        self._energy = 0.0
        self._moving_at: float | None = None    # last sample above `quiet`
        self._quiet_since: float | None = None  # start of the current quiet run
        self._quiet_frames = 0
        self._quiet_energy = 0.0

    @property
    def underway(self) -> bool:
        """Whether a movement is open right now."""
        return self._start is not None

    @property
    def current(self) -> Segment | None:
        """The open segment, or None. Readable before the movement ends.

        Includes the frames of a dip that has not yet been bridged, because
        while a movement is underway those frames are part of it. If the dip
        turns out to be the end, the closed segment reports `end` at the last
        moving frame and leaves that quiet tail out.
        """
        if self._start is None:
            return None
        return Segment(
            start=self._start,
            end=None,
            peak=self._peak_at,
            frames=self._frames + self._quiet_frames,
            energy=self._energy + self._quiet_energy,
        )

    # ----------------------------------------------------------------- update

    def update(self, activity: float, at: float) -> Segment | None:
        """Feed one frame's activity. Returns a Segment when one closes.

        A returned segment has already passed the minimum duration, so a caller
        never sees the noise spikes. Segments that were rejected are counted in
        `rejected` rather than reported, since a spike is not news.
        """
        activity = float(activity)
        if not math.isfinite(activity):
            # A NaN out of a division by a vanished scale is a broken read, not
            # a large movement. Treating it as quiet keeps it out of `peak` and
            # `energy`, where one non-finite value would poison every later
            # comparison.
            activity = 0.0

        closed: Segment | None = None
        if self._last_at is not None and (
                at < self._last_at or at - self._last_at > STALE_GAP_SECONDS):
            # A break in the stream, or a clock that went backwards. Close what
            # is open on its own evidence and start again from here, rather
            # than bridging a movement across a hole of unknown length.
            closed = self._close()
            self._forget()

        gap = 0.0 if self._last_at is None else max(at - self._last_at, 0.0)
        area = activity * gap
        self._last_at = at

        if self._start is None:
            if activity >= self.active:
                self._open(at, activity, area)
            else:
                # Held in case this turns out to be the run-up to a movement.
                self._preroll.append((at, activity, area))
            return closed

        if activity > self.quiet:
            if self._quiet_frames:
                # The dip came back up, so it was mid-flight and not the end.
                # Folding the dip in rather than discarding it is what keeps a
                # punch one punch instead of two halves.
                self._frames += self._quiet_frames
                self._energy += self._quiet_energy
                self._quiet_frames = 0
                self._quiet_energy = 0.0
                self._quiet_since = None
            self._frames += 1
            self._energy += area
            self._moving_at = at
            if activity > self._peak:
                self._peak = activity
                self._peak_at = at
            return closed

        # At or below `quiet`. Provisional: the movement is only over once the
        # stream has stayed this quiet for `bridge` seconds.
        self._quiet_frames += 1
        self._quiet_energy += area
        if self._quiet_since is None:
            self._quiet_since = at
        if at - self._quiet_since >= self.bridge:
            finished = self._close()
            if finished is not None:
                closed = finished
        return closed

    def _open(self, at: float, activity: float, area: float) -> None:
        """Start a movement, backdated to when activity left the baseline.

        Entry is on `active` so noise cannot start a segment, but the movement
        began when it first rose over `quiet`, several frames earlier for
        anything with a real run-up. Reporting the crossing of `active` as the
        start would put every boundary late by however sharply the person
        moved, so the pre-roll is walked back while it stays over `quiet`.
        """
        self._clear_open()
        self._start = at
        self._frames = 1
        self._energy = area
        self._peak = activity
        self._peak_at = at
        self._moving_at = at

        run: list[tuple[float, float, float]] = []
        for sample in reversed(self._preroll):
            if sample[1] <= self.quiet:
                break
            run.append(sample)
        for sample_at, _, sample_area in run:
            self._start = sample_at
            self._frames += 1
            self._energy += sample_area
        self._preroll.clear()

    def _close(self) -> Segment | None:
        """End the open movement. Returns it, unless it was too short.

        `end` is the last frame that was actually moving, not the frame that
        noticed the movement had stopped. The bridge means noticing is
        `bridge` late by construction, and a boundary reported `bridge` late
        would be wrong rather than cautious.
        """
        if self._start is None:
            return None
        end = self._moving_at if self._moving_at is not None else self._start
        segment = Segment(start=self._start, end=end, peak=self._peak_at,
                          frames=self._frames, energy=self._energy)
        self._clear_open()
        if segment.end - segment.start < self.min_seconds:
            self.rejected += 1
            return None
        self.recent.append(segment)
        return segment


def activity_of(features_now, features_before, dt: float) -> float:
    """How much is happening, as one number between 0 and 1.

    The two arguments are movement feature vectors of the same shape, `dt`
    apart. Any descriptor works: the named quantities out of `posture.extract`,
    a wrist position and pinch out of a `motion.Sample`, a `bodymotion.features`
    vector. What comes back is the rate at which that description is changing,
    which is what a movement is. `bodymotion.py` puts it plainly: a punch is
    not a shape, it is a shape changing.

    Give it a span of about `ACTIVITY_SPAN_SECONDS` rather than two adjacent
    frames. Adjacent frames divide the difference by 33 ms and so multiply
    per-frame landmark noise by 30. Measured on `bodyaction`, adjacent frames
    put an idle body at 3.6 against a jab at 4.0, which cannot be thresholded;
    across 100 ms the same idle reads 0.7 against 4.6. Noise does not grow with
    the span, the movement does.

    The rate is squashed into 0 to 1 by r squared over r squared plus a
    reference squared. Three things that buys, each fixing a real problem:
    a tracker jump that sends the rate to a thousand saturates near 1 instead
    of dominating `energy` for the whole segment; the same two thresholds work
    for the body lane and the hand lane, whose raw rates differ by roughly
    twice; and squaring gives the pair of thresholds room to sit apart, where a
    linear map leaves a quiet baseline a third of the way up the scale with
    nowhere to put `quiet`.

    Scale is not normalised away, because a descriptor in different units needs
    thresholds in those units. That is a calibration, not a bug, and the
    alternative is dividing by a running estimate of the baseline, which makes
    "how much is happening" depend on what happened a minute ago.
    """
    now = np.asarray(features_now, dtype=np.float64).ravel()
    before = np.asarray(features_before, dtype=np.float64).ravel()
    if now.shape != before.shape:
        raise ValueError(f"feature shapes differ: {now.shape} and {before.shape}")
    if now.size == 0:
        return 0.0

    step = now - before
    if not np.isfinite(step).all():
        # One unseen joint extrapolated to a NaN must not make the whole frame
        # read as a movement, and must not make it read as stillness either.
        # Dropping the broken dimensions keeps the rest of the body usable.
        step = step[np.isfinite(step)]
        if step.size == 0:
            return 0.0

    # Root mean square across dimensions rather than a plain norm, so a 12
    # dimensional descriptor and a 50 dimensional one produce comparable
    # numbers and the thresholds do not move when the descriptor grows.
    rate = float(np.linalg.norm(step) / math.sqrt(step.size) / max(dt, MIN_DT))
    return rate * rate / (rate * rate + HALF_ACTIVITY_RATE * HALF_ACTIVITY_RATE)
