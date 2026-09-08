"""Committing live, while the movement is still happening.

The offline lane already settled the argument. `tell/lead.py` measured the
constraint: the median gap between one action ending and the next starting is
263 ms, and confirming a gesture the way `machine.py` does costs 250 ms, so a
call that waits for the movement to finish arrives with 13 ms to spare. End to
end that is the difference between 37% useful warnings when committing early at
0.85 confidence and 7% when waiting for the movement to complete.

`tell/early.py` proved a model can commit on a prefix. This is the live half of
it: the decision layer that actually runs on a stream and does not wait.

What changed from `machine.py`, and why:

**Evidence, not duration.** The old layer needed a label held for
`dwell_frames` consecutive frames on top of an exponential smoother, and both
of those are pure delay. Here a call fires the moment `min_frames` of evidence
put a class over `threshold`. Four frames at 30 fps is 133 ms, which fits
inside the 263 ms gap with room for the rest of the pipeline.

**A window, not a smoother.** An exponential smoother lags by construction: it
takes several frames for a new answer to overcome the old one. Instead the
recent frames are averaged flat, exactly as `early.py` builds features over a
prefix of the movement. A confident stream reads 1.0 on its first frame, so
averaging costs nothing when the model is sure, while a single wrong frame
still cannot carry a decision on its own.

**Onset, so latency means something.** A movement begins when the distribution
leaves the neutral classes. Without that moment recorded there is no honest
latency to report, only a timestamp.

**One call per movement.** A punch held in front of the camera is one punch.
Repeating a call at frame rate is the failure that makes a live layer unusable,
so a call is spent until the subject returns to neutral. A *different* call
inside the same continuous movement is a real event (a punch flowing into a
kick), and that one is gated by `cooldown` instead. Same label needs rest, a
new label needs time.

The trade is real and is not hidden: on a genuinely ambiguous opening the
window holds the ambiguous frames and the call is late, and on an opening that
confidently lies this commits to the lie sooner than a dwell layer would.
`tests/test_live_commit.py` measures both rather than asserting they are small.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

# How many recent frames are averaged into one decision. This is a ceiling,
# not a requirement: a confident stream commits at `min_frames`. The ceiling
# exists so an ambiguous opening cannot poison the average forever. With a
# plain running mean over the whole movement, ten ambiguous frames keep a
# clean class below 0.85 for another twenty frames, and a call that never
# arrives is worse than a late one.
EVIDENCE_WINDOW_FRAMES = 8

# A movement has begun when less than this much probability mass sits on the
# neutral classes. Mass rather than argmax, because an uncertain distribution
# has an argmax that flickers between non-neutral classes frame to frame, and
# onset detected off that flicker would start the clock on noise.
ONSET_NEUTRAL_MASS = 0.5

# And it is over when neutral mass comes back up to here. The gap between the
# two is deliberate: one threshold for both directions makes the state chatter
# on any stream that sits near the line.
REST_NEUTRAL_MASS = 0.6


@dataclass(frozen=True)
class Call:
    """One committed read, with the cost of making it attached.

    `frames` and `latency` are not diagnostics. They are the product claim, so
    every call carries the numbers needed to check it against the 263 ms gap.
    """

    label: str
    confidence: float
    at: float
    frames: int           # frames of evidence since onset that it committed on
    latency: float        # seconds from when the movement began

    @property
    def milliseconds(self) -> float:
        """Latency in the unit `lead.py` states the gap in, to save the sums."""
        return self.latency * 1000.0


class EarlyCommitter:
    """A streaming decision layer that calls a movement before it ends.

    Drop-in alternative to `machine.GestureMachine` for the case where the
    answer is worth less the longer it takes. There is no engagement gate here:
    engagement exists to stop a hand controlling a keyboard by accident, and a
    read of what someone just did commands nothing.
    """

    def __init__(self, classes: list[str], threshold: float = 0.85,
                 min_frames: int = 4, cooldown: float = 0.4,
                 neutral: tuple[str, ...] = ("idle", "none", "rest")):
        self.classes = list(classes)
        self.threshold = threshold
        self.min_frames = min_frames
        self.cooldown = cooldown
        self.neutral = neutral

        # Resolved once, here, rather than by name on every frame: this runs
        # thirty times a second and the class list cannot change under it.
        self._neutral_index = np.array(
            [i for i, name in enumerate(self.classes) if name in neutral], dtype=int
        )
        self._callable_index = np.array(
            [i for i, name in enumerate(self.classes) if name not in neutral], dtype=int
        )

        self.calls: list[Call] = []
        self._window: deque[np.ndarray] = deque(maxlen=EVIDENCE_WINDOW_FRAMES)
        self._onset: float | None = None
        self._frames = 0
        self._confidence = 0.0
        self._spent: str | None = None
        self._moved_on = False
        self._last_call = -1e9
        self._absent_since: float | None = None

    # ------------------------------------------------------------------ state

    def reset(self) -> None:
        self._forget_evidence()
        self.calls = []
        self._rearm()
        self._last_call = -1e9
        self._absent_since = None

    def _rearm(self) -> None:
        """Forget which call was last made, so the same one can be made again."""
        self._spent = None
        self._moved_on = False

    def _forget_evidence(self) -> None:
        """Drop everything about the movement in progress, keep the history.

        Called whenever the movement ends or the subject disappears. The point
        is that no half-built trigger survives: evidence from before a gap is
        evidence about a different movement.
        """
        self._window.clear()
        self._onset = None
        self._frames = 0
        self._confidence = 0.0

    @property
    def watching(self) -> bool:
        """Whether a movement is underway, which is when a call can happen."""
        return self._onset is not None

    @property
    def progress(self) -> float:
        """How close the current movement is to being called, 0 to 1.

        Shown on screen for the same reason `machine.py` shows its dwell meter:
        without it a movement that is one frame short of a call looks exactly
        like one being ignored. This meter reports the *binding* constraint of
        the two, so a user who is confident but early and one who is patient
        but unclear see different things.
        """
        if not self.watching or self._frames == 0:
            return 0.0
        return float(min(1.0,
                         self._frames / self.min_frames,
                         self._confidence / self.threshold))

    @property
    def evidence_frames(self) -> int:
        """Frames of evidence gathered on the movement in progress."""
        return self._frames

    # ------------------------------------------------------------------ update

    def update(self, probabilities: np.ndarray | None, at: float) -> Call | None:
        """Feed one frame's distribution. Returns a call, or None.

        Pass `None` when there is no subject in frame. That is information, not
        an absence, and it clears the evidence: coming back after a dropout
        must not fire off a trigger that was primed before it.
        """
        if probabilities is None:
            self._forget_evidence()
            if self._absent_since is None:
                self._absent_since = at
            elif at - self._absent_since > self.cooldown:
                # Gone longer than the cooldown counts as a break between
                # movements, so the last call stops being spent. One dropped
                # frame does not, or a flickering tracker would re-fire the
                # call the subject is still holding.
                self._rearm()
            return None
        self._absent_since = None

        probabilities = np.asarray(probabilities, dtype=float)
        total = float(probabilities.sum())
        if total > 0:
            # Onset is a fraction of the mass, so an unnormalised vector would
            # quietly move the point at which a movement is judged to start.
            probabilities = probabilities / total

        neutral_mass = (float(probabilities[self._neutral_index].sum())
                        if len(self._neutral_index) else 0.0)

        if not self.watching:
            if neutral_mass >= ONSET_NEUTRAL_MASS:
                self._rearm()           # at rest, so the last call is re-armed
                return None
            self._onset = at            # the movement starts here, and latency
            self._frames = 0            # is measured from here, not from the
            self._window.clear()        # frame that happens to commit
        elif neutral_mass >= REST_NEUTRAL_MASS:
            self._forget_evidence()
            self._rearm()
            return None

        if (self._spent is not None and not self._moved_on
                and len(self._callable_index)):
            frame_best = int(self._callable_index[
                probabilities[self._callable_index].argmax()])
            if self.classes[frame_best] != self._spent:
                # The movement already called is over and a different one has
                # started without a pause. Its tail is still in the window, and
                # averaging that tail into the next call delays it by the whole
                # window: measured 767 ms on a punch flowing straight into a
                # kick, against 100 ms once the window starts fresh. Only the
                # evidence origin moves. A full `min_frames` of new evidence is
                # still required, and this can happen once per call, so an
                # argmax flickering between two movements cannot keep
                # restarting the window and starve the call entirely.
                self._moved_on = True
                self._window.clear()
                self._frames = 0
                self._onset = at

        self._window.append(probabilities)
        self._frames += 1

        mean = np.mean(np.stack(self._window), axis=0)
        if not len(self._callable_index):
            return None                 # every class is neutral, nothing to call

        # The argmax is taken over callable classes only. Neutral is what ends
        # a movement, so it must never be an outcome, and excluding it here is
        # a stronger guarantee than checking for it afterwards. Confidence
        # stays a share of the whole distribution, so a window that is half
        # rest reads 0.5 and cannot cross the threshold.
        best = int(self._callable_index[mean[self._callable_index].argmax()])
        label, confidence = self.classes[best], float(mean[best])
        self._confidence = confidence

        if self._frames < self.min_frames or confidence < self.threshold:
            return None
        if label == self._spent:
            return None                 # already called, waiting for neutral
        if at - self._last_call < self.cooldown:
            return None                 # a new call, but too soon after the last

        call = Call(label=label, confidence=confidence, at=at,
                    frames=self._frames, latency=at - self._onset)
        self.calls.append(call)
        self._spent, self._moved_on, self._last_call = label, False, at

        # Start a fresh window from the commit. Two reasons: the next call in a
        # continuous sequence must be made on new evidence rather than on the
        # frames that already fired, and its latency must be measured from here
        # or it would inherit the first movement's duration and report a delay
        # that never happened.
        self._window.clear()
        self._frames = 0
        self._onset = at
        return call
