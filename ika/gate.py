"""The gate that makes swipes usable, or rather makes them safe.

The dynamic lane was measured at 3.3 false firings a minute against an idle
hand, while catching 38% of real swipes. That is worse than having no swipes,
and it is not a training problem. A swipe is "the hand moved fast in a
straight line", which is also what reaching for a cup looks like, so
displacement cannot separate intent from traffic no matter how good the model
gets.

So this is interaction design rather than machine learning. A swipe only
counts if the hand held a deliberate pose while it travelled. That is the same
mechanism the engage gesture uses one level up: pick something a hand does not
pass through by accident, and the accidents stop.

The cost is honest and worth stating: you now have to mean it. A lazy swipe
with a relaxed hand will be ignored, and that is the trade being made.
"""

from __future__ import annotations

from collections import deque

from .control import SWIPE_GATE, SWIPE_GATE_SHARE


class PoseGate:
    """Tracks what pose a hand held recently, over the movement window.

    A rolling record rather than an instantaneous check, because the pose
    classifier is briefly uncertain at the start and end of any movement. A
    single frame's disagreement should not veto a swipe that was otherwise
    performed deliberately, and requiring every frame to agree rejected
    genuine swipes in favour of nothing at all.
    """

    def __init__(
        self,
        poses: tuple[str, ...] = SWIPE_GATE,
        share: float = SWIPE_GATE_SHARE,
        seconds: float = 0.6,
    ):
        self.poses = tuple(poses)
        self.share = share
        self.seconds = seconds
        self._seen: deque[tuple[float, str]] = deque()

    def push(self, pose: str, at: float) -> None:
        self._seen.append((at, pose))
        while self._seen and at - self._seen[0][0] > self.seconds:
            self._seen.popleft()

    def clear(self) -> None:
        self._seen.clear()

    @property
    def held(self) -> float:
        """Fraction of the window spent in a gate pose."""
        if not self._seen:
            return 0.0
        return sum(1 for _, pose in self._seen if pose in self.poses) / len(self._seen)

    def open(self) -> bool:
        """Whether a swipe performed now should be allowed through."""
        return self.held >= self.share

    def __len__(self) -> int:
        return len(self._seen)
