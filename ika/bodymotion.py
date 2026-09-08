"""What a body did over a window, rather than what it looked like in a frame.

`posture` answers "what shape is this body in". A punch is not a shape, it is
a shape *changing*: an arm going from folded to straight. So the features here
are almost entirely rates and differences over a window of frames.

The named quantities from `posture` do most of the work, because their
derivatives are the domain's verbs. The rate of change of elbow extension is a
punch being thrown. The rate of change of guard height is a guard dropping. A
model handed those does not have to rediscover them from 99 coordinates.

Two lessons from the hand lane are baked in. Nothing here encodes how long the
window was or how many frames it held, because those are properties of the
windowing rather than the movement, and last time they leaked in the model
learned to lean on them and then failed on any other window length. And every
distance is divided by torso length, so a movement means the same thing near
the camera and far from it.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from . import posture
from .body import LEFT_WRIST, RIGHT_WRIST
from .posture import DERIVED

# For each of the 14 named quantities: value at the end, net change, peak rate.
# Plus per wrist: net displacement (2), peak speed (1). Plus posture drift and
# wobble.
N_NAMED = len(DERIVED)
FEATURE_DIM = N_NAMED * 3 + 2 * 3 + 2

NAMED_ORDER = tuple(DERIVED)


@dataclass
class Frame:
    """One observation, as the window needs it."""

    at: float                 # seconds
    features: np.ndarray      # posture.extract output for this frame
    wrists: np.ndarray        # (2, 2) left then right, frame-normalised x,y
    scale: float              # apparent torso length in the frame
    visibility: np.ndarray    # (33,) so a read can be discounted


def _named(frame: Frame) -> np.ndarray:
    """The 14 named quantities out of a full posture vector."""
    return np.array([frame.features[DERIVED[k]] for k in NAMED_ORDER], dtype=np.float64)


def features(window: list[Frame]) -> np.ndarray:
    """Movement features for a window. Shape (FEATURE_DIM,)."""
    if len(window) < 2:
        raise ValueError("need at least two frames to describe a movement")

    times = np.array([f.at for f in window])
    gaps = np.maximum(np.diff(times), 1e-4)

    named = np.stack([_named(f) for f in window])          # (t, 14)
    rates = np.diff(named, axis=0) / gaps[:, None]

    # Where it ended, how far it travelled, and how fast it moved at its
    # quickest. The peak rate is what separates a punch from a slow reach.
    ends = named[-1]
    deltas = named[-1] - named[0]
    peaks = np.max(np.abs(rates), axis=0)

    scale = max(float(np.median([f.scale for f in window])), 1e-4)
    wrists = np.stack([f.wrists for f in window]) / scale   # (t, 2, 2)

    wrist_block = []
    for side in range(2):
        path = wrists[:, side, :]
        net = path[-1] - path[0]
        steps = np.linalg.norm(np.diff(path, axis=0), axis=1) / gaps
        wrist_block.extend([net[0], net[1], float(steps.max())])

    canonical = np.stack([f.features[:99] for f in window])
    drift = float(np.linalg.norm(canonical[-1] - canonical[0]))
    wobble = float(canonical.std(axis=0).mean())

    return np.concatenate([
        ends, deltas, peaks, np.array(wrist_block), [drift, wobble]
    ]).astype(np.float32)


def index_of(quantity: str, kind: str = "delta") -> int:
    """Where a specific named read sits in the vector.

    Named lookups rather than raw offsets, because removing one feature from
    the hand lane's motion vector silently shifted every index after it while
    the tests kept passing on the wrong columns.
    """
    if quantity not in NAMED_ORDER:
        raise KeyError(f"unknown quantity {quantity!r}")
    base = {"end": 0, "delta": N_NAMED, "peak": 2 * N_NAMED}[kind]
    return base + NAMED_ORDER.index(quantity)


class BodyWindow:
    """A rolling window of body frames, bounded by time rather than count.

    Time rather than frames because movements happen at human speed while the
    frame rate moves with load, and a window that shrank when the machine got
    busy would change what a punch means.
    """

    def __init__(self, seconds: float = 0.6):
        self.seconds = seconds
        self._frames: deque[Frame] = deque()

    def push(self, frame: Frame) -> None:
        self._frames.append(frame)
        while self._frames and frame.at - self._frames[0].at > self.seconds:
            self._frames.popleft()

    def clear(self) -> None:
        self._frames.clear()

    def __len__(self) -> int:
        return len(self._frames)

    @property
    def ready(self) -> bool:
        """A low bar on purpose: the point is to answer while it is happening.

        The hand lane set this at 60% of the window length, which meant 13
        frames of history for a 14-frame gesture, so the window refused to
        look at a swipe until it was over.
        """
        return len(self._frames) >= 4 and self._frames[-1].at - self._frames[0].at >= 0.1

    def features(self) -> np.ndarray | None:
        return features(list(self._frames)) if self.ready else None

    def frames(self) -> list[Frame]:
        return list(self._frames)


def frame_from_body(body, at: float) -> Frame:
    """Build a window frame from a detected body.

    Wrist positions come from *image* landmarks, because that is where
    translation survives; posture comes from world landmarks, because that is
    where shape is cleanest. Using the wrong one for either is the easiest
    mistake here and produces something that almost works.
    """
    image = np.asarray(body.image, dtype=np.float64)
    world = np.asarray(body.world, dtype=np.float64)
    shoulders = (image[posture.LEFT_SHOULDER] + image[posture.RIGHT_SHOULDER]) / 2.0
    hips = (image[posture.LEFT_HIP] + image[posture.RIGHT_HIP]) / 2.0
    return Frame(
        at=at,
        features=posture.extract(world, body.visibility),
        wrists=np.stack([image[LEFT_WRIST, :2], image[RIGHT_WRIST, :2]]),
        scale=float(np.linalg.norm(shoulders[:2] - hips[:2])),
        visibility=np.asarray(body.visibility, dtype=np.float32),
    )
