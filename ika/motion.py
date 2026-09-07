"""Gestures that only exist in time, and the awkward fact that `features`
throws away exactly what they are made of.

The static pipeline works by discarding position: the wrist becomes the origin,
so a fist is a fist wherever it is in frame. That invariance is the reason the
pose classifier generalises at all.

A swipe is the opposite. A swipe *is* translation. Normalise it away and every
swipe becomes an open palm sitting perfectly still. So the two lanes need
different inputs, and this module builds the one the static lane deliberately
destroys: where the hand is going, how fast, and how straight.

Motion alone is not enough either. A swipe and a drag can trace the same path;
what separates them is the shape of the hand while it travels. So the window
carries both, and the features below mix trajectory with pose stability.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .schema import INDEX_TIP, MIDDLE_MCP, THUMB_TIP, WRIST

# Trajectory: net displacement 2, path length 1, straightness 1, peak speed 1,
# mean speed 1, direction 2. Pose: pinch start/end/min/range 4, pose drift 1,
# spread of pose over window 1.
#
# Window duration and frame count were in here and have been removed. They are
# properties of *the window*, not of the gesture, and because every training
# window was the same length the model learned to lean on them. Feeding a
# shorter window at inference then read as out-of-distribution and came back
# "none" with total confidence, which is how a swipe could be classified
# perfectly in training and never once detected live.
MOTION_DIM = 14

# Named offsets into that vector. Worth the ceremony: removing one feature
# silently shifted every index after it, and the tests that indexed by number
# kept passing while asserting things about the wrong columns.
NET_X, NET_Y = 0, 1
PATH = 2
STRAIGHTNESS = 3
PEAK_SPEED, MEAN_SPEED = 4, 5
DIR_X, DIR_Y = 6, 7
PINCH_START, PINCH_END, PINCH_MIN, PINCH_RANGE = 8, 9, 10, 11
POSE_DRIFT = 12
POSE_WOBBLE = 13


@dataclass
class Sample:
    """One frame, as the window needs it."""

    at: float               # seconds
    position: np.ndarray    # (2,) wrist in frame-normalised coordinates
    span: float             # apparent hand size in the frame, for scaling
    pinch: float            # thumb-to-index distance in palm lengths
    pose: np.ndarray        # the static feature vector, for stability checks


class MotionWindow:
    """A rolling window of recent frames.

    Length is in seconds rather than frames because gestures happen at human
    speed while frame rate varies with load, and a window that shrinks when the
    machine is busy would change what a swipe means.
    """

    def __init__(self, seconds: float = 0.6):
        self.seconds = seconds
        self._samples: deque[Sample] = deque()

    def push(self, sample: Sample) -> None:
        self._samples.append(sample)
        while self._samples and sample.at - self._samples[0].at > self.seconds:
            self._samples.popleft()

    def clear(self) -> None:
        self._samples.clear()

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def ready(self) -> bool:
        """Enough history to say anything, without waiting out the gesture.

        The bar used to be 60% of the window length, which at a 0.7s window
        meant 13 frames of history. A swipe is about 14 frames long, so the
        window would only agree to look at it for the final frame or two, and
        the dwell timer downstream never got the consecutive readings it needed.
        A fixed, low bar means the window starts answering while the movement
        is still underway, which is when an answer is useful.
        """
        return (
            len(self._samples) >= 8
            and self._samples[-1].at - self._samples[0].at >= 0.25
        )

    def features(self) -> np.ndarray | None:
        return features(list(self._samples)) if self.ready else None


def features(samples: list[Sample]) -> np.ndarray:
    """Motion features for a window. Shape (MOTION_DIM,).

    Distances are divided by the apparent hand size rather than left in frame
    units, so a swipe means the same thing whether you are close to the camera
    or across the room. This is the one invariance the motion lane does want:
    scale, but emphatically not translation.
    """
    times = np.array([s.at for s in samples])
    points = np.stack([s.position for s in samples])
    span = float(np.median([s.span for s in samples]))
    span = max(span, 1e-4)

    scaled = points / span

    net = scaled[-1] - scaled[0]
    steps = np.diff(scaled, axis=0)
    step_lengths = np.linalg.norm(steps, axis=1)
    path = float(step_lengths.sum())

    # Straightness separates a swipe from a fidget that ends up somewhere else.
    # A deliberate stroke goes almost directly; a wandering hand covers far
    # more ground than it displaces.
    straightness = float(np.linalg.norm(net) / path) if path > 1e-6 else 0.0

    gaps = np.maximum(np.diff(times), 1e-4)
    speeds = step_lengths / gaps
    direction = net / max(np.linalg.norm(net), 1e-6)

    pinches = np.array([s.pinch for s in samples])
    poses = np.stack([s.pose for s in samples])
    # How much the hand *shape* changed while travelling. Near zero means a
    # held pose being carried, which is what a swipe and a drag both are; a
    # snap is the opposite, barely moving while the shape changes sharply.
    drift = float(np.linalg.norm(poses[-1] - poses[0]))
    wobble = float(poses.std(axis=0).mean())

    return np.array(
        [
            net[0], net[1],
            path,
            straightness,
            float(speeds.max()),
            float(speeds.mean()),
            direction[0], direction[1],
            float(pinches[0]), float(pinches[-1]),
            float(pinches.min()), float(pinches.max() - pinches.min()),
            drift,
            wobble,
        ],
        dtype=np.float32,
    )


def sample_from_hand(hand, pose_vector: np.ndarray, at: float) -> Sample:
    """Build a window sample from a detected hand.

    Position comes from the *image* landmarks, because that is the only place
    translation survives. Pose comes from world landmarks, because that is
    where shape is cleanest. Using the wrong one for either is the easiest
    mistake to make here and produces a model that almost works.
    """
    image = hand.image
    span = float(np.linalg.norm(image[MIDDLE_MCP, :2] - image[WRIST, :2]))
    world = hand.world
    pinch = float(
        np.linalg.norm(world[THUMB_TIP] - world[INDEX_TIP])
        / max(np.linalg.norm(world[MIDDLE_MCP] - world[WRIST]), 1e-8)
    )
    return Sample(
        at=at,
        position=image[WRIST, :2].astype(np.float64).copy(),
        span=span,
        pinch=pinch,
        pose=pose_vector,
    )
