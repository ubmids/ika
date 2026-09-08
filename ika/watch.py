"""One pipeline: video in, identified and stabilised bodies out.

The four pieces below were each built and measured on their own. This is where
they compose, and the order they compose in is not arbitrary.

    frame -> pose -> subject mask -> camera motion -> compensate -> track

**Pose runs before stabilisation, not after.** That looks backwards until you
see why the mask exists. Camera motion is estimated by finding the largest
group of points that moved together, and a fighter filling the frame *is* that
largest group, so the estimate becomes the fighter's motion reported as the
camera's. Measured on a synthetic pan: with a subject over 40% of the frame the
estimate came back at -0.0625 against a truth of +0.0156, confidently wrong.
Masking the subject out recovers +0.0156. The mask has to come from somewhere,
and pose is the cheapest source of it, so pose goes first.

**Compensation uses accumulated motion, not the last step.** Movement features
are built over a window of about 0.6 seconds, so what corrupts them is the
camera's drift across that whole window. Undoing only the most recent frame
pair would leave the drift in.

**Compensation is skipped when the estimate is not trusted.** A textureless or
motion-blurred frame produces a confident-looking number that is wrong, and
applying it would inject camera motion rather than remove it. Doing nothing is
the better failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

from . import clip, posture
from .body import N_LANDMARKS, PoseTracker
from .stabilise import Motion, Stabiliser
from .tracking import Track, Tracker

# How far past the landmarks the subject mask reaches, as a fraction of the
# body's own bounding box. Landmarks sit inside the silhouette, so a tight box
# leaves the fighter's outline outside the mask, and an outline is exactly what
# a corner detector likes best.
MASK_PADDING = 0.35

# Below this the camera estimate is ignored rather than applied.
TRUST_FLOOR = 0.25


@dataclass(frozen=True)
class Sighting:
    """One identified body in one frame, with its posture already computed."""

    id: int
    track: Track
    features: np.ndarray          # posture.extract output
    missing: int                  # 0 means seen this frame, >0 means held

    @property
    def held(self) -> bool:
        """True when this is a remembered body rather than an observed one."""
        return self.missing > 0


@dataclass(frozen=True)
class Observation:
    """Everything one frame yielded."""

    at: float
    index: int
    sightings: list[Sighting]
    camera: Motion
    compensated: bool
    timeline_source: str = "unknown"

    @property
    def seen(self) -> list[Sighting]:
        return [s for s in self.sightings if not s.held]


def subject_mask(bodies, shape: tuple[int, int], padding: float = MASK_PADDING) -> np.ndarray | None:
    """Non-zero over every detected body, which is what `stabilise` expects.

    Note the polarity: non-zero means "this pixel is the subject", the inverse
    of OpenCV's own mask convention. Getting it backwards masks out the
    background instead, which leaves the subject as the only thing voting and
    produces exactly the failure the mask exists to prevent. It cost me a
    wrong measurement before I read the docstring.
    """
    height, width = shape
    if not bodies:
        return None

    mask = np.zeros((height, width), dtype=np.uint8)
    marked = False
    for body in bodies:
        points = np.asarray(body.image, dtype=np.float64)[:, :2]
        visible = np.asarray(body.visibility) >= 0.3
        if visible.sum() < 4:
            continue
        points = points[visible]
        x0, y0 = points.min(axis=0)
        x1, y1 = points.max(axis=0)
        pad_x = (x1 - x0) * padding
        pad_y = (y1 - y0) * padding
        left = int(max(0, (x0 - pad_x) * width))
        right = int(min(width, (x1 + pad_x) * width))
        top = int(max(0, (y0 - pad_y) * height))
        bottom = int(min(height, (y1 + pad_y) * height))
        if right > left and bottom > top:
            mask[top:bottom, left:right] = 255
            marked = True
    return mask if marked else None


def _compensated(body, motion: Motion, stabiliser: Stabiliser):
    """A copy of `body` with camera motion removed from its image landmarks.

    Only the image coordinates move. World landmarks are already hip-centred
    and carry no frame position, so there is nothing in them for a camera pan
    to corrupt.
    """
    from .body import Body

    moved = stabiliser.compensate(np.asarray(body.image, dtype=np.float64), motion)
    if moved.shape[1] == 2:
        rebuilt = np.asarray(body.image, dtype=np.float32).copy()
        rebuilt[:, :2] = moved
    else:
        rebuilt = moved.astype(np.float32)
    return Body(image=rebuilt, world=body.world, visibility=body.visibility)


@dataclass
class Watcher:
    """Holds the per-stream state: pose, camera, identities.

    One object rather than four, because their state has to advance together.
    A tracker fed frames out of order, or a stabiliser that missed one, both
    fail quietly rather than loudly.
    """

    variant: str = "lite"
    max_bodies: int = 2
    detection_confidence: float = 0.4
    stabilise: bool = True
    max_missing: int = 15

    _pose: PoseTracker = field(init=False, repr=False)
    _tracker: Tracker = field(init=False, repr=False)
    _stabiliser: Stabiliser = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._pose = PoseTracker(
            variant=self.variant,
            max_bodies=self.max_bodies,
            detection_confidence=self.detection_confidence,
            tracking_confidence=self.detection_confidence,
        )
        self._tracker = Tracker(max_missing=self.max_missing)
        self._stabiliser = Stabiliser()

    def observe(self, rgb: np.ndarray, at: float, index: int = 0) -> Observation:
        """Run one frame through the whole pipeline."""
        bodies = self._pose(rgb, int(at * 1000))

        motion = Motion(0.0, 0.0, 1.0, 0.0, 0)
        compensated = False
        if self.stabilise:
            mask = subject_mask(bodies, rgb.shape[:2])
            motion = self._stabiliser.update(rgb, subject_mask=mask)
            drift = self._stabiliser.cumulative
            if drift.confidence >= TRUST_FLOOR and drift.magnitude > 0.0:
                bodies = [_compensated(b, drift, self._stabiliser) for b in bodies]
                compensated = True

        tracks = self._tracker.update(bodies, at)
        sightings = [
            Sighting(
                id=track.id,
                track=track,
                features=posture.extract(track.body.world, track.body.visibility),
                missing=track.missing,
            )
            for track in tracks
        ]
        return Observation(at=at, index=index, sightings=sightings,
                           camera=motion, compensated=compensated)

    def reset(self) -> None:
        self._tracker = Tracker(max_missing=self.max_missing)
        self._stabiliser.reset()

    def close(self) -> None:
        self._pose.close()

    def __enter__(self) -> Watcher:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def watch(
    path: str | Path,
    max_width: int | None = 640,
    stride: int = 1,
    max_frames: int | None = None,
    variant: str = "lite",
    max_bodies: int = 2,
    stabilise: bool = True,
) -> Iterator[Observation]:
    """Every frame of a clip, as identified and stabilised bodies.

    A generator, so a long clip is never held in memory. `clip.read` supplies
    real timestamps rather than index over nominal fps, which matters here
    because every movement feature downstream is a rate and a drifting clock
    scales all of them.
    """
    with Watcher(variant=variant, max_bodies=max_bodies, stabilise=stabilise) as watcher:
        reader = clip.Reader(path, max_width=max_width, stride=stride, max_frames=max_frames)
        with reader:
            for frame in reader:
                observation = watcher.observe(frame.rgb, frame.at, frame.index)
                yield Observation(
                    at=observation.at, index=observation.index,
                    sightings=observation.sightings, camera=observation.camera,
                    compensated=observation.compensated,
                    timeline_source=frame.source,
                )
