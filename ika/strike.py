"""A punch read from the pose, for someone standing back from the camera.

`approach.py` reads a punch as a palm growing on its way to the lens. That is
the right physics at arm's length from a laptop and the wrong physics for
shadowboxing, which nobody does 60 cm from a screen. Stand back two metres,
where the whole upper body is in shot, and a jab that travels half a metre
grows the palm by about a third: under the firing threshold, from a hand only
twenty pixels across that the hand landmarker finds in a third of frames.
Replayed over nine real follow-along workouts it caught **0%** of the punches
two labellers agreed on.

What does survive at that distance is the pose, found in 98 to 100% of frames
on every clip. So a punch here is read from the arm: how far the wrist and
elbow have left this arm's own guard, how fast, and in which direction, with
forward travel in MediaPipe's inferred depth standing in for the straight
thrown at the camera that barely moves in the image.

Two readers share those features.

`ExcursionReader` is the hand-built rule: fire when the wrist has left its
guard by a set fraction of a torso. Measured on real people it catches 74% of
punches and fires about 54 false ones a minute, because a body rotating into
one arm's punch swings the other arm too, and rolls, bobs and guard resets all
move a wrist a long way from its guard.

`StrikeReader` puts a small classifier over the same features and was fitted
on the labelled footage, scored only on people it never saw
(`scripts/train_strike.py`). It is the one the drill loop uses.

"In guard" is not a fixed pose. It is this arm's own recent median, because
people hold their hands differently, and between punches they are mostly in
guard, so a median over a second and a half sits on the guard and not on the
punches. Everything here is causal, so the live loop and a replay agree
exactly.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .body import (LEFT_ELBOW, LEFT_HIP, LEFT_SHOULDER, LEFT_WRIST, NOSE, RIGHT_ELBOW,
                   RIGHT_HIP, RIGHT_SHOULDER, RIGHT_WRIST)

ARMS = {
    "left": (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
    "right": (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
}
OTHER = {"left": "right", "right": "left"}

# Shipped inside the package, unlike the other checkpoints, because it cannot
# be rebuilt from anything in the repository alone: it was fitted on landmarks
# from public videos that are fetched, not committed. The labels that fitted
# it are committed, under bench/shadow, so it can be rebuilt by anyone who
# fetches the footage.
WEIGHTS = Path(__file__).resolve().parent / "weights" / "strike.npz"

# Seconds of this arm's history the guard is the median of. Long enough that a
# combo of four punches, about a second, cannot drag it, short enough to follow
# someone who changes stance.
GUARD_SECONDS = 1.5

# Lags, in seconds, at which the features are sampled, so the classifier sees
# a short movie of the arm rather than one frame. In seconds and not frames
# because the footage runs at 24 and 30 fps and a live camera at whatever it
# manages.
LAGS = (0.0, 0.067, 0.133, 0.2)

# Landmark visibility below which a frame is not read at all.
MIN_VISIBILITY = 0.5

# The excursion rule's threshold, in torso lengths.
EXCURSION = 0.30

# Firing: a reader calls a punch when its score crosses the threshold, and
# cannot call again from that arm until the score falls below REARM of it and
# REFRACTORY seconds have passed. A double jab is about 0.3 s apart.
REARM = 0.6
REFRACTORY = 0.18

PER_FRAME = 18     # features per arm per sampled moment


@dataclass(frozen=True)
class Strike:
    side: str
    at: float
    score: float


def _frame(body, aspect: float):
    """Per-arm raw vectors for one frame, or None if the pose was guessing."""
    needed = (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP,
              LEFT_WRIST, RIGHT_WRIST, LEFT_ELBOW, RIGHT_ELBOW)
    if body is None or min(float(body.visibility[j]) for j in needed) < MIN_VISIBILITY:
        return None
    image = np.asarray(body.image, dtype=np.float64)[:, :2].copy()
    image[:, 1] /= aspect
    world = np.asarray(body.world, dtype=np.float64)
    shoulders = (image[LEFT_SHOULDER] + image[RIGHT_SHOULDER]) / 2
    hips = (image[LEFT_HIP] + image[RIGHT_HIP]) / 2
    torso = float(np.linalg.norm(shoulders - hips))
    wshoulders = (world[LEFT_SHOULDER] + world[RIGHT_SHOULDER]) / 2
    wtorso = float(np.linalg.norm(wshoulders - (world[LEFT_HIP] + world[RIGHT_HIP]) / 2))
    if torso < 1e-6 or wtorso < 1e-6:
        return None
    out = {"centre": np.r_[shoulders / torso, hips / torso],
           "twist": float(np.arctan2(*(image[RIGHT_SHOULDER] - image[LEFT_SHOULDER])[::-1])),
           "width": float(np.linalg.norm(image[LEFT_SHOULDER] - image[RIGHT_SHOULDER]) / torso),
           "nose": (image[NOSE] - shoulders) / torso}
    for side, (s, e, w) in ARMS.items():
        flip = -1.0 if side == "left" else 1.0     # one model serves both arms
        wrist = (image[w] - image[s]) / torso
        elbow = (image[e] - image[s]) / torso
        wrist[0] *= flip
        elbow[0] *= flip
        # MediaPipe's world z grows away from the camera, so forward is minus.
        forward = -(world[w, 2] - world[s, 2]) / wtorso
        eforward = -(world[e, 2] - world[s, 2]) / wtorso
        u, v = world[s] - world[e], world[w] - world[e]
        bend = float(u @ v / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-9))
        out[side] = np.r_[wrist, forward, elbow, eforward, bend]
    return out


class StrikeFeatures:
    """Streaming features per arm. Feed frames in order; ask for a vector."""

    def __init__(self, aspect: float = 1.0, guard_seconds: float = GUARD_SECONDS):
        self.aspect = float(aspect)
        self.guard_seconds = float(guard_seconds)
        self._raw: deque = deque()          # (at, frame dict), guard window
        self._rows: dict[str, deque] = {side: deque() for side in ARMS}

    def reset(self) -> None:
        self._raw.clear()
        for rows in self._rows.values():
            rows.clear()

    def update(self, body, at: float) -> dict[str, np.ndarray] | None:
        """This frame's feature vector for each arm, or None if unreadable."""
        frame = _frame(body, self.aspect)
        if frame is None:
            return None
        while self._raw and at - self._raw[0][0] > self.guard_seconds:
            self._raw.popleft()
        if len(self._raw) < 5:
            self._raw.append((at, frame))
            return None
        guards = {side: np.median(np.stack([f[side] for _, f in self._raw]), axis=0)
                  for side in ARMS}
        rest_centre = np.median(np.stack([f["centre"] for _, f in self._raw]), axis=0)
        rest_twist = float(np.median([f["twist"] for _, f in self._raw]))
        self._raw.append((at, frame))

        body_part = np.r_[frame["centre"] - rest_centre, frame["twist"] - rest_twist,
                          frame["width"]]
        out = {}
        for side in ARMS:
            d = frame[side] - guards[side]
            o = frame[OTHER[side]] - guards[OTHER[side]]
            twist = body_part[4] * (-1.0 if side == "left" else 1.0)
            row = np.r_[d[:7], np.hypot(d[0], d[1]), np.hypot(o[0], o[1]), o[2],
                        body_part[:4], twist, body_part[5], frame["nose"][1],
                        frame[side][6] - guards[side][6]]
            rows = self._rows[side]
            rows.append((at, row))
            while rows and at - rows[0][0] > LAGS[-1] + 0.1:
                rows.popleft()
            out[side] = np.concatenate([self._lagged(rows, at - lag) for lag in LAGS])
        return out

    @staticmethod
    def _lagged(rows, when: float) -> np.ndarray:
        best = min(rows, key=lambda r: abs(r[0] - when))
        return best[1]


DIM = PER_FRAME * len(LAGS)


class _Decoder:
    """Score in, punch calls out, with rearm and refractory per arm."""

    def __init__(self, threshold: float):
        self.threshold = float(threshold)
        self._armed = {side: True for side in ARMS}
        self._last = {side: -1e9 for side in ARMS}

    def step(self, side: str, value: float, at: float) -> Strike | None:
        if self._armed[side] and value > self.threshold and at - self._last[side] >= REFRACTORY:
            self._armed[side] = False
            self._last[side] = at
            return Strike(side, at, value)
        if not self._armed[side] and value < self.threshold * REARM:
            self._armed[side] = True
        return None


class ExcursionReader:
    """The hand-built rule: the wrist has left its own guard far enough."""

    def __init__(self, threshold: float = EXCURSION, aspect: float = 1.0):
        self.features = StrikeFeatures(aspect=aspect)
        self.decoder = _Decoder(threshold)

    def observe(self, body, at: float) -> list[Strike]:
        rows = self.features.update(body, at)
        if rows is None:
            return []
        out = []
        for side, row in rows.items():
            # wrist planar excursion plus forward travel, from the current frame
            value = float(row[7] + max(row[2], 0.0))
            hit = self.decoder.step(side, value, at)
            if hit:
                out.append(hit)
        return out


@dataclass
class StrikeModel:
    """A two-layer network in numpy, so the live loop needs no torch."""

    mean: np.ndarray
    scale: np.ndarray
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    threshold: float

    def __call__(self, x: np.ndarray) -> np.ndarray:
        z = (np.atleast_2d(x) - self.mean) / self.scale
        h = np.maximum(z @ self.w1 + self.b1, 0.0)
        return 1.0 / (1.0 + np.exp(-(h @ self.w2 + self.b2).ravel()))

    def save(self, path: str | Path = WEIGHTS) -> None:
        np.savez(path, mean=self.mean, scale=self.scale, w1=self.w1, b1=self.b1,
                 w2=self.w2, b2=self.b2, threshold=np.float64(self.threshold))

    @classmethod
    def load(cls, path: str | Path = WEIGHTS) -> StrikeModel:
        m = np.load(path)
        return cls(m["mean"], m["scale"], m["w1"], m["b1"], m["w2"], m["b2"],
                   float(m["threshold"]))


class StrikeReader:
    """Punch calls from a body, using the classifier fitted on real people."""

    def __init__(self, model: StrikeModel | None = None, aspect: float = 1.0,
                 threshold: float | None = None):
        self.model = model or StrikeModel.load()
        self.features = StrikeFeatures(aspect=aspect)
        self.decoder = _Decoder(self.model.threshold if threshold is None else threshold)

    def reset(self) -> None:
        self.features.reset()
        self.decoder = _Decoder(self.decoder.threshold)

    def observe(self, body, at: float) -> list[Strike]:
        rows = self.features.update(body, at)
        if rows is None:
            return []
        sides = list(rows)
        scores = self.model(np.stack([rows[s] for s in sides]))
        out = []
        for side, value in zip(sides, scores):
            hit = self.decoder.step(side, float(value), at)
            if hit:
                out.append(hit)
        return out
