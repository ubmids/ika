"""What passes between two people, which is where a strike actually lives.

Reading one body gave 82% on gross posture and 43% on strikes. The suspicion
is that a punch is not a shape at all, it is a *relationship*: one person's
wrist travelling toward another person's body. Describe one figure in isolation
and the arm extension survives, while the thing that made it a punch rather
than a stretch does not.

So these features are all about the pair. Gap and its rate, which is closing.
Each wrist's distance to the other person's torso, which is what a strike
reduces. Whether the two overlap, which is a clinch.

This matters beyond two-person footage, and that is worth saying because the
glasses only ever see one body. In first person the second party is the camera,
so "wrist approaching the other person's torso" becomes "wrist approaching the
lens", and the same relational quantity survives in a different form. If the
missing signal here turns out to be relational, that is the shape the
first-person version has to recover too.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# gap start/end/delta/min 4, closing peak/mean 2, four wrist-to-other
# distances at end and their minima and peak rates 12, overlap share 1,
# facing 1.
FEATURE_DIM = 20

NAMES = (
    "gap_start", "gap_end", "gap_delta", "gap_min",
    "closing_peak", "closing_mean",
    "a_left_reach_end", "a_left_reach_min", "a_left_reach_rate",
    "a_right_reach_end", "a_right_reach_min", "a_right_reach_rate",
    "b_left_reach_end", "b_left_reach_min", "b_left_reach_rate",
    "b_right_reach_end", "b_right_reach_min", "b_right_reach_rate",
    "overlap_share", "size_ratio",
)
INDEX = {name: i for i, name in enumerate(NAMES)}


@dataclass(frozen=True)
class Pair:
    """Two bodies in one frame, ordered left to right on screen.

    Ordered by screen position rather than by track id because a strike is
    directional: "the left person punched" is a different event from "the right
    person punched", and track ids are assigned by arrival order which carries
    no such meaning.
    """

    at: float
    a_centre: np.ndarray      # (2,) frame-normalised
    b_centre: np.ndarray
    a_wrists: np.ndarray      # (2, 2) left then right
    b_wrists: np.ndarray
    a_scale: float            # torso length in the frame
    b_scale: float


def _reach(wrist: np.ndarray, other_centre: np.ndarray, scale: float) -> float:
    """How far a wrist is from the other person's torso, in torso lengths.

    Scaled, so a punch means the same thing whether the pair are near the
    camera or far from it.
    """
    return float(np.linalg.norm(wrist - other_centre) / max(scale, 1e-4))


def features(window: list[Pair]) -> np.ndarray:
    """Relational features for a window of paired frames. Shape (20,)."""
    if len(window) < 2:
        raise ValueError("need at least two frames to describe an interaction")

    times = np.array([p.at for p in window])
    gaps_seconds = np.maximum(np.diff(times), 1e-4)
    scale = float(np.median([(p.a_scale + p.b_scale) / 2.0 for p in window]))
    scale = max(scale, 1e-4)

    gap = np.array([
        np.linalg.norm(p.a_centre - p.b_centre) / scale for p in window
    ])
    closing = -np.diff(gap) / gaps_seconds     # positive means coming together

    reaches = {}
    for side, key in ((0, "left"), (1, "right")):
        reaches[f"a_{key}"] = np.array([
            _reach(p.a_wrists[side], p.b_centre, scale) for p in window
        ])
        reaches[f"b_{key}"] = np.array([
            _reach(p.b_wrists[side], p.a_centre, scale) for p in window
        ])

    # A clinch: the two are close enough that their torsos effectively overlap.
    overlap = float(np.mean(gap < 1.0))
    size_ratio = float(np.median([
        min(p.a_scale, p.b_scale) / max(max(p.a_scale, p.b_scale), 1e-4)
        for p in window
    ]))

    out = [gap[0], gap[-1], gap[-1] - gap[0], gap.min(),
           float(closing.max()), float(closing.mean())]
    for key in ("a_left", "a_right", "b_left", "b_right"):
        series = reaches[key]
        rate = -np.diff(series) / gaps_seconds        # positive means closing in
        out += [float(series[-1]), float(series.min()), float(rate.max())]
    out += [overlap, size_ratio]
    return np.array(out, dtype=np.float32)


def pair_from_frame(at: float, bodies: list) -> Pair | None:
    """Build a Pair from two detected bodies, or None if there are not two.

    A frame with one body is not a degraded interaction, it is an absence of
    one, and filling the gap with a guess would invent a relationship.
    """
    from .body import LEFT_WRIST, RIGHT_WRIST
    from .posture import hip_centre, shoulder_centre

    if len(bodies) < 2:
        return None
    ordered = sorted(bodies, key=lambda b: float(np.asarray(b.image)[:, 0].mean()))
    made = []
    for body in ordered[:2]:
        image = np.asarray(body.image, dtype=np.float64)
        centre = (hip_centre(image)[:2] + shoulder_centre(image)[:2]) / 2.0
        made.append((
            centre,
            np.stack([image[LEFT_WRIST, :2], image[RIGHT_WRIST, :2]]),
            float(np.linalg.norm(shoulder_centre(image)[:2] - hip_centre(image)[:2])),
        ))
    (a_centre, a_wrists, a_scale), (b_centre, b_wrists, b_scale) = made
    return Pair(at=at, a_centre=a_centre, b_centre=b_centre,
                a_wrists=a_wrists, b_wrists=b_wrists,
                a_scale=a_scale, b_scale=b_scale)
