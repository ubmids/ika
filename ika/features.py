"""Turning 21 raw points into something a model can learn from.

This module is where the project is won or lost, and it is worth being explicit
about why, because it is tempting to skip.

MediaPipe hands you landmarks in image coordinates. Feed those to a classifier
directly and it learns that "point" means *an index finger in the upper left of
the frame at roughly the size my hand was during recording*. Move closer, or
sit slightly to one side, and it falls apart. The model has memorised the
recording session rather than the gesture.

So each hand is re-expressed in a frame built from itself: origin at the wrist,
scale set by the length of the palm, axes derived from the knuckle row. What
comes out is the *shape* of the hand, independent of where it is, how big it
looks, or which way it is turned.

That invariance can be taken too far. Thumbs up and thumbs down are the same
hand shape, and differ only by orientation, so throwing orientation away would
merge two gestures we care about. The fix is to keep both: canonical shape, and
the world orientation as separate features. Nothing is lost and nothing is
entangled.
"""

from __future__ import annotations

import numpy as np

from .schema import (
    INDEX_MCP, INDEX_TIP, MIDDLE_MCP, PINKY_MCP, THUMB_TIP, TIPS, WRIST, FINGERS,
)

EPS = 1e-8

# 63 canonical coordinates, 9 orientation, 5 reach, 4 spread, 1 pinch, 5 curl.
FEATURE_DIM = 63 + 9 + 5 + 4 + 1 + 5


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > EPS else np.zeros_like(v)


def palm_scale(landmarks: np.ndarray) -> float:
    """A size reference that does not change when fingers move.

    Wrist to middle knuckle: both ends are on the rigid part of the hand, so
    curling a finger cannot change it. Using something like wrist-to-fingertip
    would make a fist look like a small hand held far away.
    """
    return float(max(np.linalg.norm(landmarks[MIDDLE_MCP] - landmarks[WRIST]), EPS))


def palm_basis(landmarks: np.ndarray) -> np.ndarray:
    """An orthonormal frame attached to the palm. Rows are the axes.

    Built by Gram-Schmidt from two directions across the rigid knuckle row, so
    it rotates with the hand and is not disturbed by finger articulation.
    """
    wrist = landmarks[WRIST]
    along = _unit(landmarks[MIDDLE_MCP] - wrist)          # wrist toward knuckles
    across = landmarks[INDEX_MCP] - landmarks[PINKY_MCP]  # index side to pinky side
    across = _unit(across - (across @ along) * along)     # orthogonalise
    normal = np.cross(along, across)                      # out of the palm
    return np.stack([along, across, normal])


def mirror_to_right(landmarks: np.ndarray, is_left: bool) -> np.ndarray:
    """Reflect a left hand so it looks like a right one.

    A left hand pointing is the mirror image of a right hand pointing. Without
    this the model has to learn every gesture twice and needs twice the data
    for the same result. Handedness is still passed along separately, so a
    gesture that genuinely depends on which hand is used stays learnable.
    """
    if not is_left:
        return landmarks
    flipped = landmarks.copy()
    flipped[:, 0] *= -1.0
    return flipped


def canonical(landmarks: np.ndarray) -> np.ndarray:
    """Landmarks in the palm's own frame: (21, 3), translation and scale free."""
    centred = (landmarks - landmarks[WRIST]) / palm_scale(landmarks)
    return centred @ palm_basis(landmarks).T


def _curls(canon: np.ndarray) -> np.ndarray:
    """How bent each finger is, as the cosine of the angle at its middle joint.

    Straight reads near 1, folded near -1. Redundant with the raw coordinates
    in principle, but handing the model the quantity it would otherwise have to
    infer makes small training sets go much further.
    """
    out = []
    for mcp, pip, tip in FINGERS.values():
        a = _unit(canon[pip] - canon[mcp])
        b = _unit(canon[tip] - canon[pip])
        out.append(float(a @ b))
    return np.array(out, dtype=np.float32)


def extract(landmarks: np.ndarray, is_left: bool = False) -> np.ndarray:
    """The full feature vector for one hand. Shape (FEATURE_DIM,).

    Laid out as: canonical coordinates, palm orientation, fingertip reach,
    fingertip spread, pinch distance, finger curls.
    """
    landmarks = np.asarray(landmarks, dtype=np.float64).reshape(-1, 3)
    landmarks = mirror_to_right(landmarks, is_left)

    canon = canonical(landmarks)
    basis = palm_basis(landmarks)
    scale = palm_scale(landmarks)

    # How far each fingertip sits from the wrist. Separates fist from open palm
    # in one number per finger.
    reach = np.linalg.norm(landmarks[TIPS] - landmarks[WRIST], axis=1) / scale

    # Gaps between neighbouring fingertips: peace sign versus two fingers held
    # together is entirely this.
    spread = np.linalg.norm(np.diff(landmarks[TIPS], axis=0), axis=1) / scale

    # The single most useful number for control, so it gets to be its own
    # feature rather than something the model has to derive.
    pinch = np.linalg.norm(landmarks[THUMB_TIP] - landmarks[INDEX_TIP]) / scale

    return np.concatenate(
        [
            canon.ravel(),
            basis.ravel(),
            reach,
            spread,
            [pinch],
            _curls(canon),
        ]
    ).astype(np.float32)


def pinch_distance(landmarks: np.ndarray) -> float:
    """Thumb tip to index tip, in palm lengths. Used directly by the controller."""
    landmarks = np.asarray(landmarks, dtype=np.float64).reshape(-1, 3)
    return float(
        np.linalg.norm(landmarks[THUMB_TIP] - landmarks[INDEX_TIP]) / palm_scale(landmarks)
    )
