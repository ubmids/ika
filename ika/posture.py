"""Turning 33 body landmarks into something a model can learn from.

The hand lane's lesson applies unchanged: raw coordinates teach a classifier
where you stood during recording rather than what you did. So each body is
re-expressed in a frame built from its own torso, which is the rigid part, the
way the knuckle row is the rigid part of a hand. Origin at the hip centre,
scale from torso length, axes from the shoulder line. What survives is posture,
independent of position, distance and which way the person is facing.

Two things are genuinely different from hands, and both matter.

**Most of a body is usually hidden.** A hand is either in frame or it is not.
A body is half occluded almost always: legs behind a bag, an arm behind a
torso, a fighter side-on. Pose landmarkers do not omit what they cannot see,
they *guess*, and return the guess with a plausible-looking coordinate. So
visibility travels with the features rather than being checked once and
forgotten, and any feature built from an unseen joint is reported as unseen too.

**The useful quantities are named, not learned.** For a hand, curls and
fingertip spread were shortcuts that let small datasets go further. For a body
they are the vocabulary of the domain: elbow extension is a punch being thrown,
wrist height against the shoulder is a guard up or dropped, ankle separation is
a stance. Handing a model the quantity a coach would name beats hoping it
rediscovers it from 99 coordinates.
"""

from __future__ import annotations

import numpy as np

from .body import (
    LEFT_ANKLE, LEFT_ELBOW, LEFT_HIP, LEFT_KNEE, LEFT_SHOULDER, LEFT_WRIST,
    N_LANDMARKS, NOSE, RIGHT_ANKLE, RIGHT_ELBOW, RIGHT_HIP, RIGHT_KNEE,
    RIGHT_SHOULDER, RIGHT_WRIST,
)

EPS = 1e-8

# 99 canonical coordinates, 9 orientation, 33 visibility, 14 named quantities.
FEATURE_DIM = 99 + 9 + N_LANDMARKS + 14

# Named offsets into the derived block, because unnamed indices have already
# cost this project one silent bug.
_DERIVED_NAMES = (
    "left_elbow_angle", "right_elbow_angle",
    "left_knee_angle", "right_knee_angle",
    "left_guard_height", "right_guard_height",
    "left_reach", "right_reach",
    "stance_width", "torso_lean",
    "shoulder_twist", "hip_twist",
    "weight_shift", "head_offset",
)
DERIVED_START = 99 + 9 + N_LANDMARKS
DERIVED = {name: DERIVED_START + i for i, name in enumerate(_DERIVED_NAMES)}


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > EPS else np.zeros_like(v)


def hip_centre(marks: np.ndarray) -> np.ndarray:
    return (marks[LEFT_HIP] + marks[RIGHT_HIP]) / 2.0


def shoulder_centre(marks: np.ndarray) -> np.ndarray:
    return (marks[LEFT_SHOULDER] + marks[RIGHT_SHOULDER]) / 2.0


def torso_length(marks: np.ndarray) -> float:
    """Hip centre to shoulder centre. The body's scale reference.

    Both ends sit on the torso, so a limb moving cannot change it. Using
    something like height would make a crouch read as a smaller person.
    """
    return float(max(np.linalg.norm(shoulder_centre(marks) - hip_centre(marks)), EPS))


def torso_basis(marks: np.ndarray) -> np.ndarray:
    """An orthonormal frame attached to the torso. Rows are the axes.

    Built by Gram-Schmidt from the spine and the shoulder line, so it turns
    with the body and is undisturbed by any limb.
    """
    up = _unit(shoulder_centre(marks) - hip_centre(marks))
    across = marks[LEFT_SHOULDER] - marks[RIGHT_SHOULDER]
    across = _unit(across - (across @ up) * up)
    forward = np.cross(up, across)
    return np.stack([up, across, forward])


def canonical(marks: np.ndarray) -> np.ndarray:
    """Landmarks in the torso's own frame: (33, 3), translation and scale free."""
    centred = (marks - hip_centre(marks)) / torso_length(marks)
    return centred @ torso_basis(marks).T


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Cosine of the angle at `b`. Straight reads near -1, folded near +1.

    Returned as a cosine rather than radians so it is bounded and needs no
    normalisation, matching how the hand lane reports finger curl.
    """
    return float(_unit(a - b) @ _unit(c - b))


def derived(marks: np.ndarray) -> np.ndarray:
    """The quantities a coach would name. Shape (14,).

    Every one is scaled by torso length or already dimensionless, so none of
    them change when the person moves nearer the camera.
    """
    scale = torso_length(marks)
    hips = hip_centre(marks)
    shoulders = shoulder_centre(marks)
    basis = torso_basis(marks)
    up = basis[0]

    def height_above_shoulder(joint: int) -> float:
        """Along the spine, so it still means "high" on a leaning fighter."""
        return float((marks[joint] - shoulders) @ up / scale)

    return np.array([
        # Elbow extension. A thrown punch is an arm straightening.
        _angle(marks[LEFT_SHOULDER], marks[LEFT_ELBOW], marks[LEFT_WRIST]),
        _angle(marks[RIGHT_SHOULDER], marks[RIGHT_ELBOW], marks[RIGHT_WRIST]),
        _angle(marks[LEFT_HIP], marks[LEFT_KNEE], marks[LEFT_ANKLE]),
        _angle(marks[RIGHT_HIP], marks[RIGHT_KNEE], marks[RIGHT_ANKLE]),
        # Guard: where the hands are relative to the shoulders.
        height_above_shoulder(LEFT_WRIST),
        height_above_shoulder(RIGHT_WRIST),
        # Reach: how far the hands are from the body's centre.
        float(np.linalg.norm(marks[LEFT_WRIST] - shoulders) / scale),
        float(np.linalg.norm(marks[RIGHT_WRIST] - shoulders) / scale),
        # Footwork.
        float(np.linalg.norm(marks[LEFT_ANKLE] - marks[RIGHT_ANKLE]) / scale),
        # Lean: how far the spine is off the world vertical.
        float(_unit(shoulders - hips)[1]),
        # Rotation of shoulders and hips about the spine. The gap between them
        # is torque, which is where power in a strike comes from.
        float(_unit(marks[LEFT_SHOULDER] - marks[RIGHT_SHOULDER])[2]),
        float(_unit(marks[LEFT_HIP] - marks[RIGHT_HIP])[2]),
        # Weight: hips relative to the midpoint of the feet.
        float(
            ((hips - (marks[LEFT_ANKLE] + marks[RIGHT_ANKLE]) / 2.0) @ basis[1]) / scale
        ),
        # Head off the centre line, which is what slipping a punch looks like.
        float(((marks[NOSE] - shoulders) @ basis[1]) / scale),
    ], dtype=np.float32)


def extract(marks: np.ndarray, visibility: np.ndarray | None = None) -> np.ndarray:
    """The full feature vector for one body. Shape (FEATURE_DIM,).

    Laid out as canonical coordinates, torso orientation, per-landmark
    visibility, then the named quantities.

    Visibility is part of the vector rather than a filter applied beforehand.
    Zeroing hidden joints would be worse than useless: it would place an
    unseen wrist at the hip centre, which is a specific and wrong posture
    rather than an absent one. Passing the confidence through lets a model
    learn to discount what the landmarker was guessing at.
    """
    marks = np.asarray(marks, dtype=np.float64).reshape(-1, 3)
    if marks.shape[0] != N_LANDMARKS:
        raise ValueError(f"expected {N_LANDMARKS} landmarks, got {marks.shape[0]}")
    if visibility is None:
        visibility = np.ones(N_LANDMARKS, dtype=np.float32)

    return np.concatenate([
        canonical(marks).ravel(),
        torso_basis(marks).ravel(),
        np.asarray(visibility, dtype=np.float32).ravel(),
        derived(marks),
    ]).astype(np.float32)


def confidence(visibility: np.ndarray, joints: tuple[int, ...]) -> float:
    """How much to trust a feature built from these joints: the weakest link.

    A quantity like elbow extension is only as good as the least visible of
    the three landmarks it is computed from, so the minimum is the honest
    summary rather than the mean.
    """
    return float(np.min(np.asarray(visibility)[list(joints)]))


# Which landmarks each named quantity actually depends on, so a caller can ask
# whether a specific read is trustworthy on this frame.
DEPENDS_ON: dict[str, tuple[int, ...]] = {
    "left_elbow_angle": (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
    "right_elbow_angle": (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
    "left_knee_angle": (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
    "right_knee_angle": (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),
    "left_guard_height": (LEFT_WRIST, LEFT_SHOULDER, RIGHT_SHOULDER),
    "right_guard_height": (RIGHT_WRIST, LEFT_SHOULDER, RIGHT_SHOULDER),
    "left_reach": (LEFT_WRIST, LEFT_SHOULDER, RIGHT_SHOULDER),
    "right_reach": (RIGHT_WRIST, LEFT_SHOULDER, RIGHT_SHOULDER),
    "stance_width": (LEFT_ANKLE, RIGHT_ANKLE),
    "torso_lean": (LEFT_HIP, RIGHT_HIP, LEFT_SHOULDER, RIGHT_SHOULDER),
    "shoulder_twist": (LEFT_SHOULDER, RIGHT_SHOULDER),
    "hip_twist": (LEFT_HIP, RIGHT_HIP),
    "weight_shift": (LEFT_ANKLE, RIGHT_ANKLE, LEFT_HIP, RIGHT_HIP),
    "head_offset": (NOSE, LEFT_SHOULDER, RIGHT_SHOULDER),
}
