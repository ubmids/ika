"""A body made of arithmetic.

Same job `synth` does for hands: a posture whose truth is known before
anything runs, so the claims in `posture` can be checked rather than believed.
The one that matters most is that the named quantities measure what their names
say. It is easy to write a function called `guard_height` and never notice it
returns something else.

Crude on purpose. It is a jointed stick figure with no anatomy beyond
proportion, which is enough to make a dropped guard genuinely different from a
raised one and a thrown punch different from a folded arm.
"""

from __future__ import annotations

import numpy as np

from .body import (
    LEFT_ANKLE, LEFT_EAR, LEFT_ELBOW, LEFT_EYE, LEFT_EYE_INNER, LEFT_EYE_OUTER,
    LEFT_FOOT, LEFT_HEEL, LEFT_HIP, LEFT_INDEX, LEFT_KNEE, LEFT_PINKY,
    LEFT_SHOULDER, LEFT_THUMB, LEFT_WRIST, MOUTH_LEFT, MOUTH_RIGHT,
    N_LANDMARKS, NOSE, RIGHT_ANKLE, RIGHT_EAR, RIGHT_ELBOW, RIGHT_EYE,
    RIGHT_EYE_INNER, RIGHT_EYE_OUTER, RIGHT_FOOT, RIGHT_HEEL, RIGHT_HIP,
    RIGHT_INDEX, RIGHT_KNEE, RIGHT_PINKY, RIGHT_SHOULDER, RIGHT_THUMB,
    RIGHT_WRIST,
)

# Proportions as fractions of torso length, which is the scale everything in
# `posture` is measured against.
TORSO = 1.0
SHOULDER_HALF = 0.42
HIP_HALF = 0.28
UPPER_ARM = 0.52
FOREARM = 0.48
THIGH = 0.78
SHIN = 0.74
HEAD = 0.32


def body_pose(
    guard: float = 1.0,
    left_extension: float = 0.0,
    right_extension: float = 0.0,
    stance: float = 1.0,
    crouch: float = 0.0,
    twist: float = 0.0,
    lean: float = 0.0,
    head_slip: float = 0.0,
) -> np.ndarray:
    """A fighter's posture. Returns (33, 3) in a world frame, y is up.

    `guard` 1 means hands at chin, 0 means hands at the waist. `*_extension`
    0 means the arm folded, 1 means straight out in front. `stance` scales
    foot separation, `crouch` bends the knees, `twist` rotates the shoulders
    against the hips, `lean` tips the spine sideways and `head_slip` moves the
    head off the centre line.
    """
    marks = np.zeros((N_LANDMARKS, 3), dtype=np.float64)

    # Hips at the origin, spine leaning by `lean`.
    marks[LEFT_HIP] = (HIP_HALF, 0.0, 0.0)
    marks[RIGHT_HIP] = (-HIP_HALF, 0.0, 0.0)
    hips = np.zeros(3)

    spine = np.array([np.sin(lean), np.cos(lean), 0.0]) * TORSO * (1.0 - 0.18 * crouch)
    shoulders = hips + spine

    # Shoulders rotate about the spine by `twist`, so the shoulder line and the
    # hip line come apart. That gap is torque.
    across = np.array([np.cos(twist), 0.0, np.sin(twist)]) * SHOULDER_HALF
    marks[LEFT_SHOULDER] = shoulders + across
    marks[RIGHT_SHOULDER] = shoulders - across

    for side, shoulder, elbow, wrist, sign in (
        ("L", LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST, 1.0),
        ("R", RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST, -1.0),
    ):
        extension = left_extension if side == "L" else right_extension
        origin = marks[shoulder]
        # Folded: elbow down and in, wrist up at the chin. Extended: both out
        # in front along +z, arm nearly straight.
        elbow_folded = origin + np.array([sign * 0.10, -UPPER_ARM * 0.85, 0.18])
        elbow_out = origin + np.array([sign * 0.16, -UPPER_ARM * 0.25, UPPER_ARM * 0.92])
        marks[elbow] = elbow_folded * (1 - extension) + elbow_out * extension

        chin = shoulders + np.array([sign * 0.14, HEAD * 0.45, 0.22])
        waist = hips + np.array([sign * 0.24, 0.16, 0.16])
        wrist_folded = chin * guard + waist * (1.0 - guard)
        wrist_out = marks[elbow] + np.array([sign * 0.03, 0.02, FOREARM * 0.97])
        marks[wrist] = wrist_folded * (1 - extension) + wrist_out * extension

        # Hand landmarks hang off the wrist; nothing here reads them, but the
        # array has to be complete for the tracker's contract.
        forward = np.array([sign * 0.04, 0.0, 0.07])
        for offset, index in ((1.0, LEFT_PINKY if side == "L" else RIGHT_PINKY),
                              (1.2, LEFT_INDEX if side == "L" else RIGHT_INDEX),
                              (0.8, LEFT_THUMB if side == "L" else RIGHT_THUMB)):
            marks[index] = marks[wrist] + forward * offset

    for hip, knee, ankle, heel, foot, sign in (
        (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE, LEFT_HEEL, LEFT_FOOT, 1.0),
        (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE, RIGHT_HEEL, RIGHT_FOOT, -1.0),
    ):
        # Two-link inverse kinematics in the sagittal plane, rather than
        # nudging the knee forward by a fudge factor. The first version did the
        # latter and produced a "crouch" whose knee angle barely moved, which
        # meant the fixture could not have caught a broken knee-angle feature.
        #
        # The foot is planted, the hip drops with `crouch`, and the knee goes
        # wherever two fixed-length bones require. That is what makes the knee
        # genuinely bend.
        spread = sign * HIP_HALF * stance
        drop = 1.0 - 0.30 * crouch
        ankle_pos = marks[hip] + np.array([spread * 0.85, -(THIGH + SHIN) * drop, 0.0])
        marks[ankle] = ankle_pos

        span = np.linalg.norm(ankle_pos - marks[hip])
        span = float(np.clip(span, abs(THIGH - SHIN) + 1e-6, THIGH + SHIN - 1e-6))
        # Distance along the hip-to-ankle line where the knee's perpendicular
        # offset is measured from, then the offset itself.
        along = (span**2 + THIGH**2 - SHIN**2) / (2 * span)
        out = float(np.sqrt(max(THIGH**2 - along**2, 0.0)))
        direction = (ankle_pos - marks[hip]) / span
        # Knees bend forwards, which is +z here.
        perpendicular = np.array([0.0, 0.0, 1.0])
        perpendicular = perpendicular - (perpendicular @ direction) * direction
        norm = np.linalg.norm(perpendicular)
        perpendicular = perpendicular / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0])
        marks[knee] = marks[hip] + direction * along + perpendicular * out

        marks[heel] = marks[ankle] + np.array([0.0, -0.06, -0.09])
        marks[foot] = marks[ankle] + np.array([0.0, -0.07, 0.16])

    head = shoulders + np.array([head_slip * 0.30, HEAD, 0.02])
    marks[NOSE] = head + np.array([0.0, 0.0, 0.10])
    for index, dx, dy in (
        (LEFT_EYE, 0.055, 0.03), (RIGHT_EYE, -0.055, 0.03),
        (LEFT_EYE_INNER, 0.03, 0.03), (RIGHT_EYE_INNER, -0.03, 0.03),
        (LEFT_EYE_OUTER, 0.08, 0.03), (RIGHT_EYE_OUTER, -0.08, 0.03),
        (LEFT_EAR, 0.11, 0.0), (RIGHT_EAR, -0.11, 0.0),
        (MOUTH_LEFT, 0.04, -0.06), (MOUTH_RIGHT, -0.04, -0.06),
    ):
        marks[index] = head + np.array([dx, dy, 0.06])

    return marks


def place(
    marks: np.ndarray,
    rotate: np.ndarray | None = None,
    translate: tuple[float, float, float] = (0.0, 0.0, 0.0),
    scale: float = 1.0,
    noise: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Put a body somewhere: rotated, moved, resized, jittered.

    Exactly what `posture.extract` claims not to care about.
    """
    out = marks * scale
    if rotate is not None:
        out = out @ rotate.T
    out = out + np.asarray(translate, dtype=float)
    if noise:
        out = out + np.random.default_rng(seed).normal(0.0, noise, out.shape)
    return out


def rotation(yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0) -> np.ndarray:
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    return (
        np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        @ np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        @ np.array([[cr, 0, sr], [0, 1, 0], [-sr, 0, cr]])
    )


POSTURES: dict[str, dict] = {
    "guard":        {"guard": 1.0, "stance": 1.0},
    "guard_down":   {"guard": 0.0, "stance": 1.0},
    "jab_left":     {"guard": 1.0, "left_extension": 1.0, "twist": 0.18},
    "cross_right":  {"guard": 1.0, "right_extension": 1.0, "twist": -0.35},
    "crouch":       {"guard": 1.0, "crouch": 1.0},
    "wide_stance":  {"guard": 1.0, "stance": 2.0},
    "slip_left":    {"guard": 1.0, "lean": 0.35, "head_slip": 1.0},
}


def posture(name: str) -> np.ndarray:
    if name not in POSTURES:
        raise KeyError(f"unknown posture {name!r}; have {sorted(POSTURES)}")
    return body_pose(**POSTURES[name])
