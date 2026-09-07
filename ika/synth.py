"""A hand made of arithmetic.

Nothing here needs a camera, a dataset or a person. That matters for two
reasons: the invariance properties in `features` are claims that should be
checked rather than hoped for, and every later stage (model, sequence model,
state machine) can be exercised before a single real gesture is recorded.

The hand is a crude kinematic chain, not anatomy. It only has to be
articulated enough that a fist and an open palm are genuinely different, and
consistent enough that rotating it is a fair test.
"""

from __future__ import annotations

import numpy as np

from .schema import FINGERS, N_LANDMARKS, PINKY_MCP, THUMB_CMC, WRIST

# Knuckle positions in a palm-shaped layout. Rows: index, middle, ring, pinky.
_MCP = np.array(
    [[0.30, 0.95, 0.0], [0.00, 1.00, 0.0], [-0.28, 0.95, 0.0], [-0.52, 0.85, 0.0]]
)
_SEGMENTS = {"index": 0.36, "middle": 0.40, "ring": 0.36, "pinky": 0.30}


def _rotate_about_x(v: np.ndarray, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([v[0], c * v[1] - s * v[2], s * v[1] + c * v[2]])


def synthetic_hand(
    curl: dict[str, float] | None = None,
    spread: float = 0.0,
    pinch: float = 0.0,
) -> np.ndarray:
    """A right hand in a canonical pose. Returns (21, 3).

    `curl` maps finger name to 0 (straight) through 1 (fully folded). `spread`
    fans the fingers apart, which is what separates a peace sign from two
    fingers held together. `pinch` draws the thumb tip toward the index tip,
    1.0 meaning they touch.

    The pinch is applied by aiming the thumb rather than by curling it, because
    a curl angle alone will not bring the tips together: the first attempt at
    this fixture produced a "pinch" whose tips were further apart than a fist's,
    which would have made every test built on it meaningless.
    """
    curl = curl or {}
    hand = np.zeros((N_LANDMARKS, 3))
    hand[WRIST] = (0.0, 0.0, 0.0)

    for row, name in enumerate(("index", "middle", "ring", "pinky")):
        mcp_index, pip_index, tip_index = FINGERS[name]
        dip_index = tip_index - 1
        base = _MCP[row].copy()
        base[0] *= 1.0 + spread
        hand[mcp_index] = base

        amount = float(np.clip(curl.get(name, 0.0), 0.0, 1.0))
        length = _SEGMENTS[name]
        point = base.copy()
        direction = np.array([0.0, 1.0, 0.0])
        # Each joint bends a little further than the last, the way a finger does.
        for joint, index in enumerate((pip_index, dip_index, tip_index)):
            direction = _rotate_about_x(direction, amount * 0.75 * (joint + 1) / 2.0)
            point = point + direction * length
            hand[index] = point

    # The thumb leaves the hand sideways rather than along it, which is the
    # whole reason a pinch is possible.
    amount = float(np.clip(curl.get("thumb", 0.0), 0.0, 1.0))
    hand[THUMB_CMC] = (0.30, 0.20, 0.05)
    direction = np.array([0.62, 0.72, 0.18])
    direction = direction / np.linalg.norm(direction)
    point = hand[THUMB_CMC].copy()
    for joint, index in enumerate((THUMB_CMC + 1, THUMB_CMC + 2, THUMB_CMC + 3)):
        turn = amount * 0.55 * (joint + 1)
        turned = np.array(
            [
                direction[0] * np.cos(turn) - direction[1] * np.sin(turn) * 0.6,
                direction[0] * np.sin(turn) * 0.6 + direction[1] * np.cos(turn),
                direction[2] - amount * 0.18 * joint,
            ]
        )
        turned = turned / max(np.linalg.norm(turned), 1e-9)
        point = point + turned * 0.30
        hand[index] = point
        direction = turned

    hand[PINKY_MCP][0] *= 1.0 + spread

    if pinch > 0.0:
        amount = float(np.clip(pinch, 0.0, 1.0))
        index_tip = hand[FINGERS["index"][2]]
        thumb_tip_index = FINGERS["thumb"][2]
        # Move the tip most of the way and the joint behind it half as far, so
        # the thumb stays a plausible chain rather than snapping straight.
        hand[thumb_tip_index] = hand[thumb_tip_index] * (1 - amount) + index_tip * amount
        joint = thumb_tip_index - 1
        target = (hand[joint] + index_tip) / 2.0
        hand[joint] = hand[joint] * (1 - amount * 0.5) + target * (amount * 0.5)
    return hand


def rotation(yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0) -> np.ndarray:
    """A 3x3 rotation, composed yaw then pitch then roll."""
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    ry = np.array([[cr, 0, sr], [0, 1, 0], [-sr, 0, cr]])
    return rz @ rx @ ry


def place(
    hand: np.ndarray,
    rotate: np.ndarray | None = None,
    translate: tuple[float, float, float] = (0.0, 0.0, 0.0),
    scale: float = 1.0,
    noise: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Put a hand somewhere in the world: rotated, moved, resized, jittered.

    Exactly the transformations `features.extract` claims not to care about, so
    this is the tool that checks the claim.
    """
    out = hand * scale
    if rotate is not None:
        out = out @ rotate.T
    out = out + np.asarray(translate, dtype=float)
    if noise:
        out = out + np.random.default_rng(seed).normal(0.0, noise, out.shape)
    return out


# Recognisable poses, so tests and demos can ask for a fist by name.
POSES: dict[str, dict] = {
    "rest":       {"curl": {"thumb": 0.4, "index": 0.45, "middle": 0.45, "ring": 0.5, "pinky": 0.55}},
    "open_palm":  {"curl": {}, "spread": 0.35},
    "fist":       {"curl": {"thumb": 0.9, "index": 1.0, "middle": 1.0, "ring": 1.0, "pinky": 1.0}},
    "point":      {"curl": {"thumb": 0.7, "index": 0.0, "middle": 1.0, "ring": 1.0, "pinky": 1.0}},
    "pinch":      {"curl": {"thumb": 0.35, "index": 0.35, "middle": 0.95, "ring": 1.0, "pinky": 1.0}, "pinch": 0.95},
    "peace":      {"curl": {"thumb": 0.8, "index": 0.0, "middle": 0.0, "ring": 1.0, "pinky": 1.0}, "spread": 0.5},
    "thumbs_up":  {"curl": {"thumb": 0.0, "index": 1.0, "middle": 1.0, "ring": 1.0, "pinky": 1.0}},
    # The same hand as thumbs_up, turned over. It is listed as a rotation
    # rather than a separate shape because that is exactly what it is, and it
    # keeps the pair honest: any augmentation that rotates freely enough to
    # confuse these two has destroyed a gesture we claim to support.
    "thumbs_down": {"curl": {"thumb": 0.0, "index": 1.0, "middle": 1.0, "ring": 1.0, "pinky": 1.0},
                    "rotate": (0.0, 0.0, np.pi)},
    "l_shape":    {"curl": {"thumb": 0.0, "index": 0.0, "middle": 1.0, "ring": 1.0, "pinky": 1.0}},
}


def pose(name: str) -> np.ndarray:
    """A named synthetic pose, including any base rotation it is defined with."""
    if name not in POSES:
        raise KeyError(f"unknown pose {name!r}; have {sorted(POSES)}")
    spec = dict(POSES[name])
    turn = spec.pop("rotate", None)
    hand = synthetic_hand(**spec)
    return hand if turn is None else place(hand, rotate=rotation(*turn))
