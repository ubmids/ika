"""Body actions as movement: a punch is a posture changing, not a posture.

This is what lets the whole `tell` and `early` pipeline run on body features
instead of on abstract action labels. Each action interpolates between
postures over time with a bell velocity profile, because a hand does not
travel at constant speed and training on constant-velocity punches would
produce a model expecting something nobody does.

Feints are here for the same reason they are in the hand lane: without a
movement whose opening lies, early commitment looks far better than it is.
A feinted jab commits to the jab for part of its length and then becomes
something else, labelled by what it becomes.

Still synthetic, and still crude. A real jab involves weight transfer, hip
drive and a retraction that this does not model. What it does capture is the
thing the features are supposed to key on: which quantity changes, how far,
and how fast.
"""

from __future__ import annotations

import numpy as np

from . import bodysynth, posture
from .body import LEFT_WRIST, N_LANDMARKS, RIGHT_WRIST
from .bodymotion import Frame

# start posture, end posture, duration range in seconds. Durations are what a
# person actually takes: strikes are fast, footwork and guard changes slower.
ACTIONS: dict[str, tuple[dict, dict, tuple[float, float]]] = {
    "idle":        (dict(guard=1.0), dict(guard=1.0), (0.30, 0.60)),
    "jab_left":    (dict(guard=1.0), dict(guard=1.0, left_extension=1.0, twist=0.20), (0.14, 0.24)),
    "cross_right": (dict(guard=1.0), dict(guard=1.0, right_extension=1.0, twist=-0.38), (0.18, 0.30)),
    "drop_guard":  (dict(guard=1.0), dict(guard=0.0), (0.22, 0.40)),
    "raise_guard": (dict(guard=0.0), dict(guard=1.0), (0.18, 0.34)),
    "slip_left":   (dict(guard=1.0), dict(guard=1.0, lean=0.38, head_slip=1.0), (0.18, 0.32)),
    "crouch":      (dict(guard=1.0), dict(guard=1.0, crouch=1.0), (0.26, 0.46)),
}

# Pairs whose openings genuinely resemble each other, so the early frames are
# honestly ambiguous rather than artificially so. A jab and a cross both start
# from a guard with a shoulder turning; a slip and a crouch both start dropping.
FEINTS = (
    ("jab_left", "cross_right"),
    ("cross_right", "jab_left"),
    ("jab_left", "drop_guard"),
    ("slip_left", "crouch"),
    ("crouch", "slip_left"),
    ("jab_left", "idle"),
)


def _smoothstep(u: np.ndarray) -> np.ndarray:
    """Ease in and out: zero velocity at both ends, peak in the middle."""
    return 3 * u**2 - 2 * u**3


def _blend(start: dict, end: dict, amount: float) -> dict:
    keys = set(start) | set(end)
    return {
        k: start.get(k, _DEFAULTS.get(k, 0.0)) * (1 - amount)
           + end.get(k, _DEFAULTS.get(k, 0.0)) * amount
        for k in keys
    }


_DEFAULTS = {"guard": 1.0, "stance": 1.0, "left_extension": 0.0,
             "right_extension": 0.0, "crouch": 0.0, "twist": 0.0,
             "lean": 0.0, "head_slip": 0.0}


def _to_frames(
    poses: list[np.ndarray],
    fps: float,
    rng: np.random.Generator,
    noise: float,
    occlude_legs: bool,
    at0: float = 0.0,
) -> list[Frame]:
    """Turn world postures into window frames, with the camera's limitations.

    Legs are marked unseen by default because that is what a webcam actually
    delivers: measured over 400 real photographs, knees were visible in 4% of
    them and ankles in none. Generating fully-visible bodies would train a
    model on information it will not have.
    """
    from .body import LEFT_ANKLE, LEFT_HEEL, LEFT_FOOT, LEFT_HIP, LEFT_KNEE
    from .body import RIGHT_ANKLE, RIGHT_HEEL, RIGHT_FOOT, RIGHT_HIP, RIGHT_KNEE

    hidden = (LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
              LEFT_HEEL, RIGHT_HEEL, LEFT_FOOT, RIGHT_FOOT)

    out: list[Frame] = []
    for i, pose in enumerate(poses):
        world = bodysynth.place(pose, noise=noise, seed=int(rng.integers(1 << 30)))
        visibility = np.ones(N_LANDMARKS, dtype=np.float32) * float(rng.uniform(0.85, 1.0))
        if occlude_legs:
            # Replace the coordinates, not just the confidence. The first
            # version only lowered visibility while still handing over the
            # true leg positions, so the model could read knees it was told it
            # could not see, and `crouch` scored 85% on a camera that cannot
            # observe it. A landmarker does not withhold an unseen joint, it
            # extrapolates a plausible one, so that is what is simulated here:
            # legs somewhere below the hips, with the error a guess carries.
            hips = (world[LEFT_HIP] + world[RIGHT_HIP]) / 2.0
            guess_rng = np.random.default_rng(int(rng.integers(1 << 30)))
            for joint, depth in zip(hidden, (0.8, 0.8, 1.5, 1.5, 1.6, 1.6, 1.7, 1.7)):
                visibility[joint] = float(guess_rng.uniform(0.0, 0.15))
                world[joint] = hips + np.array([
                    guess_rng.normal(0.0, 0.18),
                    -depth + guess_rng.normal(0.0, 0.22),
                    guess_rng.normal(0.0, 0.18),
                ])
        # A rough perspective projection: the world posture placed in frame.
        image = np.zeros((N_LANDMARKS, 3), dtype=np.float64)
        image[:, 0] = 0.5 + world[:, 0] * 0.16
        image[:, 1] = 0.55 - world[:, 1] * 0.16
        image[:, 2] = world[:, 2]
        out.append(
            Frame(
                at=at0 + i / fps,
                features=posture.extract(world, visibility),
                wrists=np.stack([image[LEFT_WRIST, :2], image[RIGHT_WRIST, :2]]),
                scale=0.16,
                visibility=visibility,
            )
        )
    return out


def make(
    name: str,
    fps: float = 30.0,
    seed: int = 0,
    noise: float = 0.006,
    occlude_legs: bool = True,
) -> list[Frame]:
    """One instance of a named body action."""
    if name not in ACTIONS:
        raise KeyError(f"unknown action {name!r}; have {sorted(ACTIONS)}")
    rng = np.random.default_rng(seed)
    start, end, (low, high) = ACTIONS[name]

    duration = float(rng.uniform(low, high))
    count = max(4, int(round(duration * fps)))
    amounts = _smoothstep(np.linspace(0.0, 1.0, count))
    # A little variation in how far the action is carried through, since
    # nobody throws the same punch twice.
    reach = float(rng.uniform(0.85, 1.0))
    poses = [bodysynth.body_pose(**_blend(start, end, a * reach)) for a in amounts]
    return _to_frames(poses, fps, rng, noise, occlude_legs)


def feint(
    looks_like: str,
    becomes: str,
    switch: float = 0.45,
    fps: float = 30.0,
    seed: int = 0,
    noise: float = 0.006,
    occlude_legs: bool = True,
) -> list[Frame]:
    """A movement that begins as one action and turns into another."""
    rng = np.random.default_rng(seed)
    first = make(looks_like, fps=fps, seed=int(rng.integers(1 << 30)),
                 noise=noise, occlude_legs=occlude_legs)
    second = make(becomes, fps=fps, seed=int(rng.integers(1 << 30)),
                  noise=noise, occlude_legs=occlude_legs)
    cut = max(2, int(len(first) * switch))
    joined = first[:cut] + second
    return [
        Frame(i / fps, f.features, f.wrists, f.scale, f.visibility)
        for i, f in enumerate(joined)
    ]


def dataset(
    per_class: int = 250,
    classes: list[str] | None = None,
    feint_rate: float = 0.3,
    fps: float = 30.0,
    seed: int = 0,
) -> tuple[list[list[Frame]], np.ndarray, list[str]]:
    """Whole movements with labels, ready for prefixing or windowing."""
    classes = classes or list(ACTIONS)
    rng = np.random.default_rng(seed)
    windows, labels = [], []
    for label, name in enumerate(classes):
        for _ in range(per_class):
            windows.append(make(name, fps=fps, seed=int(rng.integers(1 << 30))))
            labels.append(label)
    for _ in range(int(per_class * len(classes) * feint_rate)):
        looks_like, becomes = FEINTS[rng.integers(len(FEINTS))]
        windows.append(feint(looks_like, becomes,
                             switch=float(rng.uniform(0.3, 0.6)),
                             fps=fps, seed=int(rng.integers(1 << 30))))
        labels.append(classes.index(becomes))
    return windows, np.array(labels, dtype=np.int64), classes
