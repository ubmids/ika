"""Dynamic gestures, invented.

Same purpose as `synth` for static poses: a labelled problem that is genuinely
learnable, so the sequence model can be built and measured before a single real
swipe is recorded.

The velocity profile matters more than it looks. A hand does not translate at
constant speed, it accelerates and decelerates, so paths here follow a
smoothstep rather than a ramp. Training on constant-velocity swipes would
produce a model that expects something people do not do.
"""

from __future__ import annotations

import numpy as np

from . import features, synth
from .motion import Sample
from .schema import DYNAMIC_GESTURES


def _smoothstep(u: np.ndarray) -> np.ndarray:
    """Ease in and out. Zero velocity at both ends, peak in the middle."""
    return 3 * u**2 - 2 * u**3


def _frames(
    pose_name: str,
    n: int,
    fps: float,
    start: np.ndarray,
    net: np.ndarray,
    rng: np.random.Generator,
    pinch_curve: np.ndarray | None = None,
    jitter: float = 0.0015,
    span: float = 0.16,
) -> list[Sample]:
    """A held pose carried along a path, with an optional changing pinch."""
    u = np.linspace(0.0, 1.0, n)
    path = start[None, :] + net[None, :] * _smoothstep(u)[:, None]
    path = path + rng.normal(0.0, jitter, path.shape)

    out: list[Sample] = []
    for i in range(n):
        spec = dict(synth.POSES[pose_name])
        turn = spec.pop("rotate", None)
        if pinch_curve is not None:
            spec["pinch"] = float(pinch_curve[i])
        hand = synth.synthetic_hand(**spec)
        if turn is not None:
            hand = synth.place(hand, rotate=synth.rotation(*turn))
        hand = synth.place(
            hand,
            rotate=synth.rotation(*rng.normal(0, 0.05, 3)),
            noise=0.006,
            seed=int(rng.integers(1 << 30)),
        )
        out.append(
            Sample(
                at=i / fps,
                position=path[i],
                span=span * float(rng.normal(1.0, 0.02)),
                pinch=features.pinch_distance(hand),
                pose=features.extract(hand),
            )
        )
    return out


def make(name: str, fps: float = 30.0, seed: int = 0) -> list[Sample]:
    """One synthetic example of a named dynamic gesture."""
    rng = np.random.default_rng(seed)
    centre = np.array([0.5, 0.5]) + rng.normal(0, 0.05, 2)

    if name == "none":
        # Idle. Not stillness, which would be trivially separable, but the
        # aimless drift a hand actually does while you think.
        n = int(0.7 * fps)
        wander = rng.normal(0, 0.012, 2)
        pose = rng.choice(["rest", "open_palm", "point", "fist"])
        return _frames(str(pose), n, fps, centre, wander, rng, jitter=0.004)

    if name.startswith("swipe_"):
        direction = {
            "swipe_left": np.array([-1.0, 0.0]),
            "swipe_right": np.array([1.0, 0.0]),
            "swipe_up": np.array([0.0, -1.0]),
            "swipe_down": np.array([0.0, 1.0]),
        }[name]
        n = int(rng.uniform(0.35, 0.6) * fps)
        distance = rng.uniform(0.22, 0.40)
        wobble = rng.normal(0, 0.05, 2)   # nobody swipes perfectly straight
        pose = rng.choice(["open_palm", "point"])
        return _frames(str(pose), n, fps, centre, direction * distance + wobble, rng)

    if name == "pinch_drag":
        # A pinch held while the hand travels: same path shape as a swipe, but
        # slower, shorter, and with the fingers together the whole way. The
        # pose is what separates them, not the trajectory.
        n = int(rng.uniform(0.5, 0.7) * fps)
        angle = rng.uniform(0, 2 * np.pi)
        distance = rng.uniform(0.08, 0.18)
        net = np.array([np.cos(angle), np.sin(angle)]) * distance
        pinch = np.full(n, 0.95) + rng.normal(0, 0.02, n)
        return _frames("pinch", n, fps, centre, net, rng, pinch_curve=np.clip(pinch, 0, 1))

    if name == "snap":
        # Barely moves, but the shape changes fast. The mirror image of a
        # swipe, which is why straightness and pose drift both earn their place
        # in the feature vector.
        n = int(0.45 * fps)
        u = np.linspace(0, 1, n)
        # closed, then flicking open partway through
        pinch = np.where(u < 0.45, 0.95, np.clip(0.95 - (u - 0.45) * 3.0, 0.0, 0.95))
        drift = rng.normal(0, 0.02, 2)
        return _frames("pinch", n, fps, centre, drift, rng, pinch_curve=pinch, jitter=0.003)

    raise KeyError(f"unknown dynamic gesture {name!r}; have {DYNAMIC_GESTURES}")


def dataset(
    per_class: int = 300,
    classes: list[str] | None = None,
    fps: float = 30.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[str], list[list[Sample]]]:
    """Motion features and labels, plus the raw windows for the sequence model."""
    from .motion import features as motion_features

    classes = classes or list(DYNAMIC_GESTURES)
    rng = np.random.default_rng(seed)

    vectors, labels, windows = [], [], []
    for label, name in enumerate(classes):
        for _ in range(per_class):
            window = make(name, fps=fps, seed=int(rng.integers(1 << 30)))
            vectors.append(motion_features(window))
            labels.append(label)
            windows.append(window)
    return (
        np.stack(vectors),
        np.array(labels, dtype=np.int64),
        classes,
        windows,
    )


def _idle(n: int, fps: float, at0: float, centre: np.ndarray,
          rng: np.random.Generator, span: float = 0.16) -> list[Sample]:
    """Filler frames: a hand present but not doing anything.

    Deliberately varied, and more varied than any gesture class. Idle is what a
    hand does almost all of the time and it does it in endless ways, so a thin
    negative class is the fastest route to a system that fires at you while you
    think. An earlier version used two poses and a small constant drift; on
    idle motion it had not seen, it produced over twenty false firings a
    minute.
    """
    if n <= 0:
        return []
    poses = ["rest", "open_palm", "point", "fist", "peace", "l_shape"]
    # A momentum random walk rather than a straight drift: real idle motion
    # wanders, changes its mind, and occasionally moves quite fast.
    position = centre.copy()
    velocity = rng.normal(0, 0.004, 2)
    out: list[Sample] = []
    name = str(rng.choice(poses))
    for i in range(n):
        if rng.random() < 0.05:
            name = str(rng.choice(poses))
        velocity = velocity * rng.uniform(0.80, 0.97) + rng.normal(0, 0.006, 2)
        position = np.clip(position + velocity, 0.08, 0.92)
        spec = dict(POSES_FOR_IDLE.get(name, {}))
        curl = {k: float(np.clip(v + rng.normal(0, 0.15), 0, 1))
                for k, v in spec.get("curl", {}).items()}
        hand = synth.synthetic_hand(
            curl=curl,
            spread=float(max(0.0, spec.get("spread", 0.0) + rng.normal(0, 0.1))),
            pinch=float(np.clip(rng.uniform(0.0, 0.5), 0, 1)),
        )
        hand = synth.place(hand, rotate=synth.rotation(*rng.normal(0, 0.12, 3)),
                           noise=0.008, seed=int(rng.integers(1 << 30)))
        out.append(
            Sample(
                at=at0 + i / fps,
                position=position.copy(),
                span=span * float(rng.normal(1.0, 0.04)),
                pinch=features.pinch_distance(hand),
                pose=features.extract(hand),
            )
        )
    return out


POSES_FOR_IDLE = {k: v for k, v in synth.POSES.items()}


def make_in_context(
    name: str,
    fps: float = 30.0,
    window: float = 0.7,
    seed: int = 0,
    min_overlap: float = 0.6,
) -> tuple[list[Sample], bool]:
    """A gesture embedded in idle time, then cropped by a rolling window.

    This is the situation the live app is actually in. The window does not know
    where a gesture starts, so it catches beginnings, middles, ends and the
    quiet either side. Training on perfectly aligned windows produces a model
    that scores brilliantly offline and then fires at the wrong moments,
    because alignment was doing work the model never had to learn.

    Returns the cropped window and whether enough of the gesture is inside it
    to deserve the label. A window holding the tail of a swipe is honestly
    closer to nothing than to a swipe, and labelling it otherwise teaches the
    model to fire on fragments.
    """
    rng = np.random.default_rng(seed)
    centre = np.array([0.5, 0.5]) + rng.normal(0, 0.05, 2)

    if name == "none":
        n = int(window * fps * 1.6)
        return _idle(n, fps, 0.0, centre, rng)[: int(window * fps)], True

    core = make(name, fps=fps, seed=int(rng.integers(1 << 30)))
    length = int(window * fps)

    # Idle padding on both sides, generously longer than the window, so a crop
    # can genuinely miss part of the gesture. An earlier version used short
    # padding and every crop happened to contain the whole gesture, which made
    # the overlap rule below dead code and the dataset no harder than the
    # aligned one. A test now asserts that fragments actually occur.
    lead = int(rng.uniform(1.0, 1.6) * length)
    tail = int(rng.uniform(1.0, 1.6) * length)

    before = _idle(lead, fps, 0.0, core[0].position, rng)
    after = _idle(tail, fps, 0.0, core[-1].position, rng)
    timeline = [Sample(i / fps, s.position, s.span, s.pinch, s.pose)
                for i, s in enumerate(before + core + after)]

    # Slide the window across the gesture rather than the whole timeline, so
    # overlap is spread across the full range from a bare clip to the lot.
    # Cropping uniformly over the timeline would mostly return pure idle and
    # drown every class in "none".
    first = max(0, lead - length + 2)
    last = min(len(timeline) - length, lead + len(core) - 2)
    start = int(rng.integers(first, max(first + 1, last + 1)))
    cropped = timeline[start : start + length]

    inside = len(range(max(start, lead), min(start + length, lead + len(core))))
    return cropped, (inside / len(core)) >= min_overlap


def realistic_dataset(
    per_class: int = 300,
    classes: list[str] | None = None,
    fps: float = 30.0,
    window: float = 0.7,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[str], list[list[Sample]]]:
    """Like `dataset`, but with the window sliding rather than aligned.

    Fragments that fall below the overlap threshold are relabelled `none`,
    which is both honest and what makes the resulting model safe to run: it
    learns that a partial gesture is not a command.
    """
    from .motion import features as motion_features

    classes = classes or list(DYNAMIC_GESTURES)
    none_label = classes.index("none")
    rng = np.random.default_rng(seed)

    vectors, labels, windows = [], [], []
    for label, name in enumerate(classes):
        for _ in range(per_class):
            cropped, enough = make_in_context(
                name, fps=fps, window=window, seed=int(rng.integers(1 << 30))
            )
            if len(cropped) < 6:
                continue
            vectors.append(motion_features(cropped))
            labels.append(label if enough else none_label)
            windows.append(cropped)
    return np.stack(vectors), np.array(labels, dtype=np.int64), classes, windows
