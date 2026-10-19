"""Plant a known guard habit into real footage, made of real movement.

Nobody on the shadowboxing bench comes with a known habit, so the habit read
and the call had only ever been measured in simulation. This makes the next
best thing: a real person's real round, with a habit put into it on purpose,
so the whole pipeline (the punch reader, the guard read, the miner, the call)
can be asked to find something whose truth is known.

The habit is spliced, not drawn. Every planted sag is a guard drop that two
labellers marked on someone in the bench, cut out of that person's landmarks
as a displacement of the arm from where it was, and added to the recipient's
arm in the recipient's own torso lengths:

- **Real shape.** The donor's elbow, wrist and hand points move exactly as
  they did, in the image and in MediaPipe's world coordinates, resampled to
  the recipient's frame rate. Its duration and depth are the donor's, so the
  spread of plants is the spread of real drops, not one template.
- **Detrended, then tapered.** The donor's own drift over the segment (the
  line between where the arm was at its start and its end) is removed, so
  only the excursion is moved, and a raised-cosine taper takes it to exactly
  zero at both ends. The recipient's stream is never stepped.
- **Scaled to the recipient.** Displacements are carried in torso lengths and
  multiplied back by the recipient's own torso, so a small person farther from
  the camera gets a proportionally small drop.
- **Mirrored by anatomy, not image side.** Sideways movement is carried as
  "away from the other shoulder", so a left-arm drop planted on a right arm,
  or on someone facing the other way, still goes outward and not across.

The trigger is the labelled punches, the truth both labellers agreed on, not
what the punch reader caught. That is deliberate: the habit has to be really
there in the movement, as it would be in a person, whether or not the reader
sees the setup. Triggering from the reader would plant only where the reader
already succeeded and would hide the cost of its misses, which is the whole
question.

A plant is only made where it is physically plausible: inside a labelled
stretch of shadowboxing, after a combo has ended (no agreed punch for
`sag.WINDOW[1]` seconds), with the planted arm clear of any punch either
labeller marked on it (a drop is cut short, and the hand brought back up,
when that arm throws again sooner), and on frames where the pose was found.
Strength is
honoured exactly: of the eligible occurrences of the setup, round(strength x n)
get a drop, chosen with a seeded generator.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .body import (LEFT_ELBOW, LEFT_HIP, LEFT_INDEX, LEFT_PINKY, LEFT_SHOULDER,
                   LEFT_THUMB, LEFT_WRIST, NOSE, RIGHT_ELBOW, RIGHT_HIP, RIGHT_INDEX,
                   RIGHT_PINKY, RIGHT_SHOULDER, RIGHT_THUMB, RIGHT_WRIST)
from .sag import WINDOW

# The joints a planted drop moves, per arm: the elbow, the wrist, and the three
# hand points the pose model hangs off the wrist.
ARM = {
    "left": (LEFT_ELBOW, LEFT_WRIST, LEFT_PINKY, LEFT_INDEX, LEFT_THUMB),
    "right": (RIGHT_ELBOW, RIGHT_WRIST, RIGHT_PINKY, RIGHT_INDEX, RIGHT_THUMB),
}
SHOULDER = {"left": LEFT_SHOULDER, "right": RIGHT_SHOULDER}
OTHER = {"left": "right", "right": "left"}

# Seconds of the donor taken either side of the labelled drop, so the descent
# and the return are carried and not just the time the hand sat low. The
# labels are 0.1 s tiles marking when the hand is down, so the movement into
# it starts a little before the first one.
PAD = 0.2

# The raised-cosine taper at each end of a plant, in seconds.
RAMP = 0.1

# A donor drop is used only when it looks like a sag after a combo: the hand
# down for this long, no longer (a hand down for six seconds is a rest, not a
# lapse) ...
MIN_DOWN, MAX_DOWN = 0.3, 1.5
# ... and when the landmarks actually show it: the wrist on average this far
# below where the segment's own trend puts it, in torso lengths, over the
# labelled span. A drop two labellers saw and the pose did not is not
# movement this can carry.
MIN_DEPTH = 0.10
# ... with the pose found on most of its frames.
MIN_FOUND = 0.8

# The planted arm must be clear of its own punches by this much at both ends
# of the plant, the same margin the guard read uses (`sag.PUNCHING`).
CLEAR = 0.3

# When, after the setup's last punch, the hand is down. Measured on the agreed
# drops of the bench that follow a punch: median 0.7 s, quartiles 0.5 and 1.0.
# Drawn uniformly over the quartiles so a plant lands inside the second the
# guard read looks at, as most real drops after a punch do.
DOWN_AFTER = (0.5, 1.0)

# The shortest plant: the descent, the shortest real drop, and the taper back.
MIN_PLANT = PAD + MIN_DOWN + RAMP


@dataclass(frozen=True)
class Drop:
    """One real labelled drop, as a displacement of one arm.

    `image` and `world` are (frames, joints, 3), in torso lengths, with the
    sideways axis as "away from the other shoulder". `t` is seconds from the
    segment's start; the labelled hand-down moment is at `PAD`.
    """

    donor: str
    side: str
    t: np.ndarray
    image: np.ndarray
    world: np.ndarray
    down: float          # labelled seconds the hand was down
    depth: float         # mean wrist height change over the labelled span, torso lengths

    @property
    def length(self) -> float:
        return float(self.t[-1])


@dataclass(frozen=True)
class Plant:
    """A drop placed in a recipient's round: `drop` starts at `start`."""

    start: float
    side: str
    drop: int            # index into the library
    after: float         # the labelled punch that triggered it
    length: float        # seconds; the donor's, or less if the arm punches again sooner

    def down(self) -> float:
        """When the hand is down: the donor's labelled first-down moment."""
        return self.start + PAD


def _torsos(image: np.ndarray, world: np.ndarray, aspect: float):
    """Per-frame image torso (aspect corrected) and world torso."""
    img = image[:, :, :2].copy()
    img[:, :, 1] /= aspect
    mid = lambda a, i, j: (a[:, i] + a[:, j]) / 2
    torso = np.linalg.norm(mid(img, LEFT_SHOULDER, RIGHT_SHOULDER)
                           - mid(img, LEFT_HIP, RIGHT_HIP), axis=1)
    wtorso = np.linalg.norm(mid(world, LEFT_SHOULDER, RIGHT_SHOULDER)
                            - mid(world, LEFT_HIP, RIGHT_HIP), axis=1)
    return torso, wtorso


def _outward(points: np.ndarray, side: str) -> np.ndarray:
    """+1 where this side's shoulder lies at larger x than the other's, per frame."""
    s = np.sign(points[:, SHOULDER[side], 0] - points[:, SHOULDER[OTHER[side]], 0])
    s[s == 0] = 1.0
    return s


def cut(clip, side: str, start: float, end: float, punches=()) -> Drop | None:
    """A labelled drop on `clip`, from `start` to `end` (hand down), as a Drop.

    Returns None when it is not one this can carry: too short or too long to
    be a sag after a combo, the arm punching during it (`punches`, every punch
    either labeller marked, so no punch movement is carried along with the
    drop), the pose missing, or no drop visible in the landmarks.
    """
    down = end - start
    if not (MIN_DOWN <= down <= MAX_DOWN):
        return None
    a, b = start - PAD, end + PAD
    if any(p.side in (side, "?", "") and a - CLEAR < p.at < b + CLEAR for p in punches):
        return None
    keep = (clip.t >= a) & (clip.t <= b)
    if keep.sum() < 6:
        return None
    t = clip.t[keep] - a
    image = clip.body_image[keep].astype(np.float64)
    world = clip.body_world[keep].astype(np.float64)
    found = np.isfinite(image[:, 0, 0])
    if found.mean() < MIN_FOUND or not found[0] or not found[-1]:
        return None
    # Fill the odd missing frame by interpolation, so the shape stays whole.
    if not found.all():
        for arr in (image, world):
            flat = arr.reshape(len(arr), -1)
            for k in range(flat.shape[1]):
                flat[~found, k] = np.interp(t[~found], t[found], flat[found, k])
    torso, wtorso = _torsos(image, world, clip.aspect)
    if np.any(torso < 1e-6) or np.any(wtorso < 1e-6):
        return None
    img = image[:, :, :2].copy()
    img[:, :, 1] /= clip.aspect
    joints = ARM[side]
    rel_i = (img[:, joints] - img[:, [SHOULDER[side]]]) / torso[:, None, None]
    rel_w = (world[:, joints] - world[:, [SHOULDER[side]]]) / wtorso[:, None, None]
    rel_i[:, :, 0] *= _outward(img, side)[:, None]
    rel_w[:, :, 0] *= _outward(world, side)[:, None]
    rel_i = np.concatenate([rel_i, np.zeros(rel_i.shape[:2] + (1,))], axis=2)
    d_image, d_world = _detrend(t, rel_i), _detrend(t, rel_w)
    # Image y grows downward, so a hand going down is positive dy.
    span = (t >= PAD) & (t <= PAD + down)
    depth = -float(d_image[span, 1, 1].mean())
    if depth > -MIN_DEPTH:
        return None
    return Drop(clip.name, side, t, d_image, d_world, float(down), depth)


def _detrend(t: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Remove the line between the segment's first and last few frames."""
    n = max(1, min(3, len(t) // 4))
    a, b = x[:n].mean(0), x[-n:].mean(0)
    ta, tb = t[:n].mean(), t[-n:].mean()
    frac = ((t - ta) / max(tb - ta, 1e-9))[:, None, None]
    return x - (a + (b - a) * frac)


def taper(t: np.ndarray, length: float, ramp: float = RAMP) -> np.ndarray:
    """1 in the middle, raised-cosine to 0 at both ends of [0, length]."""
    ramp = min(ramp, length / 2)
    w = np.ones_like(t, dtype=np.float64)
    up = t < ramp
    w[up] = 0.5 - 0.5 * np.cos(np.pi * np.clip(t[up], 0, None) / ramp)
    down = t > length - ramp
    w[down] = 0.5 - 0.5 * np.cos(np.pi * np.clip(length - t[down], 0, None) / ramp)
    w[(t < 0) | (t > length)] = 0.0
    return w


def apply(clip, plants: list[Plant], library: list[Drop]):
    """A copy of `clip` with every plant added. The original is untouched."""
    image = clip.body_image.astype(np.float64, copy=True)
    world = clip.body_world.astype(np.float64, copy=True)
    for p in plants:
        drop = library[p.drop]
        rel = clip.t - p.start
        on = (rel >= 0) & (rel <= p.length)
        on &= np.isfinite(image[:, 0, 0])
        if not on.any():
            continue
        idx = np.flatnonzero(on)
        torso, wtorso = _torsos(image[idx], world[idx], clip.aspect)
        # One scale for the whole plant: the recipient's median torso over it,
        # so landmark jitter in the torso does not wobble the drop.
        scale_i, scale_w = float(np.median(torso)), float(np.median(wtorso))
        w = taper(rel[idx], p.length)[:, None]
        corr = np.empty((len(idx), drop.image.shape[1], 3))
        for arr, d, scale, aspect in ((image, drop.image, scale_i, clip.aspect),
                                      (world, drop.world, scale_w, None)):
            for k in range(3):
                for j in range(d.shape[1]):
                    corr[:, j, k] = np.interp(rel[idx], drop.t, d[:, j, k])
            delta = corr * scale
            delta[:, :, 0] *= _outward(arr[idx], p.side)[:, None]
            delta *= w[:, :, None]
            if aspect is not None:
                delta[:, :, 1] *= aspect      # back to raw image y
                delta[:, :, 2] = 0.0          # image z is not carried
            sub = arr[idx]
            sub[:, list(ARM[p.side])] += delta
            arr[idx] = sub
    return replace(clip, body_image=image.astype(clip.body_image.dtype),
                   body_world=world.astype(clip.body_world.dtype))


def combos(punches) -> list[list]:
    """Agreed punches grouped into combos the way the guard read groups them:
    no gap inside a combo longer than `sag.WINDOW[1]`."""
    out = []
    for p in sorted(punches, key=lambda p: p.at):
        if out and p.at - out[-1][-1].at < WINDOW[1]:
            out[-1].append(p)
        else:
            out.append([p])
    return out


def _inside(at: float, windows) -> bool:
    return any(a <= at <= b for a, b in windows)


def _spans(windows) -> list[tuple[float, float]]:
    """Adjacent labelled sheets joined into continuous spans."""
    out = []
    for a, b in sorted(windows):
        if out and a <= out[-1][1] + 1e-6:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def endings(truth, setup: tuple[str, ...] | None) -> list:
    """Combo endings in the labelled truth, matching `setup` (None: any).

    `truth` is a `shadow.Consensus`. Returns (last punch, combo) pairs, for
    combos that end inside a labelled stretch of shadowboxing.
    """
    out = []
    spans = _spans(truth.windows)
    for combo in combos(truth.agreed):
        last = combo[-1]
        if not _inside(last.at, spans):
            continue
        if setup is not None:
            sides = tuple(f"punch_{p.side}" for p in combo[-len(setup):])
            if len(combo) < len(setup) or sides != tuple(setup):
                continue
        out.append((last, combo))
    return out


def room(labelled, side: str, start: float) -> float:
    """Seconds from `start` until this arm must be back in guard: its next
    punch (by either labeller, or one whose arm is in doubt), less CLEAR."""
    nxt = [p.at for p in labelled if p.side in (side, "?", "") and p.at >= start]
    return (min(nxt) - CLEAR - start) if nxt else float("inf")


def fits(clip, truth, labelled, side: str, start: float, length: float) -> bool:
    """Whether a plant on `side` over [start, start + length] is plausible:
    inside one labelled stretch, the arm clear of its own punches by CLEAR at
    both ends, and the pose found on most of its frames."""
    end = start + length
    if length < MIN_PLANT:
        return False
    if not any(a <= start and end <= b for a, b in _spans(truth.windows)):
        return False
    for p in labelled:
        if p.side in (side, "?", "") and start - CLEAR < p.at < end + CLEAR:
            return False
    on = (clip.t >= start) & (clip.t <= end)
    return bool(on.any()) and np.isfinite(clip.body_image[on, 0, 0]).mean() >= MIN_FOUND


def plan(clip, truth, labelled, library: list[Drop], side: str,
         setup: tuple[str, ...] | None, strength: float, seed: int,
         count: int | None = None) -> tuple[list[Plant], int, int]:
    """Choose which occurrences get a drop, and which real drop each gets.

    Each occurrence gets a hand-down moment drawn from DOWN_AFTER and a donor
    drawn at random. When the arm punches again before the donor's drop
    would have ended, the drop is cut short there and the taper brings the
    hand back up: a hand sagging and then coming back for the next punch,
    which is what a real one does. An occurrence with no room even for a
    short drop is not plantable.

    `setup` None means any combo ending (the shuffled control); `count` then
    fixes how many plants, so the control carries as many drops as the habit
    it is compared with. A drop is never put back into the round it came
    from.

    Returns (plants, plantable occurrences, all occurrences).
    """
    rng = np.random.default_rng(seed)
    donors = [i for i, d in enumerate(library) if d.donor != clip.name]
    occurrences = endings(truth, setup)
    eligible = []
    for last, _combo in occurrences:
        lag = float(rng.uniform(*DOWN_AFTER))
        start = last.at + lag - PAD
        pick = int(donors[int(rng.integers(len(donors)))]) if donors else None
        if pick is None:
            continue
        length = min(library[pick].length, room(labelled, side, start))
        if fits(clip, truth, labelled, side, start, length):
            eligible.append(Plant(start, side, pick, last.at, float(length)))
    n = count if count is not None else int(round(strength * len(eligible)))
    n = min(n, len(eligible))
    chosen = sorted(rng.choice(len(eligible), size=n, replace=False).tolist()) if n else []
    return [eligible[i] for i in chosen], len(eligible), len(occurrences)
