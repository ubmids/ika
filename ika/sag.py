"""Where your hands go after you punch, measured rather than detected.

The drill's headline read is "you drop your right hand after two rights". The
first way it was built needed a dropped guard to be an event: a wrist that
crossed a threshold and stayed there. Measured on real footage that event is
the weakest thing in the whole loop. The two labellers agreed on only
about half of the drops each other marked, the detector found about 30% of
the ones they agreed on, and a simulation with those exact error rates showed
the habit read collapsing: a planted habit named 22% of the time after
twenty-four minutes of drilling, against 100% in six with a perfect sensor.

So this asks the question differently. Not "was there a drop", which is a
binary call made thirty times a second on a noisy signal, but "how high was
that hand, on average, in the second after this combo", compared with the
second after every other combo. Averaging is what rescues it. Per frame, a
wrist in a labelled drop sits 1.16 standard deviations below one in guard;
averaged over the window after a punch, and then over every time the setup
happens, the noise that ruined the event detector mostly cancels.

Guard height is read in the image, as the wrist's height above the nose in
torso lengths, and relative to this person's own recent guard: the 75th
percentile of the last twenty seconds, since between punches hands are mostly
up. The image and not MediaPipe's world coordinates, because on the labelled
footage the image signal separated drops from guard better (1.16 against 0.88).

Frames where that arm is itself punching are left out of the window. A
straight is thrown at shoulder height, which is below the nose, so without
this every one-two would look like a sagging guard.

Output is counts, in the same shape as the miner's findings, so the profile
pools it across sessions with the same statistics and the caller can arm it
the same way. An occasion counts as "hand down" when the window's average sits
more than DOWN below the guard.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .body import LEFT_HIP, LEFT_SHOULDER, NOSE, RIGHT_HIP, RIGHT_SHOULDER
import math

from .tell.habits import Finding, benjamini_hochberg

WRISTS = {"left": 15, "right": 16}

BASELINE_SECONDS = 20.0
BASELINE_PERCENTILE = 75.0

# The window after a punch that is read, in seconds from the punch. It starts
# late enough for the hand to have come back, and is short enough that the
# read belongs to this combo and not the next.
WINDOW = (0.35, 1.2)

# Frames this close to a punch by the same arm are the punch, not the guard.
PUNCHING = 0.3

# An occasion counts as "hand down" when the window's mean is this far below
# the guard, in torso lengths. On the labelled footage, frames in a drop sat
# 0.34 below the rolling guard on average and frames in guard 0.08, so this
# sits between the two.
DOWN = 0.2

# At least this many readable frames in a window, or the occasion is skipped.
MIN_FRAMES = 6

MIN_SUPPORT = 10
MIN_LIFT = 1.35

# What one session may propose to the profile. See drill_app for the measured
# trade that picked it.
PROPOSAL_FDR = 0.1


@dataclass
class _Frame:
    at: float
    height: dict[str, float]      # relative to the rolling guard; NaN if unread


class GuardTrack:
    """Streaming guard heights, relative to this person's own guard."""

    def __init__(self, aspect: float = 1.0, keep_seconds: float = 900.0):
        self.aspect = float(aspect)
        self.keep_seconds = float(keep_seconds)
        self._recent = {side: deque() for side in WRISTS}
        self.frames: deque[_Frame] = deque()

    def reset(self) -> None:
        for d in self._recent.values():
            d.clear()
        self.frames.clear()

    def update(self, body, at: float) -> None:
        heights = {side: float("nan") for side in WRISTS}
        if body is not None:
            needed = (NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)
            image = np.asarray(body.image, dtype=np.float64)[:, :2].copy()
            image[:, 1] /= self.aspect
            torso = float(np.linalg.norm(
                (image[LEFT_SHOULDER] + image[RIGHT_SHOULDER]) / 2
                - (image[LEFT_HIP] + image[RIGHT_HIP]) / 2))
            if torso > 1e-6 and min(float(body.visibility[j]) for j in needed) >= 0.5:
                for side, w in WRISTS.items():
                    if body.visibility[w] < 0.5:
                        continue
                    raw = -(image[w, 1] - image[NOSE, 1]) / torso
                    recent = self._recent[side]
                    recent.append((at, raw))
                    while recent and at - recent[0][0] > BASELINE_SECONDS:
                        recent.popleft()
                    if len(recent) >= 30:
                        guard = float(np.percentile([h for _, h in recent], BASELINE_PERCENTILE))
                        heights[side] = raw - guard
        self.frames.append(_Frame(at, heights))
        while self.frames and at - self.frames[0].at > self.keep_seconds:
            self.frames.popleft()

    def snapshot(self):
        """The track as arrays, for reading many windows without rescanning."""
        t = np.fromiter((f.at for f in self.frames), dtype=np.float64)
        h = {side: np.fromiter((f.height[side] for f in self.frames), dtype=np.float64)
             for side in WRISTS}
        return t, h

    def after(self, at: float, side: str, punches: list[tuple[float, str]],
              snapshot=None) -> float | None:
        """Mean guard height of one arm in the window after `at`, or None."""
        t, h = snapshot if snapshot is not None else self.snapshot()
        lo, hi = at + WINDOW[0], at + WINDOW[1]
        a, b = np.searchsorted(t, lo), np.searchsorted(t, hi, side="right")
        times, values = t[a:b], h[side][a:b]
        keep = np.isfinite(values)
        for p, s in punches:
            if s == side and lo - PUNCHING <= p <= hi + PUNCHING:
                keep &= np.abs(times - p) > PUNCHING
        if keep.sum() < MIN_FRAMES:
            return None
        return float(values[keep].mean())


@dataclass(frozen=True)
class Occasion:
    context: tuple[str, ...]
    side: str
    height: float

    @property
    def down(self) -> bool:
        return self.height < -DOWN


def lower_than(inside: list[float], rest: list[float]) -> float:
    """One-sided Mann-Whitney U: do `inside` heights sit lower than `rest`?

    A rank test on the heights themselves. Two earlier versions threw
    information away first and paid for it. Counting each occasion as "down"
    or "up" and testing the counts turns a 0.87 standard deviation difference
    into a coin that lands differently 68% against 34% of the time; simulated
    at the measured noise, that named a planted habit 28% of the time in
    twelve minutes. And a binomial against the other occasions' rate treats
    that rate as known when it is estimated from twenty or thirty combos,
    which on shuffled real data let 14% of nulls through. Ranks keep the
    magnitude, need no assumption about its distribution, and compare the two
    samples as the two samples they are.

    Normal approximation with a tie correction, which is accurate at the
    sample sizes allowed through (MIN_SUPPORT on each side).
    """
    n1, n2 = len(inside), len(rest)
    values = np.concatenate([inside, rest])
    order = values.argsort(kind="mergesort")
    ranks = np.empty(len(values))
    sorted_values = values[order]
    i = 0
    tie_term = 0.0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        t = j - i + 1
        tie_term += t ** 3 - t
        i = j + 1
    u = ranks[:n1].sum() - n1 * (n1 + 1) / 2.0
    mean = n1 * n2 / 2.0
    n = n1 + n2
    var = n1 * n2 / 12.0 * ((n + 1) - tie_term / (n * (n - 1)))
    if var <= 0:
        return 1.0
    z = (u - mean + 0.5) / math.sqrt(var)          # continuity-corrected, lower tail
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def combo_of(punches: list[tuple[float, str]], i: int) -> list[str]:
    """The combo punch `i` belongs to, up to and including it: the run of
    punches before it with no gap longer than WINDOW[1]. A setup never spans
    two combos, or the last right of one and the first of the next would make
    "two rights" out of what the fighter threw as two separate things."""
    j = i
    while j > 0 and punches[j][0] - punches[j - 1][0] < WINDOW[1]:
        j -= 1
    return [f"punch_{s}" for _, s in punches[j:i + 1]]


def current_combo(punches: list[tuple[float, str]]) -> list[str]:
    """The combo in progress, for the caller: what "the setup" means live."""
    ordered = sorted(punches)
    return combo_of(ordered, len(ordered) - 1) if ordered else []


def occasions(track: GuardTrack, punches: list[tuple[float, str]],
              max_context: int = 2) -> list[Occasion]:
    """Every punch, the setup it ended, and where each hand was afterwards.

    Only the last punch of a combo is an occasion. A punch followed by another
    within the window is still mid-combo, and "where was the hand after it"
    has no meaning while the next punch is being thrown.
    """
    out = []
    snap = track.snapshot()
    ordered = sorted(punches)
    for i, (at, _side) in enumerate(ordered):
        if i + 1 < len(ordered) and ordered[i + 1][0] - at < WINDOW[1]:
            continue
        combo = combo_of(ordered, i)
        for k in range(1, max_context + 1):
            if len(combo) < k:
                break
            context = tuple(combo[-k:])
            for side in WRISTS:
                h = track.after(at, side, ordered, snap)
                if h is not None:
                    out.append(Occasion(context, side, h))
    return out


def read(track: GuardTrack, punches: list[tuple[float, str]],
         fdr: float = 0.05) -> list[Finding]:
    """Which setups leave a hand lower than your other combos do.

    Each candidate is "after this setup, that hand is down", counted over
    occasions and tested against the rate on every other occasion of the same
    context length: the same binomial test and false-discovery control the
    sequence miner uses, so the two kinds of finding are held to one standard.
    """
    return read_occasions(occasions(track, punches), fdr=fdr)


def read_occasions(occ: list[Occasion], fdr: float = 0.05) -> list[Finding]:
    """`read`, from occasions already collected. Split out so a simulation can
    hand it occasions with known habits in them."""
    candidates = []
    for k in (1, 2):
        pool = [o for o in occ if len(o.context) == k]
        for side in WRISTS:
            mine = [o for o in pool if o.side == side]
            for context in {o.context for o in mine}:
                inside = [o for o in mine if o.context == context]
                rest = [o for o in mine if o.context != context]
                if len(inside) < MIN_SUPPORT or len(rest) < MIN_SUPPORT:
                    continue
                hits = sum(o.down for o in inside)
                others = sum(o.down for o in rest)
                base = max(others / len(rest), 1.0 / (len(rest) + 1))
                rate = hits / len(inside)
                # Every candidate goes into the correction, however small its
                # effect. Filtering on lift first means the correction only ever
                # sees the few that already look strong, so it never pays for
                # having looked at the rest: measured on shuffled real
                # occasions, that let 25% of nulls report a habit.
                candidates.append(Finding(
                    context=context, then=f"guard_down_{side}", support=len(inside),
                    hits=hits, probability=rate, base=base,
                    p_value=lower_than([o.height for o in inside],
                                       [o.height for o in rest])))
    keep = benjamini_hochberg([c.p_value for c in candidates], fdr=fdr)
    found = [c for c, ok in zip(candidates, keep) if ok and c.lift >= MIN_LIFT]
    # The same reduction the sequence miner makes. A two-punch setup is only
    # worth saying if it beats its own last punch; and when it does by a wide
    # margin, the one-punch version is a diluted echo of it and goes.
    def ends(longer, shorter):
        return len(shorter) < len(longer) and longer[-len(shorter):] == shorter

    kept = [f for f in found
            if not any(g.then == f.then and ends(f.context, g.context)
                       and f.probability <= g.probability * 1.2 for g in found)]
    final = [f for f in kept
             if not any(g.then == f.then and ends(g.context, f.context)
                        and g.probability > f.probability * 1.5 for g in kept)]
    return sorted(final, key=lambda f: -f.lift)
