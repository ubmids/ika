"""Real shadowboxing footage as a test bench: replay, truth, and scoring.

Everything the drill loop claimed about punches had been measured on hands
generated with arithmetic. This module is how it meets real people without
anyone standing in front of a camera: public follow-along workouts, landmarks
cached once by `scripts/shadow_extract.py`, and two independent sets of punch
labels read off contact sheets (`scripts/shadow_sheets.py`).

**Who labelled.** The labellers are not people. Each set was made by a
separate instance of a vision-language model (Claude) looking at the contact
sheets, with the same written instructions and no sight of the other's
labels. That is what made labelling twenty minutes of footage twice feasible
at all, and it is also why agreement is measured rather than assumed: two
model labellers can share a blind spot that two people would not.

Two labellers rather than one because a label set nobody has checked is an
opinion. Agreement between them is reported first, and a detector is never
scored more finely than the labels agree: if the labellers only match each
other 80% of the time, a detector at 95% against one of them is noise.

The footage itself is not in the repository, only the scripts that fetch it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .body import LEFT_SHOULDER, RIGHT_SHOULDER, Body
from .hands import Hand

ROOT = Path(__file__).resolve().parent.parent / "data" / "shadow"
# The labels are ours and small, so they are committed. The footage and the
# landmarks cached from it are not; `scripts/shadow_fetch.py` rebuilds both.
BENCH = Path(__file__).resolve().parent.parent / "bench" / "shadow"

# Two labels, or a label and a detection, are the same punch when this close.
# One contact-sheet tile is 0.1 s, a punch peak lasts about two, and a
# detector committing on the way out rather than at full extension lands a
# tile or so early. Wider than this and two punches of a fast one-two, 0.25 s
# apart, would start matching each other.
MATCH_SECONDS = 0.2


@dataclass(frozen=True)
class Punch:
    at: float
    side: str            # the thrower's own "left" or "right"
    kind: str = ""
    sure: bool = True


@dataclass
class Clip:
    """One clip's cached landmarks, replayable frame by frame."""

    name: str
    t: np.ndarray
    body_image: np.ndarray
    body_world: np.ndarray
    body_vis: np.ndarray
    hand_image: np.ndarray
    hand_left: np.ndarray
    width: int
    height: int

    @classmethod
    def load(cls, name: str, root: Path = ROOT) -> Clip:
        m = np.load(root / "marks" / f"{name}.npz")
        return cls(name, m["t"], m["body_image"], m["body_world"], m["body_vis"],
                   m["hand_image"], m["hand_left"], int(m["width"]), int(m["height"]))

    @property
    def aspect(self) -> float:
        return self.width / self.height

    def body(self, i: int) -> Body | None:
        if not np.isfinite(self.body_image[i, 0, 0]):
            return None
        return Body(self.body_image[i], self.body_world[i], self.body_vis[i])

    def hands(self, i: int) -> list[Hand]:
        return [Hand(self.hand_image[i, k], self.hand_image[i, k],
                     "Left" if self.hand_left[i, k] else "Right", 1.0)
                for k in range(self.hand_left.shape[1]) if self.hand_left[i, k] >= 0]

    def frames(self):
        for i in range(len(self.t)):
            yield float(self.t[i]), self.hands(i), self.body(i)

    def anatomical(self, at: float, image_side: str) -> str:
        """Which of the thrower's arms has its shoulder on that side of the image.

        Decided per moment from the pose rather than assumed from "facing the
        camera", because a fighter turning side-on swaps nothing anatomically
        but can put both shoulders on one half of the frame.
        """
        i = int(np.clip(np.searchsorted(self.t, at), 0, len(self.t) - 1))
        lx, rx = self.body_image[i, LEFT_SHOULDER, 0], self.body_image[i, RIGHT_SHOULDER, 0]
        if not (np.isfinite(lx) and np.isfinite(rx)):
            return "right" if image_side == "image_left" else "left"
        left_is_image_left = lx < rx
        if image_side == "image_left":
            return "left" if left_is_image_left else "right"
        return "right" if left_is_image_left else "left"


@dataclass
class Labels:
    """One labeller's view of one clip."""

    punches: list[Punch]
    windows: list[tuple[float, float]]    # spans labelled as real shadowboxing
    covered: list[tuple[float, float]]    # every span the labeller looked at


def load_labels(clip: Clip, who: str, root: Path = BENCH) -> Labels | None:
    # A labeller's later rounds live in "<who>2", "<who>3" and so on, one file
    # per clip per round, so a new round never rewrites an earlier one.
    paths = [p for p in (root / "labels" / f"{who}{n}" / f"{clip.name}.json"
                         for n in ("", "2", "3", "4")) if p.exists()]
    if not paths:
        return None
    sheets = [s for p in paths for s in json.loads(p.read_text())["sheets"]]
    punches, windows, covered = [], [], []
    for sheet in sheets:
        start = float(sheet["start"])
        span = (start, start + 3.0)
        covered.append(span)
        if sheet.get("activity") == "shadowboxing":
            windows.append(span)
        marks = sheet.get("punches", [])
        # One labeller wrote "walking" where the list of punches belonged.
        # A sheet whose punches are not a list has none.
        for p in marks if isinstance(marks, list) else []:
            at = start + int(p["tile"]) / 10.0
            punches.append(Punch(at, clip.anatomical(at, p.get("arm", "")),
                                 p.get("kind", ""), bool(p.get("sure", True))))
    punches.sort(key=lambda p: p.at)
    return Labels(punches, windows, covered)


def person_of(name: str, bench: Path = BENCH) -> str:
    """Who is in a clip. Two clips from one coach are one person, so a
    leave-one-person-out split must hold them out together or it leaks."""
    rows = json.loads((bench / "clips.json").read_text())["clips"]
    return next((r["person"] for r in rows if r["id"] == name), name)


def inside(at: float, spans) -> bool:
    return any(a <= at < b for a, b in spans)


def match(truth, found, tolerance: float = MATCH_SECONDS, sided: bool = True):
    """Greedy one-to-one matching by time, nearest first.

    Returns (pairs, missed, extra). One-to-one matters: a detector firing
    three times on one punch should score one hit and two false punches,
    not three hits.
    """
    candidates = []
    for i, a in enumerate(truth):
        for j, b in enumerate(found):
            gap = abs(a.at - b.at)
            if gap <= tolerance and (not sided or a.side == b.side):
                candidates.append((gap, i, j))
    candidates.sort()
    used_t, used_f, pairs = set(), set(), []
    for _, i, j in candidates:
        if i in used_t or j in used_f:
            continue
        used_t.add(i); used_f.add(j)
        pairs.append((truth[i], found[j]))
    missed = [p for i, p in enumerate(truth) if i not in used_t]
    extra = [p for j, p in enumerate(found) if j not in used_f]
    return pairs, missed, extra


@dataclass
class Consensus:
    """What two labellers agree on, and how much they agree."""

    agreed: list[Punch]
    disputed: list[Punch]
    windows: list[tuple[float, float]]
    agreement: float          # F1 between the two labellers, sided
    agreement_unsided: float  # the same, ignoring which arm


def consensus(a: Labels, b: Labels) -> Consensus:
    """Punches both labellers saw, matched without regard to arm.

    Which arm threw it is kept only when both say the same arm. The labellers
    agree that a punch happened far more often than they agree on the arm,
    81 to 100% against as low as 17% on a fighter seen side-on, so tying the
    two together would throw away real punches over an argument about the arm.
    """
    windows = [w for w in a.windows if w in b.windows]
    pa = [p for p in a.punches if inside(p.at, windows)]
    pb = [p for p in b.punches if inside(p.at, windows)]
    sided, _, _ = match(pa, pb)
    pairs, only_a, only_b = match(pa, pb, sided=False)
    agreed = [Punch((x.at + y.at) / 2, x.side if x.side == y.side else "?",
                    x.kind or y.kind, x.sure and y.sure)
              for x, y in pairs]
    total = len(pa) + len(pb)
    return Consensus(
        agreed=sorted(agreed, key=lambda p: p.at),
        disputed=sorted(only_a + only_b, key=lambda p: p.at),
        windows=windows,
        agreement=2 * len(sided) / total if total else 1.0,
        agreement_unsided=2 * len(pairs) / total if total else 1.0,
    )


@dataclass
class Score:
    hits: int
    missed: int
    false: int
    minutes: float
    wrong_side: int

    @property
    def recall(self) -> float:
        total = self.hits + self.missed
        return self.hits / total if total else 1.0

    @property
    def false_per_minute(self) -> float:
        return self.false / self.minutes if self.minutes else 0.0

    def __add__(self, other: Score) -> Score:
        return Score(self.hits + other.hits, self.missed + other.missed,
                     self.false + other.false, self.minutes + other.minutes,
                     self.wrong_side + other.wrong_side)


def score(found: list[Punch], truth: Consensus) -> Score:
    """Recall against what both labellers agree on; false punches against both.

    A detection landing on a disputed label is neither credited nor charged:
    one of the two labellers saw a punch there, and the honest reading is
    that nobody knows. So recall is measured on the agreed set and a false
    punch is one that neither labeller saw. The arm is scored separately, on
    the punches whose arm both labellers agree on.
    """
    found = [p for p in found if inside(p.at, truth.windows)]
    pairs, missed, extra = match(truth.agreed, found, sided=False)
    _, _, unexplained = match(truth.disputed, extra, sided=False)
    minutes = sum(b - a for a, b in truth.windows) / 60.0
    wrong = sum(1 for t, f in pairs if t.side != "?" and t.side != f.side)
    return Score(len(pairs), len(missed), len(unexplained), minutes, wrong)
