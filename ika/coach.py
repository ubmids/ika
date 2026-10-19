"""Coach mode: ika calls the combos, so it already knows the setup.

The drill has to work out what you threw before it can say anything about
what your hands did afterwards, and that is where it loses most of its
evidence. The punch reader catches 47% of punches held out by person, so a
two-punch setup is caught whole about a fifth of the time, and a strong
guard habit takes weeks of rounds to name.

Pad work and follow-along classes turn that around. The coach calls "one,
two", you throw it. Here ika is the coach. It already knows the setup,
because it said it, so the punch reader is only asked a much easier
question: did anything get thrown in the second or two after the call. The
guard read is the same one the drill uses (`sag.py`), taken in the second
after the response ends, and the same rank test asks whether one called
combo leaves a hand lower than the others do.

Numbering is the usual boxing one: odd numbers are the lead hand, even the
rear. For an orthodox stance the lead is the left.

    1 jab   2 cross   3 lead hook   4 rear hook   5 lead uppercut   6 rear uppercut

Defensive calls (slip, roll) are left out on purpose. Guard height is read
relative to the shoulders, and a slip or a roll at the end of a combo tilts
and lowers them with the hands in ways nobody has measured, so it could show
up as a change of guard that is not one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import sag
from .tell.habits import Finding

NAMES = {1: "jab", 2: "cross", 3: "lead hook", 4: "rear hook",
         5: "lead uppercut", 6: "rear uppercut"}
SPOKEN = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}

# Twelve standard combos, one to four punches. The order is only where a
# rotation starts; which ones a session uses is `choose`.
CATALOGUE = ("1-2", "1-1-2", "1-2-3", "2-3-2", "1-2-3-2", "3-2",
             "1-6-3-2", "1-2-5-2", "5-6", "2-3", "1-2-1-2", "2")

# How many combos one session drills. The rank test needs at least
# sag.MIN_SUPPORT answered calls of a combo, and as many of the others, in
# one session. A three-minute round fits about 47 calls of these combos: four
# combos would get 11 or 12 each, so two unanswered calls would leave a combo
# untested, while three get 13 to 17 each, which survives the misses the
# punch reader makes. So three, and one default round can test every combo
# it calls. More rounds add the same again.
PER_SESSION = 3

# Time from one call to the next, in seconds, for a combo of n punches:
# BASE_GAP + PER_PUNCH * n. It has to hold the call, the reaction, the punches,
# and then the full guard window after them (sag.WINDOW ends 1.2 s after the
# last punch), with room to spare before the next call is spoken. See
# `window` for the arithmetic; a test holds the two together.
BASE_GAP = 2.4
PER_PUNCH = 0.5

# When a response can land. Nobody reacts to a word in under 0.2 s; the last
# punch of n is expected by RESPONSE_FIRST + PUNCH_INTERVAL * (n - 1) and is
# accepted until RESPONSE_SLACK + RESPONSE_PER_PUNCH * n.
REACTION = 0.2
RESPONSE_FIRST = 0.7
PUNCH_INTERVAL = 0.3
RESPONSE_SLACK = 1.2
RESPONSE_PER_PUNCH = 0.4

# How long the start of a round waits after "box", so the first call is not
# spoken over it.
LEAD_IN = 2.0

# Habit contexts from coach mode carry this prefix. A drill habit's context is
# the punches the reader saw; a coach habit's is the combo that was called.
# They are different evidence and must never pool into one count.
PREFIX = "coach_"

REMINDERS = {"guard_down_left": "left hand up", "guard_down_right": "right hand up"}


@dataclass(frozen=True)
class Combo:
    name: str                       # "1-2"
    numbers: tuple[int, ...]

    @classmethod
    def parse(cls, name: str) -> Combo:
        numbers = tuple(int(n) for n in name.split("-"))
        if not numbers or any(n not in NAMES for n in numbers):
            raise ValueError(f"a combo is numbers 1 to 6 joined by '-', not {name!r}")
        return cls("-".join(str(n) for n in numbers), numbers)

    def __len__(self) -> int:
        return len(self.numbers)

    @property
    def words(self) -> str:
        return ", ".join(SPOKEN[n] for n in self.numbers)

    @property
    def context(self) -> tuple[str, ...]:
        return (PREFIX + self.name,)

    def sides(self, stance: str = "orthodox") -> tuple[str, ...]:
        lead, rear = ("left", "right") if stance == "orthodox" else ("right", "left")
        return tuple(lead if n % 2 else rear for n in self.numbers)


def gap(combo: Combo) -> float:
    return BASE_GAP + PER_PUNCH * len(combo)


def window(at: float, combo: Combo) -> tuple[float, float]:
    """When a response to a call at `at` can land."""
    return at + REACTION, at + RESPONSE_SLACK + RESPONSE_PER_PUNCH * len(combo)


def expected_end(at: float, combo: Combo) -> float:
    """When the last punch should land if the person keeps a normal pace."""
    return at + RESPONSE_FIRST + PUNCH_INTERVAL * (len(combo) - 1)


def is_coach(context) -> bool:
    return bool(context) and str(context[0]).startswith(PREFIX)


def combo_of(context) -> str:
    return str(context[0])[len(PREFIX):]


@dataclass(frozen=True)
class Call:
    at: float
    combo: Combo
    remind: str = ""

    @property
    def words(self) -> str:
        return self.combo.words + (f"... {self.remind}" if self.remind else "")


def choose(seed: int = 0, known=(), k: int = PER_SESSION,
           catalogue=CATALOGUE) -> list[Combo]:
    """Which combos a session drills: known habits first, then a seeded draw.

    A combo that is a known habit stays in, so it keeps being measured and a
    habit that has been coached out can be seen fading. The rest are drawn
    from the catalogue so that different sessions cover different combos.
    """
    picked = []
    for name in known:
        combo = Combo.parse(name)
        if combo not in picked:
            picked.append(combo)
    rng = np.random.default_rng(seed)
    for i in rng.permutation(len(catalogue)):
        if len(picked) >= k:
            break
        combo = Combo.parse(catalogue[int(i)])
        if combo not in picked:
            picked.append(combo)
    return picked[:k]


def plan(record, now: float, seed: int = 0, k: int = PER_SESSION) -> list[Combo]:
    """The combos for the next session, from what the profile already holds.

    Standing coach habits first. Then combos a past session proposed but that
    are not yet standing, most recent first: a single session cannot make a
    habit stand (the profile asks for two), so a hunch has to be called again
    to be confirmed or dropped, the way a coach would follow one up. Those
    follow-ups take at most k - 1 places, so every session also tries
    something new, and the rest come from the seeded rotation.
    """
    standing = [combo_of(h.context) for h in record.habits(now=now) if is_coach(h.context)]
    pending = sorted((o for o in record.observations
                      if is_coach(o.context) and combo_of(o.context) not in standing),
                     key=lambda o: -o.at)
    hunches = []
    for o in pending:
        if combo_of(o.context) not in hunches:
            hunches.append(combo_of(o.context))
    known = standing[:k] + hunches[:max(0, k - 1 - len(standing))]
    return choose(seed=seed, known=known, k=k)


def schedule(combos: list[Combo], seconds: float = 180.0, seed: int = 0,
             start: float = 0.0) -> list[Call]:
    """Every call of one round, in order, with when it is said.

    Dealt like cards: one of each combo, shuffled, then another hand, so every
    combo is called the same number of times to within one, and spread evenly
    through the round. Spreading matters for the statistics and not only for
    variety. The rank test compares one combo's guard heights with the
    others', and a guard that sags as the round tires someone would make
    whichever combo was called late look like a habit if calls came in blocks.
    A hand never starts with the combo the last one ended on.
    """
    if not combos:
        return []
    rng = np.random.default_rng(seed)
    calls, at, last = [], start, None
    while True:
        hand = [combos[int(i)] for i in rng.permutation(len(combos))]
        if len(hand) > 1 and hand[0] == last:
            hand[0], hand[-1] = hand[-1], hand[0]
        for combo in hand:
            if at + gap(combo) > start + seconds:
                return calls
            calls.append(Call(at, combo))
            at += gap(combo)
            last = combo


def reps_per_combo(combos: list[Combo], seconds: float = 180.0) -> float:
    """How many times each combo is called in a round, on average."""
    mean_gap = float(np.mean([gap(c) for c in combos]))
    return seconds / mean_gap / len(combos)


@dataclass(frozen=True)
class Rep:
    """One call and what followed it."""

    call: Call
    answered: bool
    end: float | None                     # when the response is taken to end
    heights: dict                         # side -> mean guard height, or None
    punches: int = 0                      # detections in the response window

    @property
    def read(self) -> bool:
        return self.answered and any(v is not None for v in self.heights.values())


def align(calls: list[Call], track, punches: list[tuple[float, str]]) -> list[Rep]:
    """Each call, whether it was answered, and where the hands sat afterwards.

    Answered means the punch reader fired at all in the response window. Not
    which punch, or how many: at 47% recall a three-punch combo is caught
    whole a tenth of the time, but something in it is caught 85% of the time.

    The response is taken to end at the later of the last detection in the
    window and when this person's last punch of this combo usually lands
    (`pace`). The later, because a reader that missed the last punch would
    otherwise end the response early, and the guard window would then sit on
    top of that punch: a straight is thrown below a guard held at the chin,
    so it reads as a dropped hand. The guard read itself is `sag.GuardTrack.after`, exactly as
    the drill reads it, frames of an arm that is itself punching left out.
    """
    ordered = sorted(punches)
    times = np.asarray([p for p, _ in ordered], dtype=np.float64)
    snap = track.snapshot()
    seen = []
    for call in calls:
        lo, hi = window(call.at, call.combo)
        a, b = np.searchsorted(times, lo), np.searchsorted(times, hi, side="right")
        seen.append((int(b - a), float(times[b - 1]) if b > a else None))
    usual = pace(calls, seen)
    reps = []
    for call, (count, last) in zip(calls, seen):
        if not count:
            reps.append(Rep(call, False, None, {}, 0))
            continue
        end = max(last, call.at + usual[call.combo])
        heights = {side: track.after(end, side, ordered, snap) for side in sag.WRISTS}
        reps.append(Rep(call, True, end, heights, count))
    return reps


# How many answered calls of a combo before its pace is learned rather than
# assumed, and which percentile of the last detection stands for the end.
PACE_MIN = 5
PACE_PERCENTILE = 75.0


def pace(calls: list[Call], seen) -> dict:
    """When, after the call, this person's last punch of each combo lands.

    Learned from the session, per combo, because assuming one pace for
    everyone was measured to invent habits. Simulated with a fighter slower
    than the assumed pace (first punch at 1.0 s rather than 0.7, then every
    0.45 s rather than 0.3), a fixed expected end put the guard window on top
    of the missed last punch of long combos more often than of short ones,
    and 11% of twelve-minute sessions with no habit at all reported one.

    The last detection is the last punch under half the time, because the
    reader misses the rest, so its median sits early; the 75th percentile sits
    at or just after the true end. The same figure is used for every call of
    a combo, habit or not, so it cannot favour one combo's reads over another.
    Until a combo has PACE_MIN answered calls, the fixed `expected_end` stands.
    """
    offsets: dict[Combo, list[float]] = {}
    for call, (count, last) in zip(calls, seen):
        if count:
            offsets.setdefault(call.combo, []).append(last - call.at)
    usual = {}
    for call in calls:
        mine = offsets.get(call.combo, [])
        usual[call.combo] = (float(np.percentile(mine, PACE_PERCENTILE))
                             if len(mine) >= PACE_MIN
                             else expected_end(0.0, call.combo))
    return usual


def occasions(reps: list[Rep]) -> list[sag.Occasion]:
    """Answered reps as the occasions `sag` tests, keyed by the called combo."""
    out = []
    for rep in reps:
        if not rep.answered:
            continue
        for side, height in rep.heights.items():
            if height is not None:
                out.append(sag.Occasion(rep.call.combo.context, side, height))
    return out


def read(reps: list[Rep], fdr: float = 0.05):
    """Which called combos leave a hand lower than the others do.

    `sag.read_occasions` unchanged: every combo and side goes into one
    Benjamini-Hochberg correction, each tested by a rank test against the
    other combos' heights. The one difference from the drill is the context,
    which is the combo that was called rather than the punches that were seen.
    """
    return sag.read_occasions(occasions(reps), fdr=fdr)


def rate(reps: list[Rep], context, then: str) -> tuple[int, int]:
    """(hand down, answered) for one combo and side, for the session log."""
    side = then.rsplit("_", 1)[1]
    mine = [o for o in occasions(reps) if o.context == tuple(context) and o.side == side]
    return sum(o.down for o in mine), len(mine)


def evidence(reps: list[Rep], record) -> list:
    """What one session puts on the profile.

    What the drill puts there: candidates read at sag.PROPOSAL_FDR, a
    shortlist and not a verdict. Plus one thing the drill cannot give. A combo
    the profile already has a hunch about, called again this session and not
    proposed again, is recorded with this session's plain counts, however
    unremarkable. The drill only ever records what looked strong, so a hunch
    is pooled from selected extremes and absence arrives as silence. Coach
    mode knows exactly how often it called each combo, so a follow-up that
    came to nothing is evidence against the hunch, and it is recorded as such.
    """
    proposed = read(reps, fdr=sag.PROPOSAL_FDR)
    have = {(f.context, f.then) for f in proposed}
    tracked = {(tuple(o.context), o.then) for o in record.observations if is_coach(o.context)}
    occ = occasions(reps)
    extra = []
    for context, then in sorted(tracked - have):
        side = then.rsplit("_", 1)[1]
        inside = [o for o in occ if o.context == context and o.side == side]
        rest = [o for o in occ if o.context != context and o.side == side]
        if not inside or not rest:
            continue
        hits = sum(o.down for o in inside)
        base = max(sum(o.down for o in rest) / len(rest), 1.0 / (len(rest) + 1))
        extra.append(Finding(context, then, len(inside), hits, hits / len(inside), base, 1.0))
    return list(proposed) + extra


def reminders(habits) -> dict[str, str]:
    """Combo name to the words appended to its call, from known coach habits.

    Both hands on one combo say "hands up"; one says which.
    """
    sides: dict[str, set[str]] = {}
    for habit in habits:
        if is_coach(habit.context) and habit.then in REMINDERS:
            sides.setdefault(combo_of(habit.context), set()).add(habit.then)
    return {name: ("hands up" if len(found) > 1 else REMINDERS[next(iter(found))])
            for name, found in sides.items()}


def with_reminders(calls: list[Call], remind: dict[str, str]) -> list[Call]:
    return [Call(c.at, c.combo, remind.get(c.combo.name, "")) for c in calls]


def describe(finding) -> str:
    """A coach finding in words: the call, then the hand."""
    side = finding.then.rsplit("_", 1)[1]
    return (f"after {combo_of(finding.context)}: {side} hand down "
            f"{finding.probability:.0%} of the time (other combos {finding.base:.0%}), "
            f"{finding.lift:.1f}x, {finding.hits}/{finding.support}")
