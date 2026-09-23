"""Does the habit read survive the sensors it actually has?

    python scripts/robustness.py

The punch reader was measured on real people (scripts/train_strike.py) and is
far from perfect. This asks the only question that matters about that: with
those exact error rates corrupting what it sees, can the drill still name a
planted guard habit, how many minutes of drilling does it take, and does a
fighter with no habit stay clean?

Two ways of reading the guard are compared, because the first one failed.

**Events**: a dropped guard is a detected event, mined as part of the stream.
At the measured rates (30% of drops caught, 4.5 false drops a minute) it
could not find a habit that was planted 80% of the time.

**Sag** (`ika/sag.py`): where the hand sits, on average, in the second after
each combo, compared across setups. The guard height noise here is measured
from the labelled footage: an occasion's average sits 0.34 torso lengths
below the guard in a drop and 0.08 when up, with a spread of 0.3.

A fighter throws about 70 punches a minute, which is what the labelled
footage shows, in combos of one to four.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ika import sag  # noqa: E402
from ika.profile import Profile  # noqa: E402
from ika.tell.habits import mine  # noqa: E402

PUNCHES_PER_MINUTE = 70
UP, DOWN, SPREAD = -0.08, -0.34, 0.30

# Held out, every person on the bench: scripts/train_strike.py. Drops: scripts/shadow_guard.py.
MEASURED = {"punch_recall": 0.47, "false_punches_per_minute": 5.6,
            "drop_recall": 0.30, "false_drops_per_minute": 4.5}
PERFECT = {"punch_recall": 1.0, "false_punches_per_minute": 0.0,
           "drop_recall": 1.0, "false_drops_per_minute": 0.0}


def combos(rng, minutes: float, habit: float):
    """True combos, each with whether the right hand dropped after it. The
    habit: after two rights to finish, the right hand drops at `habit`."""
    out, thrown = [], 0
    while thrown < minutes * PUNCHES_PER_MINUTE:
        combo = [str(rng.choice(["punch_left", "punch_right"])) for _ in range(rng.integers(1, 5))]
        thrown += len(combo)
        doubled = len(combo) >= 2 and combo[-1] == combo[-2] == "punch_right"
        dropped = (rng.random() < habit) if doubled else (rng.random() < 0.08)
        out.append((combo, dropped))
    return out


def sensed(rng, combos_, minutes: float, rates):
    """What the punch reader reports, combo by combo: misses at the measured
    recall, and false punches at the measured rate, some joining a combo and
    some arriving alone in a gap, where they end a combo of their own."""
    per_combo = rates["false_punches_per_minute"] * minutes / max(len(combos_), 1)
    out = []
    for combo, dropped in combos_:
        seen = [p for p in combo if rng.random() < rates["punch_recall"]]
        for _ in range(rng.poisson(per_combo)):
            fake = str(rng.choice(["punch_left", "punch_right"]))
            if rng.random() < 0.5:
                seen.insert(int(rng.integers(0, len(seen) + 1)), fake)
            else:
                out.append(([fake], False))     # alone, in a gap
        if seen:
            out.append((seen, dropped))
    return out


def sag_occasions(rng, seen) -> list:
    occ = []
    for combo, dropped in seen:
        for k in (1, 2):
            if len(combo) < k:
                continue
            context = tuple(combo[-k:])
            right = rng.normal(DOWN if dropped else UP, SPREAD)
            left = rng.normal(UP, SPREAD)
            occ += [sag.Occasion(context, "right", right), sag.Occasion(context, "left", left)]
    return occ


def sag_read(rng, seen) -> list:
    return sag.read_occasions(sag_occasions(rng, seen))


def pooled(habit: float, sessions: int, minutes: float = 6.0, trials: int = 40,
           seed: int = 1) -> float:
    """Several sessions into one profile, the way `ika drill` records them:
    each session proposes at sag.PROPOSAL_FDR, the profile decides."""
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(trials):
        profile = Profile("sim")
        for day in range(1, sessions + 1):
            seen = sensed(rng, combos(rng, minutes, habit), minutes, MEASURED)
            profile.add(sag.read_occasions(sag_occasions(rng, seen), fdr=sag.PROPOSAL_FDR),
                        at=day * 86400.0)
        guard = [h for h in profile.habits(now=sessions * 86400.0)
                 if h.then.startswith("guard_down")]
        hits += named(guard) if habit else bool(guard)
    return hits / trials


def event_read(rng, combos_, minutes, rates) -> list:
    stream = []
    for combo, dropped in combos_:
        stream += [p for p in combo if rng.random() < rates["punch_recall"]]
        if dropped and rng.random() < rates["drop_recall"]:
            stream.append("guard_down_right")
    for _ in range(rng.poisson(rates["false_drops_per_minute"] * minutes)):
        stream.insert(int(rng.integers(0, len(stream) + 1)), "guard_down_right")
    return mine(stream, fdr=0.05)


def named(findings) -> bool:
    return any(f.then == "guard_down_right" and f.context[-1:] == ("punch_right",)
               for f in findings)


def trial_rate(read, minutes, habit, rates, trials=60, seed=0) -> float:
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(trials):
        c = combos(rng, minutes, habit)
        if read == "sag":
            found = sag_read(rng, sensed(rng, c, minutes, rates))
        else:
            found = event_read(rng, c, minutes, rates)
        guard = [f for f in found if f.then.startswith("guard_down")]
        hits += named(guard) if habit else bool(guard)
    return hits / trials


def main() -> int:
    print("a planted habit (right hand drops after two rights), how often it is named:")
    print(f"  {'minutes':>7s} {'habit':>6s} {'events, perfect':>16s} {'events, measured':>17s} "
          f"{'sag, perfect':>13s} {'sag, measured':>14s}")
    for minutes in (3, 6, 12):
        for habit in (0.5, 0.8):
            row = [trial_rate(r, minutes, habit, rates)
                   for r, rates in (("events", PERFECT), ("events", MEASURED),
                                    ("sag", PERFECT), ("sag", MEASURED))]
            print(f"  {minutes:7d} {habit:6.0%} " + " ".join(
                f"{v:>{w}.0%}" for v, w in zip(row, (16, 17, 13, 14))))
    print("\na fighter with no guard habit, how often one is reported anyway:")
    for minutes in (3, 12):
        row = [trial_rate(r, minutes, 0.0, rates)
               for r, rates in (("events", MEASURED), ("sag", MEASURED))]
        print(f"  {minutes:7d} min   events {row[0]:4.0%}   sag {row[1]:4.0%}")
    print("\nacross sessions, the way ika drill records them (six minutes a day, measured sensors):")
    for habit in (0.8, 0.5, 0.0):
        cells = "  ".join(f"{n} sessions {pooled(habit, n):4.0%}" for n in (4, 8, 12))
        label = f"{habit:.0%} habit named" if habit else "no habit, told one"
        print(f"  {label:18s} {cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
