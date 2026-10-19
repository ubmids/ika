"""Every null the guard read's decision rule has to pass, in one place.

    python scripts/guard_nulls.py                  # the shipped rule and the old one
    python scripts/guard_nulls.py --quick          # fewer trials

The guard read (`ika/sag.py`) names a habit when a rank test, corrected for
every setup and side it looked at, is sure a setup's heights sit lower than
the other combos', AND the difference is big enough to matter. "Big enough"
was a lift bar on counts of occasions below a fixed line (1.35x); it is now
an effect size on the heights themselves (`sag.effect`, the share of pairs
where the setup's height is lower, at least sag.MIN_EFFECT). This runs every
null either rule has to pass, side by side, so loosening anything is only
ever done with its null measured:

  drill, simulated clean fighter    scripts/robustness.py's fighter, no habit, 12 min
  drill, 12 pooled sessions         the same into one profile
  drill, shuffled real sessions     scripts/sag_check.py's shuffle, 15 bench clips
  coach, simulated clean fighter    scripts/coach_check.py, 3 and 12 min
  coach, slower clean fighter       the same, first punch at 1.0 s then every 0.45 s
  coach, 12 pooled sessions         coach.plan curriculum into one profile
  coach, random schedules on bench  calls never made, over the 15 bench clips

The planted bench's own nulls (shuffled drops, nothing planted) are in
scripts/planted_check.py, which takes the same --rule.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))

from ika import coach, sag  # noqa: E402

RULES = {"lift (old)": ("lift", sag.MIN_EFFECT), "effect 0.6": ("effect", 0.6)}


def use(rule):
    sag.RULE, sag.MIN_EFFECT = rule


def drill_sim(trials):
    import robustness as rb
    return rb.trial_rate("sag", 12, 0.0, rb.MEASURED, trials=trials, seed=3)


def drill_pooled(trials):
    import robustness as rb
    return rb.pooled(0.0, 12, trials=trials, seed=4)


_REPLAYS = {}


def replays(reference):
    """Every bench clip replayed once: its guard track and its punches."""
    if reference not in _REPLAYS:
        from ika.shadow import Clip
        from ika.strike import StrikeReader
        names = sorted(p.stem for p in (HERE / "data" / "shadow" / "marks").glob("*.npz"))
        out = []
        for name in names:
            clip = Clip.load(name)
            track = sag.GuardTrack(aspect=clip.aspect, reference=reference)
            reader = StrikeReader(aspect=clip.aspect)
            punches = []
            for at, _hands, body in clip.frames():
                track.update(body, at)
                punches += [(s.at, s.side) for s in reader.observe(body, at)]
            out.append((name, track, punches, float(clip.t[-1])))
        _REPLAYS[reference] = out
    return _REPLAYS[reference]


def drill_shuffled(reference, shuffles):
    rng = random.Random(0)
    hits = trials = 0
    for _name, track, punches, _s in replays(reference):
        occ = sag.occasions(track, punches)
        for _ in range(shuffles):
            pools = {}
            for o in occ:
                pools.setdefault((len(o.context), o.side), []).append(o)
            mixed = []
            for pool in pools.values():
                ctx = [o.context for o in pool]
                rng.shuffle(ctx)
                mixed += [replace(o, context=c) for o, c in zip(pool, ctx)]
            hits += bool(sag.read_occasions(mixed))
            trials += 1
    return hits / trials


def coach_bench(reference, schedules):
    hits = trials = 0
    for _name, track, punches, seconds in replays(reference):
        for k in range(schedules):
            combos = coach.choose(seed=10_000 + k)
            calls = coach.schedule(combos, seconds - 7.0, seed=k, start=5.0)
            hits += bool(coach.read(coach.align(calls, track, punches)))
            trials += 1
    return hits / trials


def main() -> int:
    import coach_check as cc
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--reference", default=sag.REFERENCE, choices=sag.REFERENCES)
    args = ap.parse_args()
    n = 100 if args.quick else 400
    npool = 30 if args.quick else 100
    print(f"guard reference: {args.reference}; a habit reported where none exists, should be "
          "about 5% or under")
    rows, shipped = {}, (sag.RULE, sag.MIN_EFFECT)
    for label, rule in RULES.items():
        use(rule)
        r = {}
        r["drill, simulated clean fighter, 12 min"] = drill_sim(n)
        r["drill, 12 pooled six-minute sessions"] = drill_pooled(npool)
        r["drill, shuffled real sessions"] = drill_shuffled(args.reference, 20 if args.quick else 40)
        for minutes in (3, 12):
            r[f"coach, simulated clean fighter, {minutes} min"] = cc.single(
                minutes, 0.0, cc.MEASURED, n, guard="real")[0]
            r[f"coach, slower clean fighter, {minutes} min"] = cc.single(
                minutes, 0.0, cc.MEASURED, n, pace=(1.0, 0.45), guard="real")[0]
        r["coach, 12 pooled six-minute sessions"] = cc.pooled(0.0, 12, cc.MEASURED, npool,
                                                              guard="real")[0]
        r["coach, random schedules on the bench"] = coach_bench(args.reference,
                                                                 20 if args.quick else 40)
        rows[label] = r
        print(f"  done: {label}", flush=True)
    print(f"\n{'null':44s} " + " ".join(f"{k:>11s}" for k in RULES))
    for key in rows[next(iter(RULES))]:
        print(f"{key:44s} " + " ".join(f"{rows[k][key]:>11.1%}" for k in RULES))
    use(shipped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
