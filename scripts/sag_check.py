"""The guard-sag read on real footage, and a null that keeps the real signal.

    python scripts/sag_check.py

For each clip: replay the pose, read punches with the shipped detector, read
where each hand sat after every combo, and report what the read finds. Nobody
knows these people's habits, so the findings cannot be checked directly. What
can be checked is the false-discovery side on real signal: shuffle which setup
each occasion belongs to, which keeps every real guard height and every real
punch but destroys any link between them, and count what is still "found".
"""
from __future__ import annotations

import random
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ika import sag  # noqa: E402
from ika.shadow import BENCH, Clip  # noqa: E402
from ika.strike import StrikeReader  # noqa: E402


def replay(clip: Clip):
    track = sag.GuardTrack(aspect=clip.aspect)
    reader = StrikeReader(aspect=clip.aspect)
    punches = []
    for at, _hands, body in clip.frames():
        track.update(body, at)
        punches += [(s.at, s.side) for s in reader.observe(body, at)]
    return track, punches


def shuffled_read(occ, rng):
    contexts = [o.context for o in occ]
    rng.shuffle(contexts)
    mixed = [replace(o, context=c) for o, c in zip(occ, contexts)]
    original = sag.occasions
    sag.occasions = lambda *_a, **_k: mixed
    try:
        return sag.read(None, [])
    finally:
        sag.occasions = original


def main() -> int:
    names = sorted(p.stem for p in (Path(__file__).resolve().parent.parent
                                    / "data" / "shadow" / "marks").glob("*.npz"))
    rng = random.Random(0)
    total_null, trials, heights = 0, 0, []
    for n in names:
        clip = Clip.load(n)
        track, punches = replay(clip)
        occ = sag.occasions(track, punches)
        heights += [o.height for o in occ if len(o.context) == 1]
        found = sag.read(track, punches)
        # Within each context length and side the shuffle must stay inside
        # its own pool, so shuffle each pool separately.
        null_hits = 0
        for _ in range(20):
            pools = {}
            for o in occ:
                pools.setdefault((len(o.context), o.side), []).append(o)
            mixed = []
            for pool in pools.values():
                ctx = [o.context for o in pool]
                rng.shuffle(ctx)
                mixed += [replace(o, context=c) for o, c in zip(pool, ctx)]
            original = sag.occasions
            sag.occasions = lambda *_a, _m=mixed, **_k: _m
            try:
                null_hits += bool(sag.read(track, punches))
            finally:
                sag.occasions = original
        total_null += null_hits
        trials += 20
        down = np.mean([o.down for o in occ]) if occ else float("nan")
        print(f"  {n:14s} {len(punches):4d} punches  {len(occ) // 2:4d} occasions  "
              f"hand down {down:4.0%}  findings {len(found)}  "
              f"shuffled: {null_hits}/20 report anything", flush=True)
        for f in found[:3]:
            print(f"      {f.describe()}")
    h = np.array(heights)
    print(f"\n  per-occasion guard height: mean {h.mean():+.3f}, sd {h.std():.3f}, "
          f"{(h < -sag.DOWN).mean():.0%} count as down")
    print(f"  shuffled null: {total_null}/{trials} = {total_null / trials:.0%} of shuffles "
          f"report a habit that cannot exist")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
