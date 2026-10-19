"""A known guard habit, planted in real footage, read by the real pipeline.

    python scripts/planted_check.py               # 10 seeds a condition, all cores
    python scripts/planted_check.py --seeds 4     # quicker, noisier

The habit read and the call had only been measured in simulation, because
nobody on the bench comes with a known habit. This plants one (`ika/plant.py`):
real guard drops two labellers marked, spliced into the second after a chosen
setup in other people's real rounds, at a chosen strength, triggered by the
labelled punches so the habit is really there whether or not the reader sees
the setup. Then the shipped pipeline reads the planted round exactly as `ika
drill` would: the punch reader (held out: fitted and thresholded without this
person), the guard track, the sag read, and a live caller.

Two habits:
  one punch   after punch_right, the LEFT hand sags (the guard that is not punching)
  two rights  after punch_right then punch_right, the RIGHT hand sags
              (the habit the simulation in scripts/robustness.py plants)

Controls, on the same rounds:
  none        nothing planted: what the read says about these people as they are
  shuffled    as many drops as the 80% habit, after random combo endings, so the
              drops are real and the same size but belong to no setup

Only the labelled stretches of each round are read, since only there are the
punches known and plants made. A round is 0.3 to 2.3 labelled minutes, so the
minutes axis is built from composite fighters: rounds of different people,
each with the same habit planted, read as one session. Repeating one person's
round instead would count the same unplanted movement several times over, and
the "none" row below shows what that does to the null.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))

from ika import plant, sag  # noqa: E402
from ika.cue import Caller  # noqa: E402
from ika.drill import DrillReader  # noqa: E402
from ika.shadow import (BENCH, ROOT, Clip, Punch, consensus, inside,  # noqa: E402
                        load_labels, person_of)
from ika.strike import ARMS, StrikeFeatures, StrikeModel, StrikeReader, _Decoder  # noqa: E402
from ika.tell.habits import Finding  # noqa: E402

CACHE = HERE / "data" / "planted"

HABITS = {
    "one punch": (("punch_right",), "left"),
    "two rights": (("punch_right", "punch_right"), "right"),
}
STRENGTHS = (0.8, 0.5, 0.3)
MINUTES = (3, 6, 12)
CALL_WINDOW = 1.5        # as cue.measure: the lapse within this long after the call


# --- the held-out punch readers -------------------------------------------------

def bench_names() -> list[str]:
    root = BENCH / "labels"
    return sorted({p.stem for p in (root / "A").glob("*.json")}
                  & {p.stem for p in (root / "B").glob("*.json")})


def lopo_models(names) -> dict[str, StrikeModel]:
    """One punch model per person, fitted and thresholded on everyone else,
    by the procedure in scripts/train_strike.py. Cached, since they take a
    few minutes and torch to rebuild."""
    out, missing = {}, []
    people = sorted({person_of(n) for n in names})
    for person in people:
        path = CACHE / "lopo" / f"{person}.npz"
        if path.exists():
            out[person] = StrikeModel.load(path)
        else:
            missing.append(person)
    import train_strike as ts

    if not missing:
        return out

    print(f"fitting {len(missing)} held-out punch models (cached in {CACHE / 'lopo'})", flush=True)
    clips = {n: Clip.load(n) for n in names}
    data = {n: ts.dataset(clips[n]) for n in names}
    feats = {n: _features(clips[n]) for n in names}
    for person in missing:
        train = [n for n in names if person_of(n) != person]
        x = np.concatenate([data[n][0] for n in train])
        y = np.concatenate([data[n][1] for n in train])
        w = np.concatenate([data[n][2] for n in train])
        model = ts.fit(x, y, w)
        labelled = [n for n in train if data[n][3].agreed]
        grid = (0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99)
        scored = {n: _scores(feats[n], model) for n in labelled}

        def total(th):
            s = None
            for n in labelled:
                one = ts.score(_decode(scored[n], th), data[n][3])
                s = one if s is None else s + one
            return s
        model.threshold = max(grid, key=lambda th: ts.objective(total(th)))
        path = CACHE / "lopo" / f"{person}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(path)
        out[person] = model
        print(f"  {person:20s} threshold {model.threshold}", flush=True)
    return out


def held_out(names, models) -> None:
    """The held-out score of the cached models, which should be the README's."""
    import train_strike as ts

    total = None
    for n in names:
        clip = Clip.load(n)
        truth = consensus(load_labels(clip, "A"), load_labels(clip, "B"))
        if not truth.agreed:
            continue
        m = models[person_of(n)]
        s = ts.score(_decode(_scores(_features(clip), m), m.threshold), truth)
        total = s if total is None else total + s
    print(f"held-out punch reader: recall {total.recall:.0%}, "
          f"{total.false_per_minute:.1f} false a minute (README: 47%, 5.6)")


def _features(clip):
    """Every frame's feature rows, which do not depend on the model."""
    f = StrikeFeatures(aspect=clip.aspect)
    rows = []
    for at, _hands, body in clip.frames():
        r = f.update(body, at)
        if r is not None:
            rows.append((at, list(r), np.stack([r[s] for s in r])))
    return rows


def _scores(rows, model):
    return [(at, sides, model(x)) for at, sides, x in rows]


def _decode(scored, threshold):
    """The same calls StrikeReader.observe makes, from precomputed scores."""
    d = _Decoder(threshold)
    out = []
    for at, sides, values in scored:
        for side, v in zip(sides, values):
            hit = d.step(side, float(v), at)
            if hit:
                out.append(Punch(hit.at, hit.side))
    return out


# --- the bench, as truth to plant into ------------------------------------------

def guard_drops(clip):
    """Drops both guard labellers marked, overlapping, on the same arm."""
    from shadow_guard import drops
    da, _, _ = drops(clip, "A")
    db, _, _ = drops(clip, "B")
    return [(s, max(a1, a2), min(b1, b2)) for s, a1, b1 in da for t, a2, b2 in db
            if s == t and min(b1, b2) > max(a1, a2)]


def library() -> list[plant.Drop]:
    out = []
    for path in sorted((BENCH / "guard" / "A").glob("*.json")):
        clip = Clip.load(path.stem)
        labelled = load_labels(clip, "A").punches + load_labels(clip, "B").punches
        for side, a, b in guard_drops(clip):
            d = plant.cut(clip, side, a, b, labelled)
            if d is not None:
                out.append(d)
    return out


def truth_of(clip):
    a, b = load_labels(clip, "A"), load_labels(clip, "B")
    return consensus(a, b), a.punches + b.punches


# --- one planted round, through the real pipeline ------------------------------

@dataclass
class Job:
    clip: str
    habit: str            # a HABITS key
    condition: str        # "0.8", "0.5", "0.3", "shuffled", "none"
    seed: int
    depth: float = 1.0    # the drops scaled by this; 1 is as the labellers' drops were
    perfect: bool = False  # read the labelled punches instead of the reader's
    reference: str = sag.REFERENCE   # what guard height is measured from


def _scaled(lib, k):
    from dataclasses import replace
    return [replace(d, image=d.image * k, world=d.world * k, depth=d.depth * k) for d in lib]


def _endings(punches):
    """(time, combo) for each punch that ends a combo, as sag.occasions sees it."""
    ordered = sorted(punches)
    out = []
    for i, (at, _s) in enumerate(ordered):
        if i + 1 < len(ordered) and ordered[i + 1][0] - at < sag.WINDOW[1]:
            continue
        out.append((at, tuple(sag.combo_of(ordered, i))))
    return out


_LIB = None


def run(job: Job) -> dict:
    global _LIB
    if _LIB is None:
        _LIB = library()
    lib = _LIB if job.depth == 1.0 else _scaled(_LIB, job.depth)
    clip = Clip.load(job.clip)
    truth, labelled = truth_of(clip)
    setup, side = HABITS[job.habit]
    plants, eligible, occurring = [], 0, len(plant.endings(truth, setup))
    if job.condition == "shuffled":
        _, eligible, _ = plant.plan(clip, truth, labelled, lib, side, setup, 0.8, job.seed)
        plants, _, _ = plant.plan(clip, truth, labelled, lib, side, None, 0.0, job.seed + 7919,
                                  count=int(round(0.8 * eligible)))
    elif job.condition != "none":
        plants, eligible, _ = plant.plan(clip, truth, labelled, lib, side, setup,
                                         float(job.condition), job.seed)
    planted = plant.apply(clip, plants, lib) if plants else clip
    if job.perfect:
        # The punch reader replaced by the labels: what the guard read alone does.
        track = sag.GuardTrack(aspect=clip.aspect, reference=job.reference)
        for at, _hands, body in planted.frames():
            track.update(body, at)
        punches = [(p.at, p.side) for p in truth.agreed if p.side in ("left", "right")]
        occ = sag.occasions(track, punches)
        return {"clip": job.clip, "habit": job.habit, "condition": job.condition,
                "seed": job.seed, "minutes": sum(b - a for a, b in plant._spans(truth.windows)) / 60,
                "occasions": [(o.context, o.side, o.height) for o in occ]}

    model = StrikeModel.load(CACHE / "lopo" / f"{person_of(job.clip)}.npz")
    reader = DrillReader(framing="stand", aspect=clip.aspect)
    reader._strike = StrikeReader(model=model, aspect=clip.aspect)
    reader.guard_track = sag.GuardTrack(aspect=clip.aspect, reference=job.reference)
    caller = Caller([Finding(setup, f"guard_down_{side}", 10, 8, 0.8, 0.1, 0.0)])
    spans = plant._spans(truth.windows)
    for at, hands, body in planted.frames():
        for _event in reader.observe(hands, body, at):
            caller.update(sag.current_combo(reader.punches()), at)

    punches = [(a, s) for a, s in reader.punches() if inside(a, spans)]
    occ = sag.occasions(reader.guard_track, punches)
    snap = reader.guard_track.snapshot()
    # Each read combo ending, with the heights after it, so the effect of a
    # plant can be measured on exactly the occasions it landed in.
    ends = []
    for at, combo in _endings(punches):
        h = reader.guard_track.after(at, side, sorted(punches), snap)
        hit = next((p for p in plants if abs(p.after - at) <= 0.25), None)
        ends.append((at, combo, h, hit is not None))
    downs = [p.down() for p in plants]
    calls = [c.at for c in caller.cues if inside(c.at, spans)]
    leads = [min((d - c for d in downs if 0 <= d - c <= CALL_WINDOW), default=None)
             for c in calls]
    truth_setups = [last.at for last, _ in plant.endings(truth, setup)]
    caught = sum(any(abs(at - t) <= plant.CLEAR and combo[-len(setup):] == setup
                     for at, combo, _h, _p in ends) for t in truth_setups)
    return {
        "clip": job.clip, "habit": job.habit, "condition": job.condition, "seed": job.seed,
        "minutes": sum(b - a for a, b in spans) / 60.0,
        "occurring": occurring, "eligible": eligible, "planted": len(plants),
        "caught": caught,
        "occasions": [(o.context, o.side, o.height) for o in occ],
        "ends": [(combo[-2:], h, p) for _at, combo, h, p in ends],
        "calls": len(calls), "leads": [x for x in leads if x is not None],
        "covered": sum(any(0 <= d - c <= CALL_WINDOW for c in calls) for d in downs),
        "plant_spans": [(p.start, p.start + p.length) for p in plants],
        "punches": punches,
    }


# --- reading -------------------------------------------------------------------

def occasions_of(result) -> list[sag.Occasion]:
    return [sag.Occasion(tuple(c), s, h) for c, s, h in result["occasions"]]


def named(findings, habit) -> bool:
    """Named the way scripts/robustness.py counts it: a guard habit on the
    planted hand whose setup ends in the setup's last punch."""
    setup, side = HABITS[habit]
    return any(f.then == f"guard_down_{side}" and tuple(f.context)[-1:] == setup[-1:]
               for f in findings)


def any_guard(findings) -> bool:
    return any(f.then.startswith("guard_down") for f in findings)


def composite(results, minutes, rng) -> list:
    """Rounds of different people, in a random order, until there are enough
    labelled minutes. One round per person, so no movement is counted twice."""
    by_person = defaultdict(list)
    for r in results:
        by_person[person_of(r["clip"])].append(r)
    order = rng.permutation(sorted(by_person))
    got, occ = 0.0, []
    for person in order:
        seeds = sorted({r["seed"] for r in by_person[person]})
        seed = seeds[int(rng.integers(len(seeds)))]
        for r in by_person[person]:
            if r["seed"] == seed:
                occ += occasions_of(r)
                got += r["minutes"]
        if got >= minutes:
            return occ
    return []          # not enough distinct footage


# --- the simulation, for the same habits ---------------------------------------

def simulated(habit: str, strength: float, minutes: float, trials: int = 200,
              seed: int = 0) -> float:
    """scripts/robustness.py at the measured sensor rates, generalised to
    either habit, read by the same sag read. Returns how often it is named
    (or, at strength 0, how often any guard habit is reported)."""
    import robustness as rb

    setup, side = HABITS[habit]
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(trials):
        combos, thrown = [], 0
        while thrown < minutes * rb.PUNCHES_PER_MINUTE:
            combo = [str(rng.choice(["punch_left", "punch_right"]))
                     for _ in range(rng.integers(1, 5))]
            thrown += len(combo)
            hit = tuple(combo[-len(setup):]) == setup
            combos.append((combo, (rng.random() < strength) if hit else (rng.random() < 0.08)))
        seen = rb.sensed(rng, combos, minutes, rb.MEASURED)
        occ = []
        for combo, dropped in seen:
            for k in (1, 2):
                if len(combo) < k:
                    continue
                ctx = tuple(combo[-k:])
                for s in ("left", "right"):
                    mean = rb.DOWN if (dropped and s == side) else rb.UP
                    occ.append(sag.Occasion(ctx, s, rng.normal(mean, rb.SPREAD)))
        found = sag.read_occasions(occ)
        hits += named(found, habit) if strength else any_guard(found)
    return hits / trials


# --- sanity: is the plant findable for the wrong reason? -----------------------

def sanity(results, lib) -> None:
    """Two checks that the plant is not a giveaway.

    Steps: a splice that jumped would show as frame-to-frame wrist steps
    bigger than the person's own. Punches: a plant that the punch reader
    fired on would change the setups it is supposed to sit after.
    """
    print("\nsanity: is a plant detectable for the wrong reason?")
    steps_in, steps_out, n_frames = [], [], 0
    punch_in, punch_out, t_in, t_out = 0, 0, 0.0, 0.0
    seen = set()
    for r in results:
        if r["condition"] != "0.8" or not r["plant_spans"] or (r["clip"], r["habit"]) in seen:
            continue
        seen.add((r["clip"], r["habit"]))
        clip = Clip.load(r["clip"])
        truth, labelled = truth_of(clip)
        setup, side = HABITS[r["habit"]]
        plants, _, _ = plant.plan(clip, truth, labelled, lib, side, setup, 0.8, r["seed"])
        planted = plant.apply(clip, plants, lib)
        wrist = 15 if side == "left" else 16
        xy = planted.body_image[:, wrist, :2].astype(np.float64)
        xy[:, 1] /= clip.aspect
        step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        mid = (clip.t[1:] + clip.t[:-1]) / 2
        on = np.zeros(len(mid), bool)
        for a, b in r["plant_spans"]:
            on |= (mid >= a) & (mid <= b)
        ok = np.isfinite(step)
        steps_in.append(step[on & ok])
        steps_out.append(step[~on & ok])
        n_frames += int((on & ok).sum())
        spans = plant._spans(truth.windows)
        mine = [a for a, s in r["punches"] if s == side]
        dur_in = sum(b - a for a, b in r["plant_spans"])
        punch_in += sum(any(a <= p <= b for a, b in r["plant_spans"]) for p in mine)
        punch_out += sum(not any(a <= p <= b for a, b in r["plant_spans"]) for p in mine)
        t_in += dur_in
        t_out += sum(b - a for a, b in spans) - dur_in
    sin, sout = np.concatenate(steps_in), np.concatenate(steps_out)
    p99 = np.percentile(sout, 99)
    print(f"  wrist step per frame: inside plants median {np.median(sin):.4f}, "
          f"outside {np.median(sout):.4f}; above the outside 99th percentile: "
          f"{(sin > p99).mean():.1%} of {n_frames} planted frames (1% would be no change)")
    print(f"  the reader's punches on the planted arm: {punch_in / max(t_in / 60, 1e-9):.1f} a minute "
          f"inside plants, {punch_out / max(t_out / 60, 1e-9):.1f} outside")


# --- main ----------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--trials", type=int, default=300, help="composites per cell")
    parser.add_argument("--reference", default=sag.REFERENCE, choices=sag.REFERENCES,
                        help="guard height measured from the shoulders (shipped) or the nose (old)")
    parser.add_argument("--rule", default=sag.RULE, choices=sag.RULES,
                        help="the read's size bar: lift (shipped) or effect (measured, not adopted)")
    parser.add_argument("--heights", action="store_true",
                        help=f"write the measured guard heights to {HEIGHTS.name} and stop")
    parser.add_argument("--min-effect", type=float, default=sag.MIN_EFFECT, dest="min_effect")
    args = parser.parse_args()
    sag.RULE, sag.MIN_EFFECT = args.rule, args.min_effect
    print(f"guard reference {args.reference}, decision rule {args.rule}"
          + (f" {args.min_effect}" if args.rule == "effect" else ""))

    names = [n for n in bench_names()]
    lib = library()
    held_out(names, lopo_models(names))
    # A round can only carry a plant where punches are labelled.
    names = [n for n in names if truth_of(Clip.load(n))[0].agreed]
    people = sorted({person_of(n) for n in names})
    print(f"{len(lib)} real drops to transplant, from "
          f"{len({d.donor for d in lib})} people ({', '.join(sorted({d.donor for d in lib}))}); "
          f"down {min(d.down for d in lib):.1f} to {max(d.down for d in lib):.1f} s, "
          f"depth {np.median([d.depth for d in lib]):+.2f} torso lengths (median)")
    print(f"{len(people)} people in {len(names)} rounds, "
          f"{sum(sum(b - a for a, b in truth_of(Clip.load(n))[0].windows) for n in names) / 60:.1f} "
          f"labelled minutes")

    cache = CACHE / f"runs_{args.seeds}_{args.reference}.json"
    if cache.exists():
        results = json.loads(cache.read_text())
    else:
        ref = args.reference
        jobs = [Job(n, "one punch", "none", 0, reference=ref) for n in names]
        for habit in HABITS:
            for cond in (*[str(s) for s in STRENGTHS], "shuffled"):
                jobs += [Job(n, habit, cond, s, reference=ref)
                         for n in names for s in range(args.seeds)]
        print(f"replaying {len(jobs)} rounds on {args.workers} workers", flush=True)
        with ProcessPoolExecutor(args.workers) as pool:
            results = list(pool.map(run, jobs, chunksize=2))
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(results))
    for r in results:
        r["occasions"] = [(tuple(c), s, h) for c, s, h in r["occasions"]]

    if args.heights:
        export_heights(results, args.reference)
        return 0
    report(results, names, people, args.trials)
    sanity(results, lib)
    where_lost(names, args)
    return 0


HEIGHTS = HERE / "bench" / "shadow" / "guard_heights.json"


def export_heights(results, reference) -> None:
    """The guard heights the 80% one-punch habit left, as quantiles, for
    scripts/coach_check.py to draw from instead of assuming a normal shape.

    "planted": the window after a read combo ending a plant landed in;
    "rest": after every other read combo ending. Left hand, all seeds, the
    real punch reader. Quantiles rather than raw heights, so the file is
    small and holds no landmarks."""
    per = defaultdict(lambda: ([], []))
    for r in results:
        if r["habit"] != "one punch" or r["condition"] != "0.8":
            continue
        for _c, h, hit in r["ends"]:
            if h is not None:
                per[person_of(r["clip"])][0 if hit else 1].append(h)
    # Each person's heights centred on their own median after other combos.
    # A session is one person, so how high one person's guard sits against
    # another's is not part of what the read sees; left in, it is spread the
    # rank test never meets. Centred, the effect is 0.70 (shoulders) where
    # pooling people raw gives 0.72 and pairing only within a person 0.78:
    # the lowest of the three, so the simulation errs low.
    # Centred on the pooled median rather than zero, so a height still sits
    # where it does against sag.DOWN, which the "lift" rule counts against.
    centre = float(np.median([h for _p, rest in per.values() for h in rest]))
    planted_h, other_h = [], []
    for planted, rest in per.values():
        if planted and rest:
            mid = float(np.median(rest)) - centre
            planted_h += [h - mid for h in planted]
            other_h += [h - mid for h in rest]
    q = np.linspace(0, 100, 201)
    data = json.loads(HEIGHTS.read_text()) if HEIGHTS.exists() else {}
    data["about"] = ("Guard heights in the second after a combo on the planted bench "
                     "(scripts/planted_check.py --heights): 80% 'after a right, the left "
                     "sags', read with the held-out punch reader. Each person's own median after other "
                     "combos moved to the pooled one. Quantiles, torso lengths.")
    data[reference] = {
        "planted": [round(float(v), 4) for v in np.percentile(planted_h, q)],
        "rest": [round(float(v), 4) for v in np.percentile(other_h, q)],
        "n_planted": len(planted_h), "n_rest": len(other_h),
        "effect": round(sag.effect(planted_h, other_h), 3),
    }
    HEIGHTS.write_text(json.dumps(data, indent=1) + "\n")
    print(f"wrote {reference} heights to {HEIGHTS}: {len(planted_h)} planted, "
          f"{len(other_h)} rest, effect {data[reference]['effect']}")


def where_lost(names, args) -> None:
    """The same 80% one-punch habit with each weakness taken away in turn:
    the punch reader replaced by the labels, and the planted drops made twice
    as deep as the labelled ones. A diagnosis, not a claim about anyone."""
    cache = CACHE / f"lost_{args.seeds}_{args.reference}.json"
    variants = {"reader, drops as labelled": (1.0, False),
                "reader, drops twice as deep": (2.0, False),
                "labelled punches, drops as labelled": (1.0, True),
                "labelled punches, drops twice as deep": (2.0, True)}
    if cache.exists():
        got = json.loads(cache.read_text())
    else:
        jobs = {v: [Job(n, "one punch", "0.8", s, d, p, args.reference)
                    for n in names for s in range(args.seeds)]
                for v, (d, p) in variants.items() if v != "reader, drops as labelled"}
        got = {}
        with ProcessPoolExecutor(args.workers) as pool:
            for v, js in jobs.items():
                got[v] = list(pool.map(run, js, chunksize=2))
        cache.write_text(json.dumps(got))
    got["reader, drops as labelled"] = [
        r for r in json.loads((CACHE / f"runs_{args.seeds}_{args.reference}.json").read_text())
        if r["habit"] == "one punch" and r["condition"] == "0.8"]
    print("\nwhere it is lost: the 80% one-punch habit, composite fighters, named:")
    print(f"  {'':40s} " + " ".join(f"{m:>4d} min" for m in MINUTES))
    rng = np.random.default_rng(5)
    for v in variants:
        res = got[v]
        for r in res:
            r["occasions"] = [(tuple(c), s_, h) for c, s_, h in r["occasions"]]
        cells = []
        for minutes in MINUTES:
            hit = n = 0
            for _ in range(args.trials):
                occ = composite(res, minutes, rng)
                if occ:
                    n += 1
                    hit += named(sag.read_occasions(occ), "one punch")
            cells.append(f"{hit / max(n, 1):>8.0%}")
        print(f"  {v:40s} " + " ".join(cells))


def report(results, names, people, trials) -> None:
    none = {r["clip"]: r for r in results if r["condition"] == "none"}

    for habit, (setup, side) in HABITS.items():
        rows = [r for r in results if r["habit"] == habit]
        print(f"\n=== {habit}: after {' then '.join(setup)}, the {side} hand sags ===")
        print("one round per person, read on its own (named / seeds):")
        print(f"  {'person':20s} {'min':>4s} {'setups':>6s} {'plantable':>9s} "
              f"{'read':>5s} " + " ".join(f"{s:>5.0%}" for s in STRENGTHS) + "  shuffled  none")
        for person in people:
            mine = [r for r in rows if person_of(r["clip"]) == person]
            clips = sorted({r["clip"] for r in mine})
            # A person with two rounds: their occasions pooled per seed.
            def cell(cond):
                seeds = sorted({r["seed"] for r in mine if r["condition"] == cond})
                hit = 0
                for s in seeds:
                    occ = sum((occasions_of(r) for r in mine
                               if r["condition"] == cond and r["seed"] == s), [])
                    f = sag.read_occasions(occ)
                    hit += named(f, habit) if cond != "shuffled" else any_guard(f)
                return f"{hit:>2d}/{len(seeds):<2d}"
            first = [r for r in mine if r["condition"] == "0.8" and r["seed"] == 0]
            minutes = sum(r["minutes"] for r in first)
            occurring = sum(r["occurring"] for r in first)
            eligible = sum(r["eligible"] for r in first)
            caught = sum(r["caught"] for r in first)
            nocc = sum((occasions_of(none[c]) for c in clips), [])
            print(f"  {person:20s} {minutes:4.1f} {occurring:6d} {eligible:9d} {caught:5d} "
                  + " ".join(f"{cell(str(s)):>5s}" for s in STRENGTHS)
                  + f"  {cell('shuffled'):>8s}  {'yes' if any_guard(sag.read_occasions(nocc)) else 'no':>4s}")
        occ_all = sum(r["occurring"] for r in rows if r["condition"] == "0.8" and r["seed"] == 0)
        elig_all = sum(r["eligible"] for r in rows if r["condition"] == "0.8" and r["seed"] == 0)
        caught_all = sum(r["caught"] for r in rows if r["condition"] == "0.8" and r["seed"] == 0)
        print(f"  setups in the labels {occ_all}, plantable {elig_all} ({elig_all / max(occ_all, 1):.0%}), "
              f"read by the punch reader as this setup {caught_all} ({caught_all / max(occ_all, 1):.0%})")

        # The size of what was planted, on the occasions it landed in.
        planted_h, other_h = [], []
        for r in rows:
            if r["condition"] != "0.8":
                continue
            for _c, h, hit in r["ends"]:
                if h is not None:
                    (planted_h if hit else other_h).append(h)
        if planted_h:
            print(f"  guard height after a combo, {side} hand: planted {np.mean(planted_h):+.2f} "
                  f"(sd {np.std(planted_h):.2f}), the rest {np.mean(other_h):+.2f} "
                  f"(sd {np.std(other_h):.2f}); effect {sag.effect(planted_h, other_h):.2f}; "
                  f"the simulation assumes -0.34 against -0.08, sd 0.30")

        print("composite fighters, one round per person until the minutes are reached "
              f"({trials} composites a cell), named:")
        print(f"  {'minutes':>7s} " + " ".join(f"{'real ' + format(s, '.0%'):>9s} {'sim':>4s}"
                                                for s in STRENGTHS)
              + f"  {'shuffled':>8s} {'none':>5s} {'sim':>4s}")
        rng = np.random.default_rng(11)
        for minutes in MINUTES:
            cells = []
            for s in STRENGTHS:
                res = [r for r in rows if r["condition"] == str(s)]
                hit = n = 0
                for _ in range(trials):
                    occ = composite(res, minutes, rng)
                    if occ:
                        n += 1
                        hit += named(sag.read_occasions(occ), habit)
                cells.append(f"{hit / max(n, 1):>9.0%} {simulated(habit, s, minutes):>4.0%}")
            res = [r for r in rows if r["condition"] == "shuffled"]
            sh = n = 0
            for _ in range(trials):
                occ = composite(res, minutes, rng)
                if occ:
                    n += 1
                    sh += any_guard(sag.read_occasions(occ))
            nres = list(none.values())
            nh = nn = 0
            for _ in range(trials):
                occ = composite(nres, minutes, rng)
                if occ:
                    nn += 1
                    nh += any_guard(sag.read_occasions(occ))
            print(f"  {minutes:7d} " + " ".join(cells)
                  + f"  {sh / max(n, 1):>8.0%} {nh / max(nn, 1):>5.0%} "
                  f"{simulated(habit, 0.0, minutes):>4.0%}")

        print("the call, armed with the planted habit, live as the round plays:")
        for s in STRENGTHS:
            res = [r for r in rows if r["condition"] == str(s)]
            calls = sum(r["calls"] for r in res)
            right = sum(len(r["leads"]) for r in res)
            leads = [x for r in res for x in r["leads"]]
            plants = sum(r["planted"] for r in res)
            covered = sum(r["covered"] for r in res)
            print(f"  {s:4.0%} habit: {calls} calls, {right / max(calls, 1):.0%} followed by a "
                  f"planted drop, median {np.median(leads) if leads else float('nan'):.2f} s "
                  f"before the hand was down; {covered / max(plants, 1):.0%} of {plants} "
                  f"planted drops had a call before them")

    print("\nrepeating one round instead of composing: nothing planted, the same "
          "unplanted round read k times as one session, any guard habit reported:")
    for k in (1, 2, 4, 8):
        bad = 0
        for person in people:
            occ = sum((occasions_of(r) for r in none.values() if person_of(r["clip"]) == person), [])
            bad += any_guard(sag.read_occasions(occ * k))
        print(f"  x{k}: {bad}/{len(people)} people")


if __name__ == "__main__":
    raise SystemExit(main())
