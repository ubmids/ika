"""Which guard signal separates a real sag from normal movement best?

    python scripts/guard_signal.py              # every candidate, three metrics
    python scripts/guard_signal.py --seeds 2    # quicker planted metric

The guard read (`ika/sag.py`) reads the wrist's height above the nose in
torso lengths, against the 75th percentile of this person's last twenty
seconds, averaged over the second after a combo. This measures alternatives
to that per-frame signal on three things, always per person:

  frames     frames inside a drop both guard labellers marked, against frames
             in guard (inside a labelled stretch of shadowboxing, outside any
             drop either labeller marked on that arm, and clear of that arm's
             punches by sag.PUNCHING). The primary metric: real drops.
  occasions  the read's own unit on the labelled footage: the window after a
             combo ending (labelled punches), with an agreed drop on that arm
             overlapping it, against windows no labeller marked a drop in.
  planted    the end-to-end bench's unit: an 80% habit planted from the
             labelled drops (`ika/plant.py`, "after a right ends a combo, the
             left sags"), the window after each planted ending against the
             window after every other ending, with labelled punches.

Effect size is Cohen's d (the drop side lower, so positive is right) and the
AUC that a drop window sits lower than a guard window. Both are unitless, so
signals in different units compare directly.

A caution the planted column needs: a plant moves only the elbow, the wrist
and the three hand points, in image x and y and in world x, y and z. It does
not move the nose, the mouth, the shoulders, or image z. So a feature built
on image z sees no plant at all, and a feature referenced to the shoulder sees
exactly what was moved, where a real drop may also move the head. The frames
and occasions columns are real drops and are the ones to believe.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))

from ika import plant, sag  # noqa: E402
from ika.body import (LEFT_ELBOW, LEFT_HIP, LEFT_INDEX, LEFT_PINKY,  # noqa: E402
                      LEFT_SHOULDER, LEFT_THUMB, LEFT_WRIST, MOUTH_LEFT, MOUTH_RIGHT,
                      NOSE, RIGHT_ELBOW, RIGHT_HIP, RIGHT_INDEX, RIGHT_PINKY,
                      RIGHT_SHOULDER, RIGHT_THUMB, RIGHT_WRIST)
from ika.shadow import BENCH, Clip, consensus, load_labels, person_of  # noqa: E402

J = {
    "left": dict(w=LEFT_WRIST, e=LEFT_ELBOW, s=LEFT_SHOULDER,
                 hand=(LEFT_WRIST, LEFT_PINKY, LEFT_INDEX, LEFT_THUMB)),
    "right": dict(w=RIGHT_WRIST, e=RIGHT_ELBOW, s=RIGHT_SHOULDER,
                  hand=(RIGHT_WRIST, RIGHT_PINKY, RIGHT_INDEX, RIGHT_THUMB)),
}
NEEDED = (NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)
SIDES = ("left", "right")


# --- per-frame raw signals, larger = hand higher ---------------------------------

def raw(clip) -> dict[str, dict[str, np.ndarray]]:
    """Every candidate's raw per-frame value per arm, NaN where unread.

    Read the way GuardTrack reads: the torso and nose seen (visibility 0.5),
    the wrist seen. Elbow candidates also need the elbow seen.
    """
    img = clip.body_image[:, :, :3].astype(np.float64).copy()
    img[:, :, 1] /= clip.aspect
    wld = clip.body_world.astype(np.float64)
    vis = clip.body_vis.astype(np.float64)
    vis = np.where(np.isfinite(vis), vis, 0.0)
    mid = lambda a, i, j: (a[:, i] + a[:, j]) / 2
    torso = np.linalg.norm(mid(img, LEFT_SHOULDER, RIGHT_SHOULDER)[:, :2]
                           - mid(img, LEFT_HIP, RIGHT_HIP)[:, :2], axis=1)
    wtorso = np.linalg.norm(mid(wld, LEFT_SHOULDER, RIGHT_SHOULDER)
                            - mid(wld, LEFT_HIP, RIGHT_HIP), axis=1)
    base_ok = (np.nan_to_num(torso) > 1e-6) & (vis[:, list(NEEDED)].min(1) >= 0.5)
    shy = mid(img, LEFT_SHOULDER, RIGHT_SHOULDER)[:, 1]
    mouth = mid(img, MOUTH_LEFT, MOUTH_RIGHT)
    out: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    for side in SIDES:
        j = J[side]
        ok = base_ok & (vis[:, j["w"]] >= 0.5)
        oke = ok & (vis[:, j["e"]] >= 0.5)
        w, e, s, n = img[:, j["w"]], img[:, j["e"]], img[:, j["s"]], img[:, NOSE]
        up = lambda y, ref: -(y - ref) / torso
        f = {
            "wrist-nose": (up(w[:, 1], n[:, 1]), ok),
            "wrist-mouth": (up(w[:, 1], mouth[:, 1]), ok),
            "wrist-shoulder": (up(w[:, 1], s[:, 1]), ok),
            "wrist-shoulders": (up(w[:, 1], shy), ok),
            "hand-nose": (up(img[:, list(j["hand"]), 1].mean(1), n[:, 1]), ok),
            "hand-shoulders": (up(img[:, list(j["hand"]), 1].mean(1), shy), ok),
            "elbow-shoulder": (up(e[:, 1], s[:, 1]), oke),
            "wrist+elbow": ((up(w[:, 1], n[:, 1]) + up(e[:, 1], s[:, 1])) / 2, oke),
            "face distance": (-np.linalg.norm(w[:, :2] - n[:, :2], axis=1) / torso, ok),
            "forearm angle": (-(w[:, 1] - e[:, 1])
                              / np.maximum(np.linalg.norm(w[:, :2] - e[:, :2], axis=1), 1e-6), oke),
            "world wrist-nose": (-(wld[:, j["w"], 1] - wld[:, NOSE, 1]) / wtorso, ok),
            "world wrist+elbow": ((-(wld[:, j["w"], 1] - wld[:, NOSE, 1])
                                   - (wld[:, j["e"], 1] - wld[:, j["s"], 1])) / 2 / wtorso, oke),
            "world face 3d": (-np.linalg.norm(wld[:, j["w"]] - wld[:, NOSE], axis=1) / wtorso, ok),
            "image z (unplanted)": (-(img[:, j["w"], 2] - img[:, j["s"], 2]), ok),
        }
        for name, (v, m) in f.items():
            v = np.where(m & np.isfinite(v), v, np.nan)
            out[name][side] = v
    return out


def rolling(t: np.ndarray, x: np.ndarray, kind: str, seconds: float = sag.BASELINE_SECONDS,
            need: int = 30) -> np.ndarray:
    """x against a causal rolling statistic of its own last `seconds`, as
    GuardTrack does it: the window includes the current frame, NaN until
    `need` frames have been read."""
    if kind == "none":
        return x.copy()
    out = np.full(len(x), np.nan)
    idx = np.flatnonzero(np.isfinite(x))
    ts, xs = t[idx], x[idx]
    starts = np.searchsorted(ts, ts - seconds, side="left")
    for k in range(len(idx)):
        a = starts[k]
        if k + 1 - a < need:
            continue
        win = xs[a:k + 1]
        if kind.startswith("p"):
            out[idx[k]] = xs[k] - np.percentile(win, float(kind[1:]))
        elif kind == "median":
            out[idx[k]] = xs[k] - np.median(win)
        elif kind == "robust z":
            med = np.median(win)
            mad = np.median(np.abs(win - med)) * 1.4826
            out[idx[k]] = (xs[k] - np.percentile(win, 75)) / max(mad, 1e-3)
    return out


def smooth(t, x, seconds):
    """Causal exponential smoothing over read frames, time constant `seconds`."""
    if seconds <= 0:
        return x
    out = x.copy()
    prev, pt = None, None
    for i in range(len(x)):
        if not np.isfinite(x[i]):
            continue
        if prev is None:
            prev = x[i]
        else:
            a = 1 - np.exp(-(t[i] - pt) / seconds)
            prev = prev + a * (x[i] - prev)
        pt = t[i]
        out[i] = prev
    return out


# --- the windows the read takes ---------------------------------------------------

def window(t, h, at, side, punches, stat="mean"):
    """GuardTrack.after on an array: frames in sag.WINDOW after `at`, that
    arm's own punches excluded, at least sag.MIN_FRAMES of them."""
    lo, hi = at + sag.WINDOW[0], at + sag.WINDOW[1]
    a, b = np.searchsorted(t, lo), np.searchsorted(t, hi, side="right")
    times, values = t[a:b], h[a:b]
    keep = np.isfinite(values)
    for p, s in punches:
        if s == side and lo - sag.PUNCHING <= p <= hi + sag.PUNCHING:
            keep &= np.abs(times - p) > sag.PUNCHING
    if keep.sum() < sag.MIN_FRAMES:
        return None
    v = values[keep]
    if stat == "mean":
        return float(v.mean())
    if stat == "p25":
        return float(np.percentile(v, 25))
    if stat == "median":
        return float(np.median(v))
    raise ValueError(stat)


# --- effect sizes ----------------------------------------------------------------

def auc(low, high) -> float:
    """P(a value from `low` sits below one from `high`), ties half."""
    low, high = np.asarray(low, float), np.asarray(high, float)
    if not len(low) or not len(high):
        return float("nan")
    allv = np.concatenate([low, high])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    u = ranks[len(low):].sum() - len(high) * (len(high) + 1) / 2
    return float(u / (len(low) * len(high)))


def cohen(low, high) -> float:
    low, high = np.asarray(low, float), np.asarray(high, float)
    if len(low) < 2 or len(high) < 2:
        return float("nan")
    sd = np.sqrt((low.var(ddof=1) * (len(low) - 1) + high.var(ddof=1) * (len(high) - 1))
                 / (len(low) + len(high) - 2))
    return float((high.mean() - low.mean()) / max(sd, 1e-12))


# --- the labelled bench --------------------------------------------------------

def guard_clips() -> list[str]:
    a = {p.stem for p in (BENCH / "guard" / "A").glob("*.json")}
    b = {p.stem for p in (BENCH / "guard" / "B").glob("*.json")}
    return sorted(a & b)


def guard_truth(clip):
    from shadow_guard import drops
    da, busy_a, _ = drops(clip, "A")
    db, busy_b, _ = drops(clip, "B")
    busy = [w for w in busy_a if w in busy_b]
    agreed = [(s, max(a1, a2), min(b1, b2)) for s, a1, b1 in da for t, a2, b2 in db
              if s == t and min(b1, b2) > max(a1, a2)]
    return agreed, da + db, busy


def punch_truth(clip):
    a, b = load_labels(clip, "A"), load_labels(clip, "B")
    return consensus(a, b), a.punches + b.punches


def in_spans(t, spans):
    m = np.zeros(len(t), bool)
    for a, b in spans:
        m |= (t >= a) & (t < b)
    return m


def labelled_sets(clip):
    """Per side: (drop frame mask, guard frame mask), and the occasions
    (ending time, side, positive?) for the labelled-occasion metric."""
    agreed, either, busy = guard_truth(clip)
    truth, labelled = punch_truth(clip)
    t = clip.t
    inbusy = in_spans(t, busy)
    masks = {}
    for side in SIDES:
        near = np.zeros(len(t), bool)
        for p in labelled:
            if p.side in (side, "?", ""):
                near |= np.abs(t - p.at) <= sag.PUNCHING
        drop = in_spans(t, [(a, b) for s, a, b in agreed if s == side]) & inbusy & ~near
        anyd = in_spans(t, [(a, b) for s, a, b in either if s == side])
        guard = inbusy & ~anyd & ~near
        masks[side] = (drop, guard)
    occ = []
    for last, _combo in plant.endings(truth, None):
        lo, hi = last.at + sag.WINDOW[0], last.at + sag.WINDOW[1]
        if not any(a <= lo and hi <= b + 3.0 for a, b in busy):
            continue
        for side in SIDES:
            ov = max((min(b, hi) - max(a, lo) for s, a, b in agreed if s == side), default=0.0)
            anyov = max((min(b, hi) - max(a, lo) for s, a, b in either if s == side), default=0.0)
            if ov >= 0.2:
                occ.append((last.at, side, True))
            elif anyov <= 0:
                occ.append((last.at, side, False))
    punches = [(p.at, p.side) for p in labelled if p.side in SIDES]
    return masks, occ, punches


# --- candidates ------------------------------------------------------------------

def candidates():
    """(label, raw feature, baseline, smoothing seconds, window statistic)."""
    out = [("current: wrist-nose, p75", "wrist-nose", "p75", 0.0, "mean")]
    for f in ("wrist-mouth", "wrist-shoulder", "wrist-shoulders", "hand-nose",
              "elbow-shoulder", "wrist+elbow", "face distance", "forearm angle",
              "world wrist-nose", "world wrist+elbow", "world face 3d", "image z (unplanted)"):
        out.append((f"{f}, p75", f, "p75", 0.0, "mean"))
    for b in ("none", "median", "p90", "robust z"):
        out.append((f"wrist-nose, {b}", "wrist-nose", b, 0.0, "mean"))
    out.append(("wrist-nose, p75, smoothed 0.1 s", "wrist-nose", "p75", 0.1, "mean"))
    out.append(("wrist-nose, p75, window p25", "wrist-nose", "p75", 0.0, "p25"))
    out.append(("wrist-nose, p75, window median", "wrist-nose", "p75", 0.0, "median"))
    out.append(("wrist+elbow, robust z", "wrist+elbow", "robust z", 0.0, "mean"))
    out.append(("wrist-shoulder, robust z", "wrist-shoulder", "robust z", 0.0, "mean"))
    for b in ("none", "median", "robust z"):
        out.append((f"wrist-shoulders, {b}", "wrist-shoulders", b, 0.0, "mean"))
    out.append(("wrist-shoulders, robust z, window p25", "wrist-shoulders", "robust z", 0.0, "p25"))
    out.append(("wrist-shoulders, p75, window p25", "wrist-shoulders", "p75", 0.0, "p25"))
    out.append(("hand-shoulders, robust z", "hand-shoulders", "robust z", 0.0, "mean"))
    return out


def signal(clip, cand, cache):
    _label, feat, base, sm, _stat = cand
    key = (clip.name, id(clip), feat, base, sm)
    if key not in cache:
        r = cache.setdefault((clip.name, id(clip), "raw"), None) or raw(clip)
        cache[(clip.name, id(clip), "raw")] = r
        cache[key] = {s: smooth(clip.t, rolling(clip.t, r[feat][s], base), sm) for s in SIDES}
    return cache[key]


def labelled_metrics(cands):
    """frames and occasions, per guard-labelled person."""
    out = {c[0]: {"frames": {}, "occ": {}} for c in cands}
    counts = {}
    for name in guard_clips():
        clip = Clip.load(name)
        masks, occ, punches = labelled_sets(clip)
        person = person_of(name)
        counts[person] = (sum(int(m[0].sum()) for m in masks.values()),
                          sum(1 for o in occ if o[2]), sum(1 for o in occ if not o[2]))
        cache = {}
        for c in cands:
            h = signal(clip, c, cache)
            lo, hi = [], []
            for side in SIDES:
                d, g = masks[side]
                lo.append(h[side][d & np.isfinite(h[side])])
                hi.append(h[side][g & np.isfinite(h[side])])
            lo, hi = np.concatenate(lo), np.concatenate(hi)
            out[c[0]]["frames"].setdefault(person, []).append((lo, hi))
            pos, neg = [], []
            for at, side, positive in occ:
                v = window(clip.t, h[side], at, side, punches, c[4])
                if v is not None:
                    (pos if positive else neg).append(v)
            out[c[0]]["occ"].setdefault(person, []).append((pos, neg))
    return out, counts


def _summ(per_person, min_n=3):
    """Mean over people of AUC and d, and the pooled (per-person centred) d."""
    aucs, ds = [], []
    for person, parts in per_person.items():
        lo = np.concatenate([np.asarray(p[0], float) for p in parts])
        hi = np.concatenate([np.asarray(p[1], float) for p in parts])
        if len(lo) < min_n or len(hi) < min_n:
            continue
        aucs.append(auc(lo, hi))
        ds.append(cohen(lo, hi))
    return (float(np.mean(aucs)) if aucs else float("nan"),
            float(np.mean(ds)) if ds else float("nan"), len(aucs),
            float(np.min(aucs)) if aucs else float("nan"))


# --- the planted bench, labelled punches ------------------------------------------

_LIB = None


def planted_job(args):
    name, seed, cands = args
    global _LIB
    from planted_check import library, truth_of
    if _LIB is None:
        _LIB = library()
    clip = Clip.load(name)
    truth, labelled = truth_of(clip)
    setup, side = ("punch_right",), "left"
    plants, _, _ = plant.plan(clip, truth, labelled, _LIB, side, setup, 0.8, seed)
    planted = plant.apply(clip, plants, _LIB) if plants else clip
    punches = sorted((p.at, p.side) for p in truth.agreed if p.side in SIDES)
    ends = [last.at for last, _ in plant.endings(truth, None)]
    hit = {p.after for p in plants}
    cache = {}
    res = {}
    for c in cands:
        h = signal(planted, c, cache)
        lo, hi = [], []
        for at in ends:
            v = window(planted.t, h[side], at, side, punches, c[4])
            if v is not None:
                (lo if at in hit else hi).append(v)
        res[c[0]] = (lo, hi)
    return person_of(name), res


def planted_metrics(cands, seeds, workers):
    from concurrent.futures import ProcessPoolExecutor
    from planted_check import bench_names, truth_of
    names = [n for n in bench_names() if truth_of(Clip.load(n))[0].agreed]
    jobs = [(n, s, cands) for n in names for s in range(seeds)]
    out = {c[0]: {} for c in cands}
    with ProcessPoolExecutor(workers) as pool:
        for person, res in pool.map(planted_job, jobs):
            for label, pair in res.items():
                out[label].setdefault(person, []).append(pair)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-planted", action="store_true")
    ap.add_argument("--per-person", action="store_true", dest="per_person")
    ap.add_argument("--only", default="", help="only candidates whose label contains this")
    args = ap.parse_args()
    cands = [c for c in candidates() if not args.only or args.only in c[0] or c[0].startswith("current")]
    lab, counts = labelled_metrics(cands)
    print("guard-labelled people: drop frames, drop occasions, clean occasions")
    for p, (f, pos, neg) in counts.items():
        print(f"  {p:14s} {f:6d} {pos:4d} {neg:4d}")
    pl = None if args.no_planted else planted_metrics(cands, args.seeds, args.workers)
    print("\nper-person mean AUC (a drop sits lower) and Cohen's d; worst person's AUC in []")
    print(f"{'candidate':38s} {'frames AUC':>10s} {'d':>5s} {'occ AUC':>8s} {'d':>5s}"
          f"   {'planted AUC':>11s} {'d':>5s}")
    for c in cands:
        fa, fd, fn, fmin = _summ(lab[c[0]]["frames"])
        oa, od, on, omin = _summ(lab[c[0]]["occ"], min_n=2)
        line = (f"{c[0]:38s} {fa:6.3f}[{fmin:.2f}] {fd:5.2f} {oa:6.3f}[{omin:.2f}] {od:5.2f}")
        if pl is not None:
            pa, pd, pn, pmin = _summ(pl[c[0]])
            line += f"   {pa:6.3f}[{pmin:.2f}] {pd:5.2f} ({pn})"
        print(line, flush=True)
    if pl is not None and args.per_person:
        print("\nplanted AUC per person")
        people = sorted(pl[cands[0][0]])
        print(f"{'':38s} " + " ".join(f"{p[:6]:>6s}" for p in people))
        for c in cands:
            cells = []
            for p in people:
                lo = np.concatenate([np.asarray(x[0], float) for x in pl[c[0]][p]])
                hi = np.concatenate([np.asarray(x[1], float) for x in pl[c[0]][p]])
                cells.append(f"{auc(lo, hi):6.2f}" if len(lo) >= 3 and len(hi) >= 3 else "     -")
            print(f"{c[0]:38s} " + " ".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
