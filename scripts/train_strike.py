"""Fit the punch classifier on labelled real footage, and prove it on strangers.

    python scripts/train_strike.py          # leave-one-person-out, then fit all
    python scripts/train_strike.py --quick  # the held-out report only

Every clip is a different person, so leaving one clip out at a time is
leaving one person out: the number reported is how the detector does on
someone it has never seen, which is the only situation it will ever be in.
The firing threshold is chosen on the training people too, never on the one
held out.

A frame is a positive when the arm it belongs to throws a punch both
labellers agreed on, peaking within the next 0.2 s. So the model learns to
call a punch on the way out, not at full extension, which is the commit-early
rule the project is built on. Frames near a disputed label, and near a punch
whose arm the labellers disagreed on, are left out of training rather than
guessed at.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ika.shadow import (BENCH, Clip, Punch, Score, consensus, inside,  # noqa: E402
                        load_labels, person_of, score)
from ika.strike import ARMS, WEIGHTS, StrikeFeatures, StrikeModel, StrikeReader  # noqa: E402

LEAD = 0.20      # a frame this long before a peak is already "a punch coming"
AFTER = 0.03
IGNORE = 0.30    # frames this close to a label, but outside the positive span


def dataset(clip: Clip):
    a, b = load_labels(clip, "A"), load_labels(clip, "B")
    truth = consensus(a, b)
    quiet = [w for w in a.covered if w in b.covered
             and not inside(w[0] + 0.1, a.windows) and not inside(w[0] + 0.1, b.windows)]
    features = StrikeFeatures(aspect=clip.aspect)
    frames = []
    for at, _hands, body in clip.frames():
        rows = features.update(body, at)
        if rows is not None:
            frames.append((at, rows))
    agreed = [_resolve(p, frames) for p in truth.agreed]

    xs, ys, ws = [], [], []
    for at, rows in frames:
        busy, rest = inside(at, truth.windows), inside(at, quiet)
        if not (busy or rest):
            continue
        for side in ARMS:
            y, w = 0.0, 1.0
            if busy:
                for p in agreed:
                    gap = p.at - at
                    if -AFTER <= gap <= LEAD and p.side == side:
                        y, w = 1.0, 1.0
                    elif -IGNORE <= gap <= IGNORE + LEAD and p.side == side and y == 0:
                        w = 0.0
                for p in truth.disputed:
                    if abs(p.at - at) <= IGNORE:
                        w = 0.0
            xs.append(rows[side]); ys.append(y); ws.append(w)
    return (np.asarray(xs, np.float32), np.asarray(ys, np.float32),
            np.asarray(ws, np.float32), truth)


def _resolve(p: Punch, frames) -> Punch:
    """A punch both labellers saw but gave to different arms goes to the arm
    whose wrist had actually left its guard further at that moment. Dropping
    those punches instead threw away up to a third of the labels on
    fighters seen side-on, where arm is exactly what is hard to call from a picture."""
    if p.side != "?":
        return p
    near = [rows for at, rows in frames if abs(at - p.at) <= 0.1]
    if not near:
        return p
    reach = {side: max(float(r[side][7]) for r in near) for side in ARMS}
    return Punch(p.at, max(reach, key=reach.get), p.kind, p.sure)


def fit(x, y, w, seed=0, hidden=32, epochs=300) -> StrikeModel:
    keep = w > 0
    x, y = x[keep], y[keep]
    mean, scale = x.mean(0), x.std(0) + 1e-6
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(x.shape[1], hidden), torch.nn.ReLU(),
                              torch.nn.Linear(hidden, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-2)
    X = torch.tensor((x - mean) / scale)
    Y = torch.tensor(y)[:, None]
    pos = float(y.mean())
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / max(pos, 1e-6)))
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(net(X), Y)
        loss.backward()
        opt.step()
    w1, b1 = net[0].weight.detach().numpy().T, net[0].bias.detach().numpy()
    w2, b2 = net[2].weight.detach().numpy().T, net[2].bias.detach().numpy()
    return StrikeModel(mean, scale, w1, b1, w2, b2, threshold=0.5)


def detect(clip: Clip, model: StrikeModel, threshold: float) -> list[Punch]:
    reader = StrikeReader(model=model, aspect=clip.aspect, threshold=threshold)
    found = []
    for at, _hands, body in clip.frames():
        found += [Punch(s.at, s.side) for s in reader.observe(body, at)]
    return found


def objective(s: Score) -> float:
    # Recall less a price for false punches: one a minute costs two points.
    return s.recall - 0.02 * s.false_per_minute


def choose_threshold(clips, model) -> float:
    grid = (0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99)
    def total(th):
        return sum((score(detect(c, model, th), t) for c, t in clips[1:]),
                   score(detect(clips[0][0], model, th), clips[0][1]))
    return max(grid, key=lambda th: objective(total(th)))


def report(name: str, s: Score) -> None:
    print(f"  {name:22s} recall {s.recall:5.0%}  false/min {s.false_per_minute:5.1f}  "
          f"({s.hits} hit, {s.missed} missed, {s.false} false, "
          f"{s.wrong_side} wrong arm, {s.minutes:.1f} min)", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    root = BENCH / "labels"
    names = sorted({p.stem for p in (root / "A").glob("*.json")}
                   & {p.stem for p in (root / "B").glob("*.json")})
    clips = {n: Clip.load(n) for n in names}
    data = {n: dataset(clips[n]) for n in names}
    people = [n for n in names if data[n][3].agreed]
    who = {n: person_of(n) for n in names}
    print(f"{len(set(who[n] for n in people))} people in {len(people)} clips, "
          f"{sum(len(data[n][3].agreed) for n in people)} agreed punches, "
          f"{sum(int(data[n][1].sum()) for n in names)} positive frames")

    print("\nleave one person out:")
    held = None
    for out in people:
        train = [n for n in names if who[n] != who[out]]
        x = np.concatenate([data[n][0] for n in train])
        y = np.concatenate([data[n][1] for n in train])
        w = np.concatenate([data[n][2] for n in train])
        model = fit(x, y, w)
        th = choose_threshold([(clips[n], data[n][3]) for n in train if data[n][3].agreed],
                              model)
        s = score(detect(clips[out], model, th), data[out][3])
        held = s if held is None else held + s
        report(f"{out} @{th}", s)
    report("HELD OUT, pooled", held)

    if not args.quick:
        x = np.concatenate([data[n][0] for n in names])
        y = np.concatenate([data[n][1] for n in names])
        w = np.concatenate([data[n][2] for n in names])
        model = fit(x, y, w)
        model.threshold = choose_threshold([(clips[n], data[n][3]) for n in people], model)
        model.save(WEIGHTS)
        print(f"\nsaved {WEIGHTS} at threshold {model.threshold}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
