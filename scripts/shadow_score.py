"""Score punch detectors on real people, against two independent labellers.

    python scripts/shadow_score.py

Order of the report is deliberate. How well the two labellers agree comes
first, then how one labeller scores when treated as a detector against the
other, because that is the ceiling: nothing can honestly be said to read
punches better than the two labellers agree on them. (The labellers are two
separate vision-model instances, not people; see `ika/shadow.py`.) Then the detectors
that need no fitting. The fitted one is scored by `train_strike.py`, since it
can only be judged on people it was not fitted on.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ika.drill import DrillReader  # noqa: E402
from ika.shadow import (BENCH, Clip, Punch, Score, consensus, inside,  # noqa: E402
                        load_labels, match, score)
from ika.strike import ExcursionReader  # noqa: E402


def clips() -> list[str]:
    a = {p.stem for p in (BENCH / "labels" / "A").glob("*.json")}
    b = {p.stem for p in (BENCH / "labels" / "B").glob("*.json")}
    return sorted(a & b)


def approach_detector(clip: Clip) -> list[Punch]:
    """The drill loop's original reader: a palm closing on the lens."""
    reader = DrillReader(calibration=0.0, framing="close")
    found = []
    for at, hands, body in clip.frames():
        for event in reader.observe(hands, body, at):
            if event.name.startswith("punch_"):
                found.append(Punch(event.at, event.name[6:]))
    return found


def excursion_detector(clip: Clip) -> list[Punch]:
    reader = ExcursionReader(aspect=clip.aspect)
    found = []
    for at, _hands, body in clip.frames():
        found += [Punch(s.at, s.side) for s in reader.observe(body, at)]
    return found


def report(name: str, s: Score) -> None:
    print(f"  {name:22s} recall {s.recall:5.0%}  false/min {s.false_per_minute:5.1f}  "
          f"({s.hits} hit, {s.missed} missed, {s.false} false, {s.minutes:.1f} min)")


def main() -> int:
    names = clips()
    loaded = {n: Clip.load(n) for n in names}
    truths, labels = {}, {}
    print("labeller agreement (F1, same arm / any arm), on sheets both call shadowboxing:")
    for n in names:
        a, b = load_labels(loaded[n], "A"), load_labels(loaded[n], "B")
        labels[n] = (a, b)
        truth = truths[n] = consensus(a, b)
        print(f"  {n:14s} {truth.agreement:4.0%} / {truth.agreement_unsided:4.0%}   "
              f"{len(truth.agreed):3d} agreed, {len(truth.disputed):3d} disputed, "
              f"{sum(y - x for x, y in truth.windows):4.0f} s")

    hits = missed = extra = 0
    minutes = 0.0
    for n in names:
        a, b = labels[n]
        w = truths[n].windows
        pa = [p for p in a.punches if inside(p.at, w)]
        pb = [p for p in b.punches if inside(p.at, w)]
        pairs, lost, more = match(pb, pa, sided=False)
        hits, missed, extra = hits + len(pairs), missed + len(lost), extra + len(more)
        minutes += sum(y - x for x, y in w) / 60
    print("\nthe ceiling: labeller A scored as a detector against labeller B")
    report("labeller vs labeller", Score(hits, missed, extra, minutes, 0))

    print("\ndetectors that need no fitting, pooled over everyone:")
    for label, detect in (("palm closing on lens", approach_detector),
                          ("wrist leaves guard", excursion_detector)):
        total = None
        for n in names:
            s = score(detect(loaded[n]), truths[n])
            total = s if total is None else total + s
        report(label, total)
    print("\nthe fitted detector, on people it never saw: python scripts/train_strike.py --quick")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
