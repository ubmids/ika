"""Score dropped-guard detection against two labellers, on real footage.

    python scripts/shadow_guard.py

A drop both labellers marked, overlapping in time and on the same arm, is a
drop. A guard_down event is a hit if it starts inside one, or up to 0.3 s
before, and false if it lands where neither labeller saw the hand down.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ika.drill import DrillReader  # noqa: E402
from ika.shadow import BENCH, ROOT, Clip  # noqa: E402

EARLY = 0.3


def drops(clip: Clip, who: str):
    path = BENCH / "guard" / who / f"{clip.name}.json"
    raw = json.loads(path.read_text())
    out, busy, seen = [], [], []
    for sheet in raw["sheets"]:
        start = float(sheet["start"])
        seen.append((start, start + 3.0))
        if sheet.get("activity") == "shadowboxing":
            busy.append((start, start + 3.0))
        for d in sheet.get("drops", []) if isinstance(sheet.get("drops"), list) else []:
            a, b = start + int(d["from_tile"]) / 10, start + (int(d["to_tile"]) + 1) / 10
            out.append((clip.anatomical((a + b) / 2, d["arm"]), a, b))
    # join a drop that runs across a sheet edge back into one
    out.sort()
    joined = []
    for side, a, b in out:
        if joined and joined[-1][0] == side and a - joined[-1][2] <= 0.15:
            joined[-1] = (side, joined[-1][1], max(b, joined[-1][2]))
        else:
            joined.append((side, a, b))
    return joined, busy, seen


def main() -> int:
    total = {"agreed": 0, "hit": 0, "false": 0, "minutes": 0.0, "events": 0}
    for path in sorted((BENCH / "guard" / "A").glob("*.json")):
        clip = Clip.load(path.stem)
        da, busy_a, _ = drops(clip, "A")
        db, busy_b, _ = drops(clip, "B")
        busy = [w for w in busy_a if w in busy_b]
        agreed = [(s, max(a1, a2), min(b1, b2)) for s, a1, b1 in da for t, a2, b2 in db
                  if s == t and min(b1, b2) > max(a1, a2)]
        either = da + db
        reader = DrillReader(framing="stand", aspect=clip.aspect)
        events = []
        for at, hands, body in clip.frames():
            events += [e for e in reader.observe(hands, body, at) if e.name.startswith("guard_down")]
        events = [e for e in events if any(a <= e.at < b for a, b in busy)]
        sides = lambda e: ("left", "right") if e.name.endswith("both") else (e.name.rsplit("_", 1)[1],)
        hit = sum(1 for s, a, b in agreed
                  if any(s in sides(e) and a - EARLY <= e.at <= b for e in events))
        false = sum(1 for e in events
                    if not any(s in sides(e) and a - EARLY <= e.at <= b for s, a, b in either))
        minutes = sum(b - a for a, b in busy) / 60
        print(f"  {clip.name:14s} agreed drops {len(agreed):3d}  caught {hit:3d}  "
              f"events {len(events):3d}  false {false:3d}  ({minutes:.1f} min)")
        for k, v in (("agreed", len(agreed)), ("hit", hit), ("false", false),
                     ("minutes", minutes), ("events", len(events))):
            total[k] += v
    print(f"  ALL: caught {total['hit']}/{total['agreed']} "
          f"({total['hit'] / max(total['agreed'], 1):.0%}), "
          f"false {total['false'] / max(total['minutes'], 1e-6):.1f}/min "
          f"of {total['events']} events")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
