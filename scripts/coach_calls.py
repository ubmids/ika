"""Can the bench footage test coach mode? The coaches in it call combos out loud.

    python scripts/coach_calls.py

The bench clips are follow-along workouts, and several of the coaches call
numbered combos ("one, two, three, two"). If those calls can be recovered
with times, the clips are a real-footage test of coach mode: the call is
known, the person on screen answers it, and the read runs on real guard
heights.

The calls are read from word-timed transcripts of each clip's three-minute
section (OpenAI Whisper, run locally over the section audio; the transcripts
sit in data/shadow/audio and are not committed, like the footage). A call is
two or more number words, one to six, each within 0.8 s of the last, not
following "number" or "round" ("round number one" is not a call).

It reports, per clip: how many calls, how often the punch reader fired in
the response window after one against windows at random times (if the
answer detector means anything, real calls should beat random ones), and how
many times each combo was called, against the sag.MIN_SUPPORT a combo needs
before the read can test it.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from ika import coach, sag  # noqa: E402
from ika.shadow import ROOT, Clip  # noqa: E402
from ika.strike import StrikeReader  # noqa: E402

NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
JOIN = 0.8


def calls(name: str) -> list[coach.Call]:
    payload = json.loads((ROOT / "audio" / f"{name}.json").read_text())
    words = [(w["word"].strip(" ,.!?-").lower(), float(w["start"]))
             for seg in payload["segments"] for w in seg.get("words", [])]
    out, run = [], []

    def close():
        if len(run) >= 2:
            out.append(coach.Call(run[0][1], coach.Combo.parse(
                "-".join(str(n) for n, _ in run))))

    for i, (word, at) in enumerate(words):
        counted = word in NUMBERS and (i == 0 or words[i - 1][0] not in ("number", "round"))
        if counted and run and at - run[-1][1] < JOIN:
            run.append((NUMBERS[word], at))
            continue
        close()
        run = [(NUMBERS[word], at)] if counted else []
    close()
    return out


def answered(times: np.ndarray, at: float, combo: coach.Combo) -> bool:
    lo, hi = coach.window(at, combo)
    return bool(((times >= lo) & (times <= hi)).any())


def main() -> int:
    rng = np.random.default_rng(0)
    names = sorted(p.stem for p in (ROOT / "audio").glob("*.json")
                   if (ROOT / "marks" / f"{p.stem}.npz").exists())
    if not names:
        print("no transcripts with cached landmarks in data/shadow")
        return 1
    total_calls = total_real = 0
    total_random = 0.0
    for name in names:
        made = calls(name)
        if not made:
            print(f"  {name}: no numbered calls")
            continue
        clip = Clip.load(name)
        track, reader, punches = sag.GuardTrack(aspect=clip.aspect), StrikeReader(aspect=clip.aspect), []
        for at, _hands, body in clip.frames():
            track.update(body, at)
            punches += [(s.at, s.side) for s in reader.observe(body, at)]
        times = np.asarray([p for p, _ in punches])
        real = sum(answered(times, c.at, c.combo) for c in made)
        seconds = float(clip.t[-1])
        random = np.mean([answered(times, float(rng.uniform(3.0, seconds - 5.0)),
                                   made[int(rng.integers(len(made)))].combo)
                          for _ in range(2000)])
        counts = Counter(c.combo.name for c in made)
        top = max(counts.values())
        found = coach.read(coach.align(made, track, punches))
        print(f"  {name}: {len(made):3d} calls, answered {real / len(made):4.0%} "
              f"(random times {random:4.0%}); most-called combo {top}x of the "
              f"{sag.MIN_SUPPORT} the read needs; {len(found)} habit(s)")
        total_calls += len(made)
        total_real += real
        total_random += random * len(made)
    if total_calls:
        print(f"\n  all: {total_calls} calls, answered {total_real / total_calls:.0%} "
              f"against {total_random / total_calls:.0%} at random times")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
