"""Coach mode, measured the way the drill was: at the sensor error rates
measured on real footage, against a planted habit and against no habit.

    python scripts/coach_check.py            # simulation, and the bench null
    python scripts/coach_check.py --quick    # fewer trials, for a smoke run

In coach mode ika calls the combos, so the setup is known and the punch
reader only has to say whether anything was thrown after a call. What is
simulated here is a person following the calls, frame by frame, so that the
real alignment (`coach.align`) and the real guard read (`sag.GuardTrack.after`)
run on it, including the ways they can go wrong: a missed last punch that
leaves the guard window sitting on top of it, a false punch that confirms a
call nobody answered or drags the window late.

The person:

- answers 95% of calls (an assumption, not a measurement), first punch
  0.7 s after the call (sd 0.15), then one every 0.3 s (sd 0.05);
- holds each hand, in the second after a combo, at an average that is drawn
  per occasion from the same numbers robustness.py uses, measured on the
  labelled footage: 0.08 torso lengths below the rolling guard when up, 0.34
  when down, spread 0.30. Between combos the same "up" level, drawn afresh
  each call. Frame noise 0.1 on top, and 3% of frames unread;
- while an arm is punching, that wrist sits 0.5 below the guard for 0.3 s
  (a straight is thrown below the nose; the depth is an assumption);
- with a habit, drops the right hand after one called combo at the planted
  rate, and after any other combo 8% of the time, as robustness.py does.

The sensor: each thrown punch is detected at the held-out recall, 47%, a few
hundredths of a second off, plus 5.6 false punches a minute at random times.

The real-footage null lays random coach schedules over the bench clips. The
people in them were following someone else's calls, so any combo ika reports
as a habit is false by construction, and the guard heights and punches are
real.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ika import coach, sag  # noqa: E402
from ika.profile import Profile  # noqa: E402

FPS = 30.0
UP, DOWN, SPREAD = -0.08, -0.34, 0.30
FRAME_NOISE = 0.10
UNREAD = 0.03
PUNCH_DEPTH = -0.5
PUNCH_HALF = 0.15
ANSWER = 0.95
BASE_DROP = 0.08
ROUND = 180.0
REST = 60.0

MEASURED = {"recall": 0.47, "false_per_minute": 5.6}
PERFECT = {"recall": 1.0, "false_per_minute": 0.0}


class SimTrack(sag.GuardTrack):
    """A guard track whose heights were simulated rather than read off a body."""

    def __init__(self, t, h):
        super().__init__()
        self._snap = (t, h)

    def snapshot(self):
        return self._snap


def calls_for(combos, minutes: float, seed: int) -> list[coach.Call]:
    """Whole three-minute rounds with a minute's rest between, as `ika coach
    --rounds` runs them."""
    rounds = max(1, int(round(minutes * 60 / ROUND)))
    out = []
    for r in range(rounds):
        out += coach.schedule(combos, ROUND, seed=seed * 101 + r,
                              start=coach.LEAD_IN + r * (ROUND + REST))
    return out


def person(rng, calls, habit_combo, habit: float, pace=(0.7, 0.3)):
    """What a fighter following the calls does: true punches, and guard
    heights frame by frame."""
    end_time = calls[-1].at + coach.gap(calls[-1].combo) + 2.0
    t = np.arange(0.0, end_time, 1.0 / FPS)
    h = {side: np.full(len(t), np.nan) for side in sag.WRISTS}
    punches = []
    bounds = [c.at for c in calls] + [end_time]
    # Before the first call: in guard.
    first = np.searchsorted(t, bounds[0])
    for side in sag.WRISTS:
        h[side][:first] = rng.normal(UP, SPREAD)
    for i, call in enumerate(calls):
        a, b = np.searchsorted(t, call.at), np.searchsorted(t, bounds[i + 1])
        answered = rng.random() < ANSWER
        thrown = []
        if answered:
            at = call.at + float(np.clip(rng.normal(pace[0], 0.15), 0.35, pace[0] + 0.5))
            for side in call.combo.sides():
                thrown.append((at, side))
                at += float(max(0.18, rng.normal(pace[1], 0.05)))
        punches += thrown
        end = thrown[-1][0] if thrown else call.at
        after = np.searchsorted(t, end + 0.2)
        dropped = answered and (rng.random() < (habit if call.combo == habit_combo
                                                else BASE_DROP))
        for side in sag.WRISTS:
            h[side][a:after] = rng.normal(UP, SPREAD)
            level = DOWN if (dropped and side == "right") else UP
            h[side][after:b] = rng.normal(level, SPREAD)
    for side in sag.WRISTS:
        h[side] += rng.normal(0.0, FRAME_NOISE, len(t))
        for at, s in punches:
            if s == side:
                a, b = np.searchsorted(t, at - PUNCH_HALF), np.searchsorted(t, at + PUNCH_HALF)
                h[side][a:b] = PUNCH_DEPTH + rng.normal(0.0, FRAME_NOISE, b - a)
        h[side][rng.random(len(t)) < UNREAD] = np.nan
    return t, h, punches


def sensed(rng, punches, seconds: float, rates) -> list[tuple[float, str]]:
    seen = [(at + float(rng.normal(0.0, 0.03)), side) for at, side in punches
            if rng.random() < rates["recall"]]
    for _ in range(rng.poisson(rates["false_per_minute"] * seconds / 60.0)):
        seen.append((float(rng.uniform(0.0, seconds)), str(rng.choice(["left", "right"]))))
    return sorted(seen)


def session(rng, combos, habit_combo, habit, minutes, rates, seed, pace=(0.7, 0.3)):
    calls = calls_for(combos, minutes, seed)
    t, h, punches = person(rng, calls, habit_combo, habit, pace)
    detected = sensed(rng, punches, float(t[-1]), rates)
    return coach.align(calls, SimTrack(t, h), detected)


def named(found, habit_combo) -> bool:
    return any(f.then == "guard_down_right" and f.context == habit_combo.context
               for f in found)


def single(minutes, habit, rates, trials, seed=0, pace=(0.7, 0.3)):
    """One session whose combos include the habit's: named (or, with no habit,
    any habit reported), plus how many calls were answered and read."""
    rng = np.random.default_rng(seed)
    hits, answered, total = 0, 0, 0
    for trial in range(trials):
        combos = coach.choose(seed=seed * 7919 + trial)
        reps = session(rng, combos, combos[0], habit, minutes, rates, trial, pace)
        found = coach.read(reps)
        hits += named(found, combos[0]) if habit else bool(found)
        answered += sum(r.read for r in reps)
        total += len(reps)
    return hits / trials, answered / max(total, 1)


def pooled(habit, sessions, rates, trials, minutes=6.0, seed=1):
    """Sessions into one profile, the way `ika coach` records them
    (`coach.evidence`), with the curriculum `coach.plan` picks from what the
    profile holds. The habit's combo is fixed per fighter and is only in a
    session when the plan calls it: drawn in rotation until a session
    proposes it, then called again to confirm it."""
    rng = np.random.default_rng(seed)
    hits, called = 0, 0
    catalogue = [coach.Combo.parse(n) for n in coach.CATALOGUE]
    for trial in range(trials):
        record = Profile("sim")
        habit_combo = catalogue[trial % len(catalogue)]
        for day in range(1, sessions + 1):
            now = day * 86400.0
            combos = coach.plan(record, now, seed=seed * 100003 + trial * 101 + day)
            called += habit_combo in combos
            reps = session(rng, combos, habit_combo, habit, minutes, rates,
                           trial * 1000 + day)
            record.add(coach.evidence(reps, record), at=now)
        guard = [s for s in record.habits(now=sessions * 86400.0)
                 if coach.is_coach(s.context)]
        hits += named(guard, habit_combo) if habit else bool(guard)
    return hits / trials, called / (trials * sessions)


def bench_null(schedules=40):
    """Random coach schedules over the real bench clips: guard heights and
    punches are real, the calls were never made, so every finding is false."""
    from ika.shadow import Clip
    from ika.strike import StrikeReader

    names = sorted(p.stem for p in (Path(__file__).resolve().parent.parent
                                    / "data" / "shadow" / "marks").glob("*.npz"))
    if not names:
        return None
    false, trials, read, total = 0, 0, 0, 0
    per_clip = []
    for name in names:
        clip = Clip.load(name)
        track = sag.GuardTrack(aspect=clip.aspect)
        reader = StrikeReader(aspect=clip.aspect)
        punches = []
        for at, _hands, body in clip.frames():
            track.update(body, at)
            punches += [(s.at, s.side) for s in reader.observe(body, at)]
        seconds = float(clip.t[-1])
        mine = 0
        for k in range(schedules):
            combos = coach.choose(seed=10_000 + k)
            calls = coach.schedule(combos, seconds - 7.0, seed=k, start=5.0)
            reps = coach.align(calls, track, punches)
            found = coach.read(reps)
            mine += bool(found)
            read += sum(r.read for r in reps)
            total += len(reps)
        false += mine
        trials += schedules
        per_clip.append((name, mine))
    return false / trials, trials, read / max(total, 1), per_clip


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-bench", action="store_true", dest="no_bench")
    args = ap.parse_args()
    n = 40 if args.quick else 200
    npool = 20 if args.quick else 100

    print("a planted habit (right hand drops after one called combo, 80%), one session"
          " whose three combos include it:")
    print(f"  {'minutes':>7s} {'perfect':>8s} {'measured':>9s} {'calls read':>11s}")
    for minutes in (3, 6, 12):
        p, _ = single(minutes, 0.8, PERFECT, n)
        m, read = single(minutes, 0.8, MEASURED, n)
        print(f"  {minutes:7d} {p:8.0%} {m:9.0%} {read:11.0%}")
    print("\nthe same at a 50% habit, measured sensors:")
    for minutes in (3, 6, 12):
        m, _ = single(minutes, 0.5, MEASURED, n)
        print(f"  {minutes:7d} min  {m:4.0%}")
    print("\nno habit at all, how often one is reported anyway (measured sensors):")
    for minutes in (3, 12):
        m, _ = single(minutes, 0.0, MEASURED, n * 2)
        print(f"  {minutes:7d} min  {m:4.0%}  of {n * 2}")
    print("\nno habit, a slower fighter than the alignment expects (first punch at 1.0 s,"
          " then every 0.45 s):")
    for minutes in (3, 12):
        m, _ = single(minutes, 0.0, MEASURED, n * 2, pace=(1.0, 0.45))
        print(f"  {minutes:7d} min  {m:4.0%}  of {n * 2}")
    print("\nacross six-minute sessions into one profile, curriculum from coach.plan"
          " (measured sensors):")
    for habit in (0.8, 0.0):
        cells = []
        for k in (4, 8, 12):
            rate, called = pooled(habit, k, MEASURED, npool)
            cells.append(f"{k} sessions {rate:4.0%} (habit combo called in {called:.0%})"
                         if habit else f"{k} sessions {rate:4.0%}")
        label = "80% habit named" if habit else "no habit, told one"
        print(f"  {label:18s} " + "   ".join(cells))

    if not args.no_bench:
        result = bench_null()
        if result is None:
            print("\nbench null: no cached landmarks (scripts/shadow_fetch.py)")
        else:
            rate, trials, read, per_clip = result
            print(f"\nrandom coach schedules over the real bench clips: {rate:.1%} of "
                  f"{trials} report a habit (should be under 5%); {read:.0%} of calls "
                  "had a detection in the window and a guard read")
            print("  per clip: " + ", ".join(f"{name} {k}" for name, k in per_clip))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
