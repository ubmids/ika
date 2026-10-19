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

# Guard models: what the average guard height in the second after a combo
# is, with and without a drop. Each draws one occasion's level.
#
#   real          measured on the planted bench (scripts/planted_check.py
#                 --heights, bench/shadow/guard_heights.json): real guard
#                 tracks, the shipped read from the shoulders, the held-out
#                 punch reader, an 80% habit planted from real drops. "Down"
#                 draws from the heights after combos a plant landed in, "up"
#                 from every other, each person centred on their own guard.
#                 Drawn from the measured quantiles, so the shape is real and
#                 not assumed normal: the heights have a long low tail, and
#                 the rank test reads them at an effect of 0.70 where a normal
#                 curve with the same means and spreads would say about 0.65.
#   real-nose     the same, measured from the nose, the read's old reference.
#   real-normal   the previous REAL model, normal: -0.25 against -0.16, sd
#                 0.19, the means and spread of the same nose measurement.
#   sim           what robustness.py assumed from labelled drop frames,
#                 -0.34 against -0.08, sd 0.30. Optimistic; kept to compare.
HEIGHTS = Path(__file__).resolve().parent.parent / "bench" / "shadow" / "guard_heights.json"


class Normal:
    def __init__(self, up, down, spread):
        self.up, self.down, self.spread = up, down, spread

    def draw(self, rng, down: bool) -> float:
        return float(rng.normal(self.down if down else self.up, self.spread))

    def describe(self) -> str:
        return (f"normal, after the habit combo {self.down:+.2f}, after others "
                f"{self.up:+.2f}, sd {self.spread:.2f}")


class Measured:
    def __init__(self, reference):
        import json
        d = json.loads(HEIGHTS.read_text())[reference]
        self.q = np.linspace(0.0, 1.0, len(d["rest"]))
        self.rest, self.planted = np.asarray(d["rest"]), np.asarray(d["planted"])
        self.reference, self.effect = reference, d["effect"]

    def draw(self, rng, down: bool) -> float:
        return float(np.interp(rng.random(), self.q, self.planted if down else self.rest))

    def describe(self) -> str:
        return (f"measured from the {self.reference}, after a plant median "
                f"{np.median(self.planted):+.2f}, after others {np.median(self.rest):+.2f}, "
                f"effect {self.effect:.2f}")


GUARD = {"real": lambda: Measured("shoulders"), "real-nose": lambda: Measured("nose"),
         "real-normal": lambda: Normal(-0.16, -0.25, 0.19),
         "sim": lambda: Normal(UP, DOWN, SPREAD)}
_MODELS = {}


def model(guard):
    if guard not in _MODELS:
        _MODELS[guard] = GUARD[guard]()
    return _MODELS[guard]


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


def person(rng, calls, habit_combo, habit: float, pace=(0.7, 0.3), guard="sim"):
    level = model(guard)
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
        h[side][:first] = level.draw(rng, False)
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
            h[side][a:after] = level.draw(rng, False)
            h[side][after:b] = level.draw(rng, dropped and side == "right")
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


def session(rng, combos, habit_combo, habit, minutes, rates, seed, pace=(0.7, 0.3),
            guard="sim"):
    calls = calls_for(combos, minutes, seed)
    t, h, punches = person(rng, calls, habit_combo, habit, pace, guard)
    detected = sensed(rng, punches, float(t[-1]), rates)
    return coach.align(calls, SimTrack(t, h), detected)


def named(found, habit_combo) -> bool:
    return any(f.then == "guard_down_right" and f.context == habit_combo.context
               for f in found)


def single(minutes, habit, rates, trials, seed=0, pace=(0.7, 0.3), guard="sim"):
    """One session whose combos include the habit's: named (or, with no habit,
    any habit reported), plus how many calls were answered and read."""
    rng = np.random.default_rng(seed)
    hits, answered, total = 0, 0, 0
    for trial in range(trials):
        combos = coach.choose(seed=seed * 7919 + trial)
        reps = session(rng, combos, combos[0], habit, minutes, rates, trial, pace, guard)
        found = coach.read(reps)
        hits += named(found, combos[0]) if habit else bool(found)
        answered += sum(r.read for r in reps)
        total += len(reps)
    return hits / trials, answered / max(total, 1)


def pooled(habit, sessions, rates, trials, minutes=6.0, seed=1, guard="sim"):
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
                           trial * 1000 + day, guard=guard)
            record.add(coach.evidence(reps, record), at=now)
        standing = [s for s in record.habits(now=sessions * 86400.0)
                    if coach.is_coach(s.context)]
        hits += named(standing, habit_combo) if habit else bool(standing)
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


def report(guard: str, n: int, npool: int) -> None:
    print(f"=== guard model {guard}: {model(guard).describe()}, rule {sag.RULE} ===\n")
    print("a planted habit (right hand drops after one called combo, 80%), one session"
          " whose three combos include it:")
    print(f"  {'minutes':>7s} {'perfect':>8s} {'measured':>9s} {'calls read':>11s}")
    for minutes in (3, 6, 12):
        p, _ = single(minutes, 0.8, PERFECT, n, guard=guard)
        m, read = single(minutes, 0.8, MEASURED, n, guard=guard)
        print(f"  {minutes:7d} {p:8.0%} {m:9.0%} {read:11.0%}")
    print("\nthe same at a 50% habit, measured sensors:")
    for minutes in (3, 6, 12):
        m, _ = single(minutes, 0.5, MEASURED, n, guard=guard)
        print(f"  {minutes:7d} min  {m:4.0%}")
    if guard.startswith("real"):
        print("\nhow long one session (same three combos, --rounds) takes, 80% habit,"
              " measured sensors:")
        for minutes in (24, 48, 96, 192):
            m, _ = single(minutes, 0.8, MEASURED, max(n // 2, 20), guard=guard)
            print(f"  {minutes:7d} min  {m:4.0%}")
        # Why it levels off rather than climbing: sag's read also needs the
        # share of "hand down" occasions (window mean below -DOWN) to be
        # MIN_LIFT times the other combos'. At this separation an 80% habit
        # gives about 56% against 43%, a lift near 1.3, so with more minutes
        # the rank test grows certain and the lift bar still turns it away.
    print("\nno habit at all, how often one is reported anyway (measured sensors):")
    for minutes in (3, 12):
        m, _ = single(minutes, 0.0, MEASURED, n * 2, guard=guard)
        print(f"  {minutes:7d} min  {m:4.0%}  of {n * 2}")
    print("\nno habit, a slower fighter than the alignment expects (first punch at 1.0 s,"
          " then every 0.45 s):")
    for minutes in (3, 12):
        m, _ = single(minutes, 0.0, MEASURED, n * 2, pace=(1.0, 0.45), guard=guard)
        print(f"  {minutes:7d} min  {m:4.0%}  of {n * 2}")
    print("\nacross six-minute sessions into one profile, curriculum from coach.plan"
          " (measured sensors):")
    for habit in (0.8, 0.0):
        cells = []
        for k in (4, 8, 12):
            rate, called = pooled(habit, k, MEASURED, npool, guard=guard)
            cells.append(f"{k} sessions {rate:4.0%} (habit combo called in {called:.0%})"
                         if habit else f"{k} sessions {rate:4.0%}")
        label = "80% habit named" if habit else "no habit, told one"
        print(f"  {label:18s} " + "   ".join(cells))
    print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-bench", action="store_true", dest="no_bench")
    ap.add_argument("--guard", nargs="+", default=["real", "sim"], choices=sorted(GUARD),
                    help="guard model(s): real-measured separation, or the older sim one")
    ap.add_argument("--rule", default=sag.RULE, choices=sag.RULES,
                    help="the guard read's size bar: lift (shipped) or effect (measured, not adopted)")
    ap.add_argument("--min-effect", type=float, default=sag.MIN_EFFECT, dest="min_effect")
    args = ap.parse_args()
    sag.RULE, sag.MIN_EFFECT = args.rule, args.min_effect
    print(f"decision rule: {args.rule}" + (f" {args.min_effect}" if args.rule == "effect" else "") + "\n")
    n = 40 if args.quick else 200
    npool = 20 if args.quick else 100

    for guard in args.guard:
        report(guard, n, npool)

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
