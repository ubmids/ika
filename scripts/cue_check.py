"""Does the call land before the lapse, and how often is it right?

    python scripts/cue_check.py

Measured on sessions with a planted habit, because a real one needs a person
whose habit is already known, and that is the one thing public footage does
not come with. Each subject drops their guard after two right hands with a
set probability, among combos that do not. One session teaches the caller the habit; a second, fresh
session is where the calls are scored, so nothing is judged on the stream it
learned from. A subject with no habit at all measures how often it calls
something that is not there.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tests"))

from ika.cue import Caller, measure  # noqa: E402
from ika.drill import CALIBRATION_SECONDS, read_habits  # noqa: E402
from ika.sag import current_combo  # noqa: E402
from test_drill import Session  # noqa: E402


def subject(seed: int, rate: float, combos: int = 60) -> Session:
    s = Session()
    s.rest(CALIBRATION_SECONDS + 1.0)
    rng = np.random.default_rng(seed)
    for _ in range(combos):
        _combo(s, rng, rate, lambda events: None)
    return s


def _combo(s: Session, rng, rate: float, watch) -> None:
    """One exchange: two rights, or a left-right. With a habit the guard drops
    after two rights at `rate`; without one it drops after either combo at
    random, so drops happen but belong to no setup."""
    doubled = rng.random() < 0.5
    watch(s.punch(is_left=not doubled)); watch(s.rest(0.4))
    watch(s.punch()); watch(s.rest(0.4))
    drop = (doubled and rng.random() < rate) if rate else rng.random() < 0.3
    watch(s.drop_guard() if drop else s.rest(0.7))
    watch(s.rest(0.6))


def replay_calls(learned, seed: int, rate: float):
    """A fresh session, with the caller armed from an earlier one."""
    s = Session()
    caller = Caller(learned)          # the live default gap
    s.rest(CALIBRATION_SECONDS + 1.0)
    rng = np.random.default_rng(seed)

    def watch(events):
        # Asked after every event, exactly as the live loop does.
        for event in events:
            caller.update(current_combo(s.reader.punches()), event.at)

    for _ in range(60):
        _combo(s, rng, rate, watch)
    return caller.cues, list(s.reader.events), s.at


def main() -> int:
    print("planted: drops the guard after punches, at each rate; calls scored on a new session")
    for rate in (0.5, 0.75, 0.9):
        learned = read_habits(subject(seed=1, rate=rate).reader)
        lapses = [h for h in learned if h.then.startswith("guard_down")]
        cues, events, seconds = replay_calls(learned, seed=2, rate=rate)
        m = measure(cues, events)
        drops = sum(e.name.startswith("guard_down") for e in events)
        print(f"  {rate:.0%} habit: {len(lapses)} lapse habit(s) learned; {m['cues']} calls, "
              f"{m['precision']:.0%} followed by the lapse, median "
              f"{m['median_lead']:.2f}s before the hand went down; {drops} drops in the round")
    learned = read_habits(subject(seed=3, rate=0.0).reader)
    cues, events, seconds = replay_calls(learned, seed=4, rate=0.0)
    print(f"  no habit: {len(learned)} habit(s) learned, {len(cues)} calls "
          f"in {seconds / 60:.1f} minutes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
