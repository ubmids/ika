"""`ika history`, laid out to be read rather than parsed.

Coach and drill are kept apart all the way down, because they are different
evidence: a coach habit is a guard read after a combo ika called, a drill
habit is read off what the punch reader thought you threw. Each habit gets
one row: a sparkline of its rate in every session of its own mode, the last
session's count, and a label.

The labels are the profile's own decision and nothing new. `Profile.habits`
is what decides a habit stands (enough sessions, enough evidence left after
the two-week decay, a lift over chance that survives the correction for
everything looked at), and the labels only ask it at different times:

- **holding**: it stands now.
- **fading**: it stood after an earlier session and no longer does. The
  evidence behind it has decayed, or later sessions diluted it.
- **new**: seen in only one session so far, and a habit needs two
  (`profile.MIN_SESSIONS`); a coach habit needs a second session after the
  one that proposed it.
- **not yet**: seen in more sessions than that, never with enough behind it
  to stand.
"""

from __future__ import annotations

import time
from collections import defaultdict

from .profile import MIN_SESSIONS, Profile, mode_of

SPARKS = "▁▂▃▄▅▆▇█"
MISSING = "·"

LABELS = ("holding", "fading", "new", "not yet")


def spark(points) -> str:
    """One character per session: the rate as a bar, "·" where the setup never came up."""
    out = []
    for _at, hits, support in points:
        if not support:
            out.append(MISSING)
        else:
            out.append(SPARKS[min(len(SPARKS) - 1, int(hits / support * len(SPARKS)))])
    return "".join(out)


def _key(context, then) -> str:
    return " then ".join(context) + " -> " + then


def labels(record: Profile, now: float) -> dict[str, str]:
    """Each habit on record, keyed as in the log, to its label."""
    holding = {_key(h.context, h.then) for h in record.habits(now=now)}
    times = sorted({o.at for o in record.observations})
    stood: set[str] = set()
    for t in times:
        earlier = Profile(record.subject,
                          observations=[o for o in record.observations if o.at <= t])
        stood |= {_key(h.context, h.then) for h in earlier.habits(now=t)}
    sessions: dict[str, set[float]] = defaultdict(set)
    for o in record.observations:
        sessions[_key(o.context, o.then)].add(o.at)

    out = {}
    for key, seen in sessions.items():
        needed = MIN_SESSIONS + (1 if key.startswith("coach_") else 0)
        if key in holding:
            out[key] = "holding"
        elif key in stood:
            out[key] = "fading"
        elif len(seen) < needed:
            out[key] = "new"
        else:
            out[key] = "not yet"
    return out


def _hand(then: str) -> str:
    side = then.rsplit("_", 1)[-1]
    return "hands down" if side == "both" else f"{side} hand down"


def describe(key: str) -> str:
    """A habit key in words: "after right, left -> right hand down"."""
    context, _, then = key.partition(" -> ")
    if context.startswith("coach_"):
        return f"called {context[len('coach_'):]} -> {_hand(then)}"
    setup = ", ".join(c.rsplit("_", 1)[-1] for c in context.split(" then ")) if context else "anything"
    outcome = _hand(then) if then.startswith("guard_down_") else then.rsplit("_", 1)[-1]
    return f"after {setup} -> {outcome}"


def _when(at: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(at))


def _day(at: float) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(at))


def _session_line(row: dict) -> str:
    counts = row["counts"]
    minutes = f"{row['seconds'] / 60:4.1f} min"
    if mode_of(row) == "coach":
        return (f"    {_when(row['at'])}  {minutes}  coach  {counts.get('calls', 0):4d} calls  "
                f"{counts.get('answered', 0):4d} answered")
    punches = sum(v for k, v in counts.items() if k.startswith("punch_"))
    drops = sum(v for k, v in counts.items() if k.startswith("guard_down_"))
    return (f"    {_when(row['at'])}  {minutes}  drill  {punches:4d} punches "
            f"{drops:3d} guard drops")


EMPTY = """
  Nothing on record for {subject} yet.

    ika coach    ika calls combos and reads your guard after each one.
                 The one to start with: it knows what you threw.
    ika drill    you box freely; ika reads what you throw and finds the habit.

  Every round is saved, and this shows each habit session by session.
"""


def render(record: Profile | None, subject: str, now: float | None = None,
           last: int = 12) -> list[str]:
    """`ika history` as lines of text."""
    if record is None or not record.log:
        return EMPTY.format(subject=subject).split("\n")
    now = time.time() if now is None else now
    out: list[str] = []
    log = record.log
    modes = {"coach": [r for r in log if mode_of(r) == "coach"],
             "drill": [r for r in log if mode_of(r) == "drill"]}
    out.append(f"  {subject}: {len(log)} session(s), {len(modes['coach'])} coach and "
               f"{len(modes['drill'])} drill, {_day(log[0]['at'])} to {_day(log[-1]['at'])}")

    label = labels(record, now)
    series = dict(record.history())
    groups = {
        "coach": [("after a called combo", lambda k: k.startswith("coach_"))],
        "drill": [("where your hands go",
                   lambda k: not k.startswith("coach_")
                   and k.partition(" -> ")[2].startswith("guard")),
                  ("punch patterns",
                   lambda k: not k.startswith("coach_")
                   and not k.partition(" -> ")[2].startswith("guard"))],
    }
    titles = {"coach": "coach   ika called the combos",
              "drill": "drill   ika read what you threw"}

    for mode in ("coach", "drill"):
        rows = modes[mode]
        if not rows:
            continue
        out += ["", f"  {titles[mode]}"]
        shown = rows[-last:]
        if len(rows) > len(shown):
            out.append(f"    ... {len(rows) - len(shown)} earlier")
        out += [_session_line(r) for r in shown]
        times = [r["at"] for r in shown]
        for heading, wanted in groups[mode]:
            keys = sorted(k for k in series if wanted(k))
            if not keys:
                continue
            width = max(len(describe(k)) for k in keys)
            out += ["", f"    {heading}".ljust(width + 8)
                    + "each session".ljust(max(len(times), 12) + 2) + "last"]
            for key in sorted(keys, key=lambda k: (LABELS.index(label.get(k, "not yet")), k)):
                by_time = {at: (h, s) for at, h, s in series[key]}
                points = [(t, *by_time.get(t, (0, 0))) for t in times]
                latest = next(((h, s) for _t, h, s in reversed(points) if s), None)
                tail = f"{latest[0] / latest[1]:4.0%} {latest[0]:>3d}/{latest[1]:<3d}" \
                    if latest else "   -        "
                out.append(f"      {describe(key):{width}s}  "
                           f"{spark(points).ljust(max(len(times), 12))}  {tail}  "
                           f"{label.get(key, '')}")
    if not series:
        out += ["", "  no habit has been found yet. A habit needs two sessions behind it."]

    standing = record.habits(now=now)
    if standing:
        from .coach import combo_of, is_coach

        out += ["", "  standing now"]
        for habit in standing[:6]:
            text = habit.describe(now=now)
            if is_coach(habit.context):
                text += f"  (the call for {combo_of(habit.context)} now ends with a reminder)"
            out.append(f"    {text}")
    out += ["", "  holding: stands on the pooled evidence now.  fading: stood, and no longer does.",
            "  new: too few sessions to stand yet (two for a drill habit, three for coach).",
            f"  not yet: seen more than that, never with enough behind it.  "
            f"{MISSING}: the setup never came up."]
    return out
