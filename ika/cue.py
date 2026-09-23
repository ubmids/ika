"""Saying the habit out loud, while there is still time to do something about it.

Everything before this reports a habit after the round: "you drop your right
hand after two punches". That is the read. This is the call. When the last few
things you did are the setup that usually comes before your lapse, it says so
now, before the hand has gone down, which is the one moment the information is
worth anything.

A cue fires on the setup and never on the lapse itself. The setup is punches,
which the stream has the instant they are thrown; the lapse is a hand that has
to stay down for `drill.GUARD_HOLD` before it is even counted. So the call can
only ever land early if it is keyed on what came before, and that is the whole
design.

Only lapses are called. "After a jab you throw a cross" may well be a habit,
and it is useless to shout about. What a fighter can act on in the moment is
"hands", so a cue is only armed for habits whose outcome is a guard coming
down.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass

LAPSES = ("guard_down_",)

# Never two cues closer than this. A cue every second is noise, and noise is
# what a person learns to ignore.
MIN_GAP = 2.0

WORDS = {
    "guard_down_left": "left hand",
    "guard_down_right": "right hand",
    "guard_down_both": "hands up",
}


@dataclass(frozen=True)
class Cue:
    at: float
    context: tuple[str, ...]
    then: str
    probability: float

    @property
    def words(self) -> str:
        return WORDS.get(self.then, "hands")


class Caller:
    """Watches the habit stream and calls the lapse its setup predicts."""

    def __init__(self, habits=(), min_gap: float = MIN_GAP, voice: bool = False):
        self.min_gap = float(min_gap)
        self.voice = voice and _speaker() is not None
        self._last = -1e9
        self.cues: list[Cue] = []
        self.arm(habits)

    def arm(self, habits) -> None:
        """Replace what is being watched for. Accepts findings or standing habits."""
        armed = [h for h in habits if h.then.startswith(LAPSES) and h.context]
        # Longest context first: the more specific setup is the better call.
        self._habits = sorted(armed, key=lambda h: (-len(h.context), -h.probability))

    @property
    def armed(self) -> int:
        return len(self._habits)

    def update(self, stream: list[str], at: float) -> Cue | None:
        """Call if the end of the stream is a known setup. Returns the cue.

        Pass the combo in progress (`sag.current_combo`), not the whole
        session: a setup is something inside one combo, and matching across
        the gap between combos fired calls a combo early, before the real one.
        """
        if at - self._last < self.min_gap:
            return None
        for habit in self._habits:
            n = len(habit.context)
            if len(stream) >= n and tuple(stream[-n:]) == tuple(habit.context):
                cue = Cue(at, tuple(habit.context), habit.then, habit.probability)
                self._last = at
                self.cues.append(cue)
                if self.voice:
                    _say(cue.words)
                return cue
        return None


def _speaker() -> str | None:
    return shutil.which("say") if sys.platform == "darwin" else None


def _say(words: str) -> None:
    """Speak without waiting. A blocked loop would drop the frames it is reading."""
    speaker = _speaker()
    if speaker is None:
        print("\a", end="", flush=True)
        return
    subprocess.Popen([speaker, "-r", "240", words],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def measure(cues: list[Cue], events, window: float = 1.5) -> dict:
    """How many cues were followed by the lapse they called, and how early.

    `events` are the drill's events, with times. A cue is right when the lapse
    it named starts within `window` seconds after it; the lead is how long
    before the hand actually went down it was said.
    """
    right, leads = 0, []
    for cue in cues:
        # A call for one hand is right if that hand dropped, alone or with the other.
        names = {cue.then, "guard_down_both"} if cue.then.startswith("guard_down_") else {cue.then}
        after = [e for e in events if e.name in names and 0.0 <= e.at - cue.at <= window]
        if after:
            right += 1
            leads.append(after[0].at - cue.at)
    return {
        "cues": len(cues),
        "right": right,
        "precision": right / len(cues) if cues else 0.0,
        "median_lead": sorted(leads)[len(leads) // 2] if leads else 0.0,
    }
