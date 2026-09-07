"""From noisy per-frame guesses to things you actually meant.

Classification accuracy is not what makes hand control work. This is.

A classifier fires an opinion every frame, thirty times a second, and it is
wrong sometimes. Wire that straight to your keyboard and reaching for a coffee
sends four keystrokes. Worse, the failure is not occasional: your hands are in
frame constantly, doing things that mean nothing, and every one of them gets
classified as *something*.

Four mechanisms, each earning its place:

**Smoothing.** Probabilities are averaged over recent frames, so one bad frame
cannot fire anything on its own.

**Dwell.** A gesture must persist for several consecutive frames before it
counts. This is what separates holding a shape from passing through it, and
passing through is what a hand does on the way to somewhere else.

**Hysteresis.** After firing, that gesture is spent until something neutral is
seen. Otherwise holding a pose repeats it forever at frame rate.

**Engagement.** Nothing fires at all until you deliberately arm the system, and
it disarms itself when your hands go quiet. This is the difference between a
computer you control and one that flinches at you.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Intent:
    """Something the user meant, as opposed to something the model saw."""

    kind: str        # "engaged", "disengaged", "fired"
    gesture: str
    confidence: float
    at: float


@dataclass
class GestureMachine:
    classes: list[str]
    threshold: float = 0.80          # below this, no gesture is a candidate
    dwell_frames: int = 5            # frames a gesture must hold to count
    cooldown: float = 0.50           # seconds between firings
    smoothing: float = 0.6           # weight on history, 0 disables
    engage_gesture: str | None = "open_palm"
    neutral: tuple[str, ...] = ("rest", "none")
    disengage_after: float = 6.0     # seconds of neutral before disarming

    engaged: bool = field(default=False, init=False)
    smoothed: np.ndarray | None = field(default=None, init=False)
    _candidate: str | None = field(default=None, init=False)
    _streak: int = field(default=0, init=False)
    _spent: str | None = field(default=None, init=False)
    _last_fire: float = field(default=-1e9, init=False)
    _last_active: float = field(default=0.0, init=False)

    def reset(self) -> None:
        self.engaged = False
        self.smoothed = None
        self._candidate = None
        self._streak = 0
        self._spent = None
        self._last_fire = -1e9

    @property
    def progress(self) -> float:
        """How far the current gesture is toward firing, for the on-screen meter.

        Showing this is not decoration. Without it, a gesture that is nearly
        recognised is indistinguishable from one being ignored, and the user
        has no way to learn what the system wants.
        """
        return min(1.0, self._streak / self.dwell_frames) if self._candidate else 0.0

    @property
    def candidate(self) -> str | None:
        return self._candidate

    def update(self, probabilities: np.ndarray | None, now: float) -> list[Intent]:
        """Feed one frame's distribution. Returns any intents it produced.

        Pass `None` when no hand is visible, which is information rather than
        an absence: a vanished hand should clear the streak, not freeze it.
        """
        if probabilities is None:
            self._candidate, self._streak = None, 0
            if self.engaged and now - self._last_active > self.disengage_after:
                self.engaged = False
                return [Intent("disengaged", "", 0.0, now)]
            return []

        probabilities = np.asarray(probabilities, dtype=float)
        if self.smoothed is None or len(self.smoothed) != len(probabilities):
            self.smoothed = probabilities.copy()
        else:
            k = self.smoothing
            self.smoothed = k * self.smoothed + (1.0 - k) * probabilities

        best = int(self.smoothed.argmax())
        name, confidence = self.classes[best], float(self.smoothed[best])

        # Only a confident, non-neutral reading can be a candidate. Neutral
        # poses are what clears hysteresis, so they must never fire.
        if confidence < self.threshold or name in self.neutral:
            if name in self.neutral and confidence >= self.threshold:
                self._spent = None       # hand returned to rest, so re-arm
            else:
                self._last_active = now  # uncertain, but something is happening
            self._candidate, self._streak = None, 0
            if self.engaged and now - self._last_active > self.disengage_after:
                self.engaged = False
                return [Intent("disengaged", "", 0.0, now)]
            return []

        self._last_active = now
        if name == self._candidate:
            self._streak += 1
        else:
            self._candidate, self._streak = name, 1

        if self._streak < self.dwell_frames:
            return []

        if not self.engaged:
            # Disarmed: the only gesture that means anything is the one that arms.
            if self.engage_gesture is None or name == self.engage_gesture:
                self.engaged = True
                self._streak = 0
                self._spent = name
                self._last_fire = now
                return [Intent("engaged", name, confidence, now)]
            return []

        if name == self._spent or now - self._last_fire < self.cooldown:
            return []

        self._spent = name
        self._last_fire = now
        self._streak = 0
        return [Intent("fired", name, confidence, now)]
