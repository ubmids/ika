"""Fighters with habits, invented so the miner can be graded.

The point is the same as `synth` was for hand poses: a labelled problem where
the right answer is known before anything runs. Here the answer is "this
fighter drops their guard after two jabs, 70% of the time", and the miner
either finds that and nothing else, or it does not.

Timing is not decoration. A habit is only worth anything if the warning
arrives before the punch, so every action carries a real start and end, drawn
from durations a human would actually take.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Action:
    name: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class Habit:
    """A tendency: after `context`, do `then` with probability `probability`."""

    context: tuple[str, ...]
    then: str
    probability: float


# A small boxing-shaped vocabulary with plausible durations in seconds. Strikes
# are fast, footwork is slower, and a dropped guard is a lapse rather than an
# act, which is why it is the classic thing to punish.
DURATIONS: dict[str, tuple[float, float]] = {
    "jab": (0.12, 0.22),
    "cross": (0.16, 0.28),
    "hook": (0.18, 0.30),
    "uppercut": (0.18, 0.32),
    "low_kick": (0.30, 0.48),
    "step_in": (0.22, 0.40),
    "step_back": (0.22, 0.40),
    "step_left": (0.24, 0.42),
    "step_right": (0.24, 0.42),
    "guard_up": (0.15, 0.28),
    "drop_guard": (0.20, 0.40),
    "feint": (0.14, 0.26),
}

VOCABULARY = tuple(DURATIONS)

# What a fighter throws when no habit is driving them. Deliberately uneven,
# because a uniform base rate would make any elevated probability trivially
# visible and flatter the miner.
BASE_WEIGHTS: dict[str, float] = {
    "jab": 4.0, "cross": 2.5, "hook": 1.6, "uppercut": 0.9, "low_kick": 0.8,
    "step_in": 1.6, "step_back": 1.7, "step_left": 1.3, "step_right": 1.3,
    "guard_up": 1.5, "feint": 1.2, "drop_guard": 0.7,
}


@dataclass
class Fighter:
    """Emits a stream of actions, with optional planted habits.

    Habits are matched longest-context-first, so a specific tendency wins over
    a general one, the way a real read would.
    """

    habits: tuple[Habit, ...] = ()
    weights: dict[str, float] = field(default_factory=lambda: dict(BASE_WEIGHTS))
    gap: tuple[float, float] = (0.05, 0.45)   # rest between actions
    seed: int = 0

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        names = list(self.weights)
        total = sum(self.weights.values())
        self._names = names
        self._probabilities = np.array([self.weights[n] / total for n in names])
        self._by_length = sorted(self.habits, key=lambda h: -len(h.context))

    def base_rate(self, name: str) -> float:
        """The probability of `name` absent any habit. The thing a lift is over."""
        return float(self._probabilities[self._names.index(name)])

    def _next_name(self, history: list[str]) -> str:
        for habit in self._by_length:
            k = len(habit.context)
            if k and len(history) >= k and tuple(history[-k:]) == habit.context:
                if self._rng.random() < habit.probability:
                    return habit.then
                break   # the habit applied and did not fire; do not try shorter ones
        return str(self._rng.choice(self._names, p=self._probabilities))

    def sequence(self, count: int, start: float = 0.0) -> list[Action]:
        history: list[str] = []
        out: list[Action] = []
        clock = start
        for _ in range(count):
            name = self._next_name(history)
            low, high = DURATIONS.get(name, (0.2, 0.35))
            duration = float(self._rng.uniform(low, high))
            out.append(Action(name, clock, clock + duration))
            clock += duration + float(self._rng.uniform(*self.gap))
            history.append(name)
        return out


def names(stream: list[Action]) -> list[str]:
    return [a.name for a in stream]
