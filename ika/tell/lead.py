"""Lead time: how long before the move lands can we call it.

The number that decides whether any of this is a product. A read delivered
after the punch is not a read, and the gesture pipeline already showed that
recognition alone costs about 250 ms, so the margin here is not generous.

Lead time is measured from the moment the *context* is recognisable to the
moment the predicted action *starts*. Recognition is not free, so the
detection cost is charged explicitly rather than assumed away.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .actions import Action
from .habits import Finding


@dataclass(frozen=True)
class Warning_:
    at: float          # when we could have said it
    predicted: str
    happened: str
    lands_at: float
    lead: float        # seconds of notice; negative means too late

    @property
    def correct(self) -> bool:
        return self.predicted == self.happened


def lead_times(
    stream: list[Action],
    findings: list[Finding],
    detection_cost: float = 0.25,
    recognition: float = 1.0,
    vocabulary: tuple[str, ...] | None = None,
    seed: int = 0,
) -> list[Warning_]:
    """Replay a stream and record every warning a finding would have produced.

    `detection_cost` is the delay between a movement happening and the system
    knowing what it was. `recognition` is how often it gets that right.

    Those two are not independent, and pretending otherwise is the easiest way
    to produce a flattering number. Committing to a call early cuts the delay
    but gets fooled by feints; waiting for certainty costs time. `early.sweep`
    measures that trade, and this composes it with the prediction so the two
    can be judged together rather than each in isolation.

    A misrecognised action corrupts the context, so the habit lookup either
    misses entirely or matches the wrong habit, exactly as it would live.
    """
    by_context: dict[tuple[str, ...], Finding] = {}
    for finding in findings:
        # Prefer the strongest finding for a given context.
        current = by_context.get(finding.context)
        if current is None or finding.lift > current.lift:
            by_context[finding.context] = finding

    longest = max((len(c) for c in by_context), default=0)
    rng = np.random.default_rng(seed)
    vocabulary = vocabulary or tuple(sorted({a.name for a in stream}))

    # What the system *thinks* it saw, which is what the lookup actually uses.
    observed = [
        a.name if rng.random() < recognition
        else str(rng.choice([v for v in vocabulary if v != a.name]))
        for a in stream
    ]

    out: list[Warning_] = []
    for i in range(len(stream) - 1):
        for length in range(longest, 0, -1):
            if i + 1 < length:
                continue
            context = tuple(observed[i + 1 - length : i + 1])
            finding = by_context.get(context)
            if finding is None:
                continue
            ready = stream[i].end + detection_cost
            nxt = stream[i + 1]
            out.append(
                Warning_(
                    at=ready,
                    predicted=finding.then,
                    happened=nxt.name,
                    lands_at=nxt.start,
                    lead=nxt.start - ready,
                )
            )
            break
    return out


def summarise(warnings: list[Warning_]) -> dict:
    """The go/no-go numbers.

    `useful` is the one that matters: warnings that were both right and early
    enough to act on. Precision alone would hide the late ones, and lead time
    alone would hide the wrong ones.
    """
    if not warnings:
        return {"warnings": 0, "precision": 0.0, "in_time": 0.0, "useful": 0.0,
                "median_lead": 0.0}
    correct = [w for w in warnings if w.correct]
    in_time = [w for w in warnings if w.lead > 0]
    useful = [w for w in warnings if w.correct and w.lead > 0]
    leads = np.array([w.lead for w in warnings])
    return {
        "warnings": len(warnings),
        "precision": len(correct) / len(warnings),
        "in_time": len(in_time) / len(warnings),
        "useful": len(useful) / len(warnings),
        "median_lead": float(np.median(leads)),
        "median_lead_correct": float(np.median([w.lead for w in correct])) if correct else 0.0,
    }
