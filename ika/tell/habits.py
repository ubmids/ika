"""Mining a stream for habits, and refusing to believe most of what it finds.

Counting conditional frequencies is the easy half. `P(drop_guard | jab, jab)`
is arithmetic. The hard half is that a stream of any length contains thousands
of candidate contexts, and in *random* data some of them will look strongly
predictive purely by chance. A miner without a null model is a machine for
inventing tells, which is worse than useless: it would hand a fighter a
confident read of a habit their opponent does not have.

So two defences, and they are the substance of this module.

**A test per candidate.** How likely is a run of `support` observations
producing `hits` of an outcome whose base rate is `base`, if nothing were going
on? That is a binomial tail, computed exactly in log space.

**A correction for having looked so many times.** Testing 2,000 contexts at
p<0.05 yields about 100 false finds by construction. Benjamini-Hochberg
controls the *proportion* of reported findings expected to be false, which is
the quantity that matters when the output is a list someone will act on.

`ika tell --null` measures the residual false-discovery rate on fighters with
no habits at all, and that number is the honest headline for this module.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Finding:
    """A habit the miner believes in."""

    context: tuple[str, ...]
    then: str
    support: int          # times the context occurred
    hits: int             # times `then` followed it
    probability: float    # hits / support
    base: float           # unconditional rate of `then`
    p_value: float

    @property
    def lift(self) -> float:
        """How many times more likely than chance. The size of the tell."""
        return self.probability / self.base if self.base > 0 else float("inf")

    def describe(self) -> str:
        context = " then ".join(self.context) if self.context else "anything"
        return (
            f"after {context}: {self.then} "
            f"{self.probability:.0%} of the time (normally {self.base:.0%}), "
            f"{self.lift:.1f}x, seen {self.hits}/{self.support}"
        )


def _log_binomial_tail(hits: int, support: int, base: float) -> float:
    """log P(X >= hits) for X ~ Binomial(support, base).

    Computed in log space with lgamma rather than with `math.comb` and powers,
    because at a few thousand observations the intermediate terms overflow a
    float long before the answer does.
    """
    if base <= 0.0:
        return 0.0 if hits == 0 else -math.inf
    if base >= 1.0:
        return 0.0
    if hits <= 0:
        return 0.0

    log_base = math.log(base)
    log_rest = math.log1p(-base)
    terms = []
    for i in range(hits, support + 1):
        terms.append(
            math.lgamma(support + 1) - math.lgamma(i + 1) - math.lgamma(support - i + 1)
            + i * log_base + (support - i) * log_rest
        )
    peak = max(terms)
    return peak + math.log(sum(math.exp(t - peak) for t in terms))


def binomial_tail(hits: int, support: int, base: float) -> float:
    """P(X >= hits) for X ~ Binomial(support, base). The per-candidate test."""
    return float(min(1.0, math.exp(_log_binomial_tail(hits, support, base))))


def benjamini_hochberg(p_values: list[float], fdr: float = 0.05) -> list[bool]:
    """Which findings survive, controlling the false-discovery *proportion*.

    Bonferroni would control the chance of any false find at all, which is far
    too strict here: it would throw away real habits to avoid a single mistake.
    What a fighter actually wants is "most of what you tell me is true", and
    that is the false-discovery rate.
    """
    count = len(p_values)
    if count == 0:
        return []
    order = np.argsort(p_values)
    keep = np.zeros(count, dtype=bool)
    threshold_index = -1
    for rank, index in enumerate(order, start=1):
        if p_values[index] <= fdr * rank / count:
            threshold_index = rank
    if threshold_index > 0:
        keep[order[:threshold_index]] = True
    return keep.tolist()


def mine(
    stream: list[str],
    max_context: int = 3,
    min_support: int = 12,
    min_lift: float = 1.35,
    fdr: float = 0.05,
) -> list[Finding]:
    """Find the habits in a sequence of action names.

    `min_support` exists because a context seen three times can show a
    100% rate and mean nothing; the statistics need something to work with.
    `min_lift` filters findings that are real but too small to act on: a
    tendency 1.05x above chance is not a tell even when the p-value is tiny,
    and at long stream lengths plenty of those become significant.
    """
    if len(stream) < min_support + max_context:
        return []

    total = len(stream)
    counts: dict[str, int] = defaultdict(int)
    for name in stream:
        counts[name] += 1
    base = {name: n / total for name, n in counts.items()}

    # context -> successor -> count
    followers: dict[tuple[str, ...], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for length in range(1, max_context + 1):
        for i in range(total - length):
            context = tuple(stream[i : i + length])
            followers[context][stream[i + length]] += 1

    candidates: list[Finding] = []
    for context, successors in followers.items():
        support = sum(successors.values())
        if support < min_support:
            continue
        for successor, hits in successors.items():
            rate = hits / support
            reference = base.get(successor, 0.0)
            if reference <= 0.0 or rate / reference < min_lift:
                continue
            candidates.append(
                Finding(
                    context=context,
                    then=successor,
                    support=support,
                    hits=hits,
                    probability=rate,
                    base=reference,
                    p_value=binomial_tail(hits, support, reference),
                )
            )

    survivors = benjamini_hochberg([c.p_value for c in candidates], fdr=fdr)
    kept = [c for c, ok in zip(candidates, survivors) if ok]

    return _minimal(kept)


def _is_suffix(shorter: tuple[str, ...], longer: tuple[str, ...]) -> bool:
    return len(shorter) < len(longer) and longer[-len(shorter):] == shorter


def _minimal(kept: list[Finding], alpha: float = 0.05) -> list[Finding]:
    """Reduce a pile of overlapping findings to the habits actually present.

    One planted habit generates a family of findings, and reporting the family
    as separate reads would badly mislead whoever acts on it. Planting a single
    "after jab, jab: drop guard" produced four: the true one, two with
    irrelevant prefixes ("after cross, jab, jab"), and one diluted echo
    ("after jab", at a third of the strength).

    Both directions are junk, for different reasons, and each needs its own
    test rather than a fudge factor:

    **Longer than necessary.** "after cross, jab, jab" is kept only if its rate
    is significantly above what its own suffix "after jab, jab" already
    predicts. Tested as a binomial against the suffix's rate, not against the
    global base rate: the question is whether the extra context adds anything.

    **Shorter than necessary.** "after jab" looks predictive only because some
    of those jabs were the first of a pair. It is dropped when a longer kept
    context ends with it and predicts the same outcome far more strongly, since
    the short version is then a diluted view of the long one.
    """
    by_outcome: dict[str, list[Finding]] = defaultdict(list)
    for finding in kept:
        by_outcome[finding.then].append(finding)

    final: list[Finding] = []
    for findings in by_outcome.values():
        findings.sort(key=lambda f: len(f.context))

        survivors: list[Finding] = []
        for finding in findings:
            suffixes = [o for o in survivors if _is_suffix(o.context, finding.context)]
            if suffixes:
                best = max(suffixes, key=lambda o: o.probability)
                # Does the longer context beat its suffix by more than chance?
                if binomial_tail(finding.hits, finding.support, best.probability) > alpha:
                    continue
            survivors.append(finding)

        for finding in survivors:
            longer = [
                o for o in survivors
                if _is_suffix(finding.context, o.context)
                and o.probability > finding.probability * 1.5
            ]
            if not longer:
                final.append(finding)

    return sorted(final, key=lambda f: -f.lift)
