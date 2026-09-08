"""A habit record that survives the process, and forgets on purpose.

`tell.habits` mines one stream and then the answer dies with the interpreter.
Two things are wrong with that, and this module exists for both.

**One session is thin evidence.** The miner finds a planted habit at 11.5x
lift on a long stream, but it also reports a habit that does not exist in 15%
of fighters that have none. A tendency seen four times in one session and a
tendency seen twenty times across five sessions are not the same claim, and
throwing the record away at exit forces every session to make the strong claim
from the weak evidence. Pooling across sessions is how a quiet but consistent
tendency earns significance, and how a one-off coincidence never does.

**The product invalidates its own findings.** Tell a fighter they drop their
guard after a jab and they stop doing it. A habit that was true last month and
is false today is worse than no habit at all, because it is a confident read
that walks its owner into a counter. So evidence is weighted by age: every
observation halves in weight each half life, and a habit that stops being
observed loses its standing without anyone having to retract it.

The decay is the whole point. Nothing else here models the fact that being
right about a person changes them.

**What this cannot see.** `add` is fed the miner's findings, and the miner only
reports what it believes. A session where the context occurred and the outcome
did not follow produces no finding at all, so there is no negative evidence to
record: absence arrives as silence, not as a counter-count. That is why fading
has to be driven by the age of the standing evidence rather than by contrary
evidence, and why the session gates below are stated in terms of how much
un-decayed evidence is still behind a claim.

Sessions propose, the profile decides. `propose` mines a session at a
deliberately permissive false-discovery rate so that a sub-threshold tendency
still reaches the record, and the gates in `habits` do the real deciding on the
pooled evidence. Miner-strict per session plus pooling would be strictly worse:
a habit that never clears the bar alone can never clear it by repetition.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .tell.habits import Finding, benjamini_hochberg, binomial_tail, mine

SECONDS_PER_DAY = 86400.0

# Two weeks. Chosen against the coaching loop rather than any statistic: a
# fighter who is told about a habit acts on it within a session or two, so a
# month-old observation should be carrying a fraction of the weight of
# yesterday's. Longer and the record quotes habits that have been coached out;
# much shorter and five sessions of consistent evidence never accumulate.
DEFAULT_HALF_LIFE = 14.0 * SECONDS_PER_DAY

# A habit must have been seen in at least this many separate sessions. This is
# the guard on the miner's measured 15% false-find rate: an invented finding
# comes from the noise of one particular stream, and the odds of the same
# invented context and outcome recurring in an independent session are small.
# Without this a brand new subject's first thin session would be reported as a
# read 15% of the time.
MIN_SESSIONS = 2

# The same requirement again, but after decay: a session and a half of weight
# must still stand behind the claim. This is what makes a habit fade. When the
# tendency stops happening no new observations arrive, the standing ones decay,
# and the sum of their weights falls through this floor. A habit backed by
# twenty sessions survives a quiet month; one backed by two does not, which is
# the correct asymmetry. Set below 2 rather than at it so that two sessions a
# day apart still count as two sessions: at 2.0 exactly, the second session of
# a real habit is held back by a rounding-scale amount of decay.
MIN_EFFECTIVE_SESSIONS = 1.5

# Mirrors `habits.mine`'s own `min_support`. A context with less evidence than
# this has nothing for the binomial test to work with, whether it is thin
# because it was never seen or thin because what was seen has aged out.
MIN_WEIGHTED_SUPPORT = 12.0

# Mirrors `habits.mine`'s `min_lift`. A pooled tendency 1.05x above chance is
# not a tell however tight its p-value, and pooling makes tight p-values easy.
MIN_LIFT = 1.35

# The false-discovery rate the pooled findings are held to. Same value the
# miner uses, applied to a different and better-supported set of candidates.
DEFAULT_FDR = 0.05

# What a session is allowed to propose. Looser than the decision but nowhere
# near loose: this is the one knob that can quietly wreck the record, because
# pooling breaks the null model the miner is tested against. A session only
# reports its extremes, so pooling those extremes pools cherry-picked tails,
# and the binomial test in `habits` has no way to know it is looking at a
# selected sample. Measured over 20 null profiles of five 3,000-action
# sessions: at 0.2 a null session proposes 0.5 junk candidates and 0 of 20
# profiles report a habit; at 0.5 it proposes about 30, junk contexts start
# colliding across sessions, and 18 of 20 null profiles report habits at
# 2x to 3x lift. Anything above about 0.35 is unusable for that reason.
PROPOSAL_FDR = 0.2

# Past this many half lives an observation carries under 2% of its original
# weight, far too little on its own to lift a pooled count over
# MIN_WEIGHTED_SUPPORT or a session sum over MIN_EFFECTIVE_SESSIONS. Dropping
# the tail is not exactly free: measured on a 30 session record it moved a
# pooled count from 65/82 to 64/81. That is the accepted error, and a bounded
# file is worth a trial.
FORGETTABLE_AFTER_HALF_LIVES = 6.0


@dataclass
class Observation:
    """One context to outcome count, from one session, at a time.

    `base` is the unconditional rate of `then` measured in the session the
    observation came from. It is kept per observation rather than per profile
    so that pooling, forgetting and the JSON file each stay correct on their
    own: after `forget_before` drops half the record, the remaining rows still
    carry everything needed to recompute significance. It has a default so the
    five-field constructor still works.
    """

    context: tuple[str, ...]
    then: str
    hits: int
    support: int
    at: float                 # unix seconds
    base: float = 0.0

    def weight(self, now: float, half_life: float) -> float:
        """How much this observation still counts for. Halves every half life."""
        age = max(0.0, now - self.at)
        return 0.5 ** (age / half_life)


@dataclass(frozen=True)
class Standing:
    """A habit that still stands, on pooled and decayed evidence.

    Deliberately not a `habits.Finding`. A finding is a claim about one stream;
    this is a claim about a person over time, and the difference is exactly the
    three fields a finding does not have: how many sessions are behind it, how
    much of that evidence has aged away, and when it was last seen. Those are
    what a coach needs to know whether to still say it out loud.
    """

    context: tuple[str, ...]
    then: str
    hits: int                 # pooled, decayed, rounded to a count
    support: int
    probability: float
    base: float
    p_value: float
    sessions: int             # separate sessions that contributed
    effective_sessions: float # those sessions after decay
    last_seen: float          # unix seconds of the most recent observation

    @property
    def lift(self) -> float:
        return self.probability / self.base if self.base > 0 else float("inf")

    def age(self, now: float) -> float:
        """Seconds since the tendency was last observed."""
        return max(0.0, now - self.last_seen)

    def describe(self, now: float | None = None) -> str:
        context = " then ".join(self.context) if self.context else "anything"
        text = (
            f"after {context}: {self.then} "
            f"{self.probability:.0%} of the time (normally {self.base:.0%}), "
            f"{self.lift:.1f}x, {self.hits}/{self.support} across "
            f"{self.sessions} sessions ({self.effective_sessions:.1f} after decay)"
        )
        if now is not None:
            text += f", last seen {self.age(now) / SECONDS_PER_DAY:.1f} days ago"
        return text


def propose(stream: list[str], fdr: float = PROPOSAL_FDR, **kwargs) -> list[Finding]:
    """Mine one session for candidates to put on the record.

    A thin wrapper on `habits.mine` with the false-discovery rate relaxed,
    because a session is a shortlist and not a verdict. Mining each session at
    the strict 5% would make pooling pointless: a tendency that cannot clear
    the bar on one stream would never be recorded, so it could never clear it
    by being consistent, which is the one thing this module is for.
    """
    return mine(stream, fdr=fdr, **kwargs)


@dataclass
class Profile:
    """The accumulated, decaying habit record of one subject."""

    subject: str
    observations: list[Observation] = field(default_factory=list)

    # --- recording --------------------------------------------------------

    def add(self, findings, at: float) -> None:
        """Record a session's findings, stamped with when the session happened.

        Findings are stored as raw counts, not as the miner's verdict. The
        p-value a finding arrives with was computed against one stream and is
        the wrong number once the evidence is pooled, so it is dropped here and
        recomputed in `habits` on everything that stands.
        """
        for finding in findings:
            self.observations.append(
                Observation(
                    context=tuple(finding.context),
                    then=finding.then,
                    hits=int(finding.hits),
                    support=int(finding.support),
                    at=float(at),
                    base=float(finding.base),
                )
            )

    def sessions(self) -> int:
        """Separate sessions on the record. Sessions share a timestamp."""
        return len({observation.at for observation in self.observations})

    # --- reading ----------------------------------------------------------

    def habits(
        self,
        now: float,
        half_life: float = DEFAULT_HALF_LIFE,
        fdr: float = DEFAULT_FDR,
    ) -> list[Standing]:
        """What is still true about this subject, ranked by how big the tell is.

        Counts are pooled with an age weight, not averaged as rates. Averaging
        rates would let two-for-two today outvote eighteen-for-twenty last
        week, which is backwards: the second is far better evidence even after
        a week of decay. Weighted counts get that ordering right for free.
        """
        if half_life <= 0.0:
            # Refused rather than clamped: a caller passing this has a bug in
            # their units, and silently picking a half life for them would
            # hide it behind plausible looking output.
            raise ValueError(f"half_life must be positive, got {half_life}")

        pooled: dict[tuple[tuple[str, ...], str], list[Observation]] = defaultdict(list)
        for observation in self.observations:
            pooled[(observation.context, observation.then)].append(observation)

        candidates: list[Standing] = []
        for (context, then), group in pooled.items():
            weights = [observation.weight(now, half_life) for observation in group]
            weighted_support = sum(w * o.support for w, o in zip(weights, group))
            if weighted_support < MIN_WEIGHTED_SUPPORT:
                continue

            session_times = {o.at for o in group}
            if len(session_times) < MIN_SESSIONS:
                continue
            # Every observation from one session shares its timestamp, so a
            # session's weight is counted once however many contexts it found.
            effective_sessions = sum(
                0.5 ** (max(0.0, now - at) / half_life) for at in session_times
            )
            if effective_sessions < MIN_EFFECTIVE_SESSIONS:
                continue

            weighted_hits = sum(w * o.hits for w, o in zip(weights, group))
            base = sum(
                w * o.support * o.base for w, o in zip(weights, group)
            ) / weighted_support
            probability = weighted_hits / weighted_support
            if base <= 0.0 or probability / base < MIN_LIFT:
                continue

            hits, support = _as_counts(weighted_hits, weighted_support)
            candidates.append(
                Standing(
                    context=context,
                    then=then,
                    hits=hits,
                    support=support,
                    probability=probability,
                    base=base,
                    p_value=binomial_tail(hits, support, base),
                    sessions=len(session_times),
                    effective_sessions=effective_sessions,
                    last_seen=max(session_times),
                )
            )

        # Candidates dropped above were never tested, they were dropped for
        # having too little evidence to test, so they must not inflate the
        # correction's denominator. Same order as `habits.mine`.
        survivors = benjamini_hochberg([c.p_value for c in candidates], fdr=fdr)
        kept = [c for c, ok in zip(candidates, survivors) if ok]
        return sorted(kept, key=lambda s: (-s.lift, s.p_value))

    # --- bounding the file ------------------------------------------------

    def forget_before(self, at: float) -> None:
        """Drop everything observed before `at`. In place, and unrecoverable."""
        self.observations = [o for o in self.observations if o.at >= at]

    def prune(self, now: float, half_life: float = DEFAULT_HALF_LIFE) -> None:
        """Drop observations too old to affect any answer.

        A profile that only ever grows is a file that eventually stops being
        openable. Past six half lives an observation's weight is under 2%, too
        small to carry a pooled count over `MIN_WEIGHTED_SUPPORT` or a session
        sum over `MIN_EFFECTIVE_SESSIONS` on its own.

        It is not free, and the honest size of the cost is measured: on a 30
        session record the dropped tail was still worth about one trial in
        aggregate, moving a pooled count from 65/82 to 64/81. A trial, not a
        verdict, which is the trade a bounded file requires.
        """
        self.forget_before(now - FORGETTABLE_AFTER_HALF_LIVES * half_life)

    # --- the record on disk ------------------------------------------------

    def save(self, path) -> None:
        """Write the profile as JSON.

        JSON and not pickle because this is a record about a person that they
        or their coach may reasonably want to read, correct, or delete a line
        from, and because a pickle is a promise to execute whatever the file
        says next time it is opened.
        """
        path = Path(path)
        payload = {
            "subject": self.subject,
            "observations": [
                {
                    "context": list(o.context),
                    "then": o.then,
                    "hits": o.hits,
                    "support": o.support,
                    "at": o.at,
                    "base": o.base,
                }
                for o in self.observations
            ],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "Profile":
        """Read a profile back. Round-trips exactly, timestamps included."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            subject=payload["subject"],
            observations=[
                Observation(
                    context=tuple(row["context"]),
                    then=row["then"],
                    hits=int(row["hits"]),
                    support=int(row["support"]),
                    at=float(row["at"]),
                    base=float(row.get("base", 0.0)),
                )
                for row in payload["observations"]
            ],
        )


def _as_counts(weighted_hits: float, weighted_support: float) -> tuple[int, int]:
    """Turn decayed weights back into the counts `binomial_tail` needs.

    The test is a binomial over whole trials, so a weighted sum has to become
    an integer somewhere. Rounding down the support is the conservative
    direction: it claims less evidence than the weights suggest rather than
    more. Hits are then rounded and clamped so the rate cannot exceed one,
    which floor alone would allow at small counts.
    """
    support = int(math.floor(weighted_support))
    hits = min(support, int(round(weighted_hits)))
    return hits, support
