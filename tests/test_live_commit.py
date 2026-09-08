"""The live decision layer, judged on the only thing that matters: how soon.

`tell/lead.py` measured a 263 ms median gap between one action ending and the
next starting, and `machine.py` spends 250 ms of that confirming what it saw.
So these tests are not about whether a punch is recognised. They are about
whether the call arrives while it is still worth having, and they check that
against `GestureMachine` on identical input rather than against a budget
somebody picked.

Everything here runs on probability streams built with numpy: no camera, no
weights, no dataset.
"""

import numpy as np
import pytest

from ika.live_commit import EarlyCommitter
from ika.machine import GestureMachine

CLASSES = ["rest", "open_palm", "jab", "hook", "kick"]
FPS = 30.0

# The gap a call has to land inside, from lead.py. Quoted here because a
# latency assertion against a made-up number proves nothing.
GAP_BETWEEN_ACTIONS = 0.263

# What machine.py costs to confirm a gesture, also from lead.py.
CONFIRMATION_COST = 0.250


def onehot(name: str, confidence: float = 1.0) -> np.ndarray:
    p = np.full(len(CLASSES), (1.0 - confidence) / (len(CLASSES) - 1))
    p[CLASSES.index(name)] = confidence
    return p


def split(first: str, second: str, share: float = 0.45) -> np.ndarray:
    """A model that is honestly torn between two movements.

    This is what `early.py` says a model trained on feints produces during an
    ambiguous opening, and it is the stream that costs early commitment the
    most, so it needs to be easy to build.
    """
    p = np.zeros(len(CLASSES))
    p[CLASSES.index(first)] = share
    p[CLASSES.index(second)] = share
    p[CLASSES.index("rest")] = 1.0 - 2 * share
    return p


def feed(committer, name, frames, start=0.0, step=1 / FPS, confidence=1.0):
    """Hold a movement for a number of frames, collecting any calls."""
    out = []
    for i in range(frames):
        call = committer.update(onehot(name, confidence), start + i * step)
        if call is not None:
            out.append(call)
    return out


def play(committer, stream, start=0.0, step=1 / FPS):
    """Feed a prepared stream of distributions."""
    out = []
    for i, p in enumerate(stream):
        call = committer.update(p, start + i * step)
        if call is not None:
            out.append(call)
    return out


def armed_machine(**kw):
    """A GestureMachine past its engagement gate and settled back to neutral.

    The engagement handshake is not part of the comparison. It exists so a hand
    cannot drive a keyboard by accident, it is paid once, and charging it to
    the dwell layer here would flatter the committer.
    """
    m = GestureMachine(CLASSES, **kw)
    for i in range(10):
        m.update(onehot("open_palm"), i / FPS)
    assert m.engaged
    for i in range(10):
        m.update(onehot("rest"), 1.0 + i / FPS)
    return m


def frames_to_decide(stream, start=2.0, **kw):
    """Run both layers over one stream. Returns (frames, label) for each.

    Same distributions, same timestamps, same frame rate, so the only
    difference left is the decision rule.
    """
    committer, machine = EarlyCommitter(CLASSES, **kw), armed_machine()
    early = dwell = None
    for i, p in enumerate(stream):
        at = start + i / FPS
        call = committer.update(p, at)
        if call is not None and early is None:
            early = (i + 1, call.label)
        fired = [x for x in machine.update(p, at) if x.kind == "fired"]
        if fired and dwell is None:
            dwell = (i + 1, fired[0].gesture)
    return early, dwell


# --------------------------------------------------------------- the point


def test_it_commits_in_fewer_frames_than_the_dwell_machine():
    """The whole reason this module exists.

    Identical stream into both layers. The dwell layer has to see the label
    survive five consecutive frames on top of an exponential smoother that
    needs four frames just to cross its own threshold. This one needs four
    frames of evidence, full stop.
    """
    early, dwell = frames_to_decide([onehot("jab")] * 40)
    early_frames, early_label = early
    dwell_frames, dwell_label = dwell

    assert early_label == dwell_label == "jab", "both must still be right"
    assert early_frames == 4, f"committer took {early_frames} frames"
    assert dwell_frames == 8, f"GestureMachine took {dwell_frames} frames"
    assert early_frames < dwell_frames, (
        f"committer {early_frames} frames ({early_frames / FPS * 1000:.0f} ms), "
        f"GestureMachine {dwell_frames} frames "
        f"({dwell_frames / FPS * 1000:.0f} ms) at {FPS:.0f} fps"
    )


def test_the_call_lands_inside_the_gap_between_actions():
    """133 ms against a 263 ms gap leaves room for the rest of the pipeline.

    The dwell layer's 267 ms does not, which is the 37%-versus-7% result in
    lead.py restated as a single frame count.
    """
    early, dwell = frames_to_decide([onehot("jab")] * 40)
    assert early[0] / FPS < GAP_BETWEEN_ACTIONS, "a call nobody can act on"
    assert dwell[0] / FPS >= CONFIRMATION_COST, (
        "the dwell layer is expected to spend the whole confirmation budget; "
        "if it got faster, this comparison needs redoing"
    )


# --------------------------------------------------------------- committing


def test_a_confident_movement_commits_on_min_frames_of_evidence():
    committer = EarlyCommitter(CLASSES, min_frames=4)
    calls = feed(committer, "jab", 30, start=2.0)
    assert len(calls) == 1
    assert calls[0].label == "jab"
    assert calls[0].frames == 4, "committed on exactly the evidence it needed"


def test_min_frames_is_a_floor_no_matter_how_confident_the_model_is():
    """One perfect frame is still one frame, and a tracker glitch looks like
    one perfect frame."""
    committer = EarlyCommitter(CLASSES, min_frames=6)
    for i in range(5):
        assert committer.update(onehot("jab"), 2.0 + i / FPS) is None
    assert committer.update(onehot("jab"), 2.0 + 5 / FPS) is not None


def test_it_does_not_wait_for_the_movement_to_be_held():
    """The dwell rule this replaces. A movement that lasts only as long as it
    takes to read must still be readable, and holding it longer must not be
    the price of a call."""
    committer = EarlyCommitter(CLASSES, min_frames=4)
    calls = play(committer, [onehot("jab")] * 4 + [onehot("rest")] * 10, start=2.0)
    assert len(calls) == 1, "four frames of evidence was the entire movement"


def test_low_confidence_never_commits():
    """An unsure model is silent, not a vote. Staying quiet about a movement
    you cannot read is a real outcome, as early.py puts it."""
    committer = EarlyCommitter(CLASSES, threshold=0.85)
    assert feed(committer, "jab", 60, start=2.0, confidence=0.5) == []
    assert committer.watching, "still watching, just not calling"


def test_one_bad_frame_does_not_commit_to_the_wrong_call():
    """Why the window is averaged at all. A single impostor frame in an
    otherwise clean stream must not be able to carry a decision."""
    committer = EarlyCommitter(CLASSES)
    stream = [onehot("hook") if i == 2 else onehot("jab") for i in range(20)]
    calls = play(committer, stream, start=2.0)
    assert [c.label for c in calls] == ["jab"]


def test_it_never_commits_on_neutral():
    """Neutral is what ends a movement. If it could also be a call, the layer
    would fire every time the subject stopped moving."""
    committer = EarlyCommitter(CLASSES, neutral=("rest",))
    assert feed(committer, "rest", 90, start=2.0) == []
    assert not committer.watching


# --------------------------------------------------------------- one per move


def test_a_single_punch_fires_once_not_thirty_times():
    """Two seconds of the same punch is one punch. Without this the layer
    reports sixty of them and is unusable at any latency."""
    committer = EarlyCommitter(CLASSES)
    calls = feed(committer, "jab", 60, start=2.0)
    assert len(calls) == 1


def test_returning_to_neutral_lets_the_same_call_happen_again():
    committer = EarlyCommitter(CLASSES)
    assert len(feed(committer, "jab", 20, start=2.0)) == 1
    assert feed(committer, "jab", 20, start=3.0) == [], "spent until rest is seen"

    feed(committer, "rest", 10, start=4.0)
    assert not committer.watching, "rest ended the movement"
    assert len(feed(committer, "jab", 20, start=5.0)) == 1


def test_a_combo_reads_as_two_calls_without_a_pause_between_them():
    """A punch flowing straight into a kick is two events, and the second one
    must not inherit the first one's timing. Measured: 100 ms, not the 767 ms
    it took while the punch's tail was still averaged into the window."""
    committer = EarlyCommitter(CLASSES)
    calls = feed(committer, "jab", 20, start=2.0)
    calls += feed(committer, "kick", 20, start=2.0 + 20 / FPS)
    assert [c.label for c in calls] == ["jab", "kick"]
    assert all(c.frames == 4 for c in calls)
    assert all(c.latency * 1000 < GAP_BETWEEN_ACTIONS * 1000 for c in calls), (
        f"latencies {[round(c.milliseconds) for c in calls]} ms"
    )


def test_the_cooldown_holds_off_a_second_call_that_comes_too_fast():
    """Two different movements inside the cooldown are more likely one
    movement being read twice than a combo thrown in a tenth of a second."""
    committer = EarlyCommitter(CLASSES, cooldown=1.0)
    calls = feed(committer, "jab", 8, start=2.0)
    calls += feed(committer, "kick", 8, start=2.0 + 8 / FPS)
    assert [c.label for c in calls] == ["jab"], "the kick was inside the cooldown"

    later = feed(committer, "kick", 8, start=4.0)
    assert [c.label for c in later] == ["kick"]


# --------------------------------------------------------------- onset


def test_latency_is_measured_from_the_onset_of_the_movement():
    """Not from the frame that happened to commit. Latency measured from the
    commit is always zero, which is a comfortable number and a lie."""
    committer = EarlyCommitter(CLASSES, min_frames=4)
    feed(committer, "rest", 30, start=1.0)          # sitting still
    onset = 2.0
    calls = feed(committer, "jab", 10, start=onset)
    assert len(calls) == 1
    expected = (calls[0].frames - 1) / FPS          # onset frame counts as one
    assert calls[0].latency == pytest.approx(expected)
    assert calls[0].at == pytest.approx(onset + expected)


def test_watching_turns_on_at_onset_and_off_at_rest():
    """What the onset detector is for, and what `latency` is measured against."""
    committer = EarlyCommitter(CLASSES)
    feed(committer, "rest", 10, start=1.0)
    assert not committer.watching
    committer.update(onehot("jab"), 2.0)
    assert committer.watching, "the distribution left neutral"
    feed(committer, "rest", 3, start=3.0)
    assert not committer.watching


def test_an_uncertain_movement_still_counts_as_a_movement():
    """Onset is decided on neutral mass, not on the argmax. A distribution
    spread over two movements has an argmax that flickers between them, and a
    subject who is clearly not at rest is clearly not at rest."""
    committer = EarlyCommitter(CLASSES)
    committer.update(split("jab", "hook"), 2.0)
    assert committer.watching
    assert committer.progress > 0.0


# --------------------------------------------------------------- lost subject


def test_a_vanished_subject_clears_the_evidence():
    """Coming back after a dropout must not fire a trigger primed before it.
    Frames from either side of a gap are evidence about different movements."""
    committer = EarlyCommitter(CLASSES, min_frames=6)
    feed(committer, "jab", 5, start=2.0)            # one frame short
    assert committer.evidence_frames == 5
    assert committer.update(None, 2.2) is None
    assert not committer.watching and committer.progress == 0.0
    assert feed(committer, "jab", 5, start=2.3) == [], "evidence restarted at zero"


def test_a_dropout_longer_than_the_cooldown_re_arms_the_last_call():
    """A subject who left frame and came back throwing the same punch again is
    throwing a second punch. A single dropped frame is not, or a flickering
    tracker would re-call the movement still being held."""
    committer = EarlyCommitter(CLASSES, cooldown=0.4)
    assert len(feed(committer, "jab", 10, start=2.0)) == 1
    for i in range(30):                             # one second out of frame
        committer.update(None, 3.0 + i / FPS)
    assert len(feed(committer, "jab", 10, start=5.0)) == 1


def test_a_single_dropped_frame_does_not_re_arm_it():
    committer = EarlyCommitter(CLASSES, cooldown=0.4)
    assert len(feed(committer, "jab", 10, start=2.0)) == 1
    committer.update(None, 2.4)
    assert feed(committer, "jab", 10, start=2.45) == [], "still the same punch"


# --------------------------------------------------------------- meter, state


def test_progress_climbs_to_the_call_then_resets():
    """The on-screen meter. Without it, a movement one frame short of a call
    is indistinguishable from one being ignored, and nobody can learn the
    timing of a system that gives no feedback."""
    committer = EarlyCommitter(CLASSES, min_frames=10)
    seen, calls = [], []
    for i in range(40):
        call = committer.update(onehot("jab"), 2.0 + i / FPS)
        if call is not None:
            calls.append(call)
            break
        seen.append(committer.progress)
    assert seen == sorted(seen), "the meter must only ever climb"
    assert 0 < max(seen) < 1.0
    assert len(calls) == 1
    assert committer.progress == 0.0, "resets after committing"


def test_the_meter_reports_whichever_constraint_is_binding():
    """Frames when the model is sure and early, confidence when it is neither.
    A meter that only tracked frames would sit at full while nothing happened."""
    sure = EarlyCommitter(CLASSES, min_frames=8, threshold=0.85)
    feed(sure, "jab", 4, start=2.0)
    assert sure.progress == pytest.approx(4 / 8), "short of frames, not confidence"

    unsure = EarlyCommitter(CLASSES, min_frames=4, threshold=0.9)
    feed(unsure, "jab", 4, start=2.0, confidence=0.45)
    assert unsure.progress == pytest.approx(0.45 / 0.9), "short of confidence"


def test_every_call_records_the_evidence_it_was_made_on():
    """The measurement hook. A latency claim with no frame count behind it
    cannot be checked, and this layer's entire argument is a frame count."""
    committer = EarlyCommitter(CLASSES)
    feed(committer, "jab", 10, start=2.0)
    feed(committer, "rest", 5, start=3.0)
    feed(committer, "kick", 10, start=4.0)
    assert [c.label for c in committer.calls] == ["jab", "kick"]
    assert all(c.frames >= committer.min_frames for c in committer.calls)
    assert all(c.milliseconds == pytest.approx(c.latency * 1000)
               for c in committer.calls)


def test_reset_clears_everything():
    committer = EarlyCommitter(CLASSES)
    feed(committer, "jab", 10, start=2.0)
    committer.reset()
    assert committer.calls == []
    assert not committer.watching and committer.progress == 0.0
    assert committer.evidence_frames == 0
    assert len(feed(committer, "jab", 10, start=2.1)) == 1, "re-armed"


# --------------------------------------------------------------- the cost


def test_an_honestly_ambiguous_opening_delays_the_call_instead_of_guessing():
    """The price of the averaged window, measured rather than asserted away.

    Ten frames of a model genuinely torn between two movements stay in the
    window, so the call arrives at frame 16 (533 ms) instead of frame 4. That
    is outside the 263 ms gap, and it is the correct answer arriving late
    rather than a wrong answer arriving on time.
    """
    stream = [split("jab", "hook")] * 10 + [onehot("hook")] * 40
    early, dwell = frames_to_decide(stream)
    assert early == (16, "hook"), f"committer {early}"
    assert dwell == (16, "hook"), f"GestureMachine {dwell}"
    assert early[0] / FPS > GAP_BETWEEN_ACTIONS, (
        "when the model is honestly unsure, committing early buys nothing: "
        f"both layers took {early[0]} frames ({early[0] / FPS * 1000:.0f} ms)"
    )


def test_a_short_confident_lie_is_what_early_commitment_actually_costs():
    """The failure case, stated plainly.

    Six frames of a confidently wrong opening, then the truth. The dwell layer
    sits through the lie and calls the movement correctly at frame 14. This
    layer commits at frame 4 and is wrong. That is the trade lead.py priced:
    37% useful warnings against 7%, bought with calls like this one.
    """
    stream = [onehot("hook")] * 6 + [onehot("jab")] * 40
    early, dwell = frames_to_decide(stream)
    assert early == (4, "hook"), f"committer {early}, wrong and fast"
    assert dwell == (14, "jab"), f"GestureMachine {dwell}, right and late"


def test_across_a_mixed_stream_it_is_faster_and_no_less_accurate():
    """Both layers over the same sixty movements, a third of them feints.

    One stream proves nothing either way, so this is the population version:
    accuracy side by side and median frames side by side, on identical input.
    """
    rng = np.random.default_rng(7)
    movements = ["jab", "hook", "kick"]
    trials = []
    for i in range(60):
        truth = movements[i % len(movements)]
        confidence = float(rng.uniform(0.88, 1.0))
        stream = []
        if i % 3 == 0:                              # a third open ambiguously
            decoy = movements[(i + 1) % len(movements)]
            hesitation = int(rng.integers(4, 10))
            stream += [split(decoy, truth, share=float(rng.uniform(0.42, 0.48)))
                       ] * hesitation
        stream += [onehot(truth, confidence)] * 30
        trials.append((truth, stream))

    early_right = dwell_right = 0
    early_frames, dwell_frames = [], []
    for truth, stream in trials:
        early, dwell = frames_to_decide(stream)
        assert early is not None and dwell is not None, "both must decide"
        early_right += early[1] == truth
        dwell_right += dwell[1] == truth
        early_frames.append(early[0])
        dwell_frames.append(dwell[0])

    early_median = float(np.median(early_frames))
    dwell_median = float(np.median(dwell_frames))
    report = (
        f"committer: {early_right}/{len(trials)} correct, median "
        f"{early_median:.0f} frames ({early_median / FPS * 1000:.0f} ms); "
        f"GestureMachine: {dwell_right}/{len(trials)} correct, median "
        f"{dwell_median:.0f} frames ({dwell_median / FPS * 1000:.0f} ms)"
    )
    assert early_median < dwell_median, report
    assert early_right >= dwell_right, report
