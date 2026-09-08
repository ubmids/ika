"""A hand closing on the lens, checked against approaches imposed by hand.

The camera has no depth sensor, so `approach.py` claims to measure closing from
apparent size alone. That is a claim, not a fact, and these tests are what
turns it into one. Every approach here is manufactured with
`synth.place(..., scale=...)`: the fractional growth per second is chosen
first, the hand is built to have exactly that growth, and the test asserts the
module recovers the number that was imposed. No camera, no network, no weights.

The central test is `test_a_sideways_sweep_does_not_read_as_closing`. A hand
crossing the frame is the motion that produces the most pixel movement and the
least closing, and a size-based measure that cannot tell the two apart would be
worse than nothing, because it would fire on every wave.
"""

import time

import numpy as np
import pytest

from ika import bodysynth, synth
from ika.approach import (AMBIGUITIES, DEFAULT_RATE_THRESHOLD,
                          DEFAULT_SECONDS, MIN_CONFIDENCE, MIN_SAMPLES,
                          REFERENCE_OPENNESS, ROTATION_WORST_RATE,
                          TURN_TOLERANCE, Approach, ApproachTracker,
                          apparent_scale, approach_rate, body_scale,
                          frame_reliability, palm_openness)
from ika.schema import FINGERS

FPS = 30.0
FRAME_INTERVAL = 1.0 / FPS

# A punch at the lens: 0.6 m to 0.25 m in a quarter of a second, so apparent
# size grows 2.4x, which is ln(2.4) / 0.25 = 3.5 e-folds per second. The same
# arithmetic that sets DEFAULT_RATE_THRESHOLD in the module.
PUNCH_RATE = 3.5
PUNCH_FRAMES = 8            # 0.23 s, a whole strike

# Apparent palm size at the start of a punch. At 0.6 m the FaceTime HD frame is
# about 0.61 m across and a palm is about 0.09 m long, so 0.15 frame widths.
START_SCALE = 0.15

# Where the hand sits when the test does not care about placement. Low enough
# in frame that a hand growing 2.4x still has its fingertips inside the image,
# because a clipped hand is a confidence test and not a rate test.
CENTRE = (0.5, 0.40)

# The live loop already spends 31.6 ms of a 263 ms budget. This module is meant
# to be a rounding error against that, and the measured cost is 0.11 ms per
# frame. The assertion is set several times looser so a busy machine does not
# fail the suite over scheduling noise, while still catching any change that
# makes this expensive.
COST_BUDGET_MS = 0.5


def approaching(rate, frames=15, start=START_SCALE, centre=CENTRE,
                lateral=0.0, pose="fist", noise=0.0, seed=0, aspect=1.0):
    """A hand whose apparent size grows at exactly `rate` per second.

    Ground truth by construction. The synthetic fist's largest palm span is
    exactly 1.0 in its own units, so `scale=s` in `synth.place` puts the
    apparent scale at exactly `s` frame widths and the imposed fractional rate
    comes back out of `approach_rate` to floating-point precision.
    """
    hand = synth.pose(pose)
    out = []
    for i in range(frames):
        at = i * FRAME_INTERVAL
        marks = synth.place(
            hand,
            translate=(centre[0] + lateral * at, centre[1], 0.0),
            scale=start * np.exp(rate * at),
            noise=noise,
            seed=seed * 1000 + i,
        )
        if aspect != 1.0:
            # What MediaPipe reports on a non-square frame: y normalised by
            # height rather than by width.
            marks[:, 1] *= aspect
        out.append((marks, at))
    return out


def body_growing(rate, frames=15, start=0.20):
    """A torso whose apparent size grows at `rate` per second: a person
    leaning toward the screen, or being pushed toward it in a chair."""
    figure = bodysynth.body_pose()
    return [
        synth.place(figure, translate=(0.5, 0.55, 0.0),
                    scale=start * np.exp(rate * i * FRAME_INTERVAL))
        for i in range(frames)
    ]


def run(sequence, bodies=None, **kwargs):
    """Feed a sequence through a fresh tracker and return the last verdict."""
    tracker = ApproachTracker(**kwargs)
    verdict = None
    for i, (marks, at) in enumerate(sequence):
        verdict = tracker.update(marks, at, body=None if bodies is None else bodies[i])
    return verdict


def in_plane(axis_degrees, tilt_degrees):
    """A rotation by `tilt` about an axis lying in the image plane.

    Not expressible as one of `synth.rotation`'s three angles, and the axis has
    to be swept because the whole point of taking the largest palm span is that
    the damage depends on which axis the hand turns about.
    """
    alpha = np.radians(axis_degrees)
    axis = np.array([np.cos(alpha), np.sin(alpha), 0.0])
    cross = np.array([[0.0, 0.0, axis[1]],
                      [0.0, 0.0, -axis[0]],
                      [-axis[1], axis[0], 0.0]])
    theta = np.radians(tilt_degrees)
    return np.eye(3) + np.sin(theta) * cross + (1 - np.cos(theta)) * (cross @ cross)


# The axis that costs the largest palm span the most, found by sweeping every
# in-plane axis at full tilt. Pinned as a constant so the rotation tests attack
# the worst case rather than a convenient one.
WORST_ROTATION_AXIS_DEGREES = 31.0


def flipping(seconds, frames=None, start=START_SCALE, axis=WORST_ROTATION_AXIS_DEGREES):
    """A hand turning from fully edge-on to fully face-on, at constant size.

    Nothing about this is an approach, and every span across the palm grows
    while it happens. It is the hardest false positive for a size-based
    measure.
    """
    frames = frames or int(round(seconds * FPS)) + 1
    hand = synth.pose("fist")
    out = []
    for i in range(frames):
        done = i / (frames - 1)
        marks = synth.place(hand, rotate=in_plane(axis, (1.0 - done) * 90.0),
                            translate=(CENTRE[0], CENTRE[1], 0.0), scale=start)
        out.append((marks, i * seconds / (frames - 1)))
    return out


# --------------------------------------------------------------------------
# the interface


def test_the_verdict_carries_a_scale_a_rate_a_judgement_and_a_confidence():
    verdict = run(approaching(PUNCH_RATE, frames=PUNCH_FRAMES))
    assert isinstance(verdict, Approach)
    assert 0.0 < verdict.scale < 1.0
    assert isinstance(verdict.rate, float)
    assert isinstance(verdict.closing, bool)
    assert 0.0 <= verdict.confidence <= 1.0
    with pytest.raises(Exception):
        verdict.closing = False    # frozen, so a caller cannot edit the record


def test_apparent_scale_takes_a_hand_object_or_a_bare_array():
    """Synthetic fixtures and live detections must go through identical code,
    or the tests are checking a path the camera never uses."""
    from ika.hands import Hand

    marks = synth.place(synth.pose("fist"), translate=(0.5, 0.4, 0.0), scale=0.2)
    hand = Hand(image=marks.astype(np.float32), world=synth.pose("fist"),
                label="Right", score=0.9)
    assert apparent_scale(hand) == pytest.approx(apparent_scale(marks), rel=1e-6)


def test_the_documented_ambiguities_are_stated_rather_than_hidden():
    """The module is allowed to fail; it is not allowed to fail quietly. Each
    entry here has a test below that pins the failure."""
    assert len(AMBIGUITIES) >= 4
    assert any("lean" in item for item in AMBIGUITIES)
    assert any("edge-on" in item for item in AMBIGUITIES)


# --------------------------------------------------------------------------
# recovering an approach that was imposed by hand


@pytest.mark.parametrize("rate", [0.0, 0.5, 1.0, 2.0, 3.5, 5.0, -2.0, -3.5])
def test_a_known_approach_rate_is_recovered_exactly(rate):
    """Ground truth imposed with `synth.place(scale=...)`, then recovered.

    Exact rather than approximate because the fit is on the logarithm, and a
    constant-speed closing under a pinhole camera is a geometric ramp in
    apparent size, which is a straight line in log space.
    """
    sequence = approaching(rate)
    scales = [apparent_scale(marks) for marks, _ in sequence]
    times = [at for _, at in sequence]
    assert approach_rate(scales, times) == pytest.approx(rate, abs=1e-9)


def test_the_rate_means_the_same_thing_near_and_far():
    """Fractional growth, not growth in frame widths. Without this the same
    punch reads several times larger when thrown from close in, and no single
    threshold could hold across a session where the user shifts in their seat.
    """
    near = run(approaching(PUNCH_RATE, frames=PUNCH_FRAMES, start=0.22))
    far = run(approaching(PUNCH_RATE, frames=PUNCH_FRAMES, start=0.06))
    assert near.rate == pytest.approx(far.rate, abs=1e-6)
    assert near.scale > far.scale * 3      # very different apparent sizes
    assert near.closing and far.closing


def test_curling_the_fingers_does_not_change_the_apparent_scale():
    """A fist must not read as a small hand held further away.

    The same reason `features.palm_scale` measures wrist to middle knuckle:
    every point the scale is built from is on the rigid part of the hand. Using
    anything with a joint in it would make closing a fist look like retreating.
    """
    spread = 0.0
    opened = synth.synthetic_hand(curl={}, spread=spread)
    closed = synth.synthetic_hand(
        curl={name: 1.0 for name in FINGERS}, spread=spread)
    placed = [synth.place(h, translate=(0.5, 0.4, 0.0), scale=0.2)
              for h in (opened, closed)]
    assert apparent_scale(placed[0]) == pytest.approx(apparent_scale(placed[1]), rel=1e-9)
    assert palm_openness(placed[0]) == pytest.approx(palm_openness(placed[1]), rel=1e-9)


def test_a_punch_at_the_lens_reads_as_closing():
    verdict = run(approaching(PUNCH_RATE, frames=PUNCH_FRAMES))
    assert verdict.closing
    assert verdict.reason == "closing"
    assert verdict.rate == pytest.approx(PUNCH_RATE, abs=1e-6)
    assert verdict.confidence > 0.7


def test_a_hand_retreating_is_reported_as_retreating_and_not_as_closing():
    verdict = run(approaching(-PUNCH_RATE, start=0.36))
    assert not verdict.closing
    assert verdict.rate < -DEFAULT_RATE_THRESHOLD
    assert verdict.reason == "retreating"


# --------------------------------------------------------------------------
# the central test: closing is not the same as moving


def test_a_sideways_sweep_does_not_read_as_closing():
    """The test this module exists to pass.

    A hand crossing 0.6 of the frame in under half a second is the largest
    pixel movement a hand makes, and it is not an approach at all. A measure
    that confused the two would fire on every wave, which is worse than having
    no measure. Apparent size is constant here because the distance is, so the
    rate is zero and stays zero however fast the hand travels.
    """
    for lateral in (0.5, 1.0, 1.28):
        sweep = approaching(0.0, lateral=lateral, centre=(0.2, 0.40))
        verdict = run(sweep)
        assert not verdict.closing, f"a sweep at {lateral} frame widths per second fired"
        assert abs(verdict.rate) < 1e-6
        # And the caller is told which of the two "not closing" cases it is.
        assert verdict.reason == "sweep"
        assert verdict.confidence > 0.8, "a sweep across open frame is a reliable reading"

    # A sweep fast enough to leave the frame is still not an approach, but the
    # confidence drops because the hand is genuinely half out of shot by the
    # end, which is the honest reading and not a failure of the rate.
    off_frame = run(approaching(0.0, lateral=2.0, centre=(0.2, 0.40)))
    assert not off_frame.closing
    assert off_frame.confidence < 0.8


def test_a_sweep_and_a_punch_are_separated_by_the_full_threshold():
    """Not merely ordered correctly. A margin that had collapsed to nothing
    would still pass an ordering test, which is exactly how the `z` sweep found
    a classifier at 100% accuracy whose decision margin had fallen from 44x to
    1.5x."""
    sweep = run(approaching(0.0, lateral=1.28, centre=(0.2, 0.40)))
    punch = run(approaching(PUNCH_RATE, frames=PUNCH_FRAMES))
    assert punch.rate - sweep.rate > 2 * DEFAULT_RATE_THRESHOLD


def test_a_hand_held_still_does_not_read_as_closing():
    verdict = run(approaching(0.0))
    assert not verdict.closing
    assert verdict.reason == "steady"
    assert verdict.confidence > 0.8


def test_a_slow_drift_stays_under_the_threshold():
    """An ordinary lean back in a chair, 0.29 per second by the arithmetic in
    the module. It has to stay quiet or the system fires while the user reads."""
    verdict = run(approaching(0.3))
    assert not verdict.closing
    assert verdict.rate == pytest.approx(0.3, abs=1e-6)


# --------------------------------------------------------------------------
# the person leaning in, which is what the body is for


def test_a_whole_person_leaning_in_is_not_a_punch():
    """Requirement three. If the body grew at the rate the hand did, the hand
    was carried rather than thrown.

    This is the same correction `interaction.py` needed: a strike measured
    against the other party rather than in isolation, which is what took
    strikes there from 42% to 82%.
    """
    lean = 1.6                       # well over the threshold on its own
    verdict = run(approaching(lean), bodies=body_growing(lean))
    assert verdict.rate > DEFAULT_RATE_THRESHOLD, "the hand really did grow"
    assert verdict.body_rate == pytest.approx(lean, abs=1e-6)
    assert not verdict.closing
    assert verdict.reason == "lean"
    assert verdict.confidence > 0.8, "a lean is a confident negative, not a shrug"


def test_a_punch_thrown_while_stepping_in_still_reads_as_closing():
    """The body-relative correction must not throw away real strikes. A fighter
    steps in as they punch, so the body grows too, just far more slowly."""
    verdict = run(approaching(PUNCH_RATE, frames=PUNCH_FRAMES),
                  bodies=body_growing(0.8, frames=PUNCH_FRAMES))
    assert verdict.closing
    assert verdict.rate - verdict.body_rate > DEFAULT_RATE_THRESHOLD


def test_a_lean_in_cannot_be_told_from_a_punch_without_a_body():
    """A stated failure, measured rather than hidden.

    With only a hand in view there is nothing that distinguishes a hand carried
    toward the lens by the whole upper body from a hand thrown at it. This
    module calls it closing. The boundary is a fitted rate just over the
    threshold, which at 0.6 m is a closing speed of about 0.75 m/s: a lunge
    rather than a lean, but reachable by a person shifting forward hard.
    """
    body = body_growing(1.5)
    hand = approaching(1.5)
    with_body = run(hand, bodies=body)
    without_body = run(hand)

    assert not with_body.closing, "with a body in view the ambiguity is resolved"
    assert without_body.closing, "without one it is not, and this is the miss"
    assert without_body.rate == pytest.approx(with_body.rate, abs=1e-9), (
        "the two verdicts see the identical hand, so the difference is only "
        "what the body adds"
    )

    # The boundary, measured. Just under the threshold stays quiet.
    assert not run(approaching(DEFAULT_RATE_THRESHOLD - 0.05)).closing
    assert run(approaching(DEFAULT_RATE_THRESHOLD + 0.05)).closing


def test_the_body_reference_ignores_the_limbs():
    """Hip centre to shoulder centre, so a punch cannot change the reference it
    is being measured against. If it could, the relative rate would cancel the
    strike out along with the lean."""
    still = bodysynth.body_pose(right_extension=0.0)
    thrown = bodysynth.body_pose(right_extension=1.0)
    placed = [synth.place(b, translate=(0.5, 0.5, 0.0), scale=0.2)
              for b in (still, thrown)]
    assert body_scale(placed[0]) == pytest.approx(body_scale(placed[1]), rel=1e-6)


# --------------------------------------------------------------------------
# depth-free, and provably so


def test_nothing_reads_the_inferred_z_coordinate():
    """Requirement one, proved by destruction rather than by inspection.

    MediaPipe's `z` is inferred, not measured. Corrupting it with noise a
    hundred times the size of the landmark coordinates must leave every field
    of every verdict bit-for-bit identical. A synthetic depth-error sweep kept
    a two-way classifier at 100% while its decision margin fell from 44x to
    1.5x, so "still works" is not a sufficient bar: identical is.
    """
    sequence = approaching(PUNCH_RATE, frames=PUNCH_FRAMES)
    bodies = body_growing(0.8, frames=PUNCH_FRAMES)

    def verdicts(corrupt):
        rng = np.random.default_rng(7)
        tracker = ApproachTracker()
        out = []
        for (marks, at), body in zip(sequence, bodies):
            marks, body = marks.copy(), body.copy()
            if corrupt:
                marks[:, 2] = rng.normal(0.0, 100.0, len(marks))
                body[:, 2] = rng.normal(0.0, 100.0, len(body))
            out.append(tracker.update(marks, at, body=body))
        return out

    clean, corrupted = verdicts(False), verdicts(True)
    assert clean == corrupted
    assert any(v.closing for v in clean), "and the punch was still detected"


# --------------------------------------------------------------------------
# a hand turning, which grows without approaching


def test_a_hand_turning_face_on_is_not_a_punch():
    """The false positive that geometry alone cannot bound.

    Every span across the palm grows as the hand turns toward the camera even
    though the distance never changes. Taking the largest span keeps the
    fabricated rate near the threshold instead of far over it, and the openness
    term does the rest: the flip turns the palm hard while a punch does not
    turn it at all.
    """
    for seconds in (0.2, 0.3, 0.5):
        verdict = run(flipping(seconds))
        assert not verdict.closing, f"a {seconds}s flip read as a punch"
        assert verdict.reason == "turning"
        assert verdict.confidence < MIN_CONFIDENCE


def test_the_worst_rotation_fabricates_the_rate_the_threshold_was_set_against():
    """Pinned, because the margin here is only ten percent and a change that
    quietly ate it would leave the flip test passing on confidence alone."""
    sequence = flipping(0.3)
    scales = [apparent_scale(marks) for marks, _ in sequence]
    times = [at for _, at in sequence]
    assert approach_rate(scales, times) == pytest.approx(ROTATION_WORST_RATE, abs=0.02)
    assert ROTATION_WORST_RATE < DEFAULT_RATE_THRESHOLD


def test_taking_the_largest_palm_span_is_what_keeps_rotation_bounded():
    """The averaged span, which was the first thing tried, fabricates 2.23 per
    second on the same flip, which is two thirds of a full jab. Pinned here so
    the choice is not undone as a simplification."""
    from ika.approach import PALM_SPANS

    def averaged(marks):
        return float(np.mean([np.linalg.norm(marks[a, :2] - marks[b, :2])
                              for a, b in PALM_SPANS]))

    sequence = flipping(0.3)
    times = [at for _, at in sequence]
    mean_rate = approach_rate([averaged(m) for m, _ in sequence], times)
    max_rate = approach_rate([apparent_scale(m) for m, _ in sequence], times)
    assert mean_rate == pytest.approx(2.23, abs=0.02)
    assert max_rate == pytest.approx(ROTATION_WORST_RATE, abs=0.02)
    assert mean_rate > DEFAULT_RATE_THRESHOLD > max_rate


def test_a_punch_does_not_turn_the_palm_and_a_flip_does():
    """Why openness works as the discriminator: it is a ratio of area to size
    squared, so it is blind to distance and sensitive only to turning."""
    punch = approaching(PUNCH_RATE, frames=PUNCH_FRAMES)
    times = [at for _, at in punch]
    punch_turn = abs(approach_rate([palm_openness(m) for m, _ in punch], times))

    flip = flipping(0.2)
    flip_turn = abs(approach_rate(
        [max(palm_openness(m), 0.01) for m, _ in flip], [at for _, at in flip]))

    assert punch_turn < 1e-6
    assert flip_turn > TURN_TOLERANCE * 5


def test_openness_falls_to_nothing_when_the_palm_goes_edge_on():
    face_on = synth.place(synth.pose("fist"), translate=(0.5, 0.4, 0.0), scale=0.2)
    edge_on = synth.place(synth.pose("fist"),
                          rotate=in_plane(WORST_ROTATION_AXIS_DEGREES, 90.0),
                          translate=(0.5, 0.4, 0.0), scale=0.2)
    assert palm_openness(face_on) > REFERENCE_OPENNESS
    assert palm_openness(edge_on) < 1e-9


def test_a_strike_thrown_edge_on_is_reported_unreliable_and_not_closing():
    """Another stated failure. A knife hand presents no palm plane, so its
    apparent size is not measurable rather than merely noisy, and the honest
    answer is a low confidence instead of a number. It is a miss, on purpose.
    """
    hand = synth.pose("fist")
    turned = synth.place(hand, rotate=in_plane(WORST_ROTATION_AXIS_DEGREES, 90.0))
    tracker = ApproachTracker()
    verdict = None
    for i in range(PUNCH_FRAMES):
        at = i * FRAME_INTERVAL
        verdict = tracker.update(
            synth.place(turned, translate=(0.5, 0.4, 0.0),
                        scale=START_SCALE * np.exp(PUNCH_RATE * at)), at)
    assert verdict.rate == pytest.approx(PUNCH_RATE, abs=1e-6), "the size did grow"
    assert not verdict.closing
    assert verdict.confidence < MIN_CONFIDENCE


# --------------------------------------------------------------------------
# confidence, reported honestly


def test_a_hand_at_the_frame_edge_gives_low_confidence_not_a_wrong_number():
    """Requirement four. MediaPipe returns all 21 points whatever it can see,
    extrapolating the ones outside the image, so a hand at the border produces
    palm spans that look like measurements and are guesses."""
    verdict = run(approaching(PUNCH_RATE, frames=PUNCH_FRAMES, centre=(0.97, 0.40)))
    assert verdict.confidence < MIN_CONFIDENCE
    assert not verdict.closing
    assert verdict.reason == "unreliable"


def test_a_hand_hugging_the_border_is_unreliable_before_it_leaves_the_frame():
    """The check cannot wait for landmarks to go outside [0, 1]: by then the
    scale has already been wrong for several frames."""
    sequence = approaching(0.0, start=0.10, centre=(0.062, 0.02))
    inside = all(
        np.all((marks[:, :2] >= 0.0) & (marks[:, :2] <= 1.0))
        for marks, _ in sequence
    )
    assert inside, "this fixture is about a hand that is still fully in shot"
    verdict = run(sequence)
    assert verdict.confidence < MIN_CONFIDENCE
    assert verdict.reason == "unreliable"


def test_a_hand_in_open_frame_is_a_reliable_reading():
    marks, _ = approaching(0.0)[0]
    assert frame_reliability(marks) > 0.9


def test_landmark_jitter_lowers_confidence_without_moving_the_rate_much():
    """A punch measured through noisy landmarks should still be a punch, and
    should say it is slightly less sure."""
    clean = run(approaching(PUNCH_RATE, frames=PUNCH_FRAMES))
    errors, confidences = [], []
    for seed in range(20):
        noisy = run(approaching(PUNCH_RATE, frames=PUNCH_FRAMES,
                                noise=0.005, seed=seed + 1))
        errors.append(noisy.rate - PUNCH_RATE)
        confidences.append(noisy.confidence)
        assert noisy.closing, f"seed {seed} lost the punch"
    assert abs(np.mean(errors)) < 0.25
    assert np.std(errors) < 0.25
    assert np.mean(confidences) < clean.confidence


def test_confidence_falls_when_the_size_series_is_not_a_clean_expansion():
    """A hand whose apparent size jumps about is not approaching at some rate,
    it is a tracking failure, and the fitted slope through it is noise."""
    hand = synth.pose("fist")
    rng = np.random.default_rng(3)
    tracker = ApproachTracker()
    verdict = None
    for i in range(15):
        at = i * FRAME_INTERVAL
        scale = START_SCALE * float(np.exp(rng.normal(0.0, 0.35)))
        verdict = tracker.update(
            synth.place(hand, translate=(0.5, 0.4, 0.0), scale=scale), at)
    assert verdict.confidence < MIN_CONFIDENCE


def test_a_verdict_never_commits_below_the_confidence_floor():
    """Across every fixture in this file, `closing` implies a confidence the
    caller can act on. The floor existing is not the same as it being wired in.
    """
    for sequence in (approaching(PUNCH_RATE, frames=PUNCH_FRAMES),
                     approaching(PUNCH_RATE, frames=PUNCH_FRAMES, centre=(0.97, 0.40)),
                     approaching(0.0, lateral=1.28, centre=(0.2, 0.40)),
                     flipping(0.2)):
        verdict = run(sequence)
        assert not verdict.closing or verdict.confidence >= MIN_CONFIDENCE


# --------------------------------------------------------------------------
# the streaming contract


def test_a_cold_start_reports_no_evidence_rather_than_a_rate():
    """A live loop asks on the first frame it ever sees. Two samples fit a line
    perfectly, so a two-point rate reports the noise between them."""
    tracker = ApproachTracker()
    sequence = approaching(PUNCH_RATE, frames=PUNCH_FRAMES)
    for i in range(MIN_SAMPLES - 1):
        marks, at = sequence[i]
        verdict = tracker.update(marks, at)
        assert verdict.rate == 0.0
        assert verdict.confidence == 0.0
        assert verdict.reason == "cold"
        assert verdict.scale > 0.0, "the size is known even when the rate is not"


def test_too_little_history_is_answered_with_zero_rather_than_an_exception():
    assert approach_rate([0.1, 0.2], [0.0, 0.1]) == 0.0
    assert approach_rate([], []) == 0.0
    # Enough samples but crammed into too little time is the same problem.
    assert approach_rate([0.1, 0.11, 0.12, 0.13], [0.0, 0.001, 0.002, 0.003]) == 0.0


def test_a_mismatched_pair_of_series_is_an_error_and_not_a_guess():
    with pytest.raises(ValueError, match="against"):
        approach_rate([0.1, 0.2, 0.3, 0.4], [0.0, 0.1, 0.2])


def test_losing_the_hand_clears_the_window():
    """A scale series spanning a gap is two observations with a hole between
    them, and fitting a line through the hole invents whatever rate the hole
    implies."""
    tracker = ApproachTracker()
    for marks, at in approaching(PUNCH_RATE, frames=PUNCH_FRAMES):
        tracker.update(marks, at)
    assert len(tracker) > 0

    gap = tracker.update(None, PUNCH_FRAMES * FRAME_INTERVAL)
    assert len(tracker) == 0
    assert gap.reason == "no hand"
    assert gap.confidence == 0.0
    assert not gap.closing


def test_reset_clears_the_window():
    tracker = ApproachTracker()
    for marks, at in approaching(PUNCH_RATE, frames=PUNCH_FRAMES):
        tracker.update(marks, at)
    tracker.reset()
    assert len(tracker) == 0
    marks, at = approaching(0.0)[0]
    assert tracker.update(marks, at).reason == "cold"


def test_time_going_backwards_starts_a_new_stream():
    """Two recordings played through one tracker must not be fitted as one
    approach."""
    tracker = ApproachTracker()
    for marks, at in approaching(PUNCH_RATE, frames=PUNCH_FRAMES):
        tracker.update(marks, at)
    marks, _ = approaching(0.0)[0]
    verdict = tracker.update(marks, 0.0)
    assert len(tracker) == 1
    assert verdict.reason == "cold"


def test_the_window_holds_only_the_last_half_second():
    """Length in seconds and not frames, as in `motion.MotionWindow`, because
    frame rate moves with load and a window that shrank when the machine got
    busy would change what a punch is."""
    tracker = ApproachTracker(seconds=DEFAULT_SECONDS)
    for i in range(60):                      # two full seconds at 30 fps
        marks, _ = approaching(0.0)[0]
        tracker.update(marks, i * FRAME_INTERVAL)
    assert len(tracker) <= int(DEFAULT_SECONDS * FPS) + 1


def test_a_shorter_window_and_a_higher_threshold_are_both_honoured():
    fast = ApproachTracker(seconds=0.25, threshold=10.0)
    verdict = None
    for marks, at in approaching(PUNCH_RATE):
        verdict = fast.update(marks, at)
    assert len(fast) <= int(0.25 * FPS) + 1
    assert not verdict.closing, "a threshold of 10 per second must reject a jab"


# --------------------------------------------------------------------------
# the non-square frame


def test_the_frame_aspect_matters_once_the_hand_rolls_in_the_image_plane():
    """MediaPipe normalises x by width and y by height, so on a 16:9 frame the
    two axes are in different units, and a hand that only rotated appears to
    change size. Taking the largest palm span absorbs most of the 1.78x
    stretch, but 1.22x survives, which over a 0.2 s roll is a fabricated rate
    of 1.23 per second: just over the firing threshold, from a hand that never
    moved.
    """
    aspect = 16 / 9
    hand = synth.pose("fist")
    naive, corrected = [], []
    for degrees in (0.0, 30.0, 60.0, 90.0):
        marks = synth.place(hand, rotate=synth.rotation(yaw=np.radians(degrees)),
                            translate=(0.5, 0.28, 0.0), scale=0.12)
        marks[:, 1] *= aspect
        naive.append(apparent_scale(marks))
        corrected.append(apparent_scale(marks, aspect=aspect))

    assert max(naive) / min(naive) == pytest.approx(1.217, abs=0.01)
    assert np.log(max(naive) / min(naive)) / 0.2 > 0.9, (
        "which over a 0.2 s roll is most of the firing threshold"
    )
    assert max(corrected) / min(corrected) < 1.00001, "passing the aspect removes it"


@pytest.mark.parametrize("aspect_told", [16 / 9, 1.0])
def test_a_rolling_hand_on_a_widescreen_frame_does_not_read_as_a_punch(aspect_told):
    """Rolled the direction that makes the anisotropy look like growth.

    Told the truth, the rate is exactly zero. Told nothing, the rate clears the
    threshold and the verdict is still not closing, because the same wrong
    aspect distorts the palm openness and the reading comes back as turning.
    Belt and braces, and the reason it is safe to default `aspect` to 1.0.
    """
    aspect = 16 / 9
    hand = synth.pose("fist")
    tracker = ApproachTracker(aspect=aspect_told)
    verdict = None
    for i in range(7):                        # a 90 degree roll in 0.2 s
        at = i * FRAME_INTERVAL
        marks = synth.place(hand,
                            rotate=synth.rotation(yaw=np.radians(90.0 - i * 15.0)),
                            translate=(0.5, 0.28, 0.0), scale=0.12)
        marks[:, 1] *= aspect
        verdict = tracker.update(marks, at)

    assert not verdict.closing
    if aspect_told == aspect:
        assert abs(verdict.rate) < 1e-9
        assert verdict.reason == "steady"
    else:
        assert verdict.rate > DEFAULT_RATE_THRESHOLD, "the wrong aspect invents growth"
        assert verdict.reason == "turning", "and confidence is what catches it"


# --------------------------------------------------------------------------
# cost


def test_it_costs_a_small_fraction_of_a_millisecond_per_frame():
    """Requirement five. The live loop already spends 31.6 ms of a 263 ms
    budget, so anything added here has to be a rounding error against that.

    Timed with a body supplied, which is the expensive path, and over a full
    window so the regression is fitted on every call rather than short-circuited
    by the cold-start branch.
    """
    sequence = approaching(PUNCH_RATE, frames=300, start=0.02)
    body = body_growing(0.0, frames=1)[0]
    tracker = ApproachTracker()
    for marks, at in sequence[:20]:
        tracker.update(marks, at, body=body)

    best = float("inf")
    for _ in range(5):
        started = time.perf_counter()
        for marks, at in sequence:
            tracker.update(marks, at, body=body)
        best = min(best, (time.perf_counter() - started) / len(sequence) * 1000.0)

    assert best < COST_BUDGET_MS, f"{best:.3f} ms per frame"
