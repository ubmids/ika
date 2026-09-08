"""Finding movement boundaries in a stream, measured rather than asserted.

Two kinds of input here, and both are needed.

Synthetic activity traces built with numpy carry bursts placed at times this
file chose, so a recovered boundary can be compared against the truth in
milliseconds instead of being eyeballed. That is the only way to say how
accurate the boundaries are.

Real movement features from `bodyaction` and `trajectory` answer the different
question of whether `activity_of` turns a body into a signal worth
thresholding at all. A segmenter that is perfect on square waves and blind to
a punch would pass the first half of this file and be useless.

No camera, no network, no weights.
"""

import numpy as np
import pytest

from ika import bodyaction, posture, trajectory
from ika.bodymotion import Frame
from ika.motion import Sample
from ika.segment import (ACTIVE_ACTIVITY, ACTIVITY_SPAN_SECONDS,
                         DIP_BRIDGE_SECONDS, MIN_MOVEMENT_SECONDS,
                         PREROLL_FRAMES, QUIET_ACTIVITY, RECENT_SEGMENTS,
                         SEGMENT_FIELDS, STALE_GAP_SECONDS, Segment,
                         Segmenter, activity_of)

FPS = 30.0
ONE_FRAME = 1.0 / FPS

# The gap between one action ending and the next starting, from `tell/lead.py`.
# Quoted because a latency claim against a number somebody picked proves
# nothing.
GAP_BETWEEN_ACTIONS = 0.263

# How much a quiet stream reads when nothing is happening. Not zero: a tracker
# never returns the same body twice.
BASELINE = 0.04

# How many frames back the activity span looks. Three frames at 30 fps is the
# 100 ms that `ACTIVITY_SPAN_SECONDS` names.
SPAN_FRAMES = int(round(ACTIVITY_SPAN_SECONDS * FPS))

# The named quantities that need legs. A webcam does not see them: measured
# over 400 photographs in `bodyaction`, knees appeared in 4% and ankles in
# none, and the landmarker returns a guess rather than nothing. Feeding those
# guesses to `activity_of` would read their re-guessing every frame as
# movement.
LEG_QUANTITIES = ("left_knee_angle", "right_knee_angle",
                  "stance_width", "weight_shift")
LEG_COLUMNS = [posture.DERIVED[name] - posture.DERIVED_START
               for name in LEG_QUANTITIES]

# A hand span to divide positions by. Fixed, not the per-frame `span`, because
# dividing by a scale that jitters injects motion that never happened.
# `motion.features` takes the median over the window for the same reason.
NOMINAL_HAND_SPAN = 0.16


# --------------------------------------------------------------- trace makers

def burst_trace(seconds, bursts, fps=FPS, baseline=BASELINE, noise=0.0,
                seed=0):
    """A quiet baseline with bursts at times the caller chose.

    Bursts are (start, duration, amplitude) and are shaped as a half sine
    raised to a power, so they rise fast and fall fast the way a strike does
    while still having support exactly inside the stated span. A square burst
    would flatter the segmenter: nothing real turns on in one frame.
    """
    rng = np.random.default_rng(seed)
    times = np.arange(0.0, seconds, 1.0 / fps)
    activity = np.full(times.size, baseline) + rng.normal(0, noise, times.size)
    for start, duration, amplitude in bursts:
        inside = (times >= start) & (times < start + duration)
        u = (times[inside] - start) / duration
        activity[inside] += amplitude * np.sin(np.pi * u) ** 0.6
    return times, np.clip(activity, 0.0, None)


def run(times, activity, segmenter=None, **kwargs):
    """Feed a whole trace and collect the segments that closed."""
    segmenter = segmenter or Segmenter(**kwargs)
    closed = []
    for value, at in zip(activity, times):
        got = segmenter.update(float(value), float(at))
        if got is not None:
            closed.append(got)
    return segmenter, closed


def naive_segments(times, activity, threshold):
    """One threshold, no hysteresis, no minimum. The thing being avoided."""
    runs, inside = 0, False
    for value in activity:
        if value >= threshold and not inside:
            runs, inside = runs + 1, True
        elif value < threshold:
            inside = False
    return runs


# ------------------------------------------------- real features, body lane

def body_descriptor(frame):
    """What `activity_of` should read off a body frame.

    The named quantities, minus the ones built from legs the camera cannot
    see. Named rather than raw coordinates because these are the domain's
    verbs, exactly as `bodymotion` argues, and because the 33 visibility
    values in the full vector flicker on their own.
    """
    return np.delete(frame.features[posture.DERIVED_START:], LEG_COLUMNS)


def held_body(frame, count, at0, rng, jitter=0.01):
    """A body holding the posture it finished in, with landmarker jitter.

    This is what the stream looks like between actions. Splicing another
    `bodyaction.make` on instead would teleport the arm back to a guard in one
    frame, and that teleport is itself a movement, so the rest between two
    actions has to be built rather than borrowed. Jitter lands on the named
    quantities because that is what the descriptor reads.
    """
    out = []
    for i in range(count):
        features = frame.features.copy()
        block = features[posture.DERIVED_START:]
        features[posture.DERIVED_START:] = block + rng.normal(0, jitter, block.size)
        out.append(Frame(at0 + i / FPS, features, frame.wrists, frame.scale,
                         frame.visibility))
    return out


def body_stream(names, seed=0, rest=0.6):
    """Actions with rest either side, on one continuous clock.

    Returns the frames and the true span of each action, which is the thing
    the segmenter is not told.
    """
    rng = np.random.default_rng(seed)
    frames, truth = [], []
    resting = int(round(rest * FPS))
    guard = bodyaction.make("idle", fps=FPS, seed=int(rng.integers(1 << 30)))[0]
    frames += held_body(guard, max(resting, 1), 0.0, rng)
    at = len(frames) / FPS
    for name in names:
        core = bodyaction.make(name, fps=FPS, seed=int(rng.integers(1 << 30)))
        started = at
        for frame in core:
            frames.append(Frame(at, frame.features, frame.wrists, frame.scale,
                                frame.visibility))
            at += 1.0 / FPS
        truth.append((name, started, at - 1.0 / FPS))
        frames += held_body(frames[-1], resting, at, rng)
        at += resting / FPS
    return frames, truth


# ------------------------------------------------- real features, hand lane

def hand_descriptor(sample):
    """What `activity_of` should read off a hand sample.

    Where the wrist is and how pinched it is. Not the static pose vector: that
    lane deliberately destroys translation, and a swipe is translation, so pose
    alone reads a swipe as a palm sitting still.
    """
    return np.concatenate([np.asarray(sample.position) / NOMINAL_HAND_SPAN,
                           [sample.pinch]])


def held_hand(sample, count, at0, rng):
    """A hand resting where it stopped, with tracker jitter."""
    return [
        Sample(at0 + i / FPS,
               np.asarray(sample.position) + rng.normal(0, 0.002, 2),
               sample.span,
               sample.pinch + float(rng.normal(0, 0.01)),
               sample.pose + rng.normal(0, 0.004, sample.pose.size))
        for i in range(count)
    ]


def hand_stream(name, seed=0, rest=0.6):
    """One dynamic gesture with rest either side, and its true span."""
    rng = np.random.default_rng(1000 + seed)
    core = trajectory.make(name, fps=FPS, seed=seed)
    resting = int(round(rest * FPS))
    before = held_hand(core[0], resting, 0.0, rng)
    at = resting / FPS
    middle = [Sample(at + i / FPS, s.position, s.span, s.pinch, s.pose)
              for i, s in enumerate(core)]
    started, ended = at, at + (len(core) - 1) / FPS
    after = held_hand(core[-1], resting, ended + 1.0 / FPS, rng)
    return before + middle + after, (started, ended)


def feature_trace(items, descriptor, span=SPAN_FRAMES):
    """Activity over a stream of frames or samples, span frames apart."""
    vectors = [descriptor(item) for item in items]
    times = [item.at for item in items]
    out = []
    for i in range(len(items)):
        j = max(0, i - span)
        value = 0.0 if j == i else activity_of(vectors[i], vectors[j],
                                               times[i] - times[j])
        out.append((value, times[i]))
    return np.array([t for _, t in out]), np.array([a for a, _ in out])


# ================================================================ the basics

def test_a_quiet_stream_produces_no_segments():
    times, activity = burst_trace(4.0, [], noise=0.01)
    segmenter, closed = run(times, activity)
    assert closed == [] and not segmenter.underway and segmenter.current is None


def test_one_burst_becomes_one_segment():
    times, activity = burst_trace(3.0, [(1.0, 0.25, 0.8)])
    _, closed = run(times, activity)
    assert len(closed) == 1
    assert isinstance(closed[0], Segment) and closed[0].end is not None


def test_the_recovered_boundaries_land_within_a_frame_of_the_known_burst():
    """The measurement the whole module exists for, in milliseconds."""
    start, duration = 1.0, 0.25
    times, activity = burst_trace(3.0, [(start, duration, 0.8)])
    _, closed = run(times, activity)
    got = closed[0]
    start_error = abs(got.start - start)
    end_error = abs(got.end - (start + duration))
    assert start_error <= 1.5 * ONE_FRAME, start_error * 1000
    assert end_error <= 1.5 * ONE_FRAME, end_error * 1000


def test_ten_bursts_at_known_times_are_all_recovered():
    bursts = [(0.4 + i * 0.6, 0.2 + 0.02 * i, 0.7) for i in range(10)]
    times, activity = burst_trace(7.0, bursts, noise=0.01, seed=5)
    _, closed = run(times, activity)
    assert len(closed) == len(bursts)
    errors = []
    for got, (start, duration, _) in zip(closed, bursts):
        errors.append((abs(got.start - start), abs(got.end - (start + duration))))
    worst = max(max(pair) for pair in errors)
    assert worst <= 2 * ONE_FRAME, worst * 1000


def test_segments_come_in_order_and_never_overlap():
    bursts = [(0.5 + i * 0.5, 0.2, 0.8) for i in range(6)]
    times, activity = burst_trace(5.0, bursts, noise=0.01, seed=2)
    _, closed = run(times, activity)
    for earlier, later in zip(closed, closed[1:]):
        assert earlier.end <= later.start
    assert all(s.start <= s.peak <= s.end for s in closed)


# ============================================================== requirement 1

def test_hysteresis_stops_a_noisy_boundary_from_chattering():
    """One threshold emits a segment per crossing. Two emit the movement.

    The noise here sits right on the single threshold, which is the situation
    that produces dozens of one-frame segments and the reason `active` and
    `quiet` are separate numbers.
    """
    times = np.arange(0.0, 3.0, ONE_FRAME)
    rng = np.random.default_rng(7)
    middle = (ACTIVE_ACTIVITY + QUIET_ACTIVITY) / 2
    activity = np.full(times.size, BASELINE)
    # A movement that opens sharply and then coasts along the line, which is
    # what a held or slowing movement does and the worst case for one
    # threshold.
    moving = (times >= 0.5) & (times < 2.0)
    activity[moving] = middle + rng.normal(0, 0.05, moving.sum())
    activity[(times >= 0.5) & (times < 0.6)] = 0.9

    chattered = naive_segments(times, activity, middle)
    _, closed = run(times, activity)
    assert chattered > 5, chattered
    assert len(closed) == 1, [(s.start, s.end) for s in closed]


def test_active_must_exceed_quiet():
    with pytest.raises(ValueError):
        Segmenter(quiet=0.3, active=0.3)
    with pytest.raises(ValueError):
        Segmenter(quiet=0.4, active=0.2)


# ============================================================== requirement 2

def test_a_one_frame_spike_is_rejected_as_too_short():
    times = np.arange(0.0, 1.0, ONE_FRAME)
    activity = np.full(times.size, BASELINE)
    activity[10] = 0.95
    segmenter, closed = run(times, activity)
    assert closed == [] and segmenter.rejected == 1


def test_a_burst_shorter_than_the_minimum_is_rejected_and_a_longer_one_is_not():
    """The boundary is measured rather than assumed, in frames."""
    kept = {}
    for frames in range(1, 8):
        times = np.arange(0.0, 1.0, ONE_FRAME)
        activity = np.full(times.size, BASELINE)
        activity[10:10 + frames] = 0.9
        segmenter, closed = run(times, activity)
        kept[frames] = len(closed)
    shortest_kept = min(n for n, count in kept.items() if count == 1)
    assert (shortest_kept - 1) * ONE_FRAME >= MIN_MOVEMENT_SECONDS
    assert shortest_kept <= 5, kept


def test_a_rejected_segment_is_counted_but_not_reported():
    times, activity = burst_trace(2.0, [(0.5, 0.04, 0.9), (1.0, 0.3, 0.9)])
    segmenter, closed = run(times, activity)
    assert len(closed) == 1 and segmenter.rejected == 1


# ============================================================== requirement 3

def test_the_open_segment_is_readable_before_the_movement_ends():
    """Waiting for the close would spend the whole 263 ms gap."""
    start, duration = 1.0, 0.4
    times, activity = burst_trace(3.0, [(start, duration, 0.8)])
    segmenter = Segmenter()
    seen = None
    for value, at in zip(activity, times):
        segmenter.update(float(value), float(at))
        if segmenter.underway and at < start + duration / 2:
            seen = segmenter.current
            break
    assert seen is not None
    assert seen.end is None and seen.frames >= 1
    assert abs(seen.start - start) <= 1.5 * ONE_FRAME


def test_a_movement_is_declared_underway_within_two_frames_of_its_start():
    start = 1.0
    times, activity = burst_trace(3.0, [(start, 0.3, 0.8)])
    segmenter = Segmenter()
    declared = None
    for value, at in zip(activity, times):
        segmenter.update(float(value), float(at))
        if segmenter.underway:
            declared = at
            break
    assert declared is not None
    lag_frames = (declared - start) / ONE_FRAME
    assert lag_frames <= 2.0, lag_frames


def test_the_open_segment_grows_and_then_the_closed_one_agrees_with_it():
    times, activity = burst_trace(3.0, [(1.0, 0.4, 0.8)])
    segmenter = Segmenter()
    peak_seen, closed = 0, None
    for value, at in zip(activity, times):
        got = segmenter.update(float(value), float(at))
        if segmenter.current is not None:
            peak_seen = max(peak_seen, segmenter.current.frames)
        if got is not None:
            closed = got
    assert closed is not None
    assert closed.frames <= peak_seen          # the quiet tail is trimmed off
    assert segmenter.current is None and not segmenter.underway


def test_the_start_is_backdated_to_where_activity_left_the_baseline():
    """Entry is on `active`, but the movement began at the foot of the ramp.

    A slow ramp is the case that separates the two: reporting the crossing of
    `active` as the start would put the boundary several frames late.
    """
    times = np.arange(0.0, 2.0, ONE_FRAME)
    activity = np.full(times.size, BASELINE)
    ramp = (times >= 0.5) & (times < 0.9)
    activity[ramp] = np.linspace(BASELINE, 0.9, ramp.sum())
    activity[times >= 0.9] = BASELINE

    crossed_active = float(times[activity >= ACTIVE_ACTIVITY][0])
    crossed_quiet = float(times[activity > QUIET_ACTIVITY][0])
    _, closed = run(times, activity)
    assert len(closed) == 1
    assert closed[0].start < crossed_active
    assert abs(closed[0].start - crossed_quiet) <= ONE_FRAME


def test_backdating_cannot_reach_further_back_than_the_preroll():
    """A drift that creeps over `quiet` for two seconds is not a two second
    movement, so the pre-roll is a fixed length and that length is the bound."""
    times = np.arange(0.0, 4.0, ONE_FRAME)
    activity = np.full(times.size, BASELINE)
    creep = (times >= 0.5) & (times < 2.5)
    activity[creep] = (QUIET_ACTIVITY + QUIET_ACTIVITY) / 2 + 0.05
    activity[(times >= 2.5) & (times < 2.8)] = 0.9
    _, closed = run(times, activity)
    assert len(closed) == 1
    backdated = 2.5 - closed[0].start
    assert backdated <= (PREROLL_FRAMES + 1) * ONE_FRAME, backdated


# ============================================================== requirement 4

def test_a_brief_dip_does_not_split_one_punch_in_two():
    """A punch passes through near zero velocity at full extension. Closing on
    the first quiet frame would report two half-punches."""
    times = np.arange(0.0, 2.0, ONE_FRAME)
    activity = np.full(times.size, BASELINE)
    throw = (times >= 0.5) & (times < 0.9)
    activity[throw] = 0.8
    dip = (times >= 0.68) & (times < 0.72)          # two frames at 30 fps
    activity[dip] = BASELINE
    _, closed = run(times, activity)
    assert len(closed) == 1
    assert closed[0].start <= 0.52 and closed[0].end >= 0.86


def test_a_real_rest_between_two_movements_does_split_them():
    rest = GAP_BETWEEN_ACTIONS
    times, activity = burst_trace(
        3.0, [(0.5, 0.25, 0.8), (0.5 + 0.25 + rest, 0.25, 0.8)])
    _, closed = run(times, activity)
    assert len(closed) == 2
    assert closed[1].start - closed[0].end >= rest - 2 * ONE_FRAME


def test_the_shortest_rest_that_splits_two_movements_fits_inside_the_gap():
    """The number to quote when asked what cannot be split.

    Anything quieter than the bridge for less than the bridge is one movement,
    by construction. What matters is whether the split threshold is below the
    263 ms that real consecutive actions leave.
    """
    splits_at = None
    for rest_frames in range(0, 12):
        rest = rest_frames * ONE_FRAME
        times, activity = burst_trace(
            3.0, [(0.5, 0.25, 0.8), (0.75 + rest, 0.25, 0.8)])
        _, closed = run(times, activity)
        if len(closed) == 2:
            splits_at = rest
            break
    assert splits_at is not None
    assert splits_at >= DIP_BRIDGE_SECONDS - ONE_FRAME
    assert splits_at < GAP_BETWEEN_ACTIONS, splits_at * 1000


def test_a_dip_longer_than_the_bridge_does_split():
    times = np.arange(0.0, 2.0, ONE_FRAME)
    activity = np.full(times.size, BASELINE)
    activity[(times >= 0.4) & (times < 0.7)] = 0.8
    activity[(times >= 0.85) & (times < 1.2)] = 0.8
    _, closed = run(times, activity)
    assert len(closed) == 2


# ============================================================== requirement 5

def test_the_segment_carries_no_time_beyond_frames_and_duration():
    """A guard against the `MOTION_DIM` bug arriving in a new place."""
    assert SEGMENT_FIELDS == ("start", "end", "peak", "frames", "energy")
    segment = Segment(start=1.0, end=1.25, peak=1.1, frames=8, energy=0.4)
    assert segment.duration == pytest.approx(0.25)
    assert segment.milliseconds == pytest.approx(250.0)
    assert Segment(1.0, None, 1.1, 4, 0.2).duration is None


def test_the_same_burst_at_two_frame_rates_gives_the_same_boundaries():
    """Frame rate moves with load. Boundaries must not."""
    burst = (1.0, 0.3, 0.8)
    slow = run(*burst_trace(3.0, [burst], fps=30.0))[1]
    fast = run(*burst_trace(3.0, [burst], fps=60.0))[1]
    assert len(slow) == len(fast) == 1
    assert abs(slow[0].start - fast[0].start) <= ONE_FRAME
    assert abs(slow[0].end - fast[0].end) <= ONE_FRAME


def test_energy_does_not_depend_on_the_frame_rate():
    """Energy is an integral. A sum of per-frame activity would double when
    the frame rate did, which is a frame counter wearing another name."""
    burst = (1.0, 0.3, 0.8)
    slow = run(*burst_trace(3.0, [burst], fps=30.0))[1][0]
    fast = run(*burst_trace(3.0, [burst], fps=90.0))[1][0]
    assert fast.frames > 2 * slow.frames
    assert fast.energy == pytest.approx(slow.energy, rel=0.15)


def test_energy_grows_with_how_much_happened():
    small = run(*burst_trace(3.0, [(1.0, 0.3, 0.4)]))[1][0]
    louder = run(*burst_trace(3.0, [(1.0, 0.3, 0.9)]))[1][0]
    longer = run(*burst_trace(3.0, [(1.0, 0.6, 0.4)]))[1][0]
    assert louder.energy > small.energy
    assert longer.energy > small.energy


def test_the_peak_is_reported_where_activity_peaked():
    times = np.arange(0.0, 2.0, ONE_FRAME)
    activity = np.full(times.size, BASELINE)
    inside = (times >= 0.5) & (times < 0.9)
    activity[inside] = 0.5
    activity[(times >= 0.8) & (times < 0.83)] = 0.95    # late peak, not centred
    _, closed = run(times, activity)
    assert len(closed) == 1
    assert abs(closed[0].peak - 0.8) <= ONE_FRAME


# ============================================================== requirement 6

def test_memory_stays_bounded_over_a_long_stream():
    """A thousand seconds of camera, then check nothing grew with it."""
    bursts = [(0.5 + i * 1.0, 0.25, 0.8) for i in range(900)]
    times, activity = burst_trace(950.0, bursts, noise=0.01, seed=11)
    segmenter, closed = run(times, activity)
    assert len(closed) > 800
    assert len(segmenter.recent) == RECENT_SEGMENTS
    ceiling = max(RECENT_SEGMENTS, PREROLL_FRAMES)
    for name, value in vars(segmenter).items():
        if isinstance(value, (list, dict, set, tuple)) or hasattr(value, "maxlen"):
            assert len(value) <= ceiling, (name, len(value))


def test_a_long_movement_is_not_cut_short_by_a_cap():
    """Someone holding a movement gets one long segment, not an arbitrary cap:
    a cap would make duration a property of the segmenter, not the movement."""
    times = np.arange(0.0, 30.0, ONE_FRAME)
    activity = np.full(times.size, BASELINE)
    activity[(times >= 1.0) & (times < 21.0)] = 0.8
    _, closed = run(times, activity)
    assert len(closed) == 1
    assert closed[0].duration == pytest.approx(20.0, abs=2 * ONE_FRAME)


# ==================================================== breaks, resets, rubbish

def test_reset_forgets_the_movement_in_progress():
    times, activity = burst_trace(3.0, [(1.0, 0.5, 0.8)])
    segmenter = Segmenter()
    for value, at in zip(activity, times):
        segmenter.update(float(value), float(at))
        if segmenter.underway:
            break
    segmenter.reset()
    assert not segmenter.underway and segmenter.current is None
    assert len(segmenter.recent) == 0 and segmenter.rejected == 0


def test_a_break_in_the_stream_closes_the_open_movement():
    segmenter = Segmenter()
    closed = []
    for i in range(12):
        got = segmenter.update(0.8, i * ONE_FRAME)
        if got is not None:
            closed.append(got)
    assert segmenter.underway and closed == []
    late = segmenter.update(0.8, 11 * ONE_FRAME + 2 * STALE_GAP_SECONDS)
    assert late is not None
    assert late.end == pytest.approx(11 * ONE_FRAME)


def test_a_break_in_the_stream_does_not_bridge_two_movements():
    """A dropped tracker is not a movement that lasted through the dropout."""
    segmenter = Segmenter()
    closed = []
    for i in range(12):
        got = segmenter.update(0.8, i * ONE_FRAME)
        if got is not None:
            closed.append(got)
    resume = 11 * ONE_FRAME + 2 * STALE_GAP_SECONDS
    for i in range(12):
        got = segmenter.update(0.8, resume + i * ONE_FRAME)
        if got is not None:
            closed.append(got)
    for i in range(12, 20):
        got = segmenter.update(0.0, resume + i * ONE_FRAME)
        if got is not None:
            closed.append(got)
    assert len(closed) == 2
    assert closed[0].end < closed[1].start


def test_a_clock_that_goes_backwards_starts_a_new_stream():
    segmenter = Segmenter()
    for i in range(12):
        segmenter.update(0.8, 10.0 + i * ONE_FRAME)
    assert segmenter.underway
    got = segmenter.update(0.8, 0.0)
    assert got is not None and got.start >= 10.0
    assert segmenter.current is not None and segmenter.current.start == 0.0


def test_a_nan_activity_is_treated_as_quiet_not_as_movement():
    times = np.arange(0.0, 2.0, ONE_FRAME)
    activity = np.full(times.size, BASELINE)
    activity[10] = np.nan
    segmenter, closed = run(times, activity)
    assert closed == [] and not segmenter.underway

    segmenter = Segmenter()
    for i in range(12):
        segmenter.update(0.8, i * ONE_FRAME)
    for i in range(12, 20):
        segmenter.update(np.nan, i * ONE_FRAME)
    assert not segmenter.underway            # the NaN run ended it, quietly
    assert len(segmenter.recent) == 1
    assert np.isfinite(segmenter.recent[0].energy)


# ================================================================ activity_of

def test_activity_is_zero_when_nothing_changed():
    vector = np.arange(10.0)
    assert activity_of(vector, vector, ACTIVITY_SPAN_SECONDS) == 0.0


def test_activity_is_bounded_between_zero_and_one():
    small = activity_of(np.zeros(8), np.full(8, 1e-6), ACTIVITY_SPAN_SECONDS)
    huge = activity_of(np.zeros(8), np.full(8, 1e6), ACTIVITY_SPAN_SECONDS)
    assert 0.0 <= small < 0.001
    assert 0.99 < huge < 1.0


def test_activity_rises_with_the_rate_of_change():
    steps = [0.05, 0.1, 0.2, 0.4, 0.8]
    values = [activity_of(np.full(12, s), np.zeros(12), ACTIVITY_SPAN_SECONDS)
              for s in steps]
    assert all(b > a for a, b in zip(values, values[1:]))


def test_activity_does_not_move_when_the_descriptor_gains_dimensions():
    """Root mean square, not a plain norm, so adding features does not
    silently raise every threshold in the module."""
    small = np.full(6, 0.3)
    doubled = np.tile(small, 3)
    assert activity_of(small, np.zeros(6), 0.1) == pytest.approx(
        activity_of(doubled, np.zeros(18), 0.1))


def test_a_tracker_glitch_cannot_dominate_the_energy():
    """One landmark jumping across the frame saturates instead of drowning the
    rest of the segment, which is what the squash is for."""
    glitch = activity_of(np.array([1e4, 0.0]), np.zeros(2), ACTIVITY_SPAN_SECONDS)
    real = activity_of(np.array([0.4, 0.4]), np.zeros(2), ACTIVITY_SPAN_SECONDS)
    assert glitch <= 1.0
    assert glitch / real < 4.0


def test_mismatched_feature_shapes_are_refused():
    with pytest.raises(ValueError):
        activity_of(np.zeros(5), np.zeros(6), 0.1)


def test_a_broken_dimension_does_not_read_as_movement():
    """An unseen joint extrapolated to a NaN must not read as a punch, and
    must not blank out the joints that were seen either."""
    now = np.array([0.1, 0.1, np.nan])
    before = np.array([0.0, 0.0, 0.0])
    mixed = activity_of(now, before, 0.1)
    clean = activity_of(np.array([0.1, 0.1]), np.zeros(2), 0.1)
    assert np.isfinite(mixed) and mixed == pytest.approx(clean)
    assert activity_of(np.full(3, np.nan), np.zeros(3), 0.1) == 0.0


def test_a_repeated_timestamp_does_not_divide_by_zero():
    value = activity_of(np.full(4, 0.1), np.zeros(4), 0.0)
    assert np.isfinite(value) and 0.0 <= value <= 1.0


# ===================================================== real movement features

def test_the_activity_span_is_what_makes_a_real_punch_visible():
    """Why `activity_of` wants 100 ms and not two adjacent frames.

    Differencing adjacent frames divides by 33 ms and multiplies per-frame
    landmark noise by 30. The measured separation between idle and a jab is
    the claim, and it is made here rather than in a comment.
    """
    def separation(span):
        idle, jab = [], []
        for seed in range(6):
            frames, truth = body_stream(["jab_left"], seed=seed)
            times, activity = feature_trace(frames, body_descriptor, span=span)
            name, began, ended = truth[0]
            inside = (times >= began) & (times <= ended)
            idle.append(np.median(activity[~inside]))
            jab.append(activity[inside].max())
        return float(np.median(jab)) / max(float(np.median(idle)), 1e-6)

    adjacent = separation(1)
    over_span = separation(SPAN_FRAMES)
    assert over_span > adjacent
    assert over_span > 10.0, (adjacent, over_span)


def test_a_real_jab_between_rest_is_bounded_where_the_jab_was():
    """Boundaries against a movement this file did not synthesise itself."""
    errors = []
    for seed in range(8):
        frames, truth = body_stream(["jab_left"], seed=seed)
        times, activity = feature_trace(frames, body_descriptor)
        _, closed = run(times, activity)
        assert len(closed) == 1, (seed, len(closed))
        name, began, ended = truth[0]
        errors.append((closed[0].start - began, closed[0].end - ended))
    starts = np.array([e[0] for e in errors])
    ends = np.array([e[1] for e in errors])
    # Late at the start by design: `bodyaction` eases in with zero velocity, so
    # the first frames of a jab genuinely barely move, and the span has to fill
    # before there is a rate to read.
    assert 0.0 <= np.median(starts) <= 4 * ONE_FRAME, np.median(starts) * 1000
    assert abs(np.median(ends)) <= 2 * ONE_FRAME, np.median(ends) * 1000


def test_real_strikes_and_guard_changes_are_found_and_rest_is_not():
    visible = ["jab_left", "cross_right", "drop_guard", "raise_guard"]
    for name in visible:
        found = 0
        for seed in range(8):
            frames, truth = body_stream([name], seed=seed)
            times, activity = feature_trace(frames, body_descriptor)
            _, closed = run(times, activity)
            found += 1 if len(closed) == 1 else 0
        assert found == 8, (name, found)

    spurious = 0
    for seed in range(12):
        frames, _ = body_stream(["idle"], seed=seed)
        times, activity = feature_trace(frames, body_descriptor)
        _, closed = run(times, activity)
        spurious += len(closed)
    assert spurious == 0, spurious


def test_a_crouch_or_a_slip_is_too_small_to_segment_from_the_upper_body():
    """Not a bug in the segmenter, an absence in the input.

    `bodyaction` records that leaking true knee positions once scored `crouch`
    at 85% on a camera that cannot observe it. With the leg quantities left out
    as guesses, a crouch barely moves anything the camera can see, and a slip
    moves the lean and the head offset by less than the landmarker's own
    jitter. Both are recorded here as failing by design, so nobody reads the
    numbers above as covering every action in `ACTIONS`.
    """
    for name in ("crouch", "slip_left"):
        peaks = []
        for seed in range(6):
            frames, truth = body_stream([name], seed=seed)
            times, activity = feature_trace(frames, body_descriptor)
            _, closed = run(times, activity)
            _, began, ended = truth[0]
            peaks.append(activity[(times >= began) & (times <= ended)].max())
            assert closed == [], (name, seed)
        assert float(np.median(peaks)) < QUIET_ACTIVITY, (name, peaks)


def test_a_real_swipe_is_bounded_where_the_swipe_was():
    found, errors = 0, []
    for seed in range(8):
        samples, (began, ended) = hand_stream("swipe_left", seed=seed)
        times, activity = feature_trace(samples, hand_descriptor)
        _, closed = run(times, activity)
        if len(closed) == 1:
            found += 1
            errors.append((closed[0].start - began, closed[0].end - ended))
    assert found >= 7, found
    starts = np.array([e[0] for e in errors])
    ends = np.array([e[1] for e in errors])
    assert 0.0 <= np.median(starts) <= 4 * ONE_FRAME, np.median(starts) * 1000
    assert abs(np.median(ends)) <= 4 * ONE_FRAME, np.median(ends) * 1000


def test_an_idling_hand_is_not_segmented():
    """`trajectory`'s idle class is deliberately more varied than any gesture,
    because idle is what a hand does almost all of the time."""
    spurious = 0
    for seed in range(10):
        samples, _ = hand_stream("none", seed=seed)
        times, activity = feature_trace(samples, hand_descriptor)
        _, closed = run(times, activity)
        spurious += len(closed)
    assert spurious == 0, spurious


def test_two_real_punches_thrown_without_rest_come_back_as_one_segment():
    """The limitation, stated as a measurement rather than left implicit.

    A jab flowing straight into a cross never drops quiet for the 100 ms the
    bridge needs, so it is one segment covering both. `live_commit.py` handles
    this case at the classifier instead: a different label inside one
    continuous movement is a real call, gated by its cooldown. Splitting it
    here would need something the activity signal does not contain.
    """
    for seed in range(4):
        frames, truth = body_stream(["jab_left", "cross_right"], seed=seed,
                                    rest=0.0)
        # Rest at the end only, so the merged segment closes and can be read.
        rng = np.random.default_rng(seed)
        frames = frames + held_body(frames[-1], int(0.6 * FPS),
                                    frames[-1].at + ONE_FRAME, rng)
        times, activity = feature_trace(frames, body_descriptor)
        _, closed = run(times, activity)
        assert len(closed) == 1, (seed, len(closed))
        assert closed[0].start <= truth[0][1] + 4 * ONE_FRAME
        assert closed[0].end >= truth[1][2] - 2 * ONE_FRAME


def test_the_same_two_punches_with_a_real_rest_between_them_do_split():
    """The same input, one rest apart, to show the merge above is the rest and
    not the segmenter refusing to split."""
    for seed in range(4):
        frames, truth = body_stream(["jab_left", "cross_right"], seed=seed,
                                    rest=GAP_BETWEEN_ACTIONS)
        times, activity = feature_trace(frames, body_descriptor)
        _, closed = run(times, activity)
        assert len(closed) == 2, (seed, len(closed))
