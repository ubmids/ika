"""Camera motion estimation, checked against motion we imposed ourselves.

Every test here builds its own image pair by taking a textured canvas and
moving it by an amount the test chose. The truth is therefore known exactly,
which is the only way to say whether a recovered number is right rather than
merely plausible. No camera, no clips, no weights.

The claim under test is narrow and load bearing: a pan must be reported as a
pan even when a large subject is moving the other way, and a frame with nothing
to track must say so instead of guessing.
"""

import time

import cv2
import numpy as np
import pytest

from ika.stabilise import (MIN_TRUSTED_CONFIDENCE, Motion, Stabiliser, compose,
                           estimate)

FRAME_W, FRAME_H = 640, 480
# The canvas is bigger than the frame so a crop can be taken at an offset
# without running off the edge. Sliding a window over one canvas is how a known
# translation gets imposed: no interpolation, no border fill, so the only error
# in the recovered number comes from the estimator.
CANVAS_W, CANVAS_H = 1600, 1200
ORIGIN_X, ORIGIN_Y = 400, 300

# Tolerances in frame-normalised units. 0.002 is 1.3 px on a 640 wide frame.
# Measured error on a clean pair is under 0.0001, so this is loose by an order
# of magnitude on purpose: the assertions should fail when the estimator breaks,
# not when a cv2 version changes its subpixel rounding.
POSITION_TOLERANCE = 0.002
SCALE_TOLERANCE = 0.005

# Budget for one call at 640x480. Measured at 4.2 ms on an M series laptop,
# against a 263 ms per frame budget for the whole pipeline. The headroom is for
# slower and busier machines, not for a slower implementation.
FRAME_BUDGET_MS = 12.0


def textured(height, width, seed=0, cell=10):
    """A trackable background: smooth blobs, not per pixel noise.

    Per pixel noise looks textured and tracks terribly, because downscaling and
    the pyramid both alias it into different patterns at every level, so LK
    reports lost tracks on an image that a person would call detailed. Upscaled
    coarse noise gives corners that survive a resize, which is what real scenes
    do.
    """
    rng = np.random.default_rng(seed)
    coarse = rng.integers(0, 256, (height // cell + 2, width // cell + 2), dtype=np.uint8)
    grown = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(grown, (3, 3), 0)


@pytest.fixture(scope="module")
def canvas():
    return textured(CANVAS_H, CANVAS_W, seed=1)


def view(canvas, shift_x=0, shift_y=0, width=FRAME_W, height=FRAME_H):
    """A frame in which the canvas content sits `shift` pixels further along.

    Cropping further back moves the content forwards, hence the subtraction.
    Getting that sign backwards is the easy mistake here and it would make every
    test in this file assert the negation of the truth, so it lives in one place.
    """
    x = ORIGIN_X - shift_x
    y = ORIGIN_Y - shift_y
    return canvas[y:y + height, x:x + width].copy()


def zoomed(frame, factor):
    """The same frame as if the camera had zoomed in by `factor` about the centre.

    warpAffine takes the forward source to destination map, so a point p lands
    at factor * (p - centre) + centre, which is exactly the model `estimate`
    fits. Only zoom in is safe on a same size frame: zooming out would pull in
    black border that has no counterpart to track.
    """
    assert factor >= 1.0, "zoom out needs a larger source, use zoomed_out"
    height, width = frame.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 0, factor)
    return cv2.warpAffine(frame, matrix, (width, height))


def zoomed_out_pair(canvas, factor):
    """A (previous, current) pair for a zoom out, cropped from a larger warp.

    The warp is applied to an oversized region and then cropped, so the black
    edges that a zoom out creates fall outside the frame instead of inside it,
    where they would read as a textureless region and depress confidence for the
    wrong reason.
    """
    pad = 200
    big = canvas[
        ORIGIN_Y - pad:ORIGIN_Y + FRAME_H + pad,
        ORIGIN_X - pad:ORIGIN_X + FRAME_W + pad,
    ]
    height, width = big.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 0, factor)
    warped = cv2.warpAffine(big, matrix, (width, height))
    crop = (slice(pad, pad + FRAME_H), slice(pad, pad + FRAME_W))
    return big[crop].copy(), warped[crop].copy()


def with_subject(background, subject, left, top):
    """Paste a subject onto a background frame. Occlusion, as a real one is."""
    out = background.copy()
    height, width = subject.shape[:2]
    out[top:top + height, left:left + width] = subject
    return out


def box_mask(left, top, width, height):
    """Subject mask in the module's convention: non-zero means subject."""
    mask = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    mask[top:top + height, left:left + width] = 1
    return mask


def as_rgb(gray):
    return np.repeat(gray[:, :, None], 3, axis=2)


# ---------------------------------------------------------------------------
# Recovering motion we imposed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shift_x, shift_y", [(12, 5), (-30, 8), (24, -18), (0, 11)])
def test_it_recovers_a_translation_we_imposed(canvas, shift_x, shift_y):
    """The base case. A pan of a known number of pixels must come back as that
    number of pixels, expressed as a fraction of the frame."""
    motion = estimate(view(canvas), view(canvas, shift_x, shift_y))
    assert motion.dx == pytest.approx(shift_x / FRAME_W, abs=POSITION_TOLERANCE)
    assert motion.dy == pytest.approx(shift_y / FRAME_H, abs=POSITION_TOLERANCE)
    assert motion.scale == pytest.approx(1.0, abs=SCALE_TOLERANCE)
    assert motion.confidence > 0.5


@pytest.mark.parametrize("factor", [1.03, 1.08])
def test_it_recovers_a_zoom_we_imposed(canvas, factor):
    """A dolly in or a zoom shows up as scale, and must not leak into dx or dy.
    Translation is reported about the frame centre precisely so that a pure zoom
    reads as zero pan; reporting the raw fit offset would show a large spurious
    pan here and a caller would act on it."""
    previous = view(canvas)
    motion = estimate(previous, zoomed(previous, factor))
    assert motion.scale == pytest.approx(factor, abs=SCALE_TOLERANCE)
    assert abs(motion.dx) < POSITION_TOLERANCE
    assert abs(motion.dy) < POSITION_TOLERANCE
    assert motion.confidence > 0.5


def test_it_recovers_a_zoom_out_as_well_as_a_zoom_in(canvas):
    """Scale below 1 has to survive the same path as scale above 1. The two
    point scale hypothesis uses a ratio of distances, which is asymmetric around
    1, so this direction is worth its own test."""
    factor = 0.94
    previous, current = zoomed_out_pair(canvas, factor)
    motion = estimate(previous, current)
    assert motion.scale == pytest.approx(factor, abs=SCALE_TOLERANCE)
    assert motion.confidence > 0.5


def test_a_pan_and_a_zoom_at_once_are_separated(canvas):
    """Handheld footage does both at the same time, and the two parameters must
    not be traded off against each other."""
    factor = 1.05
    previous = view(canvas)
    shifted = view(canvas, 20, -10)
    motion = estimate(previous, zoomed(shifted, factor))
    assert motion.scale == pytest.approx(factor, abs=SCALE_TOLERANCE)
    # The shift was applied before the zoom about the centre, so the reported
    # centre translation is the shift scaled up by the zoom.
    assert motion.dx == pytest.approx(factor * 20 / FRAME_W, abs=POSITION_TOLERANCE)
    assert motion.dy == pytest.approx(factor * -10 / FRAME_H, abs=POSITION_TOLERANCE)


def test_identical_frames_report_no_motion_with_full_confidence(canvas):
    """Nothing moved, and the estimator has to be able to say so without
    hedging. A jittery reading here would be subtracted from real landmarks on
    every still frame of every clip."""
    frame = view(canvas)
    motion = estimate(frame, frame)
    assert motion.dx == pytest.approx(0.0, abs=1e-6)
    assert motion.dy == pytest.approx(0.0, abs=1e-6)
    assert motion.scale == pytest.approx(1.0, abs=1e-6)
    assert motion.confidence > 0.9
    assert motion.inliers > 50


def test_the_same_pan_reads_the_same_at_every_resolution(canvas):
    """A 640 wide clip and a 1920 wide one must give the same numbers, or every
    threshold tuned on one is meaningless on the other."""
    previous, current = view(canvas), view(canvas, 18, -6)
    low = estimate(previous, current)
    high = estimate(
        cv2.resize(previous, (1920, 1440), interpolation=cv2.INTER_LINEAR),
        cv2.resize(current, (1920, 1440), interpolation=cv2.INTER_LINEAR),
    )
    assert high.dx == pytest.approx(low.dx, abs=POSITION_TOLERANCE)
    assert high.dy == pytest.approx(low.dy, abs=POSITION_TOLERANCE)
    assert high.scale == pytest.approx(low.scale, abs=SCALE_TOLERANCE)


def test_the_same_pair_gives_the_same_answer_every_time(canvas):
    """RANSAC samples randomly but the estimator is seeded, because an estimator
    that answers differently on a rerun makes every disagreement downstream
    impossible to attribute."""
    previous, current = view(canvas), view(canvas, 9, 4)
    first, second = estimate(previous, current), estimate(previous, current)
    assert first == second


# ---------------------------------------------------------------------------
# The subject must not become the camera. The point of the module.
# ---------------------------------------------------------------------------


def test_a_moving_subject_does_not_become_the_camera_motion(canvas):
    """A subject over a third of the frame, moving hard the other way, while the
    camera pans right.

    This is the failure the module exists to prevent. A least squares fit over
    all tracked points, or a corner detector left to spend its budget on the
    most textured thing in view, both come back with the subject's motion and
    call it the camera. The recovered dx must have the sign and size of the
    background pan, not the subject's.
    """
    pan_x = 10
    subject = textured(300, 400, seed=77, cell=6)  # 39% of the frame, finer texture
    previous = with_subject(view(canvas), subject, left=120, top=100)
    current = with_subject(view(canvas, pan_x, 0), subject, left=95, top=112)

    motion = estimate(previous, current)
    assert motion.dx == pytest.approx(pan_x / FRAME_W, abs=POSITION_TOLERANCE)
    assert abs(motion.dy) < POSITION_TOLERANCE
    assert motion.trustworthy, "the background estimate is usable, not merely correct"


def test_masking_the_subject_recovers_the_pan_more_confidently(canvas):
    """Excluding the subject is not required for correctness above, but it must
    help: it removes the competing motion, so more tracks agree and confidence
    goes up rather than down."""
    pan_x = 10
    subject = textured(300, 400, seed=77, cell=6)
    previous = with_subject(view(canvas), subject, left=120, top=100)
    current = with_subject(view(canvas, pan_x, 0), subject, left=95, top=112)

    unmasked = estimate(previous, current)
    masked = estimate(previous, current, box_mask(120, 100, 400, 300))
    assert masked.dx == pytest.approx(pan_x / FRAME_W, abs=POSITION_TOLERANCE)
    assert masked.confidence > unmasked.confidence
    assert masked.inliers > unmasked.inliers


def test_a_subject_that_fills_most_of_the_frame_is_recovered_only_with_a_mask(canvas):
    """The honest limit of a vote based method, pinned down so it cannot regress
    silently.

    When the subject covers most of the frame it *is* the majority of the
    trackable points, and no amount of robustness can know that the majority is
    the wrong answer. Unmasked, the estimate follows the subject. That is why
    the mask argument exists, and this test proves the mask is the fix.
    """
    pan_x = 10
    subject = textured(400, 560, seed=99, cell=5)  # 73% of the frame
    previous = with_subject(view(canvas), subject, left=40, top=40)
    current = with_subject(view(canvas, pan_x, 0), subject, left=20, top=50)

    unmasked = estimate(previous, current)
    masked = estimate(previous, current, box_mask(40, 40, 560, 400))
    assert unmasked.dx < 0, "documented limit: the subject wins the vote unmasked"
    assert masked.dx == pytest.approx(pan_x / FRAME_W, abs=POSITION_TOLERANCE)


def test_a_mask_that_covers_everything_gives_no_estimate_rather_than_the_subject(canvas):
    """There is no fallback to unmasked detection, deliberately. Falling back
    would fit the subject and report it as camera motion at full confidence,
    which is worse than every other outcome available here."""
    previous, current = view(canvas), view(canvas, 12, 0)
    everything = np.ones((FRAME_H, FRAME_W), dtype=np.uint8)
    motion = estimate(previous, current, everything)
    assert motion.confidence == 0.0
    assert motion.inliers == 0
    assert (motion.dx, motion.dy, motion.scale) == (0.0, 0.0, 1.0)


def test_two_rival_motions_of_similar_size_are_not_reported_confidently(canvas):
    """Half the frame goes one way, half goes the other, same texture in both.

    Whichever half wins the vote wins it by a hair, and nothing about winning
    makes it the camera. The number returned may be either half's motion; what
    must not happen is it coming back trusted. The same scene with both halves
    moving together has to stay confident, otherwise this is just a blanket
    penalty on textured frames.
    """
    def split(left_shift, right_shift):
        frame = np.empty((FRAME_H, FRAME_W), dtype=np.uint8)
        half = FRAME_W // 2
        frame[:, :half] = canvas[300:300 + FRAME_H, 400 - left_shift:400 - left_shift + half]
        frame[:, half:] = canvas[460:460 + FRAME_H, 400 - right_shift:400 - right_shift + FRAME_W - half]
        return frame

    contested = estimate(split(0, 0), split(14, -14))
    agreeing = estimate(split(0, 0), split(14, 14))
    assert not contested.trustworthy, f"a coin flip came back at {contested.confidence:.2f}"
    assert agreeing.confidence > 0.8, "one coherent motion must stay confident"


# ---------------------------------------------------------------------------
# Confidence has to be honest
# ---------------------------------------------------------------------------


def test_a_blank_frame_yields_no_confidence():
    """A featureless wall or a blown out exposure has nothing to track, and the
    only honest answer is zero confidence. A confident number here would be
    subtracted from real landmarks and corrupt a frame that had no camera motion
    problem at all."""
    blank = np.full((FRAME_H, FRAME_W), 128, dtype=np.uint8)
    motion = estimate(blank, blank)
    assert motion.confidence == 0.0
    assert motion.inliers == 0
    assert not motion.trustworthy


def test_a_frame_that_goes_blank_yields_no_confidence(canvas):
    """The asymmetric case: texture to track in the first frame, nothing to
    track it into. The forward and backward flow check is what catches this,
    because LK itself reports success for points it has actually lost."""
    motion = estimate(view(canvas), np.full((FRAME_H, FRAME_W), 200, dtype=np.uint8))
    assert motion.confidence < MIN_TRUSTED_CONFIDENCE
    assert not motion.trustworthy


def test_a_hard_cut_is_not_reported_as_an_enormous_pan(canvas):
    """Unrelated content in consecutive frames has no camera motion between it.
    Matching one texture into another would produce a large confident pan, and
    the correction would wreck the first frame of every shot."""
    other = textured(FRAME_H, FRAME_W, seed=4242, cell=9)
    motion = estimate(view(canvas), other)
    assert not motion.trustworthy, f"a cut came back at {motion.confidence:.2f}"


def test_a_repeating_texture_is_not_reported_confidently():
    """Blinds, brickwork, a tiled floor: patterns whose period is close to the
    motion between frames.

    The flow here is genuinely ambiguous, because a shift of one period is
    indistinguishable from no shift at all, and Lucas-Kanade happily reports
    success for a point it has locked onto the wrong repeat of. The number that
    comes back is wrong and there is nothing to be done about that; what must
    not happen is it coming back trusted, because a trusted wrong pan is
    subtracted from every landmark on the frame. Tracking each point forwards
    and straight back is what catches it: a point matched to the wrong repeat
    does not return to where it started.
    """
    period = 6.0
    columns = np.sin(np.arange(FRAME_W) / period)[None, :]
    rows = np.sin(np.arange(FRAME_H) / period)[:, None]
    pattern = np.clip(127 + 100 * columns + 100 * rows, 0, 255).astype(np.uint8)
    shifted = np.roll(pattern, 19, axis=1)

    motion = estimate(pattern, shifted)
    assert not motion.trustworthy, (
        f"an ambiguous texture came back at {motion.confidence:.2f}"
    )


def test_a_motion_blurred_pan_is_still_recovered():
    """The counterweight to the test above. Blur must not be treated as failure
    by itself: horizontal smear is what a real pan looks like, and refusing to
    estimate on blurred frames would refuse exactly the frames that need
    correcting most."""
    rng = np.random.default_rng(0)
    coarse = rng.integers(0, 256, (FRAME_H // 10 + 2, FRAME_W // 10 + 2), dtype=np.uint8)
    texture = cv2.resize(coarse, (FRAME_W, FRAME_H), interpolation=cv2.INTER_LINEAR)
    smear = np.ones((1, 45), np.float32) / 45.0
    shift = 25

    previous = cv2.filter2D(texture, -1, smear)
    current = cv2.filter2D(np.roll(texture, shift, axis=1), -1, smear)
    motion = estimate(previous, current)
    assert motion.dx == pytest.approx(shift / FRAME_W, abs=POSITION_TOLERANCE)
    assert motion.trustworthy


def test_fewer_inliers_means_lower_confidence(canvas):
    """The core promise of the confidence number. Progressively occluding the
    background with a static patch removes agreeing points without changing the
    motion, so confidence must fall monotonically while dx stays right."""
    pan_x = 12
    confidences = []
    for occlusion_width in (0, 260, 420):
        patch = textured(FRAME_H, occlusion_width, seed=5, cell=7) if occlusion_width else None
        previous, current = view(canvas), view(canvas, pan_x, 0)
        if patch is not None:
            previous = with_subject(previous, patch, left=0, top=0)
            current = with_subject(current, patch, left=0, top=0)
        motion = estimate(previous, current, box_mask(0, 0, max(occlusion_width, 1), FRAME_H))
        assert motion.dx == pytest.approx(pan_x / FRAME_W, abs=POSITION_TOLERANCE)
        confidences.append(motion.confidence)
    assert confidences[0] > confidences[1] > confidences[2]


def test_mismatched_frame_sizes_are_refused(canvas):
    """Silently resizing would report the resize as a zoom."""
    with pytest.raises(ValueError, match="same size"):
        estimate(view(canvas), view(canvas, width=320, height=240))


# ---------------------------------------------------------------------------
# Compensating points, which is what the rest of the pipeline consumes
# ---------------------------------------------------------------------------


def test_a_stationary_point_comes_back_to_where_it_was(canvas):
    """A landmark that did not move in the world but slid across the frame with
    the pan must land back on its original coordinates. Anything else and the
    features in `motion` still see displacement that nobody performed."""
    pan_x, pan_y = -22, 9
    stabiliser = Stabiliser()
    stabiliser.update(as_rgb(view(canvas)))
    motion = stabiliser.update(as_rgb(view(canvas, pan_x, pan_y)))
    assert motion.trustworthy

    resting = np.array([[0.42, 0.61], [0.70, 0.33]])
    seen_now = resting + np.array([pan_x / FRAME_W, pan_y / FRAME_H])
    recovered = stabiliser.compensate(seen_now, motion)
    assert np.allclose(recovered, resting, atol=POSITION_TOLERANCE)


def test_compensation_keeps_the_motion_the_subject_actually_made(canvas):
    """Removing camera motion must not remove the gesture with it. A wrist that
    really travelled has to still show that travel afterwards."""
    pan_x = -22
    real_travel = np.array([0.18, -0.05])
    stabiliser = Stabiliser()
    stabiliser.update(as_rgb(view(canvas)))
    motion = stabiliser.update(as_rgb(view(canvas, pan_x, 0)))

    start = np.array([[0.30, 0.50]])
    seen_now = start + real_travel + np.array([pan_x / FRAME_W, 0.0])
    recovered = stabiliser.compensate(seen_now, motion)
    assert np.allclose(recovered - start, real_travel, atol=POSITION_TOLERANCE)


def test_a_zoom_is_undone_about_the_frame_centre(canvas):
    """Scale compensation has to divide about the centre, the same point the
    estimate is referenced to. Using the origin instead would leave a
    translation behind that grows with distance from the corner."""
    factor = 1.06
    previous = view(canvas)
    stabiliser = Stabiliser()
    stabiliser.update(as_rgb(previous))
    motion = stabiliser.update(as_rgb(zoomed(previous, factor)))
    assert motion.trustworthy

    resting = np.array([[0.20, 0.80], [0.50, 0.50], [0.75, 0.25]])
    seen_now = (resting - 0.5) * factor + 0.5
    recovered = stabiliser.compensate(seen_now, motion)
    assert np.allclose(recovered, resting, atol=POSITION_TOLERANCE)


def test_a_pan_stops_faking_a_swipe(canvas):
    """The whole reason this module exists, stated as the measurement it fixes.

    A hand held perfectly still while the camera pans traces a straight,
    confident stroke across the frame. `motion.features` would read that as a
    swipe. After compensation the measured displacement has to collapse to
    nothing.
    """
    pan_per_frame = -8
    still_wrist = np.array([[0.55, 0.45]])
    stabiliser = Stabiliser()
    stabiliser.update(as_rgb(view(canvas)))

    raw, corrected = still_wrist.copy(), still_wrist.copy()
    for step in range(1, 6):
        motion = stabiliser.update(as_rgb(view(canvas, pan_per_frame * step, 0)))
        assert motion.trustworthy
        raw = raw + np.array([pan_per_frame / FRAME_W, 0.0])
        corrected = stabiliser.compensate(raw, stabiliser.cumulative)

    faked = float(np.linalg.norm(raw - still_wrist))
    left = float(np.linalg.norm(corrected - still_wrist))
    assert faked > 0.05, "the uncorrected pan should look like a real gesture"
    assert left < POSITION_TOLERANCE * 3, f"compensation left {left:.4f} of fake motion"


def test_an_untrusted_estimate_is_never_applied(canvas):
    """A correction built from four points on a ruined frame is not a rough
    version of the right answer, it is an unrelated number. Leaving the frame
    alone is strictly better than adding error to it."""
    stabiliser = Stabiliser()
    points = np.array([[0.3, 0.4], [0.6, 0.7]])
    junk = Motion(dx=0.4, dy=-0.3, scale=1.5, confidence=0.05, inliers=4)
    assert np.allclose(stabiliser.compensate(points, junk), points)


def test_the_third_coordinate_is_left_alone(canvas):
    """Landmark z is a model relative depth with no calibrated relationship to
    the frame, so dividing it by an image space scale would be arithmetic on
    incompatible units."""
    stabiliser = Stabiliser()
    points = np.array([[0.3, 0.4, -0.07], [0.6, 0.7, 0.02]])
    motion = Motion(dx=0.05, dy=0.0, scale=1.1, confidence=0.9, inliers=60)
    out = stabiliser.compensate(points, motion)
    assert out.shape == points.shape
    assert np.allclose(out[:, 2], points[:, 2])
    assert not np.allclose(out[:, :2], points[:, :2])


def test_badly_shaped_points_are_refused():
    """A silent reshape here would compensate the wrong axis and the numbers
    would still look plausible."""
    stabiliser = Stabiliser()
    motion = Motion(0.01, 0.0, 1.0, 0.9, 50)
    with pytest.raises(ValueError, match=r"\(N, 2\) or \(N, 3\)"):
        stabiliser.compensate(np.zeros((4, 5)), motion)
    assert stabiliser.compensate(np.zeros((0, 2)), motion).shape == (0, 2)


# ---------------------------------------------------------------------------
# Stream behaviour
# ---------------------------------------------------------------------------


def test_the_first_frame_has_nothing_to_compare_against(canvas):
    """It must report no confidence rather than pretending the camera was
    still, because a caller that trusts the first frame gets one uncorrected
    frame at the head of every clip."""
    stabiliser = Stabiliser()
    first = stabiliser.update(as_rgb(view(canvas)))
    assert first.confidence == 0.0 and not first.trustworthy


def test_reset_stops_the_next_frame_being_matched_against_the_old_clip(canvas):
    """Without this, the first frame of a new clip is fitted against the last
    frame of the previous one and the cut is reported as an enormous pan."""
    stabiliser = Stabiliser()
    stabiliser.update(as_rgb(view(canvas)))
    assert stabiliser.update(as_rgb(view(canvas, 10, 0))).trustworthy
    stabiliser.reset()
    assert stabiliser.update(as_rgb(view(canvas, 20, 0))).confidence == 0.0
    assert stabiliser.cumulative.dx == 0.0


def test_accumulated_drift_adds_up_across_a_window(canvas):
    """The features in `motion` are computed over a window of frames, so what
    corrupts them is the drift across the whole window rather than the step
    between the last two frames."""
    step = 7
    frames = 6
    stabiliser = Stabiliser()
    for index in range(frames + 1):
        stabiliser.update(as_rgb(view(canvas, step * index, 0)))
    assert stabiliser.cumulative.dx == pytest.approx(
        frames * step / FRAME_W, abs=POSITION_TOLERANCE
    )
    assert stabiliser.cumulative.trustworthy


def test_an_unreadable_frame_does_not_poison_the_accumulated_drift(canvas):
    """One blank frame in the middle of a clip must cost one frame of
    correction, not permanently offset every estimate that follows it."""
    stabiliser = Stabiliser()
    stabiliser.update(as_rgb(view(canvas)))
    stabiliser.update(as_rgb(view(canvas, 7, 0)))
    before = stabiliser.cumulative.dx
    blank = stabiliser.update(as_rgb(np.full((FRAME_H, FRAME_W), 90, dtype=np.uint8)))
    assert not blank.trustworthy
    assert stabiliser.cumulative.dx == before, "a junk frame changed the accumulator"


def test_a_resolution_change_mid_stream_is_skipped_not_fitted(canvas):
    """Fitting across a rescale would report the rescale itself as a zoom."""
    stabiliser = Stabiliser()
    stabiliser.update(as_rgb(view(canvas)))
    smaller = cv2.resize(view(canvas, 7, 0), (320, 240), interpolation=cv2.INTER_AREA)
    assert stabiliser.update(as_rgb(smaller)).confidence == 0.0


def test_smoothing_lags_a_pan_and_no_smoothing_does_not(canvas):
    """Smoothing trades responsiveness for steadiness, which is why the default
    is off: a lagged estimate removes the wrong amount of motion on every frame
    of a real pan, a systematic error in place of mere noise."""
    step = 9
    truth = step / FRAME_W

    def first_moving_estimate(smoothing):
        stabiliser = Stabiliser(smoothing=smoothing)
        stabiliser.update(as_rgb(view(canvas)))
        return stabiliser.update(as_rgb(view(canvas, step, 0))).dx

    assert first_moving_estimate(0.0) == pytest.approx(truth, abs=POSITION_TOLERANCE)
    assert first_moving_estimate(0.8) == pytest.approx(truth, abs=POSITION_TOLERANCE)

    # Second frame of the pan: the smoothed estimate is still catching up.
    stabiliser = Stabiliser(smoothing=0.8)
    stabiliser.update(as_rgb(view(canvas)))
    stabiliser.update(as_rgb(view(canvas, step, 0)))
    lagged = stabiliser.update(as_rgb(view(canvas, 3 * step, 0)))
    assert lagged.dx < 2 * step / FRAME_W, "a smoothed estimate must lag a jump"


def test_an_impossible_smoothing_factor_is_refused():
    with pytest.raises(ValueError, match="smoothing"):
        Stabiliser(smoothing=1.0)


def test_composing_two_motions_matches_applying_them_in_turn():
    """`compose` is what builds the accumulated drift, so its algebra is worth
    checking against the definition rather than against itself."""
    first = Motion(dx=0.02, dy=-0.01, scale=1.10, confidence=0.9, inliers=80)
    second = Motion(dx=-0.05, dy=0.03, scale=1.20, confidence=0.7, inliers=40)
    together = compose(first, second)

    point = np.array([[0.30, 0.70]])
    centre = np.array([0.5, 0.5])

    def apply(motion, points):
        return (points - centre) * motion.scale + centre + np.array([motion.dx, motion.dy])

    assert np.allclose(apply(second, apply(first, point)), apply(together, point))
    assert together.confidence == 0.7, "a chain is only as good as its worst link"


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------


def test_it_costs_only_a_few_milliseconds_per_frame(canvas):
    """This runs every frame beside pose estimation inside a 263 ms budget, so
    the cost is part of the interface. Measured at 4.2 ms per 640x480 pair on an
    M series laptop; the assertion is deliberately looser than that so it fails
    on a real regression rather than on a busy machine."""
    pairs = [(view(canvas, i, i // 2), view(canvas, i - 9, i // 2 - 4)) for i in range(20)]
    estimate(*pairs[0])  # warm the cv2 code paths, they allocate on first use

    timings = []
    for previous, current in pairs:
        started = time.perf_counter()
        estimate(previous, current)
        timings.append((time.perf_counter() - started) * 1000.0)

    median = float(np.median(timings))
    print(f"\nestimate at {FRAME_W}x{FRAME_H}: median {median:.2f} ms/frame")
    assert median < FRAME_BUDGET_MS, f"{median:.2f} ms/frame exceeds the budget"


def test_the_cost_barely_grows_with_resolution(canvas):
    """Flow runs at a fixed working width, so 1080p must cost about the same as
    480p. If this ever fails, the downscale has been lost and the cost is now
    proportional to pixels, which the frame budget cannot absorb."""
    small = [(view(canvas, i, 0), view(canvas, i - 9, 0)) for i in range(8)]
    large = [
        (
            cv2.resize(a, (1920, 1080), interpolation=cv2.INTER_LINEAR),
            cv2.resize(b, (1920, 1080), interpolation=cv2.INTER_LINEAR),
        )
        for a, b in small
    ]

    def median_ms(pairs):
        estimate(*pairs[0])
        timings = []
        for previous, current in pairs:
            started = time.perf_counter()
            estimate(previous, current)
            timings.append((time.perf_counter() - started) * 1000.0)
        return float(np.median(timings))

    at_480, at_1080 = median_ms(small), median_ms(large)
    print(f"\nestimate at 1920x1080: median {at_1080:.2f} ms/frame")
    assert at_1080 < 3.0 * at_480, f"{at_480:.2f} ms at 480p but {at_1080:.2f} ms at 1080p"
