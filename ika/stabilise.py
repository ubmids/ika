"""Camera motion, so that "the wrist moved" cannot secretly mean "the camera moved".

Every position feature in `motion` is a displacement measured in the frame: net
displacement, path length, peak speed. All of them assume the frame is nailed
down. Handheld or panning footage breaks that assumption in the worst possible
way, because it does not add noise, it adds *signal*. Pan left while the subject
stands still and `motion.features` reports a clean, straight, fast stroke to the
right. The measurement does not become noisy, it becomes confidently wrong, and
nothing downstream can tell the difference.

This module estimates the global motion between two frames so it can be
subtracted back out. Three decisions here are deliberate.

**The subject is an outlier, not the data.** A fighter filling a third of the
frame is the largest moving thing in it, and any fit that minimises total error
over all tracked points will cheerfully decide the fighter is the camera. So the
fit is robust: RANSAC over a three parameter model, keeping the largest set of
points that agree with each other, then a median refit on that set. A subject
mask can exclude the person on top of that. If this module is ever going to be
wrong this is where it goes wrong, which is why it gets the most machinery.

**Translation and uniform scale only.** No rotation, no shear, no homography.
Not because camera roll never happens, but because every extra degree of freedom
is one more way for the model to bend itself around the subject and still look
like a good fit. Three parameters cover pans, tilts and dollies, and they make
the hardest model to fool. Roll shows up as unexplained residual, which lowers
confidence, which is the honest outcome rather than a silent misfit.

**Confidence is allowed to be zero.** On a motion blurred frame or a blank wall
there is nothing to track and the only honest answer is "no idea". A confident
wrong number would push a bogus correction into every downstream feature on that
frame, which is strictly worse than leaving the frame uncorrected. So few
inliers means low confidence, and `compensate` refuses to touch points when the
estimate is not trustworthy.

Everything in and out is frame-normalised, x by width and y by height, matching
the landmark coordinates `motion.Sample` already carries. A 640 wide clip and a
1920 wide clip of the same pan therefore report the same numbers, which is the
only way a threshold tuned on one clip means anything on another.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Work at a fixed small width regardless of input resolution. Two reasons, and
# the second is the important one. Speed: this runs every frame next to pose
# estimation inside a 263 ms budget, and optical flow cost scales with pixels.
# Consistency: detecting corners on a 1080p frame and on its 360p version finds
# different features and therefore gives slightly different answers, so pinning
# the working width is what actually makes the output resolution independent
# rather than merely resolution scaled.
WORK_WIDTH = 320

# Corner budget. More features means a better vote against the subject, but
# goodFeaturesToTrack and the LK pass are the whole cost of this module. 160
# well spread points is far more than the three parameter model needs, so the
# surplus is spent entirely on outlier resistance.
MAX_FEATURES = 160
# Ask the detector for far more than the budget, then thin them by position
# below. Corner strength alone is the wrong way to spend the budget: see
# FEATURE_GRID.
DETECT_FEATURES = 600
# Relative to the strongest corner in the frame. Low, because the threshold is
# relative: a subject in a patterned shirt against a plain wall raises the bar
# for the whole image, and at the usual 0.01 the wall stops producing corners
# entirely, leaving the subject as the only thing being tracked.
FEATURE_QUALITY = 0.003
# Minimum gap between corners, as a fraction of the working width. Keeps the
# detector from returning a dozen near duplicates on one strong corner.
FEATURE_SPACING_FRACTION = 0.012

# Features are thinned to at most FEATURE_GRID_CAP per cell of a grid over the
# frame. This is the mechanism that keeps the subject from becoming the camera
# motion, and it is worth being precise about why a plain corner detector is
# not enough. goodFeaturesToTrack ranks by corner strength, so a person in
# textured clothing against an ordinary wall takes most of the corner budget
# despite covering a minority of the frame, and RANSAC then finds its largest
# consensus inside the subject. Capping per cell makes the number of features
# from a region depend on the region's *area* rather than its contrast, so a
# subject covering a third of the frame gets about a third of the votes and
# loses. Without this, a finely textured subject over a smooth background wins
# outright, which was observed before the grid went in.
FEATURE_GRID = (8, 6)
FEATURE_GRID_CAP = 4

# Pyramidal Lucas-Kanade. Three extra levels lets a feature move roughly an
# eighth of the working width between frames, which covers a fast pan; a single
# level silently loses every fast pan and reports near zero motion for it.
LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
)

# Forward-backward check: track each point to the next frame and straight back,
# and require it to land where it started. LK reports success for points it has
# actually lost (in blur, in a repeating texture, at an occlusion edge) and this
# is the cheap test that catches them. Tolerance is frame-normalised.
FLOW_ROUNDTRIP_TOLERANCE = 0.004

# Below this many surviving tracks there is nothing to be robust with, so the
# answer is "no estimate" rather than a fit through six points.
MIN_TRACKED = 8
MIN_INLIERS = 5

# RANSAC over sampled point pairs. 64 hypotheses gives a better than 99% chance
# of drawing at least one pair that is entirely background when the background
# is only half the points, which is a worse case than a subject occupying a
# third of the frame. Fixed iteration count keeps the cost per frame flat.
RANSAC_ITERATIONS = 64
# Agreement threshold, frame-normalised. About 1.3 px at the working width, so
# roughly 2.5 px on a 640 wide source. Tight enough that the subject's points
# cannot quietly join the background consensus, loose enough to absorb LK's
# subpixel wobble.
RANSAC_TOLERANCE = 0.004
# A pair of points close together gives a scale estimate of noise over noise.
# Requiring a real baseline is what stops those hypotheses from winning with a
# wild scale that happens to fit a clump.
MIN_PAIR_BASELINE = 0.08
# One frame cannot legitimately zoom by more than a few percent at video rates.
# Anything outside this is a misfit, not a dolly, so it is rejected outright
# instead of being handed to `compensate` to divide points by.
MIN_SCALE, MAX_SCALE = 0.6, 1.6
# Fixed seed. The sampling is random but the answer must not be: an estimator
# that returns a slightly different number each run makes every downstream test
# flaky and makes a disagreement between two runs impossible to debug.
RANSAC_SEED = 20260908
# Refits after the winning hypothesis. It came from two points, so its inlier
# set is incomplete; refitting on the set and re-scoring recovers the points
# that were just outside the tolerance. Two is where it stops changing.
REFINEMENT_PASSES = 2

# Confidence shaping. Full marks needs this many agreeing points; the floor is
# MIN_INLIERS, where confidence is zero because a fit through five points is not
# evidence of anything.
CONFIDENT_INLIERS = 40
# Inliers must also be spread out. All the agreeing points in one small clump
# can pin down translation but says almost nothing about scale, and reporting a
# scale drawn from a clump is exactly the confident wrong number this module is
# built to avoid. Measured as the smaller of the two per-axis standard
# deviations, so a horizontal band of features does not pass on its x spread
# alone. Uniform coverage of the frame gives about 0.29.
SPREAD_TARGET = 0.12
# Below this, the least squares scale is undefined and it is reported as 1.
SPREAD_FLOOR = 1e-6

# When the losing tracks *also* form a coherent group moving differently, the
# frame contains two rigid motions and the winner was decided by a vote rather
# than by any evidence about which one is the camera. Below this share of the
# winner's size the majority is a real majority; approaching 1.0 it is a coin
# flip, and a coin flip must not come back as a confident number.
CONTEST_FREE_SHARE = 0.6
CONTEST_FLOOR = 0.25
# The rival must actually disagree. Leftover tracks always fit *something*
# slightly different, so without a separation floor this would penalise every
# clean frame for the noise in its own residuals.
CONTEST_MIN_SEPARATION = 4.0 * RANSAC_TOLERANCE

# Estimates weaker than this are treated as no estimate at all: reported as
# such, never applied. Chosen so that a frame where the background is genuinely
# visible and rigid passes easily and a blurred or mostly occluded frame does
# not.
MIN_TRUSTED_CONFIDENCE = 0.25

# Grow the subject mask before excluding it. A corner sitting on the subject's
# silhouette moves with the subject even though its own pixel is outside the
# mask, and silhouette corners are precisely the ones a corner detector likes
# best. Without the dilation, masking removes the subject's interior and keeps
# its outline, which is the worst half to keep.
SUBJECT_DILATION_FRACTION = 0.025

# Scale and translation are reported about the frame centre. Reporting the raw
# offset of the fitted map instead would mix the two: a pure zoom about the
# centre would come back as a large spurious translation, and a caller checking
# "did the camera pan" would see one where none happened.
FRAME_CENTRE = np.array([0.5, 0.5], dtype=np.float64)


@dataclass(frozen=True)
class Motion:
    """Global motion of the camera between two frames.

    Frozen because these get stored per frame and passed around; a caller
    quietly editing one after the fact would corrupt a history nobody thinks of
    as mutable.
    """

    dx: float          # frame-normalised translation of the frame centre, so
    dy: float          # resolution independent: +dx means content moved right
    scale: float       # >1 means the camera zoomed in or moved closer
    confidence: float  # 0 to 1, how well the estimate is actually supported
    inliers: int       # agreeing tracked points behind it, for diagnosis

    @property
    def trustworthy(self) -> bool:
        """Whether this estimate should be applied at all.

        Named rather than left as a bare comparison at each call site, because
        the one place that forgets the check is the place that silently
        reintroduces the bug this module exists to fix.
        """
        return self.confidence >= MIN_TRUSTED_CONFIDENCE

    @property
    def magnitude(self) -> float:
        """Frame-normalised size of the translation, for thresholding."""
        return float(np.hypot(self.dx, self.dy))


# What "the camera did not move, as far as we can tell" looks like. Note scale
# is 1 and not 0: `compensate` divides by it.
STILL = Motion(dx=0.0, dy=0.0, scale=1.0, confidence=0.0, inliers=0)


def _no_estimate(inliers: int = 0) -> Motion:
    """No usable estimate. Zero motion, and zero confidence to say so.

    Returning zero motion here is the safe default because `compensate` then
    becomes the identity, leaving the frame uncorrected. The alternative, some
    guess carried over from an earlier frame, would apply a correction that has
    no relationship to this frame at all.
    """
    return Motion(dx=0.0, dy=0.0, scale=1.0, confidence=0.0, inliers=int(inliers))


def _to_gray_u8(image: np.ndarray) -> np.ndarray:
    """Single channel uint8, whatever came in.

    Callers hand over frames from several places (a capture loop, a decoder, a
    test) in uint8, float32 0 to 1 and float32 0 to 255. Guessing wrong makes
    the corner detector return nothing and the failure looks like "the scene has
    no texture" rather than "the dtype was wrong".
    """
    array = np.asarray(image)
    if array.ndim == 3:
        if array.shape[2] == 4:
            array = cv2.cvtColor(array, cv2.COLOR_RGBA2GRAY)
        elif array.shape[2] == 3:
            array = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
        else:
            array = array[:, :, 0]
    if array.ndim != 2:
        raise ValueError(f"expected a 2D or 3D image, got shape {array.shape}")
    if array.dtype == np.uint8:
        return array
    array = array.astype(np.float32)
    # A float image that never exceeds 1 is 0 to 1 normalised; anything above is
    # already in 0 to 255. Sniffing this beats a flag nobody remembers to pass.
    peak = float(array.max()) if array.size else 0.0
    if peak <= 1.0:
        array = array * 255.0
    return np.clip(array, 0.0, 255.0).astype(np.uint8)


def _work_shape(height: int, width: int) -> tuple[int, int]:
    """Working (width, height), preserving aspect. Never upscales."""
    if width <= WORK_WIDTH:
        return width, height
    return WORK_WIDTH, max(1, int(round(height * WORK_WIDTH / width)))


def _feature_mask(subject: np.ndarray | None, size: tuple[int, int]) -> np.ndarray | None:
    """Turn a subject mask into the allowed-region mask the detector wants.

    Input convention: non-zero means "this pixel is the subject", because that
    is how every segmentation and pose mask upstream is shaped. cv2 wants the
    opposite, non-zero means "you may look here", and getting that inversion
    backwards would search *only* the subject, which is the exact failure this
    argument exists to prevent. Resized here rather than by the caller so a mask
    at the pose model's resolution works without anyone thinking about it.
    """
    if subject is None:
        return None
    array = np.asarray(subject)
    if array.ndim == 3:
        array = array.any(axis=2)
    if array.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {array.shape}")
    binary = (array != 0).astype(np.uint8)
    binary = cv2.resize(binary, size, interpolation=cv2.INTER_NEAREST)
    kernel_size = max(3, int(round(size[0] * SUBJECT_DILATION_FRACTION)) | 1)
    binary = cv2.dilate(binary, np.ones((kernel_size, kernel_size), np.uint8))
    return np.where(binary > 0, 0, 255).astype(np.uint8)


def _spread(corners: np.ndarray, width: int, height: int) -> np.ndarray:
    """Thin detected corners so their count follows area, not contrast.

    See FEATURE_GRID for why this exists. Corners arrive sorted strongest
    first, so keeping the first FEATURE_GRID_CAP per grid cell keeps the best
    corner in every region and throws away the surplus in the busy ones.
    Vectorised because it runs every frame: the per cell rank comes from a
    stable sort by cell followed by searchsorted for each run's start, which
    beats a Python loop over several hundred corners.
    """
    columns, rows = FEATURE_GRID
    cell_x = np.clip((corners[:, 0] * columns / max(width, 1)).astype(np.int32), 0, columns - 1)
    cell_y = np.clip((corners[:, 1] * rows / max(height, 1)).astype(np.int32), 0, rows - 1)
    cells = cell_y * columns + cell_x

    order = np.argsort(cells, kind="stable")
    grouped = cells[order]
    run_start = np.searchsorted(grouped, grouped, side="left")
    rank_in_cell = np.arange(len(grouped)) - run_start

    kept = np.sort(order[rank_in_cell < FEATURE_GRID_CAP])
    # Re-sorted into detection order so the final cap keeps the strongest of
    # the survivors rather than whichever cells happen to come first.
    return corners[kept[:MAX_FEATURES]].astype(np.float32)


def _fit(points_a: np.ndarray, points_b: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Robustly fit b = scale * a + offset. Returns (scale, offset, inlier mask).

    Both arrays are frame-normalised (N, 2). The model is uniform scale plus
    translation, which is the whole point: a least squares fit over every point
    would be dragged towards whichever coherent motion has the most points, and
    a large subject can be that. RANSAC picks the largest mutually agreeing set
    instead, so the subject has to *outnumber* the background to win rather than
    merely be present.
    """
    count = len(points_a)
    rng = np.random.default_rng(RANSAC_SEED)
    first = rng.integers(0, count, RANSAC_ITERATIONS)
    second = rng.integers(0, count, RANSAC_ITERATIONS)

    base_a = points_a[second] - points_a[first]
    base_b = points_b[second] - points_b[first]
    baseline = np.linalg.norm(base_a, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        scales = np.linalg.norm(base_b, axis=1) / np.maximum(baseline, 1e-12)
    usable = (
        (baseline >= MIN_PAIR_BASELINE)
        & np.isfinite(scales)
        & (scales >= MIN_SCALE)
        & (scales <= MAX_SCALE)
    )
    if not usable.any():
        # Every hypothesis was degenerate, which happens when all the features
        # landed in one clump. Fall back to pure translation by the median flow:
        # scale is unknowable from a clump, and claiming one would be a
        # fabrication. Confidence downstream is killed by the spread term.
        offset = np.median(points_b - points_a, axis=0)
        residual = np.linalg.norm(points_b - (points_a + offset), axis=1)
        return 1.0, offset, residual <= RANSAC_TOLERANCE

    offsets = points_b[first] - scales[:, None] * points_a[first]
    predicted = scales[:, None, None] * points_a[None, :, :] + offsets[:, None, :]
    residuals = np.linalg.norm(points_b[None, :, :] - predicted, axis=2)
    agree = (residuals <= RANSAC_TOLERANCE) & usable[:, None]
    best = int(np.argmax(agree.sum(axis=1)))
    inliers = agree[best]

    # See REFINEMENT_PASSES.
    scale, offset = scales[best], offsets[best]
    for _ in range(REFINEMENT_PASSES):
        if inliers.sum() < MIN_INLIERS:
            break
        kept_a, kept_b = points_a[inliers], points_b[inliers]
        centre_a, centre_b = kept_a.mean(axis=0), kept_b.mean(axis=0)
        spread_a, spread_b = kept_a - centre_a, kept_b - centre_b
        denominator = float((spread_a * spread_a).sum())
        if denominator > SPREAD_FLOOR:
            candidate = float((spread_a * spread_b).sum() / denominator)
            scale = float(np.clip(candidate, MIN_SCALE, MAX_SCALE))
        else:
            scale = 1.0
        # Median rather than mean for the offset. RANSAC has already removed the
        # gross outliers, but a couple of subject points can sit just inside the
        # tolerance, and a mean lets them pull the translation while a median
        # does not.
        offset = np.median(kept_b - scale * kept_a, axis=0)
        residual = np.linalg.norm(points_b - (scale * points_a + offset), axis=1)
        inliers = residual <= RANSAC_TOLERANCE

    return float(scale), np.asarray(offset, dtype=np.float64), inliers


def _confidence(
    points_a: np.ndarray, inliers: np.ndarray, residuals: np.ndarray
) -> float:
    """How much the data actually supports the fit, in 0 to 1.

    Four independent ways an estimate can be worthless, multiplied together
    because any one of them alone is enough to make the number useless:

    - too few agreeing points, so the fit is not evidence of anything;
    - a small fraction of the tracks agreeing, which means the scene is not one
      rigid background and whatever won the vote is only one of the motions in
      it;
    - agreeing points clustered in one spot, which pins translation but leaves
      scale essentially invented;
    - large residuals even among the inliers, which means the three parameter
      model does not describe what happened (roll, rolling shutter, a subject
      that is most of the frame).
    """
    total = len(points_a)
    kept = int(inliers.sum())
    if total == 0 or kept < MIN_INLIERS:
        return 0.0

    span = max(CONFIDENT_INLIERS - MIN_INLIERS, 1)
    count_term = float(np.clip((kept - MIN_INLIERS) / span, 0.0, 1.0))
    agreement_term = kept / total
    positions = points_a[inliers]
    spread = float(min(positions[:, 0].std(), positions[:, 1].std()))
    spread_term = float(np.clip(spread / SPREAD_TARGET, 0.0, 1.0))
    median_residual = float(np.median(residuals[inliers]))
    residual_term = float(np.clip(1.0 - median_residual / RANSAC_TOLERANCE, 0.0, 1.0))

    return float(count_term * agreement_term * spread_term * residual_term)


def _contest(
    points_a: np.ndarray,
    points_b: np.ndarray,
    inliers: np.ndarray,
    scale: float,
    offset: np.ndarray,
) -> float:
    """Penalty in 0 to 1 for a second rigid motion of comparable size.

    RANSAC returns the largest agreeing group, and nothing about being the
    largest makes a group the camera. So the rejected tracks get fitted too: if
    they form their own sizeable group moving somewhere else, then this frame
    has a subject big enough to have been a real candidate and the answer was a
    vote, not a measurement. This is the difference between "half my tracks are
    scattered noise", which is fine, and "half my tracks are another object",
    which is a coin flip.

    Note what this does not do. It cannot say *which* group is the camera, and
    it deliberately does not try. When the subject genuinely outnumbers the
    background the wrong group still wins; all this does is stop that answer
    coming back at full confidence.
    """
    rejected = ~inliers
    if int(rejected.sum()) < MIN_INLIERS:
        return 1.0
    rival_scale, rival_offset, rival_inliers = _fit(points_a[rejected], points_b[rejected])
    rival_count = int(rival_inliers.sum())
    if rival_count < MIN_INLIERS:
        return 1.0
    separation = float(np.linalg.norm(rival_offset - offset)) + abs(rival_scale - scale)
    if separation < CONTEST_MIN_SEPARATION:
        return 1.0

    share = rival_count / max(int(inliers.sum()), 1)
    if share <= CONTEST_FREE_SHARE:
        return 1.0
    excess = (share - CONTEST_FREE_SHARE) / max(1.0 - CONTEST_FREE_SHARE, 1e-6)
    return float(1.0 - min(excess, 1.0) * (1.0 - CONTEST_FLOOR))


def estimate(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    mask: np.ndarray | None = None,
) -> Motion:
    """Global camera motion from `previous_gray` to `current_gray`.

    Sparse optical flow on background corners, then a robust fit. `mask` marks
    the subject to exclude, non-zero meaning subject, and it applies to
    `previous_gray` because that is the frame the corners are detected in.

    Stateless on purpose. The `Stabiliser` below owns the frame history and the
    smoothing; keeping the estimator pure means a test can hand it two images
    with a known transform between them and check the number, which is the only
    way anyone can tell whether this works.
    """
    previous = _to_gray_u8(previous_gray)
    current = _to_gray_u8(current_gray)
    if previous.shape != current.shape:
        raise ValueError(
            f"frames must be the same size, got {previous.shape} and {current.shape}"
        )
    if min(previous.shape) < 2:
        return _no_estimate()

    width, height = _work_shape(*previous.shape[:2])
    if (width, height) != (previous.shape[1], previous.shape[0]):
        # INTER_AREA rather than a plain resize: downscaling by point sampling
        # aliases high frequency texture into fake corners that do not survive
        # to the next frame, which shows up as a low inlier ratio on perfectly
        # good footage.
        previous = cv2.resize(previous, (width, height), interpolation=cv2.INTER_AREA)
        current = cv2.resize(current, (width, height), interpolation=cv2.INTER_AREA)

    allowed = _feature_mask(mask, (width, height))
    corners = cv2.goodFeaturesToTrack(
        previous,
        maxCorners=DETECT_FEATURES,
        qualityLevel=FEATURE_QUALITY,
        minDistance=max(1.0, width * FEATURE_SPACING_FRACTION),
        mask=allowed,
    )
    # No corners at all: a blank wall, a blown out exposure, or a mask that
    # covered everything. Deliberately not retried without the mask, because a
    # fallback there would fit the subject and report it as camera motion with
    # full confidence, which is the single worst thing this module could do.
    if corners is None or len(corners) < MIN_TRACKED:
        return _no_estimate()

    corners = _spread(corners.reshape(-1, 2), width, height).reshape(-1, 1, 2)
    forward, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, corners, None, **LK_PARAMS)
    if forward is None:
        return _no_estimate()
    backward, status_back, _ = cv2.calcOpticalFlowPyrLK(
        current, previous, forward, None, **LK_PARAMS
    )
    if backward is None:
        return _no_estimate()

    scale_vector = np.array([width, height], dtype=np.float64)
    start = corners.reshape(-1, 2).astype(np.float64) / scale_vector
    end = forward.reshape(-1, 2).astype(np.float64) / scale_vector
    round_trip = backward.reshape(-1, 2).astype(np.float64) / scale_vector

    tracked = (
        (status.reshape(-1) == 1)
        & (status_back.reshape(-1) == 1)
        & (np.linalg.norm(round_trip - start, axis=1) <= FLOW_ROUNDTRIP_TOLERANCE)
        & np.isfinite(end).all(axis=1)
    )
    if int(tracked.sum()) < MIN_TRACKED:
        # Everything was lost between the two frames. Motion blur and hard cuts
        # both land here, and both deserve "no idea" rather than a fit through
        # the handful of points that happened to survive.
        return _no_estimate(int(tracked.sum()))

    start, end = start[tracked], end[tracked]
    fitted_scale, offset, inliers = _fit(start, end)
    if int(inliers.sum()) < MIN_INLIERS:
        return _no_estimate(int(inliers.sum()))

    residuals = np.linalg.norm(end - (fitted_scale * start + offset), axis=1)
    confidence = _confidence(start, inliers, residuals) * _contest(
        start, end, inliers, fitted_scale, offset
    )

    # Re-reference the translation to the frame centre so zoom and pan do not
    # contaminate each other. See FRAME_CENTRE.
    centred = offset - FRAME_CENTRE * (1.0 - fitted_scale)
    return Motion(
        dx=float(centred[0]),
        dy=float(centred[1]),
        scale=float(fitted_scale),
        confidence=float(confidence),
        inliers=int(inliers.sum()),
    )


def compose(first: Motion, second: Motion) -> Motion:
    """`first` then `second`, as one motion.

    Needed because the features in `motion` are computed over a window of
    frames, so what corrupts them is the camera motion accumulated across the
    whole window, not the step between the last two frames. Confidence is the
    *minimum* of the two rather than the product: a chain of thirty perfectly
    good estimates is still perfectly good, but multiplying 0.9 thirty times
    says it is worthless, and a chain is really only as good as its worst link.
    """
    scale = first.scale * second.scale
    dx = second.scale * first.dx + second.dx
    dy = second.scale * first.dy + second.dy
    return Motion(
        dx=float(dx),
        dy=float(dy),
        scale=float(scale),
        confidence=float(min(first.confidence, second.confidence)),
        inliers=int(min(first.inliers, second.inliers)),
    )


class Stabiliser:
    """Per frame camera motion, with the frame history kept for you.

    Feed it frames in order and it reports the motion since the last one. It
    also accumulates `cumulative`, the motion since `reset`, because the
    downstream features are windowed and need the total drift rather than the
    single step.
    """

    def __init__(self, smoothing: float = 0.0):
        """`smoothing` is an exponential factor in 0 to 1 over the raw estimates.

        Default zero, meaning none, and that default is a decision rather than
        laziness. Smoothing trades jitter for lag, and a lagged camera estimate
        applied to a real pan removes the wrong amount of motion on every frame
        of the pan, which is a systematic error where the jitter it removes was
        only noise. Raise it only if a caller specifically wants a stable
        readout of the camera path rather than the best per frame correction.
        """
        if not 0.0 <= smoothing < 1.0:
            raise ValueError(f"smoothing must be in [0, 1), got {smoothing}")
        self.smoothing = float(smoothing)
        self._previous: np.ndarray | None = None
        self._previous_mask: np.ndarray | None = None
        self._smoothed: Motion | None = None
        self.cumulative: Motion = Motion(0.0, 0.0, 1.0, 1.0, 0)

    def update(self, rgb: np.ndarray, subject_mask: np.ndarray | None = None) -> Motion:
        """Motion from the previous frame to this one.

        The mask handed in belongs to *this* frame, but corners are detected in
        the previous one, so the previous frame's mask is the one used and this
        one is held for next time. Using the current mask on the previous frame
        would leave the subject's trailing edge unmasked, and trailing edges are
        exactly where the strongest subject corners live.

        The first call has nothing to compare against and returns zero motion at
        zero confidence, so callers must check confidence rather than assume the
        first frame was perfectly still.
        """
        gray = _to_gray_u8(rgb)
        previous, previous_mask = self._previous, self._previous_mask
        self._previous, self._previous_mask = gray, subject_mask

        if previous is None or previous.shape != gray.shape:
            # A resolution change mid stream means the previous frame is not
            # comparable. Skipping one frame is better than fitting across a
            # rescale and reporting the rescale as a zoom.
            self._smoothed = None
            return _no_estimate()

        motion = estimate(previous, gray, previous_mask)
        if not motion.trustworthy:
            # Do not let a garbage frame enter the smoother or the accumulator.
            # One bad frame would otherwise leak into every later estimate, and
            # a permanent offset in the accumulated drift is far worse than a
            # single uncorrected frame.
            return motion

        if self.smoothing > 0.0:
            motion = self._smooth(motion)
        self.cumulative = compose(self.cumulative, motion)
        return motion

    def _smooth(self, motion: Motion) -> Motion:
        """Exponential average of the trusted estimates, confidence untouched.

        Confidence is deliberately not smoothed: it describes how well *this*
        frame supported an estimate, and averaging it would let a run of good
        frames vouch for a bad one.
        """
        previous = self._smoothed
        if previous is None:
            self._smoothed = motion
            return motion
        weight = self.smoothing
        blended = Motion(
            dx=weight * previous.dx + (1.0 - weight) * motion.dx,
            dy=weight * previous.dy + (1.0 - weight) * motion.dy,
            scale=weight * previous.scale + (1.0 - weight) * motion.scale,
            confidence=motion.confidence,
            inliers=motion.inliers,
        )
        self._smoothed = blended
        return blended

    def compensate(self, points: np.ndarray, motion: Motion) -> np.ndarray:
        """Undo `motion` on frame-normalised points, giving (N, 2) or (N, 3).

        `points` are measured in the current frame; the result is where they
        would have been had the camera not moved since the previous one. That is
        the inverse of the fitted map, so a wrist that stayed still while the
        camera panned comes back to the same coordinates and contributes zero
        displacement, which is the entire purpose of this module.

        An untrustworthy motion is applied as the identity rather than as a
        best guess. A correction built from four points on a blurred frame is
        not a smaller version of the right answer, it is an unrelated number,
        and injecting it would add error to a frame that had none of this kind.
        """
        array = np.asarray(points, dtype=np.float64)
        if array.size == 0:
            return array.copy()
        if array.ndim != 2 or array.shape[1] not in (2, 3):
            raise ValueError(f"points must be (N, 2) or (N, 3), got {array.shape}")

        out = array.copy()
        if not motion.trustworthy:
            return out
        scale = motion.scale if abs(motion.scale) > 1e-6 else 1.0
        shift = np.array([motion.dx, motion.dy], dtype=np.float64)
        out[:, :2] = (array[:, :2] - FRAME_CENTRE - shift) / scale + FRAME_CENTRE
        # A third column is left alone. Landmark z is a model relative depth
        # with no calibrated relationship to the frame, so dividing it by an
        # image space scale would be arithmetic on incompatible units.
        return out

    def reset(self) -> None:
        """Forget the stream. Call between clips or after a hard cut.

        Without this, the first frame of a new clip is fitted against the last
        frame of the old one and the cut is reported as an enormous pan.
        """
        self._previous = None
        self._previous_mask = None
        self._smoothed = None
        self.cumulative = Motion(0.0, 0.0, 1.0, 1.0, 0)
