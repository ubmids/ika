"""A hand closing on the lens, measured without any depth.

This module exists because of one hardware fact and one measurement.

The hardware fact: the MacBook camera is a plain RGB FaceTime HD. No LiDAR, no
TrueDepth, nothing that measures distance. The user sits facing it, so a punch
travels straight at the lens, and straight at the lens is the direction that
foreshortens to almost no pixel movement. The most common strike is therefore
the one `motion.py` can least see, because every feature in there is built from
image-plane displacement and a punch to the camera has hardly any.

The measurement: in `interaction.py` a strike turned out to be a *relationship*
and not a shape. Reading one body in isolation gave 42% on strikes; adding
relational features, a wrist travelling toward the other person's torso, gave
82%, and punch recall went from 21% to 84%. On this camera the second party IS
the camera, so the surviving form of that relation is "wrist approaching the
lens". This module measures that one quantity and nothing else.

How, with no depth. Under a pinhole camera apparent size goes as 1/distance, so
a rigid length on the hand growing in the image is a distance closing. The palm
is the rigid part, which is why `features.palm_scale` measures wrist to middle
knuckle, and its growth in *image* coordinates is a depth-free proxy for
closing.

Why not lean on MediaPipe's `z`. It is inferred, not measured. A synthetic
depth-error sweep kept a two-way classifier at 100% while the decision *margin*
collapsed from 44x to 1.5x, which is a classifier one bad frame away from being
wrong. So nothing here reads `z`, and `test_approach.py` proves it by filling
the `z` column with garbage and asserting the output is bit-for-bit identical.

The rate reported is *fractional* growth per second, d(ln s)/dt, and not raw
units per second. That is the difference between a usable number and a useless
one: raw growth per second depends on how big the hand already looks, so the
same punch reads several times larger when thrown from close in and no single
threshold can hold. The fractional rate is the same for the same fractional
closing at any distance, and it is the reciprocal of time-to-contact, which is
the quantity a system would actually act on.

What this cannot separate is written out at `AMBIGUITIES`, with numbers, at the
bottom of the file. The lean-in case is the important one.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .body import LEFT_HIP, LEFT_SHOULDER, RIGHT_HIP, RIGHT_SHOULDER
from .schema import INDEX_MCP, MIDDLE_MCP, PINKY_MCP, WRIST

EPS = 1e-8

# The rigid spans across the palm. `features.palm_scale` uses the first of
# these alone, which is right for its job and wrong for this one: wrist to
# middle knuckle foreshortens toward zero when the hand turns edge-on, so a
# hand rotating would read as a hand retreating fast and a hand rotating back
# would read as a punch.
#
# The apparent scale here is the *largest* of these spans, not the average.
# Rotating a rigid planar patch about an axis lying in its own plane preserves
# any span parallel to that axis, so with spans pointing in several directions
# there is always one that barely shrinks, and the maximum finds it. Swept over
# every in-plane axis at full tilt, the worst shrink is to 0.754 of face-on
# size for the maximum against 0.430 for the average. On the maximum's own
# worst axis, a flip completed in 0.3 s comes out as a fitted rate of 1.09 per
# second through the maximum and 2.23 through the average. One of those is
# under the firing threshold and the other is nearly twice over it.
PALM_SPANS = (
    (WRIST, MIDDLE_MCP),
    (WRIST, INDEX_MCP),
    (WRIST, PINKY_MCP),
    (INDEX_MCP, PINKY_MCP),
)

# Rolling window length. A punch at the lens is over in roughly 0.2 to 0.3
# seconds, so the window has to hold a whole strike; every extra tenth of a
# second of stationary lead-in dilutes the fitted rate, because the fit spans
# the whole window. Half a second holds a strike with a little context and no
# more. As in `motion.MotionWindow` the length is in seconds and not frames,
# because gestures happen at human speed while frame rate moves with load, and
# a window that shrank when the machine got busy would change what a punch is.
DEFAULT_SECONDS = 0.5

# The committed-judgement threshold, in e-folds of apparent size per second.
#
# Where it comes from. A jab from a relaxed guard covers something like 0.6 m
# to 0.25 m in 0.25 s, so apparent size grows 0.6 / 0.25 = 2.4x, which is
# ln(2.4) / 0.25 = 3.5 per second. A person leaning their whole upper body in,
# 0.8 m to 0.6 m over a full second, is ln(1.33) = 0.29 per second. A hand held
# still with landmark jitter sits under 0.2. The worst rate a pure rotation can
# fabricate is ROTATION_WORST_RATE. 1.2 sits above every one of those and a
# factor of nearly three below a real jab.
DEFAULT_RATE_THRESHOLD = 1.2

# Measured, not guessed. A synthetic palm flipped from fully edge-on to fully
# face-on in 0.3 s, on the worst of every in-plane rotation axis, produces this
# fitted rate through the largest-span scale. The endpoint ratio alone would say
# 0.94; the fit says 1.09 because a flip at constant angular speed is cos-like
# in time and not geometric, so it is steeper in the middle. 1.09 against a
# threshold of 1.2 is a margin of ten percent, which is thin, and that is
# exactly why rotation is rejected by the openness term below rather than by
# the threshold. A test pins this number so the thinness stays visible.
ROTATION_WORST_RATE = 1.09

# A rate needs at least this many samples and this much elapsed time. Two
# samples always fit a line perfectly, so a two-point rate has no residual left
# to judge it by and one bad landmark frame becomes the entire answer.
MIN_SAMPLES = 4
MIN_SPAN_SECONDS = 0.08

# Confidence reaches full strength once the window holds this much of a
# gesture. Deliberately far short of DEFAULT_SECONDS, and that is the lesson
# `motion.MotionWindow.ready` records the hard way: its readiness bar used to
# be 60% of the window length, which meant the window only agreed to look at a
# gesture for its final frame or two and the dwell timer downstream never got
# consecutive readings. A punch at the lens is complete in 0.25 s, so scaling
# confidence by span-over-window would mean never committing until 0.25 s after
# the strike landed. The bar is the length of the gesture, not the length of
# the buffer. Six samples is 0.2 s at 30 fps.
COMMIT_SPAN_SECONDS = 0.2
FULL_CONFIDENCE_SAMPLES = 6

# How close to the frame border the hand may sit, in palm scales, before its
# scale stops being trustworthy. A hand at the edge is partly outside the
# image, and MediaPipe still returns all 21 points, extrapolating the ones it
# cannot see, so the spans come back as confident-looking fiction. Half a palm
# of clearance is where the extrapolation starts to bite.
EDGE_MARGIN_SCALES = 0.5
EDGE_FLOOR = 0.15   # multiplier for a hand right up against the border

# How face-on the palm plane looks, as the area of the wrist-index-pinky
# triangle over the square of the scale. The synthetic palm measures 0.37
# face-on and 0.0 edge-on; the reference is set below the synthetic value so a
# real palm, whose knuckles sit differently, saturates rather than being
# penalised for anatomy. This is what makes rotation show up as low confidence
# and not just as a bounded error: apparent size is genuinely unmeasurable on a
# palm seen edge-on. Only the wrist and the knuckles are used, so a fist and an
# open palm give the same answer, which is correct because curling a finger
# does not turn the hand.
REFERENCE_OPENNESS = 0.30
OPENNESS_FLOOR = 0.25

# Openness is also the discriminator that catches the one rotation this cannot
# bound by geometry alone: a hand flipping face-on in under a quarter of a
# second, which fabricates a rate of 1.59 per second and would otherwise fire.
# Openness is a ratio of area to size squared, so it is scale free: a hand
# genuinely closing on the lens leaves it unchanged, while a hand turning moves
# it hard. So a fast change in openness means the growth in size is explained
# by rotation and is not evidence of closing. Measured: the 0.2 s flip turns at
# 18 per second, a punch at 0. The tolerance is set at 2.0 so that a corkscrew
# punch, which does turn the fist somewhat as it extends, still commits.
TURN_TOLERANCE = 2.0
MIN_OPENNESS = 0.01   # log floor, so a perfectly edge-on frame is finite

# Scatter in the log-scale fit, in log units, at which confidence halves. 0.08
# is 8% frame-to-frame wobble in apparent size, roughly what landmark noise
# gives on a still hand. Past that the series is not a clean expansion and the
# fitted slope is being driven by noise.
FIT_TOLERANCE = 0.08

# Below this, `closing` stays False whatever the rate says. A hand at the frame
# edge should give low confidence, not a confident wrong number.
MIN_CONFIDENCE = 0.35

# Lateral travel, in palm scales per second, above which a non-closing verdict
# is explained as a sweep rather than as a hand sitting still. Diagnostic only.
# It never turns a closing verdict off, because a hook comes in at an angle and
# is both lateral and closing, and gating on it would throw away the strike
# this module is for.
SWEEP_LATERAL_PER_SECOND = 1.5


@dataclass(frozen=True)
class Approach:
    """What the tracker can honestly say about one frame.

    `rate` is the hand's own fractional growth. `body_rate` is the same
    quantity for the torso when a body was supplied, and 0.0 when it was not.
    Both are reported and not only their difference, because "the hand grew 2.0
    and so did the body" and "nothing grew at all" are the same non-strike but
    very different situations, and collapsing them would hide the one case this
    module cannot resolve by itself.
    """

    scale: float             # apparent size now, in frame widths
    rate: float              # growth per second; positive means closing
    closing: bool            # committed judgement
    confidence: float        # 0-1
    body_rate: float = 0.0   # the torso's own growth, when a body was given
    reason: str = ""         # why the judgement went the way it did


def _image_xy(marks, aspect: float) -> np.ndarray:
    """Frame-normalised x and y, with y put into width units. Never `z`.

    Accepts a `hands.Hand`, a `body.Body`, or a bare landmark array, so
    synthetic fixtures and live detections go through identical code.

    Reading `.image` and not `.world` is the whole point of this module. World
    landmarks are hand-centred and metric, so a hand at arm's length and a hand
    at the lens have the *same* world palm length. Every bit of the closing
    information lives in the image lane.

    An (N, 2) array is accepted as well as (N, 3) because the functions in here
    hand each other points that are already projected. Reshaping such an array
    to (-1, 3) would silently reinterpret 21 points as 14, and running it
    through the aspect divide a second time would squash y twice.
    """
    array = np.asarray(getattr(marks, "image", marks), dtype=np.float64)
    if array.ndim != 2 or array.shape[1] not in (2, 3):
        array = array.reshape(-1, 3)
    points = array[:, :2].copy()
    if aspect != 1.0:
        points[:, 1] /= aspect
    return points


def apparent_scale(hand, aspect: float = 1.0) -> float:
    """Image-space palm length, in frame widths. Uses x and y only.

    The largest of `PALM_SPANS`, for the rotation reason set out there.

    `aspect` is frame width divided by frame height. MediaPipe normalises x by
    width and y by height, so on a 16:9 frame a vertical span of 0.1 covers
    1.78 times as many pixels as a horizontal one, and the two cannot be mixed
    in a Euclidean norm without converting. Left at 1.0 nothing is converted,
    which is exact on a square frame and harmless for the *rate* on any frame
    so long as the hand does not turn in the image plane. It stops being
    harmless the moment it does. A hand rolling 90 degrees on a 16:9 frame has
    each of its spans stretched by 1.78 as it swings onto the vertical axis;
    taking the largest span absorbs most of that, but 1.22x survives, which
    over a 0.2 s roll is a fabricated 0.98 per second and in the other
    direction 1.23, just over the firing threshold. What stops it there is the
    same openness term that stops a real rotation, and passing the real aspect
    removes it outright: the corrected scale comes back constant to within a
    part in 10^5. Pass the aspect when the caller knows it.
    """
    points = _image_xy(hand, aspect)
    return max(
        max(float(np.linalg.norm(points[a] - points[b])) for a, b in PALM_SPANS),
        EPS,
    )


def palm_openness(hand, aspect: float = 1.0) -> float:
    """Palm triangle area over scale squared: how face-on the hand looks.

    Near REFERENCE_OPENNESS when the palm plane faces the camera, near zero
    when it is edge-on. Used to discount confidence, because on an edge-on palm
    apparent size is not measurable rather than merely noisy.
    """
    points = _image_xy(hand, aspect)
    u = points[INDEX_MCP] - points[WRIST]
    v = points[PINKY_MCP] - points[WRIST]
    # Written out rather than np.cross: numpy 2 removed the 2-vector form, and
    # promoting to 3-vectors just to take one determinant would allocate on
    # every frame of the live loop.
    area = 0.5 * abs(float(u[0] * v[1] - u[1] * v[0]))
    scale = apparent_scale(points, aspect=1.0)
    return area / max(scale * scale, EPS)


def body_scale(body, aspect: float = 1.0) -> float:
    """Torso length in the image, in frame widths. Uses x and y only.

    Hip centre to shoulder centre, the reference `posture.torso_length` uses
    and for the same reason: both ends sit on the rigid torso, so a limb moving
    cannot change it, and something like standing height would make a crouch
    read as a person walking away. It is reimplemented here rather than
    imported because `posture.torso_length` takes the norm over all three
    coordinates, and the third is the inferred `z` this module refuses to
    touch.
    """
    points = _image_xy(body, aspect)
    shoulders = (points[LEFT_SHOULDER] + points[RIGHT_SHOULDER]) / 2.0
    hips = (points[LEFT_HIP] + points[RIGHT_HIP]) / 2.0
    return max(float(np.linalg.norm(shoulders - hips)), EPS)


def approach_rate(scales, times) -> float:
    """Fractional growth per second: the slope of ln(scale) against time.

    Fitted by least squares over the whole window rather than differenced end
    to end, because a single bad landmark frame at either end of a difference
    becomes the entire answer, and on a hand thrown at the lens the end frames
    are exactly where the hand is most motion-blurred.

    The fit is on the logarithm, which is what makes the result a fraction per
    second and so independent of how big the hand already looks. It also means
    a geometric ramp, which is what constant-speed closing produces under a
    pinhole camera, is recovered exactly instead of approximately.

    Returns 0.0, meaning "no evidence", rather than raising, when there is too
    little history. A live loop asks every frame and starts cold, and raising
    on a cold start would force every caller to wrap the call.
    """
    t = np.asarray(times, dtype=np.float64).ravel()
    s = np.asarray(scales, dtype=np.float64).ravel()
    if t.size != s.size:
        raise ValueError(f"{s.size} scales against {t.size} times")
    if t.size < MIN_SAMPLES or float(t[-1] - t[0]) < MIN_SPAN_SECONDS:
        return 0.0

    logs = np.log(np.maximum(s, EPS))
    centred = t - t.mean()
    spread = float(centred @ centred)
    if spread < EPS:
        return 0.0
    return float((centred @ (logs - logs.mean())) / spread)


def _fit_scatter(scales, times, rate: float) -> float:
    """Root-mean-square residual of the log-linear fit, in log units.

    Kept out of `approach_rate` so that function stays a single dot product for
    callers that only want the number.
    """
    t = np.asarray(times, dtype=np.float64).ravel()
    logs = np.log(np.maximum(np.asarray(scales, dtype=np.float64).ravel(), EPS))
    if t.size < MIN_SAMPLES:
        return 0.0
    residual = logs - (logs.mean() + rate * (t - t.mean()))
    return float(np.sqrt(residual @ residual / residual.size))


def frame_reliability(hand, aspect: float = 1.0) -> float:
    """How far a single frame's scale can be trusted, from the image alone.

    Three ways it can be wrong, and they compose by multiplication because any
    one of them is disqualifying on its own.

    Landmarks outside [0, 1] are extrapolations of a hand partly out of shot,
    so spans built from them are invented. A hand still inside the frame but
    hugging a border is one frame from that. And a palm seen edge-on has no
    measurable size at all, however well centred it is.
    """
    points = _image_xy(hand, aspect)
    inside = np.all((points >= 0.0) & (points <= 1.0), axis=1)
    coverage = float(inside.mean())
    # Squared, so losing a third of the hand costs more than a third of the
    # confidence. A truncated palm does not degrade gracefully, it reads as a
    # differently sized hand.
    coverage *= coverage

    scale = apparent_scale(points, aspect=1.0)
    margin = float(
        min(points[:, 0].min(), points[:, 1].min(),
            1.0 - points[:, 0].max(), 1.0 - points[:, 1].max())
    )
    room = float(np.clip(margin / (EDGE_MARGIN_SCALES * max(scale, EPS)), 0.0, 1.0))
    edge = EDGE_FLOOR + (1.0 - EDGE_FLOOR) * room

    face_on = float(np.clip(palm_openness(points, aspect=1.0) / REFERENCE_OPENNESS, 0.0, 1.0))
    openness = OPENNESS_FLOOR + (1.0 - OPENNESS_FLOOR) * face_on

    return coverage * edge * openness


# Column offsets into the window array. Named because removing one feature from
# `motion.py` once shifted every index after it and the tests kept passing
# while asserting things about the wrong columns.
AT, SCALE, BODY, WRIST_X, WRIST_Y, RELIABILITY, OPENNESS = range(7)
COLUMNS = 7


class ApproachTracker:
    """Rolling, streaming judgement of whether a hand is closing on the lens.

    Stateful and fed one frame at a time, the same contract as
    `motion.MotionWindow` and `hands.HandTracker`, so it drops into the live
    loop without a second buffering scheme.
    """

    def __init__(
        self,
        seconds: float = DEFAULT_SECONDS,
        threshold: float = DEFAULT_RATE_THRESHOLD,
        aspect: float = 1.0,
    ):
        self.seconds = float(seconds)
        self.threshold = float(threshold)
        self.aspect = float(aspect)
        self._window: deque[tuple[float, ...]] = deque()

    def reset(self) -> None:
        """Forget the window. Called when the hand is lost or re-acquired.

        A scale series spanning a gap is not a slow approach, it is two
        different observations with a hole between them, and fitting a line
        through the hole invents whatever rate the hole implies.
        """
        self._window.clear()

    def __len__(self) -> int:
        return len(self._window)

    def update(self, hand, at: float, body=None) -> Approach:
        """Take one frame and judge it. `body` is optional and worth supplying.

        Without a body this can only say "the hand is growing". With one it can
        say "the hand is growing faster than the person is", which is the
        difference between a punch and someone leaning toward the screen. It is
        the same correction `interaction.py` needed: measured against the other
        party rather than in isolation, which is what took strikes there from
        42% to 82%.
        """
        if hand is None:
            # Not a degraded reading, an absence of one, the same call
            # `interaction.pair_from_frame` makes for a lone body.
            self.reset()
            return Approach(scale=0.0, rate=0.0, closing=False, confidence=0.0,
                            reason="no hand")

        if self._window and at < self._window[-1][AT]:
            # Time went backwards, so this is a different stream and the window
            # in hand describes something else.
            self.reset()

        points = _image_xy(hand, self.aspect)
        scale = apparent_scale(points, aspect=1.0)
        self._window.append((
            float(at),
            scale,
            body_scale(body, aspect=self.aspect) if body is not None else np.nan,
            float(points[WRIST, 0]),
            float(points[WRIST, 1]),
            frame_reliability(points, aspect=1.0),
            max(palm_openness(points, aspect=1.0), MIN_OPENNESS),
        ))
        while self._window and at - self._window[0][AT] > self.seconds:
            self._window.popleft()

        return self._judge(scale)

    def _judge(self, scale: float) -> Approach:
        window = np.asarray(self._window, dtype=np.float64).reshape(-1, COLUMNS)
        times = window[:, AT]
        span = float(times[-1] - times[0]) if len(times) > 1 else 0.0

        if len(times) < MIN_SAMPLES or span < MIN_SPAN_SECONDS:
            # An honest cold start. A rate from two frames is a report of the
            # noise between them.
            return Approach(scale=scale, rate=0.0, closing=False, confidence=0.0,
                            reason="cold")

        scales = window[:, SCALE]
        rate = approach_rate(scales, times)

        bodies = window[:, BODY]
        have_body = bool(np.isfinite(bodies).all())
        body_growth = approach_rate(bodies, times) if have_body else 0.0

        # The relative rate is what the judgement is made on. If the whole
        # person grew at the rate the hand did, nothing was thrown: the user
        # moved, or the chair did.
        relative = rate - body_growth

        # How fast the palm is turning, as a fraction per second, measured the
        # same way the size is. Scale free, so a hand closing on the lens
        # leaves it at zero and only a rotation moves it.
        turn = abs(approach_rate(window[:, OPENNESS], times))
        confidence = self._confidence(window, rate, span, turn)

        travel = float(np.linalg.norm(np.diff(window[:, WRIST_X:WRIST_Y + 1], axis=0), axis=1).sum())
        lateral = travel / max(span, EPS) / max(float(np.median(scales)), EPS)

        closing = bool(relative > self.threshold and confidence >= MIN_CONFIDENCE)
        if closing:
            reason = "closing"
        elif confidence < MIN_CONFIDENCE:
            # Reported ahead of the geometry on purpose: an unreliable scale
            # means the rate is not evidence either way, and calling it
            # "steady" would be the confident wrong number this is meant to
            # avoid. Rotation gets named separately, because "the hand turned"
            # is an answer a caller can act on and "unreliable" is not.
            reason = "turning" if turn > TURN_TOLERANCE else "unreliable"
        elif have_body and rate > self.threshold:
            reason = "lean"
        elif relative < -self.threshold:
            reason = "retreating"
        elif lateral > SWEEP_LATERAL_PER_SECOND:
            reason = "sweep"
        else:
            reason = "steady"

        return Approach(
            scale=scale,
            rate=rate,
            closing=closing,
            confidence=confidence,
            body_rate=body_growth,
            reason=reason,
        )

    def _confidence(self, window: np.ndarray, rate: float, span: float,
                    turn: float) -> float:
        """Multiplicative, because any one of these failing is disqualifying.

        Four independent ways the number can be wrong: too little history, a
        badly placed or badly turned hand, a hand turning *while* it grows, and
        a scale series that is not a clean expansion at all. A hand at the edge
        of the frame with a beautiful fit still has an unreliable scale, so
        these compose by multiplying and not by averaging.

        The per-frame reliability is averaged over the window rather than taken
        from the newest frame, because the rate is fitted over the window. At
        the peak of a real punch the hand fills the frame and is clipped by it,
        so the last frame alone is always poor; the fit is still anchored on the
        clean earlier frames, and judging it by its worst input would mean this
        module could never fire on the strike it was built for.
        """
        times = window[:, AT]
        history = min(1.0, len(times) / FULL_CONFIDENCE_SAMPLES)
        history *= min(1.0, span / COMMIT_SPAN_SECONDS)
        placement = float(window[:, RELIABILITY].mean())
        fit = 1.0 / (1.0 + _fit_scatter(window[:, SCALE], times, rate) / FIT_TOLERANCE)
        rotating = 1.0 / (1.0 + turn / TURN_TOLERANCE)
        return float(np.clip(history * placement * fit * rotating, 0.0, 1.0))


# What this cannot separate, written out because the alternative is a number
# that looks trustworthy and is not. Every figure here is asserted in
# `tests/test_approach.py` so it stays true as the code moves.
#
# 1. A lean-in, with no body in view. Given only a hand, a person moving their
#    whole upper body toward the screen grows the hand at exactly the rate the
#    body closes, and there is nothing in the hand alone that says which
#    happened. Passing `body=` removes it: the two growths cancel and the
#    verdict comes back "lean". Without a body, a lean fast enough to clear the
#    threshold is called closing. The measured boundary is a fitted rate of
#    1.25 per second, which at 0.6 m from the lens is a closing speed of
#    0.75 m/s: a lunge rather than a lean, but a person shifting forward hard
#    in a chair will reach it.
# 2. The camera moving instead of the hand. Identical in the image and
#    unresolvable from image data alone. Tolerable here only because the laptop
#    sits on a desk; it would not be on the glasses.
# 3. Rotation. Rejected, but by confidence and not by the rate. A full edge-on
#    to face-on flip fabricates 1.09 per second over 0.3 s and 1.59 over 0.2 s,
#    and the second of those clears the threshold. What stops it is that the
#    same flip turns the palm at 18 per second while a punch turns it at 0, so
#    the verdict comes back "turning" with confidence 0.09. A rotation slow
#    enough to stay under TURN_TOLERANCE is also too slow to fabricate a
#    firing rate, which is what makes the pair of tests sufficient.
# 4. A strike thrown edge-on. A knife hand presents no palm plane, so its
#    apparent size is not measurable rather than merely noisy, and confidence
#    is capped near OPENNESS_FLOOR. It comes back "unreliable" and not
#    "closing", which is a miss, and a deliberate one.
# 5. What the hand is doing. An open palm pushed at the lens and a fist thrown
#    at it read the same, by design. Pose is `features`' job, and mixing the
#    two in here would entangle what the interaction lane worked to keep apart.
AMBIGUITIES = (
    "a lean-in with no body in view",
    "the camera moving instead of the hand",
    "rotation faster than a quarter of a second",
    "a strike thrown edge-on, reported as unreliable",
    "pose: a push and a punch read alike",
)
