"""Two fighters, and the question of which one is which.

Everything upstream of here tracks one body, and a fight has two. The pose
landmarker will happily return both, but it returns them as an unordered list
per frame, so `bodies[0]` is not the same person it was a frame ago. Feeding
that list straight into `posture` or `bodymotion` produces a fighter whose
guard, reach and footwork are a blend of two people, which is worse than
useless because it still looks like data.

So identity has to be assigned here, and it has to survive the three things a
fight does that a queue of joggers does not:

Occlusion. Fighters stand close enough that one hides the other for a handful
of frames. A track is therefore held, not deleted, for `max_missing` frames, so
the fighter who steps back out from behind the other gets his own id back
rather than being counted as a third person.

Crossing over. They circle, and left and right swap. Anything that matches on
x position alone swaps the ids with them. The cost below is the distance
between torso centres in the frame plus a penalty on the difference in torso
size, and the position it compares against is where the track was heading
rather than where it last was, so a fighter passing through the other keeps
going in the model too.

The clinch. This is the one that rules out greedy nearest-neighbour matching.
When two torsos are 0.05 of a frame apart and both drift the same way, the
nearest observation to fighter A is fighter B's, and greedy takes it because it
commits to the cheapest single pair before looking at what that leaves. The
assignment here is globally optimal instead: it minimises the total cost over
all pairings, which keeps both fighters on their own tracks in exactly that
case. It is the Hungarian algorithm, written out below, because scipy is not a
dependency of this project and pulling one in for a 2x2 to 4x4 matrix is not a
trade worth making.

Only the `image` landmarks are used. `world` is hip-centred, so it has thrown
away the very thing identity needs, which is where in the frame the body is.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .body import (
    LEFT_HIP, LEFT_SHOULDER, N_LANDMARKS, RIGHT_HIP, RIGHT_SHOULDER,
)

# Same floor `Body.seen` uses. The landmarker returns all 33 points always,
# inventing the ones it cannot see, so a hip behind the other fighter comes
# back as a confident number that is fiction. Averaging those into a torso
# centre drags the centre off the body and loses the match.
VISIBILITY_FLOOR = 0.5

# Torso length, hip centre to shoulder centre, is the scale reference. When
# only the shoulder line or only the hip line is visible, its width stands in
# for that length through these ratios, taken from the proportions in
# `bodysynth`. They are approximate under perspective, which is why a shape
# derived this way is flagged unmeasured and its size is not used in the cost.
SHOULDER_WIDTH_IN_TORSOS = 0.84
HIP_WIDTH_IN_TORSOS = 0.56

# Last resort when neither the shoulders nor the hips survived the visibility
# floor: the spread of whatever is visible, over a whole standing body's height
# in torso lengths. Crude, and only ever used to keep the number positive.
SPREAD_IN_TORSOS = 2.5

# A degenerate torso, a body seen exactly edge on or a frame of zeros, would
# otherwise divide by zero downstream.
MIN_TORSO_SCALE = 1e-3

# How much a size mismatch costs, relative to a positional one. Both terms are
# in frame-normalised units, so 1.5 means a torso 0.1 of the frame taller than
# the other counts as 0.15 of a frame further away. This is the term that stops
# the ids swapping when the two fighters occupy nearly the same pixels: their
# positions are then equally good explanations of both observations and only
# size separates them.
SIZE_WEIGHT = 1.5

# Matching compares against where a track was heading, not where it last was,
# because two fighters passing through each other are told apart by momentum
# when position alone has gone ambiguous. Velocity is smoothed so one jittery
# frame cannot fling the prediction off the body, capped so a bad association
# cannot fling it across the frame, and extrapolated for at most a fifth of a
# second so a long occlusion does not carry the prediction somewhere invented.
VELOCITY_MEMORY = 0.5
MAX_SPEED = 1.5          # frame widths per second
PREDICT_HORIZON = 0.2    # seconds

# Torso size is smoothed for the same reason velocity is: the shoulder and hip
# landmarks jitter, and an unsmoothed scale makes the size term noisy exactly
# when it is being relied on.
SCALE_MEMORY = 0.5

# Defaults for the tracker. Fifteen frames is half a second at 30 fps, which
# covers the occlusions a clinch produces without holding a fighter who has
# genuinely left the frame. A quarter of a frame width is further than a body
# moves between frames at any speed a fight happens at, so anything beyond it
# is a different person rather than the same one teleporting.
HELD_FRAMES = 15
MATCH_LIMIT = 0.25

# Stands in for "these two cannot be paired" inside the cost matrix. A forbidden
# pair cannot simply be left out, because the assignment needs a complete matrix
# to stay solvable, so it is priced out of every solution instead and the pairs
# that come back at this price are dropped afterwards. Far above any real cost,
# which is bounded by the frame diagonal plus the size term.
BLOCKED = 1e6

# Ids start at 1 so that a caller testing an id for truth cannot mistake the
# first fighter for no fighter.
FIRST_ID = 1


@dataclass(frozen=True)
class Shape:
    """What matching actually compares: where a torso is and how big it is.

    Deliberately small. Everything else about a body, guard height, reach,
    stance, changes several times a second while the fighter stays the same
    person, so none of it belongs in an identity cost.
    """

    centre: np.ndarray   # (2,) torso centre, frame-normalised image coordinates
    scale: float         # torso length in the same units
    measured: bool       # whether real torso anchors were visible, or this is a guess


def _visible_mean(points: np.ndarray, visibility: np.ndarray, indices) -> np.ndarray | None:
    """Mean of the landmarks in `indices` that the model actually saw."""
    kept = [i for i in indices if visibility[i] >= VISIBILITY_FLOOR]
    if not kept:
        return None
    return points[kept].mean(axis=0)


def shape_of(body) -> Shape:
    """Locate and size a body's torso from its `image` landmarks.

    The torso is used rather than the whole body because it is the rigid part:
    an arm thrown out in a punch moves the centroid of all 33 points by more
    than the fighter moved, and that fake displacement is enough to break a
    match or steal the other fighter's track.

    Only x and y are read. The image z is a relative depth guess with no
    reliable scale, and mixing it into the cost adds noise to a question the
    frame plane already answers.
    """
    image = np.asarray(body.image, dtype=np.float64)
    visibility = np.asarray(body.visibility, dtype=np.float64)
    if image.ndim != 2 or image.shape[0] != N_LANDMARKS or image.shape[1] < 2:
        raise ValueError(f"expected {N_LANDMARKS} landmarks of at least 2 axes, got {image.shape}")
    if visibility.shape != (N_LANDMARKS,):
        raise ValueError(f"expected {N_LANDMARKS} visibilities, got {visibility.shape}")

    points = image[:, :2]
    shoulders = _visible_mean(points, visibility, (LEFT_SHOULDER, RIGHT_SHOULDER))
    hips = _visible_mean(points, visibility, (LEFT_HIP, RIGHT_HIP))

    if shoulders is not None and hips is not None:
        centre = (shoulders + hips) / 2.0
        scale = float(np.linalg.norm(shoulders - hips))
        return Shape(centre, max(scale, MIN_TORSO_SCALE), True)

    # Half a torso. Position is still trustworthy, so the match can go ahead on
    # position; the size is inferred from a limb width and flagged as a guess so
    # the cost does not weigh it.
    for anchor, pair, ratio in (
        (shoulders, (LEFT_SHOULDER, RIGHT_SHOULDER), SHOULDER_WIDTH_IN_TORSOS),
        (hips, (LEFT_HIP, RIGHT_HIP), HIP_WIDTH_IN_TORSOS),
    ):
        if anchor is None:
            continue
        a, b = pair
        if visibility[a] >= VISIBILITY_FLOOR and visibility[b] >= VISIBILITY_FLOOR:
            width = float(np.linalg.norm(points[a] - points[b]))
            scale = width / ratio
        else:
            scale = MIN_TORSO_SCALE
        return Shape(anchor, max(scale, MIN_TORSO_SCALE), False)

    # No torso at all: a fighter cropped to a head and an arm, or a frame the
    # landmarker filled in entirely from inference. Better to place him roughly
    # and let `max_distance` judge the match than to drop the observation and
    # invent a new person for him next frame.
    seen = visibility >= VISIBILITY_FLOOR
    if not seen.any():
        return Shape(points.mean(axis=0), MIN_TORSO_SCALE, False)
    kept = points[seen]
    spread = float(np.linalg.norm(kept.max(axis=0) - kept.min(axis=0)))
    return Shape(kept.mean(axis=0), max(spread / SPREAD_IN_TORSOS, MIN_TORSO_SCALE), False)


def pair_cost(a: Shape, b: Shape) -> float:
    """How badly a track and an observation disagree, in frame widths.

    Distance between torso centres, plus a penalty for a difference in torso
    size. The size term is what survives a clinch: when both fighters are in
    the same place, position explains either pairing equally well and only size
    breaks the tie. It is dropped when either side's size was inferred rather
    than measured, because a guessed size would then reject a good match.
    """
    gap = float(np.linalg.norm(np.asarray(a.centre) - np.asarray(b.centre)))
    if a.measured and b.measured:
        gap += SIZE_WEIGHT * abs(a.scale - b.scale)
    return gap


def assign(cost) -> list[tuple[int, int]]:
    """Optimal one-to-one assignment minimising total cost. Hungarian, O(n^3).

    Returns (row, column) pairs, one per row when there are at least as many
    columns as rows and otherwise one per column, sorted by row.

    Written out rather than imported because the alternative is greedy, and
    greedy is wrong in the case this module exists for. Greedy takes the
    cheapest single pair and then lives with what it left behind, so two
    fighters 0.05 apart who both drift 0.04 the same way get their ids swapped:
    each observation is nearer the other man's track than his own. Minimising
    the total instead keeps them straight.
    """
    matrix = np.asarray(cost, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"cost must be a 2-D matrix, got shape {matrix.shape}")
    if matrix.size == 0:
        return []
    if not np.isfinite(matrix).all():
        raise ValueError("cost matrix must be finite; price impossible pairs out instead")
    if matrix.shape[0] > matrix.shape[1]:
        # The algorithm below wants rows to be the scarce side. Solving the
        # transpose and flipping the pairs back is exactly equivalent.
        return sorted((j, i) for i, j in assign(matrix.T))

    n, m = matrix.shape
    infinity = float("inf")
    # Row and column potentials, and p[j] = the row matched to column j, both
    # 1-based with index 0 as the scratch slot the augmenting path starts from.
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        least = [infinity] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = infinity
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                reduced = matrix[i0 - 1, j - 1] - u[i0] - v[j]
                if reduced < least[j]:
                    least[j] = reduced
                    way[j] = j0
                if least[j] < delta:
                    delta = least[j]
                    j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    least[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    return sorted((p[j] - 1, j - 1) for j in range(1, m + 1) if p[j] != 0)


@dataclass
class Track:
    """One fighter, as far as this module can tell.

    `missing` is the useful field for a caller: 0 means this track was matched
    to an observation in the frame just processed, anything higher means the
    body on it is stale and is being held in case he reappears. Drawing a held
    track without checking that would paint a fighter who is not there.
    """

    id: int
    body: object          # most recent observation, a `Body`
    at: float             # when last seen, seconds
    age: int              # frames observed
    missing: int          # consecutive frames not matched

    # Derived state, kept here rather than in a parallel table in the tracker
    # so that a track cannot go out of step with its own history.
    centre: np.ndarray = field(default_factory=lambda: np.zeros(2))
    scale: float = MIN_TORSO_SCALE
    measured: bool = False
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))


class Tracker:
    """Stable ids for several bodies across frames.

    Stateful and frame-ordered, the same contract as `PoseTracker`: feed it the
    list the landmarker returned and the timestamp, get back the live tracks.
    """

    def __init__(self, max_missing: int = HELD_FRAMES, max_distance: float = MATCH_LIMIT):
        if max_missing < 0:
            raise ValueError("max_missing cannot be negative")
        if max_distance <= 0:
            raise ValueError("max_distance must be positive")
        self.max_missing = int(max_missing)
        self.max_distance = float(max_distance)
        self._tracks: dict[int, Track] = {}
        self._next_id = FIRST_ID

    # --- the frame loop ---------------------------------------------------

    def update(self, bodies: list, at: float) -> list[Track]:
        """Take one frame of observations and return the live tracks.

        Order of business matters. Existing tracks get first refusal on the
        observations, then the leftovers become new people. Doing it the other
        way round would mint a new id for a fighter who was only briefly
        occluded, which is the failure this whole module is here to prevent.
        """
        shapes = [shape_of(body) for body in bodies]
        ids = sorted(self._tracks)

        pairs: list[tuple[int, int]] = []
        if ids and shapes:
            cost = np.full((len(ids), len(shapes)), BLOCKED, dtype=np.float64)
            for row, track_id in enumerate(ids):
                predicted = self._predict(self._tracks[track_id], at)
                for column, shape in enumerate(shapes):
                    price = pair_cost(predicted, shape)
                    # Beyond `max_distance` this is a different person, so the
                    # pair is priced out rather than merely made expensive.
                    # Without this a fighter who leaves and a stranger who
                    # arrives on the far side of the frame become the same man.
                    if price <= self.max_distance:
                        cost[row, column] = price
            pairs = [(r, c) for r, c in assign(cost) if cost[r, c] < BLOCKED]

        matched_tracks = {ids[r] for r, _ in pairs}
        matched_bodies = {c for _, c in pairs}

        for row, column in pairs:
            self._observe(self._tracks[ids[row]], bodies[column], shapes[column], at)

        for track_id in ids:
            if track_id in matched_tracks:
                continue
            track = self._tracks[track_id]
            track.missing += 1
            # Held for `max_missing` frames, dropped on the next one. Holding
            # is what returns his original id when he steps back out from
            # behind the other fighter; the bound is what stops a fighter who
            # left the frame from claiming a stranger who walks in later.
            if track.missing > self.max_missing:
                del self._tracks[track_id]

        for column, shape in enumerate(shapes):
            if column not in matched_bodies:
                self._start(bodies[column], shape, at)

        return self.active()

    def active(self) -> list[Track]:
        """Every live track, held ones included, oldest id first.

        Sorted by id rather than by insertion so that the order a caller sees
        does not depend on which fighter happened to be matched first.
        """
        return [self._tracks[i] for i in sorted(self._tracks)]

    # --- internals --------------------------------------------------------

    def _start(self, body, shape: Shape, at: float) -> Track:
        track = Track(
            id=self._next_id,
            body=body,
            at=at,
            age=1,
            missing=0,
            centre=np.asarray(shape.centre, dtype=np.float64).copy(),
            scale=shape.scale,
            measured=shape.measured,
            velocity=np.zeros(2),
        )
        self._tracks[track.id] = track
        # Ids are never reused, even after a track expires. Reusing them would
        # make two different people indistinguishable in anything already
        # recorded against the number.
        self._next_id += 1
        return track

    def _observe(self, track: Track, body, shape: Shape, at: float) -> None:
        centre = np.asarray(shape.centre, dtype=np.float64)
        elapsed = at - track.at
        if elapsed > 0:
            instant = (centre - track.centre) / elapsed
            track.velocity = VELOCITY_MEMORY * track.velocity + (1.0 - VELOCITY_MEMORY) * instant
            speed = float(np.linalg.norm(track.velocity))
            if speed > MAX_SPEED:
                track.velocity = track.velocity * (MAX_SPEED / speed)

        if shape.measured:
            # A measured size supersedes a guessed one outright, and is
            # smoothed into another measured one.
            track.scale = (
                SCALE_MEMORY * track.scale + (1.0 - SCALE_MEMORY) * shape.scale
                if track.measured
                else shape.scale
            )
            track.measured = True
        # An unmeasured observation leaves the size alone. Overwriting a real
        # torso length with one inferred from a shoulder width would poison the
        # only term that separates two fighters standing in the same place.

        track.centre = centre.copy()
        track.body = body
        track.at = at
        track.age += 1
        track.missing = 0

    def _predict(self, track: Track, at: float) -> Shape:
        """Where the track should be now if the fighter kept doing what he was.

        Matching against this rather than the last seen position is what
        carries two crossing fighters through the frames where their positions
        are identical, and what keeps a track over a fighter who kept walking
        while he was hidden.
        """
        elapsed = min(max(at - track.at, 0.0), PREDICT_HORIZON)
        return Shape(track.centre + track.velocity * elapsed, track.scale, track.measured)
