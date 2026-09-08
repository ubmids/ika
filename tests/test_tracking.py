"""Identity across frames, and the three ways a fight breaks it.

The bodies here are built by hand rather than detected, so no camera, model or
network is involved and the truth of every frame is known before the tracker
runs. What is being tested is not whether the landmarker sees two fighters, it
is whether the fighter who was id 1 is still id 1 after he is occluded, after
he crosses over, and after he is clinched up with the other man.

The claims that matter are the four marked critical below. The rest guard the
mechanisms those four depend on.
"""

import itertools

import numpy as np
import pytest

from ika.body import (Body, LEFT_ANKLE, LEFT_EAR, LEFT_ELBOW, LEFT_HIP,
                      LEFT_KNEE, LEFT_SHOULDER, LEFT_WRIST, N_LANDMARKS, NOSE,
                      RIGHT_ANKLE, RIGHT_EAR, RIGHT_ELBOW, RIGHT_HIP,
                      RIGHT_KNEE, RIGHT_SHOULDER, RIGHT_WRIST)
from ika.tracking import (MATCH_LIMIT, SIZE_WEIGHT, Shape, Track, Tracker,
                          assign, pair_cost, shape_of)

FPS = 30.0
STEP = 1.0 / FPS

# Proportions of torso length, the same ones `bodysynth` uses, so a synthetic
# body is at least shaped like the thing the tracker will meet.
SHOULDER_HALF = 0.42
HIP_HALF = 0.28
LIMB = 0.5


def fighter(x: float, y: float = 0.5, size: float = 0.20, hidden=()) -> Body:
    """A body with its torso centred on (x, y) and `size` torso length.

    Coordinates are frame-normalised the way MediaPipe's image landmarks are,
    y growing downward. `hidden` names landmarks the model could not see, which
    it still returns as a guess; those get a low visibility and a deliberately
    wrong position so that any code failing to ignore them shows up.
    """
    marks = np.zeros((N_LANDMARKS, 3), dtype=np.float32)
    half = size / 2.0
    shoulder_y, hip_y = y - half, y + half

    marks[LEFT_SHOULDER] = (x - SHOULDER_HALF * size, shoulder_y, 0.0)
    marks[RIGHT_SHOULDER] = (x + SHOULDER_HALF * size, shoulder_y, 0.0)
    marks[LEFT_HIP] = (x - HIP_HALF * size, hip_y, 0.0)
    marks[RIGHT_HIP] = (x + HIP_HALF * size, hip_y, 0.0)
    marks[NOSE] = (x, shoulder_y - 0.3 * size, 0.0)
    marks[LEFT_EAR] = (x - 0.1 * size, shoulder_y - 0.3 * size, 0.0)
    marks[RIGHT_EAR] = (x + 0.1 * size, shoulder_y - 0.3 * size, 0.0)
    marks[LEFT_ELBOW] = (x - 0.5 * size, y, 0.0)
    marks[RIGHT_ELBOW] = (x + 0.5 * size, y, 0.0)
    marks[LEFT_WRIST] = (x - 0.3 * size, shoulder_y + 0.1 * size, 0.0)
    marks[RIGHT_WRIST] = (x + 0.3 * size, shoulder_y + 0.1 * size, 0.0)
    marks[LEFT_KNEE] = (x - HIP_HALF * size, hip_y + LIMB * size, 0.0)
    marks[RIGHT_KNEE] = (x + HIP_HALF * size, hip_y + LIMB * size, 0.0)
    marks[LEFT_ANKLE] = (x - HIP_HALF * size, hip_y + 2 * LIMB * size, 0.0)
    marks[RIGHT_ANKLE] = (x + HIP_HALF * size, hip_y + 2 * LIMB * size, 0.0)

    visibility = np.full(N_LANDMARKS, 0.95, dtype=np.float32)
    for index in hidden:
        visibility[index] = 0.05
        marks[index] = (x + 0.5, y + 0.5, 0.0)   # the model's guess, off the body

    return Body(image=marks, world=marks.copy(), visibility=visibility)


def ids(tracks) -> list[int]:
    return [t.id for t in tracks]


def id_at(tracks, x: float) -> int:
    """The id of the track whose torso centre is nearest x. How a caller asks
    "which id is the fighter on the left" without knowing the answer."""
    return min(tracks, key=lambda t: abs(t.centre[0] - x)).id


# --- the shape a match is made from ---------------------------------------

def test_the_torso_centre_sits_between_the_hips_and_the_shoulders():
    shape = shape_of(fighter(0.4, 0.5, size=0.2))
    assert shape.centre == pytest.approx([0.4, 0.5], abs=1e-6)
    assert shape.scale == pytest.approx(0.2, abs=1e-6)
    assert shape.measured


def test_a_thrown_punch_does_not_move_the_body():
    """The torso is used instead of all 33 points because an arm shot out in
    front moves a whole-body centroid further than the fighter moved, and that
    fake displacement is enough to lose the match or steal the other track."""
    still = fighter(0.4)
    punching = fighter(0.4)
    reach = punching.image.copy()
    reach[RIGHT_WRIST] = (0.4 + 0.30, 0.48, 0.0)
    reach[RIGHT_ELBOW] = (0.4 + 0.18, 0.49, 0.0)
    punching = Body(image=reach, world=reach, visibility=punching.visibility)
    assert pair_cost(shape_of(still), shape_of(punching)) < 1e-6


def test_landmarks_the_model_could_not_see_are_ignored():
    """The landmarker invents hidden points, so a body whose shoulders are
    behind the other fighter reports two confident numbers that are fiction.
    Averaging them in drags the torso centre off the body entirely."""
    honest = shape_of(fighter(0.4, 0.5, size=0.2))
    occluded = shape_of(fighter(0.4, 0.5, size=0.2, hidden=(LEFT_SHOULDER, RIGHT_SHOULDER)))
    assert occluded.centre[0] == pytest.approx(0.4, abs=1e-6)
    assert abs(occluded.centre[1] - honest.centre[1]) < 0.11
    assert not occluded.measured, "a size inferred from the hip line is a guess, not a measurement"


def test_a_guessed_size_is_not_charged_against_a_match():
    """A size inferred from half a torso is unreliable, so pricing it in would
    reject a perfectly good match on a fighter who is partly hidden."""
    measured = Shape(np.array([0.4, 0.5]), 0.20, True)
    guessed = Shape(np.array([0.4, 0.5]), 0.40, False)
    assert pair_cost(measured, guessed) == pytest.approx(0.0)
    other = Shape(np.array([0.4, 0.5]), 0.40, True)
    assert pair_cost(measured, other) == pytest.approx(SIZE_WEIGHT * 0.20)


def test_a_body_with_the_wrong_landmark_count_is_refused():
    broken = Body(image=np.zeros((21, 3)), world=np.zeros((21, 3)), visibility=np.zeros(21))
    with pytest.raises(ValueError, match="expected 33"):
        shape_of(broken)


# --- optimal assignment ----------------------------------------------------

def brute_force(cost):
    """Optimal assignment by trying every pairing. Only viable for tiny
    matrices, which is exactly why it can be trusted as the reference."""
    cost = np.asarray(cost, dtype=float)
    n, m = cost.shape
    if n <= m:
        best = min(itertools.permutations(range(m), n), key=lambda c: sum(cost[range(n), list(c)]))
        return sum(cost[range(n), list(best)])
    best = min(itertools.permutations(range(n), m), key=lambda r: sum(cost[list(r), range(m)]))
    return sum(cost[list(best), range(m)])


@pytest.mark.parametrize("shape", [(1, 1), (2, 2), (2, 3), (3, 2), (4, 4), (3, 5), (5, 3)])
def test_the_assignment_is_optimal_not_merely_reasonable(shape):
    rng = np.random.default_rng(7)
    for _ in range(40):
        cost = rng.random(shape)
        pairs = assign(cost)
        assert len(pairs) == min(shape)
        assert len({r for r, _ in pairs}) == len(pairs), "a row cannot be used twice"
        assert len({c for _, c in pairs}) == len(pairs), "a column cannot be used twice"
        total = sum(cost[r, c] for r, c in pairs)
        assert total == pytest.approx(brute_force(cost))


def test_greedy_matching_swaps_the_clinch_and_optimal_matching_does_not():
    """The reason the Hungarian algorithm is in this project at all. Two tracks
    0.05 apart, both fighters drifting 0.04 the same way: each observation is
    nearer the other man's track than his own, so greedy takes the cheapest
    single pair and swaps the ids. The total-cost solution keeps them."""
    cost = np.array([[0.04, 0.09],
                     [0.01, 0.04]])

    greedy: dict[int, int] = {}
    taken: set[int] = set()
    for row, column in sorted(np.ndindex(*cost.shape), key=lambda rc: cost[rc]):
        if row not in greedy and column not in taken:
            greedy[row] = column
            taken.add(column)
    assert greedy == {0: 1, 1: 0}, "greedy is expected to swap here; that is the point"

    assert assign(cost) == [(0, 0), (1, 1)]


def test_an_infinite_cost_is_refused_rather_than_silently_mishandled():
    with pytest.raises(ValueError, match="finite"):
        assign(np.array([[0.0, np.inf], [np.inf, 0.0]]))


# --- the four critical behaviours -----------------------------------------

def test_the_first_frame_gives_every_fighter_a_track():
    tracker = Tracker()
    tracks = tracker.update([fighter(0.3), fighter(0.7)], 0.0)
    assert len(tracks) == 2 and len(set(ids(tracks))) == 2
    assert all(t.age == 1 and t.missing == 0 for t in tracks)


def test_ids_are_stable_while_both_fighters_stay_in_frame():
    tracker = Tracker()
    tracker.update([fighter(0.30, size=0.18), fighter(0.70, size=0.24)], 0.0)
    left, right = id_at(tracker.active(), 0.30), id_at(tracker.active(), 0.70)
    for frame in range(1, 30):
        at = frame * STEP
        # Circling, not standing still: both drift and bob.
        a = 0.30 + 0.004 * frame
        b = 0.70 - 0.003 * frame
        tracks = tracker.update(
            [fighter(a, 0.5 + 0.01 * np.sin(frame), size=0.18),
             fighter(b, 0.5 - 0.01 * np.sin(frame), size=0.24)],
            at,
        )
        assert id_at(tracks, a) == left and id_at(tracks, b) == right
    assert tracker.active()[0].age == 30


def test_an_id_survives_an_occlusion_gap():
    """Critical. One fighter steps behind the other for five frames. He must
    come back as himself, not as a third man, or every count and every feature
    recorded against him restarts."""
    tracker = Tracker(max_missing=15)
    tracker.update([fighter(0.35, size=0.18), fighter(0.60, size=0.24)], 0.0)
    hidden = id_at(tracker.active(), 0.35)
    visible = id_at(tracker.active(), 0.60)

    for frame in range(1, 6):
        tracks = tracker.update([fighter(0.60, size=0.24)], frame * STEP)
        held = [t for t in tracks if t.id == hidden]
        assert held, "the occluded fighter must be held, not deleted"
        assert held[0].missing == frame
        assert next(t for t in tracks if t.id == visible).missing == 0

    tracks = tracker.update([fighter(0.37, size=0.18), fighter(0.60, size=0.24)], 6 * STEP)
    assert id_at(tracks, 0.37) == hidden, "he got a new id, so he is a new person now"
    assert id_at(tracks, 0.60) == visible
    assert len(tracks) == 2, "no phantom third fighter"


def test_two_fighters_crossing_over_keep_their_ids():
    """Critical. They lunge past each other, one small and one big, until left
    and right have swapped.

    The speed is the point. At 0.06 of a frame per step they cover as much
    ground between frames as the gap between them near the crossing, so around
    the middle each body is *nearer the other man's last position than its own*.
    Position alone therefore does not merely tie, it actively prefers the swap,
    and a tracker matching on where the torsos were swaps the ids and credits
    every punch after the crossing to the wrong fighter. Only the size term and
    the momentum in the prediction hold them apart. Disable both in `tracking`
    and this test fails at frame 7, which is how it was checked.
    """
    tracker = Tracker()
    small, big = 0.17, 0.25
    step, frames = 0.06, 11
    tracker.update([fighter(0.20, size=small), fighter(0.80, size=big)], 0.0)
    lean = id_at(tracker.active(), 0.20)
    heavy = id_at(tracker.active(), 0.80)
    assert lean != heavy

    for frame in range(1, frames):
        a = 0.20 + step * frame      # the small fighter, moving right
        b = 0.80 - step * frame      # the big one, moving left
        tracks = tracker.update([fighter(a, size=small), fighter(b, size=big)], frame * STEP)
        assert len(tracks) == 2, f"a fighter was lost or duplicated at frame {frame}"
        by_id = {track.id: track for track in tracks}
        assert by_id[lean].scale < by_id[heavy].scale, (
            f"the ids swapped at frame {frame}: the small fighter's track is on the big body"
        )

    assert id_at(tracker.active(), 0.80) == lean, "the small fighter ended up on the right"
    assert id_at(tracker.active(), 0.20) == heavy


def test_two_fighters_of_the_same_build_crossing_at_different_speeds_keep_their_ids():
    """The harder crossing: identical torso sizes, so the size term is worth
    nothing and momentum is the only thing left. One rushes, the other backs
    off slowly, which means the cheapest pairing on position alone is the swap
    rather than a tie that could resolve either way by luck. Matching against
    where each track was heading is what gets them through. Set
    `PREDICT_HORIZON` to 0 and this test fails."""
    tracker = Tracker()
    size, rush, retreat = 0.20, 0.07, 0.03
    tracker.update([fighter(0.18, size=size), fighter(0.74, size=size)], 0.0)
    charging = id_at(tracker.active(), 0.18)
    retreating = id_at(tracker.active(), 0.74)

    a, b = 0.18, 0.74
    for frame in range(1, 9):
        a, b = a + rush, b - retreat
        tracks = tracker.update([fighter(a, size=size), fighter(b, size=size)], frame * STEP)
        assert len(tracks) == 2, f"a fighter was lost or duplicated at frame {frame}"
        by_id = {track.id: track for track in tracks}
        assert by_id[charging].centre[0] == pytest.approx(a, abs=1e-6), (
            f"the ids swapped at frame {frame}"
        )
        assert by_id[retreating].centre[0] == pytest.approx(b, abs=1e-6)


def test_a_genuinely_new_person_gets_a_new_id():
    """Critical. A referee steps in. He must not inherit either fighter."""
    tracker = Tracker()
    tracker.update([fighter(0.25, size=0.18), fighter(0.45, size=0.24)], 0.0)
    known = set(ids(tracker.active()))

    tracks = tracker.update(
        [fighter(0.25, size=0.18), fighter(0.45, size=0.24), fighter(0.85, size=0.21)],
        STEP,
    )
    assert len(tracks) == 3
    fresh = [t for t in tracks if t.id not in known]
    assert len(fresh) == 1 and fresh[0].age == 1
    assert known <= set(ids(tracks)), "the two fighters kept their own ids"


def test_a_track_expires_after_max_missing():
    """Critical. Held is not forever. A fighter who left the frame must be gone
    before a stranger walks in, or the stranger inherits his identity."""
    tracker = Tracker(max_missing=3)
    tracker.update([fighter(0.4)], 0.0)
    gone = tracker.active()[0].id

    for frame in range(1, 4):
        tracks = tracker.update([], frame * STEP)
        assert ids(tracks) == [gone] and tracks[0].missing == frame

    assert tracker.update([], 4 * STEP) == []
    assert tracker.active() == []

    reappeared = tracker.update([fighter(0.4)], 5 * STEP)
    assert ids(reappeared) != [gone], "an expired id must not be handed out again"


# --- rejecting the impossible ---------------------------------------------

def test_a_body_that_teleports_across_the_frame_is_not_the_same_body():
    tracker = Tracker(max_distance=MATCH_LIMIT)
    tracker.update([fighter(0.2)], 0.0)
    first = tracker.active()[0].id

    tracks = tracker.update([fighter(0.85)], STEP)
    assert len(tracks) == 2, "the far body is a new person and the old track is held"
    far = min(tracks, key=lambda t: abs(t.centre[0] - 0.85))
    assert far.id != first and far.age == 1
    assert next(t for t in tracks if t.id == first).missing == 1


def test_a_move_just_inside_the_limit_is_still_the_same_body():
    tracker = Tracker(max_distance=0.10)
    tracker.update([fighter(0.40)], 0.0)
    first = tracker.active()[0].id
    tracks = tracker.update([fighter(0.49)], STEP)
    assert ids(tracks) == [first] and tracks[0].age == 2


def test_size_alone_can_tell_two_fighters_apart_in_a_clinch():
    """Clinched and pivoting: torso centres 0.05 apart, turning around each
    other fast enough that each fighter lands nearer where the *other* one was
    than where he was himself.

    Position is then not merely ambiguous, it is actively wrong: the cheapest
    pairing by distance is the swap, and momentum is no help because a pivot is
    not a straight line. The difference in build is the only honest information
    in the frame. Set `SIZE_WEIGHT` to 0 and this test fails at the first
    frame, which is how it was checked."""
    tracker = Tracker()
    lean_size, heavy_size = 0.16, 0.26
    radius, turn = 0.025, 2.4      # frame widths, radians per frame

    def spot(frame: int, side: int) -> tuple[float, float]:
        angle = turn * frame + side * np.pi
        return 0.5 + radius * np.cos(angle), 0.5 + radius * np.sin(angle)

    tracker.update(
        [fighter(*spot(0, 0), size=lean_size), fighter(*spot(0, 1), size=heavy_size)], 0.0
    )
    lean = min(tracker.active(), key=lambda t: t.scale).id
    heavy = max(tracker.active(), key=lambda t: t.scale).id

    for frame in range(1, 10):
        tracks = tracker.update(
            [fighter(*spot(frame, 0), size=lean_size),
             fighter(*spot(frame, 1), size=heavy_size)],
            frame * STEP,
        )
        assert len(tracks) == 2, f"a fighter was lost or duplicated at frame {frame}"
        by_id = {t.id: t for t in tracks}
        # A generous margin: a swap blends the two builds into one number, so
        # the gap between the tracked sizes collapses rather than merely
        # narrowing.
        assert by_id[lean].scale < by_id[heavy].scale - 0.02, f"builds swapped at frame {frame}"


def test_the_order_the_landmarker_returns_bodies_in_does_not_matter():
    """The pose landmarker's list order is not stable between frames, which is
    the whole reason this module exists. Shuffling it must change nothing."""
    tracker = Tracker()
    tracker.update([fighter(0.30, size=0.18), fighter(0.70, size=0.24)], 0.0)
    left, right = id_at(tracker.active(), 0.30), id_at(tracker.active(), 0.70)

    for frame in range(1, 10):
        bodies = [fighter(0.30, size=0.18), fighter(0.70, size=0.24)]
        if frame % 2:
            bodies.reverse()
        tracks = tracker.update(bodies, frame * STEP)
        assert id_at(tracks, 0.30) == left and id_at(tracks, 0.70) == right


# --- bookkeeping ----------------------------------------------------------

def test_age_counts_frames_observed_and_missing_counts_the_gap():
    tracker = Tracker(max_missing=5)
    tracker.update([fighter(0.4)], 0.0)
    tracker.update([fighter(0.41)], STEP)
    tracker.update([], 2 * STEP)
    tracker.update([], 3 * STEP)
    held = tracker.active()[0]
    assert held.age == 2 and held.missing == 2 and held.at == pytest.approx(STEP)

    tracker.update([fighter(0.42)], 4 * STEP)
    back = tracker.active()[0]
    assert back.age == 3 and back.missing == 0 and back.at == pytest.approx(4 * STEP)


def test_update_and_active_report_the_same_tracks():
    tracker = Tracker()
    returned = tracker.update([fighter(0.3), fighter(0.7)], 0.0)
    assert returned == tracker.active()
    assert ids(returned) == sorted(ids(returned)), "order must not depend on match order"


def test_an_empty_frame_creates_nothing():
    tracker = Tracker()
    assert tracker.update([], 0.0) == []
    assert tracker.active() == []


def test_a_track_carries_the_body_it_was_last_matched_to():
    tracker = Tracker()
    tracker.update([fighter(0.4)], 0.0)
    latest = fighter(0.42)
    track = tracker.update([latest], STEP)[0]
    assert track.body is latest
    assert isinstance(track, Track)


def test_nonsense_tracker_settings_are_refused():
    with pytest.raises(ValueError, match="max_distance"):
        Tracker(max_distance=0.0)
    with pytest.raises(ValueError, match="max_missing"):
        Tracker(max_missing=-1)
