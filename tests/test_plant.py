"""Planting a known guard habit into real footage, made of real movement.

These run on synthetic rounds so they need no fetched footage: a body
standing still in guard, with drops and punches put exactly where a test
wants them.
"""

import numpy as np
import pytest

from ika import plant
from ika.body import (LEFT_ELBOW, LEFT_HIP, LEFT_SHOULDER, LEFT_WRIST, NOSE, RIGHT_ELBOW,
                      RIGHT_HIP, RIGHT_SHOULDER, RIGHT_WRIST)
from ika.shadow import Clip, Consensus, Punch

FPS = 30.0


def standing(seconds=60.0, fps=FPS, scale=1.0, x0=0.5, name="r", mirrored=False):
    """A body facing the camera in guard. `scale` is the torso, so a smaller
    person or one further back. `mirrored` puts the left shoulder on the
    image's left, as for someone with their back turned."""
    t = np.arange(0, seconds, 1 / fps)
    image = np.full((len(t), 33, 3), 0.0)
    world = np.full((len(t), 33, 3), 0.0)
    sign = -1.0 if mirrored else 1.0
    pose = {
        NOSE: (0.0, -0.35),
        LEFT_SHOULDER: (0.12 * sign, 0.0), RIGHT_SHOULDER: (-0.12 * sign, 0.0),
        LEFT_HIP: (0.08 * sign, 0.3), RIGHT_HIP: (-0.08 * sign, 0.3),
        LEFT_ELBOW: (0.14 * sign, 0.12), RIGHT_ELBOW: (-0.14 * sign, 0.12),
        LEFT_WRIST: (0.07 * sign, -0.2), RIGHT_WRIST: (-0.07 * sign, -0.2),
    }
    for j, (x, y) in pose.items():
        image[:, j, 0] = x0 + x * scale
        image[:, j, 1] = 0.5 + y * scale
        world[:, j, 0] = x * 1.5
        world[:, j, 1] = y * 1.5
    for j, w in ((17, LEFT_WRIST), (19, LEFT_WRIST), (21, LEFT_WRIST),
                 (18, RIGHT_WRIST), (20, RIGHT_WRIST), (22, RIGHT_WRIST)):
        image[:, j] = image[:, w]
        world[:, j] = world[:, w]
    vis = np.ones((len(t), 33))
    return Clip(name, t, image.astype(np.float32), world.astype(np.float32),
                vis.astype(np.float32), np.zeros((len(t), 2, 21, 3), np.float32),
                -np.ones((len(t), 2), np.int8), 640, 360)


def with_dip(clip, side, start, end, depth=0.3, outward=0.0):
    """The wrist and elbow of `side` go down by `depth` torso lengths between
    `start` and `end`, smoothly, and come back."""
    image = clip.body_image.copy()
    world = clip.body_world.copy()
    ramp = 0.15
    t = clip.t
    w = np.clip(np.minimum(t - (start - ramp), (end + ramp) - t) / ramp, 0, 1)
    w = 0.5 - 0.5 * np.cos(np.pi * w)
    wrist = LEFT_WRIST if side == "left" else RIGHT_WRIST
    elbow = LEFT_ELBOW if side == "left" else RIGHT_ELBOW
    away = 1.0 if side == "left" else -1.0
    for j in (wrist, elbow, *((17, 19, 21) if side == "left" else (18, 20, 22))):
        # `standing` has a torso of 0.3 in raw image y and 0.45 in world
        image[:, j, 1] += (w * depth * 0.3).astype(np.float32)
        image[:, j, 0] += (w * outward * 0.3 * away).astype(np.float32)
        world[:, j, 1] += (w * depth * 0.45).astype(np.float32)
    from dataclasses import replace
    return replace(clip, body_image=image, body_world=world)


def test_cut_carries_a_real_dip():
    donor = with_dip(standing(name="donor"), "left", 10.0, 10.8, depth=0.3)
    drop = plant.cut(donor, "left", 10.0, 10.8)
    assert drop is not None
    assert drop.length == pytest.approx(0.8 + 2 * plant.PAD, abs=1 / FPS)
    assert drop.depth == pytest.approx(-0.3, abs=0.05)
    # detrended: the excursion starts and ends where the arm was
    assert abs(drop.image[0, 1, 1]) < 0.02 and abs(drop.image[-1, 1, 1]) < 0.02


def test_cut_refuses_what_is_not_a_sag():
    flat = standing(name="donor")
    assert plant.cut(flat, "left", 10.0, 10.8) is None                       # nothing moved
    rest = with_dip(flat, "left", 10.0, 16.0)
    assert plant.cut(rest, "left", 10.0, 16.0) is None                       # a rest, not a lapse
    dipped = with_dip(flat, "left", 10.0, 10.8)
    assert plant.cut(dipped, "left", 10.0, 10.8, [Punch(10.3, "left")]) is None   # arm punching
    assert plant.cut(dipped, "left", 10.0, 10.8, [Punch(10.3, "right")]) is not None


def test_taper_is_zero_at_the_ends_and_whole_in_the_middle():
    t = np.linspace(0, 1, 101)
    w = plant.taper(t, 1.0)
    assert w[0] == pytest.approx(0) and w[-1] == pytest.approx(0)
    assert np.all(w[(t > plant.RAMP) & (t < 1 - plant.RAMP)] == 1.0)
    assert np.all(np.diff(w[t <= 0.5]) >= 0)


def _one_drop(depth=0.3, side="left", outward=0.0):
    donor = with_dip(standing(name="donor"), side, 10.0, 10.8, depth=depth, outward=outward)
    return [plant.cut(donor, side, 10.0, 10.8)]


def test_apply_is_smooth_scaled_and_leaves_the_rest_alone():
    lib = _one_drop()
    for scale in (1.0, 0.5):
        clip = standing(scale=scale)
        p = plant.Plant(20.0, "left", 0, 19.5, lib[0].length)
        out = plant.apply(clip, [p], lib)
        dy = (out.body_image[:, LEFT_WRIST, 1] - clip.body_image[:, LEFT_WRIST, 1]) / clip.aspect
        torso = 0.3 * scale / clip.aspect
        outside = (clip.t < p.start) | (clip.t > p.start + p.length)
        assert np.all(dy[outside] == 0)
        # the deepest point is the donor's depth in this person's torso lengths
        assert dy.max() / torso == pytest.approx(0.3, abs=0.05)
        # no step: frame to frame it moves no more than the donor did, scaled
        donor_step = np.abs(np.diff(lib[0].image[:, 1, 1])).max() * torso
        assert np.abs(np.diff(dy)).max() <= donor_step * 1.5 + 1e-6
        # the other arm and the original clip are untouched
        assert np.array_equal(out.body_image[:, RIGHT_WRIST], clip.body_image[:, RIGHT_WRIST])
        assert not np.shares_memory(out.body_image, clip.body_image)
        # world coordinates go down too (MediaPipe world y grows downward)
        assert (out.body_world[:, LEFT_WRIST, 1] - clip.body_world[:, LEFT_WRIST, 1]).max() > 0.1


def test_sideways_movement_is_mirrored_by_anatomy():
    lib = _one_drop(outward=0.2)             # a left hand that drops and swings outward
    for mirrored in (False, True):
        clip = standing(mirrored=mirrored)
        out = plant.apply(clip, [plant.Plant(20.0, "right", 0, 19.5, lib[0].length)], lib)
        dx = out.body_image[:, RIGHT_WRIST, 0] - clip.body_image[:, RIGHT_WRIST, 0]
        away = np.sign(clip.body_image[0, RIGHT_SHOULDER, 0] - clip.body_image[0, LEFT_SHOULDER, 0])
        moved = dx[np.argmax(np.abs(dx))]
        assert np.sign(moved) == away        # outward from the other shoulder, either way round


def _bench(n_combos=40, spacing=4.0, second=None):
    """A round of combos ending in a right, with labelled truth. `second`
    adds, after each combo, a punch by that arm this many seconds later."""
    agreed, labelled = [], []
    for k in range(n_combos):
        at = 5.0 + k * spacing
        combo = [Punch(at, "left"), Punch(at + 0.35, "right")]
        agreed += combo
    labelled = list(agreed)
    if second is not None:
        side, gap = second
        extra = [Punch(p.at + gap, side) for p in agreed[1::2]]
        labelled += extra
    truth = Consensus(sorted(agreed, key=lambda p: p.at), [], [(0.0, 5.0 + n_combos * spacing)],
                      1.0, 1.0)
    return truth, labelled


def test_strength_is_honoured_exactly():
    clip = standing(seconds=180.0)
    lib = _one_drop() + [plant.cut(with_dip(standing(name="d2"), "right", 30, 30.5),
                                   "right", 30, 30.5)]
    truth, labelled = _bench()
    for strength in (0.0, 0.3, 0.5, 0.8, 1.0):
        plants, eligible, occurring = plant.plan(clip, truth, labelled, lib, "left",
                                                 ("punch_right",), strength, seed=3)
        assert occurring == 40 and eligible == 40
        assert len(plants) == round(strength * eligible)
        assert all(p.after in [q.at for q in truth.agreed if q.side == "right"] for p in plants)


def test_no_plant_while_that_arm_is_punching():
    clip = standing(seconds=180.0)
    lib = _one_drop()
    # The left arm throws again 1.2 to 1.6 s after each combo: plants must be
    # cut short to clear it, or skipped.
    for gap in (1.2, 1.4, 1.6):
        truth, labelled = _bench(second=("left", gap))
        plants, eligible, _ = plant.plan(clip, truth, labelled, lib, "left",
                                         ("punch_right",), 1.0, seed=1)
        for p in plants:
            for q in labelled:
                if q.side == "left":
                    assert not (p.start - plant.CLEAR < q.at < p.start + p.length + plant.CLEAR)
            assert p.length >= plant.MIN_PLANT
    # with room, the donor's whole drop is used
    truth, labelled = _bench()
    plants, _, _ = plant.plan(clip, truth, labelled, lib, "left", ("punch_right",), 1.0, seed=1)
    assert all(p.length == pytest.approx(lib[0].length) for p in plants)


def test_a_plant_lands_after_the_setup_inside_the_read_window():
    clip = standing(seconds=180.0)
    truth, labelled = _bench()
    plants, _, _ = plant.plan(clip, truth, labelled, _one_drop(), "left",
                              ("punch_right",), 1.0, seed=2)
    for p in plants:
        assert plant.DOWN_AFTER[0] - 1e-9 <= p.down() - p.after <= plant.DOWN_AFTER[1] + 1e-9


def test_planning_is_deterministic_with_a_seed():
    clip = standing(seconds=180.0)
    truth, labelled = _bench()
    lib = _one_drop()
    a = plant.plan(clip, truth, labelled, lib, "left", ("punch_right",), 0.5, seed=7)
    b = plant.plan(clip, truth, labelled, lib, "left", ("punch_right",), 0.5, seed=7)
    c = plant.plan(clip, truth, labelled, lib, "left", ("punch_right",), 0.5, seed=8)
    assert a == b and a[0] != c[0]
    assert np.array_equal(plant.apply(clip, a[0], lib).body_image,
                          plant.apply(clip, b[0], lib).body_image)


def test_setups_are_combo_endings_only():
    truth, _ = _bench(n_combos=5)
    assert len(plant.endings(truth, ("punch_right",))) == 5
    assert len(plant.endings(truth, ("punch_left", "punch_right"))) == 5
    assert plant.endings(truth, ("punch_left",)) == []          # the left is mid-combo
    assert len(plant.endings(truth, None)) == 5


def test_the_shuffled_control_carries_a_fixed_count_after_any_ending():
    clip = standing(seconds=180.0)
    truth, labelled = _bench()
    plants, eligible, occurring = plant.plan(clip, truth, labelled, _one_drop(), "left",
                                             None, 0.0, seed=4, count=12)
    assert len(plants) == 12 and occurring == 40


def test_a_drop_never_goes_back_into_its_own_round():
    clip = standing(seconds=180.0, name="donor")
    truth, labelled = _bench()
    plants, eligible, _ = plant.plan(clip, truth, labelled, _one_drop(), "left",
                                     ("punch_right",), 1.0, seed=0)
    assert plants == [] and eligible == 0


def test_no_plant_where_the_pose_was_lost():
    clip = standing(seconds=180.0)
    clip.body_image[:, :, :] = np.nan
    truth, labelled = _bench()
    plants, eligible, _ = plant.plan(clip, truth, labelled, _one_drop(), "left",
                                     ("punch_right",), 1.0, seed=0)
    assert eligible == 0
