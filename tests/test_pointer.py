"""Cursor mapping and smoothing."""

import numpy as np

from ika.pointer import CursorSmoother, map_to_screen


def test_the_centre_of_the_frame_is_the_centre_of_the_screen():
    x, y = map_to_screen(0.5, 0.5, 1000, 800)
    assert abs(x - 499) <= 1 and abs(y - 399) <= 1


def test_the_inner_rectangle_reaches_the_screen_edges():
    """The point of the margin: you must be able to reach a corner without
    your hand leaving the camera's view."""
    assert map_to_screen(0.18, 0.18, 1000, 800, margin=0.18) == (999, 0)
    assert map_to_screen(0.82, 0.82, 1000, 800, margin=0.18) == (0, 799)


def test_going_past_the_margin_clamps_rather_than_overshooting():
    for nx, ny in [(-0.5, 0.5), (1.5, 0.5), (0.5, -0.2), (0.5, 1.4)]:
        x, y = map_to_screen(nx, ny, 1000, 800)
        assert 0 <= x < 1000 and 0 <= y < 800


def test_mirroring_makes_right_mean_right():
    """A front camera shows a mirror image, so without this the cursor goes
    the wrong way and the whole thing feels broken."""
    left, _ = map_to_screen(0.3, 0.5, 1000, 800, mirrored=True)
    right, _ = map_to_screen(0.7, 0.5, 1000, 800, mirrored=True)
    assert right < left
    left, _ = map_to_screen(0.3, 0.5, 1000, 800, mirrored=False)
    right, _ = map_to_screen(0.7, 0.5, 1000, 800, mirrored=False)
    assert right > left


def test_the_first_point_is_not_smoothed():
    """Otherwise the cursor flies in from wherever it was last time."""
    assert CursorSmoother()(300.0, 400.0) == (300, 400)


def test_a_still_hand_produces_a_still_cursor():
    """Jitter rejection. Landmarks wobble a pixel or two every frame and the
    cursor must not."""
    smoother = CursorSmoother()
    rng = np.random.default_rng(0)
    smoother(500.0, 500.0)
    positions = [smoother(500 + rng.normal(0, 1.5), 500 + rng.normal(0, 1.5)) for _ in range(60)]
    spread = np.std(np.array(positions[10:]), axis=0)
    assert spread.max() < 1.0, f"cursor wandered by {spread}"


def test_a_fast_move_is_not_left_behind():
    """The other half: heavy smoothing at speed would feel like lag."""
    smoother = CursorSmoother()
    smoother(100.0, 100.0)
    for _ in range(8):
        x, y = smoother(900.0, 100.0)
    assert x > 800, f"only reached {x} of 900 after 8 frames"


def test_smoothing_is_gentler_when_moving_than_when_still():
    """The property the speed-dependent design exists to provide."""
    still, moving = CursorSmoother(), CursorSmoother()
    still(500.0, 500.0)
    moving(500.0, 500.0)
    slow_step = still(502.0, 500.0)[0] - 500
    for _ in range(3):
        moving(700.0, 500.0)
    fast = CursorSmoother()
    fast(500.0, 500.0)
    fast(700.0, 500.0)
    fast_fraction = (fast(700.0, 500.0)[0] - 500) / 200.0
    slow_fraction = slow_step / 2.0
    assert fast_fraction > slow_fraction


def test_reset_forgets_the_previous_position():
    smoother = CursorSmoother()
    smoother(100.0, 100.0)
    smoother.reset()
    assert smoother(800.0, 600.0) == (800, 600)
