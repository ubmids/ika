"""Turning a fingertip into a cursor.

Two problems, both of which make the difference between usable and infuriating.

**Reach.** Mapping the frame straight onto the screen means the screen corners
sit at the frame corners, and your hand leaves the camera's view before it gets
there. So an inner rectangle of the frame maps to the whole screen, and the
margin outside it is slack you never have to enter.

**Jitter.** Landmark estimates wobble by a pixel or two every frame. On a
2 mm target that wobble is the whole game, so positions are smoothed. Smoothing
costs lag, and lag on a cursor feels worse than noise does, which is why this
uses a speed-dependent filter: heavy smoothing while the hand is still, light
smoothing while it moves. You get a steady cursor at rest and a responsive one
in flight, instead of having to choose.
"""

from __future__ import annotations

import numpy as np


def map_to_screen(
    nx: float,
    ny: float,
    screen_w: int,
    screen_h: int,
    margin: float = 0.18,
    mirrored: bool = True,
) -> tuple[int, int]:
    """Frame-normalised coordinates to screen pixels.

    `margin` is the fraction of the frame ignored on each edge. `mirrored`
    should stay true for a front camera: the image is a mirror, so moving your
    hand right must move the cursor right.
    """
    span = max(1e-6, 1.0 - 2.0 * margin)
    x = (nx - margin) / span
    y = (ny - margin) / span
    if mirrored:
        x = 1.0 - x
    x = float(np.clip(x, 0.0, 1.0))
    y = float(np.clip(y, 0.0, 1.0))
    return int(round(x * (screen_w - 1))), int(round(y * (screen_h - 1)))


class CursorSmoother:
    """Speed-dependent smoothing: still hands get steady, moving hands get quick.

    A plain exponential average with one constant forces a choice between a
    twitchy cursor and a laggy one. Here the constant is derived from how fast
    the point is travelling, so both behaviours come from one filter.
    """

    def __init__(self, slow: float = 0.82, fast: float = 0.25, knee: float = 40.0):
        self.slow = slow      # weight on history when barely moving
        self.fast = fast      # weight on history when moving quickly
        self.knee = knee      # pixels per frame at which we consider it "fast"
        self._point: np.ndarray | None = None

    def reset(self) -> None:
        self._point = None

    def __call__(self, x: float, y: float) -> tuple[int, int]:
        target = np.array([x, y], dtype=float)
        if self._point is None:
            self._point = target
            return int(round(x)), int(round(y))

        speed = float(np.linalg.norm(target - self._point))
        # Blend between the two constants over the knee, so there is no
        # discontinuity where the cursor visibly changes character.
        blend = min(1.0, speed / self.knee)
        weight = self.slow * (1.0 - blend) + self.fast * blend
        self._point = weight * self._point + (1.0 - weight) * target
        return int(round(self._point[0])), int(round(self._point[1]))


def screen_size() -> tuple[int, int]:
    """The main display's size in points, or a sane guess."""
    try:
        from AppKit import NSScreen

        frame = NSScreen.mainScreen().frame()
        return int(frame.size.width), int(frame.size.height)
    except Exception:
        return 1440, 900
