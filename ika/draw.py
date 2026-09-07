"""Drawing hands and text on frames. Presentation only, no logic."""

from __future__ import annotations

import cv2
import numpy as np

from .schema import CONNECTIONS, TIPS


def hand(frame: np.ndarray, marks: np.ndarray, colour=(90, 220, 140), thickness: int = 2) -> None:
    """Skeleton over a BGR frame, given (21, 3) frame-normalised landmarks."""
    h, w = frame.shape[:2]
    points = [(int(x * w), int(y * h)) for x, y, _ in marks]
    for a, b in CONNECTIONS:
        cv2.line(frame, points[a], points[b], colour, thickness, cv2.LINE_AA)
    for i, point in enumerate(points):
        radius = 5 if i in TIPS else 3
        cv2.circle(frame, point, radius, colour, -1, cv2.LINE_AA)


def text(frame: np.ndarray, lines: list[str], origin: tuple[int, int] = (10, 22),
         scale: float = 0.5) -> None:
    """Dark stroke under light fill, so it stays readable over any scene."""
    x, y = origin
    for i, line in enumerate(lines):
        at = (x, y + i * int(20 * scale / 0.5))
        cv2.putText(frame, line, at, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, at, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def bar(frame: np.ndarray, value: float, at: tuple[int, int], width: int = 120,
        height: int = 8, colour=(90, 220, 140)) -> None:
    """A 0-to-1 meter, for confidence and dwell progress."""
    x, y = at
    cv2.rectangle(frame, (x, y), (x + width, y + height), (60, 60, 60), -1)
    filled = int(width * float(np.clip(value, 0.0, 1.0)))
    if filled:
        cv2.rectangle(frame, (x, y), (x + filled, y + height), colour, -1)
