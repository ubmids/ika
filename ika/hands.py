"""The eyes: MediaPipe's hand landmarker, wrapped.

This is the one part deliberately not trained here. Getting 21 stable points
off a moving hand in changing light is a problem Google solved with more data
than a laptop will ever see, and reproducing it badly would sink the project
before the interesting parts begin. Everything downstream is ours.

Two coordinate systems come back and both are useful, for different jobs:

`image` landmarks are normalised to the frame, x and y in [0, 1]. Perspective
is baked in, which is exactly what you want for pointing at something: where
the finger appears is where the cursor should go.

`world` landmarks are metric and centred on the hand itself, so a hand held at
arm's length has the same measurements as one held close. That is the right
input for deciding *what shape* a hand is making, so it is what feeds
`features`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

MODEL = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"

HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker"
    "/hand_landmarker/float16/1/hand_landmarker.task"
)


class ModelMissing(FileNotFoundError):
    """Raised with the command to fix it.

    The weights are deliberately not in the repo, so a fresh clone has none.
    A bare FileNotFoundError would be a puzzle; this is a instruction.
    """

    def __init__(self, path: Path, url: str = HAND_MODEL_URL):
        self.path = path
        super().__init__(
            f"landmarker model missing at {path}\n\n"
            f"  mkdir -p {path.parent}\n"
            f"  curl -sL -o {path} \\\n    {url}\n"
        )


@dataclass(frozen=True)
class Hand:
    """One detected hand, in both coordinate systems."""

    image: np.ndarray   # (21, 3) normalised to the frame
    world: np.ndarray   # (21, 3) metric, hand-centred
    label: str          # "Left" or "Right", as seen by the camera
    score: float

    @property
    def is_left(self) -> bool:
        return self.label == "Left"

    def pixel(self, index: int, width: int, height: int) -> tuple[int, int]:
        """Where a landmark falls in the frame, in pixels."""
        return int(self.image[index, 0] * width), int(self.image[index, 1] * height)


class HandTracker:
    """Stateful tracker. Feed it frames in order and it tracks across them.

    Video mode rather than single-image mode on purpose: it reuses the previous
    frame's detection to seed the next, which is both faster and far steadier
    than detecting from scratch every frame. Steadiness matters more than
    accuracy here, because a cursor that jitters is unusable even if it is
    on average correct.
    """

    def __init__(
        self,
        max_hands: int = 2,
        detection_confidence: float = 0.5,
        tracking_confidence: float = 0.5,
        model_path: str | Path | None = None,
    ):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        path = Path(model_path or MODEL)
        if not path.exists():
            raise ModelMissing(path)

        self._vision = vision
        self._landmarker = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(path)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=max_hands,
                min_hand_detection_confidence=detection_confidence,
                min_tracking_confidence=tracking_confidence,
            )
        )

    def __call__(self, rgb: np.ndarray, timestamp_ms: int) -> list[Hand]:
        """Detect hands in one RGB uint8 frame. Timestamps must increase."""
        import mediapipe as mp

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._landmarker.detect_for_video(image, int(timestamp_ms))

        hands: list[Hand] = []
        for i, marks in enumerate(result.hand_landmarks):
            world = result.hand_world_landmarks[i]
            handedness = result.handedness[i][0]
            hands.append(
                Hand(
                    image=np.array([[p.x, p.y, p.z] for p in marks], dtype=np.float32),
                    world=np.array([[p.x, p.y, p.z] for p in world], dtype=np.float32),
                    label=handedness.category_name,
                    score=float(handedness.score),
                )
            )
        return hands

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> HandTracker:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
