"""The body: 33 landmarks instead of 21, and the same idea one level up.

The hand lane proved the shape of the pipeline. This is the part that decides
whether the `tell` work is a simulation or a product, because every figure in
it assumes actions can be recognised from real video, and that has never been
tested.

Deliberately the same structure as `hands.py`, down to the two coordinate
systems, so everything downstream keeps working. `image` landmarks carry
position in the frame; `world` landmarks are metric and hip-centred, which is
what makes a stance mean the same thing near or far from the camera.

MediaPipe ships three pose models. `lite` and `full` are both here because the
choice is a real trade at 30 fps on a laptop and `ika pose --benchmark`
measures it rather than guessing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

MODELS = Path(__file__).resolve().parent.parent / "models"

POSE_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker"
            "/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker"
            "/pose_landmarker_full/float16/1/pose_landmarker_full.task",
}

# The 33 pose landmarks, named so nothing downstream indexes a magic number.
NOSE = 0
LEFT_EYE_INNER, LEFT_EYE, LEFT_EYE_OUTER = 1, 2, 3
RIGHT_EYE_INNER, RIGHT_EYE, RIGHT_EYE_OUTER = 4, 5, 6
LEFT_EAR, RIGHT_EAR = 7, 8
MOUTH_LEFT, MOUTH_RIGHT = 9, 10
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_PINKY, RIGHT_PINKY = 17, 18
LEFT_INDEX, RIGHT_INDEX = 19, 20
LEFT_THUMB, RIGHT_THUMB = 21, 22
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
LEFT_HEEL, RIGHT_HEEL = 29, 30
LEFT_FOOT, RIGHT_FOOT = 31, 32

N_LANDMARKS = 33

# The torso, which is the rigid part of a body the way the knuckle row is the
# rigid part of a hand. The canonical frame is built from it for the same
# reason: shoulders and hips do not move relative to each other when a limb does.
TORSO = (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)

LIMBS = [
    (LEFT_SHOULDER, RIGHT_SHOULDER), (LEFT_SHOULDER, LEFT_HIP),
    (RIGHT_SHOULDER, RIGHT_HIP), (LEFT_HIP, RIGHT_HIP),
    (LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_HIP, LEFT_KNEE), (LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_HIP, RIGHT_KNEE), (RIGHT_KNEE, RIGHT_ANKLE),
    (LEFT_ANKLE, LEFT_HEEL), (LEFT_HEEL, LEFT_FOOT),
    (RIGHT_ANKLE, RIGHT_HEEL), (RIGHT_HEEL, RIGHT_FOOT),
    (NOSE, LEFT_SHOULDER), (NOSE, RIGHT_SHOULDER),
]

# What a fight is made of, in body terms. Which hand is which matters, so
# these are sided.
EXTREMITIES = (LEFT_WRIST, RIGHT_WRIST, LEFT_ANKLE, RIGHT_ANKLE)


@dataclass(frozen=True)
class Body:
    image: np.ndarray    # (33, 3) normalised to the frame
    world: np.ndarray    # (33, 3) metric, hip-centred
    visibility: np.ndarray   # (33,) how sure the model is of each point

    def seen(self, index: int, threshold: float = 0.5) -> bool:
        """Whether a landmark is actually visible rather than inferred.

        Pose landmarkers always return all 33 points, guessing at the ones
        they cannot see, so a limb behind a torso comes back as a confident
        looking number that is fiction. Anything reading a specific joint
        should check this first.
        """
        return bool(self.visibility[index] >= threshold)


class PoseTracker:
    """Stateful pose tracker. Same contract as `HandTracker`."""

    def __init__(
        self,
        variant: str = "lite",
        max_bodies: int = 1,
        detection_confidence: float = 0.4,
        tracking_confidence: float = 0.4,
        model_path: str | Path | None = None,
    ):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        from .hands import ModelMissing

        path = Path(model_path or MODELS / f"pose_landmarker_{variant}.task")
        if not path.exists():
            raise ModelMissing(path, POSE_URLS.get(variant, POSE_URLS["lite"]))

        self.variant = variant
        self._landmarker = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(path)),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=max_bodies,
                min_pose_detection_confidence=detection_confidence,
                min_tracking_confidence=tracking_confidence,
                output_segmentation_masks=False,
            )
        )

    def __call__(self, rgb: np.ndarray, timestamp_ms: int) -> list[Body]:
        import mediapipe as mp

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._landmarker.detect_for_video(image, int(timestamp_ms))

        out: list[Body] = []
        for i, marks in enumerate(result.pose_landmarks):
            world = result.pose_world_landmarks[i]
            out.append(
                Body(
                    image=np.array([[p.x, p.y, p.z] for p in marks], dtype=np.float32),
                    world=np.array([[p.x, p.y, p.z] for p in world], dtype=np.float32),
                    visibility=np.array([p.visibility for p in marks], dtype=np.float32),
                )
            )
        return out

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> PoseTracker:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
