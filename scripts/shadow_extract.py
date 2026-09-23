"""Run the landmarkers over real shadowboxing clips and cache what they saw.

Raw landmarks only, never features, for the reason `dataset.py` records:
features are a guess about what the detector needs, and that guess changes.

    python scripts/shadow_extract.py data/shadow/raw/*.mp4

Writes data/shadow/marks/<clip>.npz with, per frame: the timestamp, the body
(image, world, visibility) or NaN, and up to two hands (image, handedness).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ika.body import PoseTracker  # noqa: E402
from ika.hands import HandTracker  # noqa: E402


def extract(path: Path, out: Path, variant: str) -> dict:
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    times, body_img, body_world, body_vis = [], [], [], []
    hand_img, hand_left = [], []
    blank33 = np.full((33, 3), np.nan, np.float32)
    blank21 = np.full((2, 21, 3), np.nan, np.float32)

    with PoseTracker(variant=variant, max_bodies=1) as pose, \
         HandTracker(max_hands=2, detection_confidence=0.3) as hands:
        index = 0
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            # Container clock, not frame count over nominal fps: `clip.py`
            # records a variable-rate file that broke the counted clock.
            at = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if index and at <= times[-1]:
                at = times[-1] + 1.0 / fps
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            stamp = int(at * 1000)
            bodies = pose(rgb, stamp)
            seen = hands(rgb, stamp)

            times.append(at)
            if bodies:
                b = bodies[0]
                body_img.append(b.image)
                body_world.append(b.world)
                body_vis.append(b.visibility)
            else:
                body_img.append(blank33)
                body_world.append(blank33)
                body_vis.append(np.zeros(33, np.float32))

            h_img = blank21.copy()
            h_left = np.full(2, -1, np.int8)
            for slot, hand in enumerate(seen[:2]):
                h_img[slot] = hand.image
                h_left[slot] = int(bool(hand.is_left))
            hand_img.append(h_img)
            hand_left.append(h_left)
            index += 1

    capture.release()
    np.savez_compressed(
        out,
        t=np.asarray(times, np.float64),
        body_image=np.asarray(body_img),
        body_world=np.asarray(body_world),
        body_vis=np.asarray(body_vis),
        hand_image=np.asarray(hand_img),
        hand_left=np.asarray(hand_left),
        width=np.int32(frame_size(path)[0]),
        height=np.int32(frame_size(path)[1]),
    )
    bodies_found = float(np.isfinite(np.asarray(body_img)[:, 0, 0]).mean())
    hands_found = float((np.asarray(hand_left) >= 0).any(axis=1).mean())
    return {"frames": len(times), "fps": fps, "body": bodies_found, "hand": hands_found}


def frame_size(path: Path) -> tuple[int, int]:
    capture = cv2.VideoCapture(str(path))
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    capture.release()
    return size


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("clips", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/shadow/marks"))
    parser.add_argument("--variant", default="full")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for clip in args.clips:
        stats = extract(clip, args.out / f"{clip.stem}.npz", args.variant)
        print(f"  {clip.stem:14s} {stats['frames']:5d} frames {stats['fps']:5.1f} fps  "
              f"body {stats['body']:4.0%}  hand {stats['hand']:4.0%}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
