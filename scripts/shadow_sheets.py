"""Contact sheets for labelling punches from the picture, independent of any detector.

Each sheet is three seconds of one clip at 10 frames per second, 30 tiles,
every tile cropped to the fighter and stamped with its time and index. The crop
comes from the pose box over the whole window, not per tile, so the fighter
stays put across tiles and a punch reads as an arm moving rather than a camera.

    python scripts/shadow_sheets.py 9Yzke26tOXU --start 20 --seconds 60

Labels are made by looking at these, which is the point: the detector being
scored never touches them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent / "data" / "shadow"
RATE = 10.0
SHEET_SECONDS = 3.0
COLS, ROWS = 6, 5
TILE_H = 220


def person_box(marks, t0: float, t1: float, w: int, h: int):
    t = marks["t"]
    pick = (t >= t0) & (t < t1)
    pts = marks["body_image"][pick][:, :25, :2]      # head to hips and arms
    pts = pts[np.isfinite(pts[:, 0, 0])]
    if not len(pts):
        return 0, 0, w, h
    xs, ys = pts[..., 0] * w, pts[..., 1] * h
    x0, x1 = np.percentile(xs, 1), np.percentile(xs, 99)
    y0, y1 = np.percentile(ys, 1), np.percentile(ys, 99)
    pad = 0.25 * max(x1 - x0, y1 - y0)
    x0, x1 = max(0, x0 - pad), min(w, x1 + pad)
    y0, y1 = max(0, y0 - pad), min(h, y1 + pad)
    return int(x0), int(y0), int(x1), int(y1)


def build(clip: str, start: float, seconds: float, out: Path) -> list[Path]:
    video = ROOT / "raw" / f"{clip}.mp4"
    marks = np.load(ROOT / "marks" / f"{clip}.npz")
    capture = cv2.VideoCapture(str(video))
    w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out.mkdir(parents=True, exist_ok=True)
    made = []
    t = start
    while t < start + seconds - 1e-6:
        box = person_box(marks, t, t + SHEET_SECONDS, w, h)
        tiles = []
        for i in range(int(SHEET_SECONDS * RATE)):
            at = t + i / RATE
            capture.set(cv2.CAP_PROP_POS_MSEC, at * 1000.0)
            ok, bgr = capture.read()
            if not ok:
                bgr = np.zeros((h, w, 3), np.uint8)
            crop = bgr[box[1]:box[3], box[0]:box[2]]
            scale = TILE_H / max(crop.shape[0], 1)
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            cv2.rectangle(crop, (0, 0), (104, 20), (0, 0, 0), -1)
            cv2.putText(crop, f"{i:02d} {at:6.1f}s", (3, 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(crop)
        tile_w = max(tile.shape[1] for tile in tiles)
        tiles = [cv2.copyMakeBorder(x, 0, 0, 0, tile_w - x.shape[1], cv2.BORDER_CONSTANT)
                 for x in tiles]
        rows = [np.hstack(tiles[r * COLS:(r + 1) * COLS]) for r in range(ROWS)]
        sheet = np.vstack(rows)
        path = out / f"{clip}_{t:06.1f}.jpg"
        cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        made.append(path)
        t += SHEET_SECONDS
    capture.release()
    return made


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("clip")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--out", type=Path, default=ROOT / "sheets")
    args = parser.parse_args()
    for path in build(args.clip, args.start, args.seconds, args.out):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
