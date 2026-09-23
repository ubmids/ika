"""Pick the most punch-dense stretch of each clip, from pose alone.

    python scripts/shadow_pick.py data/shadow/marks/*.npz

Wrist speed relative to the shoulders, in torso lengths per second, summed
over a sliding window, with frames where the person is not upright (planks,
burpees) scored as zero. Only chooses where to look; it labels nothing.
"""
import sys
from pathlib import Path
import numpy as np

def pick(path, seconds=45.0):
    m = np.load(path); t = m["t"]; p = m["body_image"][:, :, :2]
    sh = (p[:, 11] + p[:, 12]) / 2; hip = (p[:, 23] + p[:, 24]) / 2
    torso = np.linalg.norm(sh - hip, axis=1)
    upright = (hip[:, 1] - sh[:, 1]) > 0.6 * torso
    rel = np.concatenate([p[:, 15] - sh, p[:, 16] - sh], axis=1) / torso[:, None]
    speed = np.nan_to_num(np.linalg.norm(np.diff(rel, axis=0), axis=1) / np.diff(t))
    speed = np.minimum(speed, 20) * upright[1:]
    dt = np.median(np.diff(t)); n = int(seconds / dt)
    c = np.convolve(speed, np.ones(n), "valid")
    i = int(np.argmax(c))
    return round(float(t[i]) / 3) * 3, float(upright.mean())

for f in sys.argv[1:]:
    start, up = pick(f)
    print(Path(f).stem, start, f"upright {up:.0%}")
