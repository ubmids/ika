"""Fetch UT-Interaction and measure real action recognition from pose.

This is the script that replaced belief with evidence. Every figure in the
recognise-predict-warn lane had been measured on movement generated with
arithmetic, and this is the first time any of it met footage of people.

    .venv/bin/python scripts/ut_interaction.py --fetch
    .venv/bin/python scripts/ut_interaction.py

UT-Interaction was chosen over the alternatives for one reason: whole bodies.
HMDB51's official download is dead on every path including the Google Drive
links on its own page, UCF101 has no download smaller than 6.9 GB, Weizmann's
host resets during the TLS handshake, and Kinetics ships YouTube URLs with
substantial link rot. UT-Interaction is 30 MB, needs no account, and shows two
complete figures head to toe on a static camera.

Its licence is a citation request rather than a named licence, which is weaker
footing than HMDB51's CC BY 4.0, so that is worth knowing before anything built
on it goes anywhere.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

BASE = "https://cvrc.ece.utexas.edu/SDHA2010/videos/competition_1"
ARCHIVES = {
    "ut-interaction_segmented_set1.zip": 13_265_124,
    "ut-interaction_segmented_set2.zip": 16_423_456,
}
# The label is the last token of the filename.
LABELS = {"0": "hand_shake", "1": "hug", "2": "kick",
          "3": "point", "4": "punch", "5": "push"}
ROOT = Path("data/ut")


def fetch(into: Path) -> int:
    """Download and unpack, checking byte counts.

    The host truncated one archive silently on first fetch, returning 7.5 MB of
    a 13.2 MB file with no error, so the expected sizes above are verified
    rather than trusted.
    """
    into.mkdir(parents=True, exist_ok=True)
    for name, expected in ARCHIVES.items():
        target = into / name
        if target.exists() and target.stat().st_size == expected:
            print(f"  {name} already complete")
            continue
        print(f"  fetching {name} ({expected/1e6:.1f} MB)")
        subprocess.run(["curl", "-sL", "--retry", "3", "-C", "-",
                        "-o", str(target), f"{BASE}/{name}"], check=True)
        size = target.stat().st_size
        if size != expected:
            print(f"  TRUNCATED: got {size} bytes, expected {expected}. "
                  f"Delete it and retry.")
            return 1
        subprocess.run(["unzip", "-oq", str(target), "-d", str(into)], check=True)

    count = 0
    for avi in sorted(into.rglob("*.avi")):
        label = LABELS.get(avi.stem.split("_")[-1])
        if label is None:
            continue
        (ROOT / label).mkdir(parents=True, exist_ok=True)
        # Keep the sequence id, so clips from one recording stay identifiable.
        shutil.copy2(avi, ROOT / label / f"{avi.parent.name}_{avi.stem}.avi")
        count += 1
    print(f"  organised {count} clips into {ROOT}")
    return 0


def measure(per_class: int | None, width: int, stride: int) -> int:
    import numpy as np

    from ika.corpus import find_clips, read_clip, report
    from ika.watch import Watcher

    if not ROOT.is_dir():
        print(f"  no corpus at {ROOT}. Run with --fetch first.")
        return 1

    clips, classes = find_clips(ROOT, per_class=per_class)
    print(f"  {len(clips)} clips, {len(classes)} classes\n")

    with Watcher(max_bodies=1, stabilise=False) as watcher:
        for i, clip in enumerate(clips):
            read_clip(clip, watcher, max_width=width, stride=stride)
            if (i + 1) % 20 == 0:
                print(f"    {i + 1}/{len(clips)}", flush=True)

    print()
    print(report(clips))
    frames = sum(c.frames for c in clips)
    found = sum(c.with_body for c in clips)
    print(f"\n  a body was found in {found}/{frames} frames "
          f"({100 * found / max(1, frames):.0f}%)")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true", help="download and organise")
    parser.add_argument("--per-class", type=int, default=None, dest="per_class")
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--stride", type=int, default=2)
    args = parser.parse_args(argv)

    if args.fetch:
        return fetch(Path("data/ut_download"))
    return measure(args.per_class, args.width, args.stride)


if __name__ == "__main__":
    sys.exit(main())
