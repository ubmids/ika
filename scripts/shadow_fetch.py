"""Fetch the shadowboxing bench and cache its landmarks.

    python scripts/shadow_fetch.py

Downloads each three-minute section listed in bench/shadow/clips.json at the
resolution the labels were made on (360p), then runs the pose landmarker over
it once, so every later measurement replays cached landmarks and not video.
Needs `yt-dlp` on the path. Nothing it writes is committed: the footage is
other people's, only the labels in bench/shadow are ours.

YouTube started asking the default client to prove it was not a bot after a
handful of downloads; the `mweb` client did not, which is why it is named.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from ika.shadow import BENCH, ROOT  # noqa: E402


def main() -> int:
    if shutil.which("yt-dlp") is None:
        print("  needs yt-dlp: brew install yt-dlp, or pip install yt-dlp")
        return 1
    clips = json.loads((BENCH / "clips.json").read_text())["clips"]
    raw = ROOT / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for clip in clips:
        target = raw / f"{clip['id']}.mp4"
        if target.exists():
            continue
        start, end = clip["section"]
        print(f"  fetching {clip['id']} ({clip['channel']}) {start}-{end}s", flush=True)
        subprocess.run([
            "yt-dlp", "-q", "--no-warnings",
            "--extractor-args", "youtube:player_client=mweb",
            "-f", "b[height<=720]/bv*[height<=720]+ba", "--merge-output-format", "mp4",
            "--download-sections", f"*{start}-{end}",
            "-o", str(raw / f"{clip['id']}.%(ext)s"),
            f"https://www.youtube.com/watch?v={clip['id']}",
        ], check=False)
    missing = [c["id"] for c in clips if not (ROOT / "marks" / f"{c['id']}.npz").exists()
               and (raw / f"{c['id']}.mp4").exists()]
    if missing:
        subprocess.run([sys.executable, str(HERE / "scripts" / "shadow_extract.py"),
                        *[str(raw / f"{m}.mp4") for m in missing]], check=False)
    have = sum((ROOT / "marks" / f"{c['id']}.npz").exists() for c in clips)
    print(f"  {have} of {len(clips)} clips ready. Next: python scripts/train_strike.py --quick")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
