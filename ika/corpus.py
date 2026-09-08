"""Turning a folder of labelled video clips into body-action training data.

This is the bridge between the simulation and reality. Every figure in the
recognise-predict-warn lane was measured on movement generated with
arithmetic, and this is what replaces it with movement that happened.

The layout expected is the one every public action dataset uses:

    root/
      punch/    clip_0001.avi  clip_0002.avi ...
      kick/     ...
      wave/     ...

Two things here are not obvious and both come from earlier mistakes.

**A clip that yields no body is reported, not silently dropped.** A dataset of
tightly cropped or low-resolution footage can fail pose estimation almost
entirely, and if that failure is quiet it looks exactly like a hard problem
rather than an unusable input. The counts come back so the difference is
visible before anything is trained on it.

**Clips are grouped by clip, never split by frame.** Frames from one clip are
near-duplicates of each other, so a random frame-level split puts the same
movement on both sides and reports an accuracy that is mostly memory. This is
the same trap as splitting HaGRID by image instead of by person, which
inflated that comparison until it was split by `user_id`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

VIDEO_SUFFIXES = (".avi", ".mp4", ".mov", ".mkv", ".webm", ".mpg", ".mpeg")

# Below this many frames with a body found, a clip is not worth keeping: the
# movement features need a window, and a handful of scattered detections
# describe nothing.
MIN_USABLE_FRAMES = 6


@dataclass
class Clip:
    """One labelled video, and what pose estimation actually got out of it."""

    path: Path
    label: str
    frames: int = 0            # frames decoded
    with_body: int = 0         # frames where a body was found
    windows: list = field(default_factory=list)   # bodymotion.Frame sequences

    @property
    def usable(self) -> bool:
        return self.with_body >= MIN_USABLE_FRAMES

    @property
    def detection_rate(self) -> float:
        return self.with_body / self.frames if self.frames else 0.0


def find_clips(root: str | Path, classes: list[str] | None = None,
               per_class: int | None = None) -> tuple[list[Clip], list[str]]:
    """Discover labelled clips from a directory of per-class folders."""
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"not a clip corpus: {root}")

    found = sorted(p.name for p in root.iterdir() if p.is_dir())
    classes = classes or found
    missing = [c for c in classes if c not in found]
    if missing:
        raise FileNotFoundError(f"no such classes under {root}: {missing}")

    clips: list[Clip] = []
    for label in classes:
        videos = sorted(
            p for p in (root / label).iterdir()
            if p.suffix.lower() in VIDEO_SUFFIXES
        )
        if per_class:
            videos = videos[:per_class]
        clips.extend(Clip(path=p, label=label) for p in videos)
    return clips, list(classes)


def read_clip(clip: Clip, watcher, max_width: int = 480, stride: int = 1,
              max_frames: int | None = None) -> Clip:
    """Run one clip through the pipeline, collecting per-frame body frames.

    Takes a `Watcher` rather than making one, because building a pose model
    per clip would dominate the runtime on a corpus of thousands.
    """
    from . import clip as clip_io
    from .bodymotion import Frame

    watcher.reset()
    sequence: list = []
    frames = 0
    with clip_io.Reader(clip.path, max_width=max_width, stride=stride,
                        max_frames=max_frames) as reader:
        for video_frame in reader:
            frames += 1
            observation = watcher.observe(video_frame.rgb, video_frame.at,
                                          video_frame.index)
            seen = observation.seen
            if not seen:
                continue
            # The largest body, because a crowd scene should be read as its
            # subject rather than as whoever happens to be track id 1.
            best = max(seen, key=lambda s: s.track.scale)
            sequence.append(Frame(
                at=video_frame.at,
                features=best.features,
                wrists=np.stack([
                    best.track.body.image[15, :2], best.track.body.image[16, :2]
                ]),
                scale=max(float(best.track.scale), 1e-4),
                visibility=best.track.body.visibility,
            ))
    clip.frames = frames
    clip.with_body = len(sequence)
    clip.windows = [sequence] if len(sequence) >= MIN_USABLE_FRAMES else []
    return clip


def split_by_clip(clips: list[Clip], fraction: float = 0.25, seed: int = 0):
    """Stratified split at clip level, never at frame level.

    Frames within a clip are near-duplicates, so a frame-level split reports
    an accuracy that is mostly memory of the same movement seen from one
    frame earlier.
    """
    rng = np.random.default_rng(seed)
    train, val = [], []
    by_label: dict[str, list[int]] = {}
    for i, clip in enumerate(clips):
        by_label.setdefault(clip.label, []).append(i)
    for indices in by_label.values():
        order = np.array(indices)
        rng.shuffle(order)
        cut = max(1, int(round(len(order) * fraction))) if len(order) > 1 else 0
        val.extend(order[:cut].tolist())
        train.extend(order[cut:].tolist())
    return np.array(train, dtype=int), np.array(val, dtype=int)


def report(clips: list[Clip]) -> str:
    """What pose estimation managed, per class.

    Printed before any training, because a corpus where bodies are rarely
    found is an unusable input rather than a hard problem, and the two look
    identical once a number comes out the far end.
    """
    from collections import defaultdict

    per_class: dict[str, list[Clip]] = defaultdict(list)
    for clip in clips:
        per_class[clip.label].append(clip)

    lines = [f"  {'class':<20} {'clips':>6} {'usable':>7} {'body found':>11}"]
    for label in sorted(per_class):
        group = per_class[label]
        usable = sum(c.usable for c in group)
        rate = np.mean([c.detection_rate for c in group]) if group else 0.0
        flag = "   <- too few bodies to use" if usable < len(group) * 0.5 else ""
        lines.append(f"  {label:<20} {len(group):>6} {usable:>7} {rate:>10.0%}{flag}")
    total = len(clips)
    usable = sum(c.usable for c in clips)
    lines.append(f"  {'total':<20} {total:>6} {usable:>7}")
    return "\n".join(lines)
