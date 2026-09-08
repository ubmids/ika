"""Reading a video file frame by frame, with timestamps you can trust.

Everything downstream of this module is time-sensitive. `motion` differentiates
positions to get velocity, `sequence` resamples windows to a fixed length, and
`dynamic` decides a gesture fired by how long it was held. All of that reads a
timestamp, so a timeline that is quietly wrong produces a pipeline that is
quietly wrong, which is the worst kind of broken because every number still
looks reasonable.

Two habits from the live camera lane do not survive contact with files.

**`index / fps` is not a timestamp.** It is only correct if the file has one
constant frame rate and no dropped frames, which is true of what a `VideoWriter`
produces and often false of what a phone produces. Phone video is variable rate:
the encoder drops the rate in low light to hold exposure, so a clip labelled
30 fps contains stretches at 24 and `index / fps` drifts a little further behind
with every frame. The drift is silent. Nothing errors, the clip just gets
slower than reality. So the container's own per-frame timestamp is used where
there is one, `index / fps` is the fallback rather than the default, and which
of the two produced a given `at` is recorded on the frame itself.

**A failed read is not the end of the file.** `VideoCapture.read` returns the
same `False` for "the clip ended" and "that packet was damaged", so the obvious
loop, `while True: ok, frame = cap.read()`, silently truncates a clip at its
first bad frame. On a deliberately damaged 30 frame test file that loop returns
10 frames and no error. This module returns 29 of the 30, with the timestamps
of everything after the damage still correct, because it separates advancing
through the stream (`grab`) from decoding pixels (`retrieve`) and seeks past a
position that will not advance. See `test_a_corrupt_frame_does_not_truncate_the_clip`
for both numbers.

One OpenCV quirk shapes the recovery code and is worth knowing about before
changing it. After `set(CAP_PROP_POS_FRAMES, n)` this build reports
`CAP_PROP_POS_FRAMES` as the position it was at *before* the seek, even though
the seek worked and the next `grab` returns frame n. So progress past a damaged
position is tracked in a local counter here. A guard that waited for the
capture's own position to move would abandon every recoverable file, which it
did until this was found.

What this module does not fix: a source with no real timestamps at all. Hand
OpenCV a folder of PNGs as `frames_%03d.png` and it will report 25 fps and
timestamps 40 ms apart with total confidence, because the image sequence backend
invents both. Nothing here can detect that, since there is no second opinion to
compare against. Only real containers get checked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

MS_PER_SECOND = 1000.0

# Which clock produced a frame's `at`, recorded per frame because the answer
# can change part way through a clip.
FROM_CONTAINER = "container"     # CAP_PROP_POS_MSEC, the file's own timestamp
FROM_INDEX = "index/fps"         # counted position over a frame rate

# A frame rate outside this band is metadata, not fact. Real footage runs from
# roughly 1 fps timelapse to 1000 fps high speed; anything else is a container
# reporting 0, a NaN, or a nonsense number, and using it would put every
# fallback timestamp somewhere absurd.
MIN_PLAUSIBLE_FPS = 0.01
MAX_PLAUSIBLE_FPS = 1000.0

# Used only when a file gives no usable rate at all and a frame still needs an
# `at`. It is a guess, and `Timeline.source` says so, so a caller that cares
# can refuse the clip rather than trust the number.
ASSUMED_FPS = 30.0

# How far the rate implied by real timestamps may sit from the rate the
# container advertises before the advertised one is called wrong. Deliberately
# loose: 29.97 stored as 30 is a rounding convention, not a lie.
FPS_DISAGREEMENT = 0.05

# A clip whose longest frame gap is more than this multiple of its shortest one
# is variable rate. 1.5x is well beyond the sub-millisecond wobble of container
# timebase rounding and well below the 1.25x-plus swings of real phone footage.
VARIABLE_RATE_SPREAD = 1.5

# Timing decisions need a sample, not the whole clip. 512 gaps is 17 seconds of
# 30 fps footage, enough to characterise a rate, and bounded so a feature length
# file does not accumulate a list of every frame gap in it.
TIMING_SAMPLE = 512

# Damaged packets cluster, so recovery has to be able to step over a run of
# them, but it must also give up rather than seek forever over a file that ends
# in garbage. 30 consecutive dead positions is a full second of 30 fps video.
MAX_CONSECUTIVE_SKIPS = 30

# When the container will not say how many frames it holds, a failed advance
# cannot be told from the end of the file by counting. Probe a few positions
# past it and stop if none of them decodes.
MAX_BLIND_PROBES = 3

# Timestamps are compared with a tolerance rather than for equality because
# container timebases are rational and rounding leaves sub-millisecond dust.
TIME_EPSILON = 1e-6

CHANNELS_RGB = 3


class ClipError(OSError):
    """Base for anything that stops a clip being read.

    One base so a caller can guard an entire ingest with a single `except`,
    while the two subclasses stay distinguishable because the fixes differ:
    a missing file is the caller's path bug, an undecodable one is a codec or
    a damaged download.
    """


class ClipMissing(ClipError, FileNotFoundError):
    """Nothing at that path.

    Also inherits FileNotFoundError so existing code that already guards file
    access keeps catching it.
    """

    def __init__(self, path: Path):
        self.path = path
        super().__init__(f"no video file at {path}")


class ClipUndecodable(ClipError):
    """The file is there and OpenCV cannot get frames out of it.

    Carries the likely causes, because the bare failure OpenCV gives back is
    a boolean and a message on stderr that a caller never sees.
    """

    def __init__(self, path: Path, detail: str):
        self.path = path
        super().__init__(
            f"cannot decode {path}: {detail}\n"
            "the file may be truncated, may be a container OpenCV was built "
            "without, or may not be video at all. `ffprobe` on it will say which."
        )


@dataclass(frozen=True)
class Frame:
    """One decoded frame and the moment it belongs to.

    `rgb` is RGB, not the BGR that OpenCV hands out, because every consumer
    here is a MediaPipe landmarker and those take RGB. Converting once at the
    source beats each caller remembering to, and a caller who forgets gets a
    result that is subtly wrong rather than an error.
    """

    index: int          # position within this read, counting from the start offset
    at: float           # seconds from the beginning of the clip
    rgb: np.ndarray     # (H, W, 3) uint8
    source: str = FROM_CONTAINER   # which clock produced `at`

    @property
    def is_timed_by_container(self) -> bool:
        """Whether `at` came from the file rather than from arithmetic."""
        return self.source == FROM_CONTAINER


@dataclass
class Timeline:
    """What the reader learned about the clip's clock while reading it.

    Live, not final: it is filled in as frames come out, so read it after
    iterating. It exists so a caller can tell a trustworthy timeline from a
    reconstructed one instead of assuming, which matters because a fallback
    timeline on variable rate footage is the exact failure this module is
    built to make visible.
    """

    reported_fps: float = 0.0        # what the container claims
    fps: float = ASSUMED_FPS         # what is actually used for fallback timing
    source: str = FROM_CONTAINER     # the clock most frames were timed by
    fps_was_wrong: bool = False      # reported rate disagrees with real timestamps
    variable_rate: bool = False      # frame gaps are not constant
    advanced: int = 0                # stream positions successfully stepped over
    yielded: int = 0                 # frames decoded and handed to the caller
    skipped: int = 0                 # positions lost to damage
    gaps: list[float] = field(default_factory=list)   # observed frame gaps, seconds

    @property
    def measured_fps(self) -> float:
        """Frame rate implied by the timestamps, or 0.0 if none were usable."""
        if not self.gaps:
            return 0.0
        median = float(np.median(self.gaps))
        return 1.0 / median if median > 0 else 0.0

    def describe(self) -> str:
        """One line for a log, so a bad clip is visible without a debugger."""
        note = []
        if self.fps_was_wrong:
            note.append(f"reported {self.reported_fps:.3f} fps but measured "
                        f"{self.measured_fps:.3f}")
        if self.variable_rate:
            note.append("variable frame rate")
        if self.skipped:
            note.append(f"{self.skipped} damaged position(s) skipped")
        return (f"{self.yielded} frames, timed by {self.source}"
                + (", " + ", ".join(note) if note else ""))


def _plausible_fps(value: float) -> bool:
    """Whether a reported frame rate can be used for arithmetic at all."""
    return bool(np.isfinite(value)) and MIN_PLAUSIBLE_FPS <= value <= MAX_PLAUSIBLE_FPS


def _is_pattern(path: str) -> bool:
    """Whether this is an image sequence pattern rather than a real path.

    OpenCV accepts `frames_%03d.png` as a video source. Such a string names no
    existing file, so the existence check has to be skipped for it or every
    sequence read would fail as missing.
    """
    return "%" in path


def _open(path: str | Path) -> tuple[cv2.VideoCapture, Path]:
    """Open a capture or raise something a person can act on.

    The existence check is separate from the open because OpenCV returns the
    same closed capture for a missing file, a directory, and an unsupported
    codec, and those need three different responses from whoever called this.
    """
    text = str(path)
    resolved = Path(text)
    if not _is_pattern(text):
        if not resolved.exists():
            raise ClipMissing(resolved)
        if resolved.is_dir():
            raise ClipUndecodable(resolved, "that is a directory, not a file")

    capture = cv2.VideoCapture(text)
    if not capture.isOpened():
        capture.release()
        raise ClipUndecodable(resolved, "OpenCV could not open a video stream")
    return capture, resolved


def probe(path: str | Path, verify: bool = False) -> dict:
    """Read a clip's shape without decoding its pixels.

    Metadata is used where it is plausible and measured where it is not, and
    the result says which, because "30 fps" read off a container and "30 fps"
    counted off timestamps are different facts and only one of them survives a
    variable rate file.

    `verify` walks the whole stream to count frames that actually decode. Off
    by default because it costs a full pass, on when the count has to be right:
    containers routinely overstate it. A damaged file used in the tests here
    advertises 30 frames and yields 29.

    Returns fps, frames, duration, width and height, plus the `*_source`
    fields saying where each came from. Sources are "container" for metadata,
    "measured" for counted, and "assumed" for the last-resort guess.
    """
    capture, resolved = _open(path)
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        reported_fps = float(capture.get(cv2.CAP_PROP_FPS))
        reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        # Zero dimensions mean the stream opened but carries no video, which is
        # what a container holding only audio, or a header without a payload,
        # looks like from here.
        if width <= 0 or height <= 0:
            raise ClipUndecodable(resolved, "stream reports no frame size")

        fps, fps_source = reported_fps, "container"
        frames, frames_source = reported_frames, "container"
        if not _plausible_fps(fps):
            fps, fps_source = 0.0, "measured"
        if frames <= 0:
            frames, frames_source = 0, "measured"

        if verify or fps_source == "measured" or frames_source == "measured":
            counted, stamps = _walk(capture, total=reported_frames)
            if counted == 0:
                raise ClipUndecodable(resolved, "no frame could be decoded")
            if verify or frames_source == "measured":
                frames, frames_source = counted, "measured"
            if fps_source == "measured":
                fps = _fps_from(stamps)
                if fps <= 0:
                    fps, fps_source = ASSUMED_FPS, "assumed"

        duration = frames / fps if fps > 0 else 0.0
        return {
            "fps": float(fps),
            "frames": int(frames),
            "duration": float(duration),
            "width": width,
            "height": height,
            "fps_source": fps_source,
            "frames_source": frames_source,
            "reported_fps": reported_fps,
            "reported_frames": reported_frames,
        }
    finally:
        capture.release()


def _walk(capture: cv2.VideoCapture, total: int = 0) -> tuple[int, list[float]]:
    """Count positions and collect timestamps without decoding pixels.

    `grab` advances the stream, `retrieve` is what converts and copies a frame,
    so counting with `grab` alone avoids paying for pixels nobody will look at.
    Recovery is the same seek-past trick the reader uses, so the count here
    matches what the reader will actually deliver. Progress is tracked in a
    local rather than read back from CAP_PROP_POS_FRAMES: after a seek this
    build of OpenCV reports the position it was at before the seek, so trusting
    it would either stall the walk or loop it.
    """
    seconds: list[float] = []
    counted = 0
    consecutive = 0
    blind_probes = 0
    target = 0      # our own monotone position, because POS_FRAMES is not
    while True:
        position = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
        if not capture.grab():
            target = max(position, target) + 1
            if total > 0 and target >= total:
                break
            if consecutive >= MAX_CONSECUTIVE_SKIPS:
                break
            if total <= 0:
                # Without a frame count there is no way to tell damage from the
                # end of the file, so probe a little and then accept the end.
                if blind_probes >= MAX_BLIND_PROBES:
                    break
                blind_probes += 1
            consecutive += 1
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, target):
                break
            continue
        consecutive = 0
        blind_probes = 0
        target = max(position, target)
        counted += 1
        stamp = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / MS_PER_SECOND
        if np.isfinite(stamp) and stamp >= 0:
            seconds.append(stamp)
    return counted, seconds


def _fps_from(seconds: list[float]) -> float:
    """Frame rate implied by a list of timestamps, or 0.0 if it cannot be.

    The median gap rather than the mean, because one damaged position leaves
    one double-length gap and a mean would smear that across the whole clip.
    """
    if len(seconds) < 2:
        return 0.0
    gaps = [b - a for a, b in zip(seconds, seconds[1:]) if b > a]
    if not gaps:
        return 0.0
    median = float(np.median(gaps))
    rate = 1.0 / median if median > 0 else 0.0
    return rate if _plausible_fps(rate) else 0.0


class Reader:
    """A clip, read once, forwards, one frame at a time.

    Exists alongside `read` so the timing verdict is reachable: `read` is a
    generator and a generator has nowhere to hang a `Timeline` off. Use `read`
    for the common case, this when you need to know afterwards whether the
    timestamps were real.
    """

    def __init__(
        self,
        path: str | Path,
        max_width: int | None = None,
        stride: int = 1,
        max_frames: int | None = None,
        start: float = 0.0,
    ):
        if stride < 1:
            raise ValueError(f"stride must be at least 1, got {stride}")
        if max_frames is not None and max_frames < 0:
            raise ValueError(f"max_frames cannot be negative, got {max_frames}")
        if max_width is not None and max_width < 1:
            raise ValueError(f"max_width must be at least 1, got {max_width}")
        if start < 0:
            raise ValueError(f"start cannot be negative, got {start}")

        self.max_width = max_width
        self.stride = stride
        self.max_frames = max_frames
        self.start = start

        self._capture, self.path = _open(path)
        self.timeline = Timeline()

        reported = float(self._capture.get(cv2.CAP_PROP_FPS))
        self.timeline.reported_fps = reported
        self.timeline.fps = reported if _plausible_fps(reported) else ASSUMED_FPS
        # An implausible rate is already known to be wrong before a single
        # frame is read, and saying so here means a caller that only checks the
        # flag still learns about a file reporting 0 fps.
        if not _plausible_fps(reported):
            self.timeline.fps_was_wrong = True

        self._total = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self._origin = self._seek(start)
        self._closed = False

    def _seek(self, start: float) -> float:
        """Move to `start` seconds and report where we actually landed.

        Seeking is by time rather than by frame number because frame numbers
        assume a constant rate, which is the assumption this module exists to
        avoid. The landing point is read back rather than assumed: a seek goes
        to a keyframe, so asking for 2.0 s can leave you at 1.87 s, and using
        the requested time as the origin would offset every timestamp after it.

        A landing time of zero after seeking forwards is the container declining
        to answer, not a seek that went back to the beginning, so the requested
        time is kept instead. Believing the zero put the whole second half of a
        clip at the timestamps of its first half.
        """
        if start <= 0:
            return 0.0
        self._capture.set(cv2.CAP_PROP_POS_MSEC, start * MS_PER_SECOND)
        landed = float(self._capture.get(cv2.CAP_PROP_POS_MSEC)) / MS_PER_SECOND
        return landed if np.isfinite(landed) and landed > TIME_EPSILON else start

    def _note_gap(self, previous: float, current: float) -> None:
        """Record a frame gap and update the variable rate verdict.

        Gaps are what tell a constant rate clip from a variable one, and the
        verdict has to be available from the first few frames rather than at
        the end, so a caller that stops early still gets an answer.
        """
        gap = current - previous
        if gap <= 0:
            return
        if len(self.timeline.gaps) < TIMING_SAMPLE:
            self.timeline.gaps.append(gap)
        gaps = self.timeline.gaps
        if len(gaps) >= 2:
            shortest, longest = min(gaps), max(gaps)
            self.timeline.variable_rate = (
                shortest > 0 and longest / shortest > VARIABLE_RATE_SPREAD
            )
            measured = self.timeline.measured_fps
            reported = self.timeline.reported_fps
            if measured > 0 and _plausible_fps(reported):
                drift = abs(measured - reported) / reported
                if drift > FPS_DISAGREEMENT:
                    # Trust the timestamps over the header. The header is one
                    # number written once; the timestamps are evidence.
                    self.timeline.fps_was_wrong = True
                    self.timeline.fps = measured

    def _shrink(self, bgr: np.ndarray) -> np.ndarray:
        """Downscale to `max_width`, keeping the aspect ratio.

        INTER_AREA because it averages the pixels it discards, so a hand at
        small scale keeps its shape instead of aliasing into a different one.
        Never upscales: enlarging cannot add detail a landmarker could use, and
        INTER_AREA is not even an enlargement filter, so a clip narrower than
        `max_width` is passed through untouched.
        """
        height, width = bgr.shape[:2]
        if self.max_width is None or width <= self.max_width:
            return bgr
        scale = self.max_width / width
        # At least one row, or a very wide letterboxed frame scaled hard would
        # round to zero height and cv2.resize would raise.
        target = (self.max_width, max(1, round(height * scale)))
        return cv2.resize(bgr, target, interpolation=cv2.INTER_AREA)

    def __iter__(self) -> Iterator[Frame]:
        """Yield frames in order. Iterating twice yields nothing the second time."""
        capture = self._capture
        position = 0            # positions advanced since the read began
        previous: float | None = None
        consecutive = 0
        blind_probes = 0
        target = 0              # our own monotone position, because POS_FRAMES is not
        recovered = False       # whether a seek-past has happened yet
        first_decode_failed_at: int | None = None

        while True:
            if self.max_frames is not None and self.timeline.yielded >= self.max_frames:
                return

            source_position = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
            if not capture.grab():
                # Indistinguishable from the end of the file, so decide with
                # the frame count if there is one and by probing if there is not.
                #
                # The seek target is kept here rather than read back from the
                # capture: after `set` this build of OpenCV still reports the
                # position it was at beforehand, so a guard that waited for
                # POS_FRAMES to move would abandon every recoverable file.
                target = max(source_position, target) + 1
                if self._total > 0 and target >= self._total:
                    break
                if consecutive >= MAX_CONSECUTIVE_SKIPS:
                    break
                if self._total <= 0:
                    if blind_probes >= MAX_BLIND_PROBES:
                        break
                    blind_probes += 1
                if not capture.set(cv2.CAP_PROP_POS_FRAMES, target):
                    break
                consecutive += 1
                recovered = True
                self.timeline.skipped += 1
                position += 1   # the timeline keeps the lost slot
                continue
            target = max(source_position, target)

            stamp = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / MS_PER_SECOND
            # A timestamp is usable if it is a real number, not negative, and
            # not behind the last one. Zero is the exception and needs its own
            # rule, because a backend with no timestamps at all answers 0.0 for
            # every frame and that reads as a perfectly monotonic timeline while
            # carrying no information. Zero is only believable for the first
            # frame of a read that began at the start of the clip; anywhere else
            # it is the container declining to answer.
            at_clip_start = previous is None and self._origin <= TIME_EPSILON
            zero_stamp = stamp <= TIME_EPSILON
            usable = (
                bool(np.isfinite(stamp))
                and stamp >= 0
                and (not zero_stamp or at_clip_start)
                and (previous is None or stamp > previous - TIME_EPSILON)
            )

            if recovered and usable and previous is not None and stamp <= previous:
                # A seek can land before where we already were, and handing back
                # a frame that goes backwards in time would corrupt any velocity
                # computed from it. Drop it rather than emit it.
                position += 1
                continue

            keep = position % self.stride == 0
            if keep:
                ok, bgr = capture.retrieve()
                if not ok:
                    # The position advanced but the pixels did not arrive. The
                    # slot is still spent, so the timeline moves on and the
                    # clip does not end here.
                    if first_decode_failed_at is None:
                        first_decode_failed_at = position
                    self.timeline.skipped += 1
                    position += 1
                    blind_probes = 0
                    continue

            consecutive = 0
            blind_probes = 0
            self.timeline.advanced += 1

            if usable:
                at, timed_by = stamp, FROM_CONTAINER
                if previous is not None:
                    self._note_gap(previous, stamp)
                previous = stamp
            else:
                # No usable timestamp, so count from the seek origin at the
                # best rate known. `position` counts every slot including the
                # damaged ones, which is what keeps this from compressing the
                # timeline around a drop.
                at = self._origin + position / max(self.timeline.fps, MIN_PLAUSIBLE_FPS)
                timed_by = FROM_INDEX
                self.timeline.source = FROM_INDEX

            if keep:
                rgb = cv2.cvtColor(self._shrink(bgr), cv2.COLOR_BGR2RGB)
                self.timeline.yielded += 1
                yield Frame(index=position, at=at, rgb=rgb, source=timed_by)

            position += 1

        if first_decode_failed_at is not None and self.timeline.yielded == 0:
            raise ClipUndecodable(self.path, "every frame failed to decode")

    def close(self) -> None:
        """Release the capture. Safe to call twice."""
        if not self._closed:
            self._capture.release()
            self._closed = True

    def __enter__(self) -> Reader:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def __del__(self) -> None:
        # A generator abandoned part way through leaves the capture open, and a
        # long ingest that does that per file runs out of descriptors.
        try:
            self.close()
        except Exception:
            pass


def read(
    path: str | Path,
    max_width: int | None = None,
    stride: int = 1,
    max_frames: int | None = None,
    start: float = 0.0,
) -> Iterator[Frame]:
    """Iterate a clip's frames, one decoded at a time.

    A generator, so an hour of 4K costs one frame of memory rather than an
    hour of it. That is not a nicety: the whole point of this lane is running
    a pose model over long footage on a laptop, and a list of frames would
    make the shape of the file decide whether the job runs at all.

    `stride` samples every nth position and leaves `at` alone, so every third
    frame of 30 fps footage reports gaps of about a third of a second rather
    than pretending to be 10 fps footage. `max_frames` caps what is handed back.
    `start` is seconds into the clip; `at` stays absolute across it, while
    `index` counts from the start point.

    `max_width` downscales with INTER_AREA and never enlarges.

    Raises ClipMissing or ClipUndecodable before yielding anything.
    """
    reader = Reader(
        path, max_width=max_width, stride=stride,
        max_frames=max_frames, start=start,
    )
    try:
        yield from reader
    finally:
        reader.close()
