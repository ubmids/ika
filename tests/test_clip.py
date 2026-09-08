"""What `clip` promises about time, damage and memory, measured on real files.

Every video here is written by the test that needs it, so this suite needs no
camera, no network, no weights and no dataset. That also means the files are
well behaved by construction, which is the catch: a `VideoWriter` will not
produce a variable frame rate clip or a container that lies about its rate, and
those are exactly the cases the module exists for. So the timing tests come in
two halves. Real files pin the behaviour that can be produced honestly, and a
stand-in capture pins the behaviour that cannot, by answering the same handful
of OpenCV calls with the timestamps a phone would have given.

The claims with numbers attached are the ones worth reading:

`test_a_corrupt_frame_does_not_truncate_the_clip` shows the naive read loop
returning 10 of 30 frames and this module returning 29.

`test_index_over_fps_drifts_on_variable_rate_footage` shows how far the
arithmetic timeline lands from the truth over 300 frames of phone-like video.

`test_a_long_clip_never_loads_into_memory_at_once` shows the resident memory
of streaming a clip against the memory of holding it.
"""

import inspect
import resource
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from ika import clip
from ika.clip import ClipError, ClipMissing, ClipUndecodable, Frame, Reader, probe, read

# Small enough that a test writes one in milliseconds, big enough that
# downscaling to a third of the width is still a legal frame size.
TEST_WIDTH, TEST_HEIGHT = 96, 72
TEST_FPS = 10.0
TEST_FRAMES = 20

MJPG = "MJPG"      # every frame independent, so one can be damaged in isolation
MP4V = "mp4v"      # a real mp4 timebase, which is where POS_MSEC earns its keep

BYTES_PER_MB = 1024 * 1024
# macOS reports the resident high-water mark in bytes, Linux in kilobytes.
RSS_SCALE = 1 if sys.platform == "darwin" else 1024

# One frame period of slack. Container timebases are rational, so a timestamp
# lands on a multiple of 1/timebase rather than on a round number of seconds.
def _tolerance(fps: float) -> float:
    return 1.0 / fps


def _paint(index: int, width: int = TEST_WIDTH, height: int = TEST_HEIGHT) -> np.ndarray:
    """A BGR frame whose blue channel encodes its position.

    Encoding the index in the pixels is what lets a test tell *which* frames
    came back after a damaged one, rather than only how many.
    """
    frame = np.zeros((height, width, 3), np.uint8)
    frame[:, :, 0] = index * 5 % 256      # blue
    frame[:, :, 1] = 40                   # green
    frame[:, :, 2] = 200                  # red
    return frame


def _write(
    path: Path,
    frames: int = TEST_FRAMES,
    fps: float = TEST_FPS,
    fourcc: str = MJPG,
    width: int = TEST_WIDTH,
    height: int = TEST_HEIGHT,
    painter=_paint,
) -> Path:
    """Write a clip and fail loudly if this OpenCV build cannot."""
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*fourcc), fps, (width, height)
    )
    assert writer.isOpened(), f"this OpenCV cannot write {fourcc} to {path.suffix}"
    for index in range(frames):
        writer.write(painter(index, width, height))
    writer.release()
    assert path.stat().st_size > 0
    return path


def _damage(source: Path, target: Path, which: int = 10) -> Path:
    """Wipe the payload of one JPEG frame inside an MJPG AVI.

    Damaging the middle of an mp4 is no use for this: it destroys the moov atom
    and the file will not open at all, which is a different failure. MJPG keeps
    each frame as a standalone JPEG, so wiping one leaves a container that opens
    cleanly, reports its full frame count, and dies on exactly one packet. That
    is the shape of a real damaged download.
    """
    raw = bytearray(source.read_bytes())
    starts = [i for i in range(len(raw) - 1) if raw[i] == 0xFF and raw[i + 1] == 0xD8]
    assert len(starts) > which + 1, f"only found {len(starts)} JPEG frames"
    for i in range(starts[which] + 2, starts[which + 1] - 2):
        raw[i] = 0x00
    target.write_bytes(bytes(raw))
    return target


def _naive_frame_count(path: Path) -> int:
    """The loop everybody writes, so the test can quote what it costs."""
    capture = cv2.VideoCapture(str(path))
    count = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        count += 1
    capture.release()
    return count


def _rss_mb() -> float:
    """Resident high-water mark, which only ever rises, so it is safe to diff."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * RSS_SCALE / BYTES_PER_MB


class FakeCapture:
    """A capture that answers with timestamps a VideoWriter cannot produce.

    Variable frame rate footage, a container reporting zero fps, and a packet
    that advances but will not decode are the three failures this module is
    built around, and none of them can be written to disk with the tools a test
    is allowed to use here. This stands in for the codec so those paths get
    exercised rather than described.

    `stamps_ms` is one entry per stream position, `None` meaning the container
    has no timestamp for it. A position listed in `undecodable` grabs but fails
    to retrieve; one in `ungrabbable` will not advance at all.
    """

    def __init__(self, stamps_ms, reported_fps=TEST_FPS, reported_frames=None,
                 undecodable=(), ungrabbable=()):
        self.stamps_ms = list(stamps_ms)
        self.reported_fps = reported_fps
        self.reported_frames = (
            len(self.stamps_ms) if reported_frames is None else reported_frames
        )
        self.undecodable = set(undecodable)
        self.ungrabbable = set(ungrabbable)
        self.position = 0
        self._grabbed = None
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return self.reported_fps
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return self.reported_frames
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return TEST_WIDTH
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return TEST_HEIGHT
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return self.position
        if prop == cv2.CAP_PROP_POS_MSEC:
            # Before the first grab this is the frame about to be read, which is
            # how a reader learns where a seek actually landed.
            at = self._grabbed if self._grabbed is not None else self.position
            if at >= len(self.stamps_ms) or self.stamps_ms[at] is None:
                return 0.0
            return self.stamps_ms[at]
        return 0.0

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self.position = int(value)
            return True
        if prop == cv2.CAP_PROP_POS_MSEC:
            # Seek by time the way a codec does: land on the first position at
            # or after the request, or count it out of the rate when the
            # positions carry no timestamps of their own.
            known = [i for i, ms in enumerate(self.stamps_ms)
                     if ms is not None and ms >= value]
            if known:
                self.position = known[0]
            elif self.reported_fps > 0:
                self.position = int(round(value / clip.MS_PER_SECOND
                                          * self.reported_fps))
            else:
                self.position = 0
            return True
        return False

    def grab(self):
        if self.position >= len(self.stamps_ms):
            return False
        if self.position in self.ungrabbable:
            self.position += 1   # the codec consumed the packet and gave nothing
            return False
        self._grabbed = self.position
        self.position += 1
        return True

    def retrieve(self):
        if self._grabbed in self.undecodable:
            return False, None
        return True, _paint(self._grabbed)

    def release(self):
        self.released = True


def _inject(monkeypatch, tmp_path, capture) -> Path:
    """Point `clip` at a stand-in capture, keeping a real file on disk.

    A real file so the existence check under test is the real one, and only the
    codec is replaced. Patching the attribute `clip` looks up means the rest of
    OpenCV, the colour conversion and the resize, stays genuine.
    """
    path = tmp_path / "stand_in.avi"
    path.write_bytes(b"placeholder")
    monkeypatch.setattr(clip.cv2, "VideoCapture", lambda *_a, **_k: capture)
    return path


# --- probe ----------------------------------------------------------------

def test_probe_reports_the_shape_of_a_clip(tmp_path):
    path = _write(tmp_path / "a.avi")
    info = probe(path)
    assert info["width"] == TEST_WIDTH
    assert info["height"] == TEST_HEIGHT
    assert info["frames"] == TEST_FRAMES
    assert info["fps"] == pytest.approx(TEST_FPS, abs=0.01)
    assert info["duration"] == pytest.approx(TEST_FRAMES / TEST_FPS, abs=0.01)


def test_probe_says_where_every_number_came_from(tmp_path):
    """A rate read off a header and a rate counted off timestamps are different
    facts, and a caller deciding whether to trust a clip needs to know which
    one it was handed."""
    info = probe(_write(tmp_path / "a.avi"))
    assert info["fps_source"] == "container"
    assert info["frames_source"] == "container"
    assert info["reported_fps"] == pytest.approx(TEST_FPS, abs=0.01)


def test_probe_measures_the_frame_count_when_the_container_overstates_it(tmp_path):
    """The damaged file advertises every frame it was written with and cannot
    deliver one of them, which is what makes the advertised count unusable for
    anything that needs to be right."""
    pristine = _write(tmp_path / "a.avi", frames=30)
    damaged = _damage(pristine, tmp_path / "bad.avi")

    claimed = probe(damaged)
    verified = probe(damaged, verify=True)

    assert claimed["frames"] == 30, "the container still claims all 30"
    assert verified["frames"] == 29, "one position cannot be decoded"
    assert verified["frames_source"] == "measured"
    assert verified["reported_frames"] == 30


def test_probe_falls_back_to_measuring_when_the_container_reports_no_rate(
    monkeypatch, tmp_path
):
    """A rate of zero is the common way a container declines to answer, and
    dividing an index by it would put every timestamp at infinity."""
    stamps = [i * 50.0 for i in range(12)]      # a real 20 fps
    capture = FakeCapture(stamps, reported_fps=0.0, reported_frames=0)
    path = _inject(monkeypatch, tmp_path, capture)

    info = probe(path)
    assert info["fps_source"] == "measured"
    assert info["frames_source"] == "measured"
    assert info["frames"] == 12
    assert info["fps"] == pytest.approx(20.0, abs=0.1)
    assert info["reported_fps"] == 0.0


def test_probe_assumes_a_rate_only_when_there_is_nothing_to_measure(
    monkeypatch, tmp_path
):
    """One frame with no timestamp gives no gap and so no rate. The number that
    comes back is a guess and the source field is the only thing stopping a
    caller treating it as a fact."""
    capture = FakeCapture([None], reported_fps=0.0, reported_frames=0)
    info = probe(_inject(monkeypatch, tmp_path, capture))
    assert info["fps_source"] == "assumed"
    assert info["fps"] == clip.ASSUMED_FPS


def test_probe_and_read_agree_about_how_many_frames_there_are(tmp_path):
    path = _write(tmp_path / "a.avi")
    assert probe(path)["frames"] == sum(1 for _ in read(path))


# --- decoding -------------------------------------------------------------

def test_reading_a_clip_yields_every_frame_once_and_in_order(tmp_path):
    path = _write(tmp_path / "a.avi")
    frames = list(read(path))
    assert len(frames) == TEST_FRAMES
    assert [f.index for f in frames] == list(range(TEST_FRAMES))
    assert all(b.at > a.at for a, b in zip(frames, frames[1:]))


def test_frames_come_back_as_rgb_not_the_bgr_opencv_hands_out(tmp_path):
    """A landmarker fed BGR does not error, it just reads a differently
    coloured world and quietly does worse, so the channel order has to be
    pinned rather than assumed."""
    red_bgr = np.zeros((TEST_HEIGHT, TEST_WIDTH, 3), np.uint8)
    red_bgr[:, :, 2] = 255      # pure red in OpenCV's order
    path = _write(tmp_path / "red.avi", frames=4, painter=lambda *_a: red_bgr)

    frame = next(iter(read(path)))
    mean = frame.rgb.reshape(-1, clip.CHANNELS_RGB).mean(axis=0)
    assert mean[0] > 200, f"red should be in channel 0, got {mean}"
    assert mean[2] < 60


def test_frames_are_the_shape_and_dtype_the_landmarkers_expect(tmp_path):
    path = _write(tmp_path / "a.avi", frames=3)
    frame = next(iter(read(path)))
    assert frame.rgb.shape == (TEST_HEIGHT, TEST_WIDTH, clip.CHANNELS_RGB)
    assert frame.rgb.dtype == np.uint8
    assert frame.rgb.flags["C_CONTIGUOUS"], "MediaPipe needs a contiguous buffer"


def test_a_frame_is_immutable_so_a_consumer_cannot_rewrite_its_timestamp(tmp_path):
    frame = next(iter(read(_write(tmp_path / "a.avi", frames=3))))
    with pytest.raises(Exception):
        frame.at = 99.0


# --- timestamps -----------------------------------------------------------

def test_timestamps_come_from_the_container_not_from_arithmetic(tmp_path):
    path = _write(tmp_path / "a.mp4", fourcc=MP4V)
    frames = list(read(path))
    assert all(f.source == clip.FROM_CONTAINER for f in frames)
    assert all(f.is_timed_by_container for f in frames)


def test_a_constant_rate_clip_agrees_with_index_over_fps(tmp_path):
    """The easy case, checked so that a disagreement anywhere else means
    something rather than being noise in the reader."""
    path = _write(tmp_path / "a.mp4", fourcc=MP4V)
    for frame in read(path):
        assert frame.at == pytest.approx(frame.index / TEST_FPS, abs=1e-3)


@pytest.mark.parametrize("fps", [29.97, 23.976, 60.0])
def test_an_awkward_frame_rate_survives_the_round_trip(tmp_path, fps):
    """23.976 and 29.97 are the rates real footage actually uses, and both are
    stored as ratios rather than decimals, so they are where a reader that
    rounds shows up."""
    path = _write(tmp_path / f"{fps}.mp4", frames=15, fps=fps, fourcc=MP4V)
    frames = list(read(path))
    gaps = [b.at - a.at for a, b in zip(frames, frames[1:])]
    assert np.median(gaps) == pytest.approx(1.0 / fps, rel=1e-3)


def test_a_container_reporting_zero_fps_is_detected_and_reported(monkeypatch, tmp_path):
    """Zero is not a rate. If the reader had to fall back it would be dividing
    by it, so the flag is raised before the first frame is read."""
    capture = FakeCapture([i * 100.0 for i in range(8)], reported_fps=0.0)
    reader = Reader(_inject(monkeypatch, tmp_path, capture))
    assert reader.timeline.fps_was_wrong
    assert reader.timeline.fps == clip.ASSUMED_FPS

    frames = list(reader)
    assert len(frames) == 8
    # The rate was useless but the timestamps were not, so `at` is still real.
    assert all(f.source == clip.FROM_CONTAINER for f in frames)
    assert frames[-1].at == pytest.approx(0.7)


def test_a_lying_frame_rate_is_caught_by_the_timestamps(monkeypatch, tmp_path):
    """The header is one number written once. The timestamps are evidence, so
    when they disagree the header loses."""
    real_fps = 10.0
    capture = FakeCapture(
        [i * (clip.MS_PER_SECOND / real_fps) for i in range(40)],
        reported_fps=30.0,
    )
    reader = Reader(_inject(monkeypatch, tmp_path, capture))
    list(reader)

    assert reader.timeline.reported_fps == 30.0
    assert reader.timeline.fps_was_wrong
    assert reader.timeline.measured_fps == pytest.approx(real_fps, rel=0.01)
    assert reader.timeline.fps == pytest.approx(real_fps, rel=0.01)


def test_a_rounding_difference_is_not_called_a_lie(monkeypatch, tmp_path):
    """29.97 footage whose header says 30 is a convention, not a fault, and
    flagging it would train callers to ignore the flag."""
    capture = FakeCapture(
        [i * (clip.MS_PER_SECOND / 29.97) for i in range(40)], reported_fps=30.0
    )
    reader = Reader(_inject(monkeypatch, tmp_path, capture))
    list(reader)
    assert not reader.timeline.fps_was_wrong


def test_variable_frame_rate_footage_is_flagged(monkeypatch, tmp_path):
    """Nothing downstream can compensate for a rate that moves, but plenty of
    it can behave differently if told, so this has to be visible."""
    gap_ms = np.tile([33.3, 41.7, 33.3, 50.0], 10)      # 30, 24, 30, 20 fps
    stamps = np.concatenate([[0.0], np.cumsum(gap_ms)])
    reader = Reader(_inject(monkeypatch, tmp_path, FakeCapture(stamps, 30.0)))
    list(reader)
    assert reader.timeline.variable_rate


def test_a_constant_rate_clip_is_not_flagged_as_variable(tmp_path):
    """The other half of the same claim, on a real file, because a detector
    that fires on everything is not a detector."""
    reader = Reader(_write(tmp_path / "a.mp4", frames=40, fourcc=MP4V))
    list(reader)
    assert not reader.timeline.variable_rate
    assert not reader.timeline.fps_was_wrong


def test_index_over_fps_drifts_on_variable_rate_footage(monkeypatch, tmp_path):
    """The measurement behind the whole module.

    300 frames of footage labelled 30 fps that actually alternates between 30
    and 20, the way a phone does when it drops the rate to hold exposure. The
    arithmetic timeline ends more than two seconds short of the truth, and
    nothing about it looks wrong: the numbers are smooth, evenly spaced, and
    completely detached from when things happened. A gesture held for half a
    second in the middle of that clip is measured somewhere else entirely.
    """
    labelled_fps = 30.0
    gap_ms = np.tile([1000 / 30, 1000 / 20], 150)
    stamps = np.concatenate([[0.0], np.cumsum(gap_ms)])
    reader = Reader(_inject(monkeypatch, tmp_path, FakeCapture(stamps, labelled_fps)))
    frames = list(reader)

    truth = stamps[-1] / clip.MS_PER_SECOND
    arithmetic = (len(frames) - 1) / labelled_fps
    drift = truth - arithmetic

    assert frames[-1].at == pytest.approx(truth, abs=1e-3), "the reader tracks truth"
    assert drift > 2.0, f"drift over {truth:.1f}s was only {drift:.3f}s"
    assert reader.timeline.variable_rate
    assert reader.timeline.fps_was_wrong


def test_the_reader_falls_back_to_arithmetic_when_there_are_no_timestamps(
    monkeypatch, tmp_path
):
    """Some backends answer 0.0 for every frame. That reads as a valid,
    monotonic timeline and is pure fiction, so it has to be rejected and the
    fallback has to say it was used."""
    capture = FakeCapture([None] * 10, reported_fps=TEST_FPS)
    reader = Reader(_inject(monkeypatch, tmp_path, capture))
    frames = list(reader)

    assert len(frames) == 10
    assert reader.timeline.source == clip.FROM_INDEX
    assert [f.source for f in frames[1:]] == [clip.FROM_INDEX] * 9
    assert frames[-1].at == pytest.approx(9 / TEST_FPS)


# --- damage ---------------------------------------------------------------

def test_a_corrupt_frame_does_not_truncate_the_clip(tmp_path):
    """The headline claim, with both numbers in the assertion.

    One JPEG frame of thirty is wiped. The ordinary read loop stops dead at it
    and reports no error, so the caller gets a third of the clip and believes
    it is the whole thing. Advancing past the bad position instead costs the
    one damaged frame and keeps the rest.
    """
    pristine = _write(tmp_path / "a.avi", frames=30)
    damaged = _damage(pristine, tmp_path / "bad.avi", which=10)

    naive = _naive_frame_count(damaged)
    recovered = list(read(damaged))

    assert naive == 10, f"the naive loop got {naive}"
    assert len(recovered) == 29, f"recovery got {len(recovered)} of 30"
    assert len(recovered) > naive * 2


def test_the_timeline_survives_a_corrupt_frame(tmp_path):
    """Skipping is only useful if what comes after keeps its real time. If the
    reader closed the gap instead, every timestamp past the damage would be one
    frame early and a pose track would drift against its own audio.
    """
    pristine = _write(tmp_path / "a.avi", frames=30)
    damaged = _damage(pristine, tmp_path / "bad.avi", which=10)

    good = [f.at for f in read(pristine)]
    after = [f.at for f in read(damaged)]

    tolerance = _tolerance(TEST_FPS) / 2
    for at in after:
        assert min(abs(at - g) for g in good) < tolerance, (
            f"{at:.3f}s matches no real frame time"
        )
    assert max(after) == pytest.approx(max(good), abs=tolerance)
    assert len(after) == len(good) - 1


def test_the_damage_is_counted_not_hidden(tmp_path):
    pristine = _write(tmp_path / "a.avi", frames=30)
    reader = Reader(_damage(pristine, tmp_path / "bad.avi"))
    list(reader)
    assert reader.timeline.skipped >= 1
    assert "skipped" in reader.timeline.describe()


def test_a_frame_that_advances_but_will_not_decode_is_skipped(monkeypatch, tmp_path):
    """The other damage shape: the packet is there, the pixels are not. It has
    to cost one frame, not the remainder of the clip."""
    capture = FakeCapture([i * 100.0 for i in range(10)], undecodable={4})
    reader = Reader(_inject(monkeypatch, tmp_path, capture))
    frames = list(reader)

    assert len(frames) == 9
    assert reader.timeline.skipped == 1
    assert 4 not in [f.index for f in frames]
    assert frames[-1].at == pytest.approx(0.9), "the tail keeps its real time"


def test_a_run_of_damage_ends_the_stream_rather_than_looping_forever(
    monkeypatch, tmp_path
):
    """Recovery has to give up somewhere, or a file that ends in garbage would
    seek past the end for as long as the process lives."""
    stamps = [i * 100.0 for i in range(200)]
    dead = set(range(5, 200))
    capture = FakeCapture(stamps, ungrabbable=dead, reported_frames=len(stamps))
    frames = list(Reader(_inject(monkeypatch, tmp_path, capture)))
    assert len(frames) == 5
    assert capture.position <= 5 + clip.MAX_CONSECUTIVE_SKIPS + 1


# --- stride, limits, seeking ---------------------------------------------

def test_stride_keeps_at_truthful(tmp_path):
    """Every third frame of 10 fps footage is 10 fps footage sampled thinly,
    not 3.3 fps footage. Reporting the latter would make a gesture look three
    times slower than it was."""
    path = _write(tmp_path / "a.mp4", frames=30, fourcc=MP4V)
    frames = list(read(path, stride=3))

    assert [f.index for f in frames] == list(range(0, 30, 3))
    gaps = [b.at - a.at for a, b in zip(frames, frames[1:])]
    assert np.median(gaps) == pytest.approx(3 / TEST_FPS, rel=0.02)
    assert frames[-1].at == pytest.approx(27 / TEST_FPS, abs=1e-3)


def test_stride_reports_the_same_times_as_reading_everything(tmp_path):
    """The strongest form of the claim: a strided frame's timestamp is the one
    the full read gives the same frame, to the millisecond."""
    path = _write(tmp_path / "a.mp4", frames=24, fourcc=MP4V)
    full = {f.index: f.at for f in read(path)}
    for frame in read(path, stride=4):
        assert frame.at == pytest.approx(full[frame.index], abs=1e-3)


def test_stride_of_one_is_the_default_and_changes_nothing(tmp_path):
    assert inspect.signature(read).parameters["stride"].default == 1
    path = _write(tmp_path / "a.avi", frames=8)
    assert len(list(read(path, stride=1))) == len(list(read(path)))


def test_max_frames_stops_the_stream_early(tmp_path):
    path = _write(tmp_path / "a.avi", frames=40)
    frames = list(read(path, max_frames=5))
    assert len(frames) == 5
    assert [f.index for f in frames] == [0, 1, 2, 3, 4]


def test_max_frames_counts_what_is_handed_back_not_what_is_read(tmp_path):
    """With a stride in play the cap has to mean frames the caller sees,
    otherwise asking for 5 frames every third gives you fewer than 5."""
    path = _write(tmp_path / "a.avi", frames=40)
    frames = list(read(path, stride=3, max_frames=5))
    assert len(frames) == 5
    assert [f.index for f in frames] == [0, 3, 6, 9, 12]


def test_asking_for_no_frames_is_allowed(tmp_path):
    """A zero cap comes out of arithmetic in a caller, and raising on it would
    turn an empty batch into a crash."""
    assert list(read(_write(tmp_path / "a.avi", frames=5), max_frames=0)) == []


def test_start_skips_ahead_and_at_stays_absolute(tmp_path):
    """`at` means time in the clip, not time since the read began. A caller
    that seeks into a clip still needs its timestamps to line up with the
    source it seeked into."""
    path = _write(tmp_path / "a.mp4", frames=40, fourcc=MP4V)
    frames = list(read(path, start=2.0))

    assert frames, "seeking into the middle of a clip returned nothing"
    assert frames[0].at == pytest.approx(2.0, abs=_tolerance(TEST_FPS))
    assert frames[0].index == 0, "index counts from the start point"
    assert len(frames) == pytest.approx(20, abs=2)


def test_seeking_into_a_clip_with_no_timestamps_still_reports_absolute_time(
    monkeypatch, tmp_path
):
    """The two hard cases together. Without container timestamps `at` has to be
    counted, and counting from zero after a seek would report the second half of
    a clip as though it were the first."""
    capture = FakeCapture([None] * 40, reported_fps=TEST_FPS)
    reader = Reader(_inject(monkeypatch, tmp_path, capture), start=2.0)
    frames = list(reader)

    assert reader.timeline.source == clip.FROM_INDEX
    assert len(frames) == 20, "half the clip is behind the seek point"
    assert frames[0].at == pytest.approx(2.0, abs=_tolerance(TEST_FPS))
    assert frames[-1].at == pytest.approx(3.9, abs=_tolerance(TEST_FPS))


def test_seeking_past_the_end_returns_nothing_rather_than_failing(tmp_path):
    path = _write(tmp_path / "a.avi", frames=10)
    assert list(read(path, start=999.0)) == []


# --- downscaling ----------------------------------------------------------

def test_max_width_downscales_and_keeps_the_aspect_ratio(tmp_path):
    path = _write(tmp_path / "a.avi", frames=4, width=120, height=90)
    frame = next(iter(read(path, max_width=40)))
    assert frame.rgb.shape[1] == 40
    assert frame.rgb.shape[0] == 30, "90 * 40 / 120"


@pytest.mark.parametrize("target", [1, 7, 31, 64])
def test_downscaling_never_produces_a_zero_sized_frame(tmp_path, target):
    """A very wide frame scaled hard rounds its height towards zero, and
    cv2.resize raises on a zero dimension rather than returning anything."""
    path = _write(tmp_path / "wide.avi", frames=2, width=320, height=24)
    frame = next(iter(read(path, max_width=target)))
    assert frame.rgb.shape[0] >= 1
    assert frame.rgb.shape[1] == min(target, 320)


def test_a_clip_narrower_than_max_width_is_left_alone(tmp_path):
    """Enlarging cannot add detail a landmarker could use, and INTER_AREA is
    not an enlargement filter, so it would only add blur."""
    path = _write(tmp_path / "a.avi", frames=3, width=64, height=48)
    frame = next(iter(read(path, max_width=640)))
    assert frame.rgb.shape[:2] == (48, 64)


def test_downscaling_uses_area_averaging(tmp_path):
    """Pinned by comparing against the filter itself rather than by trusting
    the argument was passed. Nearest-neighbour would alias a hand at small
    scale into a different shape, and the result would still look plausible."""
    path = _write(tmp_path / "a.avi", frames=3, width=120, height=90)
    full = next(iter(read(path)))
    small = next(iter(read(path, max_width=40)))
    expected = cv2.resize(full.rgb, (40, 30), interpolation=cv2.INTER_AREA)
    assert np.array_equal(small.rgb, expected)


def test_downscaling_does_not_disturb_the_timeline(tmp_path):
    path = _write(tmp_path / "a.mp4", frames=20, fourcc=MP4V, width=160, height=120)
    full = [f.at for f in read(path)]
    small = [f.at for f in read(path, max_width=48)]
    assert full == pytest.approx(small, abs=1e-6)


# --- errors ---------------------------------------------------------------

def test_a_missing_file_says_which_file_is_missing(tmp_path):
    missing = tmp_path / "not_here.mp4"
    with pytest.raises(ClipMissing) as raised:
        list(read(missing))
    assert str(missing) in str(raised.value)
    assert isinstance(raised.value, FileNotFoundError), "existing guards keep working"
    assert isinstance(raised.value, ClipError)


def test_a_file_that_is_not_video_says_so_and_says_how_to_check(tmp_path):
    """OpenCV's own answer to this is a boolean and a line on stderr the caller
    never sees, which is how an ingest ends up reporting nothing at all."""
    path = tmp_path / "notes.mp4"
    path.write_text("this is not a video")
    with pytest.raises(ClipUndecodable) as raised:
        list(read(path))
    assert "ffprobe" in str(raised.value)


def test_a_truncated_file_is_undecodable_rather_than_empty(tmp_path):
    """Half an mp4 has no moov atom, so it will not open at all. Returning zero
    frames would let a broken download pass as an empty clip."""
    whole = _write(tmp_path / "a.mp4", frames=30, fourcc=MP4V)
    half = tmp_path / "half.mp4"
    half.write_bytes(whole.read_bytes()[: whole.stat().st_size // 2])
    with pytest.raises(ClipUndecodable):
        list(read(half))


def test_a_directory_is_not_a_clip(tmp_path):
    with pytest.raises(ClipUndecodable) as raised:
        probe(tmp_path)
    assert "directory" in str(raised.value)


def test_probe_raises_the_same_errors_as_read(tmp_path):
    with pytest.raises(ClipMissing):
        probe(tmp_path / "nope.mp4")


def test_the_error_arrives_before_the_first_frame_is_asked_for(tmp_path):
    """A generator that only fails on the first `next` puts the error a long
    way from the call that caused it. Constructing the Reader is where a bad
    path should blow up."""
    with pytest.raises(ClipMissing):
        Reader(tmp_path / "nope.mp4")


@pytest.mark.parametrize("kwargs", [
    {"stride": 0},
    {"stride": -1},
    {"max_frames": -1},
    {"max_width": 0},
    {"start": -1.0},
])
def test_nonsense_arguments_are_rejected_immediately(tmp_path, kwargs):
    path = _write(tmp_path / "a.avi", frames=3)
    with pytest.raises(ValueError):
        Reader(path, **kwargs)


# --- resources and cost ---------------------------------------------------

def test_read_is_a_generator_not_a_list(tmp_path):
    path = _write(tmp_path / "a.avi", frames=6)
    stream = read(path)
    assert inspect.isgenerator(stream)
    assert isinstance(next(stream), Frame)
    stream.close()


def test_a_long_clip_never_loads_into_memory_at_once(tmp_path):
    """The measurement behind calling `read` a generator.

    300 frames of 480x360 is about 155 MB of pixels. Streaming them holds one
    frame; collecting them holds all of it. Both numbers are printed by the
    assertion when it fails, so the margin is visible rather than asserted.
    """
    width, height, frames = 480, 360, 300
    path = _write(tmp_path / "long.avi", frames=frames, width=width, height=height)
    total_mb = frames * width * height * clip.CHANNELS_RGB / BYTES_PER_MB

    base = _rss_mb()
    seen = 0
    for _ in read(path):
        seen += 1
    streamed = _rss_mb() - base

    held = [f.rgb for f in read(path)]
    greedy = _rss_mb() - base - streamed
    del held

    assert seen == frames
    assert streamed < total_mb / 10, (
        f"streaming {total_mb:.0f} MB of pixels grew resident memory by "
        f"{streamed:.1f} MB"
    )
    assert greedy > streamed, (
        f"holding every frame grew by {greedy:.1f} MB against {streamed:.1f} MB "
        "streamed"
    )


def test_the_first_frame_arrives_without_decoding_the_whole_clip(tmp_path):
    """Laziness with a number on it. A pipeline that peeks at frame one of a
    long clip should not pay for the rest of it."""
    path = _write(tmp_path / "long.avi", frames=300, width=320, height=240)

    started = time.perf_counter()
    first = next(iter(read(path)))
    to_first = time.perf_counter() - started

    started = time.perf_counter()
    total = sum(1 for _ in read(path))
    to_all = time.perf_counter() - started

    assert total == 300
    assert first.index == 0
    assert to_first < to_all / 10, (
        f"first frame took {to_first * 1000:.1f} ms, all {total} took "
        f"{to_all * 1000:.1f} ms"
    )


def test_stride_costs_less_than_decoding_every_frame(tmp_path):
    """Advancing the stream and decoding pixels are separate calls, so a
    strided read pays for the pixels it keeps and not the ones it drops. If
    this ever regresses, stride has become a filter applied after the work."""
    path = _write(tmp_path / "long.avi", frames=300, width=480, height=360)

    started = time.perf_counter()
    full = sum(1 for _ in read(path))
    full_seconds = time.perf_counter() - started

    started = time.perf_counter()
    strided = sum(1 for _ in read(path, stride=5))
    strided_seconds = time.perf_counter() - started

    assert full == 300 and strided == 60
    assert strided_seconds < full_seconds, (
        f"stride 5 took {strided_seconds * 1000:.0f} ms, all frames took "
        f"{full_seconds * 1000:.0f} ms"
    )


def test_a_reader_releases_its_capture(monkeypatch, tmp_path):
    """An ingest that walks a directory of clips and leaks one descriptor each
    time dies part way through with an error about open files, a long way from
    the code that caused it."""
    capture = FakeCapture([i * 100.0 for i in range(4)])
    path = _inject(monkeypatch, tmp_path, capture)

    with Reader(path) as reader:
        list(reader)
    assert capture.released


def test_an_abandoned_read_still_releases_its_capture(monkeypatch, tmp_path):
    """The common case is worse than the tidy one: a caller takes two frames
    and walks away, leaving the generator suspended."""
    capture = FakeCapture([i * 100.0 for i in range(50)])
    path = _inject(monkeypatch, tmp_path, capture)

    stream = read(path)
    next(stream)
    next(stream)
    stream.close()
    assert capture.released


def test_the_timeline_describes_itself_in_one_line(monkeypatch, tmp_path):
    """So a bad clip is visible in a log rather than only under a debugger."""
    capture = FakeCapture(
        [i * (clip.MS_PER_SECOND / 10) for i in range(20)], reported_fps=30.0
    )
    reader = Reader(_inject(monkeypatch, tmp_path, capture))
    list(reader)
    line = reader.timeline.describe()
    assert "20 frames" in line
    assert "30.000" in line and "10.000" in line
