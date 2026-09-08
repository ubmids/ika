"""Ingesting a folder of labelled clips, which is how real footage arrives.

The two properties worth testing are both about not fooling ourselves. A clip
that yields no body has to be reported rather than silently dropped, because
an unusable corpus and a hard problem look identical once a number comes out
the far end. And splits happen at clip level, never frame level, since frames
within a clip are near-duplicates and a frame-level split reports memory as
accuracy. That is the same trap that inflated the HaGRID comparison until it
was split by person.
"""

from pathlib import Path

import numpy as np
import pytest

from ika.corpus import (MIN_USABLE_FRAMES, Clip, find_clips, read_clip,
                        report, split_by_clip)

MODEL = Path(__file__).resolve().parent.parent / "models" / "pose_landmarker_lite.task"
HAGRID = Path(__file__).resolve().parent.parent / "data" / "hagrid" / "annotations.json"
needs_model = pytest.mark.skipif(not MODEL.exists(), reason="pose model not downloaded")
needs_photos = pytest.mark.skipif(not HAGRID.exists(), reason="no photographs to build clips")


def fake(label, name="c.avi", frames=30, with_body=30):
    clip = Clip(path=Path(name), label=label)
    clip.frames, clip.with_body = frames, with_body
    return clip


# --- discovery ------------------------------------------------------------

def test_it_finds_clips_by_folder_name(tmp_path):
    for label in ("punch", "kick"):
        (tmp_path / label).mkdir()
        for i in range(3):
            (tmp_path / label / f"{i}.avi").write_bytes(b"")
    clips, classes = find_clips(tmp_path)
    assert classes == ["kick", "punch"]
    assert len(clips) == 6


def test_it_ignores_files_that_are_not_video(tmp_path):
    (tmp_path / "punch").mkdir()
    (tmp_path / "punch" / "a.avi").write_bytes(b"")
    (tmp_path / "punch" / "notes.txt").write_bytes(b"")
    (tmp_path / "punch" / ".DS_Store").write_bytes(b"")
    clips, _ = find_clips(tmp_path)
    assert [c.path.name for c in clips] == ["a.avi"]


def test_per_class_caps_the_count(tmp_path):
    (tmp_path / "punch").mkdir()
    for i in range(10):
        (tmp_path / "punch" / f"{i:02d}.avi").write_bytes(b"")
    clips, _ = find_clips(tmp_path, per_class=4)
    assert len(clips) == 4


def test_a_missing_class_is_an_error_not_a_silent_omission(tmp_path):
    (tmp_path / "punch").mkdir()
    with pytest.raises(FileNotFoundError, match="kick"):
        find_clips(tmp_path, classes=["punch", "kick"])


def test_a_non_directory_is_refused(tmp_path):
    target = tmp_path / "not_a_dir.txt"
    target.write_bytes(b"")
    with pytest.raises(NotADirectoryError):
        find_clips(target)


# --- usability, which must be visible ------------------------------------

def test_a_clip_with_too_few_bodies_is_unusable():
    assert not fake("punch", with_body=MIN_USABLE_FRAMES - 1).usable
    assert fake("punch", with_body=MIN_USABLE_FRAMES).usable


def test_the_detection_rate_is_reported():
    """A corpus where bodies are rarely found is an unusable input, not a hard
    problem, and the two are indistinguishable once training has happened."""
    clip = fake("punch", frames=100, with_body=12)
    assert clip.detection_rate == pytest.approx(0.12)


def test_an_empty_clip_does_not_divide_by_zero():
    assert Clip(path=Path("x.avi"), label="punch").detection_rate == 0.0


def test_the_report_flags_a_class_pose_cannot_see():
    text = report([fake("punch", with_body=30) for _ in range(4)]
                  + [fake("bagwork", with_body=1) for _ in range(4)])
    assert "punch" in text and "bagwork" in text
    lines = {line.split()[0]: line for line in text.splitlines() if line.strip()}
    assert "too few bodies" in lines["bagwork"]
    assert "too few bodies" not in lines["punch"]


# --- splitting ------------------------------------------------------------

def test_the_split_is_by_clip_not_by_frame():
    """Frames within a clip are near-duplicates, so a frame-level split
    reports memory of the same movement as accuracy."""
    clips = [fake("punch", name=f"p{i}.avi") for i in range(8)]
    clips += [fake("kick", name=f"k{i}.avi") for i in range(8)]
    train, val = split_by_clip(clips, fraction=0.25, seed=0)
    assert len(train) + len(val) == len(clips)
    assert not (set(train.tolist()) & set(val.tolist()))


def test_every_class_survives_the_split():
    clips = [fake("punch", name=f"p{i}.avi") for i in range(6)]
    clips += [fake("kick", name=f"k{i}.avi") for i in range(6)]
    for seed in range(5):
        _, val = split_by_clip(clips, fraction=0.25, seed=seed)
        assert {clips[i].label for i in val} == {"punch", "kick"}


def test_a_single_clip_class_goes_to_training_rather_than_vanishing():
    clips = [fake("punch", name="p.avi")] + [fake("kick", name=f"k{i}.avi") for i in range(4)]
    train, val = split_by_clip(clips, fraction=0.25, seed=0)
    assert 0 in train.tolist()


# --- end to end -----------------------------------------------------------

@needs_model
@needs_photos
def test_it_reads_bodies_out_of_a_real_corpus(tmp_path):
    """A miniature corpus built from photographs of actual people, because a
    pose model finds nothing in synthetic stick figures."""
    import cv2

    from ika.vision import load_hagrid
    from ika.watch import Watcher

    records, _ = load_hagrid(HAGRID.parent)
    rng = np.random.default_rng(0)
    picks = [records[i] for i in rng.choice(len(records), 2, replace=False)]

    for label, record in zip(("left", "right"), picks):
        (tmp_path / label).mkdir()
        image = cv2.imread(str(record.path))
        sprite = cv2.resize(image, (int(image.shape[1] * 380 / image.shape[0]), 380))
        for take in range(2):
            path = tmp_path / label / f"{take}.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                     30.0, (480, 400))
            for i in range(14):
                frame = np.full((400, 480, 3), 40, dtype=np.uint8)
                drift = i * (3 if label == "right" else -3)
                x = 120 + drift
                sh, sw = sprite.shape[:2]
                left, right = max(0, x), min(480, x + sw)
                if right > left:
                    frame[400 - sh : 400, left:right] = sprite[:, left - x : right - x]
                writer.write(frame)
            writer.release()

    clips, classes = find_clips(tmp_path)
    assert len(clips) == 4 and classes == ["left", "right"]

    with Watcher(max_bodies=1) as watcher:
        clips = [read_clip(c, watcher, max_width=480) for c in clips]

    assert all(c.frames == 14 for c in clips), [c.frames for c in clips]
    usable = [c for c in clips if c.usable]
    assert usable, f"no clip yielded a body: {report(clips)}"

    from ika.posture import FEATURE_DIM

    for clip in usable:
        sequence = clip.windows[0]
        assert len(sequence) >= MIN_USABLE_FRAMES
        for frame in sequence:
            assert frame.features.shape == (FEATURE_DIM,)
            assert np.isfinite(frame.features).all()
        # Timestamps must advance, or every movement feature is a rate of zero.
        times = [f.at for f in sequence]
        assert times == sorted(times) and times[-1] > times[0]
