"""Checking a clone for the pieces that are not in it.

Everything here runs against a tree built in `tmp_path`, so no test needs the
network, a download, or the real `models/` and `data/` folders. That matters
twice over: the suite has to pass on a fresh clone where none of those files
exist yet, and a checker whose own tests depend on the files it checks for
proves nothing.

The property under test throughout is that complete and merely present are not
the same thing. A half-downloaded model satisfies `path.exists()` and then
fails inside MediaPipe with a flatbuffer error naming no cause, which is the
failure this module exists to move forward in time.
"""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from ika import setup_check
from ika.setup_check import (HAGRID_ANNOTATIONS, HAGRID_ANNOTATIONS_BYTES,
                             HAGRID_IMAGES, HAGRID_TRAINING_CLASSES,
                             HAND_LANDMARKER, HAND_LANDMARKER_BYTES, POSE_FULL,
                             POSE_FULL_BYTES, POSE_LITE, POSE_LITE_BYTES,
                             STATIC_CLASSIFIER, describe, missing, ready,
                             survey)

REAL_ROOT = Path(__file__).resolve().parent.parent

# The three landmarkers, as name to relative path to complete size.
MODELS = {
    HAND_LANDMARKER: ("models/hand_landmarker.task", HAND_LANDMARKER_BYTES),
    POSE_LITE: ("models/pose_landmarker_lite.task", POSE_LITE_BYTES),
    POSE_FULL: ("models/pose_landmarker_full.task", POSE_FULL_BYTES),
}
CHECKPOINT_BYTES = 4096   # any non-empty length, the real one is not fixed


def write(path: Path, size: int) -> Path:
    """Make a file of exactly `size` bytes, parents and all."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0" * size)
    return path


def complete_clone(root: Path) -> Path:
    """Build a tree where every requirement is satisfied."""
    for relative, size in MODELS.values():
        write(root / relative, size)
    write(root / "data" / "hagrid" / "annotations.json", HAGRID_ANNOTATIONS_BYTES)
    for label in HAGRID_TRAINING_CLASSES:
        write(root / "data" / "hagrid" / label / "a.jpg", 12)
    write(root / "checkpoints" / "static.pt", CHECKPOINT_BYTES)
    return root


def named(requirements, name):
    return next(r for r in requirements if r.name == name)


# --- the shape of a survey -------------------------------------------------

def test_a_survey_covers_every_piece_a_fresh_clone_lacks(tmp_path):
    names = [r.name for r in survey(tmp_path)]
    assert names == [HAND_LANDMARKER, POSE_LITE, POSE_FULL,
                     HAGRID_ANNOTATIONS, HAGRID_IMAGES, STATIC_CLASSIFIER]
    assert len(set(names)) == len(names)


def test_an_empty_directory_needs_everything(tmp_path):
    assert not ready(tmp_path)
    assert len(missing(tmp_path)) == len(survey(tmp_path))
    assert all(not r.exists for r in missing(tmp_path))


def test_a_complete_tree_is_ready(tmp_path):
    complete_clone(tmp_path)
    assert missing(tmp_path) == []
    assert ready(tmp_path)


def test_a_survey_reads_and_writes_nothing(tmp_path):
    complete_clone(tmp_path)
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    survey(tmp_path)
    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*")) == before


def test_a_requirement_cannot_be_edited_after_the_survey(tmp_path):
    hand = named(survey(tmp_path), HAND_LANDMARKER)
    with pytest.raises(FrozenInstanceError):
        hand.present = True


def test_a_survey_with_no_root_looks_at_the_repository(tmp_path):
    # Not at the working directory, which is why `ika live` works from anywhere.
    hand = named(survey(), HAND_LANDMARKER)
    assert hand.path == REAL_ROOT / "models" / "hand_landmarker.task"


# --- size, not just existence ---------------------------------------------

def test_a_truncated_model_is_reported_as_missing_not_present(tmp_path):
    complete_clone(tmp_path)
    relative, size = MODELS[HAND_LANDMARKER]
    write(tmp_path / relative, size - 1)

    hand = named(survey(tmp_path), HAND_LANDMARKER)
    assert not hand.present
    assert hand.exists          # the trap: it is there, it is just not whole
    assert hand.wrong_size
    assert hand.bytes_found == size - 1


def test_a_model_longer_than_expected_is_wrong_too(tmp_path):
    complete_clone(tmp_path)
    relative, size = MODELS[POSE_LITE]
    write(tmp_path / relative, size + 1)

    assert not named(survey(tmp_path), POSE_LITE).present


def test_an_absent_file_is_missing_but_not_wrong_size(tmp_path):
    hand = named(survey(tmp_path), HAND_LANDMARKER)
    assert not hand.present
    assert not hand.wrong_size
    assert hand.bytes_found is None


def test_a_checkpoint_with_no_known_size_is_judged_on_being_non_empty(tmp_path):
    complete_clone(tmp_path)
    checkpoint = tmp_path / "checkpoints" / "static.pt"
    assert named(survey(tmp_path), STATIC_CLASSIFIER).present

    write(checkpoint, 0)        # what an interrupted torch.save leaves behind
    assert not named(survey(tmp_path), STATIC_CLASSIFIER).present


# --- the dataset is two things ---------------------------------------------

def test_annotations_without_photographs_are_not_a_dataset(tmp_path):
    complete_clone(tmp_path)
    for label in HAGRID_TRAINING_CLASSES:
        (tmp_path / "data" / "hagrid" / label / "a.jpg").unlink()

    found = survey(tmp_path)
    assert named(found, HAGRID_ANNOTATIONS).present
    assert not named(found, HAGRID_IMAGES).present


def test_photographs_without_annotations_are_not_a_dataset(tmp_path):
    complete_clone(tmp_path)
    (tmp_path / "data" / "hagrid" / "annotations.json").unlink()

    found = survey(tmp_path)
    assert named(found, HAGRID_IMAGES).present
    assert not named(found, HAGRID_ANNOTATIONS).present


def test_a_dataset_short_one_gesture_is_incomplete(tmp_path):
    complete_clone(tmp_path)
    (tmp_path / "data" / "hagrid" / "no_gesture" / "a.jpg").unlink()

    assert not named(survey(tmp_path), HAGRID_IMAGES).present


def test_gesture_folders_that_hold_no_photographs_do_not_count(tmp_path):
    complete_clone(tmp_path)
    # A half finished unzip leaves the folders behind without their contents.
    (tmp_path / "data" / "hagrid" / "fist" / "a.jpg").unlink()
    (tmp_path / "data" / "hagrid" / "fist").mkdir(exist_ok=True)

    assert not named(survey(tmp_path), HAGRID_IMAGES).present


# --- fetchable against buildable ------------------------------------------

def test_the_landmarkers_and_the_dataset_can_be_downloaded(tmp_path):
    for name in (HAND_LANDMARKER, POSE_LITE, POSE_FULL,
                 HAGRID_ANNOTATIONS, HAGRID_IMAGES):
        assert named(survey(tmp_path), name).url is not None


def test_the_classifier_cannot_be_downloaded_only_trained(tmp_path):
    classifier = named(survey(tmp_path), STATIC_CLASSIFIER)
    assert classifier.url is None
    assert "scripts/train_from_hagrid.py" in classifier.note


def test_every_note_names_the_command_that_stops_working(tmp_path):
    for requirement in survey(tmp_path):
        assert "ika " in requirement.note or "scripts/" in requirement.note


def test_the_urls_match_the_modules_that_load_the_files():
    # The copies in setup_check exist so it needs no third party imports.
    # This is what stops them drifting from the code that uses them.
    from ika.body import POSE_URLS
    from ika.hands import HAND_MODEL_URL

    assert setup_check.HAND_MODEL_URL == HAND_MODEL_URL
    assert setup_check.POSE_LITE_URL == POSE_URLS["lite"]
    assert setup_check.POSE_FULL_URL == POSE_URLS["full"]


# --- what a person reads ---------------------------------------------------

def test_the_description_lists_everything_and_says_what_breaks(tmp_path):
    text = describe(survey(tmp_path))
    for requirement in survey(tmp_path):
        assert requirement.name in text
        assert requirement.note in text
    assert "MISSING" in text
    assert "6 of 6 still needed" in text


def test_the_description_of_a_complete_tree_raises_no_alarm(tmp_path):
    text = describe(survey(complete_clone(tmp_path)))
    assert "MISSING" not in text
    assert "WRONG SIZE" not in text
    assert "all 6 present" in text


def test_the_description_of_a_wrong_size_file_says_to_delete_it(tmp_path):
    complete_clone(tmp_path)
    relative, size = MODELS[POSE_FULL]
    write(tmp_path / relative, size - 1)

    text = describe(survey(tmp_path))
    assert "WRONG SIZE" in text
    assert "Delete it yourself" in text
    assert str(size) in text and str(size - 1) in text


def test_a_missing_download_is_described_with_its_url(tmp_path):
    text = describe(survey(tmp_path))
    assert setup_check.HAND_MODEL_URL in text
    assert setup_check.HAGRID_URL in text


# --- the expected sizes, when the real files happen to be here -------------
# Skipped on a fresh clone by design. Where the files are present they are the
# only check available on the constants themselves, since verifying them any
# other way would mean downloading.

@pytest.mark.parametrize("name", list(MODELS))
def test_a_landmarker_already_here_matches_its_expected_size(name):
    relative, size = MODELS[name]
    path = REAL_ROOT / relative
    if not path.is_file():
        pytest.skip("landmarker not downloaded")
    assert path.stat().st_size == size


def test_the_annotations_already_here_match_their_expected_size():
    path = REAL_ROOT / "data" / "hagrid" / "annotations.json"
    if not path.is_file():
        pytest.skip("HaGRID not downloaded")
    assert path.stat().st_size == HAGRID_ANNOTATIONS_BYTES
