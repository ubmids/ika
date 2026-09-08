"""The real-data comparison, which is the module backing the loudest claim.

The 96.1% against 80.8% figure comes from here, and until now nothing checked
the parts that could quietly inflate it. Three things in particular:

The **person-grouped split**. HaGRID carries a user_id and the same person
appears in many photos, so a random split lets a model score by recognising
individuals. A leak here would raise both numbers and invalidate the
comparison.

The **label picking**. An image filed under `peace` usually contains a second
hand labelled `no_gesture`, so taking the first box would train on the wrong
hand for a fraction of every class.

The **identical examples** rule. About 8% of hands have no landmark
annotation. Give the CNN 3,391 images and the landmark model 3,109 and the gap
between them is partly a difference in dataset rather than in method.
"""

from pathlib import Path

import numpy as np
import pytest

from ika import vision

ROOT = Path(__file__).resolve().parent.parent / "data" / "hagrid"
needs_data = pytest.mark.skipif(
    not (ROOT / "annotations.json").exists(),
    reason="HaGRID subset not downloaded; see the README",
)


# --- things that need no dataset -----------------------------------------

def test_crop_is_square_and_padded():
    """A tight box cuts the wrist off, and wrist angle distinguishes several
    gestures. Squaring avoids the aspect squash a plain resize would apply."""
    left, top, right, bottom = vision.crop_box(1000, 800, (0.4, 0.4, 0.1, 0.05), pad=0.25)
    assert right > left and bottom > top
    assert abs((right - left) - (bottom - top)) <= 2, "crop should be square"
    assert (right - left) > 0.1 * 1000, "and larger than the raw box"


def test_crop_stays_inside_the_image():
    for bbox in [(0.0, 0.0, 0.1, 0.1), (0.95, 0.95, 0.1, 0.1), (0.5, 0.0, 0.4, 0.05)]:
        left, top, right, bottom = vision.crop_box(640, 480, bbox)
        assert 0 <= left < right <= 640
        assert 0 <= top < bottom <= 480


def test_split_by_user_never_leaks_a_person():
    """The property the whole comparison rests on."""
    records = [
        vision.Record(Path(f"{i}.jpg"), "peace", (0.1, 0.1, 0.2, 0.2),
                      np.zeros((21, 2), np.float32), f"user{i % 40}")
        for i in range(400)
    ]
    for seed in range(5):
        train, val, held, total = vision.split_by_user(records, fraction=0.25, seed=seed)
        assert total == 40 and held > 0
        train_users = {records[i].user_id for i in train}
        val_users = {records[i].user_id for i in val}
        assert not (train_users & val_users), f"seed {seed} leaked {train_users & val_users}"
        assert len(train) + len(val) == len(records)


def test_split_by_user_is_not_a_random_split():
    """A random split of these records would put every user on both sides."""
    records = [
        vision.Record(Path(f"{i}.jpg"), "peace", (0.1, 0.1, 0.2, 0.2),
                      np.zeros((21, 2), np.float32), f"user{i % 5}")
        for i in range(100)
    ]
    train, val, _, _ = vision.split_by_user(records, fraction=0.4, seed=0)
    assert not ({records[i].user_id for i in train} & {records[i].user_id for i in val})


def test_landmark_features_pad_two_dimensions_to_three():
    """HaGRID landmarks are 2D. The depth our live pipeline gets from
    MediaPipe's world landmarks simply is not in this dataset, so the palm
    basis here is built inside the image plane."""
    from ika.features import FEATURE_DIM

    rng = np.random.default_rng(0)
    records = [
        vision.Record(Path("a.jpg"), "peace", (0.1, 0.1, 0.2, 0.2),
                      rng.random((21, 2)).astype(np.float32), "u1")
    ]
    out = vision.landmark_features(records)
    assert out.shape == (1, FEATURE_DIM)
    assert np.isfinite(out).all()


def test_landmark_features_handle_a_missing_hand():
    records = [vision.Record(Path("a.jpg"), "peace", (0.1,) * 4, None, "u1")]
    assert np.isfinite(vision.landmark_features(records)).all()


def test_augmentation_never_mirrors():
    """Several HaGRID classes are distinguished by orientation, and `peace`
    against `peace_inverted` is exactly the pair a horizontal flip would
    merge. Tested by behaviour: an earlier version of this test searched the
    source for the word "flip" and failed on the comment explaining why there
    is none.
    """
    rng = np.random.default_rng(0)
    # Left half bright, right half dark, so a mirror is unmistakable.
    crop = np.zeros((1, 16, 16, 3), dtype=np.uint8)
    crop[:, :, :8] = 255
    for _ in range(20):
        out = vision.to_tensor(crop, augment=True, rng=rng).numpy()[0]
        left = out[:, :, :8].mean()
        right = out[:, :, 8:].mean()
        assert left > right, "augmentation mirrored the image"


def test_augmentation_only_changes_brightness_and_noise():
    """Mild on purpose: geometry is signal here, not nuisance."""
    rng = np.random.default_rng(1)
    rng2 = np.random.default_rng(1)
    crops = np.full((4, 16, 16, 3), 120, dtype=np.uint8)
    plain = vision.to_tensor(crops).numpy()
    noisy = vision.to_tensor(crops, augment=True, rng=rng).numpy()
    assert plain.shape == noisy.shape
    assert not np.allclose(plain, noisy), "augment=True did nothing"
    # Deterministic given the same generator, so runs are reproducible.
    assert np.allclose(noisy, vision.to_tensor(crops, augment=True, rng=rng2).numpy())


def test_to_tensor_normalises_and_reorders():
    crops = np.full((3, 16, 16, 3), 128, dtype=np.uint8)
    out = vision.to_tensor(crops)
    assert tuple(out.shape) == (3, 3, 16, 16), "should be NCHW"
    assert abs(float(out.mean())) < 1.0, "should be roughly zero-centred"


def test_backbones_accept_the_class_count():
    for name in ("mobilenet_v3_small",):
        model = vision.build_backbone(name, 7)
        out = model(vision.to_tensor(np.zeros((2, 64, 64, 3), np.uint8)))
        assert tuple(out.shape) == (2, 7)


def test_unknown_backbone_is_refused():
    with pytest.raises(ValueError, match="unknown backbone"):
        vision.build_backbone("not_a_model", 3)


# --- things that need the dataset ----------------------------------------

@needs_data
def test_every_record_has_the_landmarks_it_promises():
    """`require_landmarks` is what makes both models see the same examples."""
    records, _ = vision.load_hagrid(ROOT, require_landmarks=True)
    assert records
    assert all(r.landmarks is not None for r in records)
    assert all(r.landmarks.shape == (21, 2) for r in records)


@needs_data
def test_requiring_landmarks_actually_drops_some():
    strict, _ = vision.load_hagrid(ROOT, require_landmarks=True)
    loose, _ = vision.load_hagrid(ROOT, require_landmarks=False)
    assert len(loose) > len(strict), "if these match, the flag is doing nothing"


@needs_data
def test_the_labelled_hand_is_the_one_performing_the_gesture():
    """Images commonly hold a second hand labelled no_gesture. Taking the
    first box would label that other hand as the gesture being performed."""
    import json

    annotations = json.loads((ROOT / "annotations.json").read_text())
    records, _ = vision.load_hagrid(ROOT)
    multi = 0
    for record in records:
        entry = annotations[record.label][record.path.stem]
        labels = entry["labels"]
        if len(labels) > 1:
            multi += 1
            which = labels.index(record.label)
            assert record.bbox == tuple(entry["bboxes"][which])
    assert multi > 50, f"only {multi} multi-hand images; test proves little"


@needs_data
def test_every_class_survives_the_split():
    """A person-grouped split can accidentally empty a class, and then the
    reported accuracy is measuring something other than what it claims."""
    records, classes = vision.load_hagrid(ROOT)
    _, val, _, _ = vision.split_by_user(records, fraction=0.25, seed=0)
    present = {records[i].label for i in val}
    assert len(present) == len(classes), sorted(set(classes) - present)


@needs_data
def test_crops_come_back_as_images_not_blanks():
    records, _ = vision.load_hagrid(ROOT)
    crops = vision.load_crops(records[:24], size=64)
    assert crops.shape == (24, 64, 64, 3) and crops.dtype == np.uint8
    blank = [i for i, c in enumerate(crops) if c.std() < 1.0]
    assert len(blank) <= 1, f"{len(blank)} of 24 crops are flat"
