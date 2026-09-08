"""The integrated pipeline: video in, identified stabilised bodies out.

The unit tests here are mostly about the mask, because the mask is the join
between two modules and its polarity is inverted from OpenCV's. Getting it
backwards masks out the background and leaves the subject as the only thing
voting on camera motion, which is precisely the failure it exists to prevent.
I got it backwards once and measured a wrong number before reading the
docstring, so it is pinned here.
"""

from pathlib import Path

import numpy as np
import pytest

from ika import watch as watch_mod
from ika.body import Body, N_LANDMARKS
from ika.stabilise import Motion
from ika.watch import MASK_PADDING, TRUST_FLOOR, Observation, Sighting, subject_mask

MODEL = Path(__file__).resolve().parent.parent / "models" / "pose_landmarker_lite.task"
HAGRID = Path(__file__).resolve().parent.parent / "data" / "hagrid" / "annotations.json"
needs_model = pytest.mark.skipif(not MODEL.exists(), reason="pose model not downloaded")
needs_data = pytest.mark.skipif(not HAGRID.exists(), reason="no photographs to build a clip from")


def body_in(x0, y0, x1, y1, visible=1.0):
    marks = np.zeros((N_LANDMARKS, 3), dtype=np.float32)
    marks[:, 0] = np.linspace(x0, x1, N_LANDMARKS)
    marks[:, 1] = np.linspace(y0, y1, N_LANDMARKS)
    return Body(image=marks, world=marks.copy(),
                visibility=np.full(N_LANDMARKS, visible, dtype=np.float32))


# --- the mask -------------------------------------------------------------

def test_the_mask_marks_the_subject_not_the_background():
    """Non-zero means subject, the inverse of OpenCV's convention. Inverting
    this leaves the subject as the only thing voting on camera motion."""
    mask = subject_mask([body_in(0.4, 0.4, 0.6, 0.6)], (480, 640))
    assert mask is not None
    centre = mask[int(0.5 * 480), int(0.5 * 640)]
    corner = mask[5, 5]
    assert centre > 0, "the subject must be marked"
    assert corner == 0, "the background must not be"


def test_no_bodies_means_no_mask_rather_than_an_empty_one():
    """An all-zero mask and no mask mean different things to `estimate`, and
    with nothing detected there is no subject to exclude."""
    assert subject_mask([], (480, 640)) is None


def test_the_mask_reaches_past_the_landmarks():
    """Landmarks sit inside the silhouette, and an outline is exactly what a
    corner detector likes best, so a tight box leaves the fighter's edge
    unmasked and still voting."""
    tight = subject_mask([body_in(0.45, 0.45, 0.55, 0.55)], (480, 640),
                         padding=0.0)
    padded = subject_mask([body_in(0.45, 0.45, 0.55, 0.55)], (480, 640),
                          padding=MASK_PADDING)
    assert (padded > 0).sum() > (tight > 0).sum()


def test_both_fighters_are_masked():
    mask = subject_mask([body_in(0.05, 0.4, 0.2, 0.9),
                         body_in(0.75, 0.4, 0.95, 0.9)], (480, 640))
    assert mask[int(0.6 * 480), int(0.12 * 640)] > 0
    assert mask[int(0.6 * 480), int(0.85 * 640)] > 0
    assert mask[int(0.6 * 480), int(0.5 * 640)] == 0, "the gap between them is background"


def test_an_invisible_body_is_not_masked():
    """A body the landmarker guessed at should not blank out a region of
    perfectly good background."""
    assert subject_mask([body_in(0.4, 0.4, 0.6, 0.6, visible=0.05)], (480, 640)) is None


def test_the_mask_is_clipped_to_the_frame():
    mask = subject_mask([body_in(-0.5, -0.5, 1.5, 1.5)], (240, 320))
    assert mask.shape == (240, 320) and (mask > 0).all()


# --- observation bookkeeping ---------------------------------------------

def test_a_held_sighting_is_not_a_seen_one():
    """A body remembered through an occlusion must be distinguishable from one
    actually observed, or downstream code will treat a guess as evidence."""
    seen = Sighting(id=1, track=None, features=np.zeros(3), missing=0)
    held = Sighting(id=2, track=None, features=np.zeros(3), missing=4)
    assert not seen.held and held.held
    obs = Observation(at=0.0, index=0, sightings=[seen, held],
                      camera=Motion(0, 0, 1, 1, 0), compensated=False)
    assert [s.id for s in obs.seen] == [1]


def test_the_trust_floor_is_not_zero():
    """An untrusted camera estimate must be ignored, because applying a
    confident-looking wrong number injects camera motion instead of removing
    it. Doing nothing is the better failure."""
    assert TRUST_FLOOR > 0.0


# --- end to end -----------------------------------------------------------

@needs_model
@needs_data
def test_it_reads_two_stable_identities_off_a_panning_clip(tmp_path):
    """The whole point of the module, on real people with a known camera pan.

    Synthetic stick figures were tried first and a pose model found nothing in
    60 frames, so the figures here are photographs of actual people.
    """
    import cv2

    from ika.vision import load_hagrid

    records, _ = load_hagrid(HAGRID.parent)
    rng = np.random.default_rng(1)
    picks = [records[i] for i in rng.choice(len(records), 2, replace=False)]
    sprites = []
    for record in picks:
        image = cv2.imread(str(record.path))
        height = 360
        sprites.append(cv2.resize(image, (int(image.shape[1] * height / image.shape[0]), height)))

    width, frame_h, count, pan = 640, 480, 24, 1.5
    background = cv2.cvtColor(
        cv2.GaussianBlur((rng.random((frame_h, width * 2)) * 255).astype(np.uint8), (9, 9), 0),
        cv2.COLOR_GRAY2BGR,
    )
    path = tmp_path / "pair.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, frame_h))
    for i in range(count):
        offset = int(i * pan)
        frame = background[:, offset : offset + width].copy()
        for k, sprite in enumerate(sprites):
            sh, sw = sprite.shape[:2]
            x = int((120 + k * 300) - offset)
            left, right = max(0, x), min(width, x + sw)
            if right > left:
                frame[frame_h - sh : frame_h, left:right] = sprite[:, left - x : right - x]
        writer.write(frame)
    writer.release()

    observations = list(watch_mod.watch(path, max_width=width))
    assert len(observations) == count

    identities = {tuple(sorted(s.id for s in o.sightings)) for o in observations}
    assert identities == {(1, 2)}, f"identities were not stable: {identities}"

    from ika.posture import FEATURE_DIM

    for observation in observations:
        for sighting in observation.sightings:
            assert sighting.features.shape == (FEATURE_DIM,)
            assert np.isfinite(sighting.features).all()

    # The camera pan must be seen and removed. Cumulative drift is what
    # compensation uses, and a per-frame pan of 1.5 px is sub-pixel after the
    # stabiliser's own downscale, so the per-frame figure under-reads while the
    # accumulated one does not.
    assert sum(o.compensated for o in observations) > count * 0.8


def test_one_watcher_survives_many_clips():
    """MediaPipe rejects a timestamp that does not increase, and every clip in
    a corpus starts again at zero. Reusing one Watcher across a folder of clips
    therefore made time run backwards and threw, which would have hit on the
    first real dataset. The landmarker gets the Watcher's own rising clock
    while the caller's real seconds travel in the Observation.
    """
    import pytest as _pytest

    from ika.watch import Watcher

    if not MODEL.exists():
        _pytest.skip("pose model not downloaded")

    rng = np.random.default_rng(0)
    frames = [(rng.random((120, 160, 3)) * 255).astype(np.uint8) for _ in range(4)]

    with Watcher(max_bodies=1, stabilise=False) as watcher:
        for clip in range(3):
            watcher.reset()
            for i, frame in enumerate(frames):
                # every clip restarts at zero, which is the whole point
                observation = watcher.observe(frame, i / 30.0, i)
                assert observation.at == pytest.approx(i / 30.0), (
                    "the caller's real time must survive, since every movement "
                    "feature downstream is a rate"
                )
