# ika — control a computer with bare hands

ìka, Yoruba for finger. Webcam in, cursor and keystrokes out. No gloves, no
sensors, no cloud.

Named for the finger rather than the hand because ọwọ́ (hand) and owó (money)
collide in ASCII, and `~/Desktop/owo` was already a finance app.

```bash
pip install -e '.[dev]'

ika train --synthetic          # a model from invented hands, no recording needed
ika live                       # webcam, dry run: prints what it would do
ika record                     # capture your own gestures
ika train data/session.npz     # a model that knows your hands
ika live --live                # actually drive the machine
ika bindings                   # what each gesture does
pytest -q                      # 66 tests, no camera, no network
```

`ika live` is a dry run unless you pass `--live`. Everything is visible and
nothing is sent to the OS. That is the mode to develop in, because a gesture
classifier wired to a keyboard is a program that types whatever it
hallucinates.

## What is pretrained and what we actually train

Worth being explicit, because "fine-tune a vision model" can easily become
theatre where the pretrained thing does all the work.

**Pretrained:** MediaPipe's hand landmarker, 21 points per hand. Getting stable
landmarks off a moving hand in changing light took Google more data than a
laptop will see, and reproducing it badly would sink the project before the
interesting parts start. This is the eyes, and only the eyes.

**Ours, trained here:** the gesture classifier, the sequence model for gestures
that only exist as movement, and the entire decision layer. On an M2 Pro, no
external GPU.

## Measured on this machine

**Landmark throughput**, 640x480, one hand, M2 Pro CPU:

| | |
|---|---|
| detection | 30.9 ms/frame, **32.4 fps** |
| two hands in a real photo | both found, 0.94 and 0.96 confidence |

**Classifier accuracy** on synthetic hands, as the problem gets more realistic:

| landmarks | accuracy | weakest gesture |
|---|---|---|
| clean | 100.0% | all at 100% |
| noisy | 96.1% | point, 88% |
| noisy plus some lost | 93.1% | point, 85% |
| harsh | 83.6% | fist, 69% |

That first row is not a result, it is a wiring check. Every synthetic class is
a tight cloud around a pose designed to be distinctive, so 100% means the
pipeline is connected, nothing more. The interesting rows are the ones below.

**Gesture to action latency**, which is what your hand actually feels:

| dwell frames | smoothing | frames | at 32 fps |
|---|---|---|---|
| 3 | 0.0 | 3 | 94 ms |
| 3 | 0.6 | 6 | 188 ms |
| 5 | 0.6 | 8 | **250 ms** (default) |
| 8 | 0.8 | 15 | 469 ms |

Both knobs buy noise rejection with time. 250 ms is the default because it
feels deliberate rather than twitchy, and `--dwell 3 --smoothing 0.4` is
noticeably snappier if you would rather have that.

## What the confusion matrix caught

`point` and `l_shape` get mixed up in both directions, 10 and 7 times out of
100. That is not a training problem to tune away, it is a vocabulary problem:
`l_shape` *is* `point` with the thumb extended, so the two differ by one finger
and nothing else. Either drop one, or accept that whichever you use will
misfire into the other. The honest fix is a smaller vocabulary, and a gesture
you cannot perform by accident is worth more than a large alphabet.

## Why the feature engineering is the whole game

Feed raw landmarks to a classifier and it learns that "point" means *an index
finger in the upper left of frame, at roughly the size my hand was during
recording*. Lean in, or shift sideways, and it collapses. It memorised the
recording session, not the gesture.

So every hand is re-expressed in a frame built from itself: origin at the
wrist, scale from the palm length, axes from the knuckle row, which is the only
rigid part of a hand. What survives is shape, independent of position, distance
and rotation. There are tests asserting each of those invariances.

That can be taken too far. Thumbs up and thumbs down are the same shape,
differing only by orientation, so discarding orientation would silently merge
two gestures. Both are kept: canonical shape, and world orientation as separate
features. There is a test for that too, and the synthetic augmentation is
deliberately bounded below the half turn that separates the pair, because
rotating freely would have merged the classes and inflated the accuracy of a
model that had in fact learned to ignore orientation.

## The state machine, which matters more than accuracy

A classifier has an opinion every frame, thirty times a second, and is
sometimes wrong. Your hands are in shot constantly doing things that mean
nothing, and every one of them gets classified as *something*. Four mechanisms
against that:

- **Smoothing** so one bad frame cannot fire anything alone
- **Dwell**, a pose must persist, which separates holding a shape from passing
  through it on the way somewhere
- **Hysteresis**, a fired gesture is spent until something neutral is seen,
  otherwise holding a pose repeats it at frame rate
- **Engagement**, nothing fires until you deliberately arm it with an open
  palm, and it disarms itself when your hands go quiet

Continuous control deliberately bypasses all of it. A cursor must follow your
finger *this frame*, so pointing and pinching are read geometrically, with two
thresholds on the pinch so landmark noise cannot flicker a click.

## Bindings

| gesture | does |
|---|---|
| open palm | engage / disengage |
| point | move the cursor |
| pinch | click, or hold to drag |
| fist | play / pause |
| peace | next desktop |
| L shape | previous desktop |
| thumbs up / down | volume |

Nothing destructive is bound, and a test enforces it. No quit, no close, no
delete. A gesture system will misfire, so a misfire should cost an annoyance,
never lost work.

## Notes for whoever hits these next

- **mediapipe 1.0.1 is broken on Apple Silicon.** `TensorsToDetectionsCalculator::Open`
  aborts inside `DrishtiMetalHelper` with "Service is unavailable", in both
  IMAGE and VIDEO mode, and forcing the CPU delegate does not help. 0.10.35 is
  the newest working release, hence the `<1.0` pin.
- **Datasets store raw landmarks, never features.** Features are a guess about
  what the model needs and that guess will change. Storing them would mean
  every improvement to `features.py` threw away all the recording work.
- **macOS needs Accessibility permission** for `--live` (System Settings,
  Privacy and Security, Accessibility). pynput fails silently without it.

## Not done yet

- Dynamic gestures (swipe, snap, pinch-drag) need the sequence model. The
  vocabulary is defined in `schema.py`, the model is not written.
- The vision-backbone fine-tune on a real gesture dataset, as a comparison
  against the landmark classifier. Landmarks should win on speed and probably
  on accuracy, and being able to say so with numbers is the point.
- Nothing has met a real hand yet beyond the landmarker benchmark. Every
  accuracy figure above is synthetic, and `ika record` exists to fix that.
