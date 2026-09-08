# ika

**Read movement, find the habit, and call the next move while there is still
time to use it.**

Not gesture recognition. Gesture recognition answers "what is this hand doing",
which is a solved and largely uninteresting problem. This answers a different
question: *what is this person about to do, and can I say so before it
happens.*

The difference is the whole project. Recognising a punch after it lands is
trivia. Naming the habit that produces it, a quarter of a second before it
starts, is a read.

---

## The flow

Four stages. Each one is measured, and each one turned out to constrain the
next in ways that were not obvious from the outside.

```
  video ──▶ SEE ──▶ RECOGNISE ──▶ PREDICT ──▶ WARN
            │         │             │           │
       landmarks   which move    what comes   in time
       per frame   is this,      next, from   to act on
                   committed     the mined
                   early         habit
```

| stage | what it does | state | measured |
|---|---|---|---|
| **See** | landmarks from video | working | hands **32 fps**, body **108 fps** |
| **Recognise** | which movement, committed before it ends | working | **80%** at 100 ms |
| **Predict** | what follows, from habits mined on the fly | working | **5-11x** better than chance |
| **Warn** | delivered while it is still useful | measured | **37%** of warnings land right and in time |

The last row is the product. The first three exist to serve it.

---

## The finding that shaped everything

The gap between one action ending and the next beginning is **263 ms**. That
gap is the entire budget.

A conventional gesture pipeline spends **250 ms** deciding what it just saw,
because it waits for the movement to finish and then confirms it with a dwell
timer. Which means a conventional pipeline has almost exactly zero time left
to do anything with the answer.

So the design rule for the whole system, arrived at by measurement rather than
taste:

> **Commit early and accept being faked out.** The latency cost of certainty is
> worse than the accuracy cost of speed.

Composed end to end, with recognition errors corrupting the prediction exactly
as they would in life:

| strategy | delay | recognition | right | in time | **useful** |
|---|---|---|---|---|---|
| commit at 0.50 confidence | 100 ms | 70% | 35% | 88% | 32% |
| **commit at 0.85 confidence** | **100 ms** | **78%** | **41%** | **89%** | **37%** |
| commit at 0.95 confidence | 233 ms | 97% | 56% | 57% | 33% |
| commit at 0.99 confidence | 267 ms | 99% | 58% | 48% | 28% |
| wait for the movement to finish | 400 ms | 100% | 59% | 13% | **7%** |

Being 78% right at 100 ms beats being 99% right at 267 ms. Waiting for the
movement to end, which is what every gesture system does, collapses to 7%.

And 0.85 is a genuine peak rather than a monotonic trend: commit sooner and too
many calls are wrong, wait longer and too many arrive after the punch. That
peak only appears because both halves were measured together instead of each
on its own.

---

## Two lanes

**Hands are the proving ground.** Cheap to iterate on, a camera and your own
hands, and every stage of the pipeline can be built and broken there first. It
also happens to be a usable thing in itself: `ika live` will drive a cursor and
media keys from bare hands.

**The body is the target.** 33 landmarks instead of 21, and the same pipeline
throughout. This is where the idea actually lives, because a fight, a serve or
a golf swing is a whole-body movement. The pose landmarker runs at 108 fps,
three times faster than the hand one, which leaves real budget for everything
downstream.

### Seeing a body works. Seeing its feet does not.

Measured on **400 real photographs** at ordinary webcam framing, 14 ms each:

| | |
|---|---|
| a body was found | **96%** of photographs |
| features computed cleanly | 100% of those |
| head, shoulders, elbows, wrists, hips visible | **100%** |
| knees visible | **4%** |
| ankles visible | **0%** |

Ten of the fourteen named reads are usable in every frame: guard height, elbow
extension, reach, torso lean, shoulder twist, head slip. The four that are not
are all the leg-dependent ones: knee angles, stance width, weight shift.

That is a product constraint, not a bug. **At laptop framing, footwork is
invisible**, and footwork is half of fighting. Either the camera has to see the
whole body, which means a tripod and a wider shot, or the system reads only
guard, hands and torso and says so. A test pins the split, so no read that
depends on legs can quietly be trusted.

This also justifies carrying visibility inside the feature vector rather than
filtering on it once. Pose landmarkers do not omit what they cannot see, they
guess, and an unseen ankle comes back as a confident-looking coordinate. Zeroing
it would be worse: that places the joint at the hip centre, which is a specific
wrong posture rather than an absent one.

---

## Quick start

```bash
pip install -e '.[dev]'

# the landmarkers are not in the repo; ika tells you this command if you skip it
curl -sL -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

ika train --synthetic   # a classifier from invented hands, no recording needed
ika live                # the terminal app: hands drawn in braille, live reads
ika tell                # can we find a habit and call the next move?
ika early               # how early can we commit, and what does it cost?
ika compare             # landmarks vs a fine-tuned CNN, on real photographs
pytest -q               # 218 tests, no camera, no network
```

`ika live` is a dry run unless you pass `--live`. It shows everything and sends
nothing, which is the mode to develop in: a classifier wired to a keyboard is a
program that types whatever it hallucinates.

---

## What is proven, and what is not

**Proven on real data:** the landmarkers work at the frame rates quoted, on
this laptop. The gesture classifier reaches **96.1%** across 34 classes on
3,109 real photographs from HaGRID, held out **by person** so the model cannot
score by recognising individuals.

**Proven, but synthetic:** every number in the predict-and-warn lane. Habits
are found at 11.5x lift, false ones appear in 15% of clean fighters, and the
latency budget above is real arithmetic over simulated movement. What this
establishes is the *shape* of the problem. The fighters are a crude model with
fixed action durations, no fatigue, no ring position, and no reaction to being
read.

**Not tested at all:** whether actions are recognisable from real sparring
video. Every figure in the warn lane assumes they are. That is the next thing
worth doing and the biggest remaining unknown.

---

## Findings worth keeping

### Landmarks beat a fine-tuned CNN, and not for the reason you would guess

Same 3,109 photographs, same person-grouped split, both models:

| route | params | accuracy | ms per hand |
|---|---|---|---|
| 21 landmarks → 87 features → MLP | 60,578 | **96.1%** | 0.06 |
| 128px hand crop → MobileNetV3, fine-tuned | 1,552,706 | 80.8% | 5.52 |

15 points better, 26x smaller, 92x faster. The CNN's weakest classes say why:
`one` 52%, `three3` 58%, `peace_inverted` 60%. Finger counting and orientation,
trivial from joint coordinates and hard from 128 pixels.

The honest reading is *not* "landmarks beat CNNs". MediaPipe's landmarker was
pretrained on vastly more hands than 2,300, so that route stands on an enormous
amount of transfer while the CNN learns hands from scratch. On a laptop with a
small dataset the pretrained landmarker is the right call by a wide margin.

### Feature invariance is where the hand lane was won

Feed raw landmarks to a classifier and it learns that "point" means *an index
finger in the upper left of frame, at roughly the size my hand was during
recording*. Every hand is instead re-expressed in a frame built from its own
knuckle row, the only rigid part of a hand, leaving shape independent of
position, distance and rotation. There are tests asserting each invariance.

Taken too far it destroys the thing it protects: thumbs up and thumbs down are
the same shape at two angles. Both are kept, canonical shape and world
orientation as separate features, with a test for that too.

### The state machine matters more than accuracy

A classifier has an opinion thirty times a second and is sometimes wrong. Your
hands are in shot constantly doing things that mean nothing. Four defences:
smoothing, dwell, hysteresis, and an explicit engage gesture, so reaching for a
coffee sends nothing. A test proves a flailing hand fires zero actions across
150 frames, and nothing destructive is bound to any gesture.

### Bugs that only measurement caught

- **Window duration was a feature.** Every training window was the same length,
  so the model leaned on it, and any other length at inference read as
  out-of-distribution and came back "idle" with total confidence. A swipe
  classified perfectly in training and was never once detected live.
- **The window would not answer until it had more history than a gesture
  lasts:** 13 frames of warm-up for a 14-frame swipe.
- **Per-level gains went negative** at low amplification, so the amplifier
  quietly *subtracted* real signal.
- **One habit was reported as four**, the same habit with irrelevant prefixes
  plus a diluted echo. Reporting a family of overlapping reads as separate
  findings would badly mislead whoever acted on it.
- **99% accuracy from four frames**, which was too clean to be true: every
  synthetic action began distinctively, so the model was reading the starting
  pose rather than the movement. Nothing ever lied. Adding feints dropped it to
  80% and made the confidence threshold mean something.

---

## Off by default, and why

**Swipes and snaps.** Built, trained, measured at **3.3 false firings per
minute** while catching 38% of real swipes. The cause is not the model: a fast
straight hand movement is indistinguishable from reaching for a cup.
Displacement cannot separate intent from traffic, and the fix is interaction
design, gating swipes behind a pose nobody holds by accident. Needs `--dynamic`
to enable.

---

## Notes for whoever hits these next

- **mediapipe 1.0.1 is broken on Apple Silicon.**
  `TensorsToDetectionsCalculator::Open` aborts inside `DrishtiMetalHelper` with
  "Service is unavailable", in both IMAGE and VIDEO mode, and forcing the CPU
  delegate does not help. 0.10.35 is the newest working release, hence the
  `<1.0` pin.
- **The `full` pose model finds nothing on close crops** where `lite` succeeds.
  `lite` is also three times faster, so it is the default.
- **Video mode on a single still is unreliable.** Detection on one frame at
  timestamp 0 gives inconsistent answers; use IMAGE mode for stills.
- **Datasets store raw landmarks, never features.** Features are a guess about
  what the model needs and that guess changes; storing them would mean every
  improvement to `features.py` threw away all the recording work.
- **macOS needs Accessibility permission** for `ika live --live`, and pynput
  fails silently without it.
- **A terminal app has to own a terminal.** `ika live` cannot start from a pipe
  or a tool call; `run.sh` exists to launch it with a real tty, and
  `ika live --stream` is the pipe-friendly mode.

---

## Next

1. **Point the pose lane at real video.** Replaces the largest assumption in
   the project with evidence. `ika/body.py` reads bodies at 108 fps; nothing
   yet turns those landmarks into actions.
2. **Feed early commitment back into `ika live`**, which still confirms rather
   than commits, and is therefore running the 7% strategy.
3. **Gate the swipes** behind a pose, if hand gestures stay part of the product.
4. **Record real hands** with `ika record`, four seconds per gesture, to replace
   the synthetic classifier that ships today.

## Attribution

HaGRID is by Kapitanov et al., licensed CC BY-SA 4.0. The subset used here:

```bash
curl -L -o data/hagrid/export_100.zip \
  "https://huggingface.co/datasets/GestureDetectionConnoisseurs/hagrid_subsets/resolve/main/hagrid-export_100_images.zip"
cd data/hagrid && unzip -q export_100.zip
```

Hand and pose landmarkers are Google MediaPipe releases. Everything else in
here is MIT, see LICENSE.
