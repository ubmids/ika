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
ika compare                    # landmarks vs a fine-tuned backbone, on real data
ika train-dynamic              # the movement classifier (swipes, snap)
ika tell                       # can we find a habit and call the next move?
ika bindings                   # what each gesture does
pytest -q                      # 125 tests, no camera, no network
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

## Landmarks or pixels? Measured, on real hands

The question the project was really for. One route keeps 21 points and throws
the image away; the other keeps the pixels and throws the points away. Both are
used in real products.

Run on **HaGRID**, 3,109 real photographs across 34 gesture classes, with the
identical examples and the identical split given to both models:

| route | params | val accuracy | ms per hand |
|---|---|---|---|
| 21 landmarks → 87 features → MLP | 60,578 | **96.1%** | 0.06 |
| 128px hand crop → MobileNetV3-Small, fine-tuned | 1,552,706 | 80.8% | 5.52 |

The landmark route wins by 15 points, with 26x fewer parameters, 92x faster.
The backbone's weakest classes say why: `one` 52%, `three3` 58%,
`peace_inverted` 60%, `thumb_index` 61%. Those are finger-count and orientation
distinctions, which are trivial once you have joint coordinates and genuinely
hard from 128 pixels.

**The honest reading is not "landmarks beat CNNs".** It is that MediaPipe's
landmarker was pretrained on vastly more hands than 2,300, so the landmark
route is quietly standing on an enormous amount of transfer while the backbone
gets 2,300 images to learn hands from scratch. Given HaGRID's full 700 GB the
gap would close and might reverse. On a laptop with a small dataset, the
pretrained landmarker is doing the heavy lifting and it is the right choice by
a wide margin.

**Split by person, not by image.** HaGRID carries a `user_id`, and the same
person appears in many photos. A random split puts the same hands on both
sides, so the model is rewarded for recognising people and the accuracy is
inflated. Every number above holds 548 of 2,192 users entirely out of training.
A test asserts zero user overlap.

## The dynamic lane does not work yet, and is off by default

Swipes, snap and pinch-drag are built, trained and measured, and the honest
result is that they are not usable. Measured end to end on a continuous stream
of idle hand with real swipes injected every ten seconds:

| threshold | dwell | false firings per minute | real swipes caught |
|---|---|---|---|
| 0.70 | 2 | 13.9 | 35% |
| 0.85 | 3 | 3.3 | 38% |
| 0.95 | 3 | 0.8 | 25% |

Firing three times a minute at nothing while missing two thirds of what you
meant is worse than having no swipes at all, so `ika live` leaves this off
unless you pass `--dynamic`.

**The cause is not the model, and more training will not fix it.** A swipe is
"the hand moved fast in a straight line", and an idle hand does that all the
time: reaching for a cup, waving while thinking, scratching an ear.
Displacement cannot separate intent from traffic. The fix is interaction
design, gating swipes behind a pose nobody holds by accident, which is the same
trick the engage gesture already uses for static poses.

Two real bugs surfaced on the way, both worth remembering:

- **Window duration and frame count were motion features.** Every training
  window was the same length, so the model leaned on them; a shorter window at
  inference read as out-of-distribution and returned `none` with total
  confidence. A swipe classified perfectly in training and was never once
  detected live. The features are now named constants, because removing one
  silently shifted every index after it while the tests kept passing.
- **The window would not answer until it had more history than a gesture
  lasts.** The readiness bar was 60% of a 0.7s window, so 13 frames, and a
  swipe is about 14 frames. It agreed to look at the movement for its final
  frame or two.

## Does a sequence model earn its keep?

A GRU over raw per-frame sequences, against those 16 hand-picked motion
features, three seeds each:

| examples per class | features + MLP | GRU |
|---|---|---|
| 8 | **100.0%** | 97.6% |
| 20 | 100.0% | 100.0% |
| 60 | 100.0% | 100.0% |
| 200 | 99.7% | 99.8% |

It never wins, and it loses when data is scarce, so the shipped classifier is
the MLP at a quarter of the parameters. The GRU stays in the tree because real
swipes will be messier than generated ones and a sequence model has more
headroom if that mess carries signal, but the default follows the evidence that
exists rather than the evidence one might wish for.

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

## Reading the tell: can we call the next move in time?

`ika tell` runs the experiment that decides whether the interesting product is
possible. The hand lane answers "what is this doing". This answers "what will
it do next", which is a different problem.

Movement becomes a stream of actions, the stream is mined for conditional
habits, and every habit is then judged on the only measure that matters: how
long before the move lands could we have said it.

**Finding the habit works.** Planting "after jab, jab: drop guard, 70% of the
time" in a synthetic fighter and mining 3,000 actions recovers it at 73%, 11.5x
above the base rate.

**It mostly does not invent them.** Over 30 fighters with no habits at all, a
false find turns up in 15% of them, 0.20 per fighter. That number is the real
headline, because a miner that reports tells someone does not have is worse
than no miner: a fighter will act on it. Two defences get it there. A binomial
tail per candidate, computed in log space since the naive form overflows at
these counts, and a Benjamini-Hochberg correction for having tested thousands
of contexts. Bonferroni would be far too strict, throwing away real habits to
avoid a single mistake; what you actually want is "most of what you tell me is
true", which is the false-discovery rate.

**One habit must produce one finding.** A single planted habit generates a
family of overlapping candidates: the same habit with irrelevant prefixes
("after cross, jab, jab") and diluted shorter versions ("after jab", at a third
of the strength). The first cut reported four findings for one habit. Both
directions are junk for different reasons and each gets its own test: a longer
context survives only if it beats its own suffix by more than chance, and a
shorter one is dropped when a longer kept context ending with it predicts the
same outcome far more strongly.

**How much history you need:**

| actions observed | habit found |
|---|---|
| 500 | 78% to 95% |
| 1,500 | 100% |
| 3,000 | 98% to 100% |

At roughly 1.5 s per action, 1,500 actions is about half an hour of continuous
action. So this reads a fighter across a bout or from footage of earlier ones,
not from a standing start in round one.

### The verdict, and it is not the one I expected

Replayed against unseen action from the same fighter:

| recognition delay | right | in time | useful | median lead |
|---|---|---|---|---|
| 0 ms | 58% | 100% | 58% | 281 ms |
| 100 ms | 58% | 88% | 51% | 181 ms |
| **250 ms** | 58% | 62% | **39%** | **31 ms** |
| 400 ms | 58% | 12% | 6% | -119 ms |

The prediction is good: 58% precision against base rates of 6% to 11%, so the
reads are five to eleven times better than chance, and the accuracy does not
change with delay because the prediction is the same either way.

**The bottleneck is recognition latency, and it is brutal.** The median gap
between one action ending and the next beginning is 263 ms, and the 10th
percentile is 93 ms. That gap is the entire window. Our own gesture pipeline
takes 250 ms to recognise something, which eats almost all of it, and at 400 ms
the median warning arrives *after* the punch.

So the whole engineering problem is getting recognition under about 100 ms, and
the obvious route is not a faster model. It is refusing to wait for the action
to finish: commit early from the first third of a movement instead of
confirming a completed one. The 250 ms figure is mostly dwell and smoothing,
which exist to confirm gestures, and confirmation is exactly what prediction
cannot afford.

Every figure above is synthetic and the fighters are a crude model of a real
one. What the experiment establishes is the *shape* of the problem: habits are
findable, prediction beats chance comfortably, and latency is where this lives
or dies.

## Not done yet

- **Nothing has met *your* hands yet.** The 96.1% is on HaGRID's people, and
  every latency and false-firing figure is measured, but the classifier that
  ships was trained on synthetic poses. `ika record` is the fix and takes about
  five minutes.
- **Gate the swipes behind a pose** so the dynamic lane becomes usable. That is
  a design change, not a training run.
- **Accessibility permission** is the only thing `--live` waits on, and it is
  yours to grant: System Settings, Privacy and Security, Accessibility.

## Attribution

HaGRID is by Kapitanov et al., licensed CC BY-SA 4.0. The subset used here is
the `hagrid_subsets` export, 34 classes at 100 images each with bounding boxes
and 21-point landmarks:

```bash
curl -L -o data/hagrid/export_100.zip \
  "https://huggingface.co/datasets/GestureDetectionConnoisseurs/hagrid_subsets/resolve/main/hagrid-export_100_images.zip"
cd data/hagrid && unzip -q export_100.zip
```
