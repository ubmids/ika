# ika

**Shadowbox in front of your laptop. It finds your habit, then says it out
loud a beat before you do it.**

Not "you threw a jab". You know that. The read is the thing you cannot see
yourself: *after two rights, your right hand comes down*. Once it knows that
about you, it watches for the two rights and says "right hand" while there is
still time to keep it up.

```bash
pip install -e .
ika drill            # stand back, hold your guard for three seconds, then box
ika history          # every habit, session by session, and whether it is fading
```

The first `ika drill` fetches the two pose models it needs (15 MB, Google
MediaPipe). `ika setup` does the same and also checks the camera. Nothing else
is needed: no account, no network after that, no recording of your own.

Stand back until the camera sees you from head to hips with both hands in
shot, about two metres from a laptop. This is a real round, replayed from a
public follow-along workout (`ika drill round.mp4 --silent`), unedited:

```
  round: 180s at 45 fps
  169 punches (56 a minute), guard dropped 33 times
  this session:
    after punch_right: guard_down_left 65% of the time (normally 6%), 10.4x, seen 15/23
    after punch_right: guard_down_right 52% of the time (normally 6%), 8.3x, seen 12/23
    after punch_right then punch_left: punch_right 86% of the time (normally 45%), 1.9x, seen 51/59
    after punch_right: punch_left 79% of the time (normally 50%), 1.6x, seen 59/75
    after punch_left: punch_right 70% of the time (normally 45%), 1.6x, seen 59/84
  called it 27 times; the lapse followed 6 (22%), 0.8s before the hand went down
```

The first two lines are the read: when this person finishes on a right, both
hands sit lower for the next second than after anything else they throw.
Checked by eye it is a lean, not an obvious drop, so treat it as the kind of
thing ika says rather than as proof it is right. The last three are patterns,
true and mostly just how people box. During the round, once a guard habit is
known, the call is spoken as the setup happens; "the lapse followed 22%" is
scored against the dropped-guard event detector, which is the weak sensor
described below, so it undercounts.

`--silent` prints the call instead of speaking it, `--quiet` drops the
per-event lines, and `ika drill some_round.mp4` replays a recorded round
instead of the camera, which is how everything below was measured.

---

## What it can honestly do

Everything in this table was measured on real people, on footage none of the
code was tuned on, except where it says simulated.

| | measured | against |
|---|---|---|
| a body is found | **95 to 100%** of frames, all 15 clips | |
| punches caught, on people the model never saw | **47%**, with **5.6** false punches a minute | the labellers themselves: 90% and 4.9 |
| the same, front-facing people only | **51%**, **3.8** false a minute | |
| the original palm-on-lens reader, standing back | **0%** | |
| a phantom guard habit on shuffled real footage | **3%** of shuffles | should be under 5% |
| a strong guard habit named, with sensors as measured (simulated) | **40%** of fighters by 12 six-minute sessions | 93% in one 12-minute session with perfect sensors |
| a fighter with no habit told they have one (simulated) | **2%** after 12 sessions | |
| the call, when the habit is real (simulated) | lands **0.8 s** before the hand drops, right as often as the habit is | |

So: the loop works end to end, from a camera to a spoken call, and it rarely
makes things up. It is also slow to be sure of anything, because the punch
reader misses about half of what is thrown, and a two-punch setup needs both
punches caught. For a strong habit that means a couple of weeks of short daily
rounds, and even then only for some people. A subtle one mostly does not show
up at all.

That is the finished state, measured and stated rather than tuned to look
better. [`DONE.md`](DONE.md) has the criteria it was held to and where each
one landed.

### How it was measured without anyone standing in front of a camera

Public follow-along shadowboxing workouts: 15 clips, 14 people (13 with punches both
labellers agreed on), three minutes each, fetched by `scripts/shadow_fetch.py` and never committed.
Most face the camera; a few are side-on, one circles the fighter, and two are
Muay Thai. `bench/shadow/clips.json` lists them.

Punches were labelled from contact sheets, 3 seconds of video per sheet at 10
frames a second, cropped to the fighter, one tile per punch at full extension.
**The labellers are not people.** Each clip was labelled twice, by two
separate instances of a vision-language model (Claude) working from the same
written instructions and never seeing each other's labels. That is what made
labelling twenty minutes of footage twice possible, and it is why agreement
is measured rather than assumed. On whether a punch happened they agree 81 to
100% per clip, except 55% on a Muay Thai elbow drill. On which arm threw it
they agree far less, 58% on a fighter seen side-on, so arm is scored
separately and never counts against catching the punch.

A punch counts as the truth when both labellers marked it within 0.2 s of
each other. A detection is a hit if it lands within 0.2 s of one, false if
neither labeller saw anything there, and neither if only one did. The
labellers scored against each other set the ceiling in the table: 90% of the
other's punches, with 4.9 a minute the other did not mark.

The punch model is scored leaving one person out at a time: fitted on
everyone else, thresholded on everyone else, scored on the one it never saw.
Two clips turned out to be the same coach in two settings, so they are one
person and are always held out together.

```bash
python scripts/shadow_fetch.py        # the footage, then landmarks cached from it
python scripts/shadow_score.py        # labeller agreement, the ceiling, the unfitted readers
python scripts/train_strike.py --quick   # the fitted reader, held out by person
python scripts/shadow_guard.py        # dropped guards, as events, against labels
python scripts/sag_check.py           # the guard read on real footage, and its shuffled null
python scripts/robustness.py          # does the habit read survive these sensors?
python scripts/cue_check.py           # the call: how early, how often right
```

---

## What the real footage changed

Before this, every number in the predict-and-warn lane had been measured on
movement generated with arithmetic. Several things broke on contact with real
people, and each one changed the design.

### A palm closing on the lens is the wrong physics standing back

The drill read a punch as a palm growing in the image on its way to the lens.
That is right at arm's length from a laptop and wrong for shadowboxing, which
nobody does 60 cm from a screen. Two metres back, a jab that travels half a
metre grows the palm by a third, under the firing threshold, from a hand
twenty pixels across that the hand landmarker found in 14 to 92% of frames.
Replayed over the labelled footage it caught **1 punch in 1,448**.

The pose, meanwhile, was there in 95 to 100% of frames. So standing back, a
punch is read from the arm (`ika/strike.py`): how far the wrist and elbow have
left this arm's own guard, how fast and in which direction, with MediaPipe's
inferred depth standing in for the straight thrown at the camera that barely
moves in the image. A hand-built rule on that ("the wrist left its guard by a
third of a torso") caught 60% but fired 48 false punches a minute, because a
body rotating into one arm's punch swings the other arm too, and every roll,
bob and guard reset moves a wrist a long way. A small network over the same
features, fitted on the labels, gets to the table's 47% at 5.6.

`ika drill --close` keeps the palm reader for sitting at arm's length.

### More labels helped, and then stopped helping

Held out, front-facing people, `full` pose model:

| labelled | people | recall | false a minute |
|---|---|---|---|
| 7 minutes | 7 | 49% | 6.3 |
| 13 minutes | 12 | 59% | 5.6 |
| 19 minutes | 13 | 55% | 4.5 |

More people was worth about ten points. More minutes of the same people was
worth nothing. Neither a longer time window, a wider network, an ensemble,
normalising each person against their own movement, nor MediaPipe's `heavy`
pose model moved it either (heavy: 45% against 49% on the same three people,
at three to four times the cost). `full` beat `lite` by 5 to 9 points and is
what the drill uses. What limits the reader now is what the landmarks can
show of a fist coming at the camera from two metres, at the 360p these
workouts were published at.

### A dropped guard is not an event

The first design counted a dropped guard as an event: a wrist crossing a
threshold and staying there. On real footage it is the weakest thing in the
loop. The two labellers agreed on only about half of each other's drops. The
detector found 30% of the agreed ones at 4.5 false a minute, and no threshold,
reference point or hold time did better. Put those error rates into a
simulation and a habit planted 80% of the time was named in 12% of fighters
after twelve minutes, against 97% within three with a perfect sensor.

So the guard is now read the way a coach would watch it (`ika/sag.py`): not
"was there a drop", but "how high was that hand, on average, in the second
after this combo, compared with the second after your other combos". Per
frame, a wrist in a labelled drop sits 1.16 standard deviations below one in
guard; averaged over a second, and then over every time the setup comes up,
most of the noise that ruined the event detector cancels. Guard height is the
wrist's height above the nose in torso lengths, against the 75th percentile
of this person's last twenty seconds, since between punches hands are mostly
up. Frames where that arm is itself punching are left out, or every one-two
would look like a sagging guard.

A setup lives inside one combo. Matching "two rights" across the gap between
combos made calls fire a combo early and then blocked the right one; bounded
by the combo, a call is right as often as the habit is.

Across the people on the bench, the read reports a guard habit for one, the
round shown at the top. By eye that one is a lean rather than an obvious
drop. An earlier version of the punch reader surfaced a different person
whose habit was plainly visible (after a left, both hands came down into the
squat-and-jump of that workout); the final reader catches fewer of her
combo endings and no longer clears the bar. Why has not been checked;
the likeliest reason is the reader's recall, but that is a guess.

### Three statistical bugs, each found by a null that should have been clean

- **Rediscovering the state machine.** A clean session reported "after
  guard_down_both: guard_up_both, 100% of the time, 7.8x". A guard that is down
  can only come up. Restorations are recorded and never mined.
- **Filtering on effect size before correcting for multiple tests.** The
  correction only ever saw the candidates that already looked strong, so it
  never paid for having looked at the rest. 25% of shuffled real sessions
  reported a habit. Everything tested now goes into the correction.
- **Treating an estimated base rate as known.** Even corrected, a binomial
  against the other combos' rate let 14% through, because that rate is itself
  estimated from twenty or thirty combos. The guard read now uses a rank test
  on the heights themselves: calibrated (5.1% under a simulated null, 3% on
  shuffled real footage), and it keeps the magnitude that counting "down or
  not" threw away, which raised the planted-habit hit rate from 73% to 93% at
  twelve minutes with perfect sensors.

### What each sensor costs the read

Simulated at the measured rates, a strong habit (80%), six-minute sessions:

| punches | guard | named after 12 minutes |
|---|---|---|
| perfect | perfect, as events | 100% |
| as measured | perfect, as events | 98% |
| perfect | as measured, as events | 45% |
| as measured | as measured, as events | 12% |
| perfect | as measured, as sag | 93% |
| as measured | as measured, as sag | 23% |

The guard read was the first bottleneck and is no longer the worst one. The
punch reader is: a two-punch setup survives both punches being caught about a
fifth of the time. That is the one number that would change what ika can do,
and the bench to measure it on is in `bench/shadow`.

### Limits, plainly

- **Framing.** It needs head to hips in shot. At laptop distance legs are
  invisible, so footwork, level changes and weight shift are not read at all.
- **Resolution.** The bench is 360p YouTube video; a 720p webcam at two
  metres gives the pose more to work with, but that has not been measured.
- **Side-on fighters** read worse, and which arm threw a punch is unreliable
  whenever the shoulders overlap.
- **The labels are a model's labels.** Two instances agreeing is evidence, not
  proof, and they may share blind spots two people would not.
- **The habit and the call were measured in simulation**, at the error rates
  measured on real footage, because nobody on the bench comes with a known
  habit. The first real test of the whole promise is someone who knows their
  own habit standing in front of it.

---

# The research that led here

Everything below is how the project got to the drill: reading hands, reading
bodies, predicting from habits, and a two-fighter lane on real fight footage.
It is kept as it was written, as findings. The glasses, fight-footage
analysis, cursor control and swipe lanes are frozen: working where they work,
not being pushed further.

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

`ika watch <clip.mp4>` is the SEE stage end to end: real timestamps, camera
motion removed, two fighters held apart by identity, posture per frame.

```
  60 frames, timed by container
  bodies per frame: 2 in 60
  identities: (1, 2) held for 60/60 frames
  camera compensated on 59/60 frames
```

The order inside it is not arbitrary. **Pose runs before stabilisation**,
which looks backwards until you see why: camera motion is estimated by finding
the largest group of points that moved together, and a fighter filling the
frame *is* that group, so the estimate becomes the fighter reported as the
camera. Measured with a subject over 40% of the frame, the answer came back
-0.0625 against a truth of +0.0156, confidently wrong. Masking the subject out
recovers +0.0156, and pose is the cheapest source of that mask.

Compensation uses **accumulated** drift rather than the last frame pair,
because movement features span about 0.6 seconds and it is the drift across
that window which corrupts them. On a 60-frame pan the accumulated figure came
to 99% of truth while the per-frame median under-read at 70%, since 1.5 px per
frame is sub-pixel after the stabiliser's own downscale. The errors average
out rather than accumulating.

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

### Reading a body early: punches yes, level changes no

The whole pipeline, run on body features instead of abstract action labels.
Movements are generated as postures changing over time with a bell velocity
profile, feints included, and the classifier commits before the movement ends.

At 0.85 confidence, **100 ms**, having seen about half the movement:

| action | laptop webcam (legs hidden) | tripod (legs visible) |
|---|---|---|
| raise guard | **100%** | 100% |
| cross, right | 98% | 100% |
| slip left | 97% | 83% |
| jab, left | 94% | 94% |
| drop guard | 92% | 92% |
| idle | 72% | 98% |
| **crouch** | **59%** | 88% |
| overall | **89%** | 93% |

Everything above the waist survives an occluded camera. `crouch` collapses,
because its only evidence is knee angle, and it drags `idle` down with it since
the two become confusable. **So at laptop framing, read punches and guard and
say nothing about level changes.** With a tripod you get all of it.

### Two fixtures that lied before this number was honest

Both scored around 86% on `crouch` with the legs hidden, which is an action a
webcam physically cannot observe. Neither was caught by a test; both were
caught by disbelieving a suspiciously good result.

**The spine got shorter.** `body_pose` shortened the torso by up to 18% when
crouching. Since `posture` normalises everything by torso length, that made
every limb read as proportionally longer and handed the classifier a crouch cue
that existed only in the fixture. Real crouching bends knees and hips; it does
not compress your spine.

**Occlusion was only a flag.** The second version lowered the visibility
number but still passed the true leg coordinates, so the model read knees it
had been told were invisible. A landmarker does not withhold an unseen joint,
it extrapolates a plausible wrong one, so that is what is simulated now. The
knee signal separating crouch from idle went from **1577 sigma** with legs
visible to **0.1 sigma** hidden, which is what "cannot see" should look like.

## The first drill loop, before real footage

This is the drill as first built, on synthetic bodies and the palm-on-lens
punch reader. Real footage replaced both halves (see the top of this README);
it is kept because the flaw it exposed still shapes the miner.

Stand where the camera sees your head, shoulders and hands. Hold still for
three seconds while it learns your resting guard. Then drill. It reports the
habit, not the punches.

**Why this works without a trained action classifier**, which is the thing
that unblocked it. Every route up to here needed a model trained on labelled
movement in laptop framing, and no such dataset exists publicly. But the two
events a drill coach actually needs are *geometric*:

- a **punch** is a hand closing on the lens, measured from apparent palm size
- a **dropped guard** is a wrist below its own learned baseline

Neither needs a model trained on anyone's hands. So the loop closes now, and
recording your own gestures became an optional refinement rather than a
prerequisite.

Verified end to end on a synthetic subject with a habit planted at 75%:

```
after punch_right then punch_right: guard_down_both
79% of the time (normally 21%), 3.7x, seen 23/29
```

That is the read. Not "you threw a jab", which you already know, but "you drop
your hands after two punches", which you cannot see yourself.

Two things are deliberately learned rather than assumed. **Guard thresholds
are calibrated per person**, because the +0.14 resting and -0.84 dropped
figures were measured on synthetic bodies and real people are built
differently and sit at different distances. And **a punch is never reported
without a body in view**: measured, a lean-in with no torso visible is
indistinguishable from a strike, crossover at 1.25/s, and telling someone they
punched when they leaned is how they stop trusting it.

Habits accumulate into a profile on disk, and fade when they stop happening.

### The flaw the first end-to-end run exposed

It reported seven habits where one was planted, and four were phantoms. Both
hands drop together, so `guard_down_left` and `guard_down_right` landed a frame
apart and the miner announced *"after guard_down_left: guard_down_right, 100%
of the time, 6.7x"* as a discovery. It was not a discovery, it was the order
the two events happened to be emitted in. **Sequential mining over
simultaneous events invents dependencies.** Both sides dropping is now one
`guard_down_both`, merged in the stream rather than within a frame, since the
two never cross the threshold on the same frame.

Three findings still survive that are true but useless, of the form "you raise
your guard after dropping it". Those are state machine necessities rather than
habits. They are now recorded and never mined, so they cannot be reported.

---

## Research commands

The commands behind the findings below. The drill itself only needs
`ika setup`; these fetch or train what each experiment used.

```bash
pip install -e '.[dev]'

# the landmarkers are not in the repo; ika tells you this command if you skip it
curl -sL -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

python scripts/train_from_hagrid.py   # a classifier from 2,192 real people
ika train --synthetic   # or from invented hands, which is much worse (below)
ika drill               # watch a drill session and report the habit
ika live                # the terminal app: hands drawn in braille, live reads
ika tell                # can we find a habit and call the next move?
ika early               # how early can we commit, and what does it cost?
ika compare             # landmarks vs a fine-tuned CNN, on real photographs
ika body                # can a camera read a body early, and what does it miss?
ika watch clip.mp4      # a whole clip: two fighters, camera motion removed
python scripts/ut_interaction.py --fetch   # real footage, then measure on it
pytest -q               # the whole suite, no camera, no network
```

`ika live` is a dry run unless you pass `--live`. It shows everything and sends
nothing, which is the mode to develop in: a classifier wired to a keyboard is a
program that types whatever it hallucinates.

`--body` adds a posture lane: guard height, elbow extension, reach, torso
lean and shoulder twist, drawn live beside the hands. Only reads that survive
webcam framing are offered, since knees are visible in 4% of frames from a
laptop and ankles in none, and a test enforces that nothing leg-derived
appears. Any read whose joints the landmarker was guessing at is greyed out
rather than shown as a fact.

Cost measured before wiring it in: hands and gesture classification take
22.1 ms, pose and posture extraction 9.5 ms, both together 31.6 ms for 32 fps.
That is 12% of the 263 ms budget, so it is cheap enough to leave running.

It **commits early rather than waiting for a gesture to be held**: 7 frames
against 11, 233 ms against 367 ms, measured on identical streams. Two layers,
because arming and acting want opposite things. Engagement keeps the dwell
timer, since the machine should not go live because a hand passed through an
open palm and 250 ms of deliberation is a feature there. Firing does not.
`--confirm` restores the old behaviour.

The split is safe in this lane for a specific reason. Committing early is
fooled by a movement that lies about its opening, which against a fighter who
feints is a real trade. Someone gesturing at their own computer is not feinting
at themselves, so the accuracy penalty nearly vanishes here while the speed is
kept.

---

## Nobody needs to record their own hands

The shipped classifier used to be trained on hands generated with arithmetic.
Judged on real held-out people, that model managed **46.4%** across seven
classes. The same architecture trained on HaGRID photographs of **2,192
different people**, split so no person appears on both sides, reaches **98.8%**
on people it has never seen.

| trained on | accuracy on real held-out people |
|---|---|
| synthetic hands | 46.4% |
| **2,192 real people** | **98.8%** |

So `scripts/train_from_hagrid.py` is now how the live classifier is built, and
recording your own hands is an optional refinement rather than a prerequisite.
It was the last thing on the list that genuinely needed a person.

HaGRID names seven of our nine gestures. The two it misses are the two the
classifier does not need: `l_shape` is `point` plus an extended thumb and the
pair confuses in both directions, which was already a known vocabulary flaw,
and `pinch` is measured geometrically from thumb-to-index distance rather than
classified. `l_shape`'s binding to "previous desktop" is therefore unreachable
with this model.

## Real footage, at last: the answer is yes, with a catch

Everything in the recognise-predict-warn lane had been measured on movement
generated with arithmetic. This is the first time any of it met video of
people. **UT-Interaction**: 120 clips, six classes, two complete bodies on a
static camera, `scripts/ut_interaction.py` fetches and measures it.

**Pose sees them.** A body was found in **99%** of frames, and all 120 clips
were usable, every class between 98 and 100%. That was the go/no-go and it
passed cleanly.

**Actions are recognisable, well above chance.** 5-fold, split by clip so no
clip appears on both sides:

| class | recall |
|---|---|
| hand shake | **90%** |
| hug | 85% |
| point | 70% |
| punch | 55% |
| push | 40% |
| **kick** | **35%** |
| overall | **62.5%** against 17% chance |

**The catch is which classes fail.** Gross posture works at 82%. The strikes,
which are the ones the product needs, sit at **43%**. And that is not a feature
design problem: describing the whole clip, describing only the peak-activity
window, and describing both together all land within four points of each other,
while shuffling which classes suffer.

| features | overall | strikes | gross posture |
|---|---|---|---|
| whole clip averaged | 62.5% | 43% | 82% |
| peak-activity window | 59.2% | 53% | 65% |
| both together | 62.5% | 48% | 77% |

Localising the strike helps strikes and hurts postures, which makes sense: a
punch is a brief event an average dilutes, and a hug is a sustained posture a
short window truncates. But nothing moves the strikes past ~53%.

**The most likely cause is a design choice of mine, not a limit.** These are
*interactions*, and the pipeline reads one body. A punch is one person's arm
extending toward another, and taking only the largest figure throws away the
relationship, which is very likely where the signal is. Testing that means
running the pair through and building features across both, which is the next
thing worth doing.

### A strike is a relationship, not a shape

The strikes sat at 43% while gross posture reached 82%, and feature design
could not move them. The hypothesis: a punch is not a shape at all, it is one
person's wrist travelling toward another person's body, and reading a single
figure throws that away. Arm extension survives; the thing that made it a
punch rather than a stretch does not.

`ika/interaction.py` adds twenty relational features: gap and its closing
rate, each of four wrists' distance to the other person's torso and how fast
it is shrinking, whether the two overlap. Measured on the **same clips**, same
folds, so the comparison is controlled:

| features | overall | strikes | punch | kick | push |
|---|---|---|---|---|---|
| one body only | 54.9% | 42% | 21% | 58% | 47% |
| **the relationship** | **82.5%** | **82%** | **84%** | **79%** | **84%** |
| both together | 73.4% | 63% | 53% | 68% | 68% |

**Punch recall quadrupled, 21% to 84%.** The hypothesis was right, and by a
wider margin than expected.

Two honest caveats. Combining both feature sets is *worse* than the
relationship alone, because seventy features across ninety-eight clips is
data-starved rather than richer. And this subset is effectively five classes,
not six: `point` survives in only one clip because two bodies are **never**
found in those clips, so its 0% is the pipeline failing to see a relationship
rather than a classifier failing to learn one. For the same reason the 82.5%
is not comparable with the 62.5% measured over all 120 clips.

That failure is itself informative. Two people are tracked in 96% of
hand-shake frames and 82% of punch frames, but 48% of hug frames and 0% of
point frames. The pipeline sees a relationship when the pair are close and
distinct, and loses one when they clinch or stand apart.

**And this carries to the glasses.** First person only ever shows one body,
the opponent, so the relational quantity has to be recovered in a different
form: wrist approaching the *lens* rather than wrist approaching another
torso. If the signal is relational, and it clearly is, that is the shape the
first-person version must reconstruct.

### The framing inversion

The named reads flip completely between the two camera setups measured, and
neither gives everything:

| read | webcam framing | wide static shot |
|---|---|---|
| shoulders, hips, torso lean | 100% | 100% |
| arms, guard height, reach | **100%** | **54-67%** |
| knees, stance, weight | **0-4%** | **85-95%** |

A laptop camera sees arms and no legs. A wide shot at ~200 px person height
sees legs, and resolves arms far less reliably. So the camera decides what you
can read, and at these resolutions there is no single framing that reads a
whole fighter. That directly explains the result above: the classes that
survive are the ones carried by the torso.

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

**Now tested:** actions are recognisable from real video at 62.5% over six
classes against 17% chance, and pose finds a body in 99% of frames. The
strikes specifically are weak at 43%, most likely because the pipeline reads
one body where the footage shows an interaction between two.

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

## The swipe gate: safe now, still insensitive

Swipes fired **3 times a minute at an idle hand** while catching half of real
ones. Not a training problem: a fast straight hand movement is what reaching
for a cup looks like, so displacement cannot separate intent from traffic
however good the model gets.

The fix is interaction design. A swipe only counts if the hand held a
deliberate pose while it travelled, which is the engage gesture's trick one
level down. The whole thing turns on **how rare that pose is**:

| gate pose | share of idle | idle firings/min | swipes caught |
|---|---|---|---|
| none | n/a | **3.00** | 50% |
| open_palm + point | 35% | 1.17 | 42% |
| **pinch** | 3.3% | **0.00** | 39% |
| thumbs_up | 0.3% | **0.00** | 33% |

My first gate was `open_palm` and `point`, which together are 35% of what an
idle hand reads as, and it only halved the problem. A gate the idle state
already satisfies is not a gate. `pinch` is 3.3% and removes idle firings
entirely, and it reads naturally, since pinching while moving is grab and drag
and pinch is already the drag gesture.

So the lane went from dangerous to safe, and stays **off by default** for the
other reason: 39% detection means you swipe and nothing happens six times in
ten. That is sensitivity in the movement classifier, not the gate. `--dynamic`
enables it.

One correction while measuring this. The earlier "3.3 false firings" figure
conflated two errors: 3.00 idle firings, which the gate fixes, and 0.33 real
swipes named as the wrong gesture, which it cannot.

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

## Frozen

These lanes work where they work and are not being pushed further:

- **Glasses**: reading movement from a head-worn camera.
- **Fight-footage analysis**: `ika watch`, two fighters from broadcast video.
- **Cursor control**: `ika live --live`, hands driving the computer.
- **Swipes**: gated behind a pose, safe, still insensitive.

## Attribution

HaGRID is by Kapitanov et al., licensed CC BY-SA 4.0. The subset used here:

```bash
curl -L -o data/hagrid/export_100.zip \
  "https://huggingface.co/datasets/GestureDetectionConnoisseurs/hagrid_subsets/resolve/main/hagrid-export_100_images.zip"
cd data/hagrid && unzip -q export_100.zip
```

The shadowboxing bench is built from public follow-along workouts on YouTube,
by Boxing Ready, El Yuyu, FightCamp, Gabriel Varga, Jeff Chan MMAShredded, Laura Wells Fitness, NateBowerFitness, Oracle Boxing, Precision Striking, Spence Crosby, Spring Sia, Sylvia Nasser, Tony Jeffries and Well+Good. The
video is fetched locally by `scripts/shadow_fetch.py` and never committed;
`bench/shadow/clips.json` lists each clip and section. Only the labels made
from it are in the repo.

Hand and pose landmarkers are Google MediaPipe releases. Everything else in
here is MIT, see LICENSE.
