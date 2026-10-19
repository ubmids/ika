# What finished means

ika is a project, not a product, and it lives in the terminal. It is finished
when `ika drill` gives a real person a habit read that is true, and is good to
use. Everything else in the repo is research that fed this, and stays frozen.

| # | criterion | pass when | state |
|---|---|---|---|
| 1 | **Measured on real people** | punches scored against hand labels on public shadowboxing footage of several different people: recall ≥ 85%, at most 1 false punch a minute, calibration holding for every person. A miss is reported, not tuned away | **measured, missed.** 47% recall at 5.6 false a minute held out by person (51% at 3.8 front-facing), against a labeller ceiling of 90% at 4.9. Labels are from two Claude instances, not humans |
| 2 | **No phantom habits** | a session with no planted habit reports none, and state-machine necessities ("guard up after guard down") are never reported as habits | **done.** 3% of shuffled real sessions report a guard habit; a simulated clean fighter is told they have one 2% of the time after 12 sessions; restorations are recorded and never mined |
| 3 | **Calls it before it happens** | during a round, a spoken or bell cue fires when the setup that precedes the habit occurs, measured on how often it lands before the habit | **done, in simulation.** The call is right as often as the habit is and lands 0.77 s before the hand drops, at the sensor error rates measured on real footage; no real person with a known habit has tested it yet |
| 4 | **Remembers you** | an end-of-round summary, and `ika history` showing each habit's rate across sessions | **done.** End-of-round summary, session log in the profile, `ika history` |
| 5 | **One command to start** | a fresh clone runs one command that fetches the models, checks the camera and walks through calibration | **done.** `ika setup` fetches the models and checks the camera; `ika drill` fetches them itself if skipped and calibrates on a three-second guard |
| 6 | **README leads with the drill** | the drill first, research after as findings, the limits of what a camera sees stated plainly | **done.** Drill first, then what it can honestly do, then the research as findings, limits stated plainly |

Criterion 1 is the one that missed, and it is what limits everything downstream:
a two-punch setup survives both punches being caught about a fifth of the time,
so a strong habit takes a couple of weeks of short rounds to name. The bench to
improve it on is `bench/shadow`.

Frozen and not blocking: glasses, fight-footage analysis, cursor control, swipes.

# v2: coach mode

`ika coach` calls numbered combos and reads the guard after each, so the
setup is known instead of read off a punch reader that misses half of what
is thrown. Measured in simulation at the sensor rates measured on real
footage (`scripts/coach_check.py`), under the guard separation measured by
planting sags into real footage (-0.25 against -0.16, sd 0.19), with the
older, optimistic guard model (-0.34 against -0.08, sd 0.30) reported beside
it for comparison. The null is also measured on real footage.

| | pass when | state |
|---|---|---|
| **Names a habit faster than the drill** | a strong (80%) habit named more often than the drill in 12 minutes | **measured, weak.** Real-measured guard: 27% in one 12-minute session, 14% across 12 pooled six-minute sessions, and no session length reaches 50% (peak 37% at 24 minutes, 22% at 192), capped by sag's 1.35 lift bar since an 80% habit gives a lift near 1.3 at this separation. Old sim guard: 80% and 67%, against the drill's 23% and 40% on the same model. The planted test on real footage named an 80% habit in 6% of 12-minute composites with labelled punches, so the real-measured column is likely an upper bound |
| **No phantom habits** | a clean fighter, a fighter slower than the assumed pace, and real footage all report a habit about 5% of the time or less | **done.** 5 to 6% of single sessions (400 each, the rank test's level) under either guard model, 4 to 6% for the slower fighter, 2% after 12 pooled sessions, 1% of 600 random schedules over the 15 bench clips |
| **Calls it live** | a known coach habit adds its reminder to the call | **done.** "one, two... right hand up", from history and from earlier rounds of the same session |
| **Remembers you** | coach sessions in the profile, tagged, and shown by `ika history` | **done.** Rows carry a mode; history lists coach and drill sessions and habits apart; the drill never watches for a coach habit |
| **Tested on real calls** | the bench coaches' own calls recovered and the read run on them | **not possible on this bench.** 71 calls recovered from word-timed transcripts across 7 clips, answered 75% against 60% at random times, but no combo is called more than 6 times in a clip against the 10 the read needs (`scripts/coach_calls.py`) |

Two design changes came from nulls that should have been clean. A fixed
expected pace put the guard window on top of a missed last punch, more for
long combos than short ones, and told 11% of slower simulated fighters they
had a habit in 12 minutes; the pace is now learned per combo from the
session. And following up a combo one session proposed pooled that
selected session in, which told 6 to 10% of clean fighters they had a habit
after 12 sessions; a coach habit is now confirmed only by the sessions
after the one that proposed it, which record their plain counts.

What would move the real-measured number is the guard read itself, not the
setup: the separation after a habit combo is small next to its spread, and
the lift bar on "hand down" counts throws away what the rank test finds.
Loosening that bar would need its own null measured first.

Not measured: anyone following real calls from ika, how often a real person
answers (95% is assumed), and whether hearing the reminder changes the habit.
