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
