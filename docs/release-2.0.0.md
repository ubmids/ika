# ika 2.0.0

ika is a terminal project: shadowbox in front of a laptop camera, and it
finds the habit you cannot see yourself, like your left hand sagging after
you finish on a right, then says it out loud as the setup happens.

## What's new

**`ika coach`.** ika calls numbered combos out loud, like pad work, and
reads your guard in the second after each one. Because it called the
combo, it knows the setup instead of reading it off a punch reader that
catches about half of what is thrown. Once a combo is a known habit, its
call ends with a reminder: "one, two... right hand up". This is the mode to
start with.

**A screen you can read from two metres away.** In a terminal, `ika coach`
and `ika drill` now draw the round: a big round clock, the call in large
type, a stick figure of what the camera sees, a guard meter per hand, the
punches it caught, and whether you are boxing or resting. A bell starts
and ends each round (`--no-bell` to turn it off). Pipes, `--plain`,
`--quiet` and replays with `--silent` still get the plain transcript.

**The guard is read from the shoulders**, not the nose, in both modes.
The head moves on its own; the shoulders separate real drops from guard
better after a combo (0.93 against 0.86) and planted ones too (0.68
against 0.64).

**A habit planted in real footage.** Sags cut from real labelled drops,
added to other people's real rounds, and read with the shipped pipeline.
This is the closest thing to a real person with a known habit the bench
has, and it is where the drill's numbers below come from.

**`ika history`** keeps coach and drill apart, draws each habit as a
sparkline across sessions, and labels it holding, fading, new or not yet,
straight from the profile's own pooled and decayed decision.

**Install** is one line, on Python 3.10 to 3.14, and `ika --version`
exists.

## The numbers, as measured

Coach mode, simulated at the punch reader's measured error rates, with
guard heights drawn from the planted rounds:

| | |
|---|---|
| a strong (80%) habit named in one 12-minute session | 60% (34% with the guard read from the nose) |
| the same, 24 minutes | 88% |
| named across 12 six-minute sessions, pooled | 69% |
| a 50% habit, 12 minutes | 16% |
| a fighter with no habit told one, 12 minutes | 5.7% of 1,000 |
| the same, a fighter slower than the read expects | 5.1% of 1,000 |
| the same, after 12 pooled sessions | 4.3% of 300 |
| a habit reported on random schedules over real footage | 1% of 600 |

Part of the jump from earlier coach numbers is ika getting better (the
shoulder read: 34% to 60% on the same heights) and part is the simulation
getting more honest. Treat 60% as an upper bound: nobody has yet followed
ika's calls in front of a camera.

What did not improve:

| | |
|---|---|
| punches caught, on people the model never saw | 47%, with 5.6 false a minute (unchanged) |
| a habit planted in real rounds, read through the drill's punch reader | named in 0% by 12 minutes, from the nose or the shoulders; the simulation said 42% |
| the same, with the labelled punches in place of the reader | 8% by 12 minutes (6% from the nose) |
| the call, on those planted rounds | 0.9 s before the drop, but only 12% of calls followed by one |
| a combo called often enough on the bench to test coach on real calls | none: no combo is called more than 6 times in a clip, the read needs 10 |

An effect-size bar measured in place of the lift bar did no better under
every null, so the lift bar stays.

## Install

```bash
pipx install git+https://github.com/ubmids/ika    # or: uv tool install git+https://github.com/ubmids/ika
ika setup      # fetches the two pose models (15 MB) and checks the camera
ika coach
```

Python 3.10 to 3.14 on Apple Silicon Macs, x86-64 Linux and Windows, where
MediaPipe 0.10 publishes wheels. Python 3.14.1 exactly is refused by
torchvision. Models and history live in `~/.ika` (`IKA_HOME` moves them).
Upgrading from 1.0.0 in a clone keeps your profile where it was, and a
1.0.0 session reads as a drill session. Moving from a clone to an
installed tool, copy `data/profiles/` to `~/.ika/data/profiles/` to keep
your history.

## Limits

- Stand back about two metres with head to hips in shot. Legs are not
  read, so footwork and level changes are invisible.
- The punch reader catches about half of what is thrown, which is why
  the drill takes weeks to name a habit, when it names one at all.
- Side-on fighters read worse, and which arm threw a punch is unreliable
  when the shoulders overlap.
- The bench labels are from a vision-language model labelling twice, not
  from people.
- Every habit number is simulated or planted. Nobody with a known habit
  of their own has tested either mode yet; that is the first real test.
