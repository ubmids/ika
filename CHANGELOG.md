# Changelog

## 2.0.0, unreleased

### Added

- `ika coach`. ika calls numbered combos out loud, like pad work, checks
  each one was answered, and reads your guard in the second after it. The
  setup is known instead of read off a punch reader that misses about half
  of what is thrown. `--rounds`, `--rest`, `--combos`, `--seed`. Once a
  combo is a known habit its call ends with a reminder: "one, two... right
  hand up".
- A full-screen view for `ika coach` and `ika drill` in a terminal: a big
  round clock, the call or cue in large type, a stick figure of the pose,
  a guard meter per hand, punch ticks per hand, and the round's phase. A
  round bell (one ring to start, three to end), off with `--no-bell` and
  never with `--silent` or a replay. `--plain`, `--quiet`, a pipe, or a
  replay with `--silent` keep the plain transcript.
- A planted-habit test on real footage (`ika/plant.py`,
  `scripts/planted_check.py`): sags cut from real labelled drops, added to
  other people's real rounds, and read with the shipped pipeline.
- `ika --version`.

### Changed

- The guard is read from the shoulders instead of the nose, in coach mode
  and the drill both, after measuring candidate signals
  (`scripts/guard_signal.py`).
- `ika history` separates coach and drill, gives each habit a sparkline
  across sessions, dates every session, and labels each habit holding,
  fading, new or not yet from the profile's own pooled, decayed decision.
  With nothing on record it says what to run.
- Python 3.10 to 3.14 (was 3.10 to 3.12). CI runs all five.
- Installed as a tool (`pipx`, `uv tool`), models and history live in
  `~/.ika` instead of inside the virtual environment. `IKA_HOME` moves
  them. A clone still keeps them in the clone.
- `ika setup` says which Python it is running on, and what to do on one
  outside the tested range.
- Sessions in the profile carry a mode, so coach and drill evidence never
  pool. Rows written by 1.0.0 read as drill.

### Measured

Simulated at the punch reader's measured error rates, with guard heights
drawn from the planted rounds:

- A strong (80%) coach habit is named in 60% of one 12-minute session (34%
  with the guard read from the nose), 88% at 24 minutes, and 69% across 12
  pooled six-minute sessions. A 50% habit: 16% in 12 minutes.
- A fighter with no habit is told one in 5.7% of 1,000 12-minute sessions,
  5.1% if slower than the read expects, 4.3% of 300 after 12 pooled
  sessions, and 1% of 600 random schedules over real bench footage.

On real footage, with a habit planted from real drops:

- Through the drill's punch reader the planted habit was named in 0% of
  composites by 12 minutes, read from the nose or the shoulders. The
  simulation had said 42%.
- With the labelled punches in place of the reader: 8% by 12 minutes (6%
  from the nose).
- The call landed 0.9 s before the drop, but only 12% of calls were
  followed by one.

Not measured: anyone following ika's calls, how often a real person
answers (95% is assumed), and whether hearing the reminder changes the
habit.

## 1.0.0, 2026-09-24

The drill, finished as `DONE.md` defines it.

- `ika drill`: stand back, hold a guard for three seconds, box. It reads
  punches from the pose, reads the guard after each combo, finds the
  setup that comes before your guard dropping, and once it knows it, says
  the hand out loud as the setup happens. `ika drill round.mp4` replays a
  recorded round.
- `ika history`: each habit's rate session by session, from a profile
  that pools sessions and lets old evidence decay with a two-week half
  life.
- `ika setup`: fetches the pose models and checks the camera.
- Measured on 15 public shadowboxing clips: 47% of punches caught with 5.6
  false a minute, on people the model never saw. A phantom guard habit on
  3 to 4% of shuffled real sessions.
- The research lanes before it (bare-hands gestures, posture, early
  reads, two-fighter footage) are kept as findings and frozen.
