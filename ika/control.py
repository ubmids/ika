"""Driving the machine, and refusing to by default.

Everything here is inert unless `live=True` is passed explicitly. That is not
timidity: a gesture classifier wired to a keyboard is a program that types
whatever it hallucinates, and the failure mode is your own machine doing things
you did not ask for while you are trying to debug why. Dry run is how you
develop it, and it prints exactly what it would have done.

macOS needs Accessibility permission before synthetic input works at all
(System Settings, Privacy and Security, Accessibility). Without it pynput fails
quietly rather than loudly, which is worth knowing before you spend an evening
wondering why nothing happens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Action:
    """One thing the computer can be asked to do."""

    name: str
    describe: str
    perform: Callable[["Controller"], None]


@dataclass
class Controller:
    """Sends input to the OS, or pretends to and keeps a log."""

    live: bool = False
    log: list[str] = field(default_factory=list)
    _keyboard: object | None = field(default=None, init=False, repr=False)
    _mouse: object | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.live:
            from pynput.keyboard import Controller as Keyboard
            from pynput.mouse import Controller as Mouse

            self._keyboard = Keyboard()
            self._mouse = Mouse()

    def _record(self, what: str) -> None:
        self.log.append(what)
        if len(self.log) > 200:
            del self.log[:-200]

    # --- primitives -------------------------------------------------------

    def tap(self, key: str) -> None:
        """Press and release a named key, e.g. "space" or "media_play_pause"."""
        self._record(f"tap {key}")
        if not self.live:
            return
        from pynput.keyboard import Key

        resolved = getattr(Key, key, key)
        self._keyboard.press(resolved)
        self._keyboard.release(resolved)

    def hotkey(self, *keys: str) -> None:
        """Chord, e.g. hotkey("ctrl", "right") to change desktop."""
        self._record("hotkey " + "+".join(keys))
        if not self.live:
            return
        from pynput.keyboard import Key

        resolved = [getattr(Key, k, k) for k in keys]
        for k in resolved:
            self._keyboard.press(k)
        for k in reversed(resolved):
            self._keyboard.release(k)

    def move_cursor(self, x: int, y: int) -> None:
        self._record(f"move {x},{y}")
        if self.live:
            self._mouse.position = (x, y)

    def click(self) -> None:
        self._record("click")
        if not self.live:
            return
        from pynput.mouse import Button

        self._mouse.click(Button.left)

    def press_mouse(self, down: bool) -> None:
        """Hold or release the button, which is what makes dragging possible."""
        self._record("mouse down" if down else "mouse up")
        if not self.live:
            return
        from pynput.mouse import Button

        (self._mouse.press if down else self._mouse.release)(Button.left)

    def scroll(self, amount: int) -> None:
        self._record(f"scroll {amount}")
        if self.live:
            self._mouse.scroll(0, amount)


# --- the default bindings -------------------------------------------------
#
# Chosen so that nothing destructive is reachable. There is deliberately no
# binding that closes a window, quits an app or deletes anything: a gesture
# system will misfire, and the cost of a misfire should be an annoyance rather
# than lost work.

ACTIONS: dict[str, Action] = {
    "fist": Action("fist", "play / pause", lambda c: c.tap("media_play_pause")),
    "peace": Action("peace", "next desktop", lambda c: c.hotkey("ctrl", "right")),
    "l_shape": Action("l_shape", "previous desktop", lambda c: c.hotkey("ctrl", "left")),
    "thumbs_up": Action("thumbs_up", "volume up", lambda c: c.tap("media_volume_up")),
    "thumbs_down": Action("thumbs_down", "volume down", lambda c: c.tap("media_volume_down")),
    # Dynamic gestures. Swipes read naturally as navigation, and a snap is a
    # deliberate, hard-to-do-by-accident movement, so it gets play/pause too.
    "swipe_left": Action("swipe_left", "previous track", lambda c: c.tap("media_previous")),
    "swipe_right": Action("swipe_right", "next track", lambda c: c.tap("media_next")),
    "swipe_up": Action("swipe_up", "scroll up", lambda c: c.scroll(5)),
    "swipe_down": Action("swipe_down", "scroll down", lambda c: c.scroll(-5)),
    "snap": Action("snap", "play / pause", lambda c: c.tap("media_play_pause")),
}

# Poses used for continuous control rather than discrete firing. They are
# handled outside the state machine, because a cursor should track your finger
# every frame, not wait for a dwell timer.
CONTINUOUS = {"point", "pinch"}

# Handled by the dynamic lane rather than the pose classifier, so the static
# machine must never see them as candidates.
DYNAMIC = {"swipe_left", "swipe_right", "swipe_up", "swipe_down", "snap", "pinch_drag"}

# The pose a hand must be holding for a swipe to count.
#
# Measured, the swipe lane fired 3.3 times a minute at an idle hand while
# catching only 38% of real swipes, and no amount of training fixes that: a
# fast straight hand movement is what reaching for a cup looks like, so
# displacement alone cannot separate intent from traffic. The fix is
# interaction design. Requiring a deliberate pose during the movement is the
# same trick the engage gesture uses, one level down: a flat palm is not a
# shape a hand passes through while doing something else.
# Measured, and the choice is entirely about rarity. A gate the idle state
# already satisfies is not a gate:
#
#   gate pose          share of idle   idle firings/min   swipes caught
#   none                         n/a               3.00             50%
#   open_palm + point            35%               1.17             42%
#   pinch                       3.3%               0.00             39%
#   thumbs_up                   0.3%               0.00             33%
#
# The first attempt gated on open_palm and point, which together are 35% of
# what an idle hand reads as, and it only halved the problem. A pinch is 3.3%
# and removes idle firings entirely. It also reads naturally, since pinching
# while moving is grab and drag, and pinch is already the drag gesture.
SWIPE_GATE = ("pinch",)

# How much of the movement must be spent in a gate pose. Not all of it,
# because the classifier is briefly uncertain at the start and end of any
# movement, and demanding a perfect run would reject genuine swipes.
SWIPE_GATE_SHARE = 0.5


def describe_bindings() -> list[str]:
    lines = [f"  {name:<12} {action.describe}" for name, action in sorted(ACTIONS.items())]
    lines.append(f"  {'point':<12} move the cursor")
    lines.append(f"  {'pinch':<12} click, or hold to drag")
    lines.append(f"  {'open_palm':<12} engage / disengage")
    return lines


def perform(gesture: str, controller: Controller) -> bool:
    """Run the action bound to a gesture. False if nothing is bound."""
    action = ACTIONS.get(gesture)
    if action is None:
        return False
    action.perform(controller)
    return True
