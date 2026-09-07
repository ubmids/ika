"""The action layer. Fully testable because dry run is the default."""

import pytest

from ika import control


def test_dry_run_is_the_default():
    """The single most important property here. A Controller you construct by
    accident must not be able to touch the machine."""
    assert control.Controller().live is False


def test_dry_run_records_instead_of_acting():
    c = control.Controller()
    c.tap("space")
    c.hotkey("ctrl", "right")
    c.move_cursor(100, 200)
    c.click()
    c.scroll(-3)
    assert c.log == ["tap space", "hotkey ctrl+right", "move 100,200", "click", "scroll -3"]


def test_every_bound_gesture_actually_does_something():
    c = control.Controller()
    for name in control.ACTIONS:
        before = len(c.log)
        assert control.perform(name, c) is True
        assert len(c.log) > before, f"{name} is bound but did nothing"


def test_unbound_gestures_are_ignored_quietly():
    c = control.Controller()
    assert control.perform("rest", c) is False
    assert control.perform("nonsense", c) is False
    assert c.log == []


def test_nothing_bound_is_destructive():
    """A gesture system will misfire, so a misfire must cost an annoyance and
    never lost work. No quit, no close, no delete anywhere in the bindings."""
    c = control.Controller()
    for name in control.ACTIONS:
        control.perform(name, c)
    forbidden = ("cmd+q", "cmd+w", "delete", "backspace", "cmd+shift+3")
    for entry in c.log:
        assert not any(bad in entry.lower() for bad in forbidden), entry


def test_continuous_poses_are_not_also_bound_to_discrete_actions():
    """Pointing moves the cursor every frame. If it were also a discrete
    binding it would fire an action while you were merely aiming."""
    assert not (control.CONTINUOUS & set(control.ACTIONS)), "a pose cannot be both"


def test_the_log_does_not_grow_without_bound():
    c = control.Controller()
    for i in range(500):
        c.tap("space")
    assert len(c.log) <= 200


def test_bindings_are_all_described():
    text = "\n".join(control.describe_bindings())
    for name in control.ACTIONS:
        assert name in text
    for name in control.CONTINUOUS:
        assert name in text
