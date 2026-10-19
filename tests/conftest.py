"""Nothing in the test suite ever makes a sound.

Every spoken call and every round bell goes through `cue._spawn`, so it is
replaced for every test with a recorder. A test that wants to know what
would have been played reads `sounds`.
"""

import pytest


@pytest.fixture(autouse=True)
def sounds(monkeypatch):
    from ika import cue

    played: list[list[str]] = []
    monkeypatch.setattr(cue, "_spawn", played.append)
    return played
