"""Where ika keeps what it downloads and what it remembers.

From a clone, that is the clone: `models/` and `data/profiles/` beside the
code, as it always was, so the research scripts and the README's measured
replays find everything where they expect it. Installed as a tool (pipx,
uv tool), the package sits inside a virtual environment that an upgrade or
an uninstall replaces, so models and your session history go in `~/.ika`
instead, where they survive that. `IKA_HOME` overrides both.

Nothing here imports anything heavy: `setup_check` reads it before numpy or
mediapipe are known to be installed.
"""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent


def _home() -> Path:
    override = os.environ.get("IKA_HOME")
    if override:
        return Path(override).expanduser()
    checkout = PACKAGE.parent
    if (checkout / "pyproject.toml").is_file() and (checkout / "ika").is_dir():
        return checkout
    return Path.home() / ".ika"


HOME = _home()
