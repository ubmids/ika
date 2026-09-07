#!/bin/bash
# Launch the terminal app from anywhere.
#
# Exists for one reason: curses has to own a real terminal, so it cannot be
# started from a pipe, a tool call, or anything without a tty. Run this from
# any terminal window and it works.
set -e
here="$(cd "$(dirname "$(readlink -f "$0" 2>/dev/null || echo "$0")")" && pwd)"
printf '\e[8;46;100t'    # ask for 46 rows: the layout wants the height
cd "$here"
if [ $# -eq 0 ]; then exec .venv/bin/ika live; fi
exec .venv/bin/ika "$@"
