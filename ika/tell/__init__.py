"""Finding the habit in a stream of movement, and calling the next move.

The hand lane answers "what is this doing". This answers "what will it do
next", which is a different problem and the one worth solving.

Three pieces:

`actions` invents fighters with habits planted on purpose, so the miner can be
graded against a known answer instead of an impression.

`habits` mines a stream for conditional dependencies and, crucially, decides
which of them are real. Test enough contexts and something always looks
significant, so this is mostly about not being fooled.

`lead` measures the only number that decides whether any of it is a product:
how long before the move lands can we call it.
"""

from .actions import Action, Fighter, Habit
from .habits import Finding, mine
from .lead import lead_times, summarise

__all__ = ["Action", "Fighter", "Habit", "Finding", "mine", "lead_times", "summarise"]
