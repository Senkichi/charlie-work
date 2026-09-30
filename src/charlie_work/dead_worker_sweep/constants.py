"""Names the decide modules share with the code they replace.

Re-exported from their defining modules (never redeclared) so a rename there
cannot silently desynchronise the sweep. Kept out of ``decide*.py`` because
those modules may not import ``state`` (see the AST guard in
``tests/test_dws_decision_table.py``): a constant is data, not an effect.
"""

from __future__ import annotations

from ..live_handoff_finalize import ROUTED_OUTCOME_KEY
from ..orphaned_worker_no_op_drain import NO_OP_DEFERRED_HEAD_KEY
from ..rework_outcome import APPLIED_HEADS_KEY
from ..state import PASSIVE_OPEN_STATUS

__all__ = [
    "APPLIED_HEADS_KEY",
    "NO_OP_DEFERRED_HEAD_KEY",
    "PASSIVE_OPEN_STATUS",
    "ROUTED_OUTCOME_KEY",
]
