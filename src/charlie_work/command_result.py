"""``CommandResult``: the leaf value type every orchestrator command returns.

Lives in its own module so ``supervise`` and ``status_snapshot`` can import it
at module top without a cycle through ``workflow``. ``workflow`` binds it
privately as ``_CommandResult`` for its own internals; it is not re-exported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CommandResult:
    ok: bool
    message: str
    data: dict[str, Any]
