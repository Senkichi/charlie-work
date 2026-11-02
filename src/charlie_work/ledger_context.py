"""Test-ledger run context for worker and gate test runs.

The ci-fleet pytest recorder reads ``CI_FLEET_LEDGER_CONTEXT`` (``worker`` or
``gate``) and ``CI_FLEET_LEDGER_TICKET`` from the environment. The ticket is the
bare issue number: the recorder qualifies it with the repo slug it derives
itself, so launchers need no slug lookup.
"""

from __future__ import annotations

CONTEXT_VAR = "CI_FLEET_LEDGER_CONTEXT"
TICKET_VAR = "CI_FLEET_LEDGER_TICKET"


def ledger_env(context: str, issue_number: int) -> dict[str, str]:
    return {CONTEXT_VAR: context, TICKET_VAR: str(issue_number)}
