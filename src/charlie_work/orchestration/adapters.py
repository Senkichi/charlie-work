"""Property / staticmethod adapter delegates for ``OrchestratorApp`` (L09, #1640).

Track 2 Phase B's final leaf. The three members here are the only
``OrchestratorApp`` members that are *not* plain instance methods -- a
``@property`` (``layout``) and two ``@staticmethod``s (``_is_dead_blocker``,
``_write_json``). The derived installer (``workflow_delegation``) attaches a
plain routed ``def`` as an instance method, which is wrong for these three, so
each is tagged with the ``as_property`` / ``as_staticmethod`` marker decorator
(design Section 3.3). The marker leaves the object a plain ``FunctionDef`` -- the
AST-equivalence gate (#1607) still sees the byte-identical moved body -- and
``_install_delegates`` wraps it into ``property(fn)`` / ``staticmethod(fn)`` at
attach time via ``_adapt``.

Module-namespace rule (#1627): none of these three bodies reaches a
workflow-defined name, so this module needs no ``import charlie_work.workflow as
_wf`` seam at all. ``ResolvedLayout``, ``Path`` and ``Any`` are annotation-only
and imported directly from their defining modules (matching the convention in
the sibling ``dispatch_state.py``); ``json`` is a runtime dependency of
``_write_json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from charlie_work.paths import ResolvedLayout
from charlie_work.workflow_delegation import as_property, as_staticmethod


@as_staticmethod
def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


@as_staticmethod
def _is_dead_blocker(
    blocker_number: int,
    state: dict[str, Any],
    pr_by_issue: dict[int, dict[str, Any]],
) -> bool:
    """True when a blocker issue can never resolve through any automated path.

    Used by dispatch()'s blocked-chain attention check: "dead" means the
    blocker issue itself is escalated, or its tracked open PR's status is
    escalated/janitor_blocked. Pure local-state lookup, no GitHub calls --
    this only names an already-known dead end, it never widens one.

    Issue #1133: ``janitor_blocked`` conflates a durably-stuck population
    (failed checks, merge conflict, body gate, CI-never-created) with a
    transient one -- a brand-new PR whose required checks simply haven't
    reported yet, which self-heals within one CI cycle. The transient
    case is identified structurally by ``is_missing_checks_only_block``
    (the SOLE janitor failure is "Required check(s) missing") combined
    with the absence of a ``ci_run_never_created_head`` marker (which
    would mean CI was confirmed to have never started for this head -- a
    durable condition). Such a PR is NOT dead: it is actively progressing
    and will unblock on the next janitor pass once CI reports. The
    ``escalated`` status stays dead unconditionally.
    """
    issue_entry = state.get("issues", {}).get(str(blocker_number), {})
    if isinstance(issue_entry, dict) and issue_entry.get("status") == "escalated":
        return True
    pr = pr_by_issue.get(blocker_number)
    if pr is not None:
        pr_number = pr.get("number")
        if pr_number is not None:
            pr_state = state.get("prs", {}).get(str(pr_number), {})
            pr_status = pr_state.get("status")
            if pr_status == "escalated":
                return True
            if pr_status == "janitor_blocked":
                # Issue #1133: a brand-new PR whose only janitor failure is
                # "Required check(s) missing" (checks not reported yet) is
                # transient, not dead -- unless ``ci_run_never_created_head``
                # is set, which means CI was confirmed to have never started
                # for this head (the durable population the alert exists for).
                # Branch on the structured flag, never on failure-message
                # text (same rule as is_draft_only_block consumers).
                if pr_state.get("is_missing_checks_only_block") and not pr_state.get(
                    "ci_run_never_created_head"
                ):
                    return False
                return True
    return False


@as_property
def layout(self) -> ResolvedLayout:
    """Public, read-only view of the resolved state-child layout.

    This is the contract module-level helpers (e.g. ``supervise.run_supervised``)
    that take an ``app`` argument should use, mirroring the existing public
    ``paths`` attribute -- callers outside this class should never reach into
    the private ``self._layout`` cache directly.
    """
    return self._layout
