"""The shared plumbing one apply walk threads through every action (issue #3119).

Split out of ``apply.py`` so the engine's pieces stay under the per-module
size cap: ``apply.py`` keeps the public entry point and the Turn-level
bookkeeping, ``apply_checks.py`` the guard layer every effect passes first,
and ``apply_handlers.py`` the per-type handlers. This module holds what all
of them share — the run context, the verdict shape, the outcome/event-kind
vocabulary, and the marker/digest helpers.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import LabelConfig
from .charter import Charter

OUTCOME_APPLIED = "applied"
OUTCOME_REFUSED = "refused"
#: A recorded action on replay, or one the pause-stop left unprocessed.
OUTCOME_SKIPPED = "skipped"

#: The repo's private-slug baseline; its presence arms the public-repo slug
#: refusal (issue #1502's ratchet baseline, reused as a data source).
_SLUG_BASELINE_DIRNAME = ".private-slug-baseline"
_SLUG_SETS_DIRNAME = "slugs"

#: Color/description used when Apply bootstraps the ownership label. The
#: purple matches ``helpers_labels._label_color``'s non-ready convention.
_OWNER_LABEL_COLOR = "5319E7"

#: The hidden marker trailing every foreman-filed issue body. ``{turn}`` and
#: ``{key}`` carry the Turn id and action key, so a crash between create and
#: record is found by searching the ownership label for this exact string.
_MARKER_TEMPLATE = "<!-- foreman:{name} turn:{turn} key:{key} -->"


def _marker_pattern(foreman_name: str) -> re.Pattern[str]:
    """Match this foreman's hidden marker inside an authored issue body."""
    return re.compile(
        r"<!--\s*foreman:" + re.escape(foreman_name) + r"\s+turn:(\S+)\s+key:(\S+)\s*-->"
    )


def _marker(foreman_name: str, turn_id: str, key: str) -> str:
    return _MARKER_TEMPLATE.format(name=foreman_name, turn=turn_id, key=key)


def _body_hash(body: str) -> str:
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _fingerprint(raw_action: Any) -> str:
    """Canonical digest of one raw action object (replay-skip identity)."""
    return hashlib.sha256(
        json.dumps(raw_action, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass
class _Run:
    """The mutable bookkeeping one apply walk threads through every action."""

    foreman_root: Path
    turn_id: str
    charter: Charter
    gh: Any
    labels: LabelConfig
    owner_label: str
    signal_marked: bool
    dry_run: bool
    open_issues_used: int
    file_issues_applied: int
    marker_map: dict[tuple[str, str], int]
    key_numbers: dict[str, int] = field(default_factory=dict)
    failed_keys: set[str] = field(default_factory=set)
    cycle_keys: frozenset[str] = frozenset()
    duplicate_file_keys: frozenset[str] = frozenset()
    duplicate_item_keys: frozenset[str] = frozenset()
    known_file_keys: frozenset[str] = frozenset()

    @property
    def repo_root(self) -> Path | None:
        return getattr(self.gh, "repo_root", None)

    @property
    def state_dir(self) -> str | None:
        state_dir = getattr(self.gh, "state_dir", None)
        if state_dir is None:
            state_dir = getattr(getattr(self.gh, "runtime", None), "state_dir", None)
        return state_dir


@dataclass(frozen=True)
class _Verdict:
    """A handler's per-action result before the index is folded in."""

    outcome: str
    reason: str | None = None
    detail: str | None = None
    issue_number: int | None = None
    item_id: str | None = None


def _refused(reason: str) -> _Verdict:
    return _Verdict(outcome=OUTCOME_REFUSED, reason=reason)


def _applied(detail: str, issue_number: int | None = None, item_id: str | None = None) -> _Verdict:
    return _Verdict(
        outcome=OUTCOME_APPLIED, detail=detail, issue_number=issue_number, item_id=item_id
    )
