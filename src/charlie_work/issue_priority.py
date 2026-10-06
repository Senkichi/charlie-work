"""Generic issue priority: the ``priority:<level>`` label (TIS-CW-7).

Any issue may carry ``priority:<level>``; nothing special-cases who filed it.
Two places read it:

* fresh dispatch claims ready issues in priority order, ``critical`` first
  (``OrchestratorApp._order_dispatch_candidates``, the one ordering both the
  real and the dry-run pass use) -- a stable re-rank, so the configured
  ``dispatch.order`` still decides within a level;
* the Aviator hand-off adds ``auto_merge.mergequeue_skip_line_label`` to a PR
  whose linked issue is ``critical``, so it is queued at the front.

:data:`LEVELS` is the vocabulary, most urgent first. A level outside it, or no
label, ranks as ``normal``; an issue with several levels takes the most urgent.
The prefix is ``LabelConfig.priority_prefix``; an empty prefix turns both off.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .github import label_names

LEVELS: tuple[str, ...] = ("critical", "high", "normal", "low")
CRITICAL = LEVELS[0]
DEFAULT = "normal"
_RANK = {level: rank for rank, level in enumerate(LEVELS)}


def priority_level(labels: Iterable[str], prefix: str) -> str:
    """The most urgent known level among ``<prefix><level>`` labels, else ``normal``."""
    if not prefix:
        return DEFAULT
    folded = prefix.lower()
    named = (
        label[len(prefix) :].strip().lower()
        for label in labels
        if label.lower().startswith(folded)
    )
    known = [level for level in named if level in _RANK]
    return min(known, key=_RANK.__getitem__, default=DEFAULT)


def is_critical(labels: Iterable[str], prefix: str) -> bool:
    return priority_level(labels, prefix) == CRITICAL


def order_by_priority(issues: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    """``issues`` re-ranked by level, ``critical`` first; stable within a level."""
    if not prefix:
        return list(issues)
    return sorted(issues, key=lambda issue: _RANK[priority_level(label_names(issue), prefix)])
