"""A human's outstanding request-changes, as a hold on the **Merge path** (issue #2763).

The merge lane approves on the fleet's own verdict only. On 2026-10-19 a PR
merged over a human's unresolved request-changes review because nothing on the
path read one. This module is the pure half of the fix: a classifier over the
PR payload's ``reviewDecision`` and its issue comments, and the gate
``decide_merge`` consults. The comment fetch is the shell's job
(``gather.gather_human_block``); nothing here touches ``gh``, the clock or state.

The hold is *lazy*: it is evaluated only where a merge or hand-off is about to
happen (``HoldFacts.human_block`` is ``None`` until then, and the gate raises
``FactNotGathered`` if a decision needs it unread), so a pass that was never
going to merge pays no GitHub call. It applies no ``mergequeue`` label, does
not escalate, and clears itself: the next pass re-reads the same sources, so a
later approve comment (or the formal review flipping) releases the PR with no
operator step.

Signal precedence, strongest first:

1. ``reviewDecision == "CHANGES_REQUESTED"`` on the PR payload already in
   hand: blocks. No comment read -- and a later comment cannot clear a formal
   review; GitHub's own ``reviewDecision`` does that.
2. Otherwise the issue comments, newest first, skipping the orchestrator's own
   (provenance-marker) comments. The newest remaining comment that matches a
   verdict pattern decides: request-changes blocks, approved clears. A comment
   matching neither is not a verdict and is skipped. A comment matching both
   resolves to request-changes (fail closed).
3. Unreadable comments hold (fail closed), reported as ``unavailable``.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .model import FactNotGathered, Hold, HoldFacts

#: ``PR.reviewDecision`` value of an unresolved formal request-changes review.
CHANGES_REQUESTED = "CHANGES_REQUESTED"

HUMAN_BLOCK_SOURCE_REVIEW_DECISION = "review_decision"
HUMAN_BLOCK_SOURCE_COMMENT = "comment"


@dataclass(frozen=True)
class HumanBlockRead:
    """What one read of the human-review signals concluded."""

    blocked: bool
    unavailable: bool = False
    #: ``HUMAN_BLOCK_SOURCE_*`` of the signal that blocked; None when none did.
    source: str | None = None


def compile_verdict_pattern(pattern: str) -> re.Pattern[str]:
    """The shipped patterns are case-insensitive; config validation guarantees they compile."""
    return re.compile(pattern, re.IGNORECASE)


def classify_human_block(
    review_decision: str | None,
    comments: Sequence[Mapping[str, Any]],
    *,
    request_changes: re.Pattern[str],
    approved: re.Pattern[str],
    is_orchestrator_comment: Callable[[Mapping[str, Any]], bool],
) -> HumanBlockRead:
    """Decide the human block from the signals in hand (pure).

    ``comments`` is in the API's order (oldest first); the scan runs newest
    first. ``is_orchestrator_comment`` is injected because the marker lives in
    ``charlie_work.workflow``, which this package cannot import at module load.
    """
    if review_decision == CHANGES_REQUESTED:
        return HumanBlockRead(True, source=HUMAN_BLOCK_SOURCE_REVIEW_DECISION)
    for comment in reversed(comments):
        if not isinstance(comment, Mapping) or is_orchestrator_comment(comment):
            continue
        body = comment.get("body")
        if not isinstance(body, str):
            continue
        if request_changes.search(body):
            return HumanBlockRead(True, source=HUMAN_BLOCK_SOURCE_COMMENT)
        if approved.search(body):
            return HumanBlockRead(False)
    return HumanBlockRead(False)


@dataclass(frozen=True)
class HumanBlockGate:
    """``decide_merge``'s view of the human block for one pass."""

    read_needed: bool
    blocked: bool = False
    unavailable: bool = False
    source: str | None = None

    @property
    def held(self) -> bool:
        """Blocked, or could not be ruled out: either way the PR is left alone."""
        return self.blocked or self.unavailable

    @property
    def holds(self) -> frozenset[Hold]:
        found: set[Hold] = set()
        if self.blocked:
            found.add(Hold.HUMAN_BLOCK)
        if self.unavailable:
            found.add(Hold.HUMAN_BLOCK_UNAVAILABLE)
        return frozenset(found)


def human_block_read_needed(enter: bool, holds: HoldFacts) -> bool:
    """A merge or hand-off could happen right now AND the feature is on."""
    return bool(enter and holds.config.honor_human_block)


def human_block_gate(enter: bool, holds: HoldFacts) -> HumanBlockGate:
    """The gate for this pass; ``enter`` is ``merge_entry``'s second value.

    Read-needed means a merge or hand-off could happen right now AND the
    feature is on. Raises ``FactNotGathered`` when it is needed but the shell
    did not read it -- an unread signal must never be mistaken for "no block".
    """
    if not human_block_read_needed(enter, holds):
        return HumanBlockGate(read_needed=False)
    if holds.human_block is None and not holds.human_block_unavailable:
        raise FactNotGathered("human_block is needed: a merge or hand-off is possible")
    return HumanBlockGate(
        read_needed=True,
        blocked=holds.human_block is True,
        unavailable=holds.human_block_unavailable,
        source=holds.human_block_source if holds.human_block is True else None,
    )
