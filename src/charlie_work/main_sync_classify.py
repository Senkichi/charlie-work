"""Pure main-sync decisions over a frozen snapshot of one checkout (docs/specs/main-sync.md).

No I/O: ``main_sync_git.read_checkout_state`` builds the ``CheckoutState``,
``classify`` picks exactly one action, and ``split_at_first_touch`` decides how far a
fast-forward may go before the repo's ``post_sync`` command has to run.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Literal

Action = Literal["none", "ff", "ref_update", "skip"]

# Expected, operator-chosen states: no event and no pass line.
SILENT_SKIP_REASONS = frozenset({"disabled", "no_origin", "not_primary", "excluded"})
# Waiting never resolves these (origin was re-rooted): alarm on first sight.
ALARM_ON_SIGHT_REASONS = frozenset({"unrelated_history"})
INSTRUCTION_BASENAMES = frozenset({"CLAUDE.md", "AGENTS.md"})


@dataclass(frozen=True)
class CheckoutState:
    """What git says about one primary checkout, after the fetch.

    ``local_sha``/``origin_sha`` are the tips of ``refs/heads/<default_branch>`` and
    ``refs/remotes/origin/<default_branch>``. ``ahead`` counts local commits missing
    from origin, ``behind`` origin commits missing locally. ``operation_in_progress``
    is already a skip reason (``rebase_in_progress``, ...). The working-tree fields
    are only read when ``current_branch == default_branch``.
    """

    default_branch: str
    current_branch: str | None  # None: detached HEAD
    local_sha: str
    origin_sha: str
    merge_base: str | None
    ahead: int
    behind: int
    operation_in_progress: str | None = None
    index_locked: bool = False
    default_checked_out_elsewhere: bool = False
    dirty_tracked: tuple[str, ...] = ()
    untracked: tuple[str, ...] = ()
    incoming_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class SyncDecision:
    action: Action
    reason: str | None = None
    paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class IncomingCommit:
    sha: str
    paths: tuple[str, ...]


def classify(state: CheckoutState) -> SyncDecision:
    """Exactly one row of the spec's classification table; the first match wins."""
    if state.local_sha == state.origin_sha:
        return SyncDecision("none")
    if state.operation_in_progress:
        return SyncDecision("skip", state.operation_in_progress)
    if state.index_locked:
        return SyncDecision("skip", "index_locked")
    if state.current_branch is None:
        return SyncDecision("skip", "detached_head")
    if state.merge_base is None:
        return SyncDecision("skip", "unrelated_history")
    if state.ahead > 0:
        return SyncDecision("skip", "diverged")
    if state.current_branch != state.default_branch:
        if state.default_checked_out_elsewhere:
            return SyncDecision("skip", "checked_out_elsewhere")
        return SyncDecision("ref_update")
    blockers = (set(state.dirty_tracked) | set(state.untracked)) & set(state.incoming_paths)
    if blockers:
        return SyncDecision("skip", "blocked", tuple(sorted(blockers)))
    return SyncDecision("ff")


def split_at_first_touch(
    commits: Sequence[IncomingCommit], patterns: Sequence[str]
) -> tuple[str | None, bool]:
    """``(safe_sha, needs_post_sync)`` for an oldest-first incoming range.

    ``safe_sha`` is the last commit before the first one touching ``patterns`` (None
    when that is the very first commit). ``needs_post_sync`` is False when no commit
    touches them, so the whole range lands without ``post_sync``. Empty ``patterns``
    means every commit touches.
    """
    for index, commit in enumerate(commits):
        if not patterns or any(fnmatchcase(p, pat) for p in commit.paths for pat in patterns):
            return (commits[index - 1].sha if index else None), True
    return None, False


def instruction_paths(paths: Iterable[str]) -> tuple[str, ...]:
    """Changed paths a live session loaded as instructions: ``CLAUDE.md``/``AGENTS.md``
    at any depth, or anything under a ``.claude/`` directory. Sorted, de-duplicated."""

    def is_instruction(path: str) -> bool:
        parts = path.split("/")
        return parts[-1] in INSTRUCTION_BASENAMES or ".claude" in parts[:-1]

    return tuple(sorted({p for p in paths if is_instruction(p)}))
