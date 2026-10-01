"""The ``worktree_unsafe`` sub-kinds and the reason-string classifier.

Extracted from ``worktree.py`` under the issue #1442 file-size ratchet when
issue #2019 added the third kind; re-exported through ``worktree.py``'s
facade import block so callers keep importing from ``charlie_work.worktree``.
"""

from __future__ import annotations

from .non_worker_product import UNCOMMITTED_WORK_REASON_PREFIX

# Issue #807: ``worktree_unsafe`` is split at detection time into two
# discriminable kinds, because the two triggers do not share a correct
# response. Launch-shim dirt / adapter shim materialization (uncommitted
# modifications) is genuinely mechanical — auto-clear and redispatch is the
# right move. Genuine unpushed local commits on the worktree branch are a
# judgment call — returning the issue to dispatch actively fights the safety
# system that raised the escalation and risks a second writer on a branch
# that already has divergent local work. The discriminator is the
# ``dirty_reason`` string already computed at the raise site; classifying at
# detection (rather than after the fact) makes the invalid state
# unrepresentable.
WORKTREE_UNSAFE_KIND_SHIM_DIRT = "worktree_unsafe_shim_dirt"
WORKTREE_UNSAFE_KIND_LOCAL_COMMITS = "worktree_unsafe_local_commits"
# Issue #2019: uncommitted edits to source/test files (a dead worker's real,
# unpaid-for output) are NOT shim dirt. Judgment class: never auto-cleared.
WORKTREE_UNSAFE_KIND_UNCOMMITTED_WORK = "worktree_unsafe_uncommitted_work"
WORKTREE_UNSAFE_KINDS: frozenset[str] = frozenset(
    {
        WORKTREE_UNSAFE_KIND_SHIM_DIRT,
        WORKTREE_UNSAFE_KIND_LOCAL_COMMITS,
        WORKTREE_UNSAFE_KIND_UNCOMMITTED_WORK,
    }
)


def _worktree_unsafe_kind_from_reason(reason: str) -> str:
    """Map a ``dirty_reason`` string to its ``worktree_unsafe`` sub-kind.

    The reason strings are produced by ``_worktree_refuse_to_reset_reason``
    and ``_worktree_dirty_reason``. "uncommitted modifications" denotes
    shim/adapter dirt (mechanical); "local commit(s)" denotes genuine
    divergence (judgment). A reason that matches neither known pattern
    defaults to ``WORKTREE_UNSAFE_KIND_LOCAL_COMMITS`` — the fail-closed
    classification toward judgment/human-needed — so a future reason
    string that doesn't match either pattern escalates as a judgment
    call rather than being silently treated as mechanical and
    auto-cleared (which would reproduce the #807 bug the split exists to
    prevent). This mirrors the fail-closed convention already enforced
    for an unrecognized explicit ``kind`` in ``WorktreeUnsafeError.__init__``
    and for an unrecognized ``reason_class`` in
    ``state.escalation_reason_class``.
    """
    if UNCOMMITTED_WORK_REASON_PREFIX in reason:
        return WORKTREE_UNSAFE_KIND_UNCOMMITTED_WORK
    if "local commit" in reason:
        return WORKTREE_UNSAFE_KIND_LOCAL_COMMITS
    if "uncommitted modifications" in reason:
        return WORKTREE_UNSAFE_KIND_SHIM_DIRT
    return WORKTREE_UNSAFE_KIND_LOCAL_COMMITS
