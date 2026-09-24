"""Head-branch PR discovery for the ``clean_worktrees`` lane (issue #1713).

Extracted from ``worktree.py`` under the file-size ratchet (issue #1442):
new code must not land in the over-cap monolith, so the fallback lookup --
and the ``WorktreeCleanGH`` protocol both it and ``clean_worktrees`` type
their ``gh`` parameter against -- lives here and is re-imported by
``worktree.py``.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .github import GitHubRunResult, WORKTREE_PR_HEAD_FIELDS


@runtime_checkable
class WorktreeCleanGH(Protocol):
    """Slice of :class:`GitHub` that ``clean_worktrees`` depends on.

    The cleanup lane only reads: PR merge state via ``gh pr view``, and --
    only when state.json carries no linked PR for the worktree's issue -- a
    ``gh pr list --head <branch> --state all`` fallback that discovers the
    PR by head branch (issue #1713). It needs just ``run`` -- the rest of
    ``GitHub`` (list caches,
    mutating ops, retry config) is irrelevant here. Narrowing the parameter
    type to this protocol lets test doubles satisfy the contract structurally
    without subclassing the frozen ``GitHub`` dataclass, and documents exactly
    which GitHub surface the cleanup lane relies on (issue #641).
    """

    def run(
        self, args: list[str], *, json_output: bool = False, allow_failure: bool = False
    ) -> Any: ...


def _pr_number_for_head_branch(gh: WorktreeCleanGH, branch: str) -> tuple[int | None, str | None]:
    """Fallback PR resolution for ``clean_worktrees`` (issue #1713).

    When ``state.json`` carries no linked PR for a worktree's issue -- the
    state entry was pruned, or the worktree was created outside the normal
    dispatch path -- resolve the PR by head branch instead, via a live
    ``gh pr list --head <branch> --state all``. Exactly one matching PR is
    an unambiguous link; its number is returned so the caller's unchanged
    ``gh pr view`` + merged/contained/dirty gates still decide eligibility.
    Zero matches, more than one match, a malformed payload, or an erroring
    ``gh`` call return ``None`` -- the caller keeps the fail-closed skip.

    Returns ``(pr_number, detail)``: ``detail`` is ``None`` on a successful
    single match and otherwise describes the lookup outcome for the skip
    reason, so the skip classes stay distinguishable in durable output.
    """
    result = gh.run(
        ["pr", "list", "--head", branch, "--state", "all", "--json", WORKTREE_PR_HEAD_FIELDS],
        json_output=True,
        allow_failure=True,
    )
    if not (isinstance(result, GitHubRunResult) and result.ok):
        error = result.error if isinstance(result, GitHubRunResult) else "unknown"
        return None, f"head-branch PR lookup failed: {error}"
    if not isinstance(result.value, list):
        # e.g. gh exited 0 with empty stdout: cannot distinguish an empty
        # result from an unreadable one (same ambiguity github.py's
        # allow_failure=False path calls out) -- treat as lookup failure,
        # never as "zero matches".
        return None, "head-branch PR lookup returned no usable list"
    entries = result.value
    if not entries:
        return None, "no PR found for head branch"
    if len(entries) == 1:
        number = entries[0].get("number") if isinstance(entries[0], dict) else None
        if isinstance(number, int):
            return number, None
        return None, "the head-branch PR entry has no number field"
    return None, f"{len(entries)} PRs share head branch; cannot pick one"
