"""Merge/branch capability: PR merge and branch lifecycle (Track 2, #1585).

Cluster G of the design doc's capability segmentation (Section 3.1):
``merge_pr``, ``delete_branch``, ``pr_update_branch``, ``pr_close``,
``pr_reopen``, ``push_empty_commit``, ``branch_protection``.

Track 2, issue #1592; design doc Section 5, L08: all seven Cluster G members
move here verbatim. Four of them (``branch_protection``, ``pr_close``,
``pr_reopen``, ``push_empty_commit``) are among the five members asserted
present in ``GitHubLike.__dict__`` (tests/test_githublike_protocol.py:33-84);
moving their bodies does not break those assertions because all four are
redeclared directly on the ``GitHubLike`` union body (design doc Section
4.1), independent of which sub-protocol/collaborator implements them.
"""

from __future__ import annotations

import json
from typing import Any, Protocol, runtime_checkable

# ``ci_fleet.github.GitHubError`` is imported directly here, not re-derived
# through ``charlie_work.github`` or ``_base.py``: ``delete_branch`` and
# ``pr_update_branch`` (moved below) catch it, and identity matters --
# ``github.py``'s own load-bearing comment on its ``GitHubError`` re-export
# (Track 2, issue #1585) explains why a local re-declaration would be a
# structurally-identical but *unrelated* exception type that no existing
# ``except GitHubError`` handler would catch. ``ci_fleet.github`` has no
# dependency on ``charlie_work``, so importing it directly here carries no
# circular-import risk (unlike ``charlie_work.github`` itself, which imports
# ``github_capabilities`` before its own definitions are ready). Mirrors
# ``issues.py``/``pull_requests.py``/``repo_meta.py``'s precedent for the
# same reasoning.
from ci_fleet.github import GitHubError

# ``GitHubRunResult`` lives in ``_base.py``, not ``charlie_work.github`` (Track
# 2, issue #1588; design doc Section 5, L04) -- see ``_base.py`` for the full
# circular-import rationale. Four of the seven members moved below
# (``pr_close``, ``pr_reopen``, ``push_empty_commit``, ``branch_protection``)
# perform a real runtime ``isinstance``/construction use of it, so (unlike
# before L08, when this leaf's own members hadn't moved yet) this must be a
# normal top-level import, not a ``TYPE_CHECKING``-only one -- mirroring
# ``pull_requests.py``'s L06 promotion of the same import for the same
# reason.
#
# ``_is_mutating`` also lives in ``_base.py`` (Track 2, issue #1590; design
# doc Section 5, L06) -- see ``_base.py``'s own comment for the cross-cutting
# rationale (shared with ``GitHub.run``, which stays on the owner
# permanently, and ``Transport._run_bool``, moved in L09).
# ``pr_close``/``pr_reopen`` (moved below) reference it as a bare global for
# their dry-run synthetic-result guard.
from ..config_validation import ConfigError
from ..github_transport.request import RestRequest
from ._base import CapabilityCollaborator, GitHubRunResult
from ._pr_mutations import enable_auto_merge
from ._send import send_result, send_text

# Flag constants for merge_pr -- single source of truth for both argv
# construction and config validation, moved from ``github.py`` alongside
# ``merge_pr`` (Track 2, issue #1592; design doc Section 5, L08). Both are
# referenced by ``merge_pr``'s body as bare globals, so they must be bound in
# this module's globals. Re-exported through
# ``github_capabilities/__init__.py`` and re-imported into ``github.py``,
# which still uses both directly to derive ``ORCHESTRATOR_MANAGED_MERGE_FLAGS``
# (a module-level constant that never moves -- it is not a ``GitHub`` member)
# -- the same re-export pattern already used for ``PR_LIST_FIELDS``/
# ``LABEL_LIST_FIELDS``/``_pr_number_from_url``.
_STRATEGY_FLAGS = {"merge": "--merge", "squash": "--squash", "rebase": "--rebase"}
_ADMIN_FLAG = "--admin"
_AUTO_FLAG = "--auto"
_MATCH_HEAD_FLAG = "--match-head-commit"


@runtime_checkable
class MergeBranchLike(Protocol):
    """Structural interface for PR-merge/branch-lifecycle operations."""

    def merge_pr(
        self, number: int, strategy: str, admin: bool = False, merge_flags: tuple[str, ...] = ()
    ) -> str: ...

    def delete_branch(self, branch: str) -> bool: ...

    def pr_update_branch(self, pr_number: int) -> bool: ...

    def pr_close(self, number: int) -> GitHubRunResult: ...

    def pr_reopen(self, number: int) -> GitHubRunResult: ...

    def push_empty_commit(self, branch: str) -> GitHubRunResult: ...

    def branch_protection(self, base: str) -> dict[str, Any] | None: ...


class MergeBranch(CapabilityCollaborator):
    """PR-merge/branch-lifecycle capability collaborator.

    Moved from ``GitHub`` verbatim (Track 2, issue #1592; design doc Section
    5, L08) -- all seven Cluster G members. Bodies still say
    ``self.run(...)``/``self.dry_run``/``self._list_cache``, which resolve
    through ``CapabilityCollaborator.__getattr__`` to the owner (design doc
    Section 3.3). None of the seven calls another of the seven (or any other
    moved-capability sibling) through ``self``, so -- unlike L07's
    ``are_issues_open``/``issue_view`` pair -- no internal call is at risk of
    resolving on the wrong collaborator instance; every ``self.run``/
    ``self.dry_run``/``self._list_cache`` reference in this class forwards to
    the owner exactly as it did when these bodies were ``GitHub`` methods.

    ``merge_pr`` references the module-level ``_STRATEGY_FLAGS``/
    ``_ADMIN_FLAG`` bare globals relocated above; ``delete_branch`` and
    ``pr_update_branch`` catch ``GitHubError`` (imported directly from
    ``ci_fleet.github``, the same identity-sensitive source ``github.py``
    itself re-exports from -- see this module's import block); ``pr_close``/
    ``pr_reopen`` use ``GitHubRunResult`` and ``_is_mutating`` (both
    relocated to ``_base.py`` in earlier leaves because they are shared with
    ``GitHub`` methods/module constants that have not moved -- see
    ``_base.py``'s own comments on each); ``push_empty_commit`` and
    ``branch_protection`` use ``GitHubRunResult`` only. Design doc Section
    3.3 covers only ``self.<attr>`` forwarding, not bare-global runtime
    symbols in moved bodies; this is the same disclosed design-gap
    resolution that recurs identically in L04/L05/L06/L07.
    """

    def merge_pr(
        self, number: int, strategy: str, admin: bool = False, merge_flags: tuple[str, ...] = ()
    ) -> str:
        # merge_flags takes precedence over the legacy admin field.
        flags = tuple(merge_flags) if merge_flags else ((_ADMIN_FLAG,) if admin else ())
        sha: str | None = None
        auto = False
        for flag in flags:
            name, _, value = flag.partition("=")
            if name == _ADMIN_FLAG:
                continue  # REST merge is the direct merge `--admin` selects
            if name == _AUTO_FLAG:
                auto = True
            elif name == _MATCH_HEAD_FLAG and value:
                sha = value
            else:
                # B8: the closed flag set replaces an open pass-through to gh.
                raise ConfigError(
                    f"auto_merge.merge_flags: unsupported flag {flag!r}; supported: "
                    f"{_ADMIN_FLAG}, {_AUTO_FLAG}, {_MATCH_HEAD_FLAG}=<sha>"
                )
        if auto:
            # B10: no REST route; the node id costs one extra read.
            enabled = enable_auto_merge(self, number, strategy)
            if not enabled.ok:
                raise GitHubError(
                    f"gh pr merge {number} --auto failed: {enabled.error or enabled.stderr}"
                )
            return f"merged #{number}"  # the text gh's empty stdout always fell back to
        # Branch deletion is deliberately NOT part of this call (the old
        # `gh pr merge --delete-branch` also switched the LOCAL branch and failed
        # when it was checked out in a worktree). Use `delete_branch` separately.
        body: dict[str, str] = {"merge_method": strategy}
        if sha is not None:
            body["sha"] = sha
        text = send_text(
            self,
            RestRequest.of("PUT", f"repos/{{owner}}/{{repo}}/pulls/{number}/merge", body=body),
        )
        # B8: the API's own message ("Pull Request successfully merged"); a
        # truthy value either way, so `merged` never reads False on success.
        try:
            message = json.loads(text).get("message") if text else None
        except (ValueError, AttributeError):
            message = None
        return message if isinstance(message, str) and message else f"merged #{number}"

    def delete_branch(self, branch: str) -> bool:
        """Best-effort deletion of the REMOTE head branch after a merge.

        Uses the git-refs API so local checkouts and worktrees are never
        touched. Returns False instead of raising — a deletion failure must
        never abort the merge/label sequence.

        Note: --delete-branch is in ORCHESTRATOR_MANAGED_MERGE_FLAGS because
        it's deliberately excluded from merge_pr to avoid worktree failures.
        """
        try:
            send_text(
                self,
                RestRequest.of("DELETE", f"repos/{{owner}}/{{repo}}/git/refs/heads/{branch}"),
            )
        except GitHubError:
            return False
        return True

    def pr_update_branch(self, pr_number: int) -> bool:
        """Update a PR's branch with the latest changes from its base.

        PUTs ``pulls/{n}/update-branch``. Returns True on success, False on failure
        (conflicts, network errors, etc.). Never raises — per-PR failures are
        reported as values and must not abort a batch operation.
        """
        try:
            send_text(
                self,
                RestRequest.of("PUT", f"repos/{{owner}}/{{repo}}/pulls/{pr_number}/update-branch"),
            )
            return True
        except GitHubError:
            return False

    def pr_close(self, number: int) -> GitHubRunResult:
        """Close a PR via ``gh pr close`` (issue #1274, W17).

        Half of the close/reopen stale-checks retrigger mechanism: closing
        and then reopening a PR is a common way to force GitHub to
        re-evaluate branch protection / re-create a check-suite run for a
        head where Actions never created one. Modeled exactly on
        ``pr_ready`` -- structured result, dry-run synthetic ok=True guard,
        never raises.
        """
        return send_result(
            self,
            RestRequest.of(
                "PATCH",
                f"repos/{{owner}}/{{repo}}/pulls/{number}",
                body={"state": "closed"},
            ),
        )

    def pr_reopen(self, number: int) -> GitHubRunResult:
        """Reopen a PR via ``gh pr reopen`` (issue #1274, W17).

        Paired with ``pr_close`` for the close/reopen stale-checks
        retrigger mechanism. Modeled exactly on ``pr_ready``.

        NOTE (unverified, flagged per issue #1274's binding comment item 6):
        whether reopening a PR actually causes GitHub Actions to create a
        fresh check-suite run for the PR's CURRENT head (as opposed to
        being a no-op for check-suite purposes) has not been confirmed
        against a live repository -- it cannot be verified with a real `gh`
        call inside a sandboxed/mocked test environment. The
        ``push_empty_commit`` fallback exists specifically because this is
        uncertain; an operator should confirm close/reopen's effect against
        a disposable fixture PR before relying on it as the primary
        mechanism in production.
        """
        return send_result(
            self,
            RestRequest.of(
                "PATCH",
                f"repos/{{owner}}/{{repo}}/pulls/{number}",
                body={"state": "open"},
            ),
        )

    def push_empty_commit(self, branch: str) -> GitHubRunResult:
        """Push a content-free commit onto ``branch`` via the Git Data API
        (issue #1274, W17).

        Fallback CI-retrigger mechanism, used only when ``pr_close`` +
        ``pr_reopen`` does not mechanically succeed (either call returns
        not-ok). Moves the branch tip to a new commit that has the exact
        same tree as the current tip (i.e. no content change) so a
        push-triggered workflow re-evaluates the branch at a fresh head
        SHA, without altering any file. Four `gh api` reads/writes, in
        order:

        1. GET the branch ref to find the current tip commit SHA.
        2. GET that commit object to find its tree SHA (the new commit
           reuses it unchanged).
        3. POST a new commit object with that tree and the old tip as its
           sole parent.
        4. PATCH the branch ref to point at the new commit.

        Every step returns errors as values -- this method never raises,
        per this repo's external-process-errors-as-values convention, and
        stops at the first failing step rather than attempting later steps
        against inconsistent state. Dry-run mode returns a synthetic
        ok=True result before any `gh` call is made at all (not just before
        the final PATCH): this operation is unconditionally mutating end to
        end -- there is no read-only prefix of it worth letting a
        --dry-run caller observe, unlike e.g. ``workflow_runs_for_head``.
        """
        if self.dry_run:
            return GitHubRunResult(
                ok=True, returncode=0, stdout="", stderr="", value=None, error=None
            )

        ref_result = send_result(
            self,
            RestRequest.of("GET", f"repos/{{owner}}/{{repo}}/git/refs/heads/{branch}"),
            json_output=True,
        )
        if not isinstance(ref_result, GitHubRunResult) or not ref_result.ok:
            error = (
                ref_result.error
                if isinstance(ref_result, GitHubRunResult)
                else f"unexpected response reading ref for branch {branch!r}"
            )
            return GitHubRunResult(
                ok=False,
                returncode=ref_result.returncode if isinstance(ref_result, GitHubRunResult) else 0,
                stdout="",
                stderr="",
                value=None,
                error=error or f"failed to read ref for branch {branch!r}",
            )
        ref_value = ref_result.value
        tip_sha = ref_value.get("object", {}).get("sha") if isinstance(ref_value, dict) else None
        if not isinstance(tip_sha, str) or not tip_sha:
            return GitHubRunResult(
                ok=False,
                returncode=0,
                stdout="",
                stderr="",
                value=None,
                error=f"could not determine tip SHA for branch {branch!r}",
            )

        commit_lookup = send_result(
            self,
            RestRequest.of("GET", f"repos/{{owner}}/{{repo}}/git/commits/{tip_sha}"),
            json_output=True,
        )
        if not isinstance(commit_lookup, GitHubRunResult) or not commit_lookup.ok:
            error = (
                commit_lookup.error
                if isinstance(commit_lookup, GitHubRunResult)
                else f"unexpected response reading commit {tip_sha!r}"
            )
            return GitHubRunResult(
                ok=False,
                returncode=(
                    commit_lookup.returncode if isinstance(commit_lookup, GitHubRunResult) else 0
                ),
                stdout="",
                stderr="",
                value=None,
                error=error or f"failed to read commit {tip_sha!r}",
            )
        commit_value = commit_lookup.value
        tree_sha = (
            commit_value.get("tree", {}).get("sha") if isinstance(commit_value, dict) else None
        )
        if not isinstance(tree_sha, str) or not tree_sha:
            return GitHubRunResult(
                ok=False,
                returncode=0,
                stdout="",
                stderr="",
                value=None,
                error=f"could not determine tree SHA for commit {tip_sha!r}",
            )

        new_commit = send_result(
            self,
            RestRequest.of(
                "POST",
                "repos/{owner}/{repo}/git/commits",
                body={
                    "message": "chore: retrigger CI (empty commit)",
                    "tree": tree_sha,
                    "parents": [tip_sha],
                },
            ),
            json_output=True,
        )
        if not isinstance(new_commit, GitHubRunResult) or not new_commit.ok:
            error = (
                new_commit.error
                if isinstance(new_commit, GitHubRunResult)
                else "unexpected response creating empty commit"
            )
            return GitHubRunResult(
                ok=False,
                returncode=new_commit.returncode if isinstance(new_commit, GitHubRunResult) else 0,
                stdout="",
                stderr="",
                value=None,
                error=error or "failed to create empty commit",
            )
        new_commit_value = new_commit.value
        new_sha = new_commit_value.get("sha") if isinstance(new_commit_value, dict) else None
        if not isinstance(new_sha, str) or not new_sha:
            return GitHubRunResult(
                ok=False,
                returncode=0,
                stdout="",
                stderr="",
                value=None,
                error="empty-commit creation response had no sha",
            )

        ref_update = send_result(
            self,
            RestRequest.of(
                "PATCH",
                f"repos/{{owner}}/{{repo}}/git/refs/heads/{branch}",
                body={"sha": new_sha},
            ),
            json_output=True,
        )
        if not isinstance(ref_update, GitHubRunResult):
            return GitHubRunResult(
                ok=False,
                returncode=0,
                stdout="",
                stderr="",
                value=None,
                error=f"unexpected response updating ref for branch {branch!r}",
            )
        return ref_update

    def branch_protection(self, base: str) -> dict[str, Any] | None:
        """Return branch protection settings for ``base``, or None on failure.

        Wraps ``gh api repos/{owner}/{repo}/branches/{base}/protection``.
        Returns ``None`` on any failure -- 404 (no protection configured),
        rate limit, transient network error, gh not installed. Errors are
        returned as values, never raised.

        Cached per orchestrator pass in ``_list_cache`` (issue #812): callers
        use this to derive base-freshness policy (``required_status_checks.
        strict``) instead of a hardcoded config constant, and need exactly
        one API call per base ref per pass, not one per PR. A failed read is
        cached as ``None`` too, so a 404/rate-limit does not turn into a
        per-PR retry storm within the same pass.

        Safety note for callers: ``None`` means "could not be read", not "no
        freshness required". Any caller gating a safety check on this value
        must fail closed on ``None``.
        """
        cache_key = ("branch_protection", base)
        if cache_key in self._list_cache:
            return self._list_cache[cache_key]
        result = send_result(
            self,
            RestRequest.of("GET", f"repos/{{owner}}/{{repo}}/branches/{base}/protection"),
            json_output=True,
        )
        value: dict[str, Any] | None = None
        if isinstance(result, GitHubRunResult):
            value = result.value if result.ok and isinstance(result.value, dict) else None
        elif isinstance(result, dict):
            value = result
        self._list_cache[cache_key] = value
        return value
