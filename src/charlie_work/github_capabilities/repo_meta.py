"""Repo-metadata capability: identity/compare/commit lookups (Track 2, #1585).

Cluster D of the design doc's capability segmentation (Section 3.1):
``name_with_owner``, ``compare``, ``compare_diff``, ``commit``,
``invalidate_list_cache``.

``invalidate_list_cache`` is an ambiguity call (Section 3.1): it is cache
lifecycle, pinned here because RepoMeta is the smallest write-side cluster.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

# ``ci_fleet.github.GitHubError`` is imported directly here, not re-derived
# through ``charlie_work.github`` or ``_base.py``: ``name_with_owner`` (moved
# below) raises it, and identity matters -- ``github.py``'s own load-bearing
# comment on its ``GitHubError`` re-export (Track 2, issue #1585) explains why
# a local re-declaration would be a structurally-identical but *unrelated*
# exception type that no existing ``except GitHubError`` handler would catch.
# ``ci_fleet.github`` has no dependency on ``charlie_work``, so importing it
# directly here carries no circular-import risk (unlike ``charlie_work.github``
# itself, which imports ``github_capabilities`` before its own definitions are
# ready).
from ci_fleet.github import GitHubError

# ``GitHubRunResult`` lives in ``_base.py``, not ``charlie_work.github`` (Track
# 2, issue #1588; design doc Section 5, L04) -- see ``_base.py`` for the full
# circular-import rationale. Three of the five members moved below
# (``commit``, ``compare``, ``compare_diff``) perform a real runtime
# ``isinstance(result, GitHubRunResult)`` check, so (unlike before L05) this
# must be a normal top-level import, not the ``TYPE_CHECKING``-only one this
# module used while RepoMeta was still empty in L01 -- mirroring
# ``checks.py``'s L04 promotion of the same import for the same reason.
from ._base import CapabilityCollaborator, GitHubRunResult

# Narrow field list for ``gh repo view --json nameWithOwner`` (issue #1609):
# ``name_with_owner`` needs only the repository's ``nameWithOwner`` scalar, so
# it must not go through a broader field list. Kept as a module-level constant
# rather than an inline literal so the field-list lint
# (tests/test_doctor.py::test_gh_field_lists_use_constants_no_inline_literals)
# covers the single-positional-list ``self.run([...], json_output=True)`` call
# shape this body uses -- the matcher's third branch, added by #1609, inspects
# that shape; before the fix this call was skipped entirely because it lived in
# ``github.py`` (the file the lint skips by design). It moved here with the
# RepoMeta body (Track 2, issue #1589, L05), exposing it to the lint.
REPO_NAME_WITH_OWNER_FIELDS = "nameWithOwner"


@runtime_checkable
class RepoMetaLike(Protocol):
    """Structural interface for repository-metadata operations."""

    def name_with_owner(self) -> str: ...

    def compare(self, base: str, head: str) -> dict[str, Any] | None: ...

    def compare_diff(self, base: str, head: str) -> str | None: ...

    def commit(self, sha: str) -> GitHubRunResult: ...

    def invalidate_list_cache(self) -> None: ...


class RepoMeta(CapabilityCollaborator):
    """Repository-metadata capability collaborator.

    Moved from ``GitHub`` verbatim (Track 2, issue #1589; design doc Section
    5, L05). Bodies still say ``self.run(...)``, which resolves through
    ``CapabilityCollaborator.__getattr__`` to the owner's ``run`` (design doc
    Section 3.3). ``invalidate_list_cache`` instead touches
    ``self._list_cache`` directly -- also owner-forwarded, but the shared
    ``_list_cache`` dict itself STAYS on the owner (design doc Section 3.4):
    no cache ownership moves, so the in-place ``.clear()`` is visible to every
    other collaborator and the owner alike.

    Two of the five also reference module-level bare globals that could not
    move with them: ``commit``/``compare``/``compare_diff`` use
    ``GitHubRunResult`` (relocated to ``_base.py`` in L04 and imported from
    there, not re-derived from ``github.py``, to avoid a circular import --
    see this module's own import block and ``_base.py`` for the full
    rationale), and ``name_with_owner`` raises ``GitHubError`` (imported
    directly from ``ci_fleet.github``, the same external, identity-sensitive
    source ``github.py`` itself re-exports from -- see this module's import
    block). Design doc Section 3.3 covers only ``self.<attr>`` forwarding, not
    bare-global runtime symbols in moved bodies; this is the same disclosed
    design-gap resolution L04 hit for ``GitHubRunResult``.
    """

    def invalidate_list_cache(self) -> None:
        """Drop cached list results so the next call refetches from GitHub.

        Called at the start of every orchestrator pass (``loop()``); the
        cache dedupes list calls within one pass, never across passes.
        """
        self._list_cache.clear()

    def commit(self, sha: str) -> GitHubRunResult:
        """Fetch a single commit's metadata by SHA.

        Wraps ``gh api repos/{owner}/{repo}/commits/{sha}``. Returns a
        ``GitHubRunResult`` whose ``value`` is the parsed JSON response
        (including ``parents`` and ``committer``/``commit.committer``) on
        success, or ``None`` with ``error`` set on failure. Errors are
        returned as values, never raised.

        Callers that only want the dict can use
        ``result.value if result.ok and isinstance(result.value, dict) else None``;
        callers that need the failure reason (e.g. for event payloads, issue
        #1140) read ``result.error``. Returning the full ``GitHubRunResult``
        rather than collapsing to ``None`` preserves the transport/API error
        (TLS blip vs rate limit vs auth vs 404) at the boundary that most
        needs it, consistent with this repo's errors-as-values invariant.
        """
        result = self.run(
            ["api", f"repos/{{owner}}/{{repo}}/commits/{sha}"],
            json_output=True,
            allow_failure=True,
        )
        if isinstance(result, GitHubRunResult):
            return result
        # Dry-run short-circuit or an unexpected double that returned a raw
        # value instead of a GitHubRunResult. Normalize so the contract is
        # uniform -- callers never need to branch on the return type.
        if isinstance(result, dict):
            return GitHubRunResult(ok=True, returncode=0, stdout="", stderr="", value=result)
        return GitHubRunResult(
            ok=False,
            returncode=0,
            stdout="",
            stderr="",
            value=None,
            error=f"unexpected response from gh.run: {type(result).__name__}",
        )

    def compare(self, base: str, head: str) -> dict[str, Any] | None:
        """Compare two commits and return the comparison metadata.

        Wraps ``gh api repos/{owner}/{repo}/compare/{base}...{head}``. Returns
        the parsed JSON response, including ``base_commit`` and
        ``merge_base_commit``, or ``None`` on failure. Errors are returned as
        values, never raised.
        """
        result = self.run(
            ["api", f"repos/{{owner}}/{{repo}}/compare/{base}...{head}"],
            json_output=True,
            allow_failure=True,
        )
        if isinstance(result, GitHubRunResult):
            return result.value if result.ok and isinstance(result.value, dict) else None
        return result if isinstance(result, dict) else None

    def compare_diff(self, base: str, head: str) -> str | None:
        """Return the plain unified-diff text between two commits (three-dot compare).

        Wraps the same ``gh api repos/{owner}/{repo}/compare/{base}...{head}``
        endpoint as :meth:`compare`, but requests the ``application/vnd.github.
        v3.diff`` media type so the response body is a ready-to-write unified
        diff (like :meth:`pr_diff`) instead of JSON compare metadata. Tolerates
        a rebased/diverged/GC'd ``base`` the same way GitHub's three-dot
        compare does. Returns ``None`` on any failure (404, API error, gh not
        installed) — errors are returned as values, never raised.
        """
        result = self.run(
            [
                "api",
                f"repos/{{owner}}/{{repo}}/compare/{base}...{head}",
                "-H",
                "Accept: application/vnd.github.v3.diff",
            ],
            allow_failure=True,
        )
        if isinstance(result, GitHubRunResult):
            return result.value if result.ok and isinstance(result.value, str) else None
        return result if isinstance(result, str) else None

    def name_with_owner(self) -> str:
        """Return the repository's nameWithOwner (e.g., "owner/repo").

        Uses `gh repo view --json nameWithOwner`. Raises GitHubError on failure
        (offline, not a GitHub repo, gh missing, etc.).

        Returns:
            The repository's nameWithOwner string.
        """
        result = self.run(
            ["repo", "view", "--json", REPO_NAME_WITH_OWNER_FIELDS], json_output=True
        )
        if not isinstance(result, dict):
            raise GitHubError("Expected dict from gh repo view")
        name_with_owner = result.get("nameWithOwner")
        if not isinstance(name_with_owner, str):
            raise GitHubError("Expected nameWithOwner string in gh repo view output")
        return name_with_owner
