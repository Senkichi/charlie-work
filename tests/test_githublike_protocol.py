"""Regression tests for the GitHubLike structural protocol surface."""

from __future__ import annotations

import inspect

from charlie_work.github import GitHubLike


def test_githublike_protocol_declares_branch_protection() -> None:
    """GitHubLike must declare ``branch_protection`` so ``GitHubLike``-typed
    ``self.gh`` can call it unguardedly without a pyright error.

    This is the review finding for PR #759 / issue #593.
    """
    assert "branch_protection" in GitHubLike.__dict__, "GitHubLike is missing branch_protection"
    sig = inspect.signature(GitHubLike.branch_protection)
    assert list(sig.parameters) == ["self", "base"]
    assert sig.return_annotation == "dict[str, Any] | None"


def test_githublike_protocol_declares_pr_ready() -> None:
    """GitHubLike must declare ``pr_ready`` so ``GitHubLike``-typed ``self.gh``
    can call it unguardedly without a pyright error.

    Same structural-omission class as ``branch_protection``.
    """
    assert "pr_ready" in GitHubLike.__dict__, "GitHubLike is missing pr_ready"
    sig = inspect.signature(GitHubLike.pr_ready)
    assert list(sig.parameters) == ["self", "number"]
    assert sig.return_annotation == "GitHubRunResult"
