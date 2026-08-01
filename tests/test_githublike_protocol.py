"""Regression tests for the GitHubLike structural protocol surface."""

from __future__ import annotations

import inspect
from pathlib import Path

from charlie_work.github import GitHub, GitHubLike


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


def _compatible_signature(proto_sig: inspect.Signature, concrete_sig: inspect.Signature) -> None:
    proto_params = [p for n, p in proto_sig.parameters.items() if n != "self"]
    concrete_params = [p for n, p in concrete_sig.parameters.items() if n != "self"]
    assert len(proto_params) == len(concrete_params), (
        f"parameter count differs: {proto_sig} vs {concrete_sig}"
    )
    for proto_param, concrete_param in zip(proto_params, concrete_params):
        assert proto_param.name == concrete_param.name, (
            f"parameter name differs for {proto_sig}: "
            f"{proto_param.name!r} vs {concrete_param.name!r}"
        )
        assert proto_param.kind == concrete_param.kind, (
            f"parameter kind differs for {proto_param.name}: "
            f"{proto_param.kind} vs {concrete_param.kind}"
        )
    if proto_sig.return_annotation is not inspect.Signature.empty:
        assert concrete_sig.return_annotation is not inspect.Signature.empty, (
            f"concrete missing return annotation for {proto_sig}"
        )
        assert proto_sig.return_annotation == concrete_sig.return_annotation, (
            f"return annotation differs: {proto_sig.return_annotation} "
            f"vs {concrete_sig.return_annotation}"
        )


def test_github_satisfies_githublike_protocol(tmp_path: Path) -> None:
    """The concrete GitHub class must implement the full GitHubLike surface.

    Catches protocol drift before it breaks GitHubLike-typed callers.
    """
    gh = GitHub(tmp_path)
    assert isinstance(gh, GitHubLike), "GitHub does not satisfy GitHubLike at runtime"

    for name in sorted(GitHubLike.__protocol_attrs__):
        if name == "dry_run":
            assert hasattr(gh, name), f"GitHub is missing attribute {name}"
            continue
        concrete = getattr(gh, name, None)
        assert callable(concrete), f"GitHub is missing or non-callable method {name}"
        proto_sig = inspect.signature(getattr(GitHubLike, name))
        concrete_sig = inspect.signature(getattr(GitHub, name))
        _compatible_signature(proto_sig, concrete_sig)
