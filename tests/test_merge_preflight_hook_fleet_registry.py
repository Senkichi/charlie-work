"""Tests for ``charlie_work.merge_preflight_hook`` (#894).

Fleet-registry shapes: ``_load_fleet_roots`` (missing / corrupt /
non-dict / well-formed), ``_repo_for_cwd`` containment matching, and
the ``_decide`` failure modes they feed -- unreadable registry fails
closed on both the Bash and MCP paths, a genuinely empty fleet passes
through.

Everything is mocked: no network, no real fleet.json reads, no subprocesses,
no LLM processes. Split verbatim out of ``tests/test_merge_preflight_hook.py``
for the Track-1 attachment-budget split (#1564).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from charlie_work import merge_preflight_hook as hook

# ---------------------------------------------------------------------------
# _load_fleet_roots / _repo_for_cwd
# ---------------------------------------------------------------------------


def test_load_fleet_roots_shapes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from charlie_work import layout

    registry = tmp_path / "fleet.json"
    monkeypatch.setattr(layout, "fleet_registry_path", lambda override=None: registry)

    # Missing file: no fleet configured -> {}.
    assert hook._load_fleet_roots() == {}

    # Corrupt file: read failure -> None (fail closed).
    registry.write_text("{not json", encoding="utf-8")
    assert hook._load_fleet_roots() is None

    # Non-dict JSON: also a read failure.
    registry.write_text("[]", encoding="utf-8")
    assert hook._load_fleet_roots() is None

    # Well-formed registry: owner/name lowercased -> repo_root Path.
    registry.write_text(
        '{"version": 1, "repos": {"Owner/Repo": {"repo_root": "C:/x/repo"}}}',
        encoding="utf-8",
    )
    roots = hook._load_fleet_roots()
    assert roots == {"owner/repo": Path("C:/x/repo")}


def test_repo_for_cwd_exact_match(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    roots = {"o/repo": root}
    assert hook._repo_for_cwd(roots, root) == "o/repo"


def test_repo_for_cwd_nested_match(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    nested = root / "sub" / "dir"
    nested.mkdir(parents=True)
    roots = {"o/repo": root}
    assert hook._repo_for_cwd(roots, nested) == "o/repo"


def test_repo_for_cwd_sibling_prefix_does_not_match(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    sibling = tmp_path / "repo-other"
    root.mkdir()
    sibling.mkdir()
    roots = {"o/repo": root}
    assert hook._repo_for_cwd(roots, sibling) is None


def test_repo_for_cwd_unrelated_cwd_returns_none(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    unrelated = tmp_path / "elsewhere"
    root.mkdir()
    unrelated.mkdir()
    roots = {"o/repo": root}
    assert hook._repo_for_cwd(roots, unrelated) is None


# ---------------------------------------------------------------------------
# _decide -- fleet-registry failure modes (fail closed on unreadable, pass
# through on genuinely empty)
# ---------------------------------------------------------------------------


def test_decide_bash_unreadable_registry_denies_mentioning_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # None = registry exists but cannot be read/parsed -> fail closed.
    monkeypatch.setattr(hook, "_load_fleet_roots", lambda: None)
    reason = hook._decide("Bash", {"command": "gh pr merge 1 --squash"}, tmp_path)
    assert reason is not None
    assert "registry" in reason.lower()


def test_decide_bash_unreadable_registry_denies_even_with_explicit_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An unreadable registry cannot confirm ANY repo is outside the fleet.
    monkeypatch.setattr(hook, "_load_fleet_roots", lambda: None)
    reason = hook._decide("Bash", {"command": "gh pr merge -R other/repo 1 --squash"}, tmp_path)
    assert reason is not None
    assert "registry" in reason.lower()


def test_decide_bash_genuinely_empty_fleet_passes_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # {} = no registry file / no repos: nothing fleet-managed, nothing to guard.
    monkeypatch.setattr(hook, "_load_fleet_roots", lambda: {})
    assert hook._decide("Bash", {"command": "gh pr merge 1 --squash"}, tmp_path) is None
    assert (
        hook._decide("Bash", {"command": "gh pr merge -R other/repo 1 --squash"}, tmp_path) is None
    )


def test_decide_mcp_unreadable_registry_denies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression for the review finding on PR #1195: the MCP path must honor
    # the same fail-closed contract as the Bash path when the registry is
    # unreadable — otherwise a corrupt fleet.json silently disables
    # enforcement for MCP-initiated merges.
    monkeypatch.setattr(hook, "_load_fleet_roots", lambda: None)
    called: list[Any] = []
    monkeypatch.setattr(hook, "_run_merge_check", lambda *a: called.append(a) or (True, ""))
    reason = hook._decide(
        "mcp__github__merge_pull_request",
        {"owner": "Senkichi", "repo": "charlie-work", "pullNumber": 5},
        tmp_path,
    )
    assert reason is not None
    assert "registry" in reason.lower()
    assert called == []


def test_decide_mcp_genuinely_empty_fleet_passes_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(hook, "_load_fleet_roots", lambda: {})
    reason = hook._decide(
        "mcp__github__merge_pull_request",
        {"owner": "o", "repo": "r", "pullNumber": 5},
        tmp_path,
    )
    assert reason is None
