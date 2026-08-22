"""Tests for the cross-repo scope gate and fleet-registry helper (issue #1244).

The scope gate checks an issue's *title* for a ``<repo-name>:`` prefix that
names a managed repo other than the dispatching one — the clearest signal
that the issue's deliverables live in that repo, not this one.  The
managed-repo set is derived from the fleet registry, never a hardcoded list.
"""

from __future__ import annotations

import json
from pathlib import Path

from charlie_work.cross_repo_gate import (
    CrossRepoGateResult,
    cross_repo_scope_gate,
)
from charlie_work.fleet_registry import managed_repo_names


# ---------------------------------------------------------------------------
# cross_repo_scope_gate
# ---------------------------------------------------------------------------


def test_empty_managed_repos_passes() -> None:
    """No fleet registry / single-repo deployment → nothing to check, passes."""
    result = cross_repo_scope_gate("job-cannon: fix the docs", "", "charlie-work", frozenset())
    assert result.passed
    assert "no other managed repos" in result.reason


def test_title_names_other_managed_repo_blocks() -> None:
    """The #709 pattern: title starts with another managed repo's name."""
    result = cross_repo_scope_gate(
        "job-cannon: docs/devin-orchestration/ ... stale",
        "Body text about job-cannon files.",
        "charlie-work",
        frozenset({"charlie-work", "job-cannon"}),
    )
    assert not result.passed
    assert "cross_repo_scope" in result.reason
    assert "job-cannon" in result.reason
    assert "charlie-work" in result.reason


def test_title_names_dispatching_repo_passes() -> None:
    """An issue whose title starts with the dispatching repo's own name passes."""
    result = cross_repo_scope_gate(
        "charlie-work: fix the dispatch logic",
        "",
        "charlie-work",
        frozenset({"charlie-work", "job-cannon"}),
    )
    assert result.passed


def test_title_does_not_name_any_repo_passes() -> None:
    """An issue with a generic title passes even when other repos exist."""
    result = cross_repo_scope_gate(
        "Fix the search function in the worker",
        "",
        "charlie-work",
        frozenset({"charlie-work", "job-cannon"}),
    )
    assert result.passed


def test_title_mentions_repo_not_as_prefix_passes() -> None:
    """A repo name appearing in the title but not as a prefix passes.

    ``coordinate with job-cannon on this`` mentions the repo but is not
    evidence of a cross-repo scope — the issue's deliverables may still
    live in the dispatching repo.
    """
    result = cross_repo_scope_gate(
        "Coordinate with job-cannon on the shared API",
        "",
        "charlie-work",
        frozenset({"charlie-work", "job-cannon"}),
    )
    assert result.passed


def test_case_insensitive_title_match() -> None:
    """Title-prefix matching is case-insensitive."""
    result = cross_repo_scope_gate(
        "Job-Cannon: fix the docs",
        "",
        "charlie-work",
        frozenset({"charlie-work", "job-cannon"}),
    )
    assert not result.passed


def test_leading_whitespace_in_title_handled() -> None:
    """Leading whitespace before the repo-name prefix does not defeat the check."""
    result = cross_repo_scope_gate(
        "  job-cannon: fix the docs",
        "",
        "charlie-work",
        frozenset({"charlie-work", "job-cannon"}),
    )
    assert not result.passed


def test_dispatching_repo_not_in_managed_set_blocks() -> None:
    """When the dispatching repo is not in the managed set, other repos still block."""
    result = cross_repo_scope_gate(
        "job-cannon: fix the docs",
        "",
        "some-other-repo",
        frozenset({"job-cannon"}),
    )
    assert not result.passed


def test_multiple_other_repos_one_matches_blocks() -> None:
    """When multiple other repos exist, matching any one blocks."""
    result = cross_repo_scope_gate(
        "ci-fleet: fix the runner allocation",
        "",
        "charlie-work",
        frozenset({"charlie-work", "job-cannon", "ci-fleet"}),
    )
    assert not result.passed
    assert "ci-fleet" in result.reason


def test_result_type_is_cross_repo_gate_result() -> None:
    """The scope gate returns the same result type as the file-path gate."""
    result = cross_repo_scope_gate(
        "job-cannon: fix", "", "charlie-work", frozenset({"charlie-work", "job-cannon"})
    )
    assert isinstance(result, CrossRepoGateResult)


# ---------------------------------------------------------------------------
# managed_repo_names
# ---------------------------------------------------------------------------


def _write_fleet_registry(fleet_dir: Path, repos: dict[str, dict[str, str]]) -> None:
    """Write a fleet.json with the given repo entries."""
    fleet_dir.mkdir(parents=True, exist_ok=True)
    fleet_json = fleet_dir / "fleet.json"
    data = {"version": 1, "repos": repos}
    fleet_json.write_text(json.dumps(data), encoding="utf-8")


def test_managed_repo_names_extracts_repo_segments(tmp_path: Path) -> None:
    """Repo names are the last segment of owner/repo keys."""
    _write_fleet_registry(
        tmp_path,
        {
            "Senkichi/charlie-work": {"repo_root": "/tmp/cw"},
            "Senkichi/job-cannon": {"repo_root": "/tmp/jc"},
        },
    )
    names = managed_repo_names(str(tmp_path))
    assert names == frozenset({"charlie-work", "job-cannon"})


def test_managed_repo_names_empty_registry(tmp_path: Path) -> None:
    """A missing fleet registry returns an empty set."""
    names = managed_repo_names(str(tmp_path))
    assert names == frozenset()


def test_managed_repo_names_single_repo(tmp_path: Path) -> None:
    """A single-repo fleet returns just that repo's name."""
    _write_fleet_registry(
        tmp_path,
        {"Senkichi/charlie-work": {"repo_root": "/tmp/cw"}},
    )
    names = managed_repo_names(str(tmp_path))
    assert names == frozenset({"charlie-work"})


def test_managed_repo_names_corrupt_registry_returns_empty(tmp_path: Path) -> None:
    """A corrupt fleet.json returns an empty set, not an exception."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "fleet.json").write_text("not valid json", encoding="utf-8")
    names = managed_repo_names(str(tmp_path))
    assert names == frozenset()
