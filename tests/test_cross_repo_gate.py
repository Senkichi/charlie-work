"""Tests for the cross-repo pre-flight gate (issue #1010).

The gate extracts file-path references from an issue body and checks whether
any of them exist in the target repo. If the issue references file paths but
none exist in the repo, the gate returns ``passed=False`` — the issue should
be escalated to ``agent:human-needed`` with a ``cross_repo_target`` reason.
"""

from __future__ import annotations

from pathlib import Path

from charlie_work.cross_repo_gate import (
    CrossRepoGateResult,
    cross_repo_gate,
    extract_referenced_paths,
)


def test_no_file_paths_referenced_passes(tmp_path: Path) -> None:
    """An issue with no file-path references passes the gate."""
    result = cross_repo_gate("Fix the bug in the search function.", tmp_path)
    assert result.passed
    assert result.referenced_paths == ()
    assert result.missing_paths == ()


def test_referenced_path_exists_in_repo_passes(tmp_path: Path) -> None:
    """At least one referenced path existing in the repo passes the gate."""
    (tmp_path / "src" / "charlie_work").mkdir(parents=True)
    (tmp_path / "src" / "charlie_work" / "foo.py").write_text("# foo", encoding="utf-8")

    result = cross_repo_gate(
        "The bug is in `src/charlie_work/foo.py` — fix the search function.",
        tmp_path,
    )
    assert result.passed
    assert "src/charlie_work/foo.py" in result.referenced_paths
    assert result.missing_paths == ()


def test_all_referenced_paths_missing_blocks(tmp_path: Path) -> None:
    """When every referenced path is absent from the repo, the gate blocks.

    This is the core scenario from issue #1010: the issue references
    ``suite_coverage.py`` which lives in a sibling repo, not this one.
    """
    body = (
        "`suite_coverage.py` is at "
        "`C:/Users/senki/repos/ci_runners/src/ci_fleet/suite_coverage.py`; "
        "there is no `src/charlie_work/suite_coverage.py`."
    )
    result = cross_repo_gate(body, tmp_path)
    assert not result.passed
    assert len(result.referenced_paths) >= 1
    assert result.missing_paths == result.referenced_paths
    assert "cross_repo_target" in result.reason


def test_mixed_paths_some_exist_passes(tmp_path: Path) -> None:
    """If at least one referenced path exists, the gate passes even if others don't."""
    (tmp_path / "src" / "charlie_work").mkdir(parents=True)
    (tmp_path / "src" / "charlie_work" / "foo.py").write_text("# foo", encoding="utf-8")

    body = (
        "Fix `src/charlie_work/foo.py` and also check "
        "`src/ci_fleet/suite_coverage.py` for context."
    )
    result = cross_repo_gate(body, tmp_path)
    assert result.passed
    assert "src/charlie_work/foo.py" in result.referenced_paths


def test_absolute_path_inside_repo_passes(tmp_path: Path) -> None:
    """An absolute path that resolves inside the repo root passes."""
    (tmp_path / "src").mkdir(parents=True)
    (tmp_path / "src" / "app.py").write_text("# app", encoding="utf-8")

    abs_path = tmp_path / "src" / "app.py"
    body = f"Edit `{abs_path.as_posix()}` to fix the bug."
    result = cross_repo_gate(body, tmp_path)
    assert result.passed


def test_absolute_path_outside_repo_blocks(tmp_path: Path) -> None:
    """An absolute path to a different repo blocks the gate."""
    body = "The file is at `C:/Users/senki/repos/ci_runners/src/ci_fleet/suite_coverage.py`."
    result = cross_repo_gate(body, tmp_path)
    assert not result.passed


def test_urls_are_not_treated_as_file_paths(tmp_path: Path) -> None:
    """URLs should not be extracted as file paths."""
    body = "See https://example.com/docs/api.py for reference."
    result = cross_repo_gate(body, tmp_path)
    assert result.passed
    assert result.referenced_paths == ()


def test_bare_filename_without_path_separator_not_extracted(tmp_path: Path) -> None:
    """A bare filename like ``main.py`` without a path separator is not extracted.

    This avoids false positives from prose like "run main.py" or "see foo.py".
    """
    paths = extract_referenced_paths("Run main.py and check foo.py for output.")
    assert paths == []


def test_backtick_quoted_relative_path_extracted() -> None:
    """Backtick-quoted paths with separators and extensions are extracted."""
    paths = extract_referenced_paths("Edit `src/charlie_work/foo.py` now.")
    assert "src/charlie_work/foo.py" in paths


def test_backtick_quoted_absolute_path_extracted() -> None:
    """Backtick-quoted absolute paths are extracted."""
    paths = extract_referenced_paths(
        "The file is at `C:/Users/senki/repos/ci_runners/src/ci_fleet/suite_coverage.py`."
    )
    assert any("suite_coverage.py" in p for p in paths)


def test_paths_are_deduplicated() -> None:
    """The same path appearing multiple times is extracted only once."""
    paths = extract_referenced_paths("Edit `src/foo.py` then check `src/foo.py` again.")
    assert paths.count("src/foo.py") == 1


def test_gate_result_is_frozen() -> None:
    """CrossRepoGateResult is a frozen dataclass."""
    result = CrossRepoGateResult(
        passed=True,
        referenced_paths=(),
        missing_paths=(),
        reason="test",
    )
    try:
        result.passed = False  # type: ignore[misc]
    except AttributeError:
        pass
    else:
        raise AssertionError("CrossRepoGateResult should be frozen")


def test_issue_953_scenario_blocks(tmp_path: Path) -> None:
    """The exact scenario from issue #1010/#953: issue body references
    ``suite_coverage.py`` at a path in a sibling repo, with no matching file
    in the target repo."""
    body = (
        "But **#953's code does not live in this repo.** `suite_coverage.py` is at "
        "`C:/Users/senki/repos/ci_runners/src/ci_fleet/suite_coverage.py`; there is no "
        "`src/charlie_work/suite_coverage.py`. The worker, handed an isolated checkout "
        "of a repo that does not contain the file it was asked to change, went to "
        "`C:\\Users\\senki\\repos\\ci_runners` — the **shared main checkout** — and worked there."
    )
    result = cross_repo_gate(body, tmp_path)
    assert not result.passed
    assert "cross_repo_target" in result.reason
