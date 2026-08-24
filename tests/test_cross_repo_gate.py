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


def test_posix_style_absolute_path_outside_repo_blocks(tmp_path: Path) -> None:
    """A POSIX-style absolute path (no drive letter) also keeps escalating.

    ``Path(candidate).is_absolute()`` is platform-dependent: on Windows a
    POSIX-style absolute path like ``/home/user/other-repo/foo.py`` reports
    ``is_absolute() is False`` (no drive letter), which would otherwise let
    it fall through to the single-ambiguous-candidate abstain path instead
    of escalating. This is a regression test for that specific misfire.
    """
    body = "The bug is in /home/senki/other-repo/foo.py."
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


def test_jc_1688_domain_shaped_table_cell_extracts_nothing_and_passes(
    tmp_path: Path,
) -> None:
    """jc#1688 misfire: a scheme-less domain+path fragment in a markdown table
    cell (``pultegroupinc.com/.../default.aspx``) is not a file path — it
    must extract as zero candidates so the gate passes."""
    body = (
        "| Pulte Group | type textbox 'Search Jobs' @ pultegroupinc.com/.../default.aspx | **0** |"
    )
    result = cross_repo_gate(body, tmp_path)
    assert result.referenced_paths == ()
    assert result.passed


def test_cw_1062_single_non_repo_shaped_candidate_abstains(tmp_path: Path) -> None:
    """cw#1062 misfire: the sole extracted candidate is ``Scripts/charlie.exe``
    — a venv-relative path, not a reference to this repo's code (``Scripts``
    is not a real top-level directory here). Isolates fix 2: the fragment is
    not domain-shaped, so this exercises only the decision-layer rule."""
    body = (
        "Both hits are `self_deploy_failed` on `uv sync` failing to remove "
        "Scripts/charlie.exe (the known #854 condition)."
    )
    paths = extract_referenced_paths(body)
    assert paths == ["Scripts/charlie.exe"]

    result = cross_repo_gate(body, tmp_path)
    assert result.passed
    assert result.referenced_paths == ("Scripts/charlie.exe",)
    assert result.missing_paths == ("Scripts/charlie.exe",)


def test_domain_token_excluded_leaves_real_missing_path_to_escalate(
    tmp_path: Path,
) -> None:
    """Isolates fix 1: two candidates in the body, one domain-shaped (must be
    excluded by the regex fix) and one real repo-shaped missing path. The
    domain token is dropped during extraction; the surviving candidate is a
    single repo-shaped missing path, which still escalates — proving the
    regex fix does not depend on the decision-layer fix's abstain behavior.
    """
    (tmp_path / "src" / "charlie_work").mkdir(parents=True)

    body = (
        "See pultegroupinc.com/careers/default.aspx for context; the actual "
        "bug is in `src/charlie_work/nonexistent.py`."
    )
    paths = extract_referenced_paths(body)
    assert paths == ["src/charlie_work/nonexistent.py"]

    result = cross_repo_gate(body, tmp_path)
    assert not result.passed
    assert result.referenced_paths == ("src/charlie_work/nonexistent.py",)
    assert "cross_repo_target" in result.reason


def test_single_repo_shaped_relative_candidate_still_escalates(tmp_path: Path) -> None:
    """A single missing relative candidate whose first segment IS a real
    repo_root directory (``src``) still escalates — the decision-layer
    exception only abstains for non-repo-shaped candidates."""
    (tmp_path / "src" / "charlie_work").mkdir(parents=True)

    body = "The bug is in `src/charlie_work/nonexistent.py`."
    result = cross_repo_gate(body, tmp_path)
    assert not result.passed
    assert result.referenced_paths == ("src/charlie_work/nonexistent.py",)
    assert "cross_repo_target" in result.reason


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


def test_domain_token_adjacent_to_real_path_does_not_swallow_it(
    tmp_path: Path,
) -> None:
    """A domain-shaped token packed tightly against a repo-shaped path in a
    compact markdown table cell must not swallow its neighbor: the domain
    token is stripped, the real (missing) repo-shaped candidate survives,
    and the gate still escalates on it."""
    (tmp_path / "src" / "charlie_work").mkdir(parents=True)

    body = "| Pulte | pultegroupinc.com/careers/default.aspx|src/charlie_work/real.py |"
    result = cross_repo_gate(body, tmp_path)
    assert result.referenced_paths == ("src/charlie_work/real.py",)
    assert not result.passed
    assert "cross_repo_target" in result.reason


# --- issue #1343: templated runtime-state example paths ---------------------


def test_issue_1343_placeholder_numbered_segment_dropped_at_extraction() -> None:
    """A candidate with a ``pr-N`` placeholder segment is dropped during
    extraction — a literal ``N`` standing in for an unknown number can never
    name a real file, so it cannot be a genuine cross-repo reference."""
    paths = extract_referenced_paths(
        "The decision file is at `.var/charlie-work/prs/pr-N/review-decision.json`."
    )
    assert paths == []


def test_issue_1343_angle_bracket_placeholder_dropped_at_extraction() -> None:
    """A candidate with an angle-bracket placeholder segment (``<state-dir>``)
    is dropped during extraction — template text, not a file reference."""
    paths = extract_referenced_paths(
        "The decision file is at `<state-dir>/prs/pr-N/review-decision.json`."
    )
    assert paths == []


def test_issue_1343_issue_n_placeholder_dropped_at_extraction() -> None:
    """An ``issue-N`` placeholder segment is also dropped (same rule as
    ``pr-N``)."""
    paths = extract_referenced_paths(
        "See `.var/charlie-work/issues/issue-N/dispatch.json` for the shape."
    )
    assert paths == []


def test_issue_1340_templated_state_dir_path_does_not_escalate(tmp_path: Path) -> None:
    """The exact #1340 shape: the sole candidate is a templated runtime-state
    example path (``.var/charlie-work/prs/pr-N/review-decision.json``) whose
    first segment names the real, gitignored runtime state dir.

    The placeholder segment ``pr-N`` drops the candidate at extraction, so
    the gate sees no file paths and passes — no ``dispatch_cross_repo_escalated``.
    """
    (tmp_path / ".var" / "charlie-work" / "prs").mkdir(parents=True)
    (tmp_path / ".gitignore").write_text(".var/\n", encoding="utf-8")

    body = (
        "The decision-file divergence is documented at "
        "`.var/charlie-work/prs/pr-N/review-decision.json`."
    )
    result = cross_repo_gate(body, tmp_path)
    assert result.passed
    assert result.referenced_paths == ()
    assert result.missing_paths == ()


def test_issue_1343_gitignored_top_level_dir_not_repo_shaped(tmp_path: Path) -> None:
    """Isolates fix 1 (gitignore-derived exclusion) from fix 2 (placeholder
    rejection): the sole candidate keys on a real, gitignored top-level
    directory but has NO placeholder segment. It is missing, yet the gate
    abstains — a path whose first segment names an *ignored* directory is
    not a reference to this repo's tracked code."""
    (tmp_path / ".var" / "charlie-work").mkdir(parents=True)
    (tmp_path / ".gitignore").write_text(".var/\n", encoding="utf-8")

    body = "State is mirrored in `.var/charlie-work/state.json`."
    paths = extract_referenced_paths(body)
    assert paths == [".var/charlie-work/state.json"]

    result = cross_repo_gate(body, tmp_path)
    assert result.passed
    assert result.referenced_paths == (".var/charlie-work/state.json",)
    assert result.missing_paths == (".var/charlie-work/state.json",)


def test_issue_1343_tracked_top_level_dir_still_repo_shaped_with_gitignore(
    tmp_path: Path,
) -> None:
    """A gitignore that ignores ``.var/`` must not weaken escalation on a
    genuine tracked-dir reference: ``src/charlie_work/nonexistent.py`` (``src``
    is a real, non-ignored top-level dir) still escalates."""
    (tmp_path / "src" / "charlie_work").mkdir(parents=True)
    (tmp_path / ".gitignore").write_text(".var/\n.venv/\n", encoding="utf-8")

    body = "The bug is in `src/charlie_work/nonexistent.py`."
    result = cross_repo_gate(body, tmp_path)
    assert not result.passed
    assert result.referenced_paths == ("src/charlie_work/nonexistent.py",)
    assert "cross_repo_target" in result.reason


def test_issue_1343_no_gitignore_keeps_existing_repo_shape_behavior(
    tmp_path: Path,
) -> None:
    """Without a ``.gitignore`` the gitignore-derived exclusion is a no-op:
    a single repo-shaped missing candidate still escalates, preserving the
    pre-#1343 positive behavior."""
    (tmp_path / "src" / "charlie_work").mkdir(parents=True)

    body = "The bug is in `src/charlie_work/nonexistent.py`."
    result = cross_repo_gate(body, tmp_path)
    assert not result.passed
    assert "cross_repo_target" in result.reason


# --- issue #1391: glob metacharacters and launcher-owned worktree paths -------


def test_issue_1391_glob_metacharacter_candidate_dropped_at_extraction() -> None:
    """A backtick-quoted glob like ``src/charlie_work/*.py`` is a glob pattern,
    not a literal file path — no file literally named ``*.py`` exists, so the
    candidate is always "missing" and would false-positive the gate. It is
    dropped during extraction (issue #1391, cw #1059)."""
    paths = extract_referenced_paths("The bug is in `src/charlie_work/*.py`.")
    assert paths == []


def test_issue_1391_question_mark_glob_dropped_at_extraction() -> None:
    """A ``?`` glob metacharacter is also dropped — same reasoning as ``*``."""
    paths = extract_referenced_paths("Check `src/charlie_work/foo?.py`.")
    assert paths == []


def test_issue_1391_bracket_glob_dropped_at_extraction() -> None:
    """A ``[...]`` character class glob is also dropped."""
    paths = extract_referenced_paths("Check `src/charlie_work/foo[0-9].py`.")
    assert paths == []


def test_issue_1391_glob_only_body_passes_gate(tmp_path: Path) -> None:
    """When the only candidate is a glob, extraction returns nothing and the
    gate passes — a glob is not evidence of a cross-repo target."""
    (tmp_path / "src" / "charlie_work").mkdir(parents=True)

    body = "The bug is in `src/charlie_work/*.py`."
    result = cross_repo_gate(body, tmp_path)
    assert result.passed
    assert result.referenced_paths == ()


def test_issue_1391_glob_alongside_real_path_keeps_real_path() -> None:
    """A glob candidate is dropped but a real path candidate in the same body
    survives — the glob filter does not suppress genuine evidence."""
    paths = extract_referenced_paths(
        "Fix `src/charlie_work/*.py` and specifically `src/charlie_work/foo.py`."
    )
    assert "src/charlie_work/foo.py" in paths
    assert all("*" not in p for p in paths)


def test_issue_1391_devin_dir_candidate_dropped_at_extraction() -> None:
    """A path under ``.devin/`` lives only inside agent worktrees, not in the
    repo tree — it is dropped during extraction so it cannot fire the gate
    (issue #1391)."""
    paths = extract_referenced_paths(
        "The shim writes `.devin/hooks.v1.json` and `.devin/skills/worker.md`."
    )
    assert paths == []


def test_issue_1391_git_worktree_dir_candidate_dropped_at_extraction() -> None:
    """A path under ``.git_worktree_dir/`` is launcher-owned and dropped."""
    paths = extract_referenced_paths("See `.git_worktree_dir/cache.json`.")
    assert paths == []


def test_issue_1391_devin_only_body_passes_gate(tmp_path: Path) -> None:
    """When every candidate is under ``.devin/``, extraction returns nothing
    and the gate passes — launcher-owned paths are not cross-repo evidence."""
    body = (
        "Shim residue: `.devin/AGENTS.md`, `.devin/hooks.v1.json`, "
        "`.devin/skills/foo.py`, `.devin/worker.md`."
    )
    result = cross_repo_gate(body, tmp_path)
    assert result.passed
    assert result.referenced_paths == ()


def test_issue_1391_devin_alongside_real_missing_path_keeps_real_path(
    tmp_path: Path,
) -> None:
    """A ``.devin/`` candidate is dropped but a real missing repo-shaped path
    in the same body survives and still escalates — the launcher-owned filter
    does not suppress genuine cross-repo evidence."""
    (tmp_path / "src" / "charlie_work").mkdir(parents=True)

    body = (
        "Shim wrote `.devin/hooks.v1.json`; the real bug is in `src/charlie_work/nonexistent.py`."
    )
    paths = extract_referenced_paths(body)
    assert paths == ["src/charlie_work/nonexistent.py"]

    result = cross_repo_gate(body, tmp_path)
    assert not result.passed
    assert "cross_repo_target" in result.reason
