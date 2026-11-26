"""``list_worktrees``: porcelain parsing, malformed/unknown entries and consumer-path
safety."""

from __future__ import annotations

from pathlib import Path

from _worktree_fixtures import _init_repo
from charlie_work.worktree import create_worktree, list_worktrees, remove_worktree


def test_list_worktrees_parses_porcelain_output(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    info = create_worktree(repo_root, "agent/issue-7-list", base_ref="HEAD")

    worktrees = list_worktrees(repo_root)

    assert len(worktrees) == 2  # main checkout + the new worktree
    paths = [entry["worktree"] for entry in worktrees]
    assert repo_root.resolve() in [p.resolve() for p in paths]
    assert info.path.resolve() in [p.resolve() for p in paths]
    branches = [entry.get("branch") for entry in worktrees]
    assert any(branch and branch.endswith("agent/issue-7-list") for branch in branches)


def test_list_worktrees_returns_empty_list_for_non_git_dir(tmp_path: Path) -> None:
    not_a_repo = tmp_path / "not-a-repo"
    not_a_repo.mkdir()

    assert list_worktrees(not_a_repo) == []


def test_list_worktrees_porcelain_parser_handles_flag_lines(tmp_path: Path) -> None:
    """Porcelain parser should handle flag lines (bare, detached, prunable) correctly."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    # Create a worktree to ensure we have porcelain output to parse
    info = create_worktree(repo_root, "agent/issue-131-flags", base_ref="HEAD")

    # The parser should handle flag lines without crashing
    worktrees = list_worktrees(repo_root)

    # Should have at least 2 worktrees (main + the new one)
    assert len(worktrees) >= 2

    # Each worktree entry should have a worktree key with a Path value
    for wt in worktrees:
        assert "worktree" in wt
        assert isinstance(wt["worktree"], Path)

    # Flag keys like "bare", "detached" should be present as True if applicable
    # (main worktree is typically not bare/detached, but the parser should handle them)
    for wt in worktrees:
        if "bare" in wt:
            assert wt["bare"] is True
        if "detached" in wt:
            assert wt["detached"] is True

    # Clean up
    remove_worktree(repo_root, info.path)


def test_list_worktrees_porcelain_parser_handles_malformed_worktree_line(tmp_path: Path) -> None:
    """Porcelain parser should drop malformed worktree entries entirely.

    When a worktree line is malformed (bare "worktree" with no path), the entire
    entry is dropped from the result. Sibling valid entries are unaffected.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    # Monkeypatch run_captured to return malformed porcelain output
    import charlie_work.worktree

    original_run_captured = charlie_work.worktree.run_captured

    def mock_run_captured(*args: object, **kwargs: object) -> object:
        # If this is a git worktree list --porcelain call, return malformed output
        if isinstance(args[0], list) and "worktree" in args[0] and "list" in args[0]:
            from charlie_work.subprocess_runner import RunResult

            # Malformed output: a bare "worktree" line with no path (flag-style)
            malformed_output = """worktree
bare
HEAD abc123
branch refs/heads/main

worktree /path/to/valid
HEAD def456
branch refs/heads/feature
"""
            return RunResult(
                returncode=0,
                stdout=malformed_output,
                stderr="",
                error=None,
            )
        return original_run_captured(*args, **kwargs)

    charlie_work.worktree.run_captured = mock_run_captured

    try:
        # The parser should not crash on malformed input
        worktrees = list_worktrees(repo_root)

        # Should parse only the valid entry; the malformed one is dropped entirely
        assert len(worktrees) == 1
        # The valid entry should be present
        assert worktrees[0]["worktree"] == Path("/path/to/valid")
        assert worktrees[0]["branch"] == "refs/heads/feature"
    finally:
        charlie_work.worktree.run_captured = original_run_captured


def test_list_worktrees_porcelain_parser_drops_unknown_flag_keys(tmp_path: Path) -> None:
    """Porcelain parser should drop entries with unknown space-less keys.

    Unknown flag keys (not in KNOWN_FLAG_KEYS) mark the entire entry as malformed
    and cause it to be dropped. Known flag lines (bare, detached) parse as True.
    Valued forms (prunable <reason>, locked <reason>) parse as str.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    # Monkeypatch run_captured to return porcelain output with unknown keys
    import charlie_work.worktree

    original_run_captured = charlie_work.worktree.run_captured

    def mock_run_captured(*args: object, **kwargs: object) -> object:
        # If this is a git worktree list --porcelain call, return output with unknown keys
        if isinstance(args[0], list) and "worktree" in args[0] and "list" in args[0]:
            from charlie_work.subprocess_runner import RunResult

            # Output with unknown flag key "garbage" and valued forms
            malformed_output = """worktree /path/to/valid1
HEAD abc123
branch refs/heads/feature1
bare

worktree /path/to/valid2
HEAD def456
branch refs/heads/feature2
garbage

worktree /path/to/valid3
HEAD ghi789
branch refs/heads/feature3
prunable some reason

worktree /path/to/valid4
HEAD jkl012
branch refs/heads/feature4
locked another reason
"""
            return RunResult(
                returncode=0,
                stdout=malformed_output,
                stderr="",
                error=None,
            )
        return original_run_captured(*args, **kwargs)

    charlie_work.worktree.run_captured = mock_run_captured

    try:
        worktrees = list_worktrees(repo_root)

        # Should parse only entries 1, 3, 4; entry 2 with unknown "garbage" key is dropped
        assert len(worktrees) == 3
        # Entry 1: valid with known flag "bare"
        assert worktrees[0]["worktree"] == Path("/path/to/valid1")
        assert worktrees[0]["bare"] is True
        # Entry 3: valid with valued "prunable" (not in KNOWN_FLAG_KEYS, but has a value)
        assert worktrees[1]["worktree"] == Path("/path/to/valid3")
        assert worktrees[1]["prunable"] == "some reason"
        # Entry 4: valid with valued "locked" (not in KNOWN_FLAG_KEYS, but has a value)
        assert worktrees[2]["worktree"] == Path("/path/to/valid4")
        assert worktrees[2]["locked"] == "another reason"
    finally:
        charlie_work.worktree.run_captured = original_run_captured


def test_list_worktrees_consumer_path_safe_with_malformed_entries(tmp_path: Path) -> None:
    """Consumer-path test: fresh-dispatch lookup with malformed porcelain output.

    Drives the real fresh-dispatch/existing-worktree lookup code (around L529 in
    worktree.py) with porcelain output containing one malformed entry and one valid
    entry. Verifies no KeyError occurs and the valid entry is triaged correctly.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    # Create a worktree to establish a valid path
    branch_name = "agent/issue-131-consumer"
    info1 = create_worktree(repo_root, branch_name, base_ref="HEAD")

    # Monkeypatch run_captured to return porcelain output with a malformed entry
    import charlie_work.worktree

    original_run_captured = charlie_work.worktree.run_captured

    def mock_run_captured(*args: object, **kwargs: object) -> object:
        # If this is a git worktree list --porcelain call, return output with malformed entry
        if isinstance(args[0], list) and "worktree" in args[0] and "list" in args[0]:
            from charlie_work.subprocess_runner import RunResult

            # Output with a malformed entry (bare "worktree") and the real valid entry
            # We need to include the actual worktree path from info1
            malformed_output = f"""worktree
bare
HEAD abc123
branch refs/heads/malformed

worktree {info1.path}
HEAD def456
branch refs/heads/{branch_name}
"""
            return RunResult(
                returncode=0,
                stdout=malformed_output,
                stderr="",
                error=None,
            )
        return original_run_captured(*args, **kwargs)

    charlie_work.worktree.run_captured = mock_run_captured

    try:
        # Drive the actual consumer code: call create_worktree again (fresh-dispatch path)
        # This will call list_worktrees internally and do the real lookup around L529
        # The existing worktree should be found and triaged correctly despite the malformed entry
        info2 = create_worktree(repo_root, branch_name, base_ref="HEAD")

        # Should succeed without KeyError (the malformed entry is dropped by the parser)
        assert info2 is not None
        # The worktree path should be the same (existing worktree was found)
        assert info2.path == info1.path
        # The worktree should still exist
        assert info2.path.exists()
    finally:
        charlie_work.worktree.run_captured = original_run_captured
        remove_worktree(repo_root, info1.path)
