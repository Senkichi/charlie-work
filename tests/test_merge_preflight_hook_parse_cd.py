"""Tests for ``charlie_work.merge_preflight_hook`` (#894).

``_parse_gh_merge_targets`` ``cd`` tracking (#1252 defect 1 and the
review finding): the merge's effective cwd follows ``cd`` in command
position, but not a ``cd`` that is subshell-scoped by a pipe,
backgrounding, or parentheses.

Everything is mocked: no network, no real fleet.json reads, no subprocesses,
no LLM processes. Split verbatim out of ``tests/test_merge_preflight_hook.py``
for the Track-1 attachment-budget split (#1564).
"""

from __future__ import annotations

from pathlib import Path

from charlie_work import merge_preflight_hook as hook

# ---------------------------------------------------------------------------
# #1252 defect 1: repo resolved from command's effective cwd, not hook cwd
# ---------------------------------------------------------------------------


def test_parse_cd_absolute_path_sets_cd_cwd(tmp_path: Path) -> None:
    target_dir = tmp_path / "job-cannon"
    target_dir.mkdir()
    targets = hook._parse_gh_merge_targets(
        f"cd {target_dir.as_posix()} && gh pr merge 1679", tmp_path
    )
    assert len(targets) == 1
    assert targets[0]["pr"] == 1679
    assert targets[0]["cd_cwd"] == target_dir.resolve()


def test_parse_cd_relative_path_resolved_against_cwd(tmp_path: Path) -> None:
    (tmp_path / "sub" / "repo").mkdir(parents=True)
    targets = hook._parse_gh_merge_targets("cd sub/repo && gh pr merge 5", tmp_path)
    assert len(targets) == 1
    assert targets[0]["cd_cwd"] == (tmp_path / "sub" / "repo").resolve()


def test_parse_no_cd_yields_cd_cwd_none(tmp_path: Path) -> None:
    targets = hook._parse_gh_merge_targets("gh pr merge 5", tmp_path)
    assert targets == [{"pr": 5, "repo": None, "cd_cwd": None}]


def test_parse_cd_in_subshell_does_not_leak(tmp_path: Path) -> None:
    # A cd inside (...) must not affect commands after the subshell.
    target_dir = tmp_path / "inner"
    target_dir.mkdir()
    targets = hook._parse_gh_merge_targets(
        f"(cd {target_dir.as_posix()} && ls); gh pr merge 5", tmp_path
    )
    assert len(targets) == 1
    assert targets[0]["cd_cwd"] is None


def test_parse_cd_in_subshell_applies_inside(tmp_path: Path) -> None:
    # A cd inside (...) applies to merge invocations inside the subshell.
    target_dir = tmp_path / "inner"
    target_dir.mkdir()
    targets = hook._parse_gh_merge_targets(
        f"(cd {target_dir.as_posix()} && gh pr merge 5)", tmp_path
    )
    assert len(targets) == 1
    assert targets[0]["cd_cwd"] == target_dir.resolve()


def test_parse_cd_not_in_command_position_ignored(tmp_path: Path) -> None:
    # ``echo cd /path`` must not update effective cwd — cd is an argument.
    targets = hook._parse_gh_merge_targets(
        f"echo cd {tmp_path.as_posix()} && gh pr merge 5", tmp_path
    )
    assert len(targets) == 1
    assert targets[0]["cd_cwd"] is None


def test_parse_multiple_cds_track_effective_cwd(tmp_path: Path) -> None:
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    targets = hook._parse_gh_merge_targets(
        f"cd {dir_a.as_posix()} && gh pr merge 1; cd {dir_b.as_posix()} && gh pr merge 2",
        tmp_path,
    )
    assert len(targets) == 2
    assert targets[0]["cd_cwd"] == dir_a.resolve()
    assert targets[1]["cd_cwd"] == dir_b.resolve()


# ---------------------------------------------------------------------------
# #1252 review finding: cd in a pipeline or backgrounded command runs in a
# subshell -- its cwd change must not persist to later && / ; -joined commands
# in the same chain. Without this, ``echo x | cd <repo> && gh pr merge N``
# and ``cd <repo> & gh pr merge N`` reopen the exact wrong-repo merge-check
# bypass this hook closes.
# ---------------------------------------------------------------------------


def test_parse_cd_piped_does_not_persist_cd_cwd(tmp_path: Path) -> None:
    # ``echo x | cd <repo> && gh pr merge N``: the cd is the right side of a
    # pipe, so it runs in a subshell. The ``&&`` joins the *pipeline* (not
    # the cd) to the merge, so the merge runs in the original cwd, not the
    # cd'd directory. cd_cwd must be None.
    target_dir = tmp_path / "job-cannon"
    target_dir.mkdir()
    targets = hook._parse_gh_merge_targets(
        f"echo x | cd {target_dir.as_posix()} && gh pr merge 5", tmp_path
    )
    assert len(targets) == 1
    assert targets[0]["cd_cwd"] is None


def test_parse_cd_backgrounded_does_not_persist_cd_cwd(tmp_path: Path) -> None:
    # ``cd <repo> & gh pr merge N``: the ``&`` backgrounds the cd (subshell),
    # so the merge runs in the original cwd. cd_cwd must be None.
    target_dir = tmp_path / "job-cannon"
    target_dir.mkdir()
    targets = hook._parse_gh_merge_targets(f"cd {target_dir.as_posix()} & gh pr merge 5", tmp_path)
    assert len(targets) == 1
    assert targets[0]["cd_cwd"] is None


def test_parse_cd_left_side_of_pipe_does_not_persist(tmp_path: Path) -> None:
    # ``cd <repo> | cat; gh pr merge N``: the left side of a pipe also runs
    # in a subshell, so the cd does not persist past the pipeline.
    target_dir = tmp_path / "job-cannon"
    target_dir.mkdir()
    targets = hook._parse_gh_merge_targets(
        f"cd {target_dir.as_posix()} | cat; gh pr merge 5", tmp_path
    )
    assert len(targets) == 1
    assert targets[0]["cd_cwd"] is None


def test_parse_cd_in_multi_pipe_does_not_persist(tmp_path: Path) -> None:
    # A cd in any segment of a multi-stage pipeline is subshell-scoped.
    target_dir = tmp_path / "job-cannon"
    target_dir.mkdir()
    targets = hook._parse_gh_merge_targets(
        f"echo a | echo b | cd {target_dir.as_posix()} && gh pr merge 5", tmp_path
    )
    assert len(targets) == 1
    assert targets[0]["cd_cwd"] is None


def test_parse_cd_before_pipeline_persists_after_pipeline(tmp_path: Path) -> None:
    # ``cd <repo> && echo x | cat && gh pr merge N``: the cd is before the
    # pipeline (joined by ``&&``), so it persists. The pipeline (echo | cat)
    # runs in subshells but does not undo the prior cd. The merge after the
    # pipeline runs in the cd'd directory.
    target_dir = tmp_path / "job-cannon"
    target_dir.mkdir()
    targets = hook._parse_gh_merge_targets(
        f"cd {target_dir.as_posix()} && echo x | cat && gh pr merge 5", tmp_path
    )
    assert len(targets) == 1
    assert targets[0]["cd_cwd"] == target_dir.resolve()


def test_parse_cd_piped_with_semicolon_does_not_persist(tmp_path: Path) -> None:
    # ``echo x | cd <repo>; gh pr merge N``: the semicolon ends the pipeline,
    # reverting the subshell-scoped cd. The merge runs in the original cwd.
    target_dir = tmp_path / "job-cannon"
    target_dir.mkdir()
    targets = hook._parse_gh_merge_targets(
        f"echo x | cd {target_dir.as_posix()}; gh pr merge 5", tmp_path
    )
    assert len(targets) == 1
    assert targets[0]["cd_cwd"] is None


def test_parse_cd_in_subshell_with_pipe_does_not_leak(tmp_path: Path) -> None:
    # ``(echo x | cd <repo>) && gh pr merge N``: the pipe is inside a
    # subshell, so the cd is doubly contained. The merge after ``)`` runs in
    # the original cwd.
    target_dir = tmp_path / "job-cannon"
    target_dir.mkdir()
    targets = hook._parse_gh_merge_targets(
        f"(echo x | cd {target_dir.as_posix()}) && gh pr merge 5", tmp_path
    )
    assert len(targets) == 1
    assert targets[0]["cd_cwd"] is None
