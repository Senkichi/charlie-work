"""Committed-diff-since-base tests for scripts/worker_stop_gate.py.

Split verbatim out of ``tests/test_worker_stop_gate.py`` (issue #1573,
Track 1 shoulder). Shared helpers and the ``gate``/``repo`` fixtures live
in ``tests/_worker_stop_gate_fixtures.py`` -- the ``tests/_*.py``
hoisted-fixture convention is the sanctioned import target for shared
test helpers (see ``tests/test_zero_cross_test_import_guard.py``).

Covers ``_committed_diff_files`` (the committed-but-unpushed
surface) and ``_all_changed_files`` (its union with the working tree).
"""

from __future__ import annotations

import subprocess

from _worker_stop_gate_fixtures import (
    _git_output,
    _run_git,
    _set_origin_main,
    gate as gate,
    repo as repo,
)


# ---------------------------------------------------------------------------
# Committed-diff-since-base surface (review round, #1259, merge-blocker B).
# ---------------------------------------------------------------------------


def test_committed_diff_files_finds_committed_but_unpushed_change(gate, repo):
    base_sha = _git_output(["rev-parse", "HEAD"], repo)
    _set_origin_main(repo, base_sha)
    (repo / "worker_change.py").write_text("value = 1\n", encoding="utf-8")
    _run_git(["add", "worker_change.py"], cwd=repo)
    _run_git(["commit", "-m", "feat: worker commits before stop"], cwd=repo)

    result = gate._committed_diff_files(repo)

    assert {cf.path for cf in result} == {"worker_change.py"}
    assert all(not cf.deleted for cf in result)


def test_committed_diff_files_reports_non_ascii_filename_unquoted(gate, repo):
    base_sha = _git_output(["rev-parse", "HEAD"], repo)
    _set_origin_main(repo, base_sha)
    (repo / "café.py").write_text("x = 1\n", encoding="utf-8")
    _run_git(["add", "café.py"], cwd=repo)
    _run_git(["commit", "-m", "feat: add non-ascii file"], cwd=repo)

    result = gate._committed_diff_files(repo)

    assert any(cf.path == "café.py" and not cf.deleted for cf in result)


def test_committed_diff_files_returns_empty_without_origin_main_ref(gate, repo):
    # The repo fixture has no remote configured at all -- no origin/main to
    # diverge from.
    (repo / "extra.py").write_text("x = 1\n", encoding="utf-8")
    _run_git(["add", "extra.py"], cwd=repo)
    _run_git(["commit", "-m", "feat: add extra file"], cwd=repo)

    assert gate._committed_diff_files(repo) == ()


def test_committed_diff_files_returns_empty_when_head_equals_origin_main(gate, repo):
    head_sha = _git_output(["rev-parse", "HEAD"], repo)
    _set_origin_main(repo, head_sha)

    assert gate._committed_diff_files(repo) == ()


def test_committed_diff_files_returns_empty_on_detached_head(gate, repo):
    base_sha = _git_output(["rev-parse", "HEAD"], repo)
    _set_origin_main(repo, base_sha)
    (repo / "new_file.py").write_text("z = 1\n", encoding="utf-8")
    _run_git(["add", "new_file.py"], cwd=repo)
    _run_git(["commit", "-m", "feat: add new_file"], cwd=repo)
    head_sha = _git_output(["rev-parse", "HEAD"], repo)
    _run_git(["checkout", "--detach", head_sha], cwd=repo)

    # origin/main != HEAD here (a real diff exists) -- this only comes back
    # empty if the detached-HEAD short-circuit itself fires, not because
    # there happened to be nothing to diff.
    assert gate._committed_diff_files(repo) == ()


def test_committed_diff_files_returns_empty_on_merge_base_command_failure(gate, repo, monkeypatch):
    def _fake_run(cmd, *, cwd, timeout):
        del cwd, timeout
        if cmd[:2] == ["git", "symbolic-ref"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="refs/heads/main\n", stderr="")
        if cmd[:3] == ["git", "rev-parse", "--verify"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="deadbeef\n", stderr="")
        if cmd[:2] == ["git", "merge-base"]:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="fatal: no merge base\n")
        raise AssertionError(f"unexpected command {cmd}")

    monkeypatch.setattr(gate, "_run", _fake_run)

    assert gate._committed_diff_files(repo) == ()


def test_committed_diff_files_returns_empty_on_diff_command_failure(gate, repo, monkeypatch):
    def _fake_run(cmd, *, cwd, timeout):
        del cwd, timeout
        if cmd[:2] == ["git", "symbolic-ref"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="refs/heads/main\n", stderr="")
        if cmd[:3] == ["git", "rev-parse", "--verify"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="deadbeef\n", stderr="")
        if cmd[:2] == ["git", "merge-base"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="deadbeef\n", stderr="")
        if "diff" in cmd:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="fatal: bad object\n")
        raise AssertionError(f"unexpected command {cmd}")

    monkeypatch.setattr(gate, "_run", _fake_run)

    assert gate._committed_diff_files(repo) == ()


def test_all_changed_files_combines_disjoint_committed_and_working_tree_files(gate, repo):
    base_sha = _git_output(["rev-parse", "HEAD"], repo)
    _set_origin_main(repo, base_sha)
    (repo / "committed_only.py").write_text("a = 1\n", encoding="utf-8")
    _run_git(["add", "committed_only.py"], cwd=repo)
    _run_git(["commit", "-m", "feat: commit one file"], cwd=repo)
    (repo / "working_tree_only.py").write_text("b = 1\n", encoding="utf-8")

    result = gate._all_changed_files(repo)

    assert {cf.path for cf in result} == {"committed_only.py", "working_tree_only.py"}


def test_all_changed_files_working_tree_wins_on_path_collision(gate, repo):
    base_sha = _git_output(["rev-parse", "HEAD"], repo)
    _set_origin_main(repo, base_sha)
    (repo / "flip.py").write_text("a = 1\n", encoding="utf-8")
    _run_git(["add", "flip.py"], cwd=repo)
    _run_git(["commit", "-m", "feat: add flip.py"], cwd=repo)
    # The committed-diff source alone reports flip.py as added (not
    # deleted); the working tree then deletes it. The union must reflect
    # the fresher (working-tree) state, not the committed one.
    (repo / "flip.py").unlink()

    result = gate._all_changed_files(repo)
    by_path = {cf.path: cf for cf in result}

    assert by_path["flip.py"].deleted is True
