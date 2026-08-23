"""Tests for scripts/worker_stop_gate.py (rework-RCA W2/#1259, W4/#1262).

Loaded via ``_script_loader.load_script_module`` (scripts/ is not on
sys.path) -- the same recipe as tests/test_merge_autonomy_ratio.py, and
required by tests/test_script_loader.py's
``test_no_hand_rolled_spec_from_file_location_in_tests`` guard.

No test here invokes a real ``uv``/``ruff``/``pytest`` subprocess:
``gate._run`` is monkeypatched to canned ``CompletedProcess`` stand-ins
wherever a command's *output* drives the decision under test. Where only
git's own porcelain parsing is under test, a real throwaway ``git init``
repo under ``tmp_path`` is used instead of hand-simulating porcelain text --
that exercises git's actual output format rather than this test suite's
assumptions about it, and stays hermetic (no network, no shared state,
self-contained per test).
"""

from __future__ import annotations

import io
import json
import os
import re
import subprocess
import time
from pathlib import Path
from types import ModuleType

import pytest
from _script_loader import load_script_module

_SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "worker_stop_gate.py"


def _load_module() -> ModuleType:
    return load_script_module(_SCRIPT_PATH, "worker_stop_gate_under_test")


def _run_git(args: list[str], cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _git_output(args: list[str], cwd: Path) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _set_origin_main(root: Path, sha: str) -> None:
    """Point a bare local ``refs/remotes/origin/main`` at ``sha``, decoupled
    from whatever the local ``init.defaultBranch`` happens to be -- the
    ``repo`` fixture has no real remote configured, so this is the only way
    to give ``_committed_diff_files`` something to diverge from."""
    _run_git(["update-ref", "refs/remotes/origin/main", sha], cwd=root)


@pytest.fixture()
def gate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> ModuleType:
    """A fresh module instance per test, with its state dir pinned under
    ``tmp_path`` so no test touches the real system temp dir (which would
    both pollute a real worker session's counters and leak across test
    runs). Function-scoped rather than module-scoped like
    test_merge_autonomy_ratio.py's ``mar`` fixture: several tests here
    monkeypatch module-level functions (``_run``, ``_repo_root``), and a
    module shared across tests would make patch ordering/teardown surprises
    possible.
    """
    module = _load_module()
    state_dir = tmp_path / "gate-state"
    monkeypatch.setattr(module, "_state_dir", lambda: _ensure_dir(state_dir))
    return module


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    """A real, throwaway git repo with one committed file."""
    root = tmp_path / "repo"
    root.mkdir()
    _run_git(["init"], cwd=root)
    _run_git(["config", "user.email", "test@example.com"], cwd=root)
    _run_git(["config", "user.name", "Test"], cwd=root)
    (root / "README.md").write_text("placeholder\n", encoding="utf-8")
    _run_git(["add", "README.md"], cwd=root)
    _run_git(["commit", "-m", "init"], cwd=root)
    return root


def _stdin(payload: dict) -> io.StringIO:
    return io.StringIO(json.dumps(payload))


def test_state_dir_creates_and_returns_directory_under_temp(tmp_path, monkeypatch):
    # Deliberately does not use the ``gate`` fixture: that fixture patches
    # ``_state_dir`` itself, which would defeat the point of exercising its
    # real body (tempfile.gettempdir()-based) here.
    module = _load_module()
    monkeypatch.setattr(module.tempfile, "gettempdir", lambda: str(tmp_path))

    result = module._state_dir()

    assert result == tmp_path / "worker_stop_gate"
    assert result.is_dir()


# ---------------------------------------------------------------------------
# Fast path: no diff at all.
# ---------------------------------------------------------------------------


def test_evaluate_fast_path_no_diff_skips_ruff_and_tests(gate, repo, monkeypatch):
    def _boom(*_args, **_kwargs):
        raise AssertionError("ruff/pytest must not run when there is no diff")

    monkeypatch.setattr(gate, "_run_ruff", _boom)
    monkeypatch.setattr(gate, "_run_targeted_tests", _boom)

    result = gate._evaluate(repo)

    assert result.block is False


def test_main_preexhaustion_fast_path_skips_reevaluation(gate, monkeypatch, capsys):
    session_id = "sess-preexhausted"
    gate._write_block_count(gate._counter_path(session_id), gate.MAX_BLOCKS_PER_SESSION)

    def _boom(_cwd):
        raise AssertionError("must not re-evaluate once already exhausted")

    monkeypatch.setattr(gate, "_repo_root", _boom)
    monkeypatch.setattr(gate.sys, "stdin", _stdin({"session_id": session_id}))

    rc = gate.main()

    assert rc == 0
    assert gate.EXHAUSTED_MARKER in capsys.readouterr().err
    # The fast path self-heals a stuck counter: a later, unrelated failure
    # in the same session must get its own fresh streak, not an
    # already-exhausted one.
    assert gate._read_block_count(gate._counter_path(session_id)) == 0


# ---------------------------------------------------------------------------
# W4/#1262 targeting rule.
# ---------------------------------------------------------------------------


def test_w4_targeting_rule_positive_emit_site_pulls_instrumentation_test(gate, repo):
    src_dir = repo / "src" / "charlie_work"
    src_dir.mkdir(parents=True)
    (src_dir / "new_thing.py").write_text(
        "def do_it():\n    log_event(state_path, 'thing_happened', {})\n", encoding="utf-8"
    )

    changed = gate._changed_files(repo)
    targets = gate._targeted_tests(repo, changed)

    assert gate.INSTRUMENTATION_TEST_PATH in targets


def test_w4_targeting_rule_negative_no_emit_site_no_instrumentation_test(gate, repo):
    src_dir = repo / "src" / "charlie_work"
    src_dir.mkdir(parents=True)
    (src_dir / "new_thing.py").write_text("def do_it():\n    return 1\n", encoding="utf-8")

    changed = gate._changed_files(repo)
    targets = gate._targeted_tests(repo, changed)

    assert gate.INSTRUMENTATION_TEST_PATH not in targets
    assert targets == ()


def test_targeted_tests_includes_changed_test_files_without_w4_trigger(gate, repo):
    (repo / "tests").mkdir()
    (repo / "tests" / "test_something.py").write_text(
        "def test_x():\n    assert True\n", encoding="utf-8"
    )

    changed = gate._changed_files(repo)
    targets = gate._targeted_tests(repo, changed)

    assert targets == ("tests/test_something.py",)


@pytest.mark.parametrize(
    "call_form",
    [
        "log_event(state_path, 'k', {})",
        "append_event(path, kind='k')",
        "self._record_event('k', {})",
    ],
)
def test_touches_emit_site_matches_all_three_call_forms(gate, repo, call_form):
    src_dir = repo / "src" / "charlie_work"
    src_dir.mkdir(parents=True)
    (src_dir / "site.py").write_text(f"def f():\n    {call_form}\n", encoding="utf-8")

    changed = gate._changed_files(repo)

    assert gate._touches_emit_site(repo, changed) is True


def test_touches_emit_site_ignores_deleted_files(gate, repo):
    src_dir = repo / "src" / "charlie_work"
    src_dir.mkdir(parents=True)
    target = src_dir / "gone.py"
    target.write_text("log_event(x, 'k', {})\n", encoding="utf-8")
    _run_git(["add", "-A"], cwd=repo)
    _run_git(["commit", "-m", "add gone"], cwd=repo)
    target.unlink()

    changed = gate._changed_files(repo)

    assert gate._touches_emit_site(repo, changed) is False


def test_touches_emit_site_fails_closed_when_file_unreadable(gate, repo):
    changed = (gate.ChangedFile(path="src/does_not_exist_on_disk.py", deleted=False),)

    assert gate._touches_emit_site(repo, changed) is True


# ---------------------------------------------------------------------------
# git status --porcelain parsing.
# ---------------------------------------------------------------------------


def test_changed_files_parses_deleted_and_untracked_entries(gate, repo):
    tracked = repo / "keep.py"
    tracked.write_text("x = 1\n", encoding="utf-8")
    _run_git(["add", "keep.py"], cwd=repo)
    _run_git(["commit", "-m", "add keep"], cwd=repo)

    tracked.unlink()
    (repo / "new_untracked.py").write_text("y = 2\n", encoding="utf-8")

    changed = gate._changed_files(repo)
    by_path = {cf.path: cf for cf in changed}

    assert by_path["keep.py"].deleted is True
    assert by_path["keep.py"].untracked is False  # tracked file, now deleted
    assert by_path["new_untracked.py"].deleted is False
    assert by_path["new_untracked.py"].untracked is True  # ?? entry


def test_changed_files_reports_non_ascii_filename_unquoted(gate, repo):
    # Review round, #1259 follow-up 2: without -c core.quotePath=false, git
    # C-quotes a non-ASCII byte into an octal-escaped, double-quoted string
    # (e.g. "caf\303\251.py"), which does not .endswith(".py") and silently
    # drops the file from every rule downstream.
    (repo / "café.py").write_text("x = 1\n", encoding="utf-8")

    changed = gate._changed_files(repo)

    assert any(cf.path == "café.py" and not cf.deleted for cf in changed)


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


# ---------------------------------------------------------------------------
# Fail-closed error path.
# ---------------------------------------------------------------------------


def test_main_fails_closed_on_unexpected_exception(gate, monkeypatch, capsys):
    def _boom(_cwd):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(gate, "_repo_root", _boom)
    monkeypatch.setattr(gate.sys, "stdin", _stdin({"session_id": "sess-error"}))

    rc = gate.main()

    assert rc == gate.BLOCK_EXIT_CODE
    assert "internal gate error" in capsys.readouterr().err


def test_main_fails_closed_on_gate_error_with_reason(gate, monkeypatch, capsys):
    def _boom(_cwd):
        raise gate.GateError("git binary missing")

    monkeypatch.setattr(gate, "_repo_root", _boom)
    monkeypatch.setattr(gate.sys, "stdin", _stdin({"session_id": "sess-x"}))

    rc = gate.main()

    err = capsys.readouterr().err
    assert rc == gate.BLOCK_EXIT_CODE
    assert "internal gate error" in err
    assert "git binary missing" in err


def test_main_fails_closed_when_decide_and_report_itself_raises(gate, monkeypatch, capsys):
    # A failure while *recording* the decision (e.g. the counter-file write)
    # must still block -- it must not bubble up as a bare non-2 exit, which
    # the hook contract treats as fail-OPEN (see module docstring).
    def _boom(_session_id, _result):
        raise OSError("disk full")

    monkeypatch.setattr(
        gate, "_repo_root", lambda _cwd: (_ for _ in ()).throw(gate.GateError("x"))
    )
    monkeypatch.setattr(gate, "_decide_and_report", _boom)
    monkeypatch.setattr(gate.sys, "stdin", _stdin({"session_id": "sess-decide-boom"}))

    rc = gate.main()

    assert rc == gate.BLOCK_EXIT_CODE
    assert "failing closed" in capsys.readouterr().err


def test_main_blocks_when_not_a_git_repo(gate, tmp_path, monkeypatch):
    not_a_repo = tmp_path / "not-a-repo"
    not_a_repo.mkdir()
    monkeypatch.setattr(
        gate.sys, "stdin", _stdin({"session_id": "sess-notgit", "cwd": str(not_a_repo)})
    )

    rc = gate.main()

    assert rc == gate.BLOCK_EXIT_CODE


def test_run_wraps_oserror_as_gate_error(gate, tmp_path, monkeypatch):
    def _raise(*_args, **_kwargs):
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(gate.subprocess, "run", _raise)

    with pytest.raises(gate.GateError):
        gate._run(["git", "status"], cwd=tmp_path, timeout=5)


def test_run_wraps_timeout_as_gate_error(gate, tmp_path, monkeypatch):
    def _raise(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd="git", timeout=5)

    monkeypatch.setattr(gate.subprocess, "run", _raise)

    with pytest.raises(gate.GateError):
        gate._run(["git", "status"], cwd=tmp_path, timeout=5)


def test_main_handles_malformed_stdin_without_crashing(gate, monkeypatch):
    monkeypatch.setattr(gate.sys, "stdin", io.StringIO("not json"))
    monkeypatch.setattr(
        gate, "_repo_root", lambda _cwd: (_ for _ in ()).throw(gate.GateError("unreachable"))
    )

    rc = gate.main()

    assert rc == gate.BLOCK_EXIT_CODE


# ---------------------------------------------------------------------------
# Ruff / targeted-test enforcement (subprocess mocked).
# ---------------------------------------------------------------------------


def test_evaluate_blocks_on_ruff_check_failure(gate, repo, monkeypatch):
    # Must be a .py file: with ruff scoped to the changed-set (blocker A
    # fix, review round, #1259), a non-.py dirty file would never even
    # reach a ruff invocation -- that emptiness path is covered separately
    # by test_evaluate_skips_ruff_subprocess_when_changed_set_has_no_py_files.
    # Uses a tracked-modified (`` M``) file, not ``??``: #1306 excludes
    # untracked files from ruff's scope, so an untracked dirty.py would
    # never reach ruff and this test would silently stop exercising the
    # ruff-check-failure path.
    (repo / "dirty.py").write_text("x", encoding="utf-8")

    def _fake_run(cmd, *, cwd, timeout):
        del cwd, timeout
        if cmd[:2] == ["git", "symbolic-ref"]:
            # Simulate detached HEAD so _committed_diff_files short-circuits
            # to () and this test's scope stays working-tree-only, as before
            # the committed-diff union (review round, #1259, blocker B).
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")
        if "status" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout=" M dirty.py\n", stderr="")
        if "check" in cmd:
            return subprocess.CompletedProcess(cmd, 1, stdout="E501 line too long\n", stderr="")
        raise AssertionError(f"unexpected command {cmd}")

    monkeypatch.setattr(gate, "_run", _fake_run)

    result = gate._evaluate(repo)

    assert result.block is True
    assert "ruff check failed" in result.reason


def test_evaluate_blocks_on_ruff_format_failure(gate, repo, monkeypatch):
    # Tracked-modified (`` M``), not ``??``: #1306 excludes untracked files
    # from ruff's scope -- see test_evaluate_blocks_on_ruff_check_failure.
    (repo / "dirty.py").write_text("x", encoding="utf-8")

    def _fake_run(cmd, *, cwd, timeout):
        del cwd, timeout
        if cmd[:2] == ["git", "symbolic-ref"]:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")
        if "status" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout=" M dirty.py\n", stderr="")
        if "check" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if "format" in cmd:
            return subprocess.CompletedProcess(
                cmd, 1, stdout="would reformat dirty.py\n", stderr=""
            )
        raise AssertionError(f"unexpected command {cmd}")

    monkeypatch.setattr(gate, "_run", _fake_run)

    result = gate._evaluate(repo)

    assert result.block is True
    assert "ruff format --check failed" in result.reason


def test_evaluate_blocks_on_targeted_test_failure(gate, repo, monkeypatch):
    (repo / "tests").mkdir()
    (repo / "tests" / "test_something.py").write_text(
        "def test_x():\n    assert False\n", encoding="utf-8"
    )

    def _fake_run(cmd, *, cwd, timeout):
        del cwd, timeout
        if cmd[:2] == ["git", "symbolic-ref"]:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")
        if "status" in cmd:
            return subprocess.CompletedProcess(
                cmd, 0, stdout="?? tests/test_something.py\n", stderr=""
            )
        if "ruff" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if "pytest" in cmd:
            return subprocess.CompletedProcess(cmd, 1, stdout="1 failed\n", stderr="")
        raise AssertionError(f"unexpected command {cmd}")

    monkeypatch.setattr(gate, "_run", _fake_run)

    result = gate._evaluate(repo)

    assert result.block is True
    assert "tests/test_something.py" in result.reason


def test_run_targeted_tests_skips_pytest_when_no_targets(gate, repo, monkeypatch):
    def _boom(*_args, **_kwargs):
        raise AssertionError("pytest must not run with an empty target set")

    monkeypatch.setattr(gate, "_run", _boom)

    result = gate._run_targeted_tests(repo, ())

    assert result.block is False


def test_run_ruff_skips_subprocess_entirely_when_py_files_empty(gate, repo, monkeypatch):
    def _boom(*_args, **_kwargs):
        raise AssertionError("ruff must not run with an empty py_files scope")

    monkeypatch.setattr(gate, "_run", _boom)

    result = gate._run_ruff(repo, ())

    assert result.block is False


def test_evaluate_skips_ruff_subprocess_when_changed_set_has_no_py_files(gate, repo, monkeypatch):
    # A non-empty changed set (so _evaluate does not take the fast "nothing
    # changed" path) that contains no .py files must still never invoke
    # ruff -- the emptiness check has to survive past _evaluate's own call
    # site, not just _run_ruff's internal one.
    (repo / "notes.txt").write_text("hello\n", encoding="utf-8")

    def _fake_run(cmd, *, cwd, timeout):
        del cwd, timeout
        if cmd[:2] == ["git", "symbolic-ref"]:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")
        if "status" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="?? notes.txt\n", stderr="")
        raise AssertionError(f"unexpected command {cmd} -- notes.txt is not a .py file")

    monkeypatch.setattr(gate, "_run", _fake_run)

    result = gate._evaluate(repo)

    assert result.block is False


def test_evaluate_scopes_ruff_to_explicit_changed_files_not_whole_tree(gate, repo, monkeypatch):
    """Merge-blocker A fix (review round, #1259): ruff must be invoked with
    an explicit file list derived from the changed-set, never a bare "."
    that would rescan the whole tree, including files this session never
    touched. Uses a tracked-modified (`` M``) file because #1306 excludes
    untracked ``??`` files from ruff's scope -- an untracked file would
    never reach ruff and this test would silently stop exercising the
    scoping path. (The real, unmocked empirical proof that this actually
    stops a spurious block -- a pre-existing lint/format issue outside the
    diff not blocking, while a real whole-tree scan on the same repo does
    fail -- was run as a standalone smoke test against the live script, not
    via a mocked unit test.)"""
    (repo / "session_file.py").write_text("x = 1\n", encoding="utf-8")
    captured: list[list[str]] = []

    def _fake_run(cmd, *, cwd, timeout):
        del cwd, timeout
        if cmd[:2] == ["git", "symbolic-ref"]:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")
        if "status" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout=" M session_file.py\n", stderr="")
        if "ruff" in cmd:
            captured.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected command {cmd}")

    monkeypatch.setattr(gate, "_run", _fake_run)

    result = gate._evaluate(repo)

    assert result.block is False
    assert captured, "ruff must still run when there is a tracked-modified .py file in scope"
    for cmd in captured:
        assert "." not in cmd, f"ruff invoked with whole-tree scope: {cmd}"
        assert "session_file.py" in cmd


# ---------------------------------------------------------------------------
# #1306: untracked debris must not gate ruff, but still gates tests/W4.
# ---------------------------------------------------------------------------


def test_evaluate_excludes_untracked_py_from_ruff_scope(gate, repo, monkeypatch):
    """#1306: an untracked (``??``) ``.py`` file -- which may be pre-existing
    debris that predates the session, since ``git status`` cannot tell the
    two apart -- must never reach ruff. A would-reformat failure on it must
    not block the stop. This is the core case the #1259 scoping did not
    close: the live checkout at C:\\Users\\senki\\repos\\charlie-work has
    untracked ``scripts/ac3_*.py`` files that ``ruff format --check`` fails
    on, and the scoped gate still included them because they are untracked.
    """
    (repo / "debris.py").write_text("x  =  1\n", encoding="utf-8")

    def _fake_run(cmd, *, cwd, timeout):
        del cwd, timeout
        if cmd[:2] == ["git", "symbolic-ref"]:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")
        if "status" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="?? debris.py\n", stderr="")
        if "ruff" in cmd:
            raise AssertionError(f"ruff must not run on untracked debris.py (#1306), got: {cmd}")
        raise AssertionError(f"unexpected command {cmd}")

    monkeypatch.setattr(gate, "_run", _fake_run)

    result = gate._evaluate(repo)

    assert result.block is False


def test_evaluate_excludes_untracked_py_from_ruff_but_keeps_tracked_modified(
    gate, repo, monkeypatch
):
    """#1306 mixed case: an untracked debris file and a tracked-modified
    session file coexist. Ruff must run on the tracked-modified file only
    and must not be invoked with the untracked file's path. If the
    tracked-modified file has a real ruff failure, the gate still blocks on
    *that* -- the untracked exclusion narrows scope, it does not disarm the
    gate.
    """
    (repo / "debris.py").write_text("x  =  1\n", encoding="utf-8")
    (repo / "session.py").write_text("x = 1\n", encoding="utf-8")
    captured: list[list[str]] = []

    def _fake_run(cmd, *, cwd, timeout):
        del cwd, timeout
        if cmd[:2] == ["git", "symbolic-ref"]:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")
        if "status" in cmd:
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout="?? debris.py\n M session.py\n",
                stderr="",
            )
        if "ruff" in cmd:
            captured.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected command {cmd}")

    monkeypatch.setattr(gate, "_run", _fake_run)

    result = gate._evaluate(repo)

    assert result.block is False
    assert captured, "ruff must run on the tracked-modified session.py"
    for cmd in captured:
        assert "session.py" in cmd
        assert "debris.py" not in cmd, f"ruff must not include untracked debris.py: {cmd}"


def test_evaluate_still_targets_untracked_test_files(gate, repo, monkeypatch):
    """#1306: untracked files are excluded from *ruff* only, not from test
    targeting. A brand-new untracked ``tests/*.py`` file legitimately needs
    coverage -- the gate must still run pytest on it even though ruff skips
    it. (If ruff ran on it, a format issue in the new test file would block
    before pytest even gets to run it -- the #1306 trade-off.)
    """
    (repo / "tests").mkdir()
    (repo / "tests" / "test_new.py").write_text(
        "def test_x():\n    assert True\n", encoding="utf-8"
    )

    def _fake_run(cmd, *, cwd, timeout):
        del cwd, timeout
        if cmd[:2] == ["git", "symbolic-ref"]:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")
        if "status" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="?? tests/test_new.py\n", stderr="")
        if "ruff" in cmd:
            raise AssertionError("ruff must not run on untracked test file (#1306)")
        if "pytest" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="1 passed\n", stderr="")
        raise AssertionError(f"unexpected command {cmd}")

    monkeypatch.setattr(gate, "_run", _fake_run)

    result = gate._evaluate(repo)

    assert result.block is False


def test_evaluate_untracked_src_with_emit_site_still_triggers_w4(gate, repo, monkeypatch):
    """#1306: untracked files are excluded from *ruff* only, not from the W4
    emit-site rule. A brand-new untracked ``src/*.py`` file that calls
    ``log_event``/``append_event``/``_record_event`` must still pull
    ``tests/test_instrumentation.py`` into the targeted-test set -- a new
    event-emit site legitimately needs registry-exhaustiveness coverage
    even before the file is staged.
    """
    src_dir = repo / "src" / "charlie_work"
    src_dir.mkdir(parents=True)
    (src_dir / "new_emit.py").write_text(
        "def f():\n    log_event(state_path, 'k', {})\n", encoding="utf-8"
    )

    def _fake_run(cmd, *, cwd, timeout):
        del cwd, timeout
        if cmd[:2] == ["git", "symbolic-ref"]:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")
        if "status" in cmd:
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout="?? src/charlie_work/new_emit.py\n",
                stderr="",
            )
        if "ruff" in cmd:
            raise AssertionError("ruff must not run on untracked src file (#1306)")
        if "pytest" in cmd:
            # The W4 rule must have pulled in the instrumentation test path.
            assert gate.INSTRUMENTATION_TEST_PATH in cmd, (
                f"W4 must target {gate.INSTRUMENTATION_TEST_PATH} for untracked emit-site file: {cmd}"
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="1 passed\n", stderr="")
        raise AssertionError(f"unexpected command {cmd}")

    monkeypatch.setattr(gate, "_run", _fake_run)

    result = gate._evaluate(repo)

    assert result.block is False


def test_evaluate_fast_path_when_only_change_predates_the_branch_base(gate, repo, monkeypatch):
    """Real-git companion to the mocked scoping test above: a file that is
    committed AT the origin/main base (i.e. it predates this branch's own
    diff) produces an empty _all_changed_files() result, so _evaluate takes
    the fast "nothing changed" path and never even calls _run_ruff. This is
    the exact mechanism behind the blocker-A smoke test's real, unmocked
    result run separately against the live script (a committed,
    unmodified file with a genuine ruff format violation -> gate rc=0,
    while a bare whole-tree `ruff format --check .` on the same repo fails).
    """
    (repo / "messy.py").write_text("x  =  1\n", encoding="utf-8")
    _run_git(["add", "messy.py"], cwd=repo)
    _run_git(["commit", "-m", "chore: pre-existing messy file"], cwd=repo)
    _set_origin_main(repo, _git_output(["rev-parse", "HEAD"], repo))

    def _boom(*_args, **_kwargs):
        raise AssertionError("ruff must not run -- messy.py predates the branch base")

    monkeypatch.setattr(gate, "_run_ruff", _boom)

    result = gate._evaluate(repo)

    assert result.block is False


# ---------------------------------------------------------------------------
# End-to-end main() with a fully mocked subprocess layer.
# ---------------------------------------------------------------------------


def test_main_allows_clean_session_end_to_end(gate, repo, monkeypatch):
    (repo / "untracked.txt").write_text("x", encoding="utf-8")

    def _fake_run(cmd, *, cwd, timeout):
        del cwd, timeout
        if cmd[:2] == ["git", "symbolic-ref"]:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")
        if cmd[:2] == ["git", "rev-parse"]:
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{repo}\n", stderr="")
        if "status" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="?? untracked.txt\n", stderr="")
        if "ruff" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected command {cmd}")

    monkeypatch.setattr(gate, "_run", _fake_run)
    monkeypatch.setattr(gate.sys, "stdin", _stdin({"session_id": "sess-ok", "cwd": str(repo)}))

    rc = gate.main()

    assert rc == 0


# ---------------------------------------------------------------------------
# Bounded-retry / exhaustion contract.
# ---------------------------------------------------------------------------


def test_decide_and_report_blocks_up_to_cap_then_exhausts(gate, capsys):
    # Anti-vacuity guard (review round, #1259 follow-up 1): both the loop
    # count and the expected codes below are hardcoded literals, not derived
    # from MAX_BLOCKS_PER_SESSION -- with both bounds coming from the same
    # constant under test, this assertion previously passed even with
    # MAX_BLOCKS_PER_SESSION == 0 (a gate that never actually blocks). The
    # explicit assert is a tripwire on that one specific value; the literal
    # codes list is what actually makes the rest of the test fail if the cap
    # changes out from under it.
    assert gate.MAX_BLOCKS_PER_SESSION == 3
    session_id = "sess-cap"
    failing = gate.GateResult(block=True, reason="boom")

    codes = [gate._decide_and_report(session_id, failing) for _ in range(4)]

    assert codes == [2, 2, 2, 0]
    assert gate.EXHAUSTED_MARKER in capsys.readouterr().err
    # Exhaustion ends the streak -- it must not spend a lifetime session
    # budget, or one early stumble silently disarms the gate for good.
    assert gate._read_block_count(gate._counter_path(session_id)) == 0


def test_decide_and_report_exhaustion_starts_a_fresh_streak(gate, capsys):
    session_id = "sess-cap-fresh"
    failing = gate.GateResult(block=True, reason="boom")

    for _ in range(gate.MAX_BLOCKS_PER_SESSION + 1):
        gate._decide_and_report(session_id, failing)
    capsys.readouterr()  # discard exhaustion output from the setup loop

    rc = gate._decide_and_report(session_id, failing)

    assert rc == gate.BLOCK_EXIT_CODE
    assert gate.EXHAUSTED_MARKER not in capsys.readouterr().err
    assert gate._read_block_count(gate._counter_path(session_id)) == 1


def test_decide_and_report_passing_result_never_blocks_or_increments(gate):
    session_id = "sess-pass"
    passing = gate.GateResult(block=False)

    rc = gate._decide_and_report(session_id, passing)

    assert rc == 0
    assert gate._read_block_count(gate._counter_path(session_id)) == 0


def test_decide_and_report_pass_resets_an_in_progress_streak(gate):
    session_id = "sess-reset-midstreak"
    failing = gate.GateResult(block=True, reason="boom")
    passing = gate.GateResult(block=False)

    gate._decide_and_report(session_id, failing)
    gate._decide_and_report(session_id, failing)
    assert gate._read_block_count(gate._counter_path(session_id)) == 2

    rc = gate._decide_and_report(session_id, passing)

    assert rc == 0
    assert gate._read_block_count(gate._counter_path(session_id)) == 0

    # And the next failure starts a fresh streak at count=1, not count=3.
    rc = gate._decide_and_report(session_id, failing)
    assert rc == gate.BLOCK_EXIT_CODE
    assert gate._read_block_count(gate._counter_path(session_id)) == 1


# ---------------------------------------------------------------------------
# Session-id sanitization.
# ---------------------------------------------------------------------------


def test_safe_session_id_falls_back_on_missing_or_non_string(gate):
    assert gate._safe_session_id(None) == gate._FALLBACK_SESSION_ID
    assert gate._safe_session_id("") == gate._FALLBACK_SESSION_ID
    assert gate._safe_session_id(123) == gate._FALLBACK_SESSION_ID


def test_safe_session_id_strips_path_unsafe_characters(gate):
    result = gate._safe_session_id("../../etc/passwd")

    assert "/" not in result
    assert re.fullmatch(r"[A-Za-z0-9_-]+", result)


# ---------------------------------------------------------------------------
# Counter persistence (atomic tmp+replace, per the repo's JSON-write invariant).
# ---------------------------------------------------------------------------


def test_write_block_count_is_atomic_tmp_replace(gate, tmp_path):
    target = tmp_path / "sess.count"

    gate._write_block_count(target, 2)

    assert target.exists()
    assert not target.with_suffix(target.suffix + ".tmp").exists()
    assert json.loads(target.read_text(encoding="utf-8")) == {"count": 2}


def test_read_block_count_defaults_to_zero_on_missing_or_corrupt(gate, tmp_path):
    missing = tmp_path / "missing.count"
    assert gate._read_block_count(missing) == 0

    corrupt = tmp_path / "corrupt.count"
    corrupt.write_text("not json", encoding="utf-8")
    assert gate._read_block_count(corrupt) == 0

    negative = tmp_path / "negative.count"
    negative.write_text(json.dumps({"count": -1}), encoding="utf-8")
    assert gate._read_block_count(negative) == 0


def test_append_invocation_log_is_best_effort_on_io_error(gate, monkeypatch, tmp_path):
    bad_dir = tmp_path / "is-a-dir"
    bad_dir.mkdir()
    monkeypatch.setattr(gate, "_log_path", lambda _session_id: bad_dir)

    gate._append_invocation_log("sess", "message")  # must not raise


# ---------------------------------------------------------------------------
# State-dir pruning (review round, #1259 follow-up 3).
# ---------------------------------------------------------------------------


def test_prune_stale_state_files_deletes_old_keeps_fresh_and_other_suffixes(gate, tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    old_count = state_dir / "old.count"
    old_log = state_dir / "old.log"
    fresh_count = state_dir / "fresh.count"
    other_suffix = state_dir / "old.other"
    for path in (old_count, old_log, fresh_count, other_suffix):
        path.write_text("x", encoding="utf-8")

    stale_time = time.time() - gate.STATE_FILE_MAX_AGE_SECONDS - 3600
    os.utime(old_count, (stale_time, stale_time))
    os.utime(old_log, (stale_time, stale_time))
    os.utime(other_suffix, (stale_time, stale_time))

    gate._prune_stale_state_files(state_dir)

    assert not old_count.exists()
    assert not old_log.exists()
    assert fresh_count.exists()
    assert other_suffix.exists()  # not a .count/.log suffix -- never touched


def test_prune_stale_state_files_swallows_missing_directory(gate, tmp_path):
    missing = tmp_path / "does-not-exist"

    gate._prune_stale_state_files(missing)  # must not raise


def test_prune_stale_state_files_swallows_per_file_unlink_error(gate, tmp_path, monkeypatch):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    stubborn = state_dir / "stubborn.count"
    stubborn.write_text("x", encoding="utf-8")
    stale_time = time.time() - gate.STATE_FILE_MAX_AGE_SECONDS - 3600
    os.utime(stubborn, (stale_time, stale_time))

    def _boom(self):
        raise OSError("permission denied")

    monkeypatch.setattr(Path, "unlink", _boom)

    gate._prune_stale_state_files(state_dir)  # must not raise; failure is swallowed


def test_main_prunes_stale_state_files_on_startup(gate, monkeypatch):
    calls: list[Path] = []
    monkeypatch.setattr(gate, "_prune_stale_state_files", lambda d: calls.append(d))
    monkeypatch.setattr(
        gate, "_repo_root", lambda _cwd: (_ for _ in ()).throw(gate.GateError("unreachable"))
    )
    monkeypatch.setattr(gate.sys, "stdin", _stdin({"session_id": "sess-prune"}))

    gate.main()

    assert len(calls) == 1
    assert calls[0] == gate._state_dir()
