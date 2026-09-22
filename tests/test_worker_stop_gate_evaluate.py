"""``_evaluate`` orchestration tests for scripts/worker_stop_gate.py.

Split verbatim out of ``tests/test_worker_stop_gate.py`` (issue #1573,
Track 1 shoulder). Shared helpers and the ``gate``/``repo`` fixtures live
in ``tests/_worker_stop_gate_fixtures.py`` -- the ``tests/_*.py``
hoisted-fixture convention is the sanctioned import target for shared
test helpers (see ``tests/test_zero_cross_test_import_guard.py``).

Covers the ruff/pytest enforcement passes, changed-set scoping, and
the #1306 untracked-debris narrowing, with ``gate._run`` monkeypatched
to canned ``CompletedProcess`` stand-ins.
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
# Fast path: no diff at all.
# ---------------------------------------------------------------------------


def test_evaluate_fast_path_no_diff_skips_ruff_and_tests(gate, repo, monkeypatch):
    def _boom(*_args, **_kwargs):
        raise AssertionError("ruff/pytest must not run when there is no diff")

    monkeypatch.setattr(gate, "_run_ruff", _boom)
    monkeypatch.setattr(gate, "_run_targeted_tests", _boom)

    result = gate._evaluate(repo)

    assert result.block is False


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

    def _fake_run(cmd, *, cwd, timeout, env=None):
        del cwd, timeout, env
        if cmd[:2] == ["git", "symbolic-ref"]:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")
        if "status" in cmd:
            return subprocess.CompletedProcess(
                cmd, 0, stdout="?? tests/test_something.py\n", stderr=""
            )
        if "-c" in cmd:
            # Import-anchor probe (#1793): report a location inside repo.
            anchored = repo / "src" / "charlie_work" / "__init__.py"
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{anchored}\n", stderr="")
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
    close: the operator's live checkout has
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

    The pytest command list is captured and asserted to contain
    ``tests/test_new.py`` so the test fails if untracked test-file
    targeting regresses -- not just if pytest happens to not be invoked
    at all (which would leave ``result.block is False`` true regardless).
    """
    (repo / "tests").mkdir()
    (repo / "tests" / "test_new.py").write_text(
        "def test_x():\n    assert True\n", encoding="utf-8"
    )
    captured_pytest: list[list[str]] = []

    def _fake_run(cmd, *, cwd, timeout, env=None):
        del cwd, timeout, env
        if cmd[:2] == ["git", "symbolic-ref"]:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")
        if "status" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="?? tests/test_new.py\n", stderr="")
        if "-c" in cmd:
            # Import-anchor probe (#1793): report a location inside repo.
            anchored = repo / "src" / "charlie_work" / "__init__.py"
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{anchored}\n", stderr="")
        if "ruff" in cmd:
            raise AssertionError("ruff must not run on untracked test file (#1306)")
        if "pytest" in cmd:
            captured_pytest.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout="1 passed\n", stderr="")
        raise AssertionError(f"unexpected command {cmd}")

    monkeypatch.setattr(gate, "_run", _fake_run)

    result = gate._evaluate(repo)

    assert result.block is False
    assert captured_pytest, (
        "pytest must be invoked on the untracked test file -- a passing "
        "result with no pytest call means targeting silently regressed"
    )
    assert any("tests/test_new.py" in cmd for cmd in captured_pytest), (
        f"pytest must target tests/test_new.py; got {captured_pytest}"
    )


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

    def _fake_run(cmd, *, cwd, timeout, env=None):
        del cwd, timeout, env
        if cmd[:2] == ["git", "symbolic-ref"]:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")
        if "status" in cmd:
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout="?? src/charlie_work/new_emit.py\n",
                stderr="",
            )
        if "-c" in cmd:
            # Import-anchor probe (#1793): report a location inside repo.
            anchored = repo / "src" / "charlie_work" / "__init__.py"
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{anchored}\n", stderr="")
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
