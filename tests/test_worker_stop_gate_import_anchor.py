"""Import-anchor tests for scripts/worker_stop_gate.py (issue #1793).

The gate used to run targeted tests as ``uv run --no-sync pytest`` with no
``PYTHONPATH`` -- test selection was anchored to the gated worktree's diff,
but import resolution followed whatever the resolved environment's editable
install pointed at. In a worktree that is the MAIN checkout's ``src``:
false reds for a good worktree, false greens for a broken one -- and a
fresh ``.venv`` silently created inside the worktree as a side effect.

These tests pin the fix: ``sys.executable -m pytest`` with the gated
toplevel's ``src/`` prepended to ``PYTHONPATH``, an import probe that
asserts ``charlie_work.__file__`` is inside the gated toplevel before any
result is trusted, and no ``uv`` invocation anywhere on the check path.

The two ``_real`` tests exercise the real subprocess layer end to end
against a fake worktree layout where "main" and "worktree" define a
different constant -- per the issue's acceptance criteria.

Shared helpers and the ``gate``/``repo`` fixtures live in
``tests/_worker_stop_gate_fixtures.py`` -- the ``tests/_*.py``
hoisted-fixture convention is the sanctioned import target for shared
test helpers (see ``tests/test_zero_cross_test_import_guard.py``).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from _worker_stop_gate_fixtures import gate as gate, repo as repo


def _make_src_tree(root: Path, marker: str) -> Path:
    """A minimal src-layout checkout: ``src/charlie_work/__init__.py``
    defining a distinguishing constant."""
    pkg = root / "src" / "charlie_work"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(f'MARKER = "{marker}"\n', encoding="utf-8")
    return root


def _probe_ok(repo_root: Path) -> subprocess.CompletedProcess:
    """A canned import-anchor probe result reporting an anchored location."""
    anchored = repo_root / "src" / "charlie_work" / "__init__.py"
    return subprocess.CompletedProcess([], 0, stdout=f"{anchored}\n", stderr="")


# ---------------------------------------------------------------------------
# _gated_tree_env: PYTHONPATH head is the gated tree's src; stale venv vars go.
# ---------------------------------------------------------------------------


def test_gated_tree_env_prepends_gated_src_ahead_of_inherited_pythonpath(gate, repo, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(["inherited", "tail"]))

    env = gate._gated_tree_env(repo)

    head, *tail = env["PYTHONPATH"].split(os.pathsep)
    assert head == str(repo / "src")
    assert tail == ["inherited", "tail"]


def test_gated_tree_env_drops_stale_virtual_env_vars(gate, repo, monkeypatch):
    monkeypatch.setenv("VIRTUAL_ENV", str(repo.parent / "other-checkout" / ".venv"))
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", str(repo.parent / "other-checkout" / ".venv"))

    env = gate._gated_tree_env(repo)

    assert "VIRTUAL_ENV" not in env
    assert "UV_PROJECT_ENVIRONMENT" not in env


def test_gated_tree_env_sets_pythonpath_when_none_inherited(gate, repo, monkeypatch):
    monkeypatch.delenv("PYTHONPATH", raising=False)

    env = gate._gated_tree_env(repo)

    assert env["PYTHONPATH"] == str(repo / "src")


# ---------------------------------------------------------------------------
# _path_within: resolved-path containment.
# ---------------------------------------------------------------------------


def test_path_within_accepts_child_and_root_itself(gate, tmp_path):
    root = tmp_path / "tree"
    assert gate._path_within(root / "src" / "charlie_work" / "__init__.py", root) is True
    assert gate._path_within(root, root) is True


def test_path_within_rejects_outside_and_sibling_prefix(gate, tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    assert gate._path_within(tmp_path / "elsewhere" / "x.py", root) is False
    # Exact-prefix vs glob-prefix trap: "repo2" shares the string prefix
    # "repo" but is not inside it.
    sibling = tmp_path / "repo2" / "src" / "charlie_work" / "__init__.py"
    assert gate._path_within(sibling, root) is False


def test_path_within_resolves_dotdot_before_comparing(gate, tmp_path):
    root = tmp_path / "repo"
    # A lexical escape ("repo/../outside") must not be read as contained.
    assert gate._path_within(root / ".." / "outside" / "x.py", root) is False
    # And a lexical containment that realpaths back inside still passes.
    assert gate._path_within(root / "sub" / ".." / "x.py", root) is True


# ---------------------------------------------------------------------------
# _assert_import_anchor: mocked probe outcomes.
# ---------------------------------------------------------------------------


def test_assert_import_anchor_passes_when_probe_lands_inside_root(gate, repo, monkeypatch):
    def _fake_run(cmd, *, cwd, timeout, env=None):
        del cwd, timeout, env
        assert "-c" in cmd, f"expected the -c probe, got {cmd}"
        return _probe_ok(repo)

    monkeypatch.setattr(gate, "_run", _fake_run)

    assert gate._assert_import_anchor(repo) is None


def test_assert_import_anchor_blocks_distinctly_when_probe_escapes_root(gate, repo, monkeypatch):
    """The wrong-tree control from the acceptance criteria: point the
    assertion at a resolution outside the gated tree and it must fail with
    the distinct error -- not a wall of test output."""
    outside = repo.parent / "main-checkout" / "src" / "charlie_work" / "__init__.py"

    def _fake_run(cmd, *, cwd, timeout, env=None):
        del cwd, timeout, env
        return subprocess.CompletedProcess(cmd, 0, stdout=f"{outside}\n", stderr="")

    monkeypatch.setattr(gate, "_run", _fake_run)

    result = gate._assert_import_anchor(repo)

    assert result is not None
    assert result.block is True
    assert gate.IMPORT_ANCHOR_FAILURE in result.reason
    assert str(outside) in result.reason
    assert str(repo) in result.reason


def test_assert_import_anchor_blocks_on_probe_import_failure(gate, repo, monkeypatch):
    def _fake_run(cmd, *, cwd, timeout, env=None):
        del cwd, timeout, env
        return subprocess.CompletedProcess(
            cmd, 1, stdout="", stderr="ModuleNotFoundError: No module named 'charlie_work'\n"
        )

    monkeypatch.setattr(gate, "_run", _fake_run)

    result = gate._assert_import_anchor(repo)

    assert result is not None
    assert result.block is True
    assert gate.IMPORT_ANCHOR_FAILURE in result.reason
    assert "ModuleNotFoundError" in result.reason


def test_assert_import_anchor_blocks_on_empty_probe_output(gate, repo, monkeypatch):
    """A namespace-package resolution has no __file__ -- the probe prints an
    empty line, which must be a distinct failure, never a silent pass."""

    def _fake_run(cmd, *, cwd, timeout, env=None):
        del cwd, timeout, env
        return subprocess.CompletedProcess(cmd, 0, stdout="\n", stderr="")

    monkeypatch.setattr(gate, "_run", _fake_run)

    result = gate._assert_import_anchor(repo)

    assert result is not None
    assert result.block is True
    assert gate.IMPORT_ANCHOR_FAILURE in result.reason


# ---------------------------------------------------------------------------
# _run_targeted_tests: interpreter-direct pytest, gated env, anchor ordering,
# and no ``uv`` anywhere (mocked subprocess layer).
# ---------------------------------------------------------------------------


def test_run_targeted_tests_invokes_interpreter_directly_with_gated_env(gate, repo, monkeypatch):
    captured: dict[str, object] = {}

    def _fake_run(cmd, *, cwd, timeout, env=None):
        del cwd, timeout
        if "-c" in cmd:
            return _probe_ok(repo)
        captured["cmd"] = cmd
        captured["env"] = env
        return subprocess.CompletedProcess(cmd, 0, stdout="1 passed\n", stderr="")

    monkeypatch.setattr(gate, "_run", _fake_run)
    monkeypatch.delenv("PYTHONPATH", raising=False)

    result = gate._run_targeted_tests(repo, ("tests/test_x.py",))

    assert result.block is False
    cmd = captured["cmd"]
    assert cmd[:3] == [sys.executable, "-m", "pytest"], cmd
    assert "uv" not in cmd
    env = captured["env"]
    assert env["PYTHONPATH"].split(os.pathsep)[0] == str(repo / "src")


def test_run_targeted_tests_never_reaches_pytest_when_anchor_fails(gate, repo, monkeypatch):
    """A failed anchor must short-circuit before pytest: the failure is the
    distinct anchor error, never a test-failure dump."""
    outside = repo.parent / "main-checkout" / "src" / "charlie_work" / "__init__.py"
    calls: list[list[str]] = []

    def _fake_run(cmd, *, cwd, timeout, env=None):
        del cwd, timeout, env
        calls.append(cmd)
        if "-c" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{outside}\n", stderr="")
        raise AssertionError(f"pytest must not run when the anchor fails: {cmd}")

    monkeypatch.setattr(gate, "_run", _fake_run)

    result = gate._run_targeted_tests(repo, ("tests/test_x.py",))

    assert result.block is True
    assert gate.IMPORT_ANCHOR_FAILURE in result.reason
    assert "targeted tests failed" not in result.reason
    assert len(calls) == 1, "only the probe should have run"


def test_run_ruff_invokes_interpreter_directly(gate, repo, monkeypatch):
    captured: list[list[str]] = []

    def _fake_run(cmd, *, cwd, timeout, env=None):
        del cwd, timeout, env
        captured.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(gate, "_run", _fake_run)

    result = gate._run_ruff(repo, ("src/charlie_work/foo.py",))

    assert result.block is False
    assert len(captured) == 2  # ruff check + ruff format --check
    for cmd in captured:
        assert cmd[:3] == [sys.executable, "-m", "ruff"], cmd
        assert "uv" not in cmd


def test_evaluate_invokes_no_uv_subprocess_anywhere(gate, repo, monkeypatch):
    """The gate must never create or sync a virtualenv as a side effect --
    structurally guaranteed by never invoking ``uv`` on any check path."""
    (repo / "tests").mkdir()
    (repo / "tests" / "test_x.py").write_text("def test_x():\n    assert True\n", encoding="utf-8")
    (repo / "session.py").write_text("x = 1\n", encoding="utf-8")
    captured: list[list[str]] = []

    def _fake_run(cmd, *, cwd, timeout, env=None):
        del cwd, timeout, env
        captured.append(cmd)
        if cmd[:2] == ["git", "symbolic-ref"]:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")
        if "status" in cmd:
            return subprocess.CompletedProcess(
                cmd, 0, stdout=" M session.py\n?? tests/test_x.py\n", stderr=""
            )
        if "-c" in cmd:
            return _probe_ok(repo)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(gate, "_run", _fake_run)

    result = gate._evaluate(repo)

    assert result.block is False
    assert captured, "the mocked run must have been exercised"
    for cmd in captured:
        assert Path(cmd[0]).stem != "uv", f"uv must never be invoked: {cmd}"


# ---------------------------------------------------------------------------
# Real subprocess coverage: the fake worktree layout from the acceptance
# criteria. "main" and "worktree" define a different constant; the gate must
# observe the worktree's value. The ambient PYTHONPATH points at "main"'s
# src -- a stand-in for the shared editable install that made the gate test
# the wrong tree.
# ---------------------------------------------------------------------------


def test_run_targeted_tests_real_imports_charlie_work_from_gated_worktree(
    gate, tmp_path, monkeypatch
):
    main_tree = _make_src_tree(tmp_path / "main", "main")
    worktree = _make_src_tree(tmp_path / "worktree", "worktree")
    (worktree / "tests").mkdir()
    (worktree / "tests" / "test_marker.py").write_text(
        "import charlie_work\n\n\n"
        "def test_marker_is_the_gated_tree():\n"
        "    assert charlie_work.MARKER == 'worktree'\n",
        encoding="utf-8",
    )
    # If the gate did not prepend the gated tree's src, this inherited entry
    # resolves charlie_work to "main" and the marker test fails -- the same
    # ordering the shared editable install produced before #1793.
    monkeypatch.setenv("PYTHONPATH", str(main_tree / "src"))

    result = gate._run_targeted_tests(worktree, ("tests/test_marker.py",))

    assert result.block is False, result.reason


def test_run_targeted_tests_real_blocks_distinctly_when_anchor_points_wrong_tree(
    gate, tmp_path, monkeypatch
):
    """Real-subprocess control for the acceptance criteria: the gated tree
    has no ``src/charlie_work`` of its own, so the probe resolves
    ``charlie_work`` via the inherited PYTHONPATH into the "main" tree --
    outside the gated toplevel. The gate must fail with the distinct
    import-anchor error and never run pytest."""
    main_tree = _make_src_tree(tmp_path / "main", "main")
    gated = tmp_path / "worktree"  # deliberately no src/charlie_work
    (gated / "tests").mkdir(parents=True)
    (gated / "tests" / "test_marker.py").write_text(
        "def test_never_runs():\n    assert False\n", encoding="utf-8"
    )
    monkeypatch.setenv("PYTHONPATH", str(main_tree / "src"))

    result = gate._run_targeted_tests(gated, ("tests/test_marker.py",))

    assert result.block is True
    assert gate.IMPORT_ANCHOR_FAILURE in result.reason
    # The planted always-failing test must never have executed: the reason
    # is the anchor failure alone, not a wall of wrong-tree test failures.
    assert "targeted tests failed" not in result.reason
    assert "test_never_runs" not in result.reason
