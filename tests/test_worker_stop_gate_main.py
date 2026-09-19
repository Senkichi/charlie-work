"""``main()`` entry-point tests for scripts/worker_stop_gate.py.

Split verbatim out of ``tests/test_worker_stop_gate.py`` (issue #1573,
Track 1 shoulder). Shared helpers and the ``gate``/``repo`` fixtures live
in ``tests/_worker_stop_gate_fixtures.py`` -- the ``tests/_*.py``
hoisted-fixture convention is the sanctioned import target for shared
test helpers (see ``tests/test_zero_cross_test_import_guard.py``).

Covers the fail-closed error path, the pre-exhaustion fast path,
``_run`` error wrapping, and end-to-end runs with a fully mocked
subprocess layer.
"""

from __future__ import annotations

import io
import subprocess
from pathlib import Path

import pytest

from _worker_stop_gate_fixtures import _stdin, gate as gate, repo as repo


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
