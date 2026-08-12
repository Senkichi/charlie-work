"""Tests for scripts/verify_events.py (issue #718).

The script has no ``if __name__ == "__main__":`` guard -- it is a top-level
script that runs its verification logic the moment it is loaded, using
``sys.argv[1]`` for the state.json path. ``_run_verify_events`` below loads
it the same way tests/test_heartbeat_check.py and
tests/test_backfill_stale_rework_briefs.py load their scripts (via
``_script_loader.load_script_module``, without adding scripts/ to
sys.path), but additionally sets ``sys.argv`` first and catches the
``SystemExit`` the script raises on its failure paths.

Regression coverage for #718: ``scripts/verify_events.py`` printed
``=== Verification PASSED ===`` when pointed at a state.json path with no
events.db -- including a path that doesn't exist at all -- because every
read helper it calls resolves through ``instrumentation._get_db``, which
creates a brand-new empty events.db (and any missing parent directories)
for any path that doesn't exist yet. The fix adds explicit pre-existence
checks, run before anything that can create the database, plus an
all-zero-results check after the read.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

import pytest

from _script_loader import load_script_module
from charlie_work.instrumentation import _db_path, close_db, log_event, record_loop_pass


_SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "verify_events.py"


@pytest.fixture(autouse=True)
def _close_db_after_test(tmp_path: Path) -> None:
    """Mirror test_instrumentation.py's teardown: avoid cross-test connection reuse."""
    yield
    close_db(tmp_path / "state.json")
    close_db(tmp_path / "nested" / "state.json")


def _run_verify_events(
    state_path: Path, script_path: Path = _SCRIPT_PATH
) -> tuple[int | None, ModuleType | None]:
    """Execute scripts/verify_events.py as if invoked as ``verify_events.py <state_path>``.

    ``script_path`` defaults to the real ``scripts/verify_events.py`` but can
    be overridden (see ``test_loader_registers_module_before_exec_module``
    below) to load a synthetic script through the identical recipe without
    touching the production file.

    Returns ``(exit_code, module)``. ``exit_code`` is ``None`` when the
    script ran to completion without calling ``sys.exit`` (the PASSED
    path); otherwise it is the code passed to ``sys.exit``. Output goes to
    the real stdout/stderr, so callers should wrap this in ``capsys``.

    Loading is delegated to ``_script_loader.load_script_module`` so the
    ``sys.modules`` registration and ``sys.argv`` save/restore are handled
    in one place (issue #1028).
    """
    try:
        module = load_script_module(
            script_path,
            "verify_events_under_test",
            argv=["verify_events.py", str(state_path)],
        )
        return None, module
    except SystemExit as exc:
        return exc.code, None


def test_fails_when_state_path_does_not_exist(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """The exact reported scenario: a state.json path that doesn't exist at all."""
    state_path = tmp_path / "nested" / "state.json"

    code, _ = _run_verify_events(state_path)

    captured = capsys.readouterr()
    assert code == 1
    assert "PASSED" not in captured.out
    assert str(state_path) in captured.err
    # No side effect: _get_db's mkdir(parents=True) must never have run.
    assert not (tmp_path / "nested").exists()
    assert not _db_path(state_path).exists()


def test_fails_when_events_db_absent(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """state.json exists, but its events.db has never been created."""
    state_path = tmp_path / "state.json"
    state_path.write_text("{}")

    code, _ = _run_verify_events(state_path)

    captured = capsys.readouterr()
    assert code == 1
    assert "PASSED" not in captured.out
    assert str(_db_path(state_path)) in captured.err
    assert not _db_path(state_path).exists()


def test_fails_when_db_present_but_all_zero(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """A pre-existing, genuinely-opened events.db with zero events and zero loop
    passes must not report PASSED either -- it is indistinguishable from being
    pointed at the wrong tree, which is the exact false positive #718 reports.
    """
    state_path = tmp_path / "state.json"
    state_path.write_text("{}")
    # Legitimately create the DB via the real instrumentation path (not the
    # script), so the DB pre-exists independent of anything verify_events.py
    # does.
    from charlie_work.instrumentation import _get_db

    conn = _get_db(state_path)
    assert conn is not None
    close_db(state_path)
    assert _db_path(state_path).exists()

    code, _ = _run_verify_events(state_path)

    captured = capsys.readouterr()
    assert code == 1
    assert "PASSED" not in captured.out
    assert "zero events and zero loop passes" in captured.err


def test_passes_with_recorded_events(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """A real, pre-existing, non-empty database is exactly what this script exists
    to confirm -- it must still report PASSED.
    """
    state_path = tmp_path / "state.json"
    state_path.write_text("{}")
    log_event(state_path, "test_event", {"key": "value"}, repo="test-repo")
    close_db(state_path)

    code, _ = _run_verify_events(state_path)

    captured = capsys.readouterr()
    assert code is None
    assert "=== Verification PASSED ===" in captured.out
    assert "Total events captured: 1" in captured.out


def test_passes_with_recorded_loop_pass_but_no_events(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """A recorded loop pass with zero standalone events is a real partial run,
    not an all-zero non-signal -- it must still report PASSED.
    """
    state_path = tmp_path / "state.json"
    state_path.write_text("{}")
    # record_loop_pass is a two-call INSERT-then-UPDATE protocol (see its
    # docstring): completed_at=None inserts the row, a second call with
    # completed_at set fills in the summary columns.
    record_loop_pass(state_path, "cid-1", "2026-01-01T00:00:00Z")
    record_loop_pass(
        state_path,
        "cid-1",
        "2026-01-01T00:00:00Z",
        "2026-01-01T00:01:00Z",
        ok=True,
        elapsed_seconds=60.0,
        error_count=0,
        merge_count=0,
        review_count=0,
    )
    close_db(state_path)

    code, _ = _run_verify_events(state_path)

    captured = capsys.readouterr()
    assert code is None
    assert "=== Verification PASSED ===" in captured.out
    assert "Loop passes recorded: 1" in captured.out


def test_loader_registers_module_before_exec_module(tmp_path: Path) -> None:
    """Regression test for #1023.

    scripts/verify_events.py has zero @dataclass definitions today, so the
    real script can't exercise this. Instead this loads a synthetic script,
    through the exact same _run_verify_events recipe, that carries the
    trigger pair CLAUDE.md mandates for config/value-object types: a frozen
    dataclass plus ``from __future__ import annotations``. If the loader
    doesn't register the module in sys.modules before exec_module, class
    creation dies with ``AttributeError: 'NoneType' object has no attribute
    '__dict__'`` while resolving the dataclass's string annotations -- a
    collection-time failure, not a test-body failure.
    """
    synthetic_script = tmp_path / "synthetic_dataclass_script.py"
    synthetic_script.write_text(
        "from __future__ import annotations\n"
        "from dataclasses import dataclass\n"
        "\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class RegressionProbe:\n"
        "    value: int\n"
    )
    state_path = tmp_path / "state.json"

    code, module = _run_verify_events(state_path, script_path=synthetic_script)

    assert code is None
    assert module is not None
    probe = module.RegressionProbe(value=1)
    assert probe.value == 1
