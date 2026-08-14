"""Tests for ``charlie_work.ci_fleet_anchor.ci_fleet_provenance_snapshot`` (issue #954).

The snapshot records what ``ci_fleet`` the supervisor *actually imported* --
``ci_fleet.__file__`` plus the sibling repo's HEAD, branch, and dirty-state --
so the editable-working-tree coupling is attributable rather than silent.

These tests exercise the snapshot function in isolation: ``run_command`` is
injected so no real ``git`` subprocess runs, and ``declared_ci_fleet_root`` is
monkeypatched to control the abstention vs. inspection path. ``ci_fleet`` is
importable in this venv, so ``ci_fleet_file`` is a real path in every case
except the import-failure simulation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from charlie_work.ci_fleet_anchor import (
    CiFleetProvenanceSnapshot,
    ci_fleet_provenance_payload,
    ci_fleet_provenance_snapshot,
)
from charlie_work.subprocess_runner import RunResult


def _ok(stdout: str = "") -> RunResult:
    return RunResult(returncode=0, stdout=stdout, stderr="")


def _fail(error: str = "git failed") -> RunResult:
    return RunResult(returncode=128, stdout="", stderr="fatal: not a git repo", error=error)


def _make_run_command(
    *,
    head: RunResult,
    branch: RunResult,
    status: RunResult,
) -> pytest.FixtureRequest:
    """Build a fake ``run_command`` that returns canned results by argv."""

    def _runner(command: list[str], *, cwd: Path, timeout_seconds: int) -> RunResult:
        if "rev-parse" in command and "--abbrev-ref" in command:
            return branch
        if "rev-parse" in command:
            return head
        if "status" in command:
            return status
        raise AssertionError(f"unexpected command: {command}")

    return _runner


# ---------------------------------------------------------------------------
# 1. Clean sibling — the normal production shape
# ---------------------------------------------------------------------------


def test_clean_sibling_records_head_branch_and_not_dirty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A clean sibling on main: head/branch recorded, dirty=False, error=None."""
    sibling_src = tmp_path / "ci_runners" / "src"
    sibling_src.mkdir(parents=True)
    monkeypatch.setattr("charlie_work.ci_fleet_anchor.declared_ci_fleet_root", lambda: sibling_src)

    run_command = _make_run_command(
        head=_ok("abc123def\n"),
        branch=_ok("main\n"),
        status=_ok(""),
    )
    snapshot = ci_fleet_provenance_snapshot(run_command=run_command)

    assert snapshot.ci_fleet_file is not None
    assert snapshot.sibling_root == str(sibling_src.parent)
    assert snapshot.sibling_head == "abc123def"
    assert snapshot.sibling_branch == "main"
    assert snapshot.sibling_dirty is False
    assert snapshot.error is None


# ---------------------------------------------------------------------------
# 2. Dirty sibling — the hazard the issue is about
# ---------------------------------------------------------------------------


def test_dirty_sibling_records_dirty_true(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A dirty working tree: sibling_dirty=True -- the signal #954 makes visible."""
    sibling_src = tmp_path / "ci_runners" / "src"
    sibling_src.mkdir(parents=True)
    monkeypatch.setattr("charlie_work.ci_fleet_anchor.declared_ci_fleet_root", lambda: sibling_src)

    run_command = _make_run_command(
        head=_ok("abc123def\n"),
        branch=_ok("main\n"),
        status=_ok(" M src/ci_fleet/runners.py\n"),
    )
    snapshot = ci_fleet_provenance_snapshot(run_command=run_command)

    assert snapshot.sibling_dirty is True
    assert snapshot.error is None


# ---------------------------------------------------------------------------
# 3. Abstention — no declared root (e.g. running from a worktree)
# ---------------------------------------------------------------------------


def test_abstention_when_no_declared_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No declared ci-fleet path source: sibling fields None, error=None.

    Same abstention as ``declared_ci_fleet_root``: from a worktree that is not
    a sibling of the real checkout, ``../ci_runners`` does not exist and the
    honest answer is "cannot tell", not "something is wrong". ci_fleet_file is
    still recorded -- it is the one fact available without the sibling.
    """
    monkeypatch.setattr("charlie_work.ci_fleet_anchor.declared_ci_fleet_root", lambda: None)
    snapshot = ci_fleet_provenance_snapshot(run_command=lambda *a, **k: _ok())

    assert snapshot.ci_fleet_file is not None
    assert snapshot.sibling_root is None
    assert snapshot.sibling_head is None
    assert snapshot.sibling_branch is None
    assert snapshot.sibling_dirty is None
    assert snapshot.error is None


# ---------------------------------------------------------------------------
# 4. Git probe failure — error recorded, not raised
# ---------------------------------------------------------------------------


def test_git_failure_records_error_and_never_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A git probe failure: error set, fields None, no exception propagated."""
    sibling_src = tmp_path / "ci_runners" / "src"
    sibling_src.mkdir(parents=True)
    monkeypatch.setattr("charlie_work.ci_fleet_anchor.declared_ci_fleet_root", lambda: sibling_src)

    run_command = _make_run_command(
        head=_fail("index lock"),
        branch=_ok("main\n"),
        status=_fail("index lock"),
    )
    snapshot = ci_fleet_provenance_snapshot(run_command=run_command)

    assert snapshot.sibling_root == str(sibling_src.parent)
    assert snapshot.sibling_head is None  # head probe failed
    assert snapshot.sibling_branch == "main"
    assert snapshot.sibling_dirty is None  # status probe failed
    assert snapshot.error is not None
    assert "rev-parse" in snapshot.error
    assert "status" in snapshot.error


# ---------------------------------------------------------------------------
# 4b. Import failure — ci_fleet cannot be imported (review finding)
# ---------------------------------------------------------------------------


def test_import_failure_returns_none_file_with_populated_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``import ci_fleet`` raises, the snapshot records the failure, never raises.

    Review finding for #954: the import-failure branch in
    ``ci_fleet_provenance_snapshot`` had no regression test. Setting
    ``sys.modules['ci_fleet'] = None`` makes ``import ci_fleet`` raise
    ``ModuleNotFoundError`` ("import of ci_fleet halted; None in sys.modules"),
    exercising the ``except`` branch. The snapshot must return
    ``ci_fleet_file=None`` with a populated ``error`` field rather than
    propagating -- a startup probe must not break the supervisor's entry path.
    """
    monkeypatch.setitem(sys.modules, "ci_fleet", None)
    snapshot = ci_fleet_provenance_snapshot(run_command=lambda *a, **k: _ok())

    assert snapshot.ci_fleet_file is None
    assert snapshot.sibling_root is None
    assert snapshot.sibling_head is None
    assert snapshot.sibling_branch is None
    assert snapshot.sibling_dirty is None
    assert snapshot.error is not None
    assert "import ci_fleet raised" in snapshot.error
    # The exception type is named so the failure is diagnosable from the event.
    assert "ModuleNotFoundError" in snapshot.error


# ---------------------------------------------------------------------------
# 4c. Shared payload helper — single source of truth for the event shape
# ---------------------------------------------------------------------------


def test_provenance_payload_contains_all_six_fields() -> None:
    """``ci_fleet_provenance_payload`` returns exactly the six attributable fields.

    Review finding for #954: the payload dict was verbatim-duplicated across
    ``fleet_dispatch._record_ci_fleet_provenance`` and ``supervise.run_supervised``.
    The shared helper is now the single source of truth; this test pins its
    shape so a caller that drops or renames a field is caught.
    """
    snapshot = CiFleetProvenanceSnapshot(
        ci_fleet_file="/x/ci_fleet/__init__.py",
        sibling_root="/x/ci_runners",
        sibling_head="abc123",
        sibling_branch="main",
        sibling_dirty=False,
        error=None,
    )
    payload = ci_fleet_provenance_payload(snapshot)
    assert payload == {
        "ci_fleet_file": "/x/ci_fleet/__init__.py",
        "sibling_root": "/x/ci_runners",
        "sibling_head": "abc123",
        "sibling_branch": "main",
        "sibling_dirty": False,
        "error": None,
    }


def test_provenance_payload_preserves_error_and_none_fields() -> None:
    """The payload helper forwards ``None`` and error strings verbatim.

    The abstention and failure shapes both rely on ``None`` sibling fields plus
    a populated ``error``; the helper must not coerce or drop them.
    """
    snapshot = CiFleetProvenanceSnapshot(
        ci_fleet_file=None,
        sibling_root=None,
        sibling_head=None,
        sibling_branch=None,
        sibling_dirty=None,
        error="import ci_fleet raised ModuleNotFoundError: halted",
    )
    payload = ci_fleet_provenance_payload(snapshot)
    assert payload["ci_fleet_file"] is None
    assert payload["sibling_dirty"] is None
    assert payload["error"] == "import ci_fleet raised ModuleNotFoundError: halted"


# ---------------------------------------------------------------------------
# 5. Never raises — the startup-safety property
# ---------------------------------------------------------------------------


def test_snapshot_never_raises_on_run_command_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crashing run_command must not propagate -- the probe is in the startup path."""

    def _crash(command: list[str], *, cwd: Path, timeout_seconds: int) -> RunResult:
        raise OSError("boom")

    sibling_src = tmp_path / "ci_runners" / "src"
    sibling_src.mkdir(parents=True)
    monkeypatch.setattr("charlie_work.ci_fleet_anchor.declared_ci_fleet_root", lambda: sibling_src)

    # Must not raise even though run_command raises -- the snapshot wraps the
    # git probes so a probe crash is reported as an error value, not propagated.
    snapshot = ci_fleet_provenance_snapshot(run_command=_crash)

    assert snapshot.ci_fleet_file is not None
    assert snapshot.sibling_root == str(sibling_src.parent)
    assert snapshot.sibling_head is None
    assert snapshot.sibling_dirty is None
    assert snapshot.error is not None
    assert "OSError" in snapshot.error


# ---------------------------------------------------------------------------
# 6. Frozen dataclass — CLAUDE.md invariant
# ---------------------------------------------------------------------------


def test_snapshot_is_frozen_dataclass() -> None:
    """Config/value objects are frozen dataclasses (CLAUDE.md invariant)."""
    snapshot = CiFleetProvenanceSnapshot(
        ci_fleet_file=None,
        sibling_root=None,
        sibling_head=None,
        sibling_branch=None,
        sibling_dirty=None,
    )
    with pytest.raises(Exception):
        snapshot.ci_fleet_file = "x"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 7. Real-process smoke — never raises with defaults
# ---------------------------------------------------------------------------


def test_real_process_smoke_never_raises() -> None:
    """Calling with defaults in the real test process must not raise.

    From a worktree, ``declared_ci_fleet_root`` abstains (returns None), so
    this exercises the abstention path with the real ``run_captured``. From
    the main checkout it would run real git probes against the sibling. Either
    way the call must be safe.
    """
    snapshot = ci_fleet_provenance_snapshot()

    assert isinstance(snapshot, CiFleetProvenanceSnapshot)
    # ci_fleet is importable in this venv, so __file__ is always set.
    assert snapshot.ci_fleet_file is not None
