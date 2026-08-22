"""Tests for merge_check, carved out of test_charlie_work.py (#1284)."""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

import charlie_work.state as state_module
from _merge_tripwire_fixtures import _merge_check_app, _write_decision
from charlie_work import cli


@contextlib.contextmanager
def _hold_state_lock(lock_path: Path) -> Any:
    """Hold a real, competing byte-range/exclusive lock on the state lock file.

    This is used to force ``state_lock`` to time out without involving another
    process, while still exercising the real platform locking primitive.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if not lock_path.exists():
        lock_path.write_bytes(b"\x00")
    handle = lock_path.open("r+b")
    try:
        if sys.platform == "win32":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        try:
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        handle.close()


def test_merge_check_fails_closed_with_no_decision(tmp_path: Path) -> None:
    """Issue #894. No recorded decision must never read as authorization."""
    app, _, fake_gh = _merge_check_app(tmp_path)

    result = app.merge_check(456)

    assert result.ok is False
    assert result.data["authorized"] is False
    assert result.data["reason"] == "no_decision"
    # A preflight is a pure question: it must not merge as a side effect.
    assert fake_gh.merged == []


def test_merge_check_rejects_request_changes(tmp_path: Path) -> None:
    """The PR #759 shape: a recorded `request_changes` at the live head."""
    app, _, _ = _merge_check_app(tmp_path)
    _write_decision(
        tmp_path, 456, {"decision": "request_changes", "reviewed_head_sha": "sha-abc123"}
    )

    result = app.merge_check(456)

    assert result.ok is False
    assert result.data["reason"] == "not_approved"
    assert result.data["decision"] == "request_changes"


def test_merge_check_rejects_approval_at_stale_head(tmp_path: Path) -> None:
    """Approved-but-moved is a distinct outcome from never-approved: it routes to
    re-review, not to review. Collapsing them would make the preflight
    unactionable."""
    app, _, _ = _merge_check_app(tmp_path)
    _write_decision(tmp_path, 456, {"decision": "approved", "reviewed_head_sha": "sha-old"})

    result = app.merge_check(456)

    assert result.ok is False
    assert result.data["reason"] == "head_moved"
    assert result.data["reviewed_head_sha"] == "sha-old"
    assert result.data["live_head_sha"] == "sha-abc123"


def test_merge_check_rejects_malformed_decision(tmp_path: Path) -> None:
    """Unparseable state fails closed rather than laundering uncertainty.

    Issue #1362 Stage 1: a corrupt flat file with no round-archive fallback
    now resolves to the same "missing" sentinel as no file at all
    (review_decision.resolve_decision_payload), so this reports "no_decision"
    rather than the old distinct "invalid_decision" reason -- the
    authorization outcome (fails closed) is unchanged.
    """
    app, _, _ = _merge_check_app(tmp_path)
    decision_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    decision_dir.mkdir(parents=True)
    (decision_dir / "review-decision.json").write_text("{not json", encoding="utf-8")

    result = app.merge_check(456)

    assert result.ok is False
    assert result.data["reason"] == "no_decision"


def test_merge_check_authorizes_approved_at_current_head(tmp_path: Path) -> None:
    """The positive control: without this, every assertion above would also pass
    against a merge_check that returned False unconditionally."""
    app, _, fake_gh = _merge_check_app(tmp_path)
    _write_decision(tmp_path, 456, {"decision": "approved", "reviewed_head_sha": "sha-abc123"})

    result = app.merge_check(456)

    assert result.ok is True
    assert result.data["authorized"] is True
    assert result.data["reason"] == "approved_at_head"
    assert fake_gh.merged == []


def test_merge_check_rejects_already_merged_pr(tmp_path: Path) -> None:
    app, _, fake_gh = _merge_check_app(tmp_path)
    _write_decision(tmp_path, 456, {"decision": "approved", "reviewed_head_sha": "sha-abc123"})
    for pr in fake_gh.prs:
        if pr.get("number") == 456:
            pr["state"] = "MERGED"

    result = app.merge_check(456)

    assert result.ok is False
    assert result.data["reason"] == "already_merged"


def test_merge_check_is_reachable_through_the_cli(tmp_path: Path) -> None:
    """Wiring check (L3). Every other test here calls merge_check directly and
    would still pass if the subcommand were never registered or dispatched —
    which is exactly the failure mode that makes a control inert. A hook shelling
    out to `charlie merge-check` reaches it through this path only."""
    app, _, _ = _merge_check_app(tmp_path)
    _write_decision(tmp_path, 456, {"decision": "approved", "reviewed_head_sha": "sha-abc123"})

    args = cli.build_parser().parse_args(["merge-check", "456"])
    assert args.command == "merge-check"
    assert args.pr == 456

    result = cli.run_command(app, args)
    assert result.ok is True
    assert result.data["reason"] == "approved_at_head"


def test_merge_check_does_not_fail_open_when_state_lock_is_held(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """merge_check must not carry @_guard_state_lock.

    That guard's contract (issue #398) is to return a *successful* skip
    (ok=True, reason='state_lock_busy') so a contended pass is a no-op rather
    than a crash. Correct for state-writing commands; catastrophic here, because
    a caller gating a merge on exit status would read "lock was busy" as
    "authorized". An authorization preflight must answer the question or fail
    closed — never succeed vacuously.

    This is not hypothetical: adding merge_check directly above merge_ready
    silently transferred merge_ready's decorator onto it, and only merge_ready's
    own lock-guard test noticed. Nothing asserted the property from this side.
    """
    monkeypatch.setattr(state_module, "_LOCK_TIMEOUT_SECONDS", 0.05)
    app, paths, _ = _merge_check_app(tmp_path)
    _write_decision(
        tmp_path, 456, {"decision": "request_changes", "reviewed_head_sha": "sha-abc123"}
    )

    state_path = paths.state_file
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps({"version": 1, "issues": {}, "prs": {}, "events": []}), encoding="utf-8"
    )
    lock_path = state_path.with_suffix(state_path.suffix + ".lock")

    with _hold_state_lock(lock_path):
        result = app.merge_check(456)

    assert result.ok is False
    assert result.data["reason"] == "not_approved"
    assert result.data.get("pass_skipped") is not True
