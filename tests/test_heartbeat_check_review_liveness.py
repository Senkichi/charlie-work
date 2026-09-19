"""Review-liveness tests for ``scripts/heartbeat_check.py``.

Split out of ``tests/test_heartbeat_check.py`` (issue #1556, Track-1):
``_review_claim_timestamp`` / ``_reviewer_pid_alive`` /
``check_review_liveness`` coverage, including the escalated-PR and
completed-rebuild regression cases.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from _heartbeat_check_fixtures import (
    _iso,
    _load_heartbeat_check,
    _make_repo,
)


@pytest.fixture(scope="module")
def hb() -> ModuleType:
    return _load_heartbeat_check()


class FakePsutil:
    """Stub psutil surface used by heartbeat_check._reviewer_pid_alive."""

    class Error(Exception):
        pass

    class NoSuchProcess(Error):
        pass

    class AccessDenied(Error):
        pass

    def __init__(self, mapping: dict[int, tuple[bool, float | None]]) -> None:
        self.mapping = mapping

    def pid_exists(self, pid: int) -> bool:
        return self.mapping.get(pid, (False, None))[0]

    def Process(self, pid: int) -> "FakeProcess":
        exists, create_time = self.mapping.get(pid, (False, None))
        if not exists:
            raise self.NoSuchProcess(pid)
        return FakeProcess(create_time, self.NoSuchProcess, pid)


class FakeProcess:
    def __init__(
        self,
        create_time_value: float | None,
        no_such_process_cls: type[Exception],
        pid: int,
    ) -> None:
        self._create_time_value = create_time_value
        self._no_such_process_cls = no_such_process_cls
        self.pid = pid

    def create_time(self) -> float:
        if self._create_time_value is None:
            raise self._no_such_process_cls(self.pid)
        return self._create_time_value


def _write_state(state_dir: Path, pr_number: int, pr_state: dict[str, Any]) -> None:
    state = {"version": 1, "prs": {str(pr_number): pr_state}}
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")


def _make_pr_dirs(state_dir: Path, pr_number: int, *, pr_mtime: float | None = None) -> Path:
    pr_dir = state_dir / "prs" / f"pr-{pr_number}"
    pr_dir.mkdir(parents=True, exist_ok=True)
    (pr_dir / "pr.json").write_text("{}", encoding="utf-8")
    (pr_dir / "review-decision.json").write_text(
        json.dumps({"decision": "pending"}), encoding="utf-8"
    )
    if pr_mtime is not None:
        os.utime(pr_dir, (pr_mtime, pr_mtime))
    return pr_dir


def _patch_gh(monkeypatch: Any, hb: ModuleType, numbers: list[int]) -> None:
    def fake_run_gh_json(args: list[str], cwd: Path) -> tuple[bool, Any, str]:
        return True, [{"number": n} for n in numbers], ""

    monkeypatch.setattr(hb, "run_gh_json", fake_run_gh_json)


def test_review_claim_timestamp_dispatched(hb: ModuleType) -> None:
    pr_state = {
        "review_dispatch_status": "review_dispatch_dispatched",
        "review_dispatched_at": "2026-07-20T22:00:00Z",
        "review_dispatch_pending_at": "2026-07-20T21:00:00Z",
        "review_dispatch_failed_at": "2026-07-20T20:00:00Z",
    }
    assert hb._review_claim_timestamp(pr_state) == "2026-07-20T22:00:00Z"


def test_review_claim_timestamp_pending(hb: ModuleType) -> None:
    pr_state = {
        "review_dispatch_status": "review_dispatch_pending",
        "review_dispatch_pending_at": "2026-07-20T21:05:00Z",
    }
    assert hb._review_claim_timestamp(pr_state) == "2026-07-20T21:05:00Z"


def test_review_claim_timestamp_failed(hb: ModuleType) -> None:
    pr_state = {
        "review_dispatch_status": "review_dispatch_failed",
        "review_dispatch_failed_at": "2026-07-20T20:15:00Z",
    }
    assert hb._review_claim_timestamp(pr_state) == "2026-07-20T20:15:00Z"


def test_review_claim_timestamp_unknown_uses_newest(hb: ModuleType) -> None:
    pr_state = {
        "review_dispatch_status": None,
        "review_dispatched_at": "2026-07-20T22:00:00Z",
        "review_dispatch_pending_at": "2026-07-20T23:00:00Z",
    }
    assert hb._review_claim_timestamp(pr_state) == "2026-07-20T23:00:00Z"


def test_reviewer_pid_alive_none_without_pid(hb: ModuleType, monkeypatch: Any) -> None:
    monkeypatch.setattr(hb, "psutil", FakePsutil({}))
    assert hb._reviewer_pid_alive({}) is None


def test_reviewer_pid_alive_false_for_dead_pid(hb: ModuleType, monkeypatch: Any) -> None:
    monkeypatch.setattr(hb, "psutil", FakePsutil({12345: (False, None)}))
    assert hb._reviewer_pid_alive({"reviewer_pid": 12345}) is False


def test_reviewer_pid_alive_true_when_alive(hb: ModuleType, monkeypatch: Any) -> None:
    monkeypatch.setattr(hb, "psutil", FakePsutil({12345: (True, 1000.0)}))
    assert (
        hb._reviewer_pid_alive({"reviewer_pid": 12345, "reviewer_process_start_time": 1000.0})
        is True
    )


def test_reviewer_pid_alive_false_when_recycled(hb: ModuleType, monkeypatch: Any) -> None:
    monkeypatch.setattr(hb, "psutil", FakePsutil({12345: (True, 2000.0)}))
    assert (
        hb._reviewer_pid_alive({"reviewer_pid": 12345, "reviewer_process_start_time": 1000.0})
        is False
    )


def test_reviewer_pid_alive_true_on_indeterminate_start_time(
    hb: ModuleType, monkeypatch: Any
) -> None:
    monkeypatch.setattr(hb, "psutil", FakePsutil({12345: (True, None)}))
    assert (
        hb._reviewer_pid_alive({"reviewer_pid": 12345, "reviewer_process_start_time": 1000.0})
        is True
    )


def test_check_review_liveness_uses_state_timestamp_not_packet_mtime(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """Regression for issue #517.

    The packet directory mtime is ancient, but state.json carries a recent
    review_dispatched_at for a live PID.  The heartbeat must report the age of
    the current dispatch attempt, not the monotonically growing packet age.
    """
    repo = _make_repo(hb, tmp_path)
    _patch_gh(monkeypatch, hb, [513])
    _make_pr_dirs(
        repo.state_dir,
        513,
        pr_mtime=(datetime(2020, 1, 1, tzinfo=timezone.utc)).timestamp(),
    )
    _write_state(
        repo.state_dir,
        513,
        {
            "review_dispatch_status": "review_dispatch_dispatched",
            "review_dispatched_at": _iso(10),
            "reviewer_pid": 12345,
            "reviewer_process_start_time": 1000.0,
        },
    )
    monkeypatch.setattr(hb, "psutil", FakePsutil({12345: (True, 1000.0)}))

    report = hb.Report()
    hb.check_review_liveness(report, repo)

    assert not report.anomaly
    assert report.lines and "review-liveness" in report.lines[0]
    assert "pid=12345 alive" in report.lines[0]
    assert "open_claims=1" in report.lines[0]


def test_check_review_liveness_flags_dead_pid_past_threshold(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    repo = _make_repo(hb, tmp_path)
    _patch_gh(monkeypatch, hb, [516])
    _make_pr_dirs(repo.state_dir, 516)
    _write_state(
        repo.state_dir,
        516,
        {
            "review_dispatch_status": "review_dispatch_dispatched",
            "review_dispatched_at": _iso(60),
            "reviewer_pid": 24616,
            "reviewer_process_start_time": 1000.0,
        },
    )
    monkeypatch.setattr(hb, "psutil", FakePsutil({24616: (False, None)}))

    report = hb.Report()
    hb.check_review_liveness(report, repo)

    assert report.anomaly
    assert "pid=24616 dead" in report.lines[0]
    assert "threshold=45m" in report.lines[0]


def test_check_review_liveness_ok_for_dead_pid_inside_grace_window(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """A dead reviewer inside the 30-min grace window should not yet anomaly."""
    repo = _make_repo(hb, tmp_path)
    _patch_gh(monkeypatch, hb, [516])
    _make_pr_dirs(repo.state_dir, 516)
    _write_state(
        repo.state_dir,
        516,
        {
            "review_dispatch_status": "review_dispatch_dispatched",
            "review_dispatched_at": _iso(20),
            "reviewer_pid": 24616,
            "reviewer_process_start_time": 1000.0,
        },
    )
    monkeypatch.setattr(hb, "psutil", FakePsutil({24616: (False, None)}))

    report = hb.Report()
    hb.check_review_liveness(report, repo)

    assert not report.anomaly
    assert "pid=24616 dead" in report.lines[0]


def test_check_review_liveness_uses_pending_timestamp(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """Regression for issue #828 (originally #822's class): exact rounded-minute
    match against an injected clock, not two independently-sampled `now()`s.

    Production formats `oldest_min={round(age_min)}` where `age_min` is
    computed from `now - claim_time`. With two independent samples this flips
    from 5 to 6 (round(5.5) == 6) once ~30s passes between the fixture write
    and the production check -- comfortably within an observed CI stall. `now`
    is frozen and passed to both the fixture and the check so `age_min` is
    exactly 5.0 regardless of how long the process stalls in between.
    """
    frozen_now = datetime(2026, 7, 29, 12, 0, 0, tzinfo=timezone.utc)
    repo = _make_repo(hb, tmp_path)
    _patch_gh(monkeypatch, hb, [100])
    _make_pr_dirs(repo.state_dir, 100)
    _write_state(
        repo.state_dir,
        100,
        {
            "review_dispatch_status": "review_dispatch_pending",
            "review_dispatch_pending_at": _iso(5, base=frozen_now),
        },
    )

    report = hb.Report()
    hb.check_review_liveness(report, repo, now=frozen_now)

    assert not report.anomaly
    assert "pid=None" in report.lines[0]
    assert "oldest_min=5" in report.lines[0]


def test_check_review_liveness_falls_back_to_packet_mtime_when_state_timestamp_missing(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """If state.json has no dispatch timestamp, the packet mtime is the only clock left."""
    repo = _make_repo(hb, tmp_path)
    _patch_gh(monkeypatch, hb, [200])
    _make_pr_dirs(
        repo.state_dir,
        200,
        pr_mtime=(datetime(2020, 1, 1, tzinfo=timezone.utc)).timestamp(),
    )
    _write_state(repo.state_dir, 200, {"review_dispatch_status": None})

    report = hb.Report()
    hb.check_review_liveness(report, repo)

    assert report.anomaly
    assert "pr-200" in report.lines[0]


def _write_state_multi(state_dir: Path, prs: dict[int, dict[str, Any]]) -> None:
    """Write a state.json with multiple PR entries (``_write_state`` covers one)."""
    state = {"version": 1, "prs": {str(n): pr_state for n, pr_state in prs.items()}}
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")


def test_check_review_liveness_escalated_pr_not_anomaly(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """Regression for issue #1357.

    An escalated PR (``status == "escalated"`` in state.json) keeps its
    placeholder ``decision="pending"`` packet file forever -- the escalation
    gate stops further dispatch, so no review ever completes to overwrite it.
    The liveness check must NOT count it as an open claim or trip ANOMALY; it
    should be surfaced in the facts string as ``escalated=N`` instead.
    """
    frozen_now = datetime(2026, 8, 19, 5, 13, 0, tzinfo=timezone.utc)
    repo = _make_repo(hb, tmp_path)
    # Live-case shape: pr-1736 escalated, packet dir untouched, pending
    # decision file from packet-build time ~14h before the beat.
    _patch_gh(monkeypatch, hb, [1736])
    _make_pr_dirs(
        repo.state_dir,
        1736,
        pr_mtime=(frozen_now - timedelta(hours=14)).timestamp(),
    )
    _write_state(
        repo.state_dir,
        1736,
        {
            "status": "escalated",
            "review_dispatch_status": "review_dispatch_dispatched",
            "review_dispatched_at": _iso(871, base=frozen_now),
            "reviewer_pid": None,
        },
    )

    report = hb.Report()
    hb.check_review_liveness(report, repo, now=frozen_now)

    assert not report.anomaly
    assert report.lines and "review-liveness" in report.lines[0]
    assert "open_claims=0" in report.lines[0]
    assert "escalated=1" in report.lines[0]
    # The escalated PR's stale dir must not appear in an ANOMALY detail line.
    assert "pr-1736" not in report.lines[0]


def test_check_review_liveness_escalated_mixed_with_still_stale_open_claim(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """Issue #1357 AC2: a non-escalated stale open claim still fires ANOMALY.

    With one escalated PR (skipped) and one genuinely in-flight open claim
    past the stale threshold, the check must still ANOMALY on the in-flight
    one only -- escalated accounting must not silently swallow real liveness
    failures. The escalated PR is surfaced as ``escalated=1`` in the facts.
    """
    frozen_now = datetime(2026, 8, 19, 5, 13, 0, tzinfo=timezone.utc)
    repo = _make_repo(hb, tmp_path)
    _patch_gh(monkeypatch, hb, [1736, 2000])
    _make_pr_dirs(
        repo.state_dir,
        1736,
        pr_mtime=(frozen_now - timedelta(hours=14)).timestamp(),
    )
    _make_pr_dirs(repo.state_dir, 2000)
    _write_state_multi(
        repo.state_dir,
        {
            1736: {
                "status": "escalated",
                "review_dispatch_status": "review_dispatch_dispatched",
                "review_dispatched_at": _iso(871, base=frozen_now),
                "reviewer_pid": None,
            },
            2000: {
                "review_dispatch_status": "review_dispatch_dispatched",
                "review_dispatched_at": _iso(60, base=frozen_now),
                "reviewer_pid": 24616,
                "reviewer_process_start_time": 1000.0,
            },
        },
    )
    monkeypatch.setattr(hb, "psutil", FakePsutil({24616: (False, None)}))

    report = hb.Report()
    hb.check_review_liveness(report, repo, now=frozen_now)

    assert report.anomaly
    assert "pr-2000" in report.lines[0]
    assert "threshold=45m" in report.lines[0]
    # The escalated PR is not in the ANOMALY detail but is in the facts.
    assert "escalated=1" in report.lines[0]
    assert "open_claims=1" in report.lines[0]
    # The escalated PR's dir must not be listed as a stale claim dir.
    assert "pr-1736" not in report.lines[0]


def test_check_review_liveness_non_escalated_pending_status_still_counts(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """Issue #1357 AC2 guard: only ``status == "escalated"`` is skipped.

    A PR whose state entry lacks ``status: "escalated"`` (here, no ``status``
    key at all, just a stale pending dispatch) must still be counted as an open
    claim and trip ANOMALY past the threshold -- the escalation carve-out is
    exact, not a fuzzy "pending-looking" match.
    """
    frozen_now = datetime(2026, 8, 19, 5, 13, 0, tzinfo=timezone.utc)
    repo = _make_repo(hb, tmp_path)
    _patch_gh(monkeypatch, hb, [3000])
    _make_pr_dirs(repo.state_dir, 3000)
    _write_state(
        repo.state_dir,
        3000,
        {
            "review_dispatch_status": "review_dispatch_dispatched",
            "review_dispatched_at": _iso(60, base=frozen_now),
            "reviewer_pid": 24616,
            "reviewer_process_start_time": 1000.0,
        },
    )
    monkeypatch.setattr(hb, "psutil", FakePsutil({24616: (False, None)}))

    report = hb.Report()
    hb.check_review_liveness(report, repo, now=frozen_now)

    assert report.anomaly
    assert "pr-3000" in report.lines[0]
    assert "open_claims=1" in report.lines[0]
    assert "escalated=" not in report.lines[0]


def test_review_claim_timestamp_completed_rebuilt_uses_prompt_mtime(
    hb: ModuleType, tmp_path: Path
) -> None:
    """Regression for issue #1403.

    A completed prior cycle (``review_dispatch_status ==
    review_dispatch_completed``) whose packet was rebuilt for a newer head:
    state.json still carries the PRIOR cycle's ``review_dispatched_at`` (stale),
    but the on-disk decision is back to ``pending`` head-stamped with the new
    head.  ``_review_claim_timestamp`` must anchor on the packet-rebuild
    evidence (``review-prompt.md`` mtime) instead of the stale dispatch time.
    """
    pr_dir = tmp_path / "prs" / "pr-1395"
    pr_dir.mkdir(parents=True)
    rebuild_time = datetime(2026, 8, 23, 0, 52, 19, tzinfo=timezone.utc)
    (pr_dir / "review-prompt.md").write_text("prompt", encoding="utf-8")
    os.utime(pr_dir / "review-prompt.md", (rebuild_time.timestamp(),) * 2)

    pr_state = {
        "review_dispatch_status": "review_dispatch_completed",
        # Prior cycle's dispatch time -- 138m before the beat, the false
        # ANOMALY source from the 2026-08-23T01:07Z incident.
        "review_dispatched_at": "2026-08-22T22:49:27Z",
        # Prior cycle's reviewed head.
        "reviewed_head_sha": "prior-cycle-head-sha",
    }
    decision = {"decision": "pending", "reviewed_head_sha": "new-head-sha"}

    timestamp = hb._review_claim_timestamp(pr_state, pr_dir=pr_dir, decision=decision)
    assert timestamp is not None
    parsed = hb.parse_iso(timestamp)
    assert parsed == rebuild_time


def test_review_claim_timestamp_completed_same_head_rebuilt_uses_prompt_mtime(
    hb: ModuleType, tmp_path: Path
) -> None:
    """Regression for issue #1436.

    A completed prior cycle whose packet was rebuilt for the SAME head (no SHA
    advance): the on-disk decision is back to ``pending`` with
    ``reviewed_head_sha`` EQUAL to state.json's.  The #1403 guard keyed on SHA
    inequality and fell through, so the newest-timestamp fallback dated the
    claim by the prior cycle's ``review_dispatched_at`` (false 467m ANOMALY on
    pr-1432).  The pending decision while status is completed is itself the
    rebuild evidence; ``_review_claim_timestamp`` must anchor on the
    ``review-prompt.md`` mtime regardless of SHA equality.
    """
    pr_dir = tmp_path / "prs" / "pr-1432"
    pr_dir.mkdir(parents=True)
    rebuild_time = datetime(2026, 8, 24, 9, 17, 38, tzinfo=timezone.utc)
    (pr_dir / "review-prompt.md").write_text("prompt", encoding="utf-8")
    os.utime(pr_dir / "review-prompt.md", (rebuild_time.timestamp(),) * 2)

    pr_state = {
        "review_dispatch_status": "review_dispatch_completed",
        # Prior cycle's dispatch time -- 467m before the 09:55Z beat, the
        # false ANOMALY source from the 2026-08-24T09:55Z incident.
        "review_dispatched_at": "2026-08-24T02:07:25Z",
        # Same head as the on-disk decision -- the case #1403 missed.
        "reviewed_head_sha": "93bf8ed",
    }
    decision = {"decision": "pending", "reviewed_head_sha": "93bf8ed"}

    timestamp = hb._review_claim_timestamp(pr_state, pr_dir=pr_dir, decision=decision)
    assert timestamp is not None
    parsed = hb.parse_iso(timestamp)
    assert parsed == rebuild_time


def test_review_claim_timestamp_completed_prompt_older_than_dispatch_uses_dispatch(
    hb: ModuleType, tmp_path: Path
) -> None:
    """Issue #1436 guard: the prompt mtime must not be older than the prior
    cycle's ``review_dispatched_at``.

    If the prompt file's mtime predates the completed cycle's dispatch (clock
    skew / a packet that was not actually rebuilt), the newer of the two is
    used so the change can only shrink a false age and never hide a genuinely
    stale claim from before the completed cycle.
    """
    pr_dir = tmp_path / "prs" / "pr-43"
    pr_dir.mkdir(parents=True)
    # Prompt mtime OLDER than the prior cycle's dispatch.
    stale_prompt_time = datetime(2026, 8, 22, 20, 0, 0, tzinfo=timezone.utc)
    (pr_dir / "review-prompt.md").write_text("prompt", encoding="utf-8")
    os.utime(pr_dir / "review-prompt.md", (stale_prompt_time.timestamp(),) * 2)

    pr_state = {
        "review_dispatch_status": "review_dispatch_completed",
        "review_dispatched_at": "2026-08-22T22:49:27Z",
        "reviewed_head_sha": "same-head-sha",
    }
    decision = {"decision": "pending", "reviewed_head_sha": "same-head-sha"}

    assert (
        hb._review_claim_timestamp(pr_state, pr_dir=pr_dir, decision=decision)
        == "2026-08-22T22:49:27Z"
    )


def test_review_claim_timestamp_completed_missing_prompt_falls_back_to_state(
    hb: ModuleType, tmp_path: Path
) -> None:
    """Issue #1436: when the prompt file is missing, the SHA comparison
    survives only as a tiebreak and we fall back to current behavior (the
    newest-timestamp fallback) regardless of SHA equality.
    """
    pr_dir = tmp_path / "prs" / "pr-44"
    pr_dir.mkdir(parents=True)

    pr_state = {
        "review_dispatch_status": "review_dispatch_completed",
        "review_dispatched_at": "2026-08-22T22:49:27Z",
        "reviewed_head_sha": "same-head-sha",
    }
    decision = {"decision": "pending", "reviewed_head_sha": "same-head-sha"}

    assert (
        hb._review_claim_timestamp(pr_state, pr_dir=pr_dir, decision=decision)
        == "2026-08-22T22:49:27Z"
    )


def test_check_review_liveness_completed_rebuilt_no_false_anomaly(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """Regression for issue #1403: end-to-end check.

    Production shape from the 2026-08-23T01:07Z beat on pr-1395: a prior review
    cycle completed at 22:49:27Z (``review_dispatch_status ==
    review_dispatch_completed``, ``reviewed_head_sha`` = prior head), then the
    rework cycle rebuilt the packet for a new head at 00:52:19Z (on-disk
    ``review-decision.json`` back to ``pending`` head-stamped with the new
    head, ``review-prompt.md`` rewritten).  ``dispatch_reviews()`` had not yet
    launched the next reviewer (waiting on the PR's Tests check), so
    ``review_dispatched_at`` still carried the prior cycle's 22:49:27Z.  The
    beat at 01:07Z must measure the claim age from the rebuild (~15m), not the
    stale prior dispatch (~138m), and must NOT ANOMALY.
    """
    frozen_now = datetime(2026, 8, 23, 1, 7, 0, tzinfo=timezone.utc)
    rebuild_time = datetime(2026, 8, 23, 0, 52, 19, tzinfo=timezone.utc)
    repo = _make_repo(hb, tmp_path)
    _patch_gh(monkeypatch, hb, [1395])
    pr_dir = repo.state_dir / "prs" / "pr-1395"
    pr_dir.mkdir(parents=True)
    (pr_dir / "pr.json").write_text("{}", encoding="utf-8")
    (pr_dir / "review-decision.json").write_text(
        json.dumps({"decision": "pending", "reviewed_head_sha": "new-head-sha"}),
        encoding="utf-8",
    )
    (pr_dir / "review-prompt.md").write_text("prompt", encoding="utf-8")
    os.utime(pr_dir / "review-prompt.md", (rebuild_time.timestamp(),) * 2)
    _write_state(
        repo.state_dir,
        1395,
        {
            "review_dispatch_status": "review_dispatch_completed",
            "review_dispatched_at": "2026-08-22T22:49:27Z",
            "reviewed_head_sha": "prior-cycle-head-sha",
            "reviewer_pid": None,
        },
    )

    report = hb.Report()
    hb.check_review_liveness(report, repo, now=frozen_now)

    assert not report.anomaly
    assert report.lines and "review-liveness" in report.lines[0]
    assert "open_claims=1" in report.lines[0]
    # ~15m from the rebuild, not ~138m from the stale prior dispatch.
    assert "oldest_min=15" in report.lines[0]
    assert "138" not in report.lines[0]


def test_check_review_liveness_completed_same_head_rebuilt_no_false_anomaly(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """Regression for issue #1436: end-to-end check.

    Production shape from the 2026-08-24T09:55Z beat on pr-1432: a prior review
    cycle completed at 02:07:25Z (``review_dispatch_status ==
    review_dispatch_completed``, ``reviewed_head_sha`` = 93bf8ed), then the
    rework cycle rebuilt the packet for the SAME head at 09:17:38Z (on-disk
    ``review-decision.json`` back to ``pending`` head-stamped with the same
    93bf8ed, ``review-prompt.md`` rewritten).  ``dispatch_reviews()`` had not
    yet launched the next reviewer, so ``review_dispatched_at`` still carried
    the prior cycle's 02:07:25Z.  The #1403 guard keyed on SHA inequality and
    fell through, so the beat would date the claim by the stale prior dispatch
    (~467m) and ANOMALY.  The beat at 09:55Z must measure the claim age from
    the rebuild (~38m, under the 45m threshold), not the stale prior dispatch,
    and must NOT ANOMALY.
    """
    frozen_now = datetime(2026, 8, 24, 9, 55, 0, tzinfo=timezone.utc)
    rebuild_time = datetime(2026, 8, 24, 9, 17, 38, tzinfo=timezone.utc)
    repo = _make_repo(hb, tmp_path)
    _patch_gh(monkeypatch, hb, [1432])
    pr_dir = repo.state_dir / "prs" / "pr-1432"
    pr_dir.mkdir(parents=True)
    (pr_dir / "pr.json").write_text("{}", encoding="utf-8")
    (pr_dir / "review-decision.json").write_text(
        json.dumps({"decision": "pending", "reviewed_head_sha": "93bf8ed"}),
        encoding="utf-8",
    )
    (pr_dir / "review-prompt.md").write_text("prompt", encoding="utf-8")
    os.utime(pr_dir / "review-prompt.md", (rebuild_time.timestamp(),) * 2)
    _write_state(
        repo.state_dir,
        1432,
        {
            "review_dispatch_status": "review_dispatch_completed",
            "review_dispatched_at": "2026-08-24T02:07:25Z",
            "reviewed_head_sha": "93bf8ed",
            "reviewer_pid": None,
        },
    )

    report = hb.Report()
    hb.check_review_liveness(report, repo, now=frozen_now)

    assert not report.anomaly
    assert report.lines and "review-liveness" in report.lines[0]
    assert "open_claims=1" in report.lines[0]
    # ~37m from the rebuild, not ~467m from the stale prior dispatch.
    assert "oldest_min=37" in report.lines[0]
    assert "467" not in report.lines[0]
