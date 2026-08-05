"""Tests for issue #933: durable once-per-PR tripwire detection record.

Covers three independent additions layered on the #502/#673 unauthorized-merge
tripwire:

1. ``workflow.summarize_loop_errors`` — a bounded summary of ``loop()``'s
   ``errors`` list for the ``loop_completed`` event payload.
2. ``OrchestratorApp._announce_unauthorized_merges`` /
   ``UNAUTHORIZED_MERGE_DETECTED_KEY`` — a durable record so
   ``unauthorized_merge_detected`` fires once per PR, not once per pass, while
   the finding itself keeps pinning ``ok=False`` until explicitly acked.
3. ``OrchestratorApp.tripwire_status`` / ``charlie tripwire status`` — a
   read-only reporter over that record.

Reuses the tripwire fixtures from ``test_charlie_work.py``
(``_arm_unauthorized_merge_tripwire``, ``_merged_worker_pr``,
``_ack_unauthorized_merge``, ``FakeGitHub``) rather than building a new
harness, per the existing convention in this test suite (see e.g.
``test_deescalation.py``, ``test_fix_escalation_paths.py``).
"""

from __future__ import annotations

from pathlib import Path

from charlie_work import cli
from charlie_work.config import OrchestratorConfig
from charlie_work.instrumentation import _classify_level
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state
from charlie_work.workflow import (
    OrchestratorApp,
    UNAUTHORIZED_MERGE_DETECTED_KEY,
    summarize_loop_errors,
)

from test_charlie_work import (
    FakeGitHub,
    _ack_unauthorized_merge,
    _arm_unauthorized_merge_tripwire,
    _merged_worker_pr,
)


def _make_app(tmp_path: Path, fake_gh: FakeGitHub, **kwargs) -> tuple[OrchestratorApp, object]:
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.ensure()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh, **kwargs)
    return app, paths


# ---------------------------------------------------------------------------
# summarize_loop_errors
# ---------------------------------------------------------------------------


def test_summarize_loop_errors_empty_list() -> None:
    summary = summarize_loop_errors([])
    assert summary == {
        "error_prs": [],
        "error_prs_truncated": 0,
        "error_details": [],
        "error_details_truncated": 0,
    }


def test_summarize_loop_errors_dedupes_pr_numbers_preserving_order() -> None:
    errors = [
        {"pr": 501, "error": "first"},
        {"pr": 502, "error": "second"},
        {"pr": 501, "error": "third"},
    ]
    summary = summarize_loop_errors(errors)
    assert summary["error_prs"] == [501, 502]
    assert summary["error_prs_truncated"] == 0


def test_summarize_loop_errors_skips_non_int_and_bool_pr_values() -> None:
    errors = [
        {"pr": "not-an-int", "error": "a"},
        {"pr": None, "error": "b"},
        {"pr": True, "error": "c"},  # bool is technically an int subclass
        {"error": "d"},  # missing "pr" entirely
        {"pr": 501, "error": "e"},
    ]
    summary = summarize_loop_errors(errors)
    assert summary["error_prs"] == [501]


def test_summarize_loop_errors_pr_cap_and_truncated_count() -> None:
    errors = [{"pr": n, "error": f"err {n}"} for n in range(1, 26)]  # 25 distinct PRs
    summary = summarize_loop_errors(errors, max_prs=20, max_details=5)
    assert summary["error_prs"] == list(range(1, 21))
    assert summary["error_prs_truncated"] == 5


def test_summarize_loop_errors_detail_cap_and_truncated_count() -> None:
    errors = [{"pr": n, "error": f"err {n}"} for n in range(1, 9)]  # 8 entries
    summary = summarize_loop_errors(errors, max_prs=20, max_details=5)
    assert len(summary["error_details"]) == 5
    assert summary["error_details_truncated"] == 3


def test_summarize_loop_errors_long_detail_truncated_to_exact_length() -> None:
    long_text = "x" * 1000
    errors = [{"pr": 501, "error": long_text}]
    summary = summarize_loop_errors(errors, detail_chars=300)
    detail = summary["error_details"][0]
    assert len(detail) == 300
    assert detail.endswith("...")
    assert detail == "x" * 297 + "..."


def test_summarize_loop_errors_short_detail_left_untruncated() -> None:
    errors = [{"pr": 501, "error": "short message"}]
    summary = summarize_loop_errors(errors, detail_chars=300)
    assert summary["error_details"] == ["short message"]


# ---------------------------------------------------------------------------
# Once-per-PR announcement (the core behavioural claim of #933)
# ---------------------------------------------------------------------------


def test_unauthorized_merge_detected_event_fires_once_per_pr_not_once_per_pass(
    tmp_path: Path,
) -> None:
    """The finding must be announced exactly once, no matter how many passes re-detect it.

    The tripwire re-detects the same unacked finding on every pass by design
    (that repetition is what pins ``ok=False`` until the finding is triaged).
    Without the durable record in ``UNAUTHORIZED_MERGE_DETECTED_KEY``, calling
    the detection path twice would emit two ``unauthorized_merge_detected``
    events for the same PR — an unbounded stream restating one fact forever,
    which is exactly the "a control that can never go quiet is not a control"
    failure #933 was filed on.
    """
    fake_gh = FakeGitHub()
    fake_gh.issues = []
    fake_gh.prs = [_merged_worker_pr(1408, 1404, "sha-1408")]
    app, paths = _make_app(tmp_path, fake_gh)
    _arm_unauthorized_merge_tripwire(paths)

    app._detect_unauthorized_merges(fake_gh.prs)
    app._detect_unauthorized_merges(fake_gh.prs)
    app._detect_unauthorized_merges(fake_gh.prs)

    state = load_state(paths.state_file)
    events = [e for e in state["events"] if e["kind"] == "unauthorized_merge_detected"]
    assert len(events) == 1, (
        f"expected exactly one unauthorized_merge_detected event across 3 passes "
        f"re-detecting the same PR, got {len(events)}"
    )
    assert events[0]["payload"]["pr"] == 1408

    # A second, DIFFERENT unauthorized PR appears -- exactly one more event.
    fake_gh.prs = [*fake_gh.prs, _merged_worker_pr(1392, 1268, "sha-1392")]
    app._detect_unauthorized_merges(fake_gh.prs)
    app._detect_unauthorized_merges(fake_gh.prs)

    state = load_state(paths.state_file)
    events = [e for e in state["events"] if e["kind"] == "unauthorized_merge_detected"]
    assert len(events) == 2, (
        f"a new distinct finding must add exactly one more event, got {len(events)}"
    )
    assert sorted(e["payload"]["pr"] for e in events) == [1392, 1408]


def test_unauthorized_merge_detected_record_persists_pr_details(tmp_path: Path) -> None:
    fake_gh = FakeGitHub()
    fake_gh.issues = []
    fake_gh.prs = [_merged_worker_pr(1408, 1404, "sha-1408")]
    app, paths = _make_app(tmp_path, fake_gh)
    _arm_unauthorized_merge_tripwire(paths)

    app._detect_unauthorized_merges(fake_gh.prs)

    state = load_state(paths.state_file)
    record = state[UNAUTHORIZED_MERGE_DETECTED_KEY]
    assert set(record.keys()) == {"1408"}
    assert record["1408"]["issue"] == 1404
    assert record["1408"]["detected_at"]


def test_unauthorized_merge_detected_record_does_not_suppress_the_finding(
    tmp_path: Path,
) -> None:
    """The record silences the EVENT only; the finding itself must keep pinning ok=False.

    This is the invariant most likely to be broken by a later refactor: it is
    tempting to reuse ``UNAUTHORIZED_MERGE_DETECTED_KEY`` as a second
    suppression set alongside the ack set. Only an explicit ack
    (``UNAUTHORIZED_MERGE_ACK_KEY``) may suppress a finding from
    ``_detect_unauthorized_merges``'s return value -- the detected-record is a
    log, not a filter.
    """
    fake_gh = FakeGitHub()
    fake_gh.issues = []
    fake_gh.prs = [_merged_worker_pr(1408, 1404, "sha-1408")]
    app, paths = _make_app(tmp_path, fake_gh)
    _arm_unauthorized_merge_tripwire(paths)

    first = app._detect_unauthorized_merges(fake_gh.prs)
    assert [d["pr"] for d in first] == [1408]

    # The PR is now present in the detected record (event already fired once)...
    state = load_state(paths.state_file)
    assert "1408" in state[UNAUTHORIZED_MERGE_DETECTED_KEY]

    # ...but re-detecting it must STILL return it: presence in the detected
    # record must never remove it from the reported/errors set.
    second = app._detect_unauthorized_merges(fake_gh.prs)
    assert [d["pr"] for d in second] == [1408], (
        "a PR recorded in UNAUTHORIZED_MERGE_DETECTED_KEY must still be "
        "returned by _detect_unauthorized_merges -- only an ack may suppress it"
    )

    # Confirm the actual suppression mechanism (ack) still works as before,
    # to contrast with the non-suppression asserted above.
    _ack_unauthorized_merge(paths, 1408, "root cause fixed")
    third = app._detect_unauthorized_merges(fake_gh.prs)
    assert third == [], "acking must still suppress the finding"


def test_unauthorized_merge_detected_writes_nothing_and_emits_nothing_in_dry_run(
    tmp_path: Path,
) -> None:
    fake_gh = FakeGitHub()
    fake_gh.issues = []
    fake_gh.prs = [_merged_worker_pr(1408, 1404, "sha-1408")]
    app, paths = _make_app(tmp_path, fake_gh, dry_run=True)
    _arm_unauthorized_merge_tripwire(paths)

    detected = app._detect_unauthorized_merges(fake_gh.prs)
    assert [d["pr"] for d in detected] == [1408], "dry-run must still report the finding"

    state = load_state(paths.state_file)
    assert UNAUTHORIZED_MERGE_DETECTED_KEY not in state, (
        "dry-run must not persist a detected-record"
    )
    events = [e for e in state["events"] if e["kind"] == "unauthorized_merge_detected"]
    assert events == [], "dry-run must not emit unauthorized_merge_detected"


# ---------------------------------------------------------------------------
# tripwire_status
# ---------------------------------------------------------------------------


def test_tripwire_status_reports_not_armed_when_no_baseline(tmp_path: Path) -> None:
    fake_gh = FakeGitHub()
    app, paths = _make_app(tmp_path, fake_gh)

    result = app.tripwire_status()

    assert result.ok is True
    assert "NOT ARMED" in result.message
    assert result.data["pending"] == []
    assert result.data["pending_count"] == 0


def test_tripwire_status_reports_detected_unacked_pr_as_pending(tmp_path: Path) -> None:
    fake_gh = FakeGitHub()
    fake_gh.issues = []
    fake_gh.prs = [_merged_worker_pr(1408, 1404, "sha-1408")]
    app, paths = _make_app(tmp_path, fake_gh)
    _arm_unauthorized_merge_tripwire(paths)

    app._detect_unauthorized_merges(fake_gh.prs)

    result = app.tripwire_status()
    assert result.ok is True
    assert result.data["pending_count"] == 1
    assert result.data["pending"][0]["pr"] == 1408
    assert result.data["detected_count"] == 1
    assert result.data["acknowledged_count"] == 0
    assert "1408" in result.message


def test_tripwire_status_excludes_acked_pr_from_pending(tmp_path: Path) -> None:
    fake_gh = FakeGitHub()
    fake_gh.issues = []
    fake_gh.prs = [_merged_worker_pr(1408, 1404, "sha-1408")]
    app, paths = _make_app(tmp_path, fake_gh)
    _arm_unauthorized_merge_tripwire(paths)

    app._detect_unauthorized_merges(fake_gh.prs)
    _ack_unauthorized_merge(paths, 1408, "root cause fixed")

    result = app.tripwire_status()
    assert result.data["pending"] == []
    assert result.data["pending_count"] == 0
    # The detected record itself is untouched by the ack -- both counts stay.
    assert result.data["detected_count"] == 1
    assert result.data["acknowledged_count"] == 1
    assert "no pending" in result.message.lower()


# ---------------------------------------------------------------------------
# instrumentation._classify_level
# ---------------------------------------------------------------------------


def test_classify_level_unauthorized_merge_detected_is_error() -> None:
    assert _classify_level("unauthorized_merge_detected") == "error"


def test_classify_level_paired_bookkeeping_events_stay_info() -> None:
    # Contrast case named in the diff's own comment: a triaged finding and a
    # suppressed backlog are bookkeeping, not the alarm itself.
    assert _classify_level("unauthorized_merge_acknowledged") == "info"
    assert _classify_level("unauthorized_merge_baseline_armed") == "info"


# ---------------------------------------------------------------------------
# CLI: `charlie tripwire status`
# ---------------------------------------------------------------------------


def test_cli_tripwire_status_routes_to_app(monkeypatch, tmp_path: Path) -> None:
    fake_gh = FakeGitHub()
    fake_gh.issues = []
    fake_gh.prs = [_merged_worker_pr(1408, 1404, "sha-1408")]
    app, paths = _make_app(tmp_path, fake_gh)
    _arm_unauthorized_merge_tripwire(paths)
    app._detect_unauthorized_merges(fake_gh.prs)

    monkeypatch.setattr(cli, "build_app", lambda args: app)

    exit_code = cli.main(["tripwire", "status"])
    assert exit_code == 0


def test_cli_tripwire_status_parses_with_no_extra_args(monkeypatch, tmp_path: Path) -> None:
    """`charlie tripwire status` takes no positional/required flags, unlike `ack`."""
    fake_gh = FakeGitHub()
    app, paths = _make_app(tmp_path, fake_gh)
    monkeypatch.setattr(cli, "build_app", lambda args: app)

    exit_code = cli.main(["tripwire", "status"])
    assert exit_code == 0
