"""Tests for the pure _annotation_to_required_change/_required_changes_from_checks annotation-aggregation parser, carved out of test_charlie_work.py (#1284) -- distinct from the heavier _ci_status_section integration cluster that remains there for a future review-domain PR, and distinct from the pre-existing facade-routed tests/test_dispatch_staleness.py."""

from __future__ import annotations

from typing import Any

from charlie_work.ci_findings import _annotation_to_required_change, _required_changes_from_checks


def test_annotation_to_required_change_full_annotation() -> None:
    """Issue #771: a well-formed GitHub annotation renders check/path/line/message."""
    entry = _annotation_to_required_change(
        "Lint",
        {
            "path": "src/charlie_work/workflow.py",
            "start_line": 42,
            "message": "line too long (100 > 99)",
            "annotation_level": "failure",
        },
    )
    assert entry == "Lint: src/charlie_work/workflow.py:42 — line too long (100 > 99)"


def test_annotation_to_required_change_no_message_returns_none() -> None:
    """Never fabricate a placeholder when GitHub gives no explanatory message."""
    assert (
        _annotation_to_required_change(
            "Lint", {"path": "src/foo.py", "start_line": 1, "annotation_level": "failure"}
        )
        is None
    )
    assert (
        _annotation_to_required_change(
            "Lint", {"path": "src/foo.py", "message": "", "annotation_level": "failure"}
        )
        is None
    )


def test_annotation_to_required_change_non_failure_level_returns_none() -> None:
    """Issue #993: warning/notice annotations are not required changes -- they
    are emitted on green runs too (e.g. the actions/checkout Node.js 20
    deprecation advisory), so surfacing them as rework items sends the worker
    after unrelated noise. Only ``annotation_level == "failure"`` renders."""
    warning = {
        "path": ".github",
        "start_line": 2,
        "message": "Node.js 20 is deprecated. ... actions/checkout@v4",
        "annotation_level": "warning",
    }
    notice = {
        "path": "src/foo.py",
        "start_line": 1,
        "message": "consider using X",
        "annotation_level": "notice",
    }
    missing_level = {"path": "src/foo.py", "start_line": 1, "message": "boom"}
    null_level = {
        "path": "src/foo.py",
        "start_line": 1,
        "message": "boom",
        "annotation_level": None,
    }
    assert _annotation_to_required_change("Lint", warning) is None
    assert _annotation_to_required_change("Lint", notice) is None
    assert _annotation_to_required_change("Lint", missing_level) is None
    assert _annotation_to_required_change("Lint", null_level) is None


def test_annotation_to_required_change_missing_path_falls_back_to_message_only() -> None:
    """No path/line data (e.g. a process-level crash) still surfaces the real
    message rather than being dropped -- but with no fabricated location."""
    entry = _annotation_to_required_change(
        "Tests", {"message": "process exited with code 1", "annotation_level": "failure"}
    )
    assert entry == "Tests: process exited with code 1"


def test_annotation_to_required_change_path_without_line() -> None:
    entry = _annotation_to_required_change(
        "Lint",
        {"path": "src/foo.py", "message": "file-level issue", "annotation_level": "failure"},
    )
    assert entry == "Lint: src/foo.py — file-level issue"


def test_annotation_to_required_change_non_dict_returns_none() -> None:
    assert _annotation_to_required_change("Lint", "not a dict") is None  # type: ignore[arg-type]


def test_required_changes_from_checks_aggregates_annotations_for_failing_check() -> None:
    checks = [
        {"name": "Lint", "state": "FAILURE", "databaseId": 111},
        {"name": "Tests", "state": "SUCCESS", "databaseId": 222},
    ]
    annotations_by_id = {
        111: [
            {
                "path": "src/foo.py",
                "start_line": 10,
                "message": "E501 line too long",
                "annotation_level": "failure",
            },
            {
                "path": "src/bar.py",
                "start_line": 20,
                "message": "F401 unused import",
                "annotation_level": "failure",
            },
        ],
    }
    required_changes = _required_changes_from_checks(
        checks, ("Lint",), lambda check_run_id: annotations_by_id.get(check_run_id, [])
    )
    assert required_changes == [
        "Lint: src/foo.py:10 — E501 line too long",
        "Lint: src/bar.py:20 — F401 unused import",
    ]


def test_required_changes_from_checks_skips_passing_run_of_failed_name() -> None:
    """A name with two runs (matrix legs) under worst-of semantics: only the
    FAILURE run's annotations should be fetched, not the passing sibling's."""
    fetched_ids: list[int] = []

    def fetch(check_run_id: int) -> list[dict[str, Any]]:
        fetched_ids.append(check_run_id)
        return (
            [{"path": "x.py", "start_line": 1, "message": "boom", "annotation_level": "failure"}]
            if check_run_id == 2
            else []
        )

    checks = [
        {"name": "Tests", "state": "SUCCESS", "databaseId": 1},
        {"name": "Tests", "state": "FAILURE", "databaseId": 2},
    ]
    required_changes = _required_changes_from_checks(checks, ("Tests",), fetch)
    assert fetched_ids == [2]
    assert required_changes == ["Tests: x.py:1 — boom"]


def test_required_changes_from_checks_no_databaseid_degrades_to_empty() -> None:
    """No resolvable check-run id AND no link (e.g. a bare status check with
    neither) -- degrade to [] without ever calling the annotations fetcher."""
    checks = [{"name": "Lint", "state": "FAILURE", "databaseId": None}]
    called = False

    def fetch(check_run_id: int) -> list[dict[str, Any]]:
        nonlocal called
        called = True
        return []

    assert _required_changes_from_checks(checks, ("Lint",), fetch) == []
    assert called is False


def test_required_changes_from_checks_no_databaseid_falls_back_to_link() -> None:
    """No resolvable check-run id (e.g. an external status check) but a real
    ``link`` from GitHub -- fall back to pointing at the link rather than
    silently dropping the failure, and never call the annotations fetcher
    (there is no check-run id to fetch with)."""
    checks = [
        {
            "name": "Lint",
            "state": "FAILURE",
            "databaseId": None,
            "link": "https://example.com/status/lint",
        }
    ]
    called = False

    def fetch(check_run_id: int) -> list[dict[str, Any]]:
        nonlocal called
        called = True
        return []

    result = _required_changes_from_checks(checks, ("Lint",), fetch)
    assert result == [
        "Lint: no per-line annotations available from GitHub; "
        "inspect the failing run at https://example.com/status/lint",
    ]
    assert called is False


def test_required_changes_from_checks_zero_annotations_degrades_to_empty() -> None:
    """A resolvable check run with zero annotations and no link (common for a
    process-level crash) degrades to [] -- never a fabricated file/line."""
    checks = [{"name": "Lint", "state": "FAILURE", "databaseId": 5}]
    assert _required_changes_from_checks(checks, ("Lint",), lambda _id: []) == []


def test_required_changes_from_checks_zero_annotations_falls_back_to_link() -> None:
    """A resolvable check run with zero annotations but a real ``link`` falls
    back to the link -- more useful than silence, still not fabricated."""
    checks = [
        {
            "name": "Lint",
            "state": "FAILURE",
            "databaseId": 5,
            "link": "https://github.com/o/r/actions/runs/1/jobs/5",
        }
    ]
    result = _required_changes_from_checks(checks, ("Lint",), lambda _id: [])
    assert result == [
        "Lint: no per-line annotations available from GitHub; "
        "inspect the failing run at https://github.com/o/r/actions/runs/1/jobs/5",
    ]


def test_required_changes_from_checks_no_checks_available_degrades_to_empty() -> None:
    assert _required_changes_from_checks(None, ("Lint",), lambda _id: []) == []


def test_required_changes_from_checks_no_failed_names_degrades_to_empty() -> None:
    checks = [{"name": "Lint", "state": "FAILURE", "databaseId": 5}]
    assert _required_changes_from_checks(checks, (), lambda _id: [{"message": "x"}]) == []


def test_required_changes_from_checks_filters_warning_level_annotations() -> None:
    """Issue #993: warning/notice annotations are present on green runs too
    (e.g. the actions/checkout Node.js 20 deprecation advisory), so they are
    not required changes. Only failure-level annotations render as entries;
    the warning is dropped entirely rather than crowding the rework brief
    with unrelated noise."""
    checks = [{"name": "Lint", "state": "FAILURE", "databaseId": 111}]
    annotations = [
        {
            "path": ".github",
            "start_line": 2,
            "message": "Node.js 20 is deprecated. ... actions/checkout@v4",
            "annotation_level": "warning",
        },
        {
            "path": "src/foo.py",
            "start_line": 10,
            "message": "E501 line too long",
            "annotation_level": "failure",
        },
    ]
    required_changes = _required_changes_from_checks(checks, ("Lint",), lambda _id: annotations)
    assert required_changes == ["Lint: src/foo.py:10 — E501 line too long"]


def test_required_changes_from_checks_appends_link_alongside_failure_annotations() -> None:
    """Issue #993: the failing run's ``link`` is always appended alongside
    whatever failure-level annotations rendered -- not only when *zero*
    annotations rendered. A process-level crash emits a contentless
    ``"Process completed with exit code 1."`` failure annotation that names
    no cause; the real cause (e.g. a TLS handshake timeout) lives only in
    the step log the link reaches. The old ``if entries:`` guard suppressed
    the link whenever any annotation rendered, so the fallback never fired
    in the scenario its own docstring named as common."""
    checks = [
        {
            "name": "Lint",
            "state": "FAILURE",
            "databaseId": 111,
            "link": "https://github.com/o/r/actions/runs/92297706625",
        }
    ]
    annotations = [
        {
            "path": ".github",
            "start_line": 2,
            "message": "Node.js 20 is deprecated. ... actions/checkout@v4",
            "annotation_level": "warning",
        },
        {
            "path": ".github",
            "start_line": 14,
            "message": "Process completed with exit code 1.",
            "annotation_level": "failure",
        },
    ]
    required_changes = _required_changes_from_checks(checks, ("Lint",), lambda _id: annotations)
    # The warning is filtered out; the contentless failure annotation still
    # renders (it is failure-level and carries a message), and the link is
    # appended alongside it so the worker can reach the run log where the
    # real cause lives.
    assert required_changes == [
        "Lint: .github:14 — Process completed with exit code 1.",
        "Lint: failing run — https://github.com/o/r/actions/runs/92297706625",
    ]


def test_required_changes_from_checks_link_fallback_fires_for_contentless_failure_only() -> None:
    """Issue #993: when the only annotations are non-failure-level (so
    ``entries`` is empty after filtering), the link fallback still fires
    with the "no per-line annotations available" wording -- the fallback is
    not deleted, only its guard is fixed so it no longer requires *zero*
    annotations of any level."""
    checks = [
        {
            "name": "Lint",
            "state": "FAILURE",
            "databaseId": 111,
            "link": "https://github.com/o/r/actions/runs/1",
        }
    ]
    annotations = [
        {
            "path": ".github",
            "start_line": 2,
            "message": "Node.js 20 is deprecated. ... actions/checkout@v4",
            "annotation_level": "warning",
        },
    ]
    required_changes = _required_changes_from_checks(checks, ("Lint",), lambda _id: annotations)
    assert required_changes == [
        "Lint: no per-line annotations available from GitHub; "
        "inspect the failing run at https://github.com/o/r/actions/runs/1",
    ]
