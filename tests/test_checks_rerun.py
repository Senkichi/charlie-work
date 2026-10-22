"""Code-failure rerun-debounce tests for ``charlie_work.checks``.

Split out of ``tests/test_checks.py`` (issue #1565, Track-1 shoulder):
``classify_check_failures``'s first-failure rerun request, per-head
attempt recording, pass-clears-marker, external-status definitive
failure, and run-id grouping -- plus the ``_run_id_from_link`` parser
those classifications rely on.
"""

from __future__ import annotations

from _checks_fixtures import REQUIRED, _link

from charlie_work.check_run_resolution import (
    _check_run_id_from_link,
    _check_suite_id_from_payload,
    _run_id_for_check_suite,
    _run_id_from_check_run_output,
)
from charlie_work.checks import (
    CheckDebounceResult,
    _run_id_from_link,
    classify_check_failures,
)


def test_run_id_from_link_extracts_run_id() -> None:
    link = "https://github.com/owner/repo/actions/runs/29525590823/job/87713099471?check_suite_focus=true"
    assert _run_id_from_link(link) == 29525590823


def test_run_id_from_link_returns_none_for_external_status() -> None:
    assert _run_id_from_link("https://external.ci/some/path") is None
    assert _run_id_from_link(None) is None
    assert _run_id_from_link("") is None


def test_check_run_id_from_link_parses_app_assigned_runs_link() -> None:
    """The live shape from issue #2540: a check run the Actions app created
    via ``POST /repos/{owner}/{repo}/check-runs`` from inside a merged job
    carries ``details_url`` ``.../runs/<check_run_id>`` (GitHub ignores the
    caller-supplied value)."""
    link = "https://github.com/owner/repo/runs/112064484398"
    assert _check_run_id_from_link(link) == 112064484398


def test_check_run_id_from_link_tolerates_query_and_fragment() -> None:
    assert _check_run_id_from_link("https://github.com/owner/repo/runs/7?focus=true") == 7
    assert _check_run_id_from_link("https://github.com/owner/repo/runs/7#summary") == 7


def test_check_run_id_from_link_rejects_actions_job_link() -> None:
    """The ``/runs/<id>`` segment of a real Actions job link is followed by
    ``/job/``, so the end-anchored check-run pattern must never match it --
    otherwise the resolver would fetch a workflow-run id as a check run."""
    assert (
        _check_run_id_from_link(
            "https://github.com/owner/repo/actions/runs/29525590823/job/87713099471"
            "?check_suite_focus=true"
        )
        is None
    )


def test_check_run_id_from_link_returns_none_for_external_status() -> None:
    assert _check_run_id_from_link("https://external.ci/some/path") is None
    assert _check_run_id_from_link(None) is None
    assert _check_run_id_from_link("") is None


def test_run_id_from_check_run_output_parses_summary_job_link() -> None:
    """A posting step that embeds the real job URL in its ``output.summary``
    gives the resolver a one-call path: the summary text is caller-supplied,
    so GitHub's ``details_url`` rewrite never touches it."""
    payload = {
        "output": {
            "summary": "Guard failed; see [the job]"
            "(https://github.com/owner/repo/actions/runs/37398606691/job/112063807090)."
        }
    }
    assert _run_id_from_check_run_output(payload) == 37398606691


def test_run_id_from_check_run_output_scans_title_and_text_too() -> None:
    title_only = {"output": {"title": "run 123 / job 456"}}
    assert _run_id_from_check_run_output(title_only) is None
    text_only = {
        "output": {
            "title": "no link here",
            "text": "https://github.com/owner/repo/actions/runs/55/job/66",
        }
    }
    assert _run_id_from_check_run_output(text_only) == 55


def test_run_id_from_check_run_output_none_without_actions_link() -> None:
    assert _run_id_from_check_run_output(None) is None
    assert _run_id_from_check_run_output({}) is None
    assert _run_id_from_check_run_output({"output": None}) is None
    assert _run_id_from_check_run_output({"output": {"summary": "no link"}}) is None
    # A check-run link in the output is not an Actions job link.
    assert (
        _run_id_from_check_run_output({"output": {"summary": "https://github.com/o/r/runs/9"}})
        is None
    )


def test_check_suite_id_from_payload_extracts_suite_id() -> None:
    assert _check_suite_id_from_payload({"check_suite": {"id": 4242}}) == 4242
    assert _check_suite_id_from_payload({"check_suite": None}) is None
    assert _check_suite_id_from_payload({}) is None
    assert _check_suite_id_from_payload(None) is None
    assert _check_suite_id_from_payload({"check_suite": {"id": "4242"}}) is None


def test_run_id_for_check_suite_matches_owning_run() -> None:
    runs = [
        {"id": 111, "check_suite_id": 1},
        {"id": 37398606691, "check_suite_id": 4242},
    ]
    assert _run_id_for_check_suite(runs, 4242) == 37398606691


def test_run_id_for_check_suite_no_match_or_malformed_returns_none() -> None:
    # An external-CI check run's suite belongs to no workflow run.
    assert _run_id_for_check_suite([{"id": 111, "check_suite_id": 1}], 4242) is None
    assert _run_id_for_check_suite(None, 4242) is None
    assert _run_id_for_check_suite([], 4242) is None
    assert _run_id_for_check_suite([{"id": "x", "check_suite_id": 4242}], 4242) is None
    assert _run_id_for_check_suite([{"check_suite_id": 4242}], 4242) is None


def test_classify_first_failure_requests_rerun_and_records_attempt() -> None:
    checks = [
        {"name": "Tests passed", "state": "FAILURE", "link": _link(100, 1)},
        {"name": "Lint & Format", "bucket": "pass"},
    ]
    result = classify_check_failures(
        checks,
        REQUIRED,
        pr_state=None,
        head_sha="sha-1",
    )
    assert result == CheckDebounceResult(
        rerun_run_ids=(100,),
        check_rerun_attempts={"sha-1": {"Tests passed": [100]}},
        definitive_failed=(),
    )


def test_classify_second_failure_is_definitive_and_does_not_rerun() -> None:
    checks = [
        {"name": "Tests passed", "state": "FAILURE", "link": _link(100, 1)},
        {"name": "Lint & Format", "bucket": "pass"},
    ]
    pr_state = {"check_rerun_attempts": {"sha-1": {"Tests passed": [100]}}}
    result = classify_check_failures(
        checks,
        REQUIRED,
        pr_state,
        head_sha="sha-1",
    )
    assert result.rerun_run_ids == ()
    assert result.definitive_failed == ("Tests passed",)
    # Attempts unchanged.
    assert result.check_rerun_attempts == {"sha-1": {"Tests passed": [100]}}


def test_classify_passing_check_clears_attempt_marker() -> None:
    checks = [
        {"name": "Tests passed", "state": "SUCCESS"},
        {"name": "Lint & Format", "bucket": "pass"},
    ]
    pr_state = {"check_rerun_attempts": {"sha-1": {"Tests passed": [100]}}}
    result = classify_check_failures(
        checks,
        REQUIRED,
        pr_state,
        head_sha="sha-1",
    )
    assert result.rerun_run_ids == ()
    assert result.definitive_failed == ()
    assert result.check_rerun_attempts == {"sha-1": {}}


def test_classify_new_head_resets_attempts() -> None:
    checks = [
        {"name": "Tests passed", "state": "FAILURE", "link": _link(100, 1)},
        {"name": "Lint & Format", "bucket": "pass"},
    ]
    pr_state = {"check_rerun_attempts": {"sha-old": {"Tests passed": [100]}}}
    result = classify_check_failures(
        checks,
        REQUIRED,
        pr_state,
        head_sha="sha-new",
    )
    assert result.rerun_run_ids == (100,)
    assert result.check_rerun_attempts == {"sha-new": {"Tests passed": [100]}}


def test_classify_external_status_failure_is_definitive_without_rerun() -> None:
    checks = [
        {"name": "Tests passed", "state": "FAILURE", "link": "https://external.ci/run"},
        {"name": "Lint & Format", "bucket": "pass"},
    ]
    result = classify_check_failures(
        checks,
        REQUIRED,
        pr_state=None,
        head_sha="sha-1",
    )
    assert result.rerun_run_ids == ()
    assert result.definitive_failed == ("Tests passed",)


def test_classify_record_attempts_false_does_not_consume_attempt() -> None:
    checks = [
        {"name": "Tests passed", "state": "FAILURE", "link": _link(100, 1)},
        {"name": "Lint & Format", "bucket": "pass"},
    ]
    result = classify_check_failures(
        checks,
        REQUIRED,
        pr_state=None,
        head_sha="sha-1",
        record_attempts=False,
    )
    assert result.rerun_run_ids == ()
    assert result.definitive_failed == ("Tests passed",)
    assert result.check_rerun_attempts == {"sha-1": {}}


def test_classify_groups_multiple_failed_checks_by_run_id() -> None:
    checks = [
        {"name": "Tests passed", "state": "FAILURE", "link": _link(100, 1)},
        {"name": "Lint & Format", "state": "FAILURE", "link": _link(100, 2)},
    ]
    result = classify_check_failures(
        checks,
        REQUIRED,
        pr_state=None,
        head_sha="sha-1",
    )
    # Both checks share the same workflow run id; only one rerun is requested.
    assert result.rerun_run_ids == (100,)
    assert result.check_rerun_attempts == {
        "sha-1": {"Tests passed": [100], "Lint & Format": [100]}
    }
