"""``GitHub.pr_checks`` tests: runId injection and the statusCheckRollup
disambiguation fallback (issue #846).

Split out of ``tests/test_github.py`` (issue #1572, Track 1 shoulder) --
bodies are verbatim relocations; shared helpers live in
``tests/_github_fixtures.py``.
"""

from __future__ import annotations

from pathlib import Path

from _fake_transport import FakeAdapter, check_run, checks_reply, graphql_failure, make_github


def test_pr_checks_injects_run_id(tmp_path: Path) -> None:
    """Issue #391: pr_checks derives the GitHub Actions workflow run id from link."""
    link = "https://github.com/owner/repo/actions/runs/29525590823/job/87713099471"
    gh, _http, _ = make_github(
        tmp_path,
        http=FakeAdapter("http", [checks_reply(check_run("Tests passed", "FAILURE", link))]),
    )

    checks = gh.pr_checks(456)

    assert checks == [
        {
            "name": "Tests passed",
            "state": "FAILURE",
            "bucket": "fail",
            "link": link,
            "databaseId": 87713099471,
            "runId": 29525590823,
        }
    ]


def test_pr_checks_zero_checks_returns_empty_list_not_none(tmp_path: Path) -> None:
    """Issue #846: a PR with zero checks must return [], not None.

    ``gh pr checks`` used to exit non-zero with "no checks reported" and no
    JSON for such a PR, which needed a second ``pr view`` call to tell apart
    from a real failure. The rollup read answers it directly (B7): an empty
    rollup is an empty list, in exactly one call.
    """
    gh, http, _ = make_github(tmp_path, http=FakeAdapter("http", [checks_reply()]))

    assert gh.pr_checks(700) == []
    assert len(http.api_requests) == 1


def test_pr_checks_genuine_failure_returns_none(tmp_path: Path) -> None:
    """Issue #846: pr_checks must still return None for a real outage.

    A read the API rejected must preserve the "unavailable" contract (None)
    rather than masquerade as "no checks".
    """
    reply = graphql_failure("Could not resolve to a PullRequest with the number of 700.")
    gh, _http, _ = make_github(tmp_path, http=FakeAdapter("http", [reply]))

    assert gh.pr_checks(700) is None


def test_pr_checks_fallback_maps_transient_glitch_rollup(tmp_path: Path) -> None:
    """Issue #846: real checks must never be lost to a transient glitch.

    The statusCheckRollup *is* the read now (the separate fallback is gone), so
    the same rollup maps into the shape pr_checks returns -- including the
    databaseId/runId injection every consumer (checks.py, janitor.py,
    workflow.py) relies on.
    """
    reply = checks_reply(
        check_run("Tests", "SUCCESS", "https://github.com/owner/repo/actions/runs/111/job/222"),
        check_run("Lint", "IN_PROGRESS", "https://github.com/owner/repo/actions/runs/111/job/333"),
    )
    gh, _http, _ = make_github(tmp_path, http=FakeAdapter("http", [reply]))

    checks = gh.pr_checks(679)

    assert [(c["name"], c["state"], c["databaseId"], c["runId"]) for c in checks] == [
        ("Tests", "SUCCESS", 222, 111),
        ("Lint", "IN_PROGRESS", 333, 111),
    ] or sorted((c["name"], c["state"], c["databaseId"], c["runId"]) for c in checks) == [
        ("Lint", "IN_PROGRESS", 333, 111),
        ("Tests", "SUCCESS", 222, 111),
    ]


def test_pr_checks_fallback_declines_non_checkrun_rollup_entry(tmp_path: Path) -> None:
    """Issue #846 declined a non-CheckRun rollup entry rather than guess a mapping.

    The rollup is now the primary read and a ``StatusContext`` maps the way
    ``gh pr checks`` prints it (named by its context, bucketed from its
    state), so it is a row, not a reason to return None.
    """
    reply = checks_reply({"__typename": "StatusContext", "context": "ci/x", "state": "SUCCESS"})
    gh, _http, _ = make_github(tmp_path, http=FakeAdapter("http", [reply]))

    checks = gh.pr_checks(680)

    assert [(c["name"], c["bucket"]) for c in checks] == [("ci/x", "pass")]
