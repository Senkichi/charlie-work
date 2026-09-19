"""``GitHub.pr_checks`` tests: runId injection and the statusCheckRollup
disambiguation fallback (issue #846).

Split out of ``tests/test_github.py`` (issue #1572, Track 1 shoulder) --
bodies are verbatim relocations; shared helpers live in
``tests/_github_fixtures.py``.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from charlie_work import github as github_module


def test_pr_checks_injects_run_id(monkeypatch, tmp_path: Path) -> None:
    """Issue #391: pr_checks derives the GitHub Actions workflow run id from link."""

    def fake_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout='[{"name":"Tests passed","state":"FAILURE","link":"https://github.com/owner/repo/actions/runs/29525590823/job/87713099471"}]',
            stderr="",
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    checks = gh.pr_checks(456)

    assert checks == [
        {
            "name": "Tests passed",
            "state": "FAILURE",
            "link": "https://github.com/owner/repo/actions/runs/29525590823/job/87713099471",
            "databaseId": 87713099471,
            "runId": 29525590823,
        }
    ]


def test_pr_checks_zero_checks_returns_empty_list_not_none(monkeypatch, tmp_path: Path) -> None:
    """Issue #846: a PR with zero checks must return [], not None.

    `gh pr checks` exits non-zero with "no checks reported" and no JSON when a
    PR has no checks yet -- a shape indistinguishable from a genuine command
    failure using result.ok/result.value alone (measured against PR #700 in
    this repo). pr_checks must disambiguate via the statusCheckRollup fallback
    and return [] here, not treat it as an infrastructure failure.
    """
    calls: list[list[str]] = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(cmd)
        if cmd[1:3] == ["pr", "checks"]:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=1,
                stdout="",
                stderr="no checks reported on the 'agent/issue-627-fix' branch",
            )
        assert cmd[1:3] == ["pr", "view"], cmd
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout='{"statusCheckRollup":[]}',
            stderr="",
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    checks = gh.pr_checks(700)

    assert checks == []
    # Exactly one disambiguating fallback call, not a retry storm.
    assert len(calls) == 2


def test_pr_checks_genuine_failure_returns_none(monkeypatch, tmp_path: Path) -> None:
    """Issue #846: pr_checks must still return None for a real outage.

    When `gh pr checks` fails AND the statusCheckRollup disambiguation
    fallback also fails, pr_checks must preserve its existing "unavailable"
    contract (None) rather than silently mask the outage as "no checks".
    """

    def fake_run(cmd, *args, **kwargs):
        if cmd[1:3] == ["pr", "checks"]:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=1,
                stdout="",
                stderr='Unknown JSON field: "bucket"',
            )
        assert cmd[1:3] == ["pr", "view"], cmd
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=1,
            stdout="",
            stderr="GraphQL: Could not resolve to a PullRequest with the number of 700.",
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    checks = gh.pr_checks(700)

    assert checks is None


def test_pr_checks_fallback_maps_transient_glitch_rollup(monkeypatch, tmp_path: Path) -> None:
    """Issue #846: a transient `gh pr checks` glitch must not lose real checks.

    If `gh pr checks` fails but the PR genuinely has checks, the
    statusCheckRollup fallback maps them into the same shape pr_checks
    normally returns -- including the databaseId/runId injection every
    consumer (checks.py, janitor.py, workflow.py) relies on -- rather than
    collapsing them to None.
    """

    def fake_run(cmd, *args, **kwargs):
        if cmd[1:3] == ["pr", "checks"]:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=1,
                stdout="",
                stderr="unexpected end of JSON input",
            )
        assert cmd[1:3] == ["pr", "view"], cmd
        rollup = {
            "statusCheckRollup": [
                {
                    "__typename": "CheckRun",
                    "name": "Tests",
                    "status": "COMPLETED",
                    "conclusion": "SUCCESS",
                    "detailsUrl": "https://github.com/owner/repo/actions/runs/111/job/222",
                },
                {
                    "__typename": "CheckRun",
                    "name": "Lint",
                    "status": "IN_PROGRESS",
                    "conclusion": "",
                    "detailsUrl": "https://github.com/owner/repo/actions/runs/111/job/333",
                },
            ]
        }
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout=json.dumps(rollup), stderr=""
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    checks = gh.pr_checks(679)

    assert checks == [
        {
            "name": "Tests",
            "state": "SUCCESS",
            "link": "https://github.com/owner/repo/actions/runs/111/job/222",
            "databaseId": 222,
            "runId": 111,
        },
        {
            "name": "Lint",
            "state": "IN_PROGRESS",
            "link": "https://github.com/owner/repo/actions/runs/111/job/333",
            "databaseId": 333,
            "runId": 111,
        },
    ]


def test_pr_checks_fallback_declines_non_checkrun_rollup_entry(
    monkeypatch, tmp_path: Path
) -> None:
    """Issue #846: an unrecognized rollup entry shape must not be guessed at.

    A `statusCheckRollup` entry that isn't a GitHub Actions `CheckRun` (e.g. an
    external `StatusContext`) has no field mapping verified against this repo.
    Rather than fabricate a lossy mapping, pr_checks declines and returns None
    -- the same "unavailable" outcome as before issue #846's fix, never worse.
    """

    def fake_run(cmd, *args, **kwargs):
        if cmd[1:3] == ["pr", "checks"]:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=1,
                stdout="",
                stderr="no checks reported on the 'x' branch",
            )
        assert cmd[1:3] == ["pr", "view"], cmd
        rollup = {"statusCheckRollup": [{"__typename": "StatusContext", "state": "SUCCESS"}]}
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout=json.dumps(rollup), stderr=""
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    checks = gh.pr_checks(700)

    assert checks is None
