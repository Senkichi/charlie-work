"""Issue #2540: run-id resolution for check-run-linked required checks.

A same-name context a merged Actions job posts via
``POST /repos/{owner}/{repo}/check-runs`` carries the app-assigned
``details_url`` ``https://github.com/<owner>/<repo>/runs/<check_run_id>``
(GitHub ignores the caller-supplied value), so ``pr_checks``' link-derived
``runId`` injection yields ``None`` for it and both debounce classifiers
treated its first failure as definitive -- no flake-aware rerun was ever
scheduled. These tests pin the data-boundary enrichment
(``_enrich_checks_run_ids`` / ``_resolve_run_id_via_check_run`` in
``orchestration.misc_checks``) that resolves the workflow run id through
the check run itself, the end-to-end ``review()`` behavior, and the
merge-lane wiring in ``merge_path.gather.gather_checks`` (the
``merge_ready`` data boundary).

API shapes consumed by the resolver (REST, apiVersion 2022-11-28):
``GET /repos/{owner}/{repo}/check-runs/{id}`` returns ``check_suite``
(object-or-null with integer ``id``) and caller-supplied ``output``
(title/summary/text); the workflow-run object carries ``check_suite_id``,
and ``GET /repos/{owner}/{repo}/actions/runs`` accepts a
``check_suite_id`` query filter.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from _checks_fixtures import REQUIRED
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401
from _fakes_github_rerun import FakeGitHubWithRerunCapture
from _review_fixtures import _required_checks_config
from charlie_work.check_run_resolution import _check_run_id_from_link
from charlie_work.merge_path.gather import gather_checks
from charlie_work.orchestration.check_run_ids import (
    _enrich_checks_run_ids,
    _resolve_run_id_via_check_run,
)
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state
from charlie_work.workflow import OrchestratorApp

# Live values from issue #2540's evidence: the posted context's details_url
# (.../runs/<check_run_id>) and the real Actions job owning the head
# (/actions/runs/<run_id>/job/<job_id>).
CHECK_RUN_LINK = "https://github.com/owner/repo/runs/112064484398"
CHECK_RUN_ID = 112064484398
JOB_LINK = "https://github.com/owner/repo/actions/runs/37398606691/job/112063807090"
RUN_ID = 37398606691
SUITE_ID = 4242


class _AppStub:
    """Minimal stand-in exposing the resolution delegate as a real method.

    ``_enrich_checks_run_ids`` calls ``self._resolve_run_id_via_check_run`` --
    normally installed on ``OrchestratorApp`` by the delegation installer --
    so the stub binds the free function directly.
    """

    _resolve_run_id_via_check_run = _resolve_run_id_via_check_run  # type: ignore[misc]

    def __init__(self, gh: Any) -> None:
        self.gh = gh


class _FakeCheckRunGitHub(FakeGitHubWithRerunCapture):
    """FakeGitHubWithRerunCapture plus the two issue-#2540 resolution reads."""

    def __init__(
        self,
        checks: list[dict[str, Any]] | None = None,
        *,
        check_run_payloads: dict[int, dict[str, Any] | None] | None = None,
        runs_by_check_suite: dict[int, list[dict[str, Any]] | None] | None = None,
    ) -> None:
        super().__init__(checks)
        self.check_run_payloads = check_run_payloads or {}
        self.runs_by_check_suite = runs_by_check_suite or {}
        self.check_run_calls: list[int] = []
        self.suite_calls: list[int] = []

    def check_run(self, check_run_id: int) -> dict[str, Any] | None:
        self.check_run_calls.append(check_run_id)
        return self.check_run_payloads.get(check_run_id)

    def workflow_runs_for_check_suite(self, check_suite_id: int) -> list[dict[str, Any]] | None:
        self.suite_calls.append(check_suite_id)
        return self.runs_by_check_suite.get(check_suite_id)


def _enrich(fake_gh: Any, checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _enrich_checks_run_ids(_AppStub(fake_gh), checks, REQUIRED)


def test_enrich_resolves_run_id_from_output_summary_link() -> None:
    """Evidence source 1: the posting step's own ``output.summary`` still
    links the real Actions job, giving a one-call resolution."""
    fake_gh = _FakeCheckRunGitHub(
        check_run_payloads={
            CHECK_RUN_ID: {
                "name": "Tests passed",
                "output": {"summary": f"Failed; see the [job]({JOB_LINK})."},
                "check_suite": {"id": SUITE_ID},
            }
        }
    )
    checks = [{"name": "Tests passed", "state": "FAILURE", "link": CHECK_RUN_LINK}]

    enriched = _enrich(fake_gh, checks)

    assert enriched[0]["runId"] == RUN_ID
    # The payload handed an Actions link directly: no suite lookup needed.
    assert fake_gh.check_run_calls == [CHECK_RUN_ID]
    assert fake_gh.suite_calls == []
    assert _check_run_id_from_link(CHECK_RUN_LINK) == CHECK_RUN_ID


def test_enrich_resolves_run_id_via_check_suite() -> None:
    """Evidence source 2: no Actions link in the output -- the check run's
    ``check_suite.id`` is matched against
    ``GET /actions/runs?check_suite_id=<id>`` (the workflow run that posted
    the check run owns that suite)."""
    fake_gh = _FakeCheckRunGitHub(
        check_run_payloads={CHECK_RUN_ID: {"check_suite": {"id": SUITE_ID}}},
        runs_by_check_suite={SUITE_ID: [{"id": RUN_ID, "check_suite_id": SUITE_ID}]},
    )
    checks = [{"name": "Tests passed", "state": "FAILURE", "link": CHECK_RUN_LINK}]

    enriched = _enrich(fake_gh, checks)

    assert enriched[0]["runId"] == RUN_ID
    assert fake_gh.check_run_calls == [CHECK_RUN_ID]
    assert fake_gh.suite_calls == [SUITE_ID]


def test_enrich_leaves_unresolvable_check_unchanged() -> None:
    """A fetch failure or a suite owned by no workflow run degrades to the
    pre-#2540 behavior: ``runId`` stays None and the check is passed through
    unmodified (never a guessed id)."""
    fake_gh = _FakeCheckRunGitHub(
        check_run_payloads={
            CHECK_RUN_ID: None,  # GET /check-runs/{id} failed
            999: {
                "check_suite": {"id": 777},  # suite resolves to no workflow run
            },
        },
        runs_by_check_suite={777: []},
    )
    checks = [
        {"name": "Tests passed", "state": "FAILURE", "link": CHECK_RUN_LINK},
        {"name": "Lint & Format", "state": "FAILURE", "link": "https://github.com/o/r/runs/999"},
    ]

    enriched = _enrich(fake_gh, checks)

    assert enriched[0] == checks[0]
    assert enriched[1] == checks[1]


def test_enrich_skips_everything_it_does_not_need_to_touch() -> None:
    """Only failing/infra required checks with an unparseable run id and a
    ``/runs/<id>`` link are resolved: the common fleet pass spends zero
    extra API calls."""
    actions_link = "https://github.com/owner/repo/actions/runs/29525590823/job/87713099471"
    fake_gh = _FakeCheckRunGitHub(
        checks=[
            # Required + FAIL + Actions link: runId already parseable.
            {"name": "Tests passed", "state": "FAILURE", "link": actions_link},
            # Required + failing + no link at all: external status check.
            {"name": "Lint & Format", "state": "FAILURE"},
            # Required + pass + /runs link: no rerun consumer cares.
            {"name": "Pre-commit", "state": "SUCCESS", "link": CHECK_RUN_LINK},
            # Failing /runs link but NOT required: outside the gate.
            {"name": "informational", "state": "FAILURE", "link": CHECK_RUN_LINK},
        ]
    )
    # Feed the enrichment the same dicts production hands it: pr_checks has
    # already injected runId (None for the /runs links) at that point.
    enriched = _enrich(fake_gh, fake_gh.pr_checks(1))

    assert [c.get("runId") for c in enriched] == [29525590823, None, None, None]
    assert fake_gh.check_run_calls == []
    assert fake_gh.suite_calls == []


def test_enrich_passes_through_none_and_empty() -> None:
    app = _AppStub(_FakeCheckRunGitHub())
    assert _enrich_checks_run_ids(app, None, REQUIRED) == []
    assert _enrich_checks_run_ids(app, [], REQUIRED) == []


def test_review_flake_rerun_for_check_run_linked_required_check(tmp_path: Path) -> None:
    """End to end: the first failure of a ``.../runs/<id>``-linked required
    check now triggers the flake-aware rerun (issue #2540's impact) instead
    of being treated as definitive."""
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = _FakeCheckRunGitHub(
        checks=[
            {"name": "Tests passed", "state": "FAILURE", "link": CHECK_RUN_LINK},
            {"name": "Lint & Format", "bucket": "pass"},
            {"name": "Pre-commit", "state": "SUCCESS"},
        ],
        check_run_payloads={CHECK_RUN_ID: {"check_suite": {"id": SUITE_ID}}},
        runs_by_check_suite={SUITE_ID: [{"id": RUN_ID, "check_suite_id": SUITE_ID}]},
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is False
    assert result.data.get("rerun_run_ids") == [RUN_ID]
    assert len(fake_gh.rerun_calls) == 1
    assert fake_gh.rerun_calls[0][:3] == ["run", "rerun", str(RUN_ID)]
    assert "--failed" in fake_gh.rerun_calls[0]
    state = load_state(paths.state_file)
    assert state["prs"]["456"]["check_rerun_attempts"] == {
        "sha-abc123": {"Tests passed": [RUN_ID]}
    }
    assert (123, config.labels.needs_rework) not in fake_gh.labels_added


def test_review_resolution_failure_keeps_definitive_rework_routing(tmp_path: Path) -> None:
    """Fail-safe: when the check run cannot be resolved (fetch failure, or
    an external-CI check run whose suite belongs to no workflow run), the
    failure is definitive and routes to rework exactly as before #2540."""
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = _FakeCheckRunGitHub(
        checks=[
            {"name": "Tests passed", "state": "FAILURE", "link": CHECK_RUN_LINK},
            {"name": "Lint & Format", "bucket": "pass"},
            {"name": "Pre-commit", "state": "SUCCESS"},
        ],
        check_run_payloads={CHECK_RUN_ID: None},
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is True
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "rework_requested"
    assert (123, config.labels.needs_rework) in fake_gh.labels_added
    assert fake_gh.rerun_calls == []


def test_review_infra_rerun_for_check_run_linked_required_check(tmp_path: Path) -> None:
    """The same resolution serves the infra-rerun driver: a CANCELLED
    ``.../runs/<id>``-linked required check is retried (without --failed)
    instead of escalating on first sight."""
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = _FakeCheckRunGitHub(
        checks=[
            {"name": "Tests passed", "state": "CANCELLED", "link": CHECK_RUN_LINK},
            {"name": "Lint & Format", "bucket": "pass"},
            {"name": "Pre-commit", "state": "SUCCESS"},
        ],
        check_run_payloads={CHECK_RUN_ID: {"output": {"summary": f"Job: {JOB_LINK}"}}},
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is False
    assert result.data.get("infra_rerun_run_ids") == [RUN_ID]
    assert fake_gh.rerun_calls == [["run", "rerun", str(RUN_ID)]]
    assert "--failed" not in fake_gh.rerun_calls[0]


@pytest.mark.parametrize("state", ["CANCELLED", "FAILURE"])
def test_gather_checks_resolves_run_id_for_check_run_linked_required_check(
    tmp_path: Path, state: str
) -> None:
    """Merge-lane wiring: ``gather_checks`` (the ``merge_ready``/preview data
    boundary in ``merge_path/gather.py``) must run the same
    ``_enrich_checks_run_ids`` enrichment ``review()`` does. A required
    check in a rerun-relevant state (``CANCELLED`` for the infra lane,
    ``FAILURE`` for the flake-debounce lane) whose link is the app-assigned
    ``.../runs/<id>`` check-run shape must come out of ``gather_checks``
    with ``runId`` resolved. Without that call the id stays ``None`` and
    the merge lane's ``classify_infra_failures``/``classify_check_failures``
    see an unrerunnable failure -- definitive on first sight."""
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = _FakeCheckRunGitHub(
        checks=[
            {"name": "Tests passed", "state": state, "link": CHECK_RUN_LINK},
            {"name": "Lint & Format", "bucket": "pass"},
            {"name": "Pre-commit", "state": "SUCCESS"},
        ],
        check_run_payloads={CHECK_RUN_ID: {"check_suite": {"id": SUITE_ID}}},
        runs_by_check_suite={SUITE_ID: [{"id": RUN_ID, "check_suite_id": SUITE_ID}]},
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    read = gather_checks(app, 456)

    assert not read.unavailable
    assert read.enriched[0]["name"] == "Tests passed"
    assert read.enriched[0]["runId"] == RUN_ID
    assert fake_gh.check_run_calls == [CHECK_RUN_ID]
    assert fake_gh.suite_calls == [SUITE_ID]
    if state == "CANCELLED":
        assert read.summary.infra_failed == ("Tests passed",)
    else:
        assert read.summary.failed == ("Tests passed",)


def test_merge_ready_infra_rerun_for_check_run_linked_required_check(tmp_path: Path) -> None:
    """Merge-lane end to end: an approved-at-live-head PR whose CANCELLED
    required check is ``.../runs/<id>``-linked gets the resolved workflow
    run retried via ``gh run rerun`` -- the carried-forward lane never
    re-enters ``review()``, so this is the consumer the ``gather_checks``
    wiring exists for (mirrors
    ``test_charlie_work_merge_ready_infra_rerun``'s approved-head setup).
    Without the ``gather_checks`` enrichment the runId stays ``None`` and
    the pass escalates immediately instead of rerunning."""
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = _FakeCheckRunGitHub(
        checks=[
            {"name": "Tests passed", "state": "CANCELLED", "link": CHECK_RUN_LINK},
            {"name": "Lint & Format", "bucket": "pass"},
            {"name": "Pre-commit", "state": "SUCCESS"},
        ],
        check_run_payloads={CHECK_RUN_ID: {"check_suite": {"id": SUITE_ID}}},
        runs_by_check_suite={SUITE_ID: [{"id": RUN_ID, "check_suite_id": SUITE_ID}]},
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    decision_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    decision_dir.mkdir(parents=True)
    (decision_dir / "review-decision.json").write_text(
        json.dumps({"decision": "approved", "reviewed_head_sha": "sha-abc123"}),
        encoding="utf-8",
    )

    result = app.merge_ready(456)

    assert result.ok is True
    assert result.data["can_merge"] is False
    assert result.data["merged"] is False
    assert result.data.get("infra_rerun_run_ids") == [RUN_ID]
    assert fake_gh.rerun_calls == [["run", "rerun", str(RUN_ID)]]
    assert "--failed" not in fake_gh.rerun_calls[0]
    state = load_state(paths.state_file)
    assert state["prs"]["456"]["infra_rerun_attempts"] == {
        "sha-abc123": {"Tests passed": {str(RUN_ID): 1}}
    }
    # Not escalated, not merged, no rework label -- remediation is in flight.
    assert (123, config.labels.operator_queue) not in fake_gh.labels_added
    assert (123, config.labels.needs_rework) not in fake_gh.labels_added
    assert fake_gh.merged == []
