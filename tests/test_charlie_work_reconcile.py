"""reconcile drift detection/fix: wiring, exit codes, mergequeue-label removal, partial failure, dead-session classification.

Split out of ``tests/test_charlie_work.py`` (issue #1554,
Track-1 wave 8/8).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import pytest
from _fakes_github import FakeGitHub
from charlie_work.config import (
    OrchestratorConfig,
    PostMortemConfig,
)
from charlie_work.paths import (
    resolved_layout,
    runtime_paths,
)
from charlie_work.state import (
    empty_state,
    save_state,
)
from charlie_work.workflow import OrchestratorApp
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401


def test_reconcile_wiring_reports_clean_repo(tmp_path: Path) -> None:
    class QuietGitHub(FakeGitHub):
        def run(self, arguments, *, json_output=False, allow_failure=False):
            # Handle dependency API calls
            if "dependencies" in " ".join(arguments):
                return [] if json_output else ""
            return []

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, QuietGitHub())

    result = app.reconcile()

    assert result.ok is True
    assert result.data["drift"] == []
    assert result.data["fixed"] is False


def test_reconcile_exit_nonzero_when_drift_found_and_not_fixed(tmp_path: Path) -> None:
    """mop-up without --fix must exit non-zero when drift is present (CI gateable)."""

    class DriftGitHub(FakeGitHub):
        def run(self, arguments, *, json_output=False, allow_failure=False):
            # Handle dependency API calls
            if "dependencies" in " ".join(arguments):
                return [] if json_output else ""
            # paginated PR list from reconcile._fetch_prs
            if arguments[0] == "api" and "pulls?state=all" in arguments[1]:
                return [
                    {
                        "number": 456,
                        "title": "fix",
                        "url": "u",
                        "headRefName": "agent/issue-123-x",
                        "baseRefName": "main",
                        "body": "",
                        "state": "MERGED",
                        "labels": [],
                        "isCrossRepository": False,
                    }
                ]
            # paginated issue list from reconcile._fetch_issues
            if arguments[0] == "api" and "issues?state=all" in arguments[1]:
                return [
                    {
                        "number": 123,
                        "title": "t",
                        "url": "u",
                        "body": "",
                        "labels": [{"name": "agent:in-progress"}],
                    }
                ]
            return []

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, DriftGitHub())

    result = app.reconcile(fix=False)

    assert result.ok is False
    assert result.data["fixed"] is False
    assert len(result.data["drift"]) > 0


def test_reconcile_exit_ok_when_drift_fixed(tmp_path: Path) -> None:
    """mop-up --fix must exit zero when all drift is repaired."""
    config = OrchestratorConfig()

    class DriftGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self._pr = {
                "number": 456,
                "title": "fix",
                "url": "u",
                "headRefName": "agent/issue-123-x",
                "baseRefName": "main",
                "body": "",
                "state": "MERGED",
                "labels": [],
                "isCrossRepository": False,
                "headRepositoryOwner": "owner",
                "baseRepositoryOwner": "owner",
            }
            self._issue = {
                "number": 123,
                "title": "t",
                "url": "u",
                "body": "",
                "labels": [{"name": "agent:in-progress"}],
            }

        def run(self, arguments, *, json_output=False, allow_failure=False):
            # Handle dependency API calls
            if "dependencies" in " ".join(arguments):
                return [] if json_output else ""
            if arguments[0] == "api" and "pulls?state=all" in arguments[1]:
                return [self._pr]
            if arguments[0] == "api" and "issues?state=all" in arguments[1]:
                return [self._issue]
            return []

        def remove_issue_label(self, number: int, label: str) -> None:
            super().remove_issue_label(number, label)
            self._issue["labels"] = [
                item for item in self._issue["labels"] if item.get("name") != label
            ]

        def add_issue_label(self, number: int, label: str) -> None:
            super().add_issue_label(number, label)
            names = {item.get("name") for item in self._issue["labels"]}
            if label not in names:
                self._issue["labels"].append({"name": label})

        def close_issue(self, number: int) -> bool:
            # Mirror the label overrides above: a real close is visible to
            # the next issues snapshot, so flip the state this fake serves.
            ok = super().close_issue(number)
            if number == self._issue["number"]:
                self._issue["state"] = "CLOSED"
            return ok

    app = OrchestratorApp(
        tmp_path, runtime_paths(tmp_path, config.runtime.state_dir), config, DriftGitHub()
    )

    result = app.reconcile(fix=True)

    assert result.ok is True
    assert result.data["fixed"] is True
    assert result.data["drift_before"] == 1
    assert result.data["drift_after"] == 0
    assert result.data["remaining_drift"] == []


def test_reconcile_removes_mergequeue_label_via_full_stack(tmp_path: Path) -> None:
    """Wiring check for issue #819. Every other ``detect_mergequeue_not_approved``
    test (test_reconcile.py) calls the detector directly with a config it
    builds itself; none of them prove the detector is actually reached
    through ``app.reconcile()``. Every reconcile test *in this file* uses
    the default ``OrchestratorConfig()``, where ``auto_merge.mergequeue_label``
    is ``None`` -- so the detector's very first line (``if not
    mergequeue_label: return []``) short-circuits before doing anything,
    and green tests here would prove nothing about wiring (the exists/
    substantive/wired distinction). This test configures the label and
    drives a real ``request_changes``-at-head PR through
    ``app.reconcile(fix=True)`` end to end, asserting the label actually
    comes off via ``GitHub.remove_pr_label`` -- the same mechanical step
    that was missing when Aviator merged PR #695 over a standing
    request-changes verdict."""
    config = dataclasses.replace(
        OrchestratorConfig(),
        auto_merge=dataclasses.replace(
            OrchestratorConfig().auto_merge, mergequeue_label="mergequeue"
        ),
    )
    mergequeue_label = config.auto_merge.mergequeue_label
    assert mergequeue_label is not None

    class MergequeueGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self._pr = {
                "number": 819,
                "title": "fix",
                "url": "u",
                "headRefName": "agent/issue-819-x",
                "baseRefName": "main",
                "headRefOid": "sha-819-live",
                "body": "",
                "state": "OPEN",
                "labels": [{"name": mergequeue_label}],
                "isCrossRepository": False,
                "headRepositoryOwner": "owner",
                "baseRepositoryOwner": "owner",
            }
            self.pr_labels_removed: list[tuple[int, str]] = []

        def run(self, arguments, *, json_output=False, allow_failure=False):
            if "dependencies" in " ".join(arguments):
                return [] if json_output else ""
            if arguments[0] == "api" and "pulls?state=all" in arguments[1]:
                return [self._pr]
            if arguments[0] == "api" and "issues?state=all" in arguments[1]:
                return []
            return []

        def remove_pr_label(self, number: int, label: str) -> bool:
            self.pr_labels_removed.append((number, label))
            self._pr["labels"] = [item for item in self._pr["labels"] if item.get("name") != label]
            return True

    gh = MergequeueGitHub()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    pr_dir = paths.prs / "pr-819"
    pr_dir.mkdir(parents=True, exist_ok=True)
    (pr_dir / "review-decision.json").write_text(
        json.dumps({"decision": "request_changes", "reviewed_head_sha": "sha-819-live"}),
        encoding="utf-8",
    )

    app = OrchestratorApp(tmp_path, paths, config, gh)

    app.reconcile(fix=True)

    assert gh.pr_labels_removed == [(819, mergequeue_label)]
    assert mergequeue_label not in [item.get("name") for item in gh._pr["labels"]]


def test_reconcile_partial_fix_failure_reports_remaining_drift(tmp_path: Path) -> None:
    """mop-up --fix must exit non-zero when a label removal silently fails."""
    config = OrchestratorConfig()

    class FailingRemoveGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self._issue = {
                "number": 30,
                "title": "t",
                "url": "u",
                "body": "",
                "labels": [{"name": "agent:in-progress"}],
            }

        def run(self, arguments, *, json_output=False, allow_failure=False):
            # Handle dependency API calls
            if "dependencies" in " ".join(arguments):
                return [] if json_output else ""
            if arguments[0] == "api" and "pulls?state=all" in arguments[1]:
                return []
            if arguments[0] == "api" and "issues?state=all" in arguments[1]:
                return [self._issue]
            return []

        def remove_issue_label(self, number: int, label: str) -> None:
            # Simulate allow_failure=True silently dropping the removal.
            pass

    app = OrchestratorApp(
        tmp_path, runtime_paths(tmp_path, config.runtime.state_dir), config, FailingRemoveGitHub()
    )

    result = app.reconcile(fix=True)

    assert result.ok is False
    assert result.data["fixed"] is False
    assert result.data["drift_before"] >= 1  # May be multiple if both adapters read the same issue
    assert result.data["drift_after"] >= 1
    assert len(result.data["remaining_drift"]) >= 1
    assert result.data["remaining_drift"][0]["kind"] == "issue_active_label_no_open_pr"
    assert "partially fixed" in result.message


def test_reconcile_closed_unmerged_pr_does_not_drop_escalated_issue_status(
    tmp_path: Path,
) -> None:
    """D-2 regression guard for issue #1066: an OPEN escalated issue whose
    linked PR is CLOSED-unmerged must NOT have its ``status`` key dropped by
    the ``closed_unmerged_pr_issue_state_converged`` drift kind.

    Before #1066's ``DORMANT_CONVERGENCE_EXCLUDED_STATUSES`` exclusion, this
    path dropped the ``status`` key for any ``ACTIVE_STATE_STATUSES`` member
    -- including ``"escalated"`` -- silently detaching the state entry from
    its still-live ``agent:human-needed`` label with no repair path back into
    the human queue (fired in production: issue #894 via PR #948). The
    sibling ``issue_status_normalized`` sweep already excluded escalated via
    ``ORCHESTRATOR_OWNED_ISSUE_STATUSES``; the asymmetry was the defect.

    This test calls ``detect_drift``/``apply_fixes`` directly (the same
    surface the reviewer reproduced the bug on) and asserts both that no
    ``closed_unmerged_pr_issue_state_converged`` drift item is emitted for
    the escalated issue and that the ``status`` key survives ``apply_fixes``.
    """
    from charlie_work.reconcile import apply_fixes, detect_drift

    config = OrchestratorConfig()
    gh = FakeGitHub()
    # OPEN escalated issue with the terminal human-needed label already
    # present -- the exact live shape of issue #894 at the time of the
    # production incident.
    gh.issues = [
        {
            "number": 894,
            "title": "issue 894",
            "url": "https://example.test/issues/894",
            "body": "",
            "labels": [{"name": config.labels.human_needed}],
            "state": "OPEN",
        }
    ]
    # CLOSED-unmerged PR linked to issue #894 via the branch-name convention.
    gh.prs = [
        {
            "number": 948,
            "title": "Fix #894",
            "url": "https://example.test/pull/948",
            "headRefName": "agent/issue-894-fix",
            "baseRefName": "main",
            "headRefOid": "sha-948",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #894",
            "labels": [],
            "isCrossRepository": False,
            "state": "CLOSED",
        }
    ]

    state = empty_state()
    state["issues"]["894"] = {"number": 894, "status": "escalated"}
    state["prs"]["948"] = {"number": 948, "status": "reviewing", "issue_number": 894}
    state_file = tmp_path / "state.json"
    save_state(state_file, state)

    drift = detect_drift(gh, state, config, repo_root=tmp_path)

    # No closed_unmerged_pr_issue_state_converged drift item for the escalated
    # issue -- the DORMANT_CONVERGENCE_EXCLUDED_STATUSES exclusion (#1066)
    # prevents it.
    issue_converged = [
        d
        for d in drift
        if d.kind == "closed_unmerged_pr_issue_state_converged" and d.issue_number == 894
    ]
    assert issue_converged == [], (
        f"escalated issue should be excluded from closed_unmerged_pr_issue_state_"
        f"converged, got: {issue_converged}"
    )

    new_state = apply_fixes(gh, state, drift, config, repo_root=tmp_path, state_path=state_file)

    # D-2: the escalated issue's status key must survive -- not dropped to
    # None and not rewritten to any other value.
    assert new_state["issues"]["894"]["status"] == "escalated", (
        f"escalated issue status was rewritten to {new_state['issues']['894'].get('status')!r}"
    )


def test_detect_drift_api_dead_session_provider_auth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #484 review finding: the ``elif w.adapter_kind == "api"`` branch in
    ``reconcile.detect_drift``'s dead-session lane classifies a dead api
    worker with a 401 log tail as ``provider_auth`` and emits a
    ``provider_throttle_detected`` drift item. A wiring regression that drops
    this branch leaves no throttle drift item. The sidecar is reaped by this
    lane, so the classification is asserted via the drift list.
    """
    from _api_budget_fixtures import api_worker_config, write_api_sidecar
    from charlie_work.reconcile import detect_drift

    config = OrchestratorConfig(
        api_worker=api_worker_config(),
        post_mortem=PostMortemConfig(enabled=False),
    )
    gh = FakeGitHub()
    gh.issues = [
        {
            "number": 4805,
            "title": "Test issue",
            "url": "https://example.test/issues/4805",
            "body": "",
            "labels": [{"name": config.labels.in_progress}],
            "state": "OPEN",
        }
    ]
    gh.prs = []
    state = empty_state()

    sessions_dir = resolved_layout(config, tmp_path).sessions_dir
    sessions_dir.mkdir(parents=True, exist_ok=True)
    write_api_sidecar(sessions_dir, 4805, provider="example", pid=99996)

    log_path = sessions_dir / "issue-4805.claude.log"
    log_path.write_text("Error: 401 Unauthorized. Invalid API key.\n", encoding="utf-8")

    # The dead-session lane fires only when the worker is not alive.
    monkeypatch.setattr("charlie_work.worker.is_worker_alive", lambda record: False)

    drift = detect_drift(gh, state, config, repo_root=tmp_path)

    # A provider_throttle_detected drift item is emitted with provider_auth.
    throttle_items = [d for d in drift if d.kind == "provider_throttle_detected"]
    assert len(throttle_items) == 1
    assert throttle_items[0].issue_number == 4805
    assert "provider_auth" in throttle_items[0].detail
