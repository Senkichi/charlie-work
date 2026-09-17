"""Regression tests for issue #1493: the internal merge flow marked a PR
``"merged"`` but never advanced the linked issue's ``state.json`` record,
leaving it frozen at ``"approved"`` with a stale cached-label snapshot
forever.

``merge_ready(..., merge=True)``'s post-``merge_pr`` lock block wrote
``prs[n].status = "merged"`` and cleared the issue's ``merge_alert`` latch --
but never ``issues[n].status``. Reconcile's ``stale_active_status`` sweep
deliberately excludes ``"approved"`` (merge finalization owns it), so nothing
ever repaired the record: the 120-issue corpus the issue body documents was
permanently split -- PR merged, issue approved, GitHub issue closed.

These tests live in their own module (rather than ``test_charlie_work.py``)
to stay under the file-size ratchet marks (issue #1442), mirroring how the
#1482 mirror-image fix extracted ``test_issue_1482.py`` /
``test_fix_unescalate.py``. The shared field-set helper lives in
``charlie_work.merge_finalize`` and is consumed by every lifecycle door that
records a merged PR -- internal merge, external-merge finalization, the
dispatch merged-PR-references close-out, and reconcile's
``merged_outside_orchestrator`` drift-fix.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from charlie_work.merge_finalize import _merged_issue_fields
from charlie_work.paths import runtime_paths
from charlie_work.reconcile import DriftItem, apply_fixes
from charlie_work.state import empty_state, load_state, save_state, state_lock
from charlie_work.workflow import OrchestratorApp

from _fakes_github import FakeGitHub
from _review_fixtures import _required_checks_config


def _write_approved_decision(tmp_path: Path, pr_number: int) -> None:
    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / f"pr-{pr_number}"
    pr_dir.mkdir(parents=True, exist_ok=True)
    (pr_dir / "review-decision.json").write_text(
        json.dumps({"decision": "approved", "reviewed_head_sha": "sha-abc123"}),
        encoding="utf-8",
    )


def _seed_issue(app: OrchestratorApp, issue_number: int, **fields: Any) -> dict[str, Any]:
    """Persist an issue record and return the entry that was seeded."""
    entry: dict[str, Any] = {"number": issue_number, **fields}
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["issues"][str(issue_number)] = entry
        save_state(app.paths.state_file, state)
    return entry


def _merge_app(tmp_path: Path):
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    return OrchestratorApp(tmp_path, paths, config, fake_gh), fake_gh, config


def test_internal_merge_finalizes_issue_state(tmp_path: Path) -> None:
    """Issue #1493: a successful internal merge advances the linked issue to
    the same terminal disposition as the PR -- ``status: "closed"``, the
    merge-alert latch cleared, and the stale intake label snapshot dropped --
    alongside the existing GitHub-side ``"merged"`` edge + issue close."""
    app, fake_gh, config = _merge_app(tmp_path)
    _write_approved_decision(tmp_path, 456)
    _seed_issue(
        app,
        123,
        status="approved",
        branch="agent/issue-123-fix-search",
        labels=["automated-ready", "bug"],  # stale intake-time snapshot
    )

    result = app.merge_ready(456, merge=True)
    assert result.ok is True

    state = load_state(app.paths.state_file)
    pr_entry = state["prs"]["456"]
    assert pr_entry["status"] == "merged"
    assert pr_entry["merged"] is True
    assert "merged_at" in pr_entry

    issue_entry = state["issues"]["123"]
    assert issue_entry["status"] == "closed"
    assert issue_entry["merge_alert"] == "OK"
    assert "labels" not in issue_entry

    # The GitHub-side edge is unchanged: done label on, workflow labels off,
    # issue closed.
    assert (123, config.labels.done) in fake_gh.labels_added
    removed_labels = {label for _n, label in fake_gh.labels_removed if _n == 123}
    assert config.labels.ready in removed_labels
    assert 123 in fake_gh.closed_issues


def test_internal_merge_failure_leaves_issue_untouched(tmp_path: Path) -> None:
    """Negative control: a merge that does not happen must not finalize the
    issue record -- status stays as seeded and the label cache is preserved."""
    app, _fake_gh, _config = _merge_app(tmp_path)
    _write_approved_decision(tmp_path, 456)
    _seed_issue(app, 123, status="approved", labels=["automated-ready"])

    result = app.merge_ready(456, merge=False)
    assert result.ok is True

    state = load_state(app.paths.state_file)
    issue_entry = state["issues"]["123"]
    # The evaluation-only pass may clear the merge-alert latch (pre-existing
    # benign behavior) but must never finalize the record.
    assert issue_entry["status"] == "approved"
    assert issue_entry["labels"] == ["automated-ready"]
    assert state["prs"]["456"].get("status") != "merged"


def test_merge_ready_reentry_converges_half_finalized_issue(tmp_path: Path) -> None:
    """A record the old code left half-finalized (PR merged, issue approved)
    converges when merge_ready re-enters on the already-merged short-circuit
    instead of staying stale forever."""
    app, _fake_gh, _config = _merge_app(tmp_path)
    _seed_issue(
        app,
        123,
        status="approved",
        merge_alert="merge_failed",
        labels=["agent:pr-open"],
    )
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "status": "merged",
            "merged": True,
        }
        save_state(app.paths.state_file, state)

    result = app.merge_ready(456, merge=True)
    assert result.ok is True

    issue_entry = load_state(app.paths.state_file)["issues"]["123"]
    assert issue_entry["status"] == "closed"
    assert issue_entry["merge_alert"] == "OK"
    assert "labels" not in issue_entry


class _FakeGitHubOutsideMergedWindow(FakeGitHub):
    """The merged PR exists but is invisible to ``merged_pr_list()``'s
    most-recent-500 window (issue #433 shape), forcing the external
    finalization sweep down its per-issue-search door."""

    def merged_pr_list(self):
        return []


def test_external_and_internal_doors_share_one_field_set(tmp_path: Path) -> None:
    """The external-merge finalization sweep produces byte-for-byte the same
    issue record the shared helper prescribes -- the two doors derive from one
    definition and cannot drift apart again."""
    from charlie_work.orchestration.state_merge_train import (
        _finalize_externally_merged_issues,
    )

    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = _FakeGitHubOutsideMergedWindow()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    seeded = _seed_issue(
        app,
        123,
        status="approved",
        merge_alert="merge_failed",
        labels=["agent:reviewing", "automated-ready"],
    )
    # GitHub reality: the mergequeue merged the PR and closed the issue.
    fake_gh.prs[0]["state"] = "MERGED"
    fake_gh.issues[0]["state"] = "CLOSED"
    fake_gh.issues[0]["labels"] = [{"name": config.labels.ready}]

    _finalize_externally_merged_issues(app)

    expected = _merged_issue_fields(seeded, 123)
    actual = load_state(app.paths.state_file)["issues"]["123"]
    assert actual == expected
    assert actual["status"] == "closed"
    assert "labels" not in actual


def test_dispatch_merged_pr_references_door_shares_field_set(tmp_path: Path) -> None:
    """dispatch()'s merged-PR-references close-out (the door a bound merged PR
    takes when its issue is already CLOSED on GitHub) applies the identical
    field set -- issue #427's shape, now deriving from the shared helper."""
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    seeded = _seed_issue(
        app,
        123,
        status="reviewing",
        merge_alert="merge_failed",
        labels=["agent:pr-open", "automated-ready"],
    )
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"]["456"] = {"status": "mergequeue", "issue_number": 123}
        save_state(app.paths.state_file, state)
    # The mergequeue merged the PR; GitHub auto-closed the issue.
    fake_gh.prs[0]["state"] = "MERGED"
    fake_gh.prs[0]["labels"] = [{"name": "mergequeue"}]
    fake_gh.issues[0]["state"] = "CLOSED"
    fake_gh.issues[0]["labels"] = [
        {"name": config.labels.ready},
        {"name": config.labels.pr_open},
    ]

    result = app.dispatch(limit=1)
    assert result.ok is True
    assert result.data["merged_pr_closed_issue_numbers"] == [123]

    expected = _merged_issue_fields(seeded, 123)
    actual = load_state(app.paths.state_file)["issues"]["123"]
    assert actual == expected
    assert actual["status"] == "closed"
    assert "labels" not in actual


def test_reconcile_merged_outside_orchestrator_door_shares_field_set() -> None:
    """reconcile's ``merged_outside_orchestrator`` drift-fix is the fifth
    door that records a merged PR -- an external merge discovered on a
    reconcile pass rather than through the fleet's own merge flow. It must
    apply the identical issue field set: a linked issue parked at
    ``"approved"`` converges to ``"closed"`` alongside
    ``prs[n].status = "merged"``. Without it, this door reproduced the exact
    #1493 dead zone (PR merged, issue approved, repair sweeps deliberately
    excluded) triggered by an externally-merged PR instead of ``merge_ready``."""
    config = _required_checks_config()
    gh = FakeGitHub()
    state = empty_state()
    seeded = {
        "number": 123,
        "status": "approved",
        "merge_alert": "merge_failed",
        "labels": ["agent:pr-open", "automated-ready"],
    }
    state["issues"]["123"] = dict(seeded)
    state["prs"]["456"] = {
        "number": 456,
        "issue_number": 123,
        "status": "reviewing",
    }
    drift = [
        DriftItem(
            kind="merged_outside_orchestrator",
            issue_number=123,
            pr_number=456,
            detail="PR #456 is MERGED on GitHub but state status is 'reviewing'",
            fix_actions=(
                "mark state prs[456].status = 'merged'",
                "finalize state issues[123] via merged-issue field set",
            ),
        )
    ]

    new_state = apply_fixes(gh, state, drift, config)

    assert new_state["prs"]["456"]["status"] == "merged"
    expected = _merged_issue_fields(seeded, 123)
    actual = new_state["issues"]["123"]
    assert actual == expected
    assert actual["status"] == "closed"
    assert actual["merge_alert"] == "OK"
    assert "labels" not in actual
    # apply_fixes returns a new state dict; the input record is untouched.
    assert state["issues"]["123"]["status"] == "approved"
    # The GitHub-side edge still runs alongside the state finalization --
    # labels via the "merged" transition, then the same explicit issue
    # close every other merged-PR door performs (without it the issue
    # lingers agent:done+OPEN and issue_status_normalized re-flags the
    # "closed" status as fresh drift on the next pass).
    assert (123, config.labels.done) in gh.labels_added
    assert 123 in gh.closed_issues


def test_merged_issue_fields_preserves_unrelated_fields() -> None:
    """The helper is a finalization overlay, not a record reset: fields it
    does not own (worker bookkeeping, escalation fields) pass through
    verbatim."""
    entry = {
        "number": 123,
        "status": "approved",
        "branch": "agent/issue-123-fix-search",
        "worker_pid": 4242,
        "labels": ["stale"],
    }
    merged = _merged_issue_fields(entry, 123)
    assert merged["status"] == "closed"
    assert merged["merge_alert"] == "OK"
    assert merged["branch"] == "agent/issue-123-fix-search"
    assert merged["worker_pid"] == 4242
    assert "labels" not in merged
    # Input dict is not mutated.
    assert entry["status"] == "approved"
    assert entry["labels"] == ["stale"]
