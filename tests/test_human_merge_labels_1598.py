"""Tests for issue #1598: configurable human-merge labels.

Covers every enforcement point:

* ``DispatchConfig.human_merge_labels`` parsing/normalization (default empty,
  YAML list, single string, invalid types).
* ``state.ESCALATION_REASON_CLASSES`` accepts ``"policy"``.
* ``labels`` exposes a ``human_merge_required`` edge that adds
  ``operator_queue``.
* ``_merge_train_candidates`` excludes a bound PR whose issue carries a
  configured human-merge label.
* ``merge_ready`` (real path) never merges or queues such a PR, transitions
  the issue to ``agent:operator-queue`` with ``reason_class="policy"``, and
  posts the orchestrator comment exactly once across passes.
* Removing the label mid-flight restores normal queue/merge behaviour on
  the next pass (live-label decision, no ``charlie unescalate`` required).
* Default empty ``human_merge_labels`` preserves existing behaviour.
* The janitor surfaces an informational warning (not a failure) when the
  bound issue carries a configured human-merge label.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from _fakes_github import FakeGitHub
from _helpers import EXAMPLES_DIR
from _review_fixtures import _approved_automerge
from charlie_work.config import (
    AutoMergeConfig,
    ConfigError,
    DispatchConfig,
    LabelConfig,
    OrchestratorConfig,
    load_config,
)
from charlie_work.github import GitHubError
from charlie_work.janitor import run_janitor
from charlie_work.labels import _edges
from charlie_work.paths import runtime_paths
from charlie_work.state import ESCALATION_REASON_CLASSES, load_state
from charlie_work.workflow import OrchestratorApp


def _mergequeue_automerge(label: str = "mergequeue") -> AutoMergeConfig:
    return AutoMergeConfig(
        required_checks=(),
        require_approved_review=True,
        mergequeue_label=label,
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_dispatch_config_human_merge_labels_default_empty() -> None:
    """Default DispatchConfig has an empty human_merge_labels tuple."""
    cfg = DispatchConfig()
    assert cfg.human_merge_labels == ()


def test_dispatch_config_human_merge_labels_normalizes_string_to_tuple() -> None:
    """A bare string is normalized to a one-element tuple, matching the
    existing comment-filter sequence normalization."""
    cfg = DispatchConfig(human_merge_labels="needs-design")
    assert cfg.human_merge_labels == ("needs-design",)


def test_dispatch_config_human_merge_labels_normalizes_list_to_tuple() -> None:
    cfg = DispatchConfig(human_merge_labels=["needs-design", "docs-only"])
    assert cfg.human_merge_labels == ("needs-design", "docs-only")


def test_dispatch_config_human_merge_labels_normalizes_tuple() -> None:
    cfg = DispatchConfig(human_merge_labels=("needs-design",))
    assert cfg.human_merge_labels == ("needs-design",)


def test_load_config_human_merge_labels_yaml_list(tmp_path: Path) -> None:
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(
        "dispatch:\n  human_merge_labels:\n    - needs-design\n    - docs-only\n",
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)
    assert cfg.dispatch.human_merge_labels == ("needs-design", "docs-only")


def test_load_config_human_merge_labels_yaml_single_string(tmp_path: Path) -> None:
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(
        "dispatch:\n  human_merge_labels:\n    - needs-design\n",
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)
    assert cfg.dispatch.human_merge_labels == ("needs-design",)


def test_load_config_human_merge_labels_rejects_non_list(tmp_path: Path) -> None:
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text("dispatch:\n  human_merge_labels: 42\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(cfg_path)


def test_load_config_human_merge_labels_rejects_non_string_element(
    tmp_path: Path,
) -> None:
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text("dispatch:\n  human_merge_labels:\n    - 42\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(cfg_path)


def test_load_config_human_merge_labels_default_empty_when_absent(
    tmp_path: Path,
) -> None:
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text("dispatch:\n  branch_prefix: agent/issue\n", encoding="utf-8")
    cfg = load_config(cfg_path)
    assert cfg.dispatch.human_merge_labels == ()


# ---------------------------------------------------------------------------
# Reason class
# ---------------------------------------------------------------------------


def test_policy_in_escalation_reason_classes() -> None:
    assert "policy" in ESCALATION_REASON_CLASSES
    assert "mechanical" in ESCALATION_REASON_CLASSES
    assert "judgment" in ESCALATION_REASON_CLASSES


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


def test_human_merge_required_edge_exists_and_adds_operator_queue() -> None:
    """The human_merge_required edge must add operator_queue (the
    agent:operator-queue label) and be a distinct named edge."""
    edges = _edges(LabelConfig())
    assert "human_merge_required" in edges
    add, _remove = edges["human_merge_required"]
    # The default LabelConfig.operator_queue is "agent:operator-queue".
    assert LabelConfig().operator_queue in add


def test_human_merge_required_edge_distinct_from_operator_queued() -> None:
    """human_merge_required is a distinct named edge from operator_queued
    so the transition is attributable in events.db."""
    edges = _edges(LabelConfig())
    assert "human_merge_required" in edges
    assert "operator_queued" in edges
    assert "human_merge_required" != "operator_queued"


# ---------------------------------------------------------------------------
# _merge_train_candidates
# ---------------------------------------------------------------------------


def _human_merge_config(
    *, mergequeue: bool = False, labels: tuple[str, ...] = ("needs-design",)
) -> OrchestratorConfig:
    return OrchestratorConfig(
        auto_merge=_mergequeue_automerge() if mergequeue else _approved_automerge(),
        dispatch=DispatchConfig(human_merge_labels=labels),
    )


def test_merge_train_candidates_excludes_human_merge_labeled_pr(
    tmp_path: Path,
) -> None:
    """A bound PR whose issue carries a configured human-merge label is
    excluded from merge-train candidates."""
    config = _human_merge_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.issues[0]["labels"] = [
        {"name": "automated-ready"},
        {"name": "needs-design"},
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="ok", verdict_provenance="fresh_llm_review")

    candidates = app._merge_train_candidates(prs=fake_gh.prs)
    pr_numbers = [pr_number for _sort_key, pr_number, _pr, _decision, _head in candidates]
    assert 456 not in pr_numbers


def test_merge_train_candidates_includes_pr_when_label_removed(
    tmp_path: Path,
) -> None:
    """Removing the human-merge label restores the PR as a candidate on the
    next pass (live-label decision)."""
    config = _human_merge_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="ok", verdict_provenance="fresh_llm_review")

    # Label present -> excluded.
    fake_gh.issues[0]["labels"] = [
        {"name": "automated-ready"},
        {"name": "needs-design"},
    ]
    candidates_with = app._merge_train_candidates(prs=fake_gh.prs)
    assert not any(pr_number == 456 for _k, pr_number, _p, _d, _h in candidates_with)

    # Label removed -> included.
    fake_gh.issues[0]["labels"] = [{"name": "automated-ready"}]
    candidates_without = app._merge_train_candidates(prs=fake_gh.prs)
    pr_numbers = [pr_number for _k, pr_number, _p, _d, _h in candidates_without]
    assert 456 in pr_numbers


def test_merge_train_candidates_default_empty_preserves_behavior(
    tmp_path: Path,
) -> None:
    """Default empty human_merge_labels preserves existing candidate
    selection."""
    config = OrchestratorConfig(auto_merge=_approved_automerge())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    # Issue carries a label that would be a human-merge label IF configured,
    # but human_merge_labels is empty so it must not be excluded.
    fake_gh.issues[0]["labels"] = [
        {"name": "automated-ready"},
        {"name": "needs-design"},
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="ok", verdict_provenance="fresh_llm_review")

    candidates = app._merge_train_candidates(prs=fake_gh.prs)
    pr_numbers = [pr_number for _k, pr_number, _p, _d, _h in candidates]
    assert 456 in pr_numbers


# ---------------------------------------------------------------------------
# merge_ready — real path
# ---------------------------------------------------------------------------


def test_merge_ready_human_merge_label_prevents_merge_and_transitions_issue(
    tmp_path: Path,
) -> None:
    """A merge-ready PR whose issue carries a human-merge label is not
    merged, not queued, transitions the issue to escalated with
    reason_class='policy', and lands in agent:operator-queue."""
    config = _human_merge_config(mergequeue=True)
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.issues[0]["labels"] = [
        {"name": "automated-ready"},
        {"name": "needs-design"},
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="ok", verdict_provenance="fresh_llm_review")

    result = app.merge_ready(456, merge=True)

    assert result.data["can_merge"] is True
    assert result.data["human_merge_hold"] is True
    assert result.data["mergequeue_label_applied"] is None
    assert fake_gh.pr_labels_added == []
    assert fake_gh.merged == []
    # Issue transitioned to escalated with reason_class="policy".
    state = load_state(paths.state_file)
    issue_entry = state["issues"]["123"]
    assert issue_entry["status"] == "escalated"
    assert issue_entry["reason_class"] == "policy"
    # operator_queue label was added to the issue.
    assert (123, config.labels.operator_queue) in fake_gh.labels_added


def test_merge_ready_human_merge_comment_posted_once(
    tmp_path: Path,
) -> None:
    """The orchestrator PR comment is posted exactly once across passes."""
    config = _human_merge_config(mergequeue=True)
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.issues[0]["labels"] = [
        {"name": "automated-ready"},
        {"name": "needs-design"},
    ]
    # Record pr_comment calls (the base fake's pr_comment is a no-op).
    pr_comments: list[tuple[int, str]] = []

    def _record_pr_comment(number: int, body_file: Path) -> None:
        pr_comments.append((number, Path(body_file).read_text(encoding="utf-8")))

    fake_gh.pr_comment = _record_pr_comment  # type: ignore[method-assign]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="ok", verdict_provenance="fresh_llm_review")

    app.merge_ready(456, merge=True)
    human_merge_comments = [c for c in pr_comments if "human merge" in c[1]]
    assert len(human_merge_comments) == 1

    app.merge_ready(456, merge=True)
    # The comment is posted once, not once per pass.
    human_merge_comments = [c for c in pr_comments if "human merge" in c[1]]
    assert len(human_merge_comments) == 1


def test_merge_ready_human_merge_label_removed_restores_merge(
    tmp_path: Path,
) -> None:
    """Removing the human-merge label after a hand-off restores normal
    queue/merge behaviour on the next pass — no ``charlie unescalate``
    required."""
    config = _human_merge_config(mergequeue=True)
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.issues[0]["labels"] = [
        {"name": "automated-ready"},
        {"name": "needs-design"},
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="ok", verdict_provenance="fresh_llm_review")

    # First pass: hand-off.
    first = app.merge_ready(456, merge=True)
    assert first.data["human_merge_hold"] is True
    assert fake_gh.merged == []

    # Remove the label.
    fake_gh.issues[0]["labels"] = [{"name": "automated-ready"}]

    # Second pass: normal mergequeue path resumes.
    second = app.merge_ready(456, merge=True)
    assert second.data["human_merge_hold"] is False
    assert second.data["mergequeue_label_applied"] is True
    # The policy escalation was cleared.
    state = load_state(paths.state_file)
    issue_entry = state["issues"]["123"]
    assert issue_entry["status"] != "escalated"


def test_merge_ready_default_empty_human_merge_labels_preserves_behavior(
    tmp_path: Path,
) -> None:
    """Default empty human_merge_labels preserves existing merge behaviour
    even when the issue carries a label that would be a human-merge label
    if configured."""
    config = OrchestratorConfig(auto_merge=_approved_automerge())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.issues[0]["labels"] = [
        {"name": "automated-ready"},
        {"name": "needs-design"},
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="ok", verdict_provenance="fresh_llm_review")

    result = app.merge_ready(456, merge=True)

    assert result.data["can_merge"] is True
    assert result.data["human_merge_hold"] is False
    # Default config has no mergequeue label, so the direct-merge path runs.
    assert result.data["merged"] is True
    assert fake_gh.merged == [(456, config.auto_merge.strategy)]


def test_merge_ready_human_merge_live_label_change_between_passes(
    tmp_path: Path,
) -> None:
    """Adding the label between passes changes the next pass's outcome
    (live-label decision, not cached state)."""
    config = _human_merge_config(mergequeue=True)
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.issues[0]["labels"] = [{"name": "automated-ready"}]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="ok", verdict_provenance="fresh_llm_review")

    # First pass: no label -> normal mergequeue handoff.
    first = app.merge_ready(456, merge=True)
    assert first.data["human_merge_hold"] is False
    assert first.data["mergequeue_label_applied"] is True

    # Reset the fake's recorded label add so the second pass is clean.
    fake_gh.pr_labels_added.clear()
    # Move the PR out of mergequeue status so it's a candidate again.
    state = load_state(paths.state_file)
    state["prs"]["456"]["status"] = "open"
    from charlie_work.state import save_state

    save_state(paths.state_file, state)

    # Add the label between passes.
    fake_gh.issues[0]["labels"] = [
        {"name": "automated-ready"},
        {"name": "needs-design"},
    ]

    # Second pass: label present -> hand-off, no mergequeue label.
    second = app.merge_ready(456, merge=True)
    assert second.data["human_merge_hold"] is True
    assert second.data["mergequeue_label_applied"] is None


# ---------------------------------------------------------------------------
# merge_ready — fail-closed human_merge_check_unavailable branch
# ---------------------------------------------------------------------------


def test_merge_ready_human_merge_check_unavailable_issue_view_raises(
    tmp_path: Path,
) -> None:
    """When ``issue_view`` raises while checking the human-merge label, the
    fleet fails closed: the PR is neither merged nor queued, and the
    unavailable flag is surfaced in the result. This is the safety/policy
    enforcement branch that had zero coverage before.
    """
    config = _human_merge_config(mergequeue=True)
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.issues[0]["labels"] = [
        {"name": "automated-ready"},
        {"name": "needs-design"},
    ]
    # Force the bound-issue fetch to fail so the human-merge check cannot
    # determine whether the label is present.
    _raise = GitHubError("simulated gh outage")

    def _failing_issue_view(number: int):
        raise _raise

    fake_gh.issue_view = _failing_issue_view  # type: ignore[method-assign]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="ok", verdict_provenance="fresh_llm_review")

    result = app.merge_ready(456, merge=True)

    assert result.data["can_merge"] is True
    assert result.data["human_merge_hold"] is False
    assert result.data["human_merge_check_unavailable"] is True
    # Fail closed: no merge, no mergequeue label.
    assert fake_gh.merged == []
    assert fake_gh.pr_labels_added == []
    assert result.data["mergequeue_label_applied"] is None


def test_merge_ready_human_merge_check_unavailable_malformed_issue(
    tmp_path: Path,
) -> None:
    """When ``issue_view`` returns a payload missing ``labels`` (malformed
    dict), the human-merge check fails closed the same way as a raise — the
    PR is not merged. Covers the second arm of the unavailable branch.
    """
    config = _human_merge_config(mergequeue=True)
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()

    # Return a dict without a 'labels' key to trip the malformed-payload arm.
    def _malformed_issue_view(number: int):
        return {"number": number, "title": "no labels here"}

    fake_gh.issue_view = _malformed_issue_view  # type: ignore[method-assign]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="ok", verdict_provenance="fresh_llm_review")

    result = app.merge_ready(456, merge=True)

    assert result.data["human_merge_check_unavailable"] is True
    assert fake_gh.merged == []
    assert fake_gh.pr_labels_added == []


# ---------------------------------------------------------------------------
# _merge_ready_dry_run — human-merge-labels mirror
# ---------------------------------------------------------------------------


def test_merge_ready_dry_run_human_merge_label_hold(tmp_path: Path) -> None:
    """The dry-run preview mirrors the real path: when the bound issue
    carries a configured human-merge label, it reports ``human_merge_hold``
    and a 'would not auto-merge' message instead of 'would merge'."""
    config = _human_merge_config(mergequeue=True)
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.issues[0]["labels"] = [
        {"name": "automated-ready"},
        {"name": "needs-design"},
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="ok", verdict_provenance="fresh_llm_review")

    result = app._merge_ready_dry_run(456, merge=True)

    assert result.data["human_merge_hold"] is True
    assert result.data["human_merge_check_unavailable"] is False
    assert "would not auto-merge" in result.message
    # Dry-run never merges.
    assert result.data["merged"] is False
    assert fake_gh.merged == []


def test_merge_ready_dry_run_human_merge_check_unavailable(tmp_path: Path) -> None:
    """The dry-run preview fails closed when the human-merge label check is
    unavailable (issue_view raises): ``ok`` is False and the unavailable
    flag is surfaced."""
    config = _human_merge_config(mergequeue=True)
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()

    def _failing_issue_view(number: int):
        raise GitHubError("simulated gh outage")

    fake_gh.issue_view = _failing_issue_view  # type: ignore[method-assign]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="ok", verdict_provenance="fresh_llm_review")

    result = app._merge_ready_dry_run(456, merge=True)

    assert result.data["human_merge_hold"] is False
    assert result.data["human_merge_check_unavailable"] is True
    # ok is False because a check was unavailable (fail-closed preview).
    assert result.ok is False
    assert "human-merge label check unavailable" in result.message


def test_merge_ready_dry_run_default_empty_preserves_behavior(
    tmp_path: Path,
) -> None:
    """Default empty ``human_merge_labels`` preserves the dry-run preview:
    no hold, no unavailable flag, even when the issue carries a label that
    would be a human-merge label if configured."""
    config = OrchestratorConfig(auto_merge=_approved_automerge())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.issues[0]["labels"] = [
        {"name": "automated-ready"},
        {"name": "needs-design"},
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="ok", verdict_provenance="fresh_llm_review")

    result = app._merge_ready_dry_run(456, merge=True)

    assert result.data["human_merge_hold"] is False
    assert result.data["human_merge_check_unavailable"] is False


# ---------------------------------------------------------------------------
# review() — issue-label-fetch glue (end-to-end)
# ---------------------------------------------------------------------------


def test_review_issue_label_fetch_glue_passes_live_labels_to_janitor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``review()`` fetches the bound issue's live labels via
    ``issue_view`` and passes them to ``run_janitor`` as the
    ``issue_labels`` kwarg. This exercises the glue end-to-end (the live
    fetch + hand-off), rather than only unit-testing ``run_janitor`` with a
    hand-built ``issue_labels`` kwarg.

    The PR is marked draft so the janitor verdict blocks (``ok=False``) and
    ``review()`` returns at the janitor gate — keeping the test bounded
    while still running the real ``run_janitor`` with the glue-fetched
    labels. The human-merge warning is informational and surfaces
    regardless of ``verdict.ok``.
    """
    config = _human_merge_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.issues[0]["labels"] = [
        {"name": "automated-ready"},
        {"name": "needs-design"},
    ]
    # A draft PR makes the janitor block (is_draft_only_block) so review()
    # returns at the gate without dispatching a reviewer, while the
    # human-merge warning still surfaces in the verdict.
    fake_gh.prs[0]["isDraft"] = True
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    captured: dict[str, object] = {}
    real_run_janitor = run_janitor

    def _spy_run_janitor(pr, checks, cfg, **kwargs):
        captured["issue_labels"] = kwargs.get("issue_labels")
        verdict = real_run_janitor(pr, checks, cfg, **kwargs)
        captured["warnings"] = list(verdict.warnings)
        return verdict

    monkeypatch.setattr("charlie_work.workflow.run_janitor", _spy_run_janitor)
    app.review(456)

    # The glue fetched the live issue labels and passed them through.
    assert captured["issue_labels"] is not None
    assert "needs-design" in captured["issue_labels"]  # type: ignore[operator]
    # The labels flowed through to the janitor's human-merge warning.
    assert any("human-merge" in w for w in captured["warnings"])  # type: ignore[arg-type]


def test_review_issue_label_fetch_glue_skipped_when_unconfigured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``human_merge_labels`` is empty (default), ``review()`` skips
    the issue-label fetch entirely and passes ``issue_labels=None`` to the
    janitor — zero overhead, matching the merge-path guard."""
    config = OrchestratorConfig(auto_merge=_approved_automerge())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.issues[0]["labels"] = [
        {"name": "automated-ready"},
        {"name": "needs-design"},
    ]
    fake_gh.prs[0]["isDraft"] = True
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    captured: dict[str, object] = {}
    real_run_janitor = run_janitor

    def _spy_run_janitor(pr, checks, cfg, **kwargs):
        captured["issue_labels"] = kwargs.get("issue_labels")
        return real_run_janitor(pr, checks, cfg, **kwargs)

    monkeypatch.setattr("charlie_work.workflow.run_janitor", _spy_run_janitor)
    app.review(456)

    assert captured["issue_labels"] is None


# ---------------------------------------------------------------------------
# Janitor
# ---------------------------------------------------------------------------


def test_janitor_human_merge_label_warning_not_failure() -> None:
    """The janitor surfaces a warning (not a failure) when the bound issue
    carries a configured human-merge label."""
    config = _human_merge_config()
    pr = {
        "number": 456,
        "title": "Fix #123: search",
        "headRefName": "agent/issue-123-fix-search",
        "baseRefName": "main",
        "body": "Closes #123\n\nTests: added.",
        "isCrossRepository": False,
        "state": "OPEN",
        "isDraft": False,
        "mergeable": True,
        "additions": 10,
        "deletions": 2,
        "mergeStateStatus": "CLEAN",
    }
    checks = [
        {"name": "Tests passed", "state": "SUCCESS"},
        {"name": "Lint & Format", "bucket": "pass"},
        {"name": "Pre-commit", "state": "SUCCESS"},
    ]
    verdict = run_janitor(
        pr,
        checks,
        config,
        issue_labels={"automated-ready", "needs-design"},
    )
    assert verdict.ok is True
    assert any("human-merge" in w for w in verdict.warnings)


def test_janitor_human_merge_label_no_warning_when_label_absent() -> None:
    """No warning when the issue does not carry a configured human-merge
    label."""
    config = _human_merge_config()
    pr = {
        "number": 456,
        "title": "Fix #123: search",
        "headRefName": "agent/issue-123-fix-search",
        "baseRefName": "main",
        "body": "Closes #123\n\nTests: added.",
        "isCrossRepository": False,
        "state": "OPEN",
        "isDraft": False,
        "mergeable": True,
        "additions": 10,
        "deletions": 2,
        "mergeStateStatus": "CLEAN",
    }
    checks = [
        {"name": "Tests passed", "state": "SUCCESS"},
        {"name": "Lint & Format", "bucket": "pass"},
        {"name": "Pre-commit", "state": "SUCCESS"},
    ]
    verdict = run_janitor(
        pr,
        checks,
        config,
        issue_labels={"automated-ready"},
    )
    assert not any("human-merge" in w for w in verdict.warnings)


def test_janitor_human_merge_label_no_warning_when_unconfigured() -> None:
    """No warning when human_merge_labels is empty (default)."""
    config = OrchestratorConfig()
    pr = {
        "number": 456,
        "title": "Fix #123: search",
        "headRefName": "agent/issue-123-fix-search",
        "baseRefName": "main",
        "body": "Closes #123\n\nTests: added.",
        "isCrossRepository": False,
        "state": "OPEN",
        "isDraft": False,
        "mergeable": True,
        "additions": 10,
        "deletions": 2,
        "mergeStateStatus": "CLEAN",
    }
    checks = [
        {"name": "Tests passed", "state": "SUCCESS"},
        {"name": "Lint & Format", "bucket": "pass"},
        {"name": "Pre-commit", "state": "SUCCESS"},
    ]
    verdict = run_janitor(
        pr,
        checks,
        config,
        issue_labels={"automated-ready", "needs-design"},
    )
    assert not any("human-merge" in w for w in verdict.warnings)


def test_janitor_human_merge_label_no_warning_when_issue_labels_none() -> None:
    """No warning when issue_labels is None (caller did not fetch)."""
    config = _human_merge_config()
    pr = {
        "number": 456,
        "title": "Fix #123: search",
        "headRefName": "agent/issue-123-fix-search",
        "baseRefName": "main",
        "body": "Closes #123\n\nTests: added.",
        "isCrossRepository": False,
        "state": "OPEN",
        "isDraft": False,
        "mergeable": True,
        "additions": 10,
        "deletions": 2,
        "mergeStateStatus": "CLEAN",
    }
    checks = [
        {"name": "Tests passed", "state": "SUCCESS"},
        {"name": "Lint & Format", "bucket": "pass"},
        {"name": "Pre-commit", "state": "SUCCESS"},
    ]
    verdict = run_janitor(pr, checks, config, issue_labels=None)
    assert not any("human-merge" in w for w in verdict.warnings)


# ---------------------------------------------------------------------------
# Example configs
# ---------------------------------------------------------------------------


def test_example_configs_load_with_human_merge_labels_comment() -> None:
    """The shipped example configs document human_merge_labels (commented
    out) and still load cleanly with the default empty value."""
    for name in (
        "orchestrator.config.devin.yaml",
        "orchestrator.config.claude-code.yaml",
    ):
        cfg = load_config(EXAMPLES_DIR / name)
        assert cfg.dispatch.human_merge_labels == ()
