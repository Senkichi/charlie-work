"""Tests for issue #592: a cached review packet must be regenerated when the
prompt template changes, not only when the PR head advances.

Without this, a template fix never reaches a static-head PR -- the orchestrator
keeps handing reviewers a packet rendered from the old template, indefinitely,
until every dispatch attempt burns out and the PR escalates. The fix stamps a
SHA-256 digest of the resolved template + referenced section partials into
``pr.json`` at render time and treats a digest mismatch as packet staleness
alongside a head-SHA mismatch.
"""

from __future__ import annotations

import json
from pathlib import Path

from charlie_work.config import CrossFamilyConfig, OrchestratorConfig
from charlie_work.paths import runtime_paths
from charlie_work.prompts import prompt_template_digest

from charlie_work.workflow import OrchestratorApp

# Reuse the shared FakeGitHub whose default PR #456 is janitor-green.
from test_charlie_work import FakeGitHub, _make_loop_app


def _write(directory: Path, name: str, text: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


def _pr_dir(tmp_path: Path, pr_number: int) -> Path:
    return tmp_path / ".var" / "charlie-work" / "prs" / f"pr-{pr_number}"


def _plant_packet(
    tmp_path: Path,
    pr_number: int,
    *,
    head_sha: str,
    template_sha: str | None,
    decision: dict | None = None,
) -> Path:
    """Plant a review packet fixture with an optional stamped template digest."""
    pr_dir = _pr_dir(tmp_path, pr_number)
    pr_dir.mkdir(parents=True, exist_ok=True)
    pr_json: dict = {"number": pr_number, "headRefOid": head_sha}
    if template_sha is not None:
        pr_json["prompt_template_sha"] = template_sha
    (pr_dir / "pr.json").write_text(json.dumps(pr_json), encoding="utf-8")
    (pr_dir / "review-prompt.md").write_text(
        f"review prompt for PR #{pr_number}", encoding="utf-8"
    )
    if decision is not None:
        (pr_dir / "review-decision.json").write_text(json.dumps(decision), encoding="utf-8")
    return pr_dir


# ---------------------------------------------------------------------------
# prompt_template_digest unit tests
# ---------------------------------------------------------------------------


def test_digest_changes_when_template_text_changes(tmp_path: Path) -> None:
    """A template edit must change the digest -- this is the staleness signal."""
    _write(tmp_path, "review.md", "# Review PR #$pr_number\n")
    digest_a = prompt_template_digest("review.md", search_dirs=(tmp_path,))
    _write(tmp_path, "review.md", "# Review PR #$pr_number (revised)\n")
    digest_b = prompt_template_digest("review.md", search_dirs=(tmp_path,))
    assert digest_a != digest_b
    assert len(digest_a) == 64  # SHA-256 hex


def test_digest_changes_when_referenced_section_partial_changes(tmp_path: Path) -> None:
    """A section partial the template references is a load-bearing input."""
    _write(tmp_path / "worker_sections", "scope_contract.md", "Scope for $branch_name.\n")
    _write(tmp_path, "review.md", "# Review #$pr_number\n\n$section_scope_contract\n")
    digest_a = prompt_template_digest("review.md", search_dirs=(tmp_path,))
    _write(tmp_path / "worker_sections", "scope_contract.md", "Scope for $branch_name (v2).\n")
    digest_b = prompt_template_digest("review.md", search_dirs=(tmp_path,))
    assert digest_a != digest_b


def test_digest_ignores_unreferenced_section_partial(tmp_path: Path) -> None:
    """An unused partial never reaches the rendered output, so editing it must
    not invalidate packets -- otherwise one stale unused file breaks every
    packet at once."""
    _write(tmp_path / "worker_sections", "unused.md", "Needs $long_gone_variable.\n")
    _write(tmp_path, "review.md", "# Review #$pr_number\n")
    digest_a = prompt_template_digest("review.md", search_dirs=(tmp_path,))
    _write(tmp_path / "worker_sections", "unused.md", "Completely different text now.\n")
    digest_b = prompt_template_digest("review.md", search_dirs=(tmp_path,))
    assert digest_a == digest_b


def test_digest_is_deterministic(tmp_path: Path) -> None:
    """Same inputs -> same digest, regardless of call order."""
    _write(tmp_path / "worker_sections", "scope_contract.md", "Scope.\n")
    _write(tmp_path, "review.md", "# Review #$pr_number\n\n$section_scope_contract\n")
    a = prompt_template_digest("review.md", search_dirs=(tmp_path,))
    b = prompt_template_digest("review.md", search_dirs=(tmp_path,))
    assert a == b


def test_digest_repo_local_override_wins_over_package_default(tmp_path: Path) -> None:
    """A repo-local override shadows the packaged template, and the digest
    reflects the override's text, not the package's."""
    package_digest = prompt_template_digest("review.md")
    _write(tmp_path, "review.md", "# Override #$pr_number\n")
    override_digest = prompt_template_digest("review.md", search_dirs=(tmp_path,))
    assert override_digest != package_digest


# ---------------------------------------------------------------------------
# review() stamps the digest into pr.json
# ---------------------------------------------------------------------------


def test_review_stamps_prompt_template_sha_into_pr_json(tmp_path: Path) -> None:
    """review() must record the current template digest in pr.json so a future
    template edit is detectable as staleness."""
    config = OrchestratorConfig(cross_family=CrossFamilyConfig(enabled=False))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub())

    result = app.review(456)
    assert result.ok is True

    pr_json_path = paths.prs / "pr-456" / "pr.json"
    pr_json = json.loads(pr_json_path.read_text(encoding="utf-8"))
    assert "prompt_template_sha" in pr_json
    # The stamped digest must match what _review_template_sha computes from the
    # current template (the package default, since no prompts_dir is configured).
    assert pr_json["prompt_template_sha"] == prompt_template_digest("review.md")


# ---------------------------------------------------------------------------
# loop() same-head skip now considers template staleness
# ---------------------------------------------------------------------------


def _pr456(head_sha: str) -> dict:
    return {
        "number": 456,
        "title": "Fix #123: search",
        "url": "https://example.test/pull/456",
        "headRefName": "agent/issue-123-fix-search",
        "headRefOid": head_sha,
        "body": "Closes #123",
        "labels": [],
        "isCrossRepository": False,
    }


def test_loop_regenerates_same_head_packet_when_template_stale(tmp_path: Path) -> None:
    """A static-head PR whose packet was rendered from an old template must be
    regenerated -- the core defect from issue #592."""
    pr = _pr456("sha-same")
    app, _ = _make_loop_app(tmp_path, prs=[pr])
    current_sha = app._review_template_sha()

    # Same head, but the packet was rendered from a different template.
    _plant_packet(tmp_path, 456, head_sha="sha-same", template_sha="stale-digest")

    review_calls: list[int] = []
    original_review = app.review

    def tracking_review(pr_number: int) -> object:
        review_calls.append(pr_number)
        return original_review(pr_number)

    app.review = tracking_review  # type: ignore[method-assign]
    result = app.loop(limit=0)

    # review() was invoked (regeneration fired)...
    assert 456 in review_calls
    # ...and the skip counter did NOT consume it.
    assert result.data["skipped_reviews"] == 0
    # A distinct event was emitted so a fleet-wide template edit is visible.
    events = _events_of_kind(paths_from_app(app), "review_packet_template_stale")
    assert any(e.get("pr_number") == 456 for e in events)
    # Sanity: the stale digest really was different from the current one.
    assert "stale-digest" != current_sha


def test_loop_skips_same_head_packet_when_template_matches(tmp_path: Path) -> None:
    """Same head AND current template -> skip regeneration (the deliberate
    idempotence guard is preserved)."""
    pr = _pr456("sha-same")
    app, _ = _make_loop_app(tmp_path, prs=[pr])
    current_sha = app._review_template_sha()

    _plant_packet(tmp_path, 456, head_sha="sha-same", template_sha=current_sha)

    review_calls: list[int] = []
    original_review = app.review

    def tracking_review(pr_number: int) -> object:
        review_calls.append(pr_number)
        return original_review(pr_number)

    app.review = tracking_review  # type: ignore[method-assign]
    result = app.loop(limit=0)

    assert 456 not in review_calls
    assert result.data["skipped_reviews"] == 1


def test_loop_treats_legacy_packet_without_digest_as_current(tmp_path: Path) -> None:
    """A packet predating issue #592 (no prompt_template_sha) is treated as
    current so the upgrade does not force a one-time fleet-wide regeneration
    burst. Packets rendered after this fix always carry the digest, so future
    template edits are still caught."""
    pr = _pr456("sha-same")
    app, _ = _make_loop_app(tmp_path, prs=[pr])

    _plant_packet(tmp_path, 456, head_sha="sha-same", template_sha=None)

    review_calls: list[int] = []
    original_review = app.review

    def tracking_review(pr_number: int) -> object:
        review_calls.append(pr_number)
        return original_review(pr_number)

    app.review = tracking_review  # type: ignore[method-assign]
    result = app.loop(limit=0)

    assert 456 not in review_calls
    assert result.data["skipped_reviews"] == 1


# ---------------------------------------------------------------------------
# review_queue() must not dispatch a template-stale packet
# ---------------------------------------------------------------------------


def test_review_queue_skips_template_stale_pending_packet(tmp_path: Path) -> None:
    """A pending packet whose template is stale must be regenerated by loop()
    before it is dispatched, not handed to a reviewer as-is."""
    pr = _pr456("sha-same")
    app, _ = _make_loop_app(tmp_path, prs=[pr])
    current_sha = app._review_template_sha()

    _plant_packet(
        tmp_path,
        456,
        head_sha="sha-same",
        template_sha="stale-digest",
        decision={"decision": "pending", "reviewed_head_sha": None},
    )

    result = app.review_queue()
    queue = result.data.get("queue", [])
    assert not any(entry["pr"] == 456 for entry in queue)
    assert "stale-digest" != current_sha


def test_review_queue_skips_template_stale_stale_decision_packet(tmp_path: Path) -> None:
    """A terminal decision on an old head with a same-head-but-template-stale
    packet must not be queued as 'stale' either -- loop() regenerates first."""
    pr = _pr456("sha-same")
    app, _ = _make_loop_app(tmp_path, prs=[pr])

    _plant_packet(
        tmp_path,
        456,
        head_sha="sha-same",
        template_sha="stale-digest",
        decision={"decision": "approved", "reviewed_head_sha": "sha-old"},
    )

    result = app.review_queue()
    queue = result.data.get("queue", [])
    assert not any(entry["pr"] == 456 for entry in queue)


def test_review_queue_skips_template_stale_vacuous_decision_packet(tmp_path: Path) -> None:
    """Issue #784 AC-8's "vacuous" branch (a content-free request_changes
    verdict at the live head) must respect template staleness the same way
    the "stale" and "pending"/"missing"/"invalid" branches do.

    dispatch_reviews()'s normal (non-cross-family) dispatch path does not
    filter its candidates by decision value -- _is_review_dispatchable
    ignores the candidate payload -- so an un-gated "vacuous" queue entry
    could still reach a fresh reviewer as a packet rendered from the old
    template: the exact defect this issue fixed for its sibling branches.
    """
    pr = _pr456("sha-same")
    app, _ = _make_loop_app(tmp_path, prs=[pr])

    _plant_packet(
        tmp_path,
        456,
        head_sha="sha-same",
        template_sha="stale-digest",
        decision={
            "decision": "request_changes",
            "reviewed_head_sha": "sha-same",
            "required_changes": [],
            "summary": "",
        },
    )

    result = app.review_queue()
    queue = result.data.get("queue", [])
    assert not any(entry["pr"] == 456 for entry in queue)


def test_review_queue_includes_vacuous_packet_when_template_matches(tmp_path: Path) -> None:
    """The template check must not over-prune the vacuous path either: a
    content-free verdict with a current-template packet is still queued as
    "vacuous" so _record_cross_family_verdicts can act on it."""
    pr = _pr456("sha-same")
    app, _ = _make_loop_app(tmp_path, prs=[pr])
    current_sha = app._review_template_sha()

    _plant_packet(
        tmp_path,
        456,
        head_sha="sha-same",
        template_sha=current_sha,
        decision={
            "decision": "request_changes",
            "reviewed_head_sha": "sha-same",
            "required_changes": [],
            "summary": "",
        },
    )

    result = app.review_queue()
    queue = result.data.get("queue", [])
    assert any(entry["pr"] == 456 and entry["decision"] == "vacuous" for entry in queue)


def test_review_queue_includes_pending_packet_when_template_matches(tmp_path: Path) -> None:
    """The template check must not over-prune: a current packet is still queued."""
    pr = _pr456("sha-same")
    app, _ = _make_loop_app(tmp_path, prs=[pr])
    current_sha = app._review_template_sha()

    _plant_packet(
        tmp_path,
        456,
        head_sha="sha-same",
        template_sha=current_sha,
        decision={"decision": "pending", "reviewed_head_sha": None},
    )

    result = app.review_queue()
    queue = result.data.get("queue", [])
    assert any(entry["pr"] == 456 for entry in queue)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def paths_from_app(app: OrchestratorApp) -> Path:
    return app.paths.state_file


def _events_of_kind(state_path: Path, kind: str) -> list[dict]:
    from charlie_work.instrumentation import query_events

    return query_events(state_path, kind=kind)
