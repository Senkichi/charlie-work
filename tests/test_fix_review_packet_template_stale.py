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

from charlie_work.config import OrchestratorConfig
from charlie_work.paths import runtime_paths
from charlie_work.prompts import prompt_template_digest

from charlie_work.workflow import OrchestratorApp

# Reuse the shared FakeGitHub whose default PR #456 is janitor-green.
from _fakes_github import FakeGitHub
from _review_fixtures import _make_loop_app


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
    config = OrchestratorConfig()
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


# ---------------------------------------------------------------------------
# Issue #1338: escalated PRs must not re-fire review_packet_template_stale
# every pass. Escalation is terminal -- review() early-returns before the
# regen path, so the staleness WARNING would fire identically every pass
# without ever converging. The recovery procedure (unescalate +
# why-charlie-hate) regenerates the packet with the current template, so the
# staleness WARNING and the cross-family regen-budget charge are suppressed
# while escalated. self.review() is still called -- its own _escalation_flags
# entry gate no-ops packet regen/label transitions, but it is the only
# per-pass path that refreshes janitor_ok/janitor_failures and runs the #776
# remediation for judgment-class escalations (PRs #1397/#1443).
# ---------------------------------------------------------------------------


def _seed_pr_escalated(app: OrchestratorApp, pr_number: int, issue_number: int) -> None:
    """Mark a PR and its linked issue as escalated in state.json."""
    from charlie_work.state import load_state, save_state, state_lock

    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"][str(pr_number)] = {
            "number": pr_number,
            "issue_number": issue_number,
            "status": "escalated",
            "escalation_reason": "test_escalation",
        }
        state["issues"][str(issue_number)] = {
            "number": issue_number,
            "status": "escalated",
            "escalation_reason": "test_escalation",
        }
        save_state(app.paths.state_file, state)


def _seed_issue_escalated(app: OrchestratorApp, issue_number: int) -> None:
    """Mark only the linked issue as escalated (PR status left non-escalated)."""
    from charlie_work.state import load_state, save_state, state_lock

    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["issues"][str(issue_number)] = {
            "number": issue_number,
            "status": "escalated",
            "escalation_reason": "test_escalation",
        }
        save_state(app.paths.state_file, state)


def test_loop_skips_template_stale_warning_for_escalated_pr(tmp_path: Path) -> None:
    """An escalated PR with a stale-template packet must NOT emit
    ``review_packet_template_stale`` on every pass -- the regen is
    unreachable (review() early-returns "escalated; review skipped") and the
    WARNING would spam events.db identically every pass without converging
    (issue #1338). The recovery procedure regenerates the packet, so the
    staleness WARNING is suppressed while escalated.

    self.review() is still called each pass -- its own _escalation_flags
    entry gate no-ops packet regen/label transitions, but it refreshes
    janitor diagnostics and runs the #776 remediation lane. Skipping it
    entirely would freeze janitor_ok/janitor_failures (PRs #1397/#1443)."""
    pr = _pr456("sha-same")
    app, _ = _make_loop_app(tmp_path, prs=[pr])
    current_sha = app._review_template_sha()

    _plant_packet(tmp_path, 456, head_sha="sha-same", template_sha="stale-digest")
    _seed_pr_escalated(app, pr_number=456, issue_number=123)
    assert "stale-digest" != current_sha

    review_calls: list[int] = []
    original_review = app.review

    def tracking_review(pr_number: int) -> object:
        review_calls.append(pr_number)
        return original_review(pr_number)

    app.review = tracking_review  # type: ignore[method-assign]

    # Run two passes -- the bug fired the WARNING every pass.
    app.loop(limit=0)
    app.loop(limit=0)

    # review() WAS invoked each pass -- it no-ops packet regen via its own
    # escalation gate but still refreshes janitor diagnostics / the #776 lane.
    assert review_calls.count(456) == 2
    # No staleness WARNING was emitted -- escalation is terminal and the
    # recovery procedure handles regen, so the WARNING is suppressed while
    # escalated.
    events = _events_of_kind(paths_from_app(app), "review_packet_template_stale")
    assert not any(e.get("pr_number") == 456 for e in events)


def test_loop_skips_template_stale_warning_for_issue_escalated(tmp_path: Path) -> None:
    """A PR whose linked ISSUE is escalated (even if the PR's own status is
    not "escalated") must likewise suppress the staleness WARNING --
    review()'s entry gate checks both pr_escalated and issue_escalated, so
    the regen is unreachable here too and the WARNING would spam identically.
    self.review() is still called each pass for the same reason as the
    pr-escalated case (janitor diagnostics + #776 lane)."""
    pr = _pr456("sha-same")
    app, _ = _make_loop_app(tmp_path, prs=[pr])
    current_sha = app._review_template_sha()

    _plant_packet(tmp_path, 456, head_sha="sha-same", template_sha="stale-digest")
    _seed_issue_escalated(app, issue_number=123)
    assert "stale-digest" != current_sha

    review_calls: list[int] = []
    original_review = app.review

    def tracking_review(pr_number: int) -> object:
        review_calls.append(pr_number)
        return original_review(pr_number)

    app.review = tracking_review  # type: ignore[method-assign]

    app.loop(limit=0)
    app.loop(limit=0)

    assert review_calls.count(456) == 2
    events = _events_of_kind(paths_from_app(app), "review_packet_template_stale")
    assert not any(e.get("pr_number") == 456 for e in events)


def test_loop_regenerates_template_stale_after_unescalate(tmp_path: Path) -> None:
    """Once the PR is de-escalated, the staleness WARNING and regen must
    resume -- the #592 behavior is preserved for non-escalated PRs. This
    proves the #1338 suppression is scoped to the escalated window only, not
    a permanent suppression. While escalated, review() is still called (for
    janitor diagnostics); after unescalate, the staleness WARNING fires too."""
    pr = _pr456("sha-same")
    app, _ = _make_loop_app(tmp_path, prs=[pr])
    current_sha = app._review_template_sha()

    _plant_packet(tmp_path, 456, head_sha="sha-same", template_sha="stale-digest")
    _seed_pr_escalated(app, pr_number=456, issue_number=123)
    assert "stale-digest" != current_sha

    # Pass 1 while escalated: no warning, but review() IS called (janitor
    # diagnostics refresh; packet regen no-ops via review()'s own gate).
    review_calls: list[int] = []
    original_review = app.review

    def tracking_review(pr_number: int) -> object:
        review_calls.append(pr_number)
        return original_review(pr_number)

    app.review = tracking_review  # type: ignore[method-assign]
    app.loop(limit=0)
    assert 456 in review_calls
    assert not any(
        e.get("pr_number") == 456
        for e in _events_of_kind(paths_from_app(app), "review_packet_template_stale")
    )

    # De-escalate: clear the escalated status so the staleness check resumes.
    from charlie_work.state import load_state, save_state, state_lock

    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"]["456"]["status"] = "reviewing"
        state["issues"]["123"]["status"] = "reviewing"
        save_state(app.paths.state_file, state)

    # Pass 2 after unescalate: review() is invoked again and the staleness
    # event fires exactly as today (#592 preserved).
    app.loop(limit=0)
    assert review_calls.count(456) == 2
    events = _events_of_kind(paths_from_app(app), "review_packet_template_stale")
    assert any(e.get("pr_number") == 456 for e in events)


def _make_loop_app_with_required_checks(
    tmp_path: Path, *, prs: list[dict], required_checks: tuple[str, ...]
) -> tuple[OrchestratorApp, FakeGitHub]:
    """Build a loop() app whose janitor gate enforces ``required_checks``.

    Mirrors ``_make_loop_app`` but configures required checks so the janitor
    can actually fail (and then heal) on them -- the default
    ``_approved_automerge`` leaves ``required_checks=()`` and the janitor is
    vacuously green, which cannot exercise the janitor-diagnostics refresh
    path the #1338 rework needs to prove.
    """
    from charlie_work.config import AutoMergeConfig, ReviewConfig

    config = OrchestratorConfig(
        review=ReviewConfig(require_tests_or_rationale=False),
        auto_merge=AutoMergeConfig(required_checks=required_checks, require_approved_review=True),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    for pr in prs:
        pr.setdefault("state", "OPEN")
    fake_gh.prs = prs
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    return app, fake_gh


def test_loop_refreshes_janitor_diagnostics_for_escalated_pr_with_stale_packet(
    tmp_path: Path,
) -> None:
    """A judgment-class escalated PR with a stale packet must still have its
    janitor diagnostics (janitor_ok/janitor_failures) refreshed across passes
    -- self.review() is called each pass and its escalated branch recomputes
    run_janitor for visibility only (PRs #1397/#1443). The #1338 rework scopes
    the skip to the staleness WARNING + cross-family regen-budget charge, so
    review() -- the only per-pass path that refreshes these diagnostics --
    stays reachable.

    This is the behavior the round-1 review identified as missing: the
    original #1338 fix skipped self.review() entirely via ``continue``,
    freezing janitor_ok/janitor_failures at whatever value they held when the
    PR escalated. This test fails against that ``continue`` (review() is
    never called, so the diagnostics never update) and passes against the
    narrowed skip.
    """
    from charlie_work.state import load_state

    pr = _pr456("sha-same")
    app, fake_gh = _make_loop_app_with_required_checks(
        tmp_path, prs=[pr], required_checks=("Tests passed",)
    )
    current_sha = app._review_template_sha()

    _plant_packet(tmp_path, 456, head_sha="sha-same", template_sha="stale-digest")
    # Judgment-class escalation: the PR escalated from a review verdict, not
    # from a janitor-gate failure. The janitor gate is therefore an
    # independent, still-meaningful signal to refresh while escalated.
    _seed_pr_escalated(app, pr_number=456, issue_number=123)
    assert "stale-digest" != current_sha

    # Drive the janitor verdict across two states: a failed required check on
    # pass 1, then the same check green on pass 2. review()'s escalated
    # branch calls self.gh.pr_checks once per pass, so a counter-based fake
    # flips the verdict between passes.
    checks_sequence: list[list[dict]] = [
        [{"name": "Tests passed", "state": "FAILURE"}],
        [{"name": "Tests passed", "state": "SUCCESS"}],
    ]
    checks_calls: list[int] = []

    def fake_pr_checks(number: int) -> list[dict]:
        checks_calls.append(number)
        return checks_sequence[min(len(checks_calls) - 1, len(checks_sequence) - 1)]

    fake_gh.pr_checks = fake_pr_checks  # type: ignore[method-assign]

    review_calls: list[int] = []
    original_review = app.review

    def tracking_review(pr_number: int) -> object:
        review_calls.append(pr_number)
        return original_review(pr_number)

    app.review = tracking_review  # type: ignore[method-assign]

    # Pass 1: required check FAILED -> janitor_ok=False, failures non-empty.
    app.loop(limit=0)
    assert 456 in review_calls
    state_after_pass1 = load_state(app.paths.state_file)
    pr_state_1 = state_after_pass1["prs"]["456"]
    assert pr_state_1.get("janitor_ok") is False
    assert pr_state_1.get("janitor_failures")  # non-empty -> check failed

    # Pass 2: required check now SUCCESS -> janitor_ok=True, failures empty.
    # The diagnostics REFRESHED across passes -- they did not stay frozen at
    # the pass-1 failure, which is the regression the round-1 review flagged.
    app.loop(limit=0)
    assert review_calls.count(456) == 2
    state_after_pass2 = load_state(app.paths.state_file)
    pr_state_2 = state_after_pass2["prs"]["456"]
    assert pr_state_2.get("janitor_ok") is True
    assert pr_state_2.get("janitor_failures") == []

    # The #1338 staleness WARNING is still suppressed while escalated -- the
    # narrowed skip did not re-introduce the WARNING spam the original fix
    # addressed.
    events = _events_of_kind(paths_from_app(app), "review_packet_template_stale")
    assert not any(e.get("pr_number") == 456 for e in events)


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
# Issue #1894: the #1338 stale-template suppression names only "escalated",
# but the identical non-convergent shape is reachable through the other
# durably-stuck statuses -- a "blocked" record on the PR or its linked
# issue (SINK_STATUSES' other member), and a "janitor_blocked" PR whose
# sole janitor failure is a permanently-missing required check
# (is_missing_checks_only_block; review()'s deterministic janitor gate
# short-circuits before packet regen, so packet_template_sha can never
# catch up). review() is still called for these statuses: the
# janitor-diagnostics refresh, ci_run_never_created detection, and the
# stale-checks-retrigger self-heal lane all live inside review()'s main
# janitor-gate path, so skipping the call would freeze the predicate's own
# inputs -- the same #1397/#1443 frozen-diagnostics failure mode.
# ---------------------------------------------------------------------------


def _seed_pr_janitor_blocked_missing_checks(
    app: OrchestratorApp, pr_number: int, issue_number: int
) -> None:
    """Mark a PR janitor_blocked on a missing-checks-only janitor failure --
    the shape review()'s janitor-gate bookkeeping persists each pass."""
    from charlie_work.state import load_state, save_state, state_lock

    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"][str(pr_number)] = {
            "number": pr_number,
            "issue_number": issue_number,
            "status": "janitor_blocked",
            "janitor_ok": False,
            "janitor_failures": ["Required check(s) missing: Tests passed"],
            "is_missing_checks_only_block": True,
        }
        save_state(app.paths.state_file, state)


def _seed_blocked(app: OrchestratorApp, pr_number: int, issue_number: int) -> None:
    """Mark a PR and its linked issue as "blocked" in state.json -- the
    shape record_review's blocked path persists (the other SINK_STATUSES
    member, a judgment verdict parked on agent:human-needed)."""
    from charlie_work.state import load_state, save_state, state_lock

    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"][str(pr_number)] = {
            "number": pr_number,
            "issue_number": issue_number,
            "status": "blocked",
        }
        state["issues"][str(issue_number)] = {
            "number": issue_number,
            "status": "blocked",
        }
        save_state(app.paths.state_file, state)


def _seed_issue_blocked(app: OrchestratorApp, issue_number: int) -> None:
    """Mark only the linked issue as "blocked" (PR status left untouched)."""
    from charlie_work.state import load_state, save_state, state_lock

    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["issues"][str(issue_number)] = {
            "number": issue_number,
            "status": "blocked",
        }
        save_state(app.paths.state_file, state)


def test_loop_skips_template_stale_warning_for_janitor_blocked_missing_checks(
    tmp_path: Path,
) -> None:
    """A janitor_blocked PR whose sole janitor failure is a permanently-
    missing required check (is_missing_checks_only_block=True) can never
    reach packet regen -- review() short-circuits in the deterministic
    janitor gate -- so the stale-template WARNING must not re-fire every
    pass (issue #1894; the swole PR #298 shape). review() is still called
    each pass -- the janitor-diagnostics refresh and the
    ci_run_never_created / stale-checks-retrigger lanes live inside it."""
    pr = _pr456("sha-same")
    app, fake_gh = _make_loop_app_with_required_checks(
        tmp_path, prs=[pr], required_checks=("Tests passed",)
    )
    current_sha = app._review_template_sha()

    # The required check never reports -> the janitor keeps failing
    # missing-checks-only and refresh-stamps is_missing_checks_only_block.
    fake_gh.pr_checks = lambda _number: []  # type: ignore[method-assign]

    _plant_packet(tmp_path, 456, head_sha="sha-same", template_sha="stale-digest")
    _seed_pr_janitor_blocked_missing_checks(app, pr_number=456, issue_number=123)
    assert "stale-digest" != current_sha

    review_calls: list[int] = []
    original_review = app.review

    def tracking_review(pr_number: int) -> object:
        review_calls.append(pr_number)
        return original_review(pr_number)

    app.review = tracking_review  # type: ignore[method-assign]

    # Run two passes -- the bug fired the WARNING every pass.
    app.loop(limit=0)
    app.loop(limit=0)

    # review() WAS invoked each pass -- it re-runs the janitor gate (which
    # refreshes janitor_ok/janitor_failures/is_missing_checks_only_block
    # and owns the stale-checks retrigger lane), then short-circuits before
    # packet regen.
    assert review_calls.count(456) == 2
    # No staleness WARNING -- the regen is unreachable while the check stays
    # missing, so the WARNING would spam identically without converging.
    events = _events_of_kind(paths_from_app(app), "review_packet_template_stale")
    assert not any(e.get("pr_number") == 456 for e in events)


def test_loop_skips_template_stale_warning_for_blocked_pr(tmp_path: Path) -> None:
    """status="blocked" is the other SINK_STATUSES member -- a terminal
    judgment verdict parked on agent:human-needed (issue #1894's second
    half). The same non-convergent suppression applies: nothing on the loop
    side can move a blocked record, so a stale-template WARNING would fire
    identically every pass until a human acts."""
    pr = _pr456("sha-same")
    app, _ = _make_loop_app(tmp_path, prs=[pr])
    current_sha = app._review_template_sha()

    _plant_packet(tmp_path, 456, head_sha="sha-same", template_sha="stale-digest")
    _seed_blocked(app, pr_number=456, issue_number=123)
    assert "stale-digest" != current_sha

    review_calls: list[int] = []
    original_review = app.review

    def tracking_review(pr_number: int) -> object:
        review_calls.append(pr_number)
        return original_review(pr_number)

    app.review = tracking_review  # type: ignore[method-assign]

    app.loop(limit=0)
    app.loop(limit=0)

    assert review_calls.count(456) == 2
    events = _events_of_kind(paths_from_app(app), "review_packet_template_stale")
    assert not any(e.get("pr_number") == 456 for e in events)


def test_loop_skips_template_stale_warning_for_blocked_issue(tmp_path: Path) -> None:
    """The swole evidence shape: the linked ISSUE carries status="blocked"
    while the PR record is janitor_blocked on missing checks. Either
    record's sink status suffices for suppression -- a human owns the
    record either way."""
    pr = _pr456("sha-same")
    app, fake_gh = _make_loop_app_with_required_checks(
        tmp_path, prs=[pr], required_checks=("Tests passed",)
    )

    fake_gh.pr_checks = lambda _number: []  # type: ignore[method-assign]

    _plant_packet(tmp_path, 456, head_sha="sha-same", template_sha="stale-digest")
    _seed_pr_janitor_blocked_missing_checks(app, pr_number=456, issue_number=123)
    _seed_issue_blocked(app, issue_number=123)

    review_calls: list[int] = []
    original_review = app.review

    def tracking_review(pr_number: int) -> object:
        review_calls.append(pr_number)
        return original_review(pr_number)

    app.review = tracking_review  # type: ignore[method-assign]

    app.loop(limit=0)
    app.loop(limit=0)

    assert review_calls.count(456) == 2
    events = _events_of_kind(paths_from_app(app), "review_packet_template_stale")
    assert not any(e.get("pr_number") == 456 for e in events)


def test_loop_fires_template_stale_warning_for_janitor_blocked_without_flag(
    tmp_path: Path,
) -> None:
    """janitor_blocked alone is NOT enough for suppression -- a record that
    lacks is_missing_checks_only_block (a different janitor failure class,
    or a record persisted before the flag existed) still has live
    remediation lanes, so the WARNING keeps firing. The suppression is
    scoped to the non-convergent missing-checks-only shape only."""
    pr = _pr456("sha-same")
    app, fake_gh = _make_loop_app_with_required_checks(
        tmp_path, prs=[pr], required_checks=("Tests passed",)
    )
    current_sha = app._review_template_sha()

    fake_gh.pr_checks = lambda _number: []  # type: ignore[method-assign]

    _plant_packet(tmp_path, 456, head_sha="sha-same", template_sha="stale-digest")

    # Seed janitor_blocked WITHOUT the flag (e.g. a record persisted by an
    # older version): pass 1 must still fire the WARNING. review() then
    # refreshes the real missing-checks shape, so pass 2 is suppressed --
    # the flag flip across passes is itself the refresh staying live.
    from charlie_work.state import load_state, save_state, state_lock

    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "status": "janitor_blocked",
            "janitor_ok": False,
            "janitor_failures": ["PR is marked as draft"],
        }
        save_state(app.paths.state_file, state)
    assert "stale-digest" != current_sha

    app.loop(limit=0)
    app.loop(limit=0)

    events = _events_of_kind(paths_from_app(app), "review_packet_template_stale")
    assert sum(1 for e in events if e.get("pr_number") == 456) == 1


def test_loop_janitor_blocked_missing_checks_heals_when_check_appears(
    tmp_path: Path,
) -> None:
    """Keeping review() on the call path is what lets the suppression lift:
    once the missing check reports, the janitor gate passes and the packet
    regenerates with the current template -- the PR converges instead of
    wedging inside its own suppressed state. This is the property the
    issue's "keep the janitor-diagnostics-refresh behavior" clause pins."""
    pr = _pr456("sha-same")
    app, fake_gh = _make_loop_app_with_required_checks(
        tmp_path, prs=[pr], required_checks=("Tests passed",)
    )
    current_sha = app._review_template_sha()

    checks_sequence: list[list[dict]] = [
        [],  # pass 1: required check still missing
        [{"name": "Tests passed", "state": "SUCCESS"}],  # pass 2: it reports
    ]
    checks_calls: list[int] = []

    def fake_pr_checks(number: int) -> list[dict]:
        checks_calls.append(number)
        return checks_sequence[min(len(checks_calls) - 1, len(checks_sequence) - 1)]

    fake_gh.pr_checks = fake_pr_checks  # type: ignore[method-assign]

    _plant_packet(tmp_path, 456, head_sha="sha-same", template_sha="stale-digest")
    _seed_pr_janitor_blocked_missing_checks(app, pr_number=456, issue_number=123)

    app.loop(limit=0)
    app.loop(limit=0)

    # review() regenerated the packet with the current template digest --
    # the suppressed state converged instead of spamming.
    pr_json_path = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456" / "pr.json"
    pr_json = json.loads(pr_json_path.read_text(encoding="utf-8"))
    assert pr_json["prompt_template_sha"] == current_sha
    events = _events_of_kind(paths_from_app(app), "review_packet_template_stale")
    assert not any(e.get("pr_number") == 456 for e in events)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def paths_from_app(app: OrchestratorApp) -> Path:
    return app.paths.state_file


def _events_of_kind(state_path: Path, kind: str) -> list[dict]:
    from charlie_work.instrumentation import query_events

    return query_events(state_path, kind=kind)
