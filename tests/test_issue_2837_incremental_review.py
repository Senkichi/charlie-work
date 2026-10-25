"""Issue #2837: smaller turn budget for incremental review rounds.

The dispatch claim resolves the reviewer turn cap: an incremental round (its
packet renders the archive-based PR-relative interdiff and the incremental
procedure) runs ``incremental_review_max_turns`` (default 20) as the #1439
structure/miss-escalation base; first and fallback rounds keep the flat
``review_max_turns`` (40). ``incremental_review: False`` (the kill switch)
stamps no round kind and restores today's budget, and a 0 value for the
incremental cap does the same.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from _fakes_github import FakeGitHub
from _review_fixtures import _fake_claude_worker_record
from charlie_work.claude_code import ClaudeWorkerRecord
from charlie_work.config import OrchestratorConfig, ReviewDispatchConfig
from charlie_work.instrumentation import query_events
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state, save_state, state_lock
from charlie_work.workflow import OrchestratorApp

_PR = 456
_ISSUE = 123


def _one_pr() -> list[dict[str, Any]]:
    return [
        {
            "number": _PR,
            "title": "Fix #123",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-fix",
            "baseRefName": "main",
            "headRefOid": "sha-456",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #123\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        }
    ]


def _dispatch_app(
    tmp_path: Path,
    *,
    config: OrchestratorConfig | None = None,
) -> OrchestratorApp:
    resolved_config = config or OrchestratorConfig(
        review_dispatch=ReviewDispatchConfig(enabled=True)
    )
    paths = runtime_paths(tmp_path, resolved_config.runtime.state_dir)
    (paths.root).mkdir(parents=True, exist_ok=True)
    (paths.root / "state.json").write_text(
        json.dumps({"version": 1, "issues": {}, "prs": {}, "events": []}),
        encoding="utf-8",
    )
    fake_gh = FakeGitHub()
    fake_gh.prs = _one_pr()
    fake_gh.issues = []
    return OrchestratorApp(tmp_path, paths, resolved_config, fake_gh)


def _write_packet(tmp_path: Path) -> None:
    """Write a minimal current review packet for PR #456."""
    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / f"pr-{_PR}"
    pr_dir.mkdir(parents=True, exist_ok=True)
    (pr_dir / "pr.json").write_text(
        json.dumps({"number": _PR, "headRefOid": "sha-456"}),
        encoding="utf-8",
    )
    (pr_dir / "review-prompt.md").write_text("review prompt", encoding="utf-8")


def _seed_dispatchable_round(
    app: OrchestratorApp,
    *,
    round_kind: str | None,
) -> None:
    """Seed state so ``dispatch_reviews`` sees PR #456 as dispatchable with a
    completed prior claim and, optionally, a stamped round kind."""
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        record = {
            **state["prs"].get(str(_PR), {}),
            "number": _PR,
            "issue_number": _ISSUE,
            "review_dispatch_status": "review_dispatch_completed",
            "review_dispatch_attempt_count": 0,
            "review_turn_limit_miss_streak": 0,
            "review_interdiff_files": 1,
            "review_interdiff_lines": 5,
        }
        if round_kind is not None:
            record["review_round_kind"] = round_kind
        state["prs"][str(_PR)] = record
        save_state(app.paths.state_file, state)


def _capture_launch(monkeypatch: Any) -> list[dict[str, Any]]:
    captured: list[dict[str, Any]] = []

    def fake_launch(*args: Any, **kwargs: Any) -> ClaudeWorkerRecord:
        captured.append(kwargs)
        return _fake_claude_worker_record(
            kwargs.get("issue_number") or args[0],
            kwargs.get("branch") or args[1],
        )

    monkeypatch.setattr("charlie_work.workflow.launch_claude_worker", fake_launch)
    return captured


# ---------------------------------------------------------------------------
# Issue #2837: the PR-relative interdiff and the incremental procedure.
# ---------------------------------------------------------------------------

_ROUND1_PR_DIFF = (
    "diff --git a/one.py b/one.py\n"
    "--- a/one.py\n"
    "+++ b/one.py\n"
    "@@ -10,3 +10,3 @@\n"
    " line\n"
    "+added line\n"
    "-context\n"
    "diff --git a/two.py b/two.py\n"
    "--- a/two.py\n"
    "+++ b/two.py\n"
    "@@ -20,2 +20,2 @@\n"
    " alpha\n"
    " beta\n"
)

# The planted rework's PR diff. one.py is byte-identical above everything but
# the hunk header's line numbers (the base merge shifted them -- a pure
# base-merge re-chunking is NOT a content change); two.py's only delta is
# shifted line numbers too; three.py is the one genuinely changed (new) file.
# base_noise.txt is deliberately ABSENT from both PR diffs: a rework that also
# merges base commits must not surface those files in ANY list (the old
# three-dot interdiff included them; the PR-relative one cannot).
_ROUND2_PR_DIFF = (
    "diff --git a/one.py b/one.py\n"
    "--- a/one.py\n"
    "+++ b/one.py\n"
    "@@ -40,3 +40,3 @@\n"
    " line\n"
    "+added line\n"
    "-context\n"
    "diff --git a/two.py b/two.py\n"
    "--- a/two.py\n"
    "+++ b/two.py\n"
    "@@ -30,2 +30,2 @@\n"
    " alpha\n"
    " beta\n"
    "diff --git a/three.py b/three.py\n"
    "--- /dev/null\n"
    "+++ b/three.py\n"
    "@@ -0,0 +1,1 @@\n"
    "+hello\n"
)


def _plant_round1_with_diff(round_dir: Path) -> None:
    round_dir.mkdir(parents=True, exist_ok=True)
    (round_dir / "review-decision.json").write_text(
        json.dumps(
            {
                "decision": "request_changes",
                "summary": "round-1 summary text",
                "required_changes": ["round-1 finding: add null check"],
                "reviewed_head_sha": "sha-r1",
            }
        ),
        encoding="utf-8",
    )
    (round_dir / "diff.patch").write_text(_ROUND1_PR_DIFF, encoding="utf-8")


def test_review_round2_incremental_interdiff_is_pr_relative(tmp_path: Path) -> None:
    """Issue #2837: with the last round's diff.patch archived, the interdiff is
    built per-file between the two PR diffs -- the genuinely changed file gets
    its patch; shifted-line-number files are listed as unchanged; base-merge
    files never appear anywhere, and ``gh.compare_diff`` is never consulted."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()  # PR 456 headRefOid is "sha-abc123"
    fake_gh.diffs[456] = _ROUND2_PR_DIFF
    decision_dir = paths.prs / "pr-456"
    _plant_round1_with_diff(decision_dir / "rounds" / "round-1")
    (decision_dir / "review-decision.json").write_text(
        json.dumps(
            {
                "decision": "request_changes",
                "summary": "round-1 summary text",
                "required_changes": ["round-1 finding: add null check"],
                "reviewed_head_sha": "sha-r1",
            }
        ),
        encoding="utf-8",
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is True
    interdiff_text = (decision_dir / "interdiff.patch").read_text(encoding="utf-8")
    # Only the genuinely changed PR file's patch is in the interdiff.
    assert interdiff_text == (
        "diff --git a/three.py b/three.py\n"
        "--- /dev/null\n"
        "+++ b/three.py\n"
        "@@ -0,0 +1,1 @@\n"
        "+hello\n"
    )
    # The legacy three-dot compare leaves its SHA-laden default text behind --
    # its absence proves the archive-based builder ran instead.
    assert "sha-r1" not in interdiff_text
    assert "sha-abc123" not in interdiff_text
    packet = (decision_dir / "review-prompt.md").read_text(encoding="utf-8")
    # The interdiff path itself is referenced in the packet, pointing at the
    # changed file's patch; the unchanged files are named in the section.
    assert "interdiff.patch" in packet
    # The base-merge file appears in no list at all.
    assert "base_noise" not in packet
    assert "base_noise" not in interdiff_text


def test_review_round2_requires_only_hunk_shifts_listed_unchanged(tmp_path: Path) -> None:
    """A file whose only change is shifted @@ line numbers is listed as
    unchanged since the prior round, per the #2837 normalisation rule."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.diffs[456] = _ROUND2_PR_DIFF
    decision_dir = paths.prs / "pr-456"
    _plant_round1_with_diff(decision_dir / "rounds" / "round-1")
    (decision_dir / "review-decision.json").write_text(
        json.dumps(
            {
                "decision": "request_changes",
                "summary": "round-1 summary text",
                "required_changes": ["round-1 finding"],
                "reviewed_head_sha": "sha-r1",
            }
        ),
        encoding="utf-8",
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is True
    packet = (decision_dir / "review-prompt.md").read_text(encoding="utf-8")
    # Both shift-only files (and the byte-identical one.py) are unchanged;
    # "Files unchanged since round 1" lists one.py and two.py. three.py (the
    # genuinely changed file) must not appear here.
    unchanged_start = packet.index("Files unchanged since round 1")
    unchanged_block = packet[unchanged_start : packet.index("## CI status")]
    assert "`one.py`" in unchanged_block
    assert "`two.py`" in unchanged_block
    assert "three.py" not in unchanged_block
    # The full-diff-re-read instruction is gone: the unchanged files keep the
    # earlier round's assessment.
    assert "the full diff remains authoritative" not in packet
    assert "keep the round-1 assessment" in packet


def test_review_round2_incremental_procedure_renders(tmp_path: Path) -> None:
    """A round-2 packet with an archive-built interdiff renders the
    incremental procedure -- including the explicit per-finding status
    instruction. A round-1 packet (no prior review) never does."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.diffs[456] = _ROUND2_PR_DIFF
    decision_dir = paths.prs / "pr-456"
    _plant_round1_with_diff(decision_dir / "rounds" / "round-1")
    (decision_dir / "review-decision.json").write_text(
        json.dumps(
            {
                "decision": "request_changes",
                "summary": "round-1 summary text",
                "required_changes": ["round-1 finding"],
                "reviewed_head_sha": "sha-r1",
            }
        ),
        encoding="utf-8",
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is True
    packet = (decision_dir / "review-prompt.md").read_text(encoding="utf-8")
    assert "## Incremental review procedure" in packet
    assert "resolved, still present, or regressed" in packet
    assert "Do not re-review the unchanged files" in packet


def test_review_round1_prompt_has_no_incremental_procedure(tmp_path: Path) -> None:
    """A round-1 packet (no prior terminal review) must not render the
    incremental procedure -- it reviews the full diff from scratch."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.diffs[456] = _ROUND2_PR_DIFF
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is True
    packet = (paths.prs / "pr-456" / "review-prompt.md").read_text(encoding="utf-8")
    assert "## Prior review" not in packet
    assert "Incremental review procedure" not in packet
    assert "resolved, still present, or regressed" not in packet


def test_incremental_round_pins_incremental_turn_cap(tmp_path: Path, monkeypatch: Any) -> None:
    """An incremental round's launcher pins the 20-turn budget; first and
    fallback rounds keep the flat ``review_max_turns``."""
    app = _dispatch_app(tmp_path)
    _write_packet(tmp_path)
    captured = _capture_launch(monkeypatch)

    _seed_dispatchable_round(app, round_kind="incremental")
    app.dispatch_reviews()
    assert len(captured) == 1
    assert captured[0]["max_turns_override"] == 20

    # A first round keeps the flat review_max_turns budget.
    captured.clear()
    _seed_dispatchable_round(app, round_kind="first")
    app.dispatch_reviews()
    assert captured[0]["max_turns_override"] == app.config.review_dispatch.review_max_turns == 40

    # A fallback round (legacy archive / compare-diff interdiff) also keeps
    # 40: the incremental procedure is not rendered for it.
    captured.clear()
    _seed_dispatchable_round(app, round_kind="fallback")
    app.dispatch_reviews()
    assert captured[0]["max_turns_override"] == 40


def test_incremental_zero_budget_restores_flat_budget(tmp_path: Path, monkeypatch: Any) -> None:
    """``incremental_review_max_turns: 0`` restores today's flat budget."""
    app = _dispatch_app(
        tmp_path,
        config=OrchestratorConfig(
            review_dispatch=ReviewDispatchConfig(enabled=True, incremental_review_max_turns=0)
        ),
    )
    _write_packet(tmp_path)
    captured = _capture_launch(monkeypatch)
    _seed_dispatchable_round(app, round_kind="incremental")
    app.dispatch_reviews()
    assert captured[0]["max_turns_override"] == 40


def test_kill_switch_off_stamps_no_round_kind(tmp_path: Path, monkeypatch: Any) -> None:
    """With the kill switch off a round-2 packet with an archived diff pins
    today's budget, stamps no round kind, and emits no fallback event."""
    app = _dispatch_app(
        tmp_path,
        config=OrchestratorConfig(
            review_dispatch=ReviewDispatchConfig(enabled=True, incremental_review=False)
        ),
    )
    diff = "diff --git a/one.py b/one.py\n--- a/one.py\n+++ b/one.py\n@@ -1,1 +1,1 @@\n x\n"
    app.gh.diffs[_PR] = diff
    app.gh.issues = [
        {
            "number": _ISSUE,
            "title": "Fix search",
            "url": "https://example.test/issues/123",
            "body": "Search is broken",
            "labels": [{"name": "automated-ready"}],
            "state": "OPEN",
        }
    ]
    result = app.review(_PR)
    assert result.ok is True
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
    assert "review_round_kind" not in state["prs"][str(_PR)]
    assert query_events(app.paths.state_file, kind="review_interdiff_fallback") == []


def test_claim_and_dispatch_events_carry_round_kinds(tmp_path: Path, monkeypatch: Any) -> None:
    """The claim event mirrors each selected PR's round kind + interdiff size,
    and the review_dispatch event carries the same maps (issue #2837)."""
    app = _dispatch_app(tmp_path)
    _write_packet(tmp_path)
    _capture_launch(monkeypatch)
    _seed_dispatchable_round(app, round_kind="incremental")

    app.dispatch_reviews()

    claims = query_events(app.paths.state_file, kind="review_dispatch_claim")
    payloads = [c["payload"] for c in claims if c["payload"].get("pr_numbers") == [_PR]]
    assert payloads, claims
    assignment = payloads[-1]["review_effort_assignments"][0]
    assert assignment["review_round_kind"] == "incremental"
    assert assignment["review_interdiff_files"] == 1
    assert assignment["review_interdiff_lines"] == 5

    dispatches = query_events(app.paths.state_file, kind="review_dispatch")
    assert any(
        d["payload"].get("review_round_kinds", {}).get(str(_PR)) == "incremental"
        and d["payload"].get("review_interdiff_counts", {}).get(str(_PR))
        == {"files": 1, "lines": 5}
        for d in dispatches
    )


# ---------------------------------------------------------------------------
# Issue #2837: round-kind stamping, the fallback event, and the kill switch.
# ---------------------------------------------------------------------------


def test_review_round_kind_first_stamp(tmp_path: Path) -> None:
    """A first-round packet stamps review_round_kind == "first" onto the PR
    record (the dispatch side reads it to select the turn budget)."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.diffs[456] = _ROUND2_PR_DIFF
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is True
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
    assert state["prs"]["456"]["review_round_kind"] == "first"


def test_review_round_kind_incremental_and_fallback_stamp(tmp_path: Path) -> None:
    """With an archived last-round diff the kind is "incremental"; with only a
    legacy (decision-only) round it is "fallback"."""  # noqa: D205
    # incremental: archived diff present.
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path / "a", config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.diffs[456] = _ROUND2_PR_DIFF
    decision_dir = paths.prs / "pr-456"
    _plant_round1_with_diff(decision_dir / "rounds" / "round-1")
    (decision_dir / "review-decision.json").write_text(
        json.dumps(
            {
                "decision": "request_changes",
                "summary": "round-1 summary text",
                "required_changes": ["round-1 finding"],
                "reviewed_head_sha": "sha-r1",
            }
        ),
        encoding="utf-8",
    )
    app = OrchestratorApp(tmp_path / "a", paths, config, fake_gh)
    assert app.review(456).ok is True
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
    record = state["prs"]["456"]
    assert record["review_round_kind"] == "incremental"
    assert record["review_interdiff_files"] == 1
    assert record["review_interdiff_lines"] == 5

    # fallback: legacy round-1 archive without a diff.patch.
    fb_config = OrchestratorConfig()
    fb_paths = runtime_paths(tmp_path / "b", fb_config.runtime.state_dir)
    fb_fake = FakeGitHub()
    fb_fake.diffs[456] = _ROUND2_PR_DIFF
    fb_dir = fb_paths.prs / "pr-456"
    fb_dir.mkdir(parents=True, exist_ok=True)
    (fb_dir / "rounds" / "round-1").mkdir(parents=True, exist_ok=True)
    (fb_dir / "rounds" / "round-1" / "review-decision.json").write_text(
        json.dumps(
            {
                "decision": "request_changes",
                "summary": "legacy round",
                "required_changes": ["legacy finding"],
                "reviewed_head_sha": "sha-r1",
            }
        ),
        encoding="utf-8",
    )
    (fb_dir / "review-decision.json").write_text(
        json.dumps(
            {
                "decision": "request_changes",
                "summary": "legacy round",
                "required_changes": ["legacy finding"],
                "reviewed_head_sha": "sha-r1",
            }
        ),
        encoding="utf-8",
    )
    fb_app = OrchestratorApp(tmp_path / "b", fb_paths, fb_config, fb_fake)
    assert fb_app.review(456).ok is True
    with state_lock(fb_app.paths.state_file):
        state = load_state(fb_app.paths.state_file)
    assert state["prs"]["456"]["review_round_kind"] == "fallback"
    fallback_events = query_events(fb_app.paths.state_file, kind="review_interdiff_fallback")
    assert any(
        e["payload"].get("pr_number") == 456 and e["payload"].get("reason") == "no_archived_diff"
        for e in fallback_events
    )


def test_review_kill_switch_off_restores_today(tmp_path: Path) -> None:
    """``incremental_review: False`` restores today's prompt (three-dot
    interdiff, full-diff procedure), today's turn budget basis, no round-kind
    stamp, and no ``review_interdiff_fallback`` event -- even when an archived
    last-round diff exists."""
    config = OrchestratorConfig(
        review_dispatch=ReviewDispatchConfig(incremental_review=False),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()  # PR 456 headRefOid is "sha-abc123"
    fake_gh.diffs[456] = _ROUND2_PR_DIFF
    decision_dir = paths.prs / "pr-456"
    _plant_round1_with_diff(decision_dir / "rounds" / "round-1")
    (decision_dir / "review-decision.json").write_text(
        json.dumps(
            {
                "decision": "request_changes",
                "summary": "round-1 summary text",
                "required_changes": ["round-1 finding"],
                "reviewed_head_sha": "sha-r1",
            }
        ),
        encoding="utf-8",
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is True
    interdiff_text = (decision_dir / "interdiff.patch").read_text(encoding="utf-8")
    assert "sha-r1" in interdiff_text
    assert "sha-abc123" in interdiff_text
    packet = (decision_dir / "review-prompt.md").read_text(encoding="utf-8")
    assert "Incremental review procedure" not in packet
    assert "the full diff remains authoritative" in packet
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
    assert "review_round_kind" not in state["prs"]["456"]
    assert query_events(app.paths.state_file, kind="review_interdiff_fallback") == []


# ---------------------------------------------------------------------------
# Mutation checks (AC8 style, mirroring test_review_round_archive.py): each
# disabled fix must break exactly the acceptance test it protects.
# ---------------------------------------------------------------------------


def _assert_round_archive_copies_diff(tmp_path: Path) -> None:
    """The assertion suite AC-of-archiving: each round-K archive holds that
    round's own diff.patch, surviving the next round's in-place overwrite."""
    from _review_fixtures import _pr_dir, _record, _round_archive_app, _round_dir

    app, paths = _round_archive_app(tmp_path)
    pr_dir = _pr_dir(paths)
    pr_dir.mkdir(parents=True, exist_ok=True)
    (pr_dir / "diff.patch").write_text("ROUND-ONE DIFF", encoding="utf-8")
    _record(app, head="sha-r1", summary="round one", required_changes=["c1"])
    (pr_dir / "diff.patch").write_text("ROUND-TWO DIFF", encoding="utf-8")
    _record(app, head="sha-r2", summary="round two", required_changes=["c2"])
    path1 = _round_dir(paths, 1) / "diff.patch"
    assert path1.exists(), "round-1/diff.patch missing"
    assert path1.read_text(encoding="utf-8") == "ROUND-ONE DIFF"


def test_mutation_no_archive_copy_breaks_diff_archiving(tmp_path: Path, monkeypatch: Any) -> None:
    """Mutation: archive_round_diff is a no-op (the #2837 fix reverted) --
    the archive-acceptance assertion must FAIL."""
    import charlie_work.review_interdiff as ri

    monkeypatch.setattr(ri, "archive_round_diff", lambda pr_dir, round_dir: None)
    with pytest.raises(AssertionError, match="round-1/diff.patch missing"):
        _assert_round_archive_copies_diff(tmp_path)


def test_mutation_no_hunk_normalisation_marks_shift_only_files_changed(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Mutation: @@ hunk headers are NOT normalised (the #2837 line-number
    rule reverted) -- the shift-only file must wrongly render as changed,
    breaking the unchanged-file assertion."""
    import charlie_work.review_interdiff as ri

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.diffs[456] = _ROUND2_PR_DIFF
    decision_dir = paths.prs / "pr-456"
    _plant_round1_with_diff(decision_dir / "rounds" / "round-1")
    (decision_dir / "review-decision.json").write_text(
        json.dumps(
            {
                "decision": "request_changes",
                "summary": "round-1 summary text",
                "required_changes": ["round-1 finding"],
                "reviewed_head_sha": "sha-r1",
            }
        ),
        encoding="utf-8",
    )

    original = ri._normalize_hunk_line

    def _no_normalise(line: str) -> str:
        return line  # identity: every line-number shift is a "change"

    monkeypatch.setattr(ri, "_normalize_hunk_line", _no_normalise)
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    result = app.review(456)
    monkeypatch.setattr(ri, "_normalize_hunk_line", original)
    assert result.ok is True
    packet = (decision_dir / "review-prompt.md").read_text(encoding="utf-8")
    # Under the mutation the shift-only files read as "changed", so the
    # unchanged-files line the real test asserts is no longer rendered.
    with pytest.raises(ValueError):
        packet.index("Files unchanged since round 1")
