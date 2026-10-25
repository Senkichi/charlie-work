"""Issue #2837: incremental review rounds -- packet facts, fallback, turn budget.

The round-2 packet builds its interdiff PR-relative from the last archived
round's ``diff.patch`` (the three round-history acceptance tests live in
``test_charlie_work_review_round_history``). This module covers everything
around that: the stale-``interdiff.patch`` hygiene, the removed-files list, the
``review_interdiff_fallback`` event, the round-kind stamp, and the dispatch
side -- an incremental round (its packet renders the archive-built interdiff
and the incremental procedure) runs ``incremental_review_max_turns`` (default
20) as the #1439 structure/miss-escalation base; first and fallback rounds
keep the flat ``review_max_turns`` (40). ``incremental_review: False`` (the
kill switch) stamps no round kind and restores today's budget, and a 0 value
for the incremental cap does the same. Budgets are asserted on the argv the
real launcher renders, not only on the kwargs the dispatch hands it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from _fakes_github import FakeGitHub
from _incremental_review_fixtures import (
    BASE_NOISE_COMPARE_DIFF,
    ROUND1_HEAD,
    ROUND1_PR_DIFF,
    ROUND2_PR_DIFF,
    capture_real_review_launches,
    max_turns_in,
    plant_round1,
)
from _review_fixtures import _fake_claude_worker_record
from charlie_work.claude_code import ClaudeWorkerRecord
from charlie_work.config import OrchestratorConfig, ReviewDispatchConfig
from charlie_work.instrumentation import query_events
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state, save_state, state_lock
from charlie_work.workflow import OrchestratorApp

_PR = 456
_ISSUE = 123
_LIVE_HEAD = "sha-456"  # _one_pr()'s headRefOid
_STALE_INTERDIFF = "STALE INTERDIFF LEFT BY AN EARLIER MOVED-HEAD ROUND\n"


def _one_pr() -> list[dict[str, Any]]:
    return [
        {
            "number": _PR,
            "title": "Fix #123",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-fix",
            "baseRefName": "main",
            "headRefOid": _LIVE_HEAD,
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
    fake_gh.issues = [
        {
            "number": _ISSUE,
            "title": "Fix search",
            "url": "https://example.test/issues/123",
            "body": "Search is broken",
            "labels": [{"name": "automated-ready"}],
            "state": "OPEN",
        }
    ]
    return OrchestratorApp(tmp_path, paths, resolved_config, fake_gh)


def _pr_dir(app: OrchestratorApp) -> Path:
    return app.paths.prs / f"pr-{_PR}"


def _pr_record(app: OrchestratorApp) -> dict[str, Any]:
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
    return state["prs"][str(_PR)]


def _write_packet(tmp_path: Path) -> None:
    """Write a minimal current review packet for PR #456."""
    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / f"pr-{_PR}"
    pr_dir.mkdir(parents=True, exist_ok=True)
    (pr_dir / "pr.json").write_text(
        json.dumps({"number": _PR, "headRefOid": _LIVE_HEAD}),
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


def _review_then_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    prior: str,
    incremental_review: bool = True,
) -> tuple[OrchestratorApp, dict[str, Any], dict[str, Any]]:
    """Packet (``review()``) then claim + launch (``dispatch_reviews()``) for
    PR #456; returns ``(app, packet-stamped PR record, the one launch capture)``.

    ``prior``: ``"none"`` (first round), ``"archived"`` (round 1 archived with
    its diff.patch) or ``"legacy"`` (round 1 archived without one).
    """
    app = _dispatch_app(
        tmp_path,
        config=OrchestratorConfig(
            review_dispatch=ReviewDispatchConfig(
                enabled=True, incremental_review=incremental_review
            )
        ),
    )
    launches = capture_real_review_launches(monkeypatch, tmp_path)
    app.gh.diffs[_PR] = ROUND2_PR_DIFF
    if prior != "none":
        plant_round1(_pr_dir(app), diff=ROUND1_PR_DIFF if prior == "archived" else None)
    assert app.review(_PR).ok is True
    record = _pr_record(app)

    app.dispatch_reviews()

    assert len(launches) == 1, launches
    return app, record, launches[0]


# ---------------------------------------------------------------------------
# Issue #2837: the incremental procedure is a round-2 feature only.
# ---------------------------------------------------------------------------


def test_review_round1_prompt_has_no_incremental_procedure(tmp_path: Path) -> None:
    """A round-1 packet (no prior terminal review) must not render the
    incremental procedure -- it reviews the full diff from scratch."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.diffs[456] = ROUND2_PR_DIFF
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is True
    packet = (paths.prs / "pr-456" / "review-prompt.md").read_text(encoding="utf-8")
    assert "## Prior review" not in packet
    assert "Incremental review procedure" not in packet
    assert "resolved, still present, or regressed" not in packet


# ---------------------------------------------------------------------------
# Issue #2837: dispatch-side turn budget, asserted on the real launcher argv.
# ---------------------------------------------------------------------------


def test_incremental_round_pins_incremental_turn_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An incremental round's launcher pins the 20-turn budget; first and
    fallback rounds keep the flat ``review_max_turns``."""
    app = _dispatch_app(tmp_path)
    _write_packet(tmp_path)
    launches = capture_real_review_launches(monkeypatch, tmp_path)

    _seed_dispatchable_round(app, round_kind="incremental")
    app.dispatch_reviews()
    assert len(launches) == 1
    assert launches[0]["kwargs"]["max_turns_override"] == 20
    assert max_turns_in(launches[0]["argv"]) == "20"

    # A first round keeps the flat review_max_turns budget.
    launches.clear()
    _seed_dispatchable_round(app, round_kind="first")
    app.dispatch_reviews()
    assert app.config.review_dispatch.review_max_turns == 40
    assert launches[0]["kwargs"]["max_turns_override"] == 40
    assert max_turns_in(launches[0]["argv"]) == "40"

    # A fallback round (legacy archive / compare-diff interdiff) also keeps
    # 40: the incremental procedure is not rendered for it.
    launches.clear()
    _seed_dispatchable_round(app, round_kind="fallback")
    app.dispatch_reviews()
    assert max_turns_in(launches[0]["argv"]) == "40"


def test_incremental_zero_budget_restores_flat_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``incremental_review_max_turns: 0`` restores today's flat budget."""
    app = _dispatch_app(
        tmp_path,
        config=OrchestratorConfig(
            review_dispatch=ReviewDispatchConfig(enabled=True, incremental_review_max_turns=0)
        ),
    )
    _write_packet(tmp_path)
    launches = capture_real_review_launches(monkeypatch, tmp_path)
    _seed_dispatchable_round(app, round_kind="incremental")
    app.dispatch_reviews()
    assert launches[0]["kwargs"]["max_turns_override"] == 40
    assert max_turns_in(launches[0]["argv"]) == "40"


@pytest.mark.parametrize(
    ("prior", "expected_kind", "expected_turns"),
    [
        pytest.param("none", "first", "40", id="first-round"),
        pytest.param("archived", "incremental", "20", id="archived-diff-incremental"),
        pytest.param("legacy", "fallback", "40", id="legacy-round-fallback"),
    ],
)
def test_round_kind_selects_launcher_turn_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prior: str,
    expected_kind: str,
    expected_turns: str,
) -> None:
    """End to end: the packet's round kind (not a seeded stamp) decides the
    ``--max-turns`` the reviewer is launched with."""
    app, record, launch = _review_then_dispatch(tmp_path, monkeypatch, prior=prior)

    assert record["review_round_kind"] == expected_kind
    assert max_turns_in(launch["argv"]) == expected_turns
    assert launch["kwargs"]["max_turns_override"] == int(expected_turns)
    claims = query_events(app.paths.state_file, kind="review_dispatch_claim")
    assert claims[-1]["payload"]["review_effort_assignments"][0]["review_round_kind"] == (
        expected_kind
    )


def test_kill_switch_off_stamps_no_round_kind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the kill switch off, a round-2 packet that HAS an archived
    last-round diff pins today's 40-turn budget, stamps no round kind, and
    emits no fallback event. The archived diff is what makes this non-vacuous:
    ``test_round_kind_selects_launcher_turn_budget`` shows the same setup with
    the switch on stamps ``incremental`` and launches with 20."""
    app, record, launch = _review_then_dispatch(
        tmp_path, monkeypatch, prior="archived", incremental_review=False
    )

    assert (_pr_dir(app) / "rounds" / "round-1" / "diff.patch").is_file()
    assert "review_round_kind" not in record
    assert "review_interdiff_files" not in record
    assert query_events(app.paths.state_file, kind="review_interdiff_fallback") == []
    assert max_turns_in(launch["argv"]) == "40"
    assert launch["kwargs"]["max_turns_override"] == 40
    assert "review_round_kind" not in _pr_record(app)


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
    fake_gh.diffs[456] = ROUND2_PR_DIFF
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
    fake_gh.diffs[456] = ROUND2_PR_DIFF
    plant_round1(paths.prs / "pr-456")
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
    fb_fake.diffs[456] = ROUND2_PR_DIFF
    plant_round1(fb_paths.prs / "pr-456", diff=None)
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
    fake_gh.diffs[456] = ROUND2_PR_DIFF
    decision_dir = paths.prs / "pr-456"
    plant_round1(decision_dir)
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
# Issue #2837 rework: packet-builder regressions (removed files, stale
# interdiff.patch hygiene, archive_read_failed).
# ---------------------------------------------------------------------------

_GONE_PY_SECTION = (
    "diff --git a/gone.py b/gone.py\n--- a/gone.py\n+++ b/gone.py\n@@ -1,1 +1,1 @@\n-x\n+y\n"
)
# ROUND1_PR_DIFF with every hunk header's line numbers shifted (a pure base
# merge above the touched regions): no file's patch content changed.
_ROUND1_SHIFTED_ONLY = ROUND1_PR_DIFF.replace("@@ -10,3 +10,3 @@", "@@ -50,3 +50,3 @@").replace(
    "@@ -20,2 +20,2 @@", "@@ -60,2 +60,2 @@"
)


def test_review_round2_names_files_removed_since_the_prior_round(tmp_path: Path) -> None:
    """A file in the archived round diff and absent from the live PR diff is
    listed by name -- not in the interdiff, and not as unchanged."""
    app = _dispatch_app(tmp_path)
    app.gh.diffs[_PR] = ROUND2_PR_DIFF
    plant_round1(_pr_dir(app), diff=ROUND1_PR_DIFF + _GONE_PY_SECTION)

    assert app.review(_PR).ok is True

    packet = (_pr_dir(app) / "review-prompt.md").read_text(encoding="utf-8")
    assert "Files removed from the PR since round 1: `gone.py`." in packet
    interdiff_text = (_pr_dir(app) / "interdiff.patch").read_text(encoding="utf-8")
    assert "gone.py" not in interdiff_text
    unchanged_start = packet.index("Files unchanged since round 1")
    assert "gone.py" not in packet[unchanged_start : packet.index("Files removed from the PR")]


def test_review_round2_all_unchanged_discards_stale_interdiff(tmp_path: Path) -> None:
    """When no file's patch changed (a shift-only rework), there is NO
    interdiff: an ``interdiff.patch`` left by an earlier moved-head round must
    not survive on disk or be measured into the round's interdiff counts."""
    app = _dispatch_app(tmp_path)
    app.gh.diffs[_PR] = _ROUND1_SHIFTED_ONLY
    plant_round1(_pr_dir(app))
    (_pr_dir(app) / "interdiff.patch").write_text(_STALE_INTERDIFF, encoding="utf-8")

    assert app.review(_PR).ok is True

    assert not (_pr_dir(app) / "interdiff.patch").exists()
    packet = (_pr_dir(app) / "review-prompt.md").read_text(encoding="utf-8")
    assert "No file's patch changed between round 1" in packet
    record = _pr_record(app)
    assert record["review_round_kind"] == "incremental"
    assert record["review_interdiff_files"] == 0
    assert record["review_interdiff_lines"] == 0


def test_review_same_head_rereview_discards_stale_interdiff(tmp_path: Path) -> None:
    """A same-head re-review has no interdiff at all: a stale
    ``interdiff.patch`` from an earlier moved-head round is removed and its
    file/line counts are NOT carried into the round's stamp."""
    app = _dispatch_app(tmp_path)
    app.gh.diffs[_PR] = ROUND2_PR_DIFF
    plant_round1(_pr_dir(app), head=_LIVE_HEAD)
    (_pr_dir(app) / "interdiff.patch").write_text(_STALE_INTERDIFF * 3, encoding="utf-8")

    assert app.review(_PR).ok is True

    assert not (_pr_dir(app) / "interdiff.patch").exists()
    packet = (_pr_dir(app) / "review-prompt.md").read_text(encoding="utf-8")
    assert "## Prior review (same head)" in packet
    record = _pr_record(app)
    assert record["review_round_kind"] == "fallback"
    assert record["review_interdiff_files"] == 0
    assert record["review_interdiff_lines"] == 0
    # A same-head round is not an archive failure: no fallback event.
    assert query_events(app.paths.state_file, kind="review_interdiff_fallback") == []


def test_review_round2_failed_compare_discards_stale_interdiff(tmp_path: Path) -> None:
    """The legacy fallback with no usable comparison also clears a stale
    ``interdiff.patch`` (and records why the archive path was not taken)."""
    app = _dispatch_app(tmp_path)
    app.gh.diffs[_PR] = ROUND2_PR_DIFF
    app.gh.compare_diff_overrides[(ROUND1_HEAD, _LIVE_HEAD)] = None
    plant_round1(_pr_dir(app), diff=None)
    (_pr_dir(app) / "interdiff.patch").write_text(_STALE_INTERDIFF, encoding="utf-8")

    assert app.review(_PR).ok is True

    assert not (_pr_dir(app) / "interdiff.patch").exists()
    packet = (_pr_dir(app) / "review-prompt.md").read_text(encoding="utf-8")
    assert "unavailable" in packet.lower()
    record = _pr_record(app)
    assert record["review_round_kind"] == "fallback"
    assert record["review_interdiff_files"] == 0


def test_review_round2_unreadable_archive_falls_back_and_says_why(tmp_path: Path) -> None:
    """An archived ``diff.patch`` that exists but cannot be decoded is
    ``archive_read_failed``: the packet falls back to the three-dot compare,
    one ``review_interdiff_fallback(reason="archive_read_failed")`` is
    recorded, and the round is stamped ``fallback`` -- not ``incremental``,
    whose procedure the legacy interdiff cannot support."""
    app = _dispatch_app(tmp_path)
    app.gh.diffs[_PR] = ROUND2_PR_DIFF
    app.gh.compare_diff_overrides[(ROUND1_HEAD, _LIVE_HEAD)] = BASE_NOISE_COMPARE_DIFF
    plant_round1(_pr_dir(app))
    (_pr_dir(app) / "rounds" / "round-1" / "diff.patch").write_bytes(
        b"\xff\xfe\x80 not utf-8 \xc3"
    )

    assert app.review(_PR).ok is True

    events = query_events(app.paths.state_file, kind="review_interdiff_fallback")
    assert len(events) == 1, events
    assert events[0]["payload"]["reason"] == "archive_read_failed"
    assert events[0]["payload"]["pr_number"] == _PR
    assert events[0]["payload"]["prior_head_sha"] == ROUND1_HEAD
    assert events[0]["payload"]["new_head_sha"] == _LIVE_HEAD
    # The legacy three-dot interdiff (here the planted base-noise compare).
    interdiff_text = (_pr_dir(app) / "interdiff.patch").read_text(encoding="utf-8")
    assert interdiff_text == BASE_NOISE_COMPARE_DIFF
    packet = (_pr_dir(app) / "review-prompt.md").read_text(encoding="utf-8")
    assert "Incremental review procedure" not in packet
    assert "the full diff remains authoritative" in packet
    record = _pr_record(app)
    assert record["review_round_kind"] == "fallback"
    assert record["review_interdiff_files"] == 1
    packet_events = query_events(app.paths.state_file, kind="review_packet")
    assert packet_events[-1]["payload"]["review_round_kind"] == "fallback"


@pytest.mark.parametrize("decision", ["missing", "pending"])
def test_prior_review_section_ignores_a_nonterminal_fallback_decision(
    tmp_path: Path, decision: str
) -> None:
    """A never-reviewed packet's ``missing``/``pending`` decision is not a
    prior review: with no archived rounds the section is empty and no
    ``review_interdiff_fallback`` is emitted (it used to render a bogus
    "Round 1 (decision: missing)" and a false fallback measurement)."""
    app = _dispatch_app(tmp_path)
    pr_dir = _pr_dir(app)
    pr_dir.mkdir(parents=True)

    section = app._build_prior_review_section(pr_dir, {"decision": decision}, _LIVE_HEAD)

    assert section == ""
    assert query_events(app.paths.state_file, kind="review_interdiff_fallback") == []


def test_prior_review_section_still_reads_the_archive_under_a_nonterminal_mirror(
    tmp_path: Path,
) -> None:
    """Positive control for the guard above: it suppresses only the SYNTHETIC
    fallback round. Archived rounds stay authoritative even when the flat
    mirror the caller passes is a ``pending`` stub."""
    app = _dispatch_app(tmp_path)
    app.gh.diffs[_PR] = ROUND2_PR_DIFF
    pr_dir = _pr_dir(app)
    plant_round1(pr_dir)
    (pr_dir / "diff.patch").write_text(ROUND2_PR_DIFF, encoding="utf-8")

    section = app._build_prior_review_section(pr_dir, {"decision": "pending"}, _LIVE_HEAD)

    assert "## Prior review" in section
    assert "round-1 finding: add null check" in section


# ---------------------------------------------------------------------------
# Mutation checks (AC8 style, mirroring test_review_round_archive.py): each
# disabled fix must break exactly the acceptance test it protects.
# ---------------------------------------------------------------------------


def _assert_round_archive_copies_diff(tmp_path: Path) -> None:
    """The assertion suite AC-of-archiving: each round-K archive holds that
    round's own diff.patch, surviving the next round's in-place overwrite."""
    from _review_fixtures import _pr_dir as _archive_pr_dir
    from _review_fixtures import _record, _round_archive_app, _round_dir

    app, paths = _round_archive_app(tmp_path)
    pr_dir = _archive_pr_dir(paths)
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
    fake_gh.diffs[456] = ROUND2_PR_DIFF
    decision_dir = paths.prs / "pr-456"
    plant_round1(decision_dir)

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
