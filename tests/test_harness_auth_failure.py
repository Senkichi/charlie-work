"""Issue #2810: a harness auth failure is a harness-level fault, not the issue's.

One dead ``devin`` credential failed 31 dispatches in 22 hours because the fleet
charged each death to its own issue and kept launching. These tests pin the fleet's
response end to end: the death is classified (``harness_auth_failed``), is charged
to no cap, blocks the harness fleet-wide through the quota ledger, alerts once per
episode, and resumes in two stages (a cheap credential check, then exactly one real
launch). Every negative assertion sits next to a planted positive control.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from _harness_auth_fixtures import (
    CHAIN,
    CLAUDE_FALLBACK,
    DEVIN_PRIMARY,
    HARNESS,
    INVALID_KEY,
    ISSUES,
    LANES,
    LAPSED,
    NOT_LOGGED_IN,
    arm,
    block,
    dead_devin_session,
    fake_launch,
    make_app,
    register,
    run_lane,
)
from charlie_work import adapter_fate_profile, harness_auth_block, layout, role_selection
from charlie_work import role_quota_ledger as ledger
from charlie_work.config import OrchestratorConfig, WorkerRoleConfig
from charlie_work.dead_worker_sweep.decide_dead_sessions import redispatch_verdict
from charlie_work.dead_worker_sweep.decide_redispatch import decide_redispatch_cap
from charlie_work.devin_failure_classification import (
    update_session_record_with_failure_classification as classify_devin_death,
)
from charlie_work.rework_attempt_exemption import (
    PROVIDER_THROTTLE_EXEMPT_KINDS,
    is_provider_throttle_rework_death,
)
from charlie_work.role_chain import RoleEntry
from charlie_work.state import load_state
from charlie_work.throttle_signatures import (
    HARNESS_AUTH_FAILED_KIND,
    devin_harness_auth_evidence,
    is_provider_throttle_failure,
)
from charlie_work.worker_fate import persisted_failure

UNKNOWN_MODEL_EMPTY = "Error: Unknown model: 'swe-9'\nAvailable:\n"


@pytest.fixture
def launched(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, str]]:
    """Capture ``(issue, adapter)`` of every launch; each succeeds with a live pid."""
    calls: list[tuple[int, str]] = []
    monkeypatch.setattr("charlie_work.adapters.dispatch_sessions", fake_launch(calls))
    return calls


@pytest.fixture
def fleet(tmp_path: Path) -> Path:
    return tmp_path / "fleet"


# --- classifier -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(NOT_LOGGED_IN, id="not-logged-in"),
        pytest.param(NOT_LOGGED_IN.replace("`devin ", "`devin.exe "), id="not-logged-in-exe"),
        pytest.param(INVALID_KEY, id="invalid-api-key"),
        pytest.param(UNKNOWN_MODEL_EMPTY, id="unknown-model-empty-list"),
    ],
)
def test_each_signature_is_recognised_with_or_without_startup_noise(text: str) -> None:
    for log in (text, f"starting devin\nloading config\n{text}"):
        evidence = devin_harness_auth_evidence(log)
        assert evidence is not None
        assert evidence.startswith("Error:")


@pytest.mark.parametrize(
    ("near_miss", "control"),
    [
        pytest.param("see the earlier Error: Not logged in", NOT_LOGGED_IN, id="mid-line"),
        pytest.param(f"  {NOT_LOGGED_IN}", NOT_LOGGED_IN, id="indented"),
        pytest.param(NOT_LOGGED_IN.removeprefix("Error: "), NOT_LOGGED_IN, id="no-error-prefix"),
        pytest.param(NOT_LOGGED_IN.replace("Error:", "error:"), NOT_LOGGED_IN, id="lowercase"),
        pytest.param(
            "Error: Unknown model: 'swe-9'\nAvailable:\n  swe-2\n  swe-1-6\n",
            UNKNOWN_MODEL_EMPTY,
            id="unknown-model-with-a-list-is-a-config-error",
        ),
        pytest.param(
            "Error: Unknown model: 'swe-9'\n", UNKNOWN_MODEL_EMPTY, id="unknown-model-no-list"
        ),
    ],
)
def test_near_misses_do_not_match_but_their_positive_controls_do(
    near_miss: str, control: str
) -> None:
    assert devin_harness_auth_evidence(control) is not None, "control must match"
    assert devin_harness_auth_evidence(near_miss) is None


def test_only_the_devin_harness_declares_a_matcher() -> None:
    profiles = adapter_fate_profile._build_profiles()
    declaring = sorted(h for h, p in profiles.items() if p.harness_auth_evidence is not None)
    assert declaring == [HARNESS]
    silent = [p for p in profiles.values() if p.harness_auth_evidence is None]
    assert {"claude-code", "api"} <= {k for p in silent for k in p.view_kinds}


def test_classify_for_is_devin_only_and_unchanged_elsewhere(tmp_path: Path) -> None:
    log = tmp_path / "w.log"
    log.write_text(NOT_LOGGED_IN + "\n", encoding="utf-8")
    assert adapter_fate_profile.classify_for("devin", log) == (HARNESS_AUTH_FAILED_KIND, None)
    profiles = adapter_fate_profile._build_profiles().values()
    for profile in profiles:
        if profile.harness_auth_evidence is not None:
            continue
        for view_kind in profile.view_kinds:
            kind, until = adapter_fate_profile.classify_for(view_kind, log)
            assert kind != HARNESS_AUTH_FAILED_KIND, view_kind
            assert until is None, view_kind


def _death(tmp_path: Path, text: str = NOT_LOGGED_IN + "\n", **kw: Any) -> tuple[Any, Path]:
    sessions = dead_devin_session(tmp_path, text, **kw)
    return classify_devin_death(sessions, 123, config=OrchestratorConfig()), sessions


def test_a_dead_devin_session_is_classified_and_blocks_the_harness(tmp_path: Path) -> None:
    (kind, until), _ = _death(tmp_path)

    assert (kind, until) == (HARNESS_AUTH_FAILED_KIND, None)  # no per-repo window is armed
    got = block()
    assert got.evidence == NOT_LOGGED_IN
    assert got.affected == (("repo", 123),)
    assert got.until > datetime.now(UTC)
    assert "devin-shell|*" in ledger._read_raw(ledger.ledger_path())["restrictions"]


def test_a_host_io_worktree_death_is_attributed_to_its_repo(tmp_path: Path) -> None:
    volume = tmp_path / "volume" / "repo" / layout.WORKTREES_DIRNAME

    (kind, _), _ = _death(tmp_path, worktrees_root=volume)

    assert kind == HARNESS_AUTH_FAILED_KIND
    assert block().affected == (("repo", 123),)


@pytest.mark.parametrize(
    ("worktree", "expected"),
    [
        pytest.param(
            Path("r") / "myrepo" / ".var" / "charlie-work" / "worktrees" / "agent-issue-9",
            "myrepo",
            id="default-layout-state-dir-leaf-is-not-the-repo",
        ),
        pytest.param(
            Path("r") / "myrepo" / ".var" / "worktrees" / "agent-issue-9",
            "myrepo",
            id="default-layout-with-state-dir-var",
        ),
        pytest.param(
            Path("v") / "fleet" / "myrepo" / "worktrees" / "agent-issue-9",
            "myrepo",
            id="host-io-volume",
        ),
        pytest.param(
            Path("v") / ".var" / "fleet" / "myrepo" / "worktrees" / "agent-issue-9",
            "myrepo",
            id="host-io-volume-under-a-var-directory",
        ),
        pytest.param(Path("v") / "elsewhere" / "agent-issue-9", None, id="no-worktrees-dir"),
        pytest.param(None, None, id="no-worktree-path"),
    ],
)
def test_the_repo_label_is_read_off_each_worktree_layout(
    tmp_path: Path, worktree: Path | None, expected: str | None
) -> None:
    payload = {} if worktree is None else {"worktree_path": str(tmp_path / worktree)}
    unlabelled_sessions = tmp_path / "elsewhere" / "sessions"  # nothing to fall back on

    assert harness_auth_block.session_repo_label(payload, unlabelled_sessions) == expected


def test_the_repo_label_falls_back_to_the_var_root_the_sessions_live_under(tmp_path: Path) -> None:
    sessions = layout.sessions_dir_default(layout.default_state_root(tmp_path / "myrepo"))
    unknown_worktree = {"worktree_path": str(tmp_path / "scratch" / "agent-issue-9")}

    assert harness_auth_block.session_repo_label(unknown_worktree, sessions) == "myrepo"
    assert harness_auth_block.session_repo_label({}, sessions) == "myrepo"


@pytest.mark.parametrize(
    ("terminal", "terminal_pid", "classified"),
    [
        pytest.param(None, None, True, id="control-no-terminal-record"),
        pytest.param({"exit_code": 1}, None, True, id="control-nonzero-exit"),
        pytest.param({"exit_code": 0}, None, False, id="zero-exit"),
        pytest.param({"exit_code": None}, None, False, id="unknown-exit"),
        pytest.param({"exit_code": 0, "worker_outcome": {"x": 1}}, None, False, id="completion"),
        pytest.param(
            {"exit_code": 0, "worker_outcome": {"x": 1}}, 7, True, id="other-pids-record"
        ),
    ],
)
def test_terminal_record_guards(
    tmp_path: Path, terminal: dict[str, Any] | None, terminal_pid: int | None, classified: bool
) -> None:
    (kind, _), _ = _death(tmp_path, terminal=terminal, terminal_pid=terminal_pid)

    assert (kind == HARNESS_AUTH_FAILED_KIND) is classified
    assert (ledger.read_harness_block(HARNESS) is not None) is classified


def test_a_completion_transcript_quoting_the_error_never_blocks(tmp_path: Path) -> None:
    transcript = f"worker summary\n{NOT_LOGGED_IN}\nall done\n"
    (control, _), _ = _death(tmp_path / "a", transcript)
    (kind, _), _ = _death(
        tmp_path / "b", transcript, terminal={"exit_code": 0, "worker_outcome": {"ok": True}}
    )
    assert control == HARNESS_AUTH_FAILED_KIND, "control: with no terminal record the line counts"
    assert kind is None


@pytest.mark.parametrize(
    ("prior_age", "hits"),
    [
        pytest.param(None, 1, id="arming-pass"),
        pytest.param(LAPSED, 2, id="control-lapsed-window-is-another-hit"),
    ],
)
def test_a_wave_dying_together_is_one_hit_charged_to_no_issue(
    tmp_path: Path,
    fleet: Path,
    launched: list,
    prior_age: timedelta | None,
    hits: int,
) -> None:
    """Three workers launched in one pass all die of the auth line: the first death
    arms the block, the other two land inside its live window -- one hit, three
    affected issues, and none of the three deaths is charged to a cap. A block whose
    window had already lapsed is the control: the same wave is then a second hit."""
    app, paths = make_app(tmp_path, "a", fleet)
    register(fleet, paths)
    app.dispatch(limit=3)
    assert sorted(n for n, _ in launched) == list(ISSUES), "the whole wave launched together"
    if prior_age is not None:
        arm(age=prior_age, repo=None, issue=None)

    kinds = [
        classify_devin_death(
            dead_devin_session(tmp_path, NOT_LOGGED_IN + "\n", issue=n, pid=4000 + n),
            n,
            config=app.config,
        )[0]
        for n in ISSUES
    ]

    assert kinds == [HARNESS_AUTH_FAILED_KIND] * len(ISSUES)
    for kind in kinds:
        assert is_provider_throttle_failure(kind)
        assert is_provider_throttle_rework_death(kind)
        assert kind in PROVIDER_THROTTLE_EXEMPT_KINDS
        assert persisted_failure({"dead_worker_failure_kind": kind}).is_throttle
        verdict = redispatch_verdict(
            ("a", "b", "c"), kind, now=datetime.now(UTC), max_auto_redispatch=3
        )
        assert not verdict.escalate and len(verdict.redispatch_at) == 3
    got = block()
    assert got.consecutive_hits == hits
    assert got.affected == tuple(("repo", n) for n in ISSUES)
    assert got.evidence == NOT_LOGGED_IN


# --- accounting: charged to no cap, in any lane ----------------------------------------


@pytest.mark.parametrize(
    ("kind", "exempt"),
    [(HARNESS_AUTH_FAILED_KIND, True), ("stalled", False)],
    ids=["auth", "stall"],
)
def test_every_cap_predicate_agrees(kind: str, exempt: bool) -> None:
    assert is_provider_throttle_failure(kind) is exempt
    assert is_provider_throttle_rework_death(kind) is exempt
    assert (kind in PROVIDER_THROTTLE_EXEMPT_KINDS) is exempt
    assert persisted_failure({"dead_worker_failure_kind": kind}).is_throttle is exempt


@pytest.mark.parametrize(
    ("kind", "escalates"), [(HARNESS_AUTH_FAILED_KIND, False), ("stalled", True)]
)
def test_the_dead_session_relabel_lane_charges_nothing(kind: str, escalates: bool) -> None:
    now = datetime.now(UTC)
    verdict = redispatch_verdict(("a", "b", "c"), kind, now=now, max_auto_redispatch=3)
    assert verdict.escalate is escalates
    assert len(verdict.redispatch_at) == (4 if escalates else 3)


@pytest.mark.parametrize(
    ("kind", "exceeded"), [(HARNESS_AUTH_FAILED_KIND, False), ("stalled", True)]
)
def test_the_orphan_redispatch_lane_charges_nothing(kind: str, exceeded: bool) -> None:
    verdict = decide_redispatch_cap(
        {"dead_worker_failure_kind": kind},
        head_details={"remote_head_sha": "a", "local_head_sha": "b"},
        window_minutes=240,
        max_auto_redispatch=0,
        stamp="t",
        now=datetime.now(UTC),
    )
    assert verdict.exceeded is exceeded


# The rework-restore lane, the no-op rework cap and ``worker_death_loop`` run through the
# real sweep in tests/test_rework_throttle_exemption.py, whose kind parametrization now
# includes ``harness_auth_failed``.


# --- block, fallback and the permit ----------------------------------------------------


def test_only_a_harness_auth_death_writes_a_harness_wide_record() -> None:
    stamp = ledger.session_stamp("worker", HARNESS, "swe-2", 0)
    payload = {"issue_number": 9, "role_entry": stamp}
    until = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
    assert ledger.record_classified_death(payload, "quota_exhausted", until, source="t")
    assert ledger.list_harness_blocks() == [], "a quota death restricts one model only"
    assert (HARNESS, "swe-2") in ledger.load_restrictions()
    assert ledger.record_classified_death(payload, HARNESS_AUTH_FAILED_KIND, None, source="t")
    assert [b.harness for b in ledger.list_harness_blocks()] == [HARNESS]


def test_a_chain_moves_past_a_blocked_harness_to_its_fallback() -> None:
    assert role_selection.select_for_launch(CHAIN).entry == DEVIN_PRIMARY, "control"
    arm()

    selection = role_selection.select_for_launch(CHAIN)

    assert selection.entry == CLAUDE_FALLBACK
    assert selection.is_fallback
    assert selection.skipped[0].to_payload()["harness_auth"] is True


def test_length_one_and_exhausted_chains_report_the_block() -> None:
    assert not role_selection.select_for_launch((DEVIN_PRIMARY,)).exhausted, "control"
    claude = (CLAUDE_FALLBACK,)
    got = arm()

    solo = role_selection.select_for_launch((DEVIN_PRIMARY,))
    both_devin = role_selection.select_for_launch((DEVIN_PRIMARY, RoleEntry(HARNESS, "swe-1-6")))

    for selection in (solo, both_devin):
        assert selection.exhausted and selection.harness_auth_blocked
        assert selection.report_fields()["chain_retry_at"].startswith(got.until.strftime("%Y-"))
    assert not role_selection.select_for_launch(claude).exhausted, "other harnesses unaffected"


@pytest.mark.parametrize("cores", [2, 4, 8])
def test_the_fixture_repo_launches_every_issue_whatever_the_hosts_core_count(
    tmp_path: Path, fleet: Path, launched: list, monkeypatch: pytest.MonkeyPatch, cores: int
) -> None:
    """The host-load cap defaults to ``cpu_count // 2``: a 4-vCPU CI runner clamped the
    three-issue wave to two launches while a 16-core workstation never noticed. 8 cores
    (cap 4, above the wave) is the control."""
    monkeypatch.setattr(os, "cpu_count", lambda: cores)
    app, _ = make_app(tmp_path, "a", fleet)

    app.dispatch(limit=3)

    assert len(launched) == len(ISSUES)


@LANES
def test_a_blocked_worker_chain_defers_with_no_claim(
    tmp_path: Path, fleet: Path, launched: list, lane: str
) -> None:
    app, paths = make_app(tmp_path, "a", fleet, rework=lane == "rework")
    register(fleet, paths)
    run_lane(app, lane)
    assert len(launched) == len(ISSUES), "control: unblocked, the lane launches"
    in_progress = app.config.labels.in_progress
    assert {n for n, label in app.gh.labels_added if label == in_progress} == set(ISSUES), (
        "control: an unblocked launch claims each issue with the in-progress label"
    )
    launched.clear()
    app, paths = make_app(tmp_path, "b", fleet, rework=lane == "rework")
    arm()

    result = run_lane(app, lane)

    assert launched == []
    assert result.data["deferred_reason"] == "harness_auth_blocked"
    assert result.data["chain_retry_at"]
    statuses = {i.get("status") for i in load_state(paths.state_file)["issues"].values()}
    assert statuses <= {"rework_requested"}, "no issue was claimed"
    assert app.gh.labels_added == [], "no agent:in-progress label was written"
    assert not layout.fleet_pause_path(override=str(fleet)).exists(), "fleet pause is not used"


@LANES
def test_a_fallback_chain_launches_on_the_other_harness(
    tmp_path: Path, fleet: Path, launched: list, lane: str
) -> None:
    chained = WorkerRoleConfig(
        harness=HARNESS, model=DEVIN_PRIMARY.model, fallbacks=(CLAUDE_FALLBACK,)
    )
    app, _ = make_app(tmp_path, "a", fleet, rework=lane == "rework", worker=chained)
    arm()

    run_lane(app, lane)

    assert launched and {adapter for _, adapter in launched} == {"claude-code"}


def test_the_reviewer_chain_defers_the_same_way(tmp_path: Path) -> None:
    from _review_fixtures import _dispatch_reviews_app, _write_review_packet

    pr = {
        "number": 100,
        "title": "Fix #10",
        "url": "https://example.test/pull/100",
        "headRefName": "agent/issue-10-fix",
        "baseRefName": "main",
        "headRefOid": "sha-100",
        "mergeStateStatus": "CLEAN",
        "body": "Closes #10",
        "labels": [],
        "isCrossRepository": False,
        "state": "OPEN",
    }
    app = _dispatch_reviews_app(tmp_path, prs=[pr])
    _write_review_packet(tmp_path, 100, "sha-100")
    arm(app.config.reviewer.harness)

    result = app.dispatch_reviews()

    assert result.data["deferred_reason"] == "harness_auth_blocked"
    assert result.data["launched_count"] == 0
    assert result.data["chain_retry_at"]
