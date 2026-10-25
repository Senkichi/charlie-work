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
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from _harness_auth_fixtures import (
    CLAUDE_FALLBACK,
    DEVIN_PRIMARY,
    EVIDENCE,
    HARNESS,
    INVALID_KEY,
    ISSUES,
    LAPSED,
    NOT_LOGGED_IN,
    arm,
    block,
    dead_devin_session,
    fake_launch,
    make_app,
    make_config,
    register,
)
from charlie_work import adapter_fate_profile, harness_auth_block, layout, role_selection
from charlie_work import fleet_provider_throttle as fpt
from charlie_work import role_quota_ledger as ledger
from charlie_work.config import OrchestratorConfig, WorkerRoleConfig
from charlie_work.dead_worker_sweep.decide_dead_sessions import redispatch_verdict
from charlie_work.dead_worker_sweep.decide_redispatch import decide_redispatch_cap
from charlie_work.devin_failure_classification import (
    update_session_record_with_failure_classification as classify_devin_death,
)
from charlie_work.instrumentation import query_events
from charlie_work.rework_attempt_exemption import (
    PROVIDER_THROTTLE_EXEMPT_KINDS,
    is_provider_throttle_rework_death,
)
from charlie_work.role_chain import RoleEntry
from charlie_work.state import load_state
from charlie_work.subprocess_runner import RunResult, run_captured
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


def _lane(app: Any, lane: str) -> Any:
    return app.dispatch(limit=3) if lane == "dispatch" else app.dispatch_rework(limit=3)


LANES = pytest.mark.parametrize("lane", ["dispatch", "rework"])


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


CHAIN = (DEVIN_PRIMARY, CLAUDE_FALLBACK)


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
    _lane(app, lane)
    assert len(launched) == len(ISSUES), "control: unblocked, the lane launches"
    launched.clear()
    app, paths = make_app(tmp_path, "b", fleet, rework=lane == "rework")
    arm()

    result = _lane(app, lane)

    assert launched == []
    assert result.data["deferred_reason"] == "harness_auth_blocked"
    assert result.data["chain_retry_at"]
    statuses = {i.get("status") for i in load_state(paths.state_file)["issues"].values()}
    assert statuses <= {"rework_requested"}, "no issue was claimed"
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

    _lane(app, lane)

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


# --- the launch gate: Stage A pending, one probe, survival, re-arm ----------------------


def test_the_gate_is_closed_until_stage_a_passes_then_admits_one_probe(fleet: Path) -> None:
    assert harness_auth_block.decide_for_harness(HARNESS, fleet_dir_override=str(fleet)) is None
    arm(age=LAPSED)

    pending = harness_auth_block.decide_for_harness(HARNESS, fleet_dir_override=str(fleet))
    assert pending.action == "defer_throttled" and pending.harness_auth
    assert pending.deferred_reason == "harness_auth_blocked"

    ledger.mark_stage_a_green(HARNESS)
    probe = harness_auth_block.decide_for_harness(HARNESS, fleet_dir_override=str(fleet))
    assert probe.action == "admit_one"
    assert probe.probe_keys == (harness_auth_block.probe_key(HARNESS),)


def test_a_live_window_defers_even_when_stage_a_was_green(fleet: Path) -> None:
    arm()
    ledger.mark_stage_a_green(HARNESS)
    decision = harness_auth_block.decide_for_harness(HARNESS, fleet_dir_override=str(fleet))
    assert decision.action == "defer_throttled"


@LANES
def test_the_probe_is_one_launch_stamped_under_both_keys(
    tmp_path: Path, fleet: Path, launched: list, lane: str
) -> None:
    app, paths = make_app(tmp_path, "a", fleet, rework=lane == "rework")
    register(fleet, paths)
    arm(age=LAPSED)
    ledger.mark_stage_a_green(HARNESS)

    _lane(app, lane)

    assert len(launched) == 1
    stamp = fpt._read_json(fpt.resume_probe_path(str(fleet)))
    assert {harness_auth_block.probe_key(HARNESS), "devin"} <= set(stamp)
    again = _lane(app, lane)
    assert len(launched) == 1
    assert again.data["deferred_reason"] == "provider_resume_staggered"


def test_a_probe_that_outlives_survival_reopens_the_harness(
    tmp_path: Path, fleet: Path, launched: list
) -> None:
    app, paths = make_app(tmp_path, "a", fleet)
    register(fleet, paths)
    arm(age=LAPSED)
    ledger.mark_stage_a_green(HARNESS)
    key = harness_auth_block.probe_key(HARNESS)
    past = datetime.now(UTC) - timedelta(seconds=fpt.RESUME_SURVIVAL_SECONDS + 5)
    fpt.note_probe_launch(key, [(os.getpid(), None)], fleet_dir_override=str(fleet), now=past)

    state_path = layout.state_file_path(fleet)
    done = harness_auth_block.run_fleet_pass(
        app.config, fleet_dir_override=str(fleet), fleet_state_path=state_path, notify_config=None
    )

    assert f"{HARNESS}:resumed" in done
    assert not block().episode_open
    (resumed,) = query_events(state_path, kind="harness_auth_resumed")
    assert resumed["level"] == "info" and resumed["payload"]["harness"] == HARNESS
    app.dispatch(limit=3)
    assert len(launched) == len(ISSUES), "the harness is open again"
    assert (
        harness_auth_block.run_fleet_pass(
            app.config,
            fleet_dir_override=str(fleet),
            fleet_state_path=state_path,
            notify_config=None,
        )
        == []
    ), "an ended episode ends once"
    assert len(query_events(state_path, kind="harness_auth_resumed")) == 1


def test_a_probe_still_inside_the_survival_period_keeps_the_harness_closed(
    tmp_path: Path, fleet: Path
) -> None:
    app, paths = make_app(tmp_path, "a", fleet)
    register(fleet, paths)
    arm(age=LAPSED)
    ledger.mark_stage_a_green(HARNESS)
    key = harness_auth_block.probe_key(HARNESS)
    fpt.note_probe_launch(key, [(os.getpid(), None)], fleet_dir_override=str(fleet))

    done = harness_auth_block.run_fleet_pass(
        app.config,
        fleet_dir_override=str(fleet),
        fleet_state_path=layout.state_file_path(fleet),
        notify_config=None,
    )

    assert f"{HARNESS}:resumed" not in done
    assert block().episode_open


def test_a_probe_that_dies_of_the_signature_rearms_with_backoff_and_no_second_alert(
    tmp_path: Path, fleet: Path, digests: list
) -> None:
    state_path = layout.state_file_path(fleet)
    arm(age=LAPSED)
    first = _pass(fleet, state_path)
    assert first == [f"{HARNESS}:alerted", f"{HARNESS}:stage_a_green"]
    sessions = dead_devin_session(tmp_path, NOT_LOGGED_IN + "\n")

    classify_devin_death(sessions, 123, config=OrchestratorConfig())  # the probe's death

    rearmed = block()
    assert rearmed.consecutive_hits == 2
    assert timedelta(minutes=29) < rearmed.until - datetime.now(UTC) <= timedelta(minutes=30)
    assert rearmed.stage_a_green_at is None
    assert _pass(fleet, state_path) == []
    assert len(query_events(state_path, kind="harness_auth_blocked")) == 1
    assert len(digests) == 1


# --- alerting ----------------------------------------------------------------------------


@pytest.fixture
def digests(monkeypatch: pytest.MonkeyPatch) -> list:
    sent: list = []
    monkeypatch.setattr(harness_auth_block, "emit_digest", lambda _cfg, d: sent.append(d))
    return sent


def _pass(fleet: Path, state_path: Path, *, stage_a: Any = None, **kw: Any) -> list[str]:
    stage = stage_a or (lambda *_a, **_k: harness_auth_block.StageAResult(True, "ok"))
    return harness_auth_block.run_fleet_pass(
        make_config(),
        fleet_dir_override=str(fleet),
        fleet_state_path=state_path,
        notify_config=object(),
        stage_a=stage,
        **kw,
    )


def test_the_episode_alerts_once(fleet: Path, digests: list) -> None:
    state_path = layout.state_file_path(fleet)
    assert _pass(fleet, state_path) == [], "no block, nothing to say"
    arm(repo="o/r1", issue=124)
    arm(repo="o/r0", issue=123)

    assert _pass(fleet, state_path) == [f"{HARNESS}:alerted"]
    assert _pass(fleet, state_path) == []

    (event,) = query_events(state_path, kind="harness_auth_blocked")
    assert event["level"] == "error"
    payload = event["payload"]
    assert payload["harness"] == HARNESS and payload["evidence"] == EVIDENCE
    assert payload["affected"] == {"o/r0": [123], "o/r1": [124]}
    assert payload["affected_repos"] == ["o/r0", "o/r1"]
    assert payload["affected_issues"] == [123, 124]
    assert payload["next_probe_at"].endswith("Z")
    (digest,) = digests
    (entry,) = digest.transitions
    assert entry.health == "HARNESS_AUTH_BLOCKED" and entry.adapter_kind == HARNESS
    assert entry.last_log_line == EVIDENCE
    assert "#123" in entry.terminal_reason and "#124" in entry.terminal_reason


def test_a_resumed_harness_that_fails_again_is_a_new_alerted_episode(
    fleet: Path, digests: list
) -> None:
    state_path = layout.state_file_path(fleet)
    arm()
    _pass(fleet, state_path)
    ledger.mark_harness_resumed(HARNESS)

    again = arm(age=timedelta(0))

    assert again.consecutive_hits == 1 and again.alerted_at is None
    assert _pass(fleet, state_path) == [f"{HARNESS}:alerted"]
    assert len(digests) == 2


def test_a_dry_run_pass_does_nothing(fleet: Path, digests: list) -> None:
    state_path = layout.state_file_path(fleet)
    arm()
    assert _pass(fleet, state_path, dry_run=True) == []
    assert digests == [] and query_events(state_path, kind="harness_auth_blocked") == []
    assert block().alerted_at is None


# --- resume, stage A ------------------------------------------------------------------------


class _Runner:
    def __init__(self, result: RunResult) -> None:
        self.result, self.calls = result, []

    def __call__(self, command: list[str], **kwargs: Any) -> RunResult:
        self.calls.append((command, kwargs))
        return self.result


def _stage_a(fleet: Path, result: RunResult, config: Any = None) -> tuple[Any, _Runner]:
    runner = _Runner(result)
    got = harness_auth_block.run_stage_a(
        HARNESS, config or make_config(), fleet_dir_override=str(fleet), runner=runner
    )
    return got, runner


LOGGED_IN = "Logged in as op@example.test\nTeam: ok\n"
TEAM_FETCH_FAILED = (
    "Logged in as op@example.test\nFailed to fetch team settings: invalid api key\n"
)


@pytest.mark.parametrize(
    ("result", "green"),
    [
        pytest.param(RunResult(0, LOGGED_IN, ""), True, id="control-healthy"),
        pytest.param(RunResult(0, TEAM_FETCH_FAILED, ""), False, id="exit-0-with-fetch-error"),
        pytest.param(RunResult(0, LOGGED_IN + INVALID_KEY, ""), False, id="exit-0-invalid-key"),
        pytest.param(RunResult(1, "", "Not logged in"), False, id="not-logged-in"),
        pytest.param(RunResult(0, "Not logged in\n", ""), False, id="exit-0-not-logged-in"),
        pytest.param(RunResult(2, LOGGED_IN, ""), False, id="nonzero-exit"),
        pytest.param(RunResult(None, "", "", timed_out=True), False, id="timeout"),
        pytest.param(RunResult(None, "", "", error="not found"), False, id="missing-binary"),
    ],
)
def test_stage_a_does_not_trust_the_exit_code(fleet: Path, result: RunResult, green: bool) -> None:
    got, runner = _stage_a(fleet, result)

    assert got.green is green, got.detail
    ((command, kwargs),) = runner.calls
    assert command[1:] == ["auth", "status"]
    assert 0 < kwargs["timeout_seconds"] <= harness_auth_block.STAGE_A_TIMEOUT_SECONDS


def test_stage_a_runs_under_the_sanitized_env_not_the_ambient_one(
    fleet: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[dict] = []
    real = harness_auth_block.sanitize_env

    def spy(target: Path, **kwargs: Any) -> dict[str, str]:
        assert "config" in kwargs, "the sanitize_env AST guard requires config="
        env = real(target, **kwargs)
        seen.append(env)
        return env

    monkeypatch.setattr(harness_auth_block, "sanitize_env", spy)
    monkeypatch.setenv("ACP_BACKEND", "ide-terminal")

    _, runner = _stage_a(fleet, RunResult(0, LOGGED_IN, ""))

    (env,) = seen
    assert "ACP_BACKEND" not in env
    assert runner.calls[0][1]["base_env"] is env
    assert Path(runner.calls[0][1]["cwd"]) != Path.cwd()


def test_run_captured_base_env_replaces_the_ambient_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("H2810_AMBIENT", "leaked")
    code = "import os; print(os.environ.get('H2810_AMBIENT'), os.environ.get('H2810_BASE'))"
    command = [sys.executable, "-c", code]

    layered = run_captured(
        command, cwd=tmp_path, timeout_seconds=30, extra_env={"H2810_BASE": "1"}
    )
    replaced = run_captured(
        command,
        cwd=tmp_path,
        timeout_seconds=30,
        base_env={k: v for k, v in os.environ.items() if k != "H2810_AMBIENT"}
        | {"H2810_BASE": "1"},
    )

    assert layered.stdout.split() == ["leaked", "1"], "control: layering keeps the ambient var"
    assert replaced.stdout.split() == ["None", "1"]


def test_a_missing_binary_is_red_through_the_real_runner(fleet: Path) -> None:
    config = OrchestratorConfig(
        devin=make_config().devin.__class__(shell_command=("definitely-not-a-real-binary-2810",))
    )
    got = harness_auth_block.run_stage_a(HARNESS, config, fleet_dir_override=str(fleet))
    assert got.green is False


def test_a_lapsed_window_with_green_stage_a_is_stamped_for_one_probe(
    fleet: Path, digests: list
) -> None:
    arm(age=LAPSED)

    assert _pass(fleet, layout.state_file_path(fleet)) == [
        f"{HARNESS}:alerted",
        f"{HARNESS}:stage_a_green",
    ]

    assert block().stage_a_green_at is not None


def test_a_lapsed_window_with_red_stage_a_rearms_without_escalating(
    fleet: Path, digests: list
) -> None:
    start = arm(age=LAPSED)

    def red(*_args: Any, **_kwargs: Any) -> harness_auth_block.StageAResult:
        return harness_auth_block.StageAResult(False, "Not logged in")

    done = _pass(fleet, layout.state_file_path(fleet), stage_a=red)

    assert done == [f"{HARNESS}:alerted", f"{HARNESS}:stage_a_red"]
    after = block()
    assert after.consecutive_hits == start.consecutive_hits, "no launch died, no new hit"
    assert after.stage_a_green_at is None
    assert timedelta(minutes=14) < after.until - datetime.now(UTC) <= timedelta(minutes=15)
    assert harness_auth_block.decide_for_harness(HARNESS, fleet_dir_override=str(fleet)).deferred


def test_a_live_window_runs_no_stage_a(fleet: Path, digests: list) -> None:
    arm()

    def boom(*_args: Any, **_kwargs: Any) -> harness_auth_block.StageAResult:
        raise AssertionError("Stage A must wait for the window to lapse")

    assert _pass(fleet, layout.state_file_path(fleet), stage_a=boom) == [f"{HARNESS}:alerted"]


# --- kill switch ------------------------------------------------------------------------------


def test_with_the_switch_off_nothing_classifies_blocks_or_alerts(
    tmp_path: Path, fleet: Path, launched: list, digests: list
) -> None:
    off = OrchestratorConfig(fleet=make_config(enabled=False).fleet)
    sessions = dead_devin_session(tmp_path / "off", NOT_LOGGED_IN + "\n")
    assert classify_devin_death(sessions, 123, config=off) == (None, None)
    assert ledger.list_harness_blocks() == []
    sessions = dead_devin_session(tmp_path / "on", NOT_LOGGED_IN + "\n")
    assert classify_devin_death(sessions, 123, config=OrchestratorConfig())[0] == (
        HARNESS_AUTH_FAILED_KIND
    ), "control: the same death classifies with the switch on"

    state_path = layout.state_file_path(fleet)
    assert (
        harness_auth_block.run_fleet_pass(
            off, fleet_dir_override=str(fleet), fleet_state_path=state_path, notify_config=object()
        )
        == []
    )
    assert digests == [] and block().alerted_at is None


def test_with_the_switch_off_a_recorded_block_is_ignored_everywhere(
    tmp_path: Path, fleet: Path, launched: list
) -> None:
    arm()
    assert role_selection.select_for_launch((DEVIN_PRIMARY,)).exhausted, "control"
    assert not role_selection.select_for_launch((DEVIN_PRIMARY,), harness_auth=False).exhausted
    assert role_selection.select_for_launch(CHAIN, harness_auth=False).entry == DEVIN_PRIMARY

    app, paths = make_app(tmp_path, "off", fleet, enabled=False)
    register(fleet, paths)

    app.dispatch(limit=3)
    assert len(launched) == len(ISSUES)
    assert fpt.decide_for_app(app).action == "open"
    on, _ = make_app(tmp_path, "on", fleet)
    assert fpt.decide_for_app(on).action == "defer_throttled", (
        "control: the switch is what opens it"
    )
