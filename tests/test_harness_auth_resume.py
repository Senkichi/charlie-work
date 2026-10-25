"""Issue #2810: the harness-auth block's lifecycle -- gate, alert, resume, kill switch.

Companion to ``tests/test_harness_auth_failure.py`` (which pins the classifier, the
accounting and the block/fallback selection). Here the block is already armed and the
tests follow it to its end: the launch gate admits exactly one probe once Stage A is
green, the episode alerts once, a probe that survives reopens the harness, and
``fleet.harness_auth_block_enabled: false`` removes the block from every read. Every
negative assertion sits next to a planted positive control.
"""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from _harness_auth_fixtures import (
    CHAIN,
    DEVIN_PRIMARY,
    EVIDENCE,
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
    make_config,
    register,
    run_lane,
)
from charlie_work import harness_auth_block, layout, role_selection
from charlie_work import fleet_provider_throttle as fpt
from charlie_work import role_quota_ledger as ledger
from charlie_work.config import OrchestratorConfig
from charlie_work.devin_failure_classification import (
    update_session_record_with_failure_classification as classify_devin_death,
)
from charlie_work.instrumentation import query_events
from charlie_work.subprocess_runner import RunResult, run_captured
from charlie_work.throttle_signatures import HARNESS_AUTH_FAILED_KIND


@pytest.fixture
def launched(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, str]]:
    """Capture ``(issue, adapter)`` of every launch; each succeeds with a live pid."""
    calls: list[tuple[int, str]] = []
    monkeypatch.setattr("charlie_work.adapters.dispatch_sessions", fake_launch(calls))
    return calls


@pytest.fixture
def fleet(tmp_path: Path) -> Path:
    return tmp_path / "fleet"


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

    run_lane(app, lane)

    assert len(launched) == 1
    stamp = fpt._read_json(fpt.resume_probe_path(str(fleet)))
    assert {harness_auth_block.probe_key(HARNESS), "devin"} <= set(stamp)
    again = run_lane(app, lane)
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
