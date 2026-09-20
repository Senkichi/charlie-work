"""Stall/dead-lane composition and stall-reap classification: once-per-pass deferral counter, phantom post-mortem sidecar composition, throttle-signature-first classification, stalled fallback, and stall-lane api-budget / provider-auth outcomes.

Split out of ``tests/test_charlie_work.py`` (issue #1551, Track-1
wave 5/8).
"""

from __future__ import annotations

import json
from datetime import (
    UTC,
    datetime,
    timedelta,
)
from pathlib import Path
from unittest.mock import patch
import pytest
from _dispatch_fixtures import _make_stalled_sidecar
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401
from _fakes_github import FakeGitHub
from _rework_dispatch_fixtures import _wg
from charlie_work.config import (
    DevinConfig,
    OrchestratorConfig,
    PostMortemConfig,
    WatchdogConfig,
    WorkerRoleConfig,
)
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state
from charlie_work.workflow import OrchestratorApp


@pytest.mark.real_activity_probe_live
def test_stall_and_dead_lane_increment_deferral_counter_at_most_once_per_pass(
    tmp_path: Path,
) -> None:
    """Issue #343 Finding 2: the stall lane (_detect_and_handle_stalled_sessions)
    and the dead lane (_classify_dead_sessions_and_update_throttle_state) both
    corroborate a not-alive, pid-bearing, error-free worker against the same
    real-activity probe, and both used to unconditionally persist Signal-1's
    inconclusive-probe deferral counter. Within a single ``loop()`` pass that
    double-incremented the counter (0->1 in the stall lane, then re-read and
    ->2 in the dead lane) -- halving the effective deferral grace period, and
    the very mechanism that opens Finding 1's pass-2 phantom-sidecar window.

    ``loop()`` (workflow.py, ~4100/~4119) always runs the stall lane
    immediately before the dead lane and passes the dead lane
    ``persist_inconclusive_probe_counter=False`` for exactly this reason --
    this test drives both lanes once in that same order, with that same
    argument, and pins the counter at exactly 1 after the pass. Every other
    caller (every existing standalone unit test, plus dispatch()/
    dispatch_rework(), which never call the dead lane at all) leaves the
    dead lane's default (True) alone, so the stall lane remains the correct
    sole writer when the dead lane doesn't run in the same pass -- see
    ``test_detect_and_handle_stalled_sessions_inconclusive_probe_deferred_
    then_escalated`` in test_worker.py, which pins that standalone case.

    MUTATION GATE: dropping the ``persist_inconclusive_probe_counter=False``
    argument from this call (i.e. reverting to the unconditional write) makes
    this test fail -- the counter would read 2, not 1.
    """
    from charlie_work.devin_shell import SessionRecord
    from charlie_work.devin_shell import _sidecar_path as devin_sidecar_path
    from charlie_work.post_mortem import ActivitySource, RealActivityProbe
    from charlie_work.workflow import (
        _classify_dead_sessions_and_update_throttle_state,
        _detect_and_handle_stalled_sessions,
    )

    config = OrchestratorConfig(
        post_mortem=PostMortemConfig(db_path=str(tmp_path / "no-such-sessions.db")),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.issues = [
        {
            "number": 343,
            "title": "Double-increment guard",
            "url": "https://example.test/issues/343",
            "body": "x",
            "labels": [{"name": config.labels.in_progress}],
        }
    ]
    fake_gh.prs = []

    sessions_dir = paths.root / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    log_path = sessions_dir / "issue-343.log"
    log_path.write_text("Working...\n", encoding="utf-8")

    sidecar_path = devin_sidecar_path(sessions_dir, 343)
    record = SessionRecord(
        issue_number=343,
        branch="agent/issue-343-x",
        worktree_path=str(tmp_path / "worktree-343"),
        prompt_path="/tmp/prompt.md",
        command=("devin", "--prompt-file", "/tmp/prompt.md"),
        pid=54321,
        started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error=None,
        process_start_time=1_700_000_000.0,
    )
    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    inconclusive_probe = RealActivityProbe(
        sources=(
            ActivitySource(
                name="devin_per_pid_log",
                timestamp=None,
                staleness_seconds=None,
                error="sessions.db unavailable",
            ),
        )
    )

    with (
        patch("charlie_work.worker.is_session_alive", return_value=False),
        patch("charlie_work.worker.real_activity_probe_for", return_value=inconclusive_probe),
    ):
        # loop() order and arguments: stall lane runs before the dead lane,
        # which is told not to persist the counter itself this pass.
        _detect_and_handle_stalled_sessions(
            sessions_dir, paths.state_file, config, write_gate=_wg(paths.state_file)
        )
        _classify_dead_sessions_and_update_throttle_state(
            sessions_dir,
            paths.state_file,
            fake_gh,
            config,
            persist_inconclusive_probe_counter=False,
            write_gate=_wg(paths.state_file),
        )

    assert sidecar_path.exists(), "sidecar must be RETAINED when the probe is inconclusive"
    persisted = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert persisted.get("inconclusive_probe_deferred_count") == 1, (
        "counter must increment at most once per worker per pass, not twice"
    )


@pytest.mark.real_activity_probe_live
def test_stall_then_dead_lane_composition_survives_phantom_post_mortem_sidecar(
    tmp_path: Path,
) -> None:
    """Issue #343 Finding 1 (composition gap): every pre-existing test drives
    ``_classify_dead_sessions_and_update_throttle_state`` in isolation. In
    production, ``loop()`` always runs the stall lane
    (``_detect_and_handle_stalled_sessions``) immediately before the dead
    lane, and the stall lane can itself reach a DEAD verdict and write a
    ``issue-N.post-mortem.json`` sidecar (via ``classify_and_record``)
    without reaping the session sidecar (it never reaps -- only the dead lane
    does). Before the ``read_session_records`` fix, that leftover post-mortem
    file was misread by the devin glob (``issue-*.json``) as a bogus phantom
    ``SessionRecord(pid=None, log_path="")``. The phantom's ``pid is None``
    skips corroboration entirely and reaches ``reap_sidecar``, which resolves
    to the SAME path as the REAL ``issue-343.json`` sidecar and deletes it --
    even when the real worker's own corroboration correctly defers it as not
    (yet) provably dead.

    Setup: forces the stall lane to reach a DEAD verdict via an unconditional
    terminal-error-marker log line (Signal 2, which "bypasses corroboration
    and still returns DEAD immediately" per classify_worker_health's
    docstring) so it writes the post-mortem sidecar without needing to
    fabricate a diverging probe. The marker line is then removed so the two
    "real" passes that follow are driven by one ordinary inconclusive probe
    throughout, exactly like the simpler double-increment test above.

    Drives the stall lane then the dead lane, in ``loop()``'s own order and
    with its own ``persist_inconclusive_probe_counter=False`` argument,
    across two passes with the stale post-mortem sidecar from setup still on
    disk throughout. Asserts the REAL sidecar survives both passes (still
    deferred) and the deferral counter advances by exactly 1 per pass
    (0 -> 1 -> 2), pinning both Finding 1 (the phantom must never be read
    back as a session) and Finding 2 (at most one increment per worker per
    pass) together.

    MUTATION GATE: reverting either the ``read_session_records`` stem
    exclusion (Finding 1) or dropping
    ``persist_inconclusive_probe_counter=False`` from the dead lane call
    (Finding 2) makes this test fail -- the sidecar is deleted mid-pass, or
    the counter overshoots to 2/4 instead of 1/2.
    """
    from charlie_work.devin_shell import SessionRecord
    from charlie_work.devin_shell import _sidecar_path as devin_sidecar_path
    from charlie_work.post_mortem import ActivitySource, RealActivityProbe
    from charlie_work.workflow import (
        _classify_dead_sessions_and_update_throttle_state,
        _detect_and_handle_stalled_sessions,
    )

    config = OrchestratorConfig(
        post_mortem=PostMortemConfig(db_path=str(tmp_path / "no-such-sessions.db")),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.issues = [
        {
            "number": 343,
            "title": "Phantom post-mortem sidecar",
            "url": "https://example.test/issues/343",
            "body": "x",
            "labels": [{"name": config.labels.in_progress}],
        }
    ]
    fake_gh.prs = []

    sessions_dir = paths.root / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    log_path = sessions_dir / "issue-343.log"

    sidecar_path = devin_sidecar_path(sessions_dir, 343)
    record = SessionRecord(
        issue_number=343,
        branch="agent/issue-343-x",
        worktree_path=str(tmp_path / "worktree-343"),
        prompt_path="/tmp/prompt.md",
        command=("devin", "--prompt-file", "/tmp/prompt.md"),
        pid=54321,
        started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error=None,
        process_start_time=1_700_000_000.0,
    )
    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    inconclusive_probe = RealActivityProbe(
        sources=(
            ActivitySource(
                name="devin_per_pid_log",
                timestamp=None,
                staleness_seconds=None,
                error="sessions.db unavailable",
            ),
        )
    )

    def _run_stall_lane() -> None:
        with (
            patch("charlie_work.worker.is_session_alive", return_value=False),
            patch(
                "charlie_work.worker.real_activity_probe_for",
                return_value=inconclusive_probe,
            ),
        ):
            _detect_and_handle_stalled_sessions(
                sessions_dir, paths.state_file, config, write_gate=_wg(paths.state_file)
            )

    def _run_dead_lane() -> None:
        with (
            patch("charlie_work.worker.is_session_alive", return_value=False),
            patch(
                "charlie_work.worker.real_activity_probe_for",
                return_value=inconclusive_probe,
            ),
        ):
            _classify_dead_sessions_and_update_throttle_state(
                sessions_dir,
                paths.state_file,
                fake_gh,
                config,
                persist_inconclusive_probe_counter=False,
                write_gate=_wg(paths.state_file),
            )

    def _run_pass() -> None:
        # loop() order and arguments: stall lane runs before the dead lane,
        # which is told not to persist the counter itself.
        _run_stall_lane()
        _run_dead_lane()

    # Setup (not one of the two counted passes): a terminal-error-marker log
    # line makes the stall lane's classify_worker_health call return DEAD
    # unconditionally (Signal 2 bypasses corroboration), so it writes the
    # post-mortem sidecar without reaping (the stall lane never reaps). The
    # marker is cleared BEFORE the dead lane runs -- matching loop()'s own
    # sequential order, where nothing else touches the log between the two
    # calls -- so the dead lane's own corroboration this pass is driven by
    # the inconclusive probe alone; otherwise Signal 2 would ALSO fire there
    # and reap the sidecar during setup.
    log_path.write_text("Error: Agent error: fatal\n", encoding="utf-8")
    _run_stall_lane()
    log_path.write_text("Working...\n", encoding="utf-8")
    _run_dead_lane()

    post_mortem_path = sessions_dir / "issue-343.post-mortem.json"
    assert post_mortem_path.exists(), "setup precondition: stall lane must have written it"
    assert sidecar_path.exists(), "setup must not reap -- dead lane saw a clean log, deferred"

    # Pass 1: ordinary inconclusive probe: both lanes defer.
    _run_pass()

    assert sidecar_path.exists(), "real sidecar must survive pass 1 (deferred, not DEAD)"
    persisted = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert persisted.get("inconclusive_probe_deferred_count") == 1

    # Pass 2: inconclusive again; the stale post-mortem sidecar from setup
    # is still on disk.
    _run_pass()

    assert post_mortem_path.exists(), "post-mortem sidecars are never reaped by this code path"
    assert sidecar_path.exists(), "real sidecar must survive pass 2 (deferred, not DEAD)"
    persisted = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert persisted.get("inconclusive_probe_deferred_count") == 2, (
        "counter must advance by exactly 1 per pass across both passes"
    )


def test_stall_reap_classifies_rate_limit_before_stalled_fallback(tmp_path: Path) -> None:
    """Issue #246: a stall-killed worker whose log tail matches the rate-limit
    signature must be classified rate_limited (with throttled_until set from
    the parsed reset-in-N-minutes cooldown), not the hardcoded "stalled"
    fallback — otherwise the very next dispatch pass relaunches into the same
    live provider rate limit.
    """
    from datetime import UTC, datetime, timedelta
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    OrchestratorApp(tmp_path, paths, config, fake_gh)

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    sidecar, _log_file = _make_stalled_sidecar(
        sessions_dir,
        1034,
        log_text=(
            "Error: Agent error: Permission denied: Permission denied: Reached "
            "overall message rate limit. Please try again later. Your limit "
            "will reset in 7 minutes.\n"
        ),
    )

    before = datetime.now(UTC)
    with (
        patch("charlie_work.worker.is_session_alive", return_value=True),
        patch("charlie_work.write_gate.kill_process_tree", return_value=[99999]),
        patch("charlie_work.dead_worker_reap.sweep_orphan_processes", return_value=[]),
    ):
        from charlie_work.workflow import _detect_and_handle_stalled_sessions

        result = _detect_and_handle_stalled_sessions(
            sessions_dir, paths.state_file, config, write_gate=_wg(paths.state_file)
        )
    after = datetime.now(UTC)

    assert any(entry["issue"] == 1034 for entry in result)

    # Sidecar must be classified rate_limited, not the hardcoded "stalled"
    updated_sidecar = json.loads(sidecar.read_text(encoding="utf-8"))
    assert updated_sidecar["failure_kind"] == "rate_limited"

    # throttled_until must be persisted to state.json, roughly now + 7 minutes
    # plus the resume margin.
    state = load_state(paths.state_file)
    throttled_until = state.get("throttled_until")
    assert throttled_until is not None
    throttle_time = datetime.fromisoformat(throttled_until.replace("Z", "+00:00"))
    margin = timedelta(seconds=config.runtime.throttle_resume_margin_s)
    assert before + timedelta(minutes=6) <= throttle_time <= after + timedelta(minutes=8) + margin

    # The reap event must carry the resolved failure_kind.
    #
    # Issue #873: this fixture's log tail opens with a terminal error marker
    # ("Agent error: Permission denied"), so classify_worker_health returns
    # DEAD at its unconditional terminal-marker signal — not STALLED. (The
    # patch on charlie_work.worker.is_session_alive above is inert here:
    # classify_worker_health checks view.is_alive(), not that module-level
    # helper.) The reap event is therefore "session_exited", not
    # "session_stalled". This test always exercised the DEAD path; before
    # #873 both healths emitted the same kind, so nothing could tell.
    #
    # Note what this proves about session_exited: it means "the process is
    # gone", NOT "the worker succeeded". Here it is a worker that died on a
    # provider rate limit. The genuine-failure signal is not lost — the
    # dead-session lane still emits error-level session_failed_escalated /
    # session_failed_relabeled carrying this same failure_kind.
    events = state.get("events", [])
    exited_events = [e for e in events if e.get("kind") == "session_exited"]
    assert len(exited_events) == 1
    assert exited_events[0]["payload"]["failure_kind"] == "rate_limited"
    assert exited_events[0]["payload"]["worker_health"] == "DEAD"
    assert [e for e in events if e.get("kind") == "session_stalled"] == []


def test_stall_reap_classifies_quota_exhausted_before_stalled_fallback(tmp_path: Path) -> None:
    """Issue #246: quota-exhaustion signature in the log tail must classify as
    quota_exhausted with the fixed 24-hour cooldown, not "stalled".
    """
    from datetime import UTC, datetime, timedelta
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    OrchestratorApp(tmp_path, paths, config, fake_gh)

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    sidecar, _log_file = _make_stalled_sidecar(
        sessions_dir,
        2001,
        log_text="Error: daily usage quota has been exhausted.\n",
    )

    before = datetime.now(UTC)
    with (
        patch("charlie_work.worker.is_session_alive", return_value=True),
        patch("charlie_work.write_gate.kill_process_tree", return_value=[99999]),
        patch("charlie_work.dead_worker_reap.sweep_orphan_processes", return_value=[]),
    ):
        from charlie_work.workflow import _detect_and_handle_stalled_sessions

        _detect_and_handle_stalled_sessions(
            sessions_dir, paths.state_file, config, write_gate=_wg(paths.state_file)
        )

    updated_sidecar = json.loads(sidecar.read_text(encoding="utf-8"))
    assert updated_sidecar["failure_kind"] == "quota_exhausted"

    state = load_state(paths.state_file)
    throttled_until = state.get("throttled_until")
    assert throttled_until is not None
    throttle_time = datetime.fromisoformat(throttled_until.replace("Z", "+00:00"))
    margin = timedelta(seconds=config.runtime.throttle_resume_margin_s)
    assert before + timedelta(hours=23) <= throttle_time <= before + timedelta(hours=25) + margin

    events = state.get("events", [])
    stalled_events = [e for e in events if e.get("kind") == "session_stalled"]
    assert stalled_events[0]["payload"]["failure_kind"] == "quota_exhausted"
    # Issue #873: unlike the rate-limit sibling above, this fixture's log tail
    # carries no terminal error marker, so the worker is classified STALLED
    # (live-but-quiet) and keeps the error-level "session_stalled" kind. Pinned
    # explicitly so the two sibling tests document both sides of the split.
    assert stalled_events[0]["payload"]["worker_health"] == "STALLED"
    assert [e for e in events if e.get("kind") == "session_exited"] == []


def test_stall_reap_falls_back_to_stalled_when_no_throttle_signature(tmp_path: Path) -> None:
    """Issue #246: a stall-killed worker with a quiet log tail (no rate-limit
    or quota signature) still falls back to failure_kind "stalled", and
    throttled_until is left untouched.
    """
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    OrchestratorApp(tmp_path, paths, config, fake_gh)

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    sidecar, _log_file = _make_stalled_sidecar(
        sessions_dir, 3007, log_text="working on the issue, one moment...\n"
    )

    with (
        patch("charlie_work.worker.is_session_alive", return_value=True),
        patch("charlie_work.write_gate.kill_process_tree", return_value=[99999]),
        patch("charlie_work.dead_worker_reap.sweep_orphan_processes", return_value=[]),
    ):
        from charlie_work.workflow import _detect_and_handle_stalled_sessions

        _detect_and_handle_stalled_sessions(
            sessions_dir, paths.state_file, config, write_gate=_wg(paths.state_file)
        )

    updated_sidecar = json.loads(sidecar.read_text(encoding="utf-8"))
    assert updated_sidecar["failure_kind"] == "stalled"

    state = load_state(paths.state_file)
    assert state.get("throttled_until") is None

    events = state.get("events", [])
    stalled_events = [e for e in events if e.get("kind") == "session_stalled"]
    assert stalled_events[0]["payload"]["failure_kind"] == "stalled"
    # Issue #873: quiet log, no terminal marker -> STALLED, so this keeps the
    # error-level kind. Note failure_kind "stalled" here is the *fallback* used
    # when no throttle signature matched; it is not evidence of a hang, which
    # is exactly why worker_health (not failure_kind) is the field that
    # distinguishes a live-but-stuck worker from a vanished one.
    assert stalled_events[0]["payload"]["worker_health"] == "STALLED"


def test_stall_lane_api_budget_kill_over_cap(tmp_path: Path) -> None:
    """Issue #484 review finding: the in-flight budget-kill block in
    ``_detect_and_handle_stalled_sessions`` (workflow.py) fires for an api
    worker whose accumulated session cost exceeds ``max_usd_per_session``:
    the process tree is killed, ``failure_kind="budget_exceeded"`` is written
    directly to the api sidecar, and a ``session_budget_exceeded`` event is
    emitted. A wiring regression that drops this block leaves the sidecar
    unmarked and the event unemitted — this assertion fails.
    """
    import os as _os

    from _api_budget_fixtures import (
        api_provider,
        write_api_events,
        write_api_sidecar,
    )
    from charlie_work.config import ApiBudgetConfig, ApiWorkerConfig

    config = OrchestratorConfig(
        api_worker=ApiWorkerConfig(
            enabled=True,
            provider="example",
            providers={"example": api_provider()},
            budget=ApiBudgetConfig(max_usd_per_session=5.0),
        ),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
        post_mortem=PostMortemConfig(enabled=False),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    OrchestratorApp(tmp_path, paths, config, fake_gh)

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    write_api_sidecar(sessions_dir, 4801, provider="example", pid=99999)
    write_api_events(sessions_dir, 4801)  # 6.15 USD > 5.0 cap

    # Stale log mtime so classify_worker_health does not short-circuit on a
    # fresh log; the budget block fires regardless of health.
    log_path = sessions_dir / "issue-4801.claude.log"
    old_time = datetime.now(UTC) - timedelta(minutes=25)
    _os.utime(log_path, (old_time.timestamp(), old_time.timestamp()))

    killed_pids: list[int] = []
    with (
        patch("charlie_work.worker.is_worker_alive", return_value=True),
        patch(
            "charlie_work.write_gate.kill_process_tree",
            side_effect=lambda pid, *_a, **_kw: killed_pids.extend([pid]) or [pid],
        ),
        patch("charlie_work.dead_worker_reap.sweep_orphan_processes", return_value=[]),
    ):
        from charlie_work.workflow import _detect_and_handle_stalled_sessions

        result = _detect_and_handle_stalled_sessions(
            sessions_dir, paths.state_file, config, write_gate=_wg(paths.state_file)
        )

    # The budget-killed session is reported in the stalled entries.
    assert any(entry["issue"] == 4801 for entry in result)
    # kill_process_tree was invoked on the api worker's pid.
    assert 99999 in killed_pids

    # The api sidecar is marked budget_exceeded directly (not via the
    # log-tail classification helper, so a coincidental throttle/auth match
    # cannot override the verdict).
    from charlie_work.claude_code import _sidecar_path as _api_sidecar_path

    sidecar = json.loads(_api_sidecar_path(sessions_dir, 4801, "api").read_text(encoding="utf-8"))
    assert sidecar["failure_kind"] == "budget_exceeded"

    # A session_budget_exceeded event was emitted with the issue + provider.
    state = load_state(paths.state_file)
    budget_events = [
        e for e in state.get("events", []) if e.get("kind") == "session_budget_exceeded"
    ]
    assert len(budget_events) == 1
    assert budget_events[0]["payload"]["issue_number"] == 4801
    assert budget_events[0]["payload"]["provider"] == "example"


def test_stall_lane_api_provider_auth_classification(tmp_path: Path) -> None:
    """Issue #484 review finding: the ``elif w.adapter_kind == "api"`` branch in
    ``_detect_and_handle_stalled_sessions`` (workflow.py stall lane) classifies
    a stalled api worker whose log tail contains a 401 signature as
    ``provider_auth`` with a 24h cooldown — not the generic ``stalled``
    fallback. A wiring regression that drops this branch leaves the sidecar
    classified ``stalled`` and ``throttled_until`` unset.
    """
    import os as _os

    from _api_budget_fixtures import api_worker_config, write_api_sidecar

    # No budget cap (dormant) so the budget-kill block does not fire; the
    # worker is STALLED and flows into the classification block.
    config = OrchestratorConfig(
        api_worker=api_worker_config(),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
        post_mortem=PostMortemConfig(enabled=False),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    OrchestratorApp(tmp_path, paths, config, fake_gh)

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    write_api_sidecar(sessions_dir, 4802, provider="example", pid=99998)

    # Write a 401 log tail and stale the mtime so the worker is STALLED.
    log_path = sessions_dir / "issue-4802.claude.log"
    log_path.write_text(
        "Working...\nError: 401 Unauthorized. Invalid API key.\n", encoding="utf-8"
    )
    old_time = datetime.now(UTC) - timedelta(minutes=25)
    _os.utime(log_path, (old_time.timestamp(), old_time.timestamp()))

    before = datetime.now(UTC)
    with (
        patch("charlie_work.worker.is_worker_alive", return_value=True),
        patch("charlie_work.write_gate.kill_process_tree", return_value=[99998]),
        patch("charlie_work.dead_worker_reap.sweep_orphan_processes", return_value=[]),
    ):
        from charlie_work.workflow import _detect_and_handle_stalled_sessions

        _detect_and_handle_stalled_sessions(
            sessions_dir, paths.state_file, config, write_gate=_wg(paths.state_file)
        )

    from charlie_work.claude_code import _sidecar_path as _api_sidecar_path

    sidecar = json.loads(_api_sidecar_path(sessions_dir, 4802, "api").read_text(encoding="utf-8"))
    assert sidecar["failure_kind"] == "provider_auth"

    # 24h cooldown persisted to state.json.
    state = load_state(paths.state_file)
    throttled_until = state.get("throttled_until")
    assert throttled_until is not None
    throttle_time = datetime.fromisoformat(throttled_until.replace("Z", "+00:00"))
    assert before + timedelta(hours=23) <= throttle_time <= before + timedelta(hours=25)
