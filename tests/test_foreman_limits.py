"""Structural limits: escalation breaker, unattended limit, verdict (issue #3123).

Design §10/§16. Covers ``foreman/limits.py`` (the three limits and their
items/events/state transitions), the ``forced_arming`` gate in
``apply_checks._arm_or_raise``, the WILD CARD/CRICKET display spellings,
the five heartbeat alarm evaluators in ``heartbeat_alarms_fleet``, and the
per-foreman doctor check. Pass-level wiring (``foreman_pass_completed``,
notify after a failed Turn, the ``--dry-run`` gate) lives in the sibling
``tests/test_foreman_limits_pass.py``. Shared world builders live in
``tests/_foreman_pass_fixtures.py``; apply-side helpers mirror
``tests/test_foreman_apply.py``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from _foreman_collect_helpers import NOW, REPO_KEY, _World
from _foreman_pass_fixtures import (
    _authored,
    _events,
    _finish,
    _patch_state,
    _world,
)
from _worktree_fixtures import (
    _wt_scratch as _register_wt_scratch,  # noqa: F401 -- registers the wt_scratch fixture
)

from charlie_work import api_budget
from charlie_work import heartbeat_alarms_foreman as haff
from charlie_work import instrumentation, layout
from charlie_work import doctor
from charlie_work.foreman import display_state
from charlie_work.foreman import items as f_items
from charlie_work.foreman import limits as flimits
from charlie_work.foreman import paths as f_paths
from charlie_work.foreman.apply import apply_turn
from charlie_work.foreman.charter import Charter, CharterBounds
from charlie_work.foreman.state import (
    ForemanState,
    save_state,
)
from charlie_work.foreman_config import ForemanBreakerConfig
from charlie_work.local_issues import LocalFileGitHub

# A small breaker so tests need three finished issues, not ten.
_BREAKER = ForemanBreakerConfig(sample=3, baseline_sample=5, ratio=1.5)

REPO = REPO_KEY


def _eval(world: _World, breaker: ForemanBreakerConfig = _BREAKER, now: datetime = NOW):
    return flimits.evaluate_limits(
        world.foreman(), fleet_dir=world.fleet_dir, breaker=breaker, now=now
    )


def _item(world: _World, item_id: str) -> f_items.Item:
    items = {i.id: i for i in f_items.load_items(world.foreman_root)}
    assert item_id in items, f"item {item_id} not in {sorted(items)}"
    return items[item_id]


def _trip_breaker(world: _World) -> ForemanState:
    """Seed a full escalated sample plus a calm baseline, then evaluate."""
    _patch_state(world, issues=_authored(20, 21, 22))
    _finish(
        world,
        (20, "human_needed"),
        (21, "operator_queue"),
        (22, "human_needed"),
        (30, "done"),
        (31, "done"),
        (32, "done"),
        (33, "done"),
        (34, "human_needed"),
    )
    return _eval(world)


# ---------------------------------------------------------------------------
# Escalation breaker
# ---------------------------------------------------------------------------


def test_breaker_trips_degrades_and_raises_item(
    tmp_path: Path, wt_scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = _world(tmp_path, wt_scratch, monkeypatch)

    state = _trip_breaker(world)

    assert state.state == "degraded"
    assert state.degraded_reasons == ("breaker",)
    assert state.breaker_item_id is not None
    tripped = _events(world, flimits.EVENT_BREAKER_TRIPPED)
    assert len(tripped) == 1
    assert tripped[0]["rate"] == 1.0 and tripped[0]["baseline_rate"] == 0.2
    item = _item(world, state.breaker_item_id)
    assert item.kind == "decision" and item.status == "open"
    assert "breaker" in (item.question or "")


def test_breaker_quiet_under_threshold(
    tmp_path: Path, wt_scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = _world(tmp_path, wt_scratch, monkeypatch)
    _patch_state(world, issues=_authored(20, 21, 22))
    # Foreman rate 1/3 vs baseline 2/5: 0.33 <= 1.5 * 0.4 — no trip.
    _finish(
        world,
        (20, "human_needed"),
        (21, "done"),
        (22, "done"),
        (30, "human_needed"),
        (31, "operator_queue"),
        (32, "done"),
        (33, "done"),
        (34, "done"),
    )

    state = _eval(world)

    assert state.state == "active"
    assert state.degraded_reasons == ()
    assert _events(world, flimits.EVENT_BREAKER_TRIPPED) == []


def test_breaker_needs_a_full_sample(
    tmp_path: Path, wt_scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = _world(tmp_path, wt_scratch, monkeypatch)
    _patch_state(world, issues=_authored(20, 21))
    _finish(
        world,
        (20, "human_needed"),
        (21, "human_needed"),
        (30, "done"),
        (31, "done"),
        (32, "done"),
        (33, "done"),
        (34, "done"),
    )

    state = _eval(world)

    assert state.degraded_reasons == ()
    assert _events(world, flimits.EVENT_BREAKER_TRIPPED) == []


def test_breaker_recovers_when_rate_falls(
    tmp_path: Path, wt_scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = _world(tmp_path, wt_scratch, monkeypatch)
    assert "breaker" in _trip_breaker(world).degraded_reasons

    # Three new done finishes push the escalated ones out of the sample.
    _patch_state(world, issues=_authored(20, 21, 22, 23, 24, 25))
    _finish(world, (23, "done"), (24, "done"), (25, "done"))
    state = _eval(world)

    assert state.state == "active"
    assert state.degraded_reasons == ()
    recovered = _events(world, flimits.EVENT_BREAKER_RECOVERED)
    assert len(recovered) == 1 and recovered[0]["via"] == "rate"


def test_breaker_answer_acknowledges_sample(
    tmp_path: Path, wt_scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = _world(tmp_path, wt_scratch, monkeypatch)
    state = _trip_breaker(world)
    answered = f_items.answer_item(
        world.foreman_root,
        state.breaker_item_id,
        choice=0,
        answered_at="2026-11-25T11:30:00Z",
    )
    assert answered.ok, answered.error

    # Still over the threshold, but the operator resolved the item: the
    # breaker recovers and acknowledges *this* sample.
    state = _eval(world)
    assert "breaker" not in state.degraded_reasons
    assert state.breaker_ack_sample
    recovered = _events(world, flimits.EVENT_BREAKER_RECOVERED)
    assert recovered[-1]["via"] == "item_resolved"

    # The acknowledgement holds while the same sample holds — no re-trip.
    _eval(world)
    assert len(_events(world, flimits.EVENT_BREAKER_TRIPPED)) == 1

    # A new finished foreman issue changes the sample — the breaker re-trips.
    _patch_state(world, issues=_authored(20, 21, 22, 23))
    _finish(world, (23, "human_needed"))
    _eval(world)
    assert len(_events(world, flimits.EVENT_BREAKER_TRIPPED)) == 2


def test_breaker_reraises_item_when_trip_create_failed(
    tmp_path: Path, wt_scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed create_item at trip time must not strand a degraded foreman.

    The item is the only operator-facing path back, so the next pass
    re-raises it while the breaker still holds — mirroring the verdict's
    forced-arming re-raise.
    """
    world = _world(tmp_path, wt_scratch, monkeypatch)
    real_create = f_items.create_item
    monkeypatch.setattr(f_items, "create_item", lambda *a, **k: f_items.Result(error="disk full"))
    state = _trip_breaker(world)
    assert "breaker" in state.degraded_reasons
    assert state.breaker_item_id is None

    monkeypatch.setattr(f_items, "create_item", real_create)
    state = _eval(world)

    assert state.state == "degraded"
    assert "breaker" in state.degraded_reasons
    assert state.breaker_item_id is not None
    item = _item(world, state.breaker_item_id)
    assert item.kind == "decision" and item.status == "open"
    assert "breaker" in (item.question or "")


def test_baseline_excludes_foreman_labeled_snapshot_issues(
    tmp_path: Path, wt_scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tracked issues are owned too — the baseline must not count them.

    The seeded snapshot carries 13 under ``foreman:g1`` (tracked — the
    operator labeled it; it is in no register) and 15 under
    ``foreman:other``. Finished, both still stay out of the "no foreman
    owns" baseline the escalation rate is computed against.
    """
    world = _world(tmp_path, wt_scratch, monkeypatch)
    _patch_state(world, issues=_authored(20, 21, 22))
    _finish(
        world,
        (20, "human_needed"),
        (21, "operator_queue"),
        (22, "human_needed"),
        (30, "done"),
        (31, "done"),
        (32, "done"),
        (33, "done"),
        (34, "done"),
        (13, "human_needed"),
        (15, "human_needed"),
    )

    state = _eval(world)

    assert state.breaker_item_id is not None
    tripped = _events(world, flimits.EVENT_BREAKER_TRIPPED)
    assert len(tripped) == 1
    assert tripped[0]["baseline"] == 5
    assert tripped[0]["baseline_rate"] == 0.0


def test_breaker_reason_is_independent_of_unattended(
    tmp_path: Path, wt_scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Clearing the breaker must not clear a concurrent unattended cause."""
    world = _world(tmp_path, wt_scratch, monkeypatch)
    stale = f_items.create_item(
        world.foreman_root,
        kind="decision",
        question="Stale question?",
        options=("a", "b"),
        recommended=0,
        created_at="2026-11-10T00:00:00Z",
    )
    assert stale.ok, stale.error
    state = _trip_breaker(world)
    assert set(state.degraded_reasons) == {"breaker", "unattended"}

    _patch_state(world, issues=_authored(20, 21, 22, 23, 24, 25))
    _finish(world, (23, "done"), (24, "done"), (25, "done"))
    state = _eval(world)

    # The breaker recovered; the stale decision item still holds degraded.
    assert state.degraded_reasons == ("unattended",)
    assert state.state == "degraded"


# ---------------------------------------------------------------------------
# Unattended limit
# ---------------------------------------------------------------------------


def test_unattended_limit_degrades_on_stale_decision(
    tmp_path: Path, wt_scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = _world(tmp_path, wt_scratch, monkeypatch)
    stale = f_items.create_item(
        world.foreman_root,
        kind="decision",
        question="Which lane?",
        options=("a", "b"),
        recommended=0,
        created_at="2026-11-10T00:00:00Z",
    )
    assert stale.ok, stale.error

    state = _eval(world)

    assert state.state == "degraded"
    assert state.degraded_reasons == ("unattended",)
    fired = _events(world, flimits.EVENT_UNATTENDED_LIMIT)
    assert len(fired) == 1
    assert fired[0]["item_id"] == stale.value.id
    assert fired[0]["max_unanswered_days"] == 7


def test_unattended_limit_recovers_on_answer(
    tmp_path: Path, wt_scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = _world(tmp_path, wt_scratch, monkeypatch)
    stale = f_items.create_item(
        world.foreman_root,
        kind="decision",
        question="Which lane?",
        options=("a", "b"),
        recommended=0,
        created_at="2026-11-10T00:00:00Z",
    )
    assert stale.ok, stale.error
    _eval(world)

    answered = f_items.answer_item(
        world.foreman_root, stale.value.id, choice=0, answered_at="2026-11-25T11:30:00Z"
    )
    assert answered.ok, answered.error
    state = _eval(world)

    assert state.state == "active"
    assert state.degraded_reasons == ()
    assert len(_events(world, flimits.EVENT_UNATTENDED_RECOVERED)) == 1


def test_unattended_ignores_fresh_and_non_decision_items(
    tmp_path: Path, wt_scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = _world(tmp_path, wt_scratch, monkeypatch)
    # The seeded world already has an open decision item from 2026-11-20
    # (5 days, inside the 7-day bound) — nothing stale, no degrade.
    state = _eval(world)
    assert "unattended" not in state.degraded_reasons

    # An old *report* item is not a decision — still no degrade.
    report = f_items.create_item(
        world.foreman_root,
        kind="report",
        text="old report",
        created_at="2026-11-01T00:00:00Z",
    )
    assert report.ok, report.error
    state = _eval(world)
    assert "unattended" not in state.degraded_reasons


# ---------------------------------------------------------------------------
# Verdict (design §16)
# ---------------------------------------------------------------------------


def _verdict_world(world: _World, outcomes: list[str]) -> None:
    """20 authored issues finished with the given terminal states, plus a done baseline."""
    numbers = list(range(20, 20 + len(outcomes)))
    _patch_state(world, issues=_authored(*numbers))
    _finish(world, *zip(numbers, outcomes))
    _finish(world, (40, "done"), (41, "done"), (42, "done"), (43, "done"), (44, "done"))


def test_verdict_passes_at_twenty_finished(
    tmp_path: Path, wt_scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = _world(tmp_path, wt_scratch, monkeypatch)
    _verdict_world(world, ["done"] * 20)

    state = _eval(world)

    assert state.verdict_raised
    assert state.forced_arming is None
    verdicts = _events(world, flimits.EVENT_VERDICT)
    assert len(verdicts) == 1
    assert verdicts[0]["passed"] is True
    assert verdicts[0]["authored_finished"] == 20
    assert verdicts[0]["done_share"] == 1.0
    reports = [i for i in f_items.load_items(world.foreman_root) if i.kind == "report"]
    assert any("verdict" in (i.text or "").lower() for i in reports)


def test_verdict_due_by_age_before_twenty_issues(
    tmp_path: Path, wt_scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = _world(tmp_path, wt_scratch, monkeypatch)
    # A first Turn older than 14 days makes the verdict due with <20 issues.
    world.current_ts = "2026-11-01T00:00:00Z"
    instrumentation.log_event(
        f_paths.state_path(world.foreman_root),
        "foreman_turn_completed",
        {"input_tokens": 10, "output_tokens": 5},
        repo=REPO_KEY,
    )

    state = _eval(world)

    assert state.verdict_raised
    verdicts = _events(world, flimits.EVENT_VERDICT)
    assert len(verdicts) == 1


def test_verdict_miss_forces_propose_and_retune_item(
    tmp_path: Path, wt_scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = _world(tmp_path, wt_scratch, monkeypatch)
    _verdict_world(world, ["human_needed"] * 20)

    state = _eval(world)

    assert state.verdict_raised
    assert state.forced_arming == "propose"
    assert state.verdict_item_id is not None
    verdicts = _events(world, flimits.EVENT_VERDICT)
    assert verdicts[0]["passed"] is False
    assert verdicts[0]["done_ok"] is False
    item = _item(world, state.verdict_item_id)
    assert item.kind == "decision" and "retire" in (item.question or "").lower()

    # Answering the retune item clears the forced arming (design §16's path back).
    answered = f_items.answer_item(
        world.foreman_root, state.verdict_item_id, choice=0, answered_at="2026-11-25T11:30:00Z"
    )
    assert answered.ok, answered.error
    state = _eval(world)
    assert state.forced_arming is None
    assert state.verdict_item_id is None


def test_verdict_token_leg_fails_on_foreman_heavy_spend(
    tmp_path: Path, wt_scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = _world(tmp_path, wt_scratch, monkeypatch)
    _verdict_world(world, ["done"] * 20)
    # Worker tokens: 100 against issue 20; foreman Turns: 30 — over the 25% cap.
    sessions = layout.sessions_dir_default(world.state_root)
    sessions.mkdir(parents=True, exist_ok=True)
    (sessions / "issue-20.events.jsonl").write_text(
        '{"tokens": 50}\n{"tokens": 100}\n', encoding="utf-8"
    )
    instrumentation.log_event(
        f_paths.state_path(world.foreman_root),
        "foreman_turn_completed",
        {"input_tokens": 25, "output_tokens": 5},
        repo=REPO_KEY,
    )

    state = _eval(world)

    verdicts = _events(world, flimits.EVENT_VERDICT)
    assert verdicts[0]["passed"] is False
    assert verdicts[0]["done_ok"] is True
    assert verdicts[0]["token_ok"] is False
    assert verdicts[0]["worker_reviewer_tokens"] == 100
    assert verdicts[0]["turn_tokens"] == 30
    assert state.forced_arming == "propose"


def test_verdict_token_leg_counts_ledger_sessions(
    tmp_path: Path, wt_scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The api-budget session ledger feeds the token leg too (issue #480)."""
    world = _world(tmp_path, wt_scratch, monkeypatch)
    _verdict_world(world, ["done"] * 20)
    # Ledger sessions: 100 tokens against authored issue 20; 999 against an
    # issue the foreman never authored — only the authored session counts.
    ledger = {
        "version": 1,
        "days": {},
        "lifetime_usd": 0.0,
        "sessions": [
            {
                "issue": 20,
                "session_id": "s1",
                "provider": "api",
                "model": "m",
                "started_at": "",
                "ended_at": "",
                "input_tokens": 60,
                "output_tokens": 40,
                "cached_tokens": 0,
                "usd": 0.0,
                "duration_s": 0.0,
                "outcome": "ok",
            },
            {
                "issue": 99,
                "session_id": "s2",
                "provider": "api",
                "model": "m",
                "started_at": "",
                "ended_at": "",
                "input_tokens": 999,
                "output_tokens": 0,
                "cached_tokens": 0,
                "usd": 0.0,
                "duration_s": 0.0,
                "outcome": "ok",
            },
        ],
    }
    api_budget.ledger_path(world.state_root).write_text(json.dumps(ledger), encoding="utf-8")
    instrumentation.log_event(
        f_paths.state_path(world.foreman_root),
        "foreman_turn_completed",
        {"input_tokens": 25, "output_tokens": 5},
        repo=REPO_KEY,
    )

    state = _eval(world)

    verdicts = _events(world, flimits.EVENT_VERDICT)
    assert verdicts[0]["worker_reviewer_tokens"] == 100
    assert verdicts[0]["turn_tokens"] == 30
    assert verdicts[0]["token_ok"] is False
    assert state.forced_arming == "propose"


def test_verdict_retries_when_report_item_fails(
    tmp_path: Path, wt_scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = _world(tmp_path, wt_scratch, monkeypatch)
    _verdict_world(world, ["done"] * 20)

    real_create = f_items.create_item
    monkeypatch.setattr(f_items, "create_item", lambda *a, **k: f_items.Result(error="disk full"))
    state = _eval(world)
    assert not state.verdict_raised
    assert _events(world, flimits.EVENT_VERDICT) == []

    monkeypatch.setattr(f_items, "create_item", real_create)
    state = _eval(world)
    assert state.verdict_raised
    assert len(_events(world, flimits.EVENT_VERDICT)) == 1


def test_verdict_not_raised_twice(
    tmp_path: Path, wt_scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = _world(tmp_path, wt_scratch, monkeypatch)
    _verdict_world(world, ["done"] * 20)
    _eval(world)
    _eval(world)
    assert len(_events(world, flimits.EVENT_VERDICT)) == 1


# ---------------------------------------------------------------------------
# forced_arming gate in apply_checks._arm_or_raise
# ---------------------------------------------------------------------------


def _apply_setup(tmp_path: Path, state: ForemanState) -> tuple[Path, LocalFileGitHub, Charter]:
    froot = tmp_path / "foremen" / "pepe"
    froot.mkdir(parents=True)
    saved = save_state(froot, state)
    assert saved.ok, saved.error
    repo_root = tmp_path / "repo"
    issues_dir = repo_root / ".issues"
    issues_dir.mkdir(parents=True)
    gh = LocalFileGitHub(repo_root=repo_root, issues_dir=issues_dir)
    charter = Charter(
        name="pepe",
        outcome="ship the outcome",
        repos=(REPO,),
        arming="auto",
        bounds=CharterBounds(),
    )
    return froot, gh, charter


def test_forced_arming_files_unarmed_with_propose_reason(tmp_path: Path) -> None:
    froot, gh, charter = _apply_setup(
        tmp_path, ForemanState(state="active", forced_arming="propose")
    )
    turn = froot / "turns" / "t1"
    turn.mkdir(parents=True)
    (turn / "actions.json").write_text(
        json.dumps(
            {
                "turn_summary": "a turn",
                "actions": [
                    {
                        "type": "file_issue",
                        "key": "a",
                        "repo": REPO,
                        "title": "Title a",
                        "body": "Body a.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = apply_turn(froot, "t1", charter=charter, gh=gh)

    assert result.ok, result.error
    (outcome,) = result.value.outcomes
    assert "unarmed (verdict forced arming: propose)" in (outcome.detail or "")
    issue = gh.issue_view(1)
    assert "agent:ready" not in {label["name"] for label in issue["labels"]}
    arming_items = [i for i in f_items.load_items(froot) if "Arm issue" in (i.question or "")]
    assert len(arming_items) == 1


# ---------------------------------------------------------------------------
# Display spellings
# ---------------------------------------------------------------------------


def test_display_state_spellings() -> None:
    assert display_state("active") == "active"
    assert display_state("faulted") == "faulted"
    assert display_state("degraded", ("breaker",)) == "WILD CARD"
    assert display_state("degraded", ("unattended",)) == "CRICKET"
    assert display_state("degraded", ("breaker", "unattended")) == "WILD CARD"
    assert display_state("degraded") == "CRICKET"


# ---------------------------------------------------------------------------
# Heartbeat alarm evaluators (heartbeat_alarms_foreman)
# ---------------------------------------------------------------------------


def test_alarm_turn_tokens_spikes_and_quiet() -> None:
    spike = haff.eval_foreman_turn_tokens("g1", [100, 100, 100, 400])
    assert spike.severity == "anomaly"
    calm = haff.eval_foreman_turn_tokens("g1", [100, 100, 100, 200])
    assert calm.severity == "ok"
    too_few = haff.eval_foreman_turn_tokens("g1", [500])
    assert too_few.severity == "ok"
    zero_median = haff.eval_foreman_turn_tokens("g1", [0, 0, 900])
    assert zero_median.severity == "ok"


def test_alarm_turns_per_day() -> None:
    today = NOW.astimezone(UTC).strftime("%Y-%m-%d")
    hit = haff.eval_foreman_turns_per_day("g1", 12, today, 12, NOW)
    assert hit.severity == "anomaly"
    under = haff.eval_foreman_turns_per_day("g1", 5, today, 12, NOW)
    assert under.severity == "ok"
    stale_date = haff.eval_foreman_turns_per_day("g1", 12, "2026-11-24", 12, NOW)
    assert stale_date.severity == "ok"
    no_bound = haff.eval_foreman_turns_per_day("g1", 99, today, None, NOW)
    assert no_bound.severity == "ok"


def test_alarm_breaker_and_state() -> None:
    assert haff.eval_foreman_breaker("g1", ("breaker",)).severity == "anomaly"
    assert haff.eval_foreman_breaker("g1", ("unattended",)).severity == "ok"
    assert haff.eval_foreman_state("g1", "faulted", ()).severity == "anomaly"
    assert haff.eval_foreman_state("g1", "degraded", ("unattended",)).severity == "warn"
    # Breaker-tripped degraded is the breaker alarm's job, not a second finding.
    assert haff.eval_foreman_state("g1", "degraded", ("breaker",)).severity == "ok"
    assert haff.eval_foreman_state("g1", "active", ()).severity == "ok"


def test_alarm_pass_freshness() -> None:
    fresh = haff.eval_foreman_pass_freshness("2026-11-25T11:55:00Z", 5, NOW)
    assert fresh.severity == "ok"
    stale = haff.eval_foreman_pass_freshness("2026-11-25T11:00:00Z", 5, NOW)
    assert stale.severity == "anomaly"
    never = haff.eval_foreman_pass_freshness(None, 5, NOW)
    assert never.severity == "anomaly"
    bad = haff.eval_foreman_pass_freshness("not-a-ts", 5, NOW)
    assert bad.severity == "anomaly"


# ---------------------------------------------------------------------------
# doctor _check_foremen
# ---------------------------------------------------------------------------


def _doctor_fleet(tmp_path: Path, *, charter: str, state: dict, items: dict | None) -> Path:
    fleet = tmp_path / "fleet"
    fdir = fleet / "foremen" / "g1"
    fdir.mkdir(parents=True)
    (fdir / "charter.yaml").write_text(charter, encoding="utf-8")
    (fdir / "state.json").write_text(json.dumps(state), encoding="utf-8")
    if items is not None:
        (fdir / "items.json").write_text(json.dumps(items), encoding="utf-8")
    (fleet / "fleet.json").write_text(
        json.dumps(
            {
                "version": 1,
                "repos": {REPO_KEY: {"state_dir": str(tmp_path / "repo-state")}},
            }
        ),
        encoding="utf-8",
    )
    return fleet


def test_doctor_foremen_reports_decode_problems(tmp_path: Path) -> None:
    fleet = _doctor_fleet(
        tmp_path,
        charter='name: "g1"\noutcome: "x"\nrepos: ["owner/repo"]\n',
        state={"state": "active"},
        items={"items": [{"id": "it-1", "kind": "decision"}]},
    )
    findings: list[tuple[str, bool, str]] = []
    doctor._check_foremen(
        lambda name, ok, detail, **kw: findings.append((name, ok, detail)),
        fleet_dir_override=str(fleet),
    )
    assert findings == [("foreman g1", True, "charter, state, and items decode; repo registered")]

    fleet = _doctor_fleet(
        tmp_path / "bad",
        charter='name: "g1"\noutcome: "x"\nrepos: ["ghost/repo"]\n',
        state={"state": "bogus"},
        items={"items": [{"id": "it-1", "kind": "nonsense"}]},
    )
    findings = []
    doctor._check_foremen(
        lambda name, ok, detail, **kw: findings.append((name, ok, detail)),
        fleet_dir_override=str(fleet),
    )
    assert len(findings) == 1 and findings[0][1] is False
    detail = findings[0][2]
    assert "ghost/repo" in detail and "bogus" in detail and "failed to decode" in detail
