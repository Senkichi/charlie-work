"""Shared world-builder for the Collect step tests (issue #3118).

``test_foreman_collect.py`` builds every scenario on the same seeded world —
a registered repo, a scaffolded foreman, hub-side ``state.json``/``events.db``,
a review-decision record, memory/journal/inbox/items, a prior Turn's
``applied.jsonl`` and a status snapshot. That machinery lives here so the
test module stays a list of assertions.

Golden briefings live under ``tests/fixtures/foreman/``; regenerate them with
``FOREMAN_REGEN_GOLDEN=1 uv run --extra dev pytest tests/test_foreman_collect.py``
after an intended render change, and review the diff like any other change.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest
from _worktree_fixtures import _init_repo

from charlie_work import instrumentation
from charlie_work import role_quota_ledger
from charlie_work.foreman import collect, journal as f_journal, paths as f_paths
from charlie_work.foreman import items as f_items
from charlie_work.foreman import memory as f_memory
from charlie_work.foreman.state import AuthoredIssue, ForemanState, save_state
from charlie_work.foreman.store import scaffold_foreman
from charlie_work.layout import status_snapshot_path

NOW = datetime(2026, 11, 25, 12, 0, 0, tzinfo=UTC)
REPO_KEY = "owner/repo"
FOREMAN_NAME = "g1"

GOLDEN_DIR = Path(__file__).parent / "fixtures" / "foreman"

_CHARTER = """\
name: g1
outcome: Ship the foreman collect step.
repos: ["owner/repo"]
standing: false
arming: auto
bounds:
  max_open_issues: 5
  max_issues_per_turn: 3
  min_turn_interval_minutes: 30
  max_quiet_hours: 24
  max_turns_per_day: 12
  max_unanswered_days: 7
"""

_INSTRUCTIONS = """\
# Instructions for g1

Keep the outcome honest. Follow the shared notes.

$section_ops-notes
"""


@dataclass
class _World:
    fleet_dir: Path
    repo_root: Path
    state_root: Path
    state_file: Path
    foreman_root: Path
    current_ts: str = "2026-11-25T08:00:00Z"

    def foreman(self) -> collect.Foreman:
        loaded = collect.load_foreman(self.fleet_dir, FOREMAN_NAME)
        assert loaded.ok, loaded.error
        return loaded.value

    def log(self, kind: str, payload: dict, ts: str) -> None:
        self.current_ts = ts
        instrumentation.log_event(self.state_file, kind, payload, level="info")

    def write_snapshot(
        self,
        issues: list[dict],
        *,
        prs: list[dict] | None = None,
        written_at: str = "2026-11-25T11:59:30Z",
    ) -> None:
        envelope = {
            "snapshot_written_at": written_at,
            "data": {"issues": issues, "prs": prs or []},
        }
        path = status_snapshot_path(self.state_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(envelope), encoding="utf-8")


def _issue(number: int, labels: list[str], title: str | None = None) -> dict:
    return {
        "number": number,
        "title": title or f"Issue {number}",
        "url": f"https://example.test/issues/{number}",
        "labels": labels,
        "dispatchable": False,
        "dependencies": {"declared": [], "open": []},
    }


def _build_world(
    tmp_path: Path,
    wt_scratch: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    heavy: bool = False,
    with_signal: bool = True,
) -> _World:
    fleet_dir = tmp_path / "fleet"
    fleet_dir.mkdir(parents=True, exist_ok=True)
    repo = wt_scratch / "repo"
    _init_repo(repo)
    state_root = repo / ".var" / "charlie-work"
    state_root.mkdir(parents=True)
    world = _World(
        fleet_dir=fleet_dir,
        repo_root=repo,
        state_root=state_root,
        state_file=state_root / "state.json",
        foreman_root=fleet_dir / "foremen" / FOREMAN_NAME,
    )
    now_s = "2026-11-25T11:00:00Z"
    (fleet_dir / "fleet.json").write_text(
        json.dumps(
            {
                "version": 1,
                "repos": {
                    REPO_KEY: {
                        "repo_root": str(repo),
                        "name_with_owner": REPO_KEY,
                        "config_path": str(repo / "orchestrator.config.yaml"),
                        "state_dir": str(state_root),
                        "first_seen": now_s,
                        "last_seen": now_s,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    scaffold = scaffold_foreman(fleet_dir, FOREMAN_NAME)
    assert scaffold.ok, scaffold.error
    charter = _CHARTER if not heavy else _CHARTER + "\n# " + "pad " * 9000
    (world.foreman_root / "charter.yaml").write_text(charter, encoding="utf-8")
    instructions = _INSTRUCTIONS if not heavy else _INSTRUCTIONS + "\n" + "pad " * 9000
    (world.foreman_root / "instructions.md").write_text(instructions, encoding="utf-8")
    shared = f_paths.shared_dir(fleet_dir)
    shared.mkdir(parents=True, exist_ok=True)
    (shared / "ops-notes.md").write_text(
        "Shared note: report at the cadence the charter sets.\n", encoding="utf-8"
    )

    save_state(
        world.foreman_root,
        ForemanState(
            state="active",
            event_cursor=5,
            inbox_cursor=0,
            signal_cursor=0,
            lifecycle_fingerprint="oldfp",
            turns_total=3,
            turns_today=2,
            turns_today_date="2026-11-25",
            consecutive_failures=0,
            last_turn_at="2026-11-25T08:00:00Z",
            issues={
                11: AuthoredIssue(number=11, body_hash="hashAAA"),
                12: AuthoredIssue(number=12, body_hash="hashBBB"),
            },
        ),
    )

    # Hub-side state.json: escalation fields for issue 11.
    world.state_file.write_text(
        json.dumps(
            {
                "issues": {
                    "11": {
                        "status": "escalated",
                        "escalation_reason": "quota_exhausted",
                        "reason_class": "mechanical",
                        "escalation_reasons_seen": ["quota_exhausted", "flaky_fixture"],
                        "terminal_since": "2026-11-25T10:00:00Z",
                    },
                    "13": {"status": "active"},
                },
                "prs": {},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(instrumentation, "_now_iso", lambda: world.current_ts)

    # events.db: ids 1-5 predate the cursor; the delta starts at 6.
    for i in range(5):
        world.log("loop_pass", {"pass": i}, "2026-11-25T08:0%d:00Z" % i)
    world.log("ready_observed", {"issue_number": 11}, "2026-11-25T09:00:00Z")
    world.log(
        "lifecycle_transition",
        {
            "issue_number": 11,
            "pr_number": 7,
            "from_state": "in_progress",
            "to_state": "needs_rework",
            "cause": "review_requested",
        },
        "2026-11-25T10:30:00Z",
    )
    world.log(
        "verdict_recorded",
        {"pr_number": 7, "issue_number": 11, "decision": "request_changes"},
        "2026-11-25T10:31:00Z",
    )
    world.log(
        "dispatch_deferred",
        {"issue_number": 13, "deferred_reason": "runner_capacity"},
        "2026-11-25T11:00:00Z",
    )
    world.log(
        "dispatch_backpressure",
        {
            "open_pr_count": 4,
            "max_open_agent_prs": 3,
            "requested_limit": 2,
            "clamped_limit": 1,
        },
        "2026-11-25T11:05:00Z",
    )
    world.log("runner_capacity_starved", {"demand": 5, "capacity": 2}, "2026-11-25T11:06:00Z")

    # Review decision record for the linked PR of issue 11.
    pr_dir = state_root / "prs" / "pr-7"
    pr_dir.mkdir(parents=True)
    required = (
        ["Tighten the section caps.", "Add a truncation case."]
        if not heavy
        else [f"required change number {i} " + "pad " * 60 for i in range(30)]
    )
    (pr_dir / "review-decision.json").write_text(
        json.dumps(
            {
                "decision": "request_changes",
                "summary": "The collect step is close but caps are missing.",
                "required_changes": required,
                "reviewed_head_sha": "abc123",
                "recorded_at": "2026-11-25T10:31:00Z",
            }
        ),
        encoding="utf-8",
    )

    # Memory files with pinned mtimes (a is oldest, d newest).
    memory_dir = f_paths.memory_dir(world.foreman_root)
    memory_dir.mkdir(parents=True, exist_ok=True)
    memory_specs = (
        [("context", "The repo is charlie-work.\n"), ("prefs", "Operator prefers short bodies.\n")]
        if not heavy
        else [(stem, f"{stem} content " + "x" * 7000) for stem in ("a", "b", "c", "d")]
    )
    for index, (stem, text) in enumerate(memory_specs):
        written = f_memory.write_memory(world.foreman_root, stem, text)
        assert written.ok, written.error
        stamp = datetime(2026, 11, 20, tzinfo=UTC).timestamp() + index * 60
        os.utime(f_paths.memory_file(world.foreman_root, stem), (stamp, stamp))

    journal_entries = [
        ("Turn t-0007: filed #11 and armed it.", "t-0007", "2026-11-24T09:00:00Z"),
        ("Turn t-0008: #11 dispatched to a worker.", "t-0008", "2026-11-24T12:00:00Z"),
        ("Turn t-0009: quiet; nothing to do.", "t-0009", "2026-11-24T15:00:00Z"),
        ("Turn t-0010: review requested changes on #11.", "t-0010", "2026-11-25T07:00:00Z"),
        ("Turn t-0011: escalated #11 on quota.", "t-0011", "2026-11-25T08:00:00Z"),
    ]
    if heavy:
        journal_entries += [
            ("Noise entry %d " % i + "x" * 3000, f"t-noise-{i}", "2026-11-25T08:0%d:00Z" % i)
            for i in range(3)
        ]
    # Two old entries so limit=5 visibly bites.
    journal_entries.insert(0, ("Ancient entry one.", None, "2026-11-20T09:00:00Z"))
    journal_entries.insert(0, ("Ancient entry two.", None, "2026-11-19T09:00:00Z"))
    for text, turn_id, ts in journal_entries:
        appended = f_journal.append_journal(world.foreman_root, text, turn_id=turn_id, ts=ts)
        assert appended.ok, appended.error

    inbox_text = (
        "Please prioritise the flaky fixture over new work."
        if not heavy
        else "Long operator note. " + "y" * 9000
    )
    appended = f_journal.append_inbox(world.foreman_root, inbox_text, ts="2026-11-25T07:00:00Z")
    assert appended.ok, appended.error

    if with_signal:
        signal_data = (
            {"text": "Big refactor landed upstream."} if not heavy else {"text": "z" * 12000}
        )
        appended = f_journal.append_signal(
            world.foreman_root, "slack", signal_data, ts="2026-11-25T07:30:00Z"
        )
        assert appended.ok, appended.error

    item = f_items.create_item(
        world.foreman_root,
        kind="decision",
        question="Which lane should the follow-up take?" if not heavy else "Q " + "q" * 5000,
        options=("worker lane", "review lane"),
        recommended=0,
        created_at="2026-11-20T10:00:00Z",
    )
    assert item.ok, item.error
    second = f_items.create_item(
        world.foreman_root,
        kind="decision",
        question="Arm issue #12?",
        options=("yes", "no"),
        recommended=0,
        created_at="2026-11-21T10:00:00Z",
    )
    assert second.ok, second.error
    answered = f_items.answer_item(
        world.foreman_root,
        second.value.id,
        choice=1,
        answered_at="2026-11-25T09:00:00Z",
    )
    assert answered.ok, answered.error

    # Previous Turn's Apply record: one applied file_issue plus a refusal.
    turn_dir = world.foreman_root / "turns" / "t-0011"
    turn_dir.mkdir(parents=True, exist_ok=True)
    applied_records = [
        {"key": "rest-labels", "type": "file_issue", "status": "applied"},
        {
            "key": "rest-core",
            "type": "file_issue",
            "status": "refused",
            "reason": "max_open_issues bound reached (5)",
        },
        {"key": "mem-1", "type": "write_memory", "status": "applied"},
    ]
    if heavy:
        applied_records += [
            {
                "key": f"overflow-{i}",
                "type": "file_issue",
                "status": "refused",
                "reason": "refused " + "r" * 200,
            }
            for i in range(30)
        ]
    (turn_dir / "applied.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in applied_records), encoding="utf-8"
    )

    # A live quota-ledger restriction (fleet dir is conftest-redirected).
    ledger = {
        "version": 1,
        "restrictions": {
            "claude-code|claude-opus-4": {
                "harness": "claude-code",
                "model": "claude-opus-4",
                "until": "2026-11-26T00:00:00Z",
                "reason": "rate_limited",
                "source": "dead_worker_classification",
                "role": "worker",
                "updated_at": "2026-11-25T06:00:00Z",
            }
        },
    }
    if heavy:
        for i in range(100):
            ledger["restrictions"][f"harness-{i}|model-{i}"] = {
                "harness": f"harness-{i}",
                "model": f"model-{i}",
                "until": "2026-11-26T00:00:00Z",
                "reason": "rate_limited",
            }
    (fleet_dir / role_quota_ledger.LEDGER_FILENAME).write_text(
        json.dumps(ledger), encoding="utf-8"
    )

    world.write_snapshot(
        [
            _issue(11, ["automated-ready", "agent:in-progress", "foreman:g1"], "Slice one"),
            _issue(13, ["automated-ready", "foreman:g1"], "Tracked slice"),
            _issue(14, ["automated-ready"], "Not ours"),
            _issue(15, ["foreman:other", "agent:queued"], "Another foreman's"),
        ]
        + (
            [
                _issue(100 + i, ["foreman:g1"], "Overflow issue %d " % i + "t" * 150)
                for i in range(100)
            ]
            if heavy
            else []
        ),
        prs=[
            {
                "number": 7,
                "title": "PR for slice one",
                "url": "https://example.test/pull/7",
                "issue_number": 11,
                "head": "agent/issue-11-slice",
                "is_draft": False,
                "reviewDecision": None,
            }
        ],
    )
    if heavy:
        for i in range(120):
            world.log(
                "lifecycle_transition",
                {
                    "issue_number": 11,
                    "from_state": f"state_{i}",
                    "to_state": f"state_{i + 1}",
                    "cause": "overflow",
                },
                f"2026-11-25T11:{i % 60:02d}:00Z",
            )
    return world


def _golden(name: str) -> Path:
    return GOLDEN_DIR / name


def _assert_golden(actual: str, name: str) -> None:
    path = _golden(name)
    if os.environ.get("FOREMAN_REGEN_GOLDEN"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(actual, encoding="utf-8")
    assert path.is_file(), f"golden {name} missing — run with FOREMAN_REGEN_GOLDEN=1"
    assert actual == path.read_text(encoding="utf-8")
