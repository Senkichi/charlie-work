"""``charlie runners shadow-status`` journal/prologue reconciliation.

Split out of ``tests/test_cli.py`` (issue #1561, Track 1) -- bodies are
verbatim relocations; shared fakes live in ``tests/_cli_fixtures.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from charlie_work import cli
from charlie_work.config import OrchestratorConfig
from charlie_work.instrumentation import log_event
from charlie_work.paths import runtime_paths
from charlie_work.workflow import CommandResult


# --------------------------------------------------------------------------
# runners shadow-status (issue #909)
# --------------------------------------------------------------------------


def _shadow_status_setup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Path, Path, argparse.Namespace]:
    """Wire find_repo_root/load_layered_config to a fresh tmp repo + fleet dir.

    Returns (repo_root, fleet_directory, args) so callers can write directly
    to the exact paths the command under test will read.
    """
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    fleet_directory = tmp_path / "fleet"
    monkeypatch.setattr(cli, "find_repo_root", lambda repo, explicit=False, **kw: repo_root)
    monkeypatch.setattr(cli, "load_layered_config", lambda *a, **k: OrchestratorConfig())
    args = cli.build_parser().parse_args(
        ["--fleet-dir", str(fleet_directory), "runners", "shadow-status"]
    )
    return repo_root, fleet_directory, args


def _allocation_state_file(repo_root: Path) -> Path:
    paths = runtime_paths(repo_root, OrchestratorConfig().runtime.state_dir)
    return paths.state_file


def _write_allocation_event(state_file: Path, *, source: str, actuating_planner: str) -> None:
    """Append one runner_allocation event, matching runner_allocation_pass.py's
    real payload shape (source/actuating_planner live inside the payload, not
    as columns -- issue #909 trap 1)."""
    log_event(
        state_file,
        "runner_allocation",
        {
            "source": source,
            "actuating_planner": actuating_planner,
            "budget": 8,
            "budget_reason": "test",
            "changes": [],
            "applied": [],
            "dry_run": False,
            "notes": [],
            "targets": [],
        },
    )


def _journal_record(
    pass_id: str,
    *,
    agreed: bool,
    changes: list[Any] | None = None,
    shadow_changes: list[Any] | None = None,
) -> dict[str, Any]:
    """One diff-journal record, matching ci_fleet.diff_journal's write shape.

    By default the shadow plan mirrors the live plan's changes so an
    ``agreed=True`` record is internally consistent. ``shadow_changes`` can be
    set independently to simulate a planner disagreement about the kind of
    change a pass emits.
    """
    live_changes = changes or []
    shadow = shadow_changes if shadow_changes is not None else list(live_changes)
    return {
        "pass_id": pass_id,
        "inputs": {},
        "live_plan": {
            "budget": 1,
            "budget_reason": "test",
            "targets": [],
            "changes": live_changes,
            "notes": [],
        },
        "shadow_plan": {
            "budget": 1,
            "budget_reason": "test",
            "targets": [],
            "changes": shadow,
            "notes": [],
        },
        "agreed": agreed,
        "outcome": "ok",
        "differences": [],
        "shadow_ms": 1.0,
    }


def _write_journal(fleet_directory: Path, records: list[dict[str, Any]]) -> Path:
    fleet_directory.mkdir(parents=True, exist_ok=True)
    journal_file = fleet_directory / "shadow-planner-diff.jsonl"
    with journal_file.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    return journal_file


def test_run_runners_shadow_status_reports_not_found_for_missing_stores(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Neither store exists yet: both must be reported as explicitly missing.

    A silent empty result is exactly the failure #909 exists to prevent -- an
    operator must be able to tell "nothing recorded yet" apart from "the
    report is broken and came back empty". Also asserts the command does not
    create either store as a side effect of looking for it: sqlite3.connect()
    creates an empty file on open, which would make a "read-only reporter"
    a lie the first time it runs against a fresh repo.
    """
    repo_root, fleet_directory, args = _shadow_status_setup(monkeypatch, tmp_path)

    result = cli.run_runners_shadow_status(args)

    assert result.ok is True
    assert result.data["events_db"]["found"] is False
    assert result.data["journal"]["found"] is False
    assert result.data["actuating"] is None
    assert result.data["configured_not_yet_in_effect"] is None
    assert result.data["agreement_streak"] is None
    assert result.data["change_agreement_streak"] is None
    assert result.data["gate"] is None
    assert not _allocation_state_file(repo_root).parent.joinpath("events.db").exists(), (
        "looking for events.db must not create it"
    )
    assert not fleet_directory.exists(), "looking for the journal must not create the fleet dir"


def test_run_runners_shadow_status_actuating_from_latest_prologue_row(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """actuating must reflect the LATEST source=prologue row, not the first."""
    repo_root, _fleet_directory, args = _shadow_status_setup(monkeypatch, tmp_path)
    state_file = _allocation_state_file(repo_root)
    _write_allocation_event(state_file, source="prologue", actuating_planner="legacy")
    _write_allocation_event(state_file, source="prologue", actuating_planner="new")

    result = cli.run_runners_shadow_status(args)

    assert result.data["events_db"]["found"] is True
    assert result.data["actuating"]["planner"] == "new"
    assert result.data["configured_not_yet_in_effect"] is None


def test_run_runners_shadow_status_newer_cli_row_is_pending_not_actuating(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Trap 2 (#909): a newer source='cli' row must never read as confirmation.

    Exact scenario the issue names: an operator flips config and runs a
    dry-run ``charlie runners allocate`` (source='cli') naming the new
    planner, but the supervisor caches config at startup and is still
    actuating the old one until it respawns. 'actuating' must stay 'legacy'
    -- the cli row must appear only in the separate, explicitly-pending
    field. Discriminates against a naive "most recent runner_allocation
    event regardless of source" implementation, which would report 'new'
    here instead.
    """
    repo_root, _fleet_directory, args = _shadow_status_setup(monkeypatch, tmp_path)
    state_file = _allocation_state_file(repo_root)
    _write_allocation_event(state_file, source="prologue", actuating_planner="legacy")
    _write_allocation_event(state_file, source="cli", actuating_planner="new")

    result = cli.run_runners_shadow_status(args)

    assert result.data["actuating"]["planner"] == "legacy", (
        "the supervisor has not respawned yet -- actuating must not jump to the cli row's planner"
    )
    assert result.data["configured_not_yet_in_effect"] is not None
    assert result.data["configured_not_yet_in_effect"]["planner"] == "new"
    assert (
        result.data["actuating"]["planner"]
        != result.data["configured_not_yet_in_effect"]["planner"]
    ), "the two fields must disagree here -- that disagreement is the whole point of trap 2"


def test_run_runners_shadow_status_older_cli_row_is_not_surfaced_as_pending(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A stale cli pre-check must not resurface once the supervisor catches up.

    Inverse of the trap-2 test: the cli row is written FIRST (the dry-run
    pre-check) and a prologue row confirming the supervisor already picked up
    the same planner is written after. Once that has happened, the old cli
    row is history, not a pending change, and must not be shown as one.
    """
    repo_root, _fleet_directory, args = _shadow_status_setup(monkeypatch, tmp_path)
    state_file = _allocation_state_file(repo_root)
    _write_allocation_event(state_file, source="cli", actuating_planner="new")
    _write_allocation_event(state_file, source="prologue", actuating_planner="new")

    result = cli.run_runners_shadow_status(args)

    assert result.data["actuating"]["planner"] == "new"
    assert result.data["configured_not_yet_in_effect"] is None


def test_run_runners_shadow_status_change_streak_excludes_noop_passes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The load-bearing streak counts only passes that emitted a real change.

    ``live_plan`` is a dict (keys budget/budget_reason/changes/notes/targets)
    that is always truthy -- a naive ``if record.get("live_plan")`` check
    would count every no-op pass as "real" and this test would see
    change_agreement_streak == {"streak": 5, "total": 5}, identical to the
    all-passes streak, which defeats the entire point of reporting it
    separately. Only ``live_plan["changes"]`` (a list) says whether a pass
    actually emitted a change.
    """
    _repo_root, fleet_directory, args = _shadow_status_setup(monkeypatch, tmp_path)
    real_change = [{"repo": "x/y", "runner": "r-1", "action": "start", "reason": "test"}]
    _write_journal(
        fleet_directory,
        [
            _journal_record("p1", agreed=True, changes=[]),
            _journal_record("p2", agreed=True, changes=[]),
            _journal_record("p3", agreed=True, changes=[]),
            _journal_record("p4", agreed=True, changes=real_change),
            _journal_record("p5", agreed=True, changes=[]),
        ],
    )

    result = cli.run_runners_shadow_status(args)

    assert result.data["journal"]["found"] is True
    assert result.data["agreement_streak"] == {"streak": 5, "total": 5}
    assert result.data["change_agreement_streak"] == {"streak": 1, "total": 1}, (
        "must count only the one changed pass, not be inflated by the 4 surrounding no-op passes"
    )


def test_run_runners_shadow_status_streaks_break_on_disagreement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Both streaks are the TRAILING agreed run, and a disagreement resets it."""
    _repo_root, fleet_directory, args = _shadow_status_setup(monkeypatch, tmp_path)
    change = [{"repo": "x/y", "runner": "r-1", "action": "start", "reason": "test"}]
    _write_journal(
        fleet_directory,
        [
            _journal_record("p1", agreed=True, changes=change),
            _journal_record("p2", agreed=False, changes=change),
            _journal_record("p3", agreed=True, changes=change),
        ],
    )

    result = cli.run_runners_shadow_status(args)

    assert result.data["agreement_streak"] == {"streak": 1, "total": 3}
    assert result.data["change_agreement_streak"] == {"streak": 1, "total": 3}


def test_run_runners_shadow_status_gate_uses_real_ci_fleet_evaluate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The gate verdict is ci_fleet's real §6.3 evaluate(), not a fabricated field.

    Proven by using REQUIRED_STREAK's real value (200): 3 agreed passes must
    NOT open the gate. A fabricated/hardcoded ``ok: True`` would pass this
    test's shape but fail this specific assertion.
    """
    from ci_fleet.shadow_gate import REQUIRED_STREAK

    _repo_root, fleet_directory, args = _shadow_status_setup(monkeypatch, tmp_path)
    _write_journal(
        fleet_directory,
        [_journal_record(f"p{i}", agreed=True, changes=[]) for i in range(3)],
    )

    result = cli.run_runners_shadow_status(args)

    gate = result.data["gate"]
    assert gate is not None
    assert gate["streak"] == 3
    assert gate["streak_required"] == REQUIRED_STREAK
    assert REQUIRED_STREAK > 3, "sanity: the fixture must be short of the real threshold"
    assert gate["streak_ok"] is False
    assert gate["ok"] is False, "3/200 must not open the gate"
    assert "GATE CLOSED" in gate["report"]


def test_shadow_status_subcommand_is_registered() -> None:
    args = cli.build_parser().parse_args(["runners", "shadow-status"])
    assert args.command == "runners"
    assert args.runners_command == "shadow-status"


def test_main_dispatches_runners_shadow_status(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_result = CommandResult(ok=True, message="runners shadow-status complete", data={})
    mock = MagicMock(return_value=mock_result)
    monkeypatch.setattr(cli, "run_runners_shadow_status", mock)

    exit_code = cli.main(["runners", "shadow-status"])

    assert exit_code == 0
    mock.assert_called_once()


def test_main_runners_shadow_status_renders_pending_planner_and_note_ordering(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """End-to-end through the real ``main()`` render path, not just ``result.data``.

    The nine data-level tests above never exercise the ~40-line print block --
    ``test_main_dispatches_runners_shadow_status`` takes the not-found branch
    with ``data={}``. The single line whose *absence* is issue #909's actual
    failure mode ("configured, not yet in effect") had zero assertions
    covering it before this test. This drives the real ``cli.main()`` with a
    prologue+cli event pair and a journal, then asserts on the rendered text:
    the pending planner line is present and names the right planner, the
    actuating line names the old one (never the pending one), and the
    load-bearing-gap qualifier note prints *after* "GATE" -- proving the fix
    for the advisor finding that "GATE OPEN" used to be the last word.
    """
    repo_root, fleet_directory, args = _shadow_status_setup(monkeypatch, tmp_path)
    state_file = _allocation_state_file(repo_root)
    _write_allocation_event(state_file, source="prologue", actuating_planner="legacy")
    _write_allocation_event(state_file, source="cli", actuating_planner="new")
    real_change = [{"repo": "x/y", "runner": "r-1", "action": "start", "reason": "test"}]
    _write_journal(
        fleet_directory,
        [
            _journal_record("p1", agreed=True, changes=[]),
            _journal_record("p2", agreed=True, changes=real_change),
        ],
    )

    exit_code = cli.main(["--fleet-dir", str(fleet_directory), "runners", "shadow-status"])

    assert exit_code == 0
    out = capsys.readouterr().out
    actuating_lines = [line for line in out.splitlines() if line.strip().startswith("Actuating")]
    assert len(actuating_lines) == 1, out
    assert "legacy" in actuating_lines[0]
    assert "new" not in actuating_lines[0]
    pending_lines = [line for line in out.splitlines() if "NOT YET IN EFFECT" in line]
    assert len(pending_lines) == 1, out
    assert "new" in pending_lines[0]
    assert "legacy" not in pending_lines[0]
    gate_idx = out.index("Gate ok=")
    note_idx = out.index("Note: criterion 1a")
    assert note_idx > gate_idx, (
        "the qualifier note must print after the gate report, not before it -- "
        "otherwise 'GATE OPEN' is the last word an operator reads, exactly the "
        "advisor-flagged regression this test pins"
    )
    assert "compared 1 time(s)" in out


def test_run_runners_shadow_status_by_action_counts_individual_changes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Per-action streak/total counts individual changes, not passes.

    A single pass can emit several start decisions; each one exercises the
    provisioning path. The split must expose that granularity, not collapse it
    to one per pass.
    """
    _repo_root, fleet_directory, args = _shadow_status_setup(monkeypatch, tmp_path)
    park = [{"repo": "x/y", "runner": "r-1", "action": "park", "reason": "test"}]
    starts = [
        {"repo": "x/y", "runner": "r-1", "action": "start", "reason": "test"},
        {"repo": "x/z", "runner": "r-2", "action": "start", "reason": "test"},
    ]
    _write_journal(
        fleet_directory,
        [
            _journal_record("p1", agreed=True, changes=park),
            _journal_record("p2", agreed=True, changes=park),
            _journal_record("p3", agreed=True, changes=park),
            _journal_record("p4", agreed=True, changes=starts),
            _journal_record("p5", agreed=True, changes=starts),
        ],
    )

    result = cli.run_runners_shadow_status(args)

    assert result.data["change_agreement_streak"] == {"streak": 5, "total": 5}
    by_action = result.data["change_agreement_streak_by_action"]
    assert by_action["park"] == {"streak": 3, "total": 3}
    assert by_action["start"] == {"streak": 4, "total": 4}


def test_run_runners_shadow_status_by_action_trailing_streak_per_action(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Each action's streak breaks when a change of that action disagrees.

    The trailing suffix of changed passes that all agreed is still one suffix,
    but a pass that does not contain a given action does not extend that
    action's trailing count, even if the overall streak continues.
    """
    _repo_root, fleet_directory, args = _shadow_status_setup(monkeypatch, tmp_path)
    park = [{"repo": "x/y", "runner": "r-1", "action": "park", "reason": "test"}]
    start = [{"repo": "x/y", "runner": "r-1", "action": "start", "reason": "test"}]
    _write_journal(
        fleet_directory,
        [
            _journal_record("p1", agreed=True, changes=park),
            _journal_record("p2", agreed=True, changes=start),
            _journal_record("p3", agreed=False, changes=start),
            _journal_record("p4", agreed=True, changes=park),
            _journal_record("p5", agreed=True, changes=park),
        ],
    )

    result = cli.run_runners_shadow_status(args)

    by_action = result.data["change_agreement_streak_by_action"]
    assert by_action["park"] == {"streak": 2, "total": 3}
    assert by_action["start"] == {"streak": 0, "total": 2}


def test_run_runners_shadow_status_by_action_skips_disputed_action_label(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A record where the planners disagree about the action is not bucketed.

    The live planner says ``start`` for a runner, the shadow says ``park`` for
    the same runner. That disagreement is the finding, so the record must not
    be counted toward either action's total or streak.
    """
    _repo_root, fleet_directory, args = _shadow_status_setup(monkeypatch, tmp_path)
    live_start = [{"repo": "x/y", "runner": "r-1", "action": "start", "reason": "test"}]
    shadow_park = [{"repo": "x/y", "runner": "r-1", "action": "park", "reason": "test"}]
    park = [{"repo": "x/y", "runner": "r-1", "action": "park", "reason": "test"}]
    _write_journal(
        fleet_directory,
        [
            _journal_record("p1", agreed=False, changes=live_start, shadow_changes=shadow_park),
            _journal_record("p2", agreed=True, changes=park),
        ],
    )

    result = cli.run_runners_shadow_status(args)

    assert result.data["change_agreement_streak"] == {"streak": 1, "total": 2}
    by_action = result.data["change_agreement_streak_by_action"]
    assert by_action["start"] == {"streak": 0, "total": 0}
    assert by_action["park"] == {"streak": 1, "total": 1}


def test_main_runners_shadow_status_renders_action_split_and_zero_note(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI renders each action with its streak and a note when 0/0."""
    repo_root, fleet_directory, args = _shadow_status_setup(monkeypatch, tmp_path)
    state_file = _allocation_state_file(repo_root)
    _write_allocation_event(state_file, source="prologue", actuating_planner="new")
    park = [{"repo": "x/y", "runner": "r-1", "action": "park", "reason": "test"}]
    _write_journal(
        fleet_directory,
        [
            _journal_record("p1", agreed=True, changes=[]),
            _journal_record("p2", agreed=True, changes=park),
        ],
    )

    exit_code = cli.main(["--fleet-dir", str(fleet_directory), "runners", "shadow-status"])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Agreement streak (passes with a real change)" in out
    assert "park:" in out
    assert "start:" in out
    assert "0/0" in out
    assert "provisioning path" in out
