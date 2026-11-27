"""Driving the foreman pass: dry-run modes, locks, pass gates, CLI (issue #3122).

Split out of the original ``test_foreman_pass.py`` under the file-size
ratchet (issue #1442): this module covers the pass from the operator's
side — ``--dry-run``/``--no-decide`` Turn modes, pass.lock/turn.lock
contention, the enabled/--only/no-foremen gates, and the ``charlie
scheme`` CLI seam itself. Wake-predicate and deferral coverage lives in
``test_foreman_pass.py``; Turn bookkeeping and resume in
``test_foreman_pass_turn.py``; shared builders in
``tests/_foreman_pass_fixtures.py``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from _foreman_collect_helpers import NOW, REPO_KEY
from _foreman_pass_fixtures import _run, _succeeding_decide, _world
from _worktree_fixtures import (
    _wt_scratch as _register_wt_scratch,  # noqa: F401 -- registers the wt_scratch fixture
)

from charlie_work.file_lock import try_acquire_byte_range_lock
from charlie_work.foreman import paths as f_paths
from charlie_work.foreman import pass_run
from charlie_work.foreman.state import read_state_snapshot
from charlie_work.foreman.store import scaffold_foreman
from charlie_work.local_issues import github_client_for
from charlie_work.github import GitHub


def _stamp(now: datetime) -> str:
    return now.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


def test_dry_run_prints_plan_and_changes_nothing(
    tmp_path: Path, wt_scratch: Path, monkeypatch: pytest.MonkeyPatch, fake_host
) -> None:
    world = _world(tmp_path, wt_scratch, monkeypatch)
    _succeeding_decide(
        fake_host,
        actions=[
            {
                "type": "file_issue",
                "key": "k1",
                "repo": REPO_KEY,
                "title": "Would file",
                "body": "body",
            }
        ],
    )

    passed = _run(world, dry_run=True)

    outcome = passed.outcomes[0]
    assert outcome.outcome == "dry_run"
    plan = json.loads(outcome.detail or "{}")
    assert plan["plan"][0]["type"] == "file_issue"
    # The dry Turn uses the dry- prefix so it is never a resume candidate.
    assert outcome.turn_id is not None
    assert outcome.turn_id.startswith(f_paths.DRY_TURN_DIR_PREFIX)
    # State is untouched and no issue was filed.
    state = read_state_snapshot(world.foreman_root)
    assert state.turns_total == 3
    assert state.last_turn_id is None
    gh = github_client_for(world.repo_root, world.foreman().config, github=GitHub, dry_run=False)
    assert gh.issue_list(state="open") == []


def test_dry_run_no_decide_stops_after_briefing(
    tmp_path: Path, wt_scratch: Path, monkeypatch: pytest.MonkeyPatch, fake_host
) -> None:
    world = _world(tmp_path, wt_scratch, monkeypatch)
    launcher = _succeeding_decide(fake_host)

    passed = _run(world, dry_run=True, no_decide=True)

    outcome = passed.outcomes[0]
    assert outcome.outcome == "dry_run" and outcome.reason == "no_decide"
    assert launcher.calls == []
    assert f_paths.briefing_path(world.foreman_root, outcome.turn_id).is_file()
    assert read_state_snapshot(world.foreman_root).turns_total == 3


def test_no_decide_requires_dry_run(
    tmp_path: Path, wt_scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = _world(tmp_path, wt_scratch, monkeypatch)
    result = pass_run.run_pass(world.fleet_dir, no_decide=True)
    assert not result.ok
    assert "--no-decide" in (result.error or "")


# ---------------------------------------------------------------------------
# Locking and pass-level gates
# ---------------------------------------------------------------------------


def test_pass_lock_contention_short_circuits(
    tmp_path: Path, wt_scratch: Path, monkeypatch: pytest.MonkeyPatch, fake_host
) -> None:
    world = _world(tmp_path, wt_scratch, monkeypatch)
    _succeeding_decide(fake_host)
    held = try_acquire_byte_range_lock(f_paths.pass_lock_path(world.fleet_dir))
    assert held is not None
    try:
        passed = _run(world)
        assert passed.locked is True
        assert passed.outcomes == ()
    finally:
        held.release()


def test_turn_lock_contention_skips_the_foreman(
    tmp_path: Path, wt_scratch: Path, monkeypatch: pytest.MonkeyPatch, fake_host
) -> None:
    world = _world(tmp_path, wt_scratch, monkeypatch)
    launcher = _succeeding_decide(fake_host)
    held = try_acquire_byte_range_lock(f_paths.turn_lock_path(world.foreman_root))
    assert held is not None
    try:
        passed = _run(world)
        outcome = passed.outcomes[0]
        assert outcome.outcome == "skipped"
        assert outcome.reason == "turn lock held"
        assert launcher.calls == []
    finally:
        held.release()


def test_pass_disabled_by_config(
    tmp_path: Path, wt_scratch: Path, monkeypatch: pytest.MonkeyPatch, fake_host
) -> None:
    world = _world(tmp_path, wt_scratch, monkeypatch)
    launcher = _succeeding_decide(fake_host)
    (world.fleet_dir / "config.yaml").write_text("foreman:\n  enabled: false\n", encoding="utf-8")

    passed = _run(world)

    assert passed.disabled is True
    assert passed.outcomes == ()
    assert launcher.calls == []


def test_no_foremen_is_an_empty_pass(tmp_path: Path) -> None:
    fleet_dir = tmp_path / "fleet"
    fleet_dir.mkdir(parents=True)
    result = pass_run.run_pass(fleet_dir, now=NOW)
    assert result.ok
    assert result.value.outcomes == ()
    assert not result.value.locked and not result.value.disabled


def test_only_restricts_the_pass(
    tmp_path: Path, wt_scratch: Path, monkeypatch: pytest.MonkeyPatch, fake_host
) -> None:
    world = _world(tmp_path, wt_scratch, monkeypatch)
    _succeeding_decide(fake_host)
    other = scaffold_foreman(world.fleet_dir, "g2")
    assert other.ok, other.error
    g2_root = world.fleet_dir / "foremen" / "g2"
    (g2_root / "charter.yaml").write_text(
        (world.foreman_root / "charter.yaml")
        .read_text(encoding="utf-8")
        .replace("name: g1", "name: g2"),
        encoding="utf-8",
    )
    # scaffold_foreman seeds state.json paused — g2 is a skip, not a Turn.

    passed = _run(world)
    assert {outcome.name for outcome in passed.outcomes} == {"g1", "g2"}

    passed = _run(world, only="g2")
    assert [(o.name, o.outcome) for o in passed.outcomes] == [("g2", "skipped")]


def test_only_unknown_foreman_is_an_error(
    tmp_path: Path, wt_scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = _world(tmp_path, wt_scratch, monkeypatch)
    result = pass_run.run_pass(world.fleet_dir, only="nope", now=NOW)
    assert not result.ok
    assert "nope" in (result.error or "")


# ---------------------------------------------------------------------------
# The CLI seam
# ---------------------------------------------------------------------------


def test_scheme_command_via_cli(
    tmp_path: Path,
    wt_scratch: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_host,
    capsys: pytest.CaptureFixture,
) -> None:
    world = _world(tmp_path, wt_scratch, monkeypatch)
    launcher = _succeeding_decide(fake_host)
    # cli.main runs the pass at real wall time — refresh the seeded snapshot
    # (stamped at the suite's fixed NOW) past the staleness bound.
    world.write_snapshot([], written_at=_stamp(datetime.now(UTC)))

    from charlie_work import cli

    rc = cli.main(
        [
            "--fleet-dir",
            str(world.fleet_dir),
            "scheme",
            "--dry-run",
            "--no-decide",
        ]
    )

    assert rc == 0
    assert launcher.calls == []
    assert "dry-run" in capsys.readouterr().out


def test_scheme_command_reports_locked(
    tmp_path: Path,
    wt_scratch: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    world = _world(tmp_path, wt_scratch, monkeypatch)
    held = try_acquire_byte_range_lock(f_paths.pass_lock_path(world.fleet_dir))
    assert held is not None
    try:
        from charlie_work import cli

        rc = cli.main(["--fleet-dir", str(world.fleet_dir), "scheme"])
    finally:
        held.release()

    assert rc == 0
    assert "pass.lock" in capsys.readouterr().out
