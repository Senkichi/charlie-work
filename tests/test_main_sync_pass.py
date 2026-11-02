"""run_main_sync_pass: registry walk, self-exclusion, budget, containment, output."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from _main_sync_fixtures import make_sync_repos

from charlie_work import main_sync
from charlie_work.config import OrchestratorConfig
from charlie_work.main_sync import (
    MainSyncPassResult,
    RepoSyncResult,
    format_pass_lines,
    health_entries,
    run_main_sync_pass,
)
from charlie_work.main_sync_state import RepoSyncState, load_sync_state, save_sync_state


def _registry(monkeypatch: pytest.MonkeyPatch, entries: list[tuple[str, Path, Path]]) -> None:
    def fake(_override: str | None, skipped: list[str]) -> Iterator[tuple[Any, ...]]:
        skipped.append("owner/unresolvable")
        for name, root, state_dir in entries:
            state_dir.mkdir(parents=True, exist_ok=True)
            yield name, root, state_dir, OrchestratorConfig()

    monkeypatch.setattr(main_sync, "_iter_registered_repo_contexts", fake)


def _run(tmp_path: Path, exclude: Path, **kwargs: Any) -> MainSyncPassResult:
    return run_main_sync_pass(
        fleet_dir_override=str(tmp_path / "fleet"), exclude_root=exclude, dry_run=False, **kwargs
    )


def test_syncs_registry_and_excludes_own_checkout_by_resolved_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    a = make_sync_repos(tmp_path / "a")
    b = make_sync_repos(tmp_path / "b")
    tip_a = a.push_commit("x.txt", "1\n")
    seed_b = b.head()
    b.push_commit("x.txt", "1\n")
    _registry(monkeypatch, [("o/a", a.clone, tmp_path / "sa"), ("o/b", b.clone, tmp_path / "sb")])

    result = _run(tmp_path, exclude=b.clone / "sub" / "..")  # a different spelling of b

    by_repo = {r.repo: r for r in result.results}
    assert by_repo["o/a"].outcome == "advanced" and a.head() == tip_a
    assert (by_repo["o/b"].outcome, by_repo["o/b"].reason) == ("skipped", "excluded")
    assert b.head() == seed_b
    assert result.skipped_repos == ("owner/unresolvable",)


def test_budget_exhaustion_skips_and_puts_repo_first_next_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    a = make_sync_repos(tmp_path / "a")
    b = make_sync_repos(tmp_path / "b")
    _registry(monkeypatch, [("o/a", a.clone, tmp_path / "sa"), ("o/b", b.clone, tmp_path / "sb")])
    ticks = iter([0.0, 0.0, 999.0])  # deadline set, o/a starts in budget, o/b does not

    first = _run(tmp_path, exclude=tmp_path / "elsewhere", monotonic=lambda: next(ticks))

    assert [(r.repo, r.reason) for r in first.results] == [
        ("o/a", None),
        ("o/b", "budget_exhausted"),
    ]
    assert load_sync_state(tmp_path / "fleet", "o/b").budget_skipped
    second = _run(tmp_path, exclude=tmp_path / "elsewhere")
    assert [r.repo for r in second.results] == ["o/b", "o/a"]
    assert not load_sync_state(tmp_path / "fleet", "o/b").budget_skipped


def test_one_repo_raising_does_not_stop_the_others(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    a = make_sync_repos(tmp_path / "a")
    _registry(
        monkeypatch, [("o/boom", tmp_path, tmp_path / "s0"), ("o/a", a.clone, tmp_path / "sa")]
    )
    real = main_sync.sync_repo

    def flaky(ctx: main_sync.SyncContext) -> RepoSyncResult:
        if ctx.repo == "o/boom":
            raise RuntimeError("surprise")
        return real(ctx)

    monkeypatch.setattr(main_sync, "sync_repo", flaky)
    result = _run(tmp_path, exclude=tmp_path / "elsewhere")
    assert [(r.repo, r.outcome) for r in result.results] == [
        ("o/boom", "failed"),
        ("o/a", "up_to_date"),
    ]


def test_registry_failure_never_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(_override: str | None, _skipped: list[str]) -> Iterator[Any]:
        raise OSError("fleet.json unreadable")

    monkeypatch.setattr(main_sync, "_iter_registered_repo_contexts", broken)
    assert _run(tmp_path, exclude=tmp_path) == MainSyncPassResult()


def test_format_pass_lines_prints_only_actionable_outcomes() -> None:
    result = MainSyncPassResult(
        results=(
            RepoSyncResult("o/a", "advanced", None, "a" * 40, "b" * 40),
            RepoSyncResult("o/b", "advanced", None, "a" * 40, "b" * 40, ref_only=True),
            RepoSyncResult("o/c", "up_to_date"),
            RepoSyncResult("o/d", "skipped", "blocked"),
            RepoSyncResult("o/e", "skipped", "unrelated_history", alarm="unrelated_history"),
            RepoSyncResult("o/f", "skipped", "budget_exhausted"),
        )
    )
    assert format_pass_lines(result) == (
        "o/a advanced aaaaaaaaa..bbbbbbbbb",
        "o/b advanced aaaaaaaaa..bbbbbbbbb (ref only)",
        "o/e skipped unrelated_history ALARM unrelated_history",
        "o/f skipped budget_exhausted",
    )


def test_health_entries_error_on_alarm_ok_on_recovery_none_in_dry_run() -> None:
    results = (
        RepoSyncResult("o/a", "skipped", "unrelated_history", alarm="unrelated_history"),
        RepoSyncResult("o/b", "advanced", None, "a" * 40, "b" * 40, recovered=True),
        RepoSyncResult("o/c", "advanced", None, "a" * 40, "b" * 40),
    )
    entries = health_entries(MainSyncPassResult(results=results))
    assert [(e.adapter_kind, e.health, e.issue_number) for e in entries] == [
        ("main-sync o/a", "ERROR", -1),
        ("main-sync o/b", "OK", -1),
    ]
    assert health_entries(MainSyncPassResult(results=results, dry_run=True)) == ()


def test_dry_run_pass_writes_no_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    a = make_sync_repos(tmp_path / "a")
    _registry(monkeypatch, [("o/a", a.clone, tmp_path / "sa")])
    save_sync_state(tmp_path / "fleet", "o/a", RepoSyncState(budget_skipped=True))
    result = run_main_sync_pass(
        fleet_dir_override=str(tmp_path / "fleet"), exclude_root=tmp_path, dry_run=True
    )
    assert result.dry_run and result.results[0].outcome == "up_to_date"
    assert load_sync_state(tmp_path / "fleet", "o/a").budget_skipped  # untouched
