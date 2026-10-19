"""Issue #2757: ``fleet.max_concurrent_sessions_by_harness`` scopes a ceiling to one harness.

``fleet.global_max_concurrent_sessions`` counts every live worker session, but a
provider quota binds only the sessions that draw on it. The per-harness map caps
one harness (e.g. a shared Devin account) without taxing a repo whose worker
chain leads with another. Covered here: config validation, the governor's report
fields, the permit gate (fall-through, clamp, total cap), tier groups, and the
unset-map no-op.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest
from _fakes_github import FakeGitHub
from charlie_work.adapters import AdapterSettings, SessionDispatchResult, SessionRequest
from charlie_work.concurrency_governor_result import ConcurrencyGovernorResult
from charlie_work.config import (
    ConfigError,
    DispatchConfig,
    FleetConfig,
    LabelConfig,
    OrchestratorConfig,
    WorkerRoleConfig,
    build_config_from_data,
)
from charlie_work.host.fakes import FakeSessionCounter, FakeWorkerLauncher
from charlie_work.instrumentation import query_events
from charlie_work.paths import runtime_paths
from charlie_work.role_chain import RoleEntry
from charlie_work.role_selection import RoleSelection, reselect_within_harness_caps
from charlie_work.worker_launch_gate import (
    CLAMPED_BY_HARNESS_MAX,
    WorkerLaunchDeferral,
    WorkerLaunchPermit,
    _launch_workers,
    issue_worker_launch_permit,
)
from charlie_work.workflow import OrchestratorApp

DEVIN = RoleEntry("devin-shell", "swe-2")
OPENCODE = RoleEntry("opencode", "big-pickle")
CLAUDE = RoleEntry("claude-code", "claude-sonnet-5-5")


# --- config ------------------------------------------------------------------


def test_map_defaults_to_empty_immutable_view() -> None:
    fleet = build_config_from_data({}).fleet
    assert dict(fleet.max_concurrent_sessions_by_harness) == {}
    assert isinstance(fleet.max_concurrent_sessions_by_harness, MappingProxyType)


def test_map_parses_and_freezes() -> None:
    data = {"fleet": {"max_concurrent_sessions_by_harness": {"devin-shell": 3, "opencode": 0}}}
    fleet = build_config_from_data(data).fleet
    assert dict(fleet.max_concurrent_sessions_by_harness) == {"devin-shell": 3, "opencode": 0}
    with pytest.raises(TypeError):
        fleet.max_concurrent_sessions_by_harness["devin-shell"] = 9  # type: ignore[index]


@pytest.mark.parametrize(
    ("value", "needle"),
    [
        ({"devin": 3}, "registered harness"),  # adapter_kind spelling, not a harness name
        ({"not-a-harness": 1}, "registered harness"),
        ({"devin-shell": -1}, "devin-shell"),
        ({"devin-shell": True}, "devin-shell"),
        ({"devin-shell": 1.5}, "devin-shell"),
        ({"devin-shell": "3"}, "devin-shell"),
        (["devin-shell"], "max_concurrent_sessions_by_harness"),
        (3, "max_concurrent_sessions_by_harness"),
    ],
)
def test_invalid_map_is_a_config_error_naming_the_key(value: Any, needle: str) -> None:
    with pytest.raises(ConfigError, match=needle) as excinfo:
        build_config_from_data({"fleet": {"max_concurrent_sessions_by_harness": value}})
    assert "fleet.max_concurrent_sessions_by_harness" in str(excinfo.value)


def test_hand_built_fleet_config_is_held_to_the_same_rules() -> None:
    with pytest.raises(ConfigError):
        FleetConfig(max_concurrent_sessions_by_harness={"nope": 1})
    built = FleetConfig(max_concurrent_sessions_by_harness={"devin-shell": 2})
    assert isinstance(built.max_concurrent_sessions_by_harness, MappingProxyType)


# --- governor report fields ----------------------------------------------------


def _gov(**kw: Any) -> ConcurrencyGovernorResult:
    base: dict[str, Any] = {
        "clamped": False,
        "max_concurrent": 0,
        "live_count": 0,
        "available_slots": 5,
        "dispatch_limit": 5,
    }
    return ConcurrencyGovernorResult(**{**base, **kw})


def test_report_fields_unchanged_when_map_unset() -> None:
    gov = _gov(max_concurrent=5)
    assert not gov.harness_enabled
    fields = gov.report_fields()
    assert "harness_concurrency_limits" not in fields
    assert "harness_live_session_counts" not in fields


def test_harness_map_alone_enables_the_report() -> None:
    gov = _gov(
        harness_caps=MappingProxyType({"devin-shell": 3}),
        harness_live_counts=MappingProxyType({"devin-shell": 2}),
    )
    assert gov.any_term_enabled
    assert gov.harness_available("devin-shell") == 1
    assert gov.harness_available("opencode") is None
    fields = gov.report_fields()
    assert fields["harness_concurrency_limits"] == {"devin-shell": 3}
    assert fields["harness_live_session_counts"] == {"devin-shell": 2}


def test_harness_available_never_negative() -> None:
    gov = _gov(
        harness_caps=MappingProxyType({"devin-shell": 1}),
        harness_live_counts=MappingProxyType({"devin-shell": 4}),
    )
    assert gov.harness_available("devin-shell") == 0


# --- pure reselection ----------------------------------------------------------


def _selection(*chain: RoleEntry) -> RoleSelection:
    return RoleSelection(chain=chain, entry=chain[0], index=0, skipped=())


def _free(table: Mapping[str, int]):
    return lambda harness: table.get(harness)


def test_reselect_is_identity_when_selected_harness_has_room_or_no_cap() -> None:
    selection = _selection(DEVIN, OPENCODE)
    assert reselect_within_harness_caps(selection, _free({})) is selection
    assert reselect_within_harness_caps(selection, _free({"devin-shell": 1})) is selection


def test_reselect_falls_through_past_a_full_harness() -> None:
    selection = _selection(DEVIN, OPENCODE)
    moved = reselect_within_harness_caps(selection, _free({"devin-shell": 0}))
    assert moved.entry == OPENCODE
    assert moved.index == 1
    assert moved.is_fallback
    assert [item.entry for item in moved.capped] == [DEVIN]
    assert moved.skipped == ()
    assert moved.report_fields()["harness_capped"] == [
        {"harness": "devin-shell", "model": "swe-2", "index": 0}
    ]


def test_reselect_exhausted_when_every_entry_is_full() -> None:
    selection = _selection(DEVIN)
    moved = reselect_within_harness_caps(selection, _free({"devin-shell": 0}))
    assert moved.entry is None


# --- the permit gate -------------------------------------------------------------


def _app(
    tmp_path: Path,
    *chain: RoleEntry,
    caps: Mapping[str, int] | None = None,
    total_cap: int = 0,
    labels: LabelConfig | None = None,
    issue_labels: tuple[str, ...] = (),
) -> OrchestratorApp:
    primary, *fallbacks = chain
    config = OrchestratorConfig(
        worker=WorkerRoleConfig(
            harness=primary.harness, model=primary.model, fallbacks=tuple(fallbacks)
        ),
        dispatch=DispatchConfig(default_limit=5),
        fleet=FleetConfig(
            global_max_concurrent_sessions=total_cap,
            max_concurrent_sessions_by_harness=caps or {},
        ),
        labels=labels or LabelConfig(),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    gh = FakeGitHub()
    gh.issues[0]["labels"] = [{"name": "automated-ready"}] + [{"name": n} for n in issue_labels]
    return OrchestratorApp(tmp_path, paths, config, gh, fleet_dir_override=str(tmp_path / "fleet"))


def _live(
    fake_host, *, devin: int = 0, opencode: int = 0, total: int | None = None
) -> FakeSessionCounter:
    by_kind = {k: v for k, v in {"devin": devin, "opencode": opencode}.items() if v}
    counter = FakeSessionCounter(
        fleet_workers=(devin + opencode if total is None else total, []),
        fleet_workers_by_adapter_kind=(by_kind, []),
    )
    fake_host(sessions=counter)
    return counter


def _permit(app: OrchestratorApp, requested: int = 5) -> WorkerLaunchPermit:
    permit = issue_worker_launch_permit(app, requested)
    assert isinstance(permit, WorkerLaunchPermit), permit
    return permit


def test_unset_map_changes_nothing(tmp_path: Path, fake_host) -> None:
    counter = _live(fake_host, devin=8, total=8)
    app = _app(tmp_path, DEVIN, OPENCODE, total_cap=20)
    with _permit(app) as permit:
        assert permit.max_launches == 5
        assert permit.role_selection.entry == DEVIN
        assert permit.role_selection.capped == ()
        assert permit.governor.clamped_by is None
        assert not permit.governor.harness_enabled
        assert "harness_concurrency_limits" not in permit.governor.report_fields()
        assert permit._ledger.harness_free == {}
    # No registry walk for per-harness counts when nothing is configured.
    assert not any(name == "fleet_live_workers_by_adapter_kind" for name, _ in counter.calls)


def test_devin_selected_launch_is_clamped_at_the_harness_cap(tmp_path: Path, fake_host) -> None:
    _live(fake_host, devin=3, total=3)
    app = _app(tmp_path, DEVIN, caps={"devin-shell": 3}, total_cap=10)
    with _permit(app) as permit:
        assert permit.max_launches == 0
        assert permit.clamped
        assert permit.governor.clamped_by == CLAMPED_BY_HARNESS_MAX
        assert permit.role_selection.entry == DEVIN  # a deferral, not a reselection
        fields = permit.governor.report_fields()
        assert fields["harness_concurrency_limits"] == {"devin-shell": 3}
        assert fields["harness_live_session_counts"] == {"devin-shell": 3}
        assert fields["clamped_by"] == CLAMPED_BY_HARNESS_MAX


def test_harness_cap_works_without_a_total_cap(tmp_path: Path, fake_host) -> None:
    """The per-harness map alone is enough: the lock and the governor both engage."""
    _live(fake_host, devin=1)
    app = _app(tmp_path, DEVIN, caps={"devin-shell": 1})
    with _permit(app) as permit:
        assert permit.max_launches == 0
        assert permit._launch_lock.valid


def test_partial_room_clamps_to_free_slots(tmp_path: Path, fake_host) -> None:
    _live(fake_host, devin=1, total=1)
    app = _app(tmp_path, DEVIN, caps={"devin-shell": 3}, total_cap=10)
    with _permit(app) as permit:
        assert permit.max_launches == 2
        assert permit.governor.clamped_by == CLAMPED_BY_HARNESS_MAX


def test_opencode_selected_launch_ignores_the_devin_count(tmp_path: Path, fake_host) -> None:
    _live(fake_host, devin=3, total=3)
    app = _app(tmp_path, OPENCODE, caps={"devin-shell": 3}, total_cap=10)
    with _permit(app) as permit:
        assert permit.max_launches == 5
        assert permit.role_selection.entry == OPENCODE
        assert permit.governor.clamped_by is None
        assert permit.governor.report_fields()["harness_live_session_counts"] == {"devin-shell": 3}


def test_total_cap_still_bounds_an_uncapped_harness(tmp_path: Path, fake_host) -> None:
    _live(fake_host, devin=3, total=8)
    app = _app(tmp_path, OPENCODE, caps={"devin-shell": 3}, total_cap=8)
    with _permit(app) as permit:
        assert permit.max_launches == 0
        assert permit.governor.clamped_by == "fleet_max"


def test_chain_falls_through_past_a_full_harness(tmp_path: Path, fake_host) -> None:
    _live(fake_host, devin=3, total=3)
    app = _app(tmp_path, DEVIN, OPENCODE, caps={"devin-shell": 3}, total_cap=10)
    with _permit(app) as permit:
        assert permit.max_launches == 5
        assert permit.role_selection.entry == OPENCODE
        assert permit.role_selection.is_fallback
        assert [item.entry for item in permit.role_selection.capped] == [DEVIN]
        assert permit.governor.clamped_by is None


def test_every_entry_full_defers_with_the_distinct_reason(tmp_path: Path, fake_host) -> None:
    _live(fake_host, devin=2, opencode=1, total=3)
    app = _app(tmp_path, DEVIN, OPENCODE, caps={"devin-shell": 2, "opencode": 1}, total_cap=10)
    with _permit(app) as permit:
        assert permit.max_launches == 0
        assert permit.governor.clamped_by == CLAMPED_BY_HARNESS_MAX
        assert not isinstance(permit, WorkerLaunchDeferral)  # never a provider throttle


def test_zero_cap_blocks_the_harness_outright(tmp_path: Path, fake_host) -> None:
    _live(fake_host)
    app = _app(tmp_path, DEVIN, caps={"devin-shell": 0})
    with _permit(app) as permit:
        assert permit.max_launches == 0


# --- tier groups -------------------------------------------------------------------


def _spy_launch(fake_host) -> list[tuple[str, int]]:
    calls: list[tuple[str, int]] = []

    def _fake(_repo, _manifest, _results, settings: AdapterSettings, requests):
        out = []
        for request in requests:
            calls.append((settings.adapter, request.issue_number))
            out.append(
                SessionDispatchResult(
                    issue_number=request.issue_number,
                    issue_title=request.issue_title,
                    prompt_path=str(request.prompt_path),
                    branch_name=request.branch_name,
                    adapter=settings.adapter,
                    ok=True,
                    pid=4242,
                    process_start_time=1.0,
                )
            )
        return out

    fake_host(worker_launch=FakeWorkerLauncher([_fake]))
    return calls


def _request(number: int, *labels: str) -> SessionRequest:
    return SessionRequest(
        issue_number=number,
        issue_title=f"issue {number}",
        prompt_path=Path(f"prompt-{number}.md"),
        branch_name=f"agent/issue-{number}",
        labels=tuple(labels),
    )


def _settings(app: OrchestratorApp) -> AdapterSettings:
    return app._adapter_settings()


def _tier_fallbacks(app: OrchestratorApp) -> list[dict[str, Any]]:
    events = query_events(app.paths.state_file, kind="worker_model_tier_fallback")
    return [e["payload"] for e in events]


def test_tier_group_on_a_full_capped_harness_falls_back_to_the_permit_selection(
    tmp_path: Path, fake_host
) -> None:
    """Permit selects the uncapped primary, so its clamp says nothing about devin-shell;
    a ``model:swe-2`` group routed onto the full devin entry must still be bounded."""
    _live(fake_host, devin=3, total=3)
    calls = _spy_launch(fake_host)
    app = _app(tmp_path, CLAUDE, DEVIN, caps={"devin-shell": 3}, total_cap=10)
    with _permit(app, 2) as permit:
        assert permit.role_selection.entry == CLAUDE
        results = _launch_workers(
            app, permit, _settings(app), [_request(1, "model:swe-2"), _request(2)]
        )
    assert all(r.ok for r in results)
    assert sorted(calls) == [("claude-code", 1), ("claude-code", 2)]
    assert [p["reason"] for p in _tier_fallbacks(app)] == ["harness_cap"]


def test_tier_group_that_fits_the_capped_harness_launches_and_draws_it_down(
    tmp_path: Path, fake_host
) -> None:
    _live(fake_host, devin=2, total=2)
    calls = _spy_launch(fake_host)
    app = _app(tmp_path, CLAUDE, DEVIN, caps={"devin-shell": 3}, total_cap=10)
    with _permit(app, 3) as permit:
        assert permit._ledger.harness_free == {"devin-shell": 1}
        results = _launch_workers(
            app,
            permit,
            _settings(app),
            [_request(1, "model:swe-2"), _request(2, "model:swe-2"), _request(3)],
        )
        # One free slot: the two-issue tier group does not fit and falls back whole.
        assert permit._ledger.harness_free == {"devin-shell": 1}
    assert all(r.ok for r in results)
    assert {adapter for adapter, _ in calls} == {"claude-code"}

    calls.clear()
    with _permit(app, 1) as permit:
        _launch_workers(app, permit, _settings(app), [_request(4, "model:swe-2")])
        assert permit._ledger.harness_free == {"devin-shell": 0}
    assert calls == [("devin-shell", 4)]


def test_tier_routing_is_untouched_when_the_map_is_unset(tmp_path: Path, fake_host) -> None:
    _live(fake_host, devin=9, total=9)
    calls = _spy_launch(fake_host)
    app = _app(tmp_path, CLAUDE, DEVIN, total_cap=20)
    with _permit(app, 1) as permit:
        _launch_workers(app, permit, _settings(app), [_request(1, "model:swe-2")])
    assert calls == [("devin-shell", 1)]
    assert _tier_fallbacks(app) == []


# --- per-harness live counts -------------------------------------------------------


class _View:
    def __init__(self, adapter_kind: str, alive: bool = True) -> None:
        self.adapter_kind = adapter_kind
        self._alive = alive

    def is_alive(self) -> bool:
        return self._alive


def test_live_counts_by_adapter_kind_count_only_live_sidecars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from charlie_work import live_session_count

    views = [_View("devin"), _View("devin"), _View("devin", alive=False), _View("opencode")]
    monkeypatch.setattr("charlie_work.worker.iter_workers", lambda _dir: views)
    assert live_session_count.count_live_sessions_by_adapter_kind(tmp_path) == {
        "devin": 2,
        "opencode": 1,
    }


def test_fleet_live_counts_by_adapter_kind_sum_across_repos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    from charlie_work import layout
    from charlie_work.fleet_registry import count_fleet_live_sessions_by_adapter_kind

    fleet_dir = tmp_path / ".fleet"
    fleet_dir.mkdir(parents=True)
    repos: dict[str, Any] = {}
    views_by_dir: dict[Path, list[_View]] = {}
    for name, views in (
        ("owner/a", [_View("devin"), _View("devin")]),
        ("owner/b", [_View("devin"), _View("opencode")]),
    ):
        root = tmp_path / name.split("/")[1]
        (root / ".git").mkdir(parents=True)
        state_dir = root / ".var" / "charlie-work"
        sessions_dir = layout.sessions_dir_default(state_dir)
        sessions_dir.mkdir(parents=True)
        views_by_dir[sessions_dir] = views
        repos[name] = {
            "repo_root": str(root),
            "name_with_owner": name,
            "config_path": str(root / "orchestrator.config.yaml"),
            "state_dir": str(state_dir),
            "first_seen": "2024-01-01T00:00:00Z",
            "last_seen": "2024-01-01T00:00:00Z",
        }
    (fleet_dir / "fleet.json").write_text(
        json.dumps({"version": 1, "repos": repos}), encoding="utf-8"
    )
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(fleet_dir))
    monkeypatch.setattr("charlie_work.worker.iter_workers", lambda d: views_by_dir[Path(d)])

    counts, skipped = count_fleet_live_sessions_by_adapter_kind(None)

    assert counts == {"devin": 3, "opencode": 1}
    assert skipped == []
