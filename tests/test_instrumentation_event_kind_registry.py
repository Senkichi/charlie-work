"""Event-kind registry tests for ``charlie_work.instrumentation``.

Split out of ``tests/test_instrumentation.py`` (issue #1569, Track-1):
``_LEVEL_BY_KIND`` membership and level assertions (#910/#1258/#1271),
the exhaustive emit-site registry scan (#910/#995) with its allow-list,
and the checks that independently verify each allow-list claim. The
scanner machinery lives in ``tests/_instrumentation_kind_scanner.py``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from _instrumentation_kind_scanner import (
    _ALLOWED_UNRESOLVED_KIND_SITES,
    _known_level,
    _scan_event_kinds,
    _scan_sweep_append_kinds,
)
from charlie_work.instrumentation import (
    EXPECTED_OPERATIONAL_KINDS,
    _LEVEL_BY_KIND,
    close_db,
    log_event,
    read_event_log,
)


@pytest.fixture(autouse=True)
def _close_db_after_test(tmp_path: Path) -> None:
    """Ensure DB connections are closed between tests to avoid cross-test contamination."""
    yield
    # Close any connections that were opened during this test
    close_db(tmp_path / "state.json")
    # Also try the variant paths used in some tests
    close_db(tmp_path / "subdir" / "state.json")
    close_db(tmp_path / "nonexistent_dir" / "state.json")


def test_event_kind_registry_exhaustive() -> None:
    """#910/#995: every emit-site kind in this package is registered or accounted for.

    Every kind resolvable to a literal set must be a member of
    ``_LEVEL_BY_KIND`` (or a registered ``_sweep`` variant). Every site the
    scanner cannot resolve must be in ``_ALLOWED_UNRESOLVED_KIND_SITES`` with
    a reason -- an unresolved, unlisted site fails the build instead of
    silently contributing nothing (#995), and a listed entry that no longer
    matches a real unresolved site fails too (a stale allow-list is a lie).
    """
    src_root = Path(__file__).parents[1] / "src" / "charlie_work"
    used, unresolved = _scan_event_kinds(src_root)

    # ci_fleet is a separate package (a sibling repo, not owned by this PR)
    # that logs through this package's sink. Its literal kinds still belong
    # in the registry check below -- an unregistered kind is a real bug
    # regardless of which repo introduced it. But its *unresolved* sites
    # deliberately do NOT feed the fail-closed assertions further down: this
    # test can only allow-list (and can only fix) unresolved sites in the
    # charlie_work tree it ships. Enforcing ci_fleet's unresolved sites here
    # would make charlie_work's CI fail on a file no charlie_work change
    # touched, and would require an allow-list entry this repo can't attach
    # a supporting test to. If ci_fleet needs the same guard, it belongs in
    # ci_fleet's own test suite, scanning its own source.
    spec = importlib.util.find_spec("ci_fleet")
    if spec is not None and spec.origin:
        ci_root = Path(spec.origin).parent
        ci_used, _ci_unresolved_not_enforced_here = _scan_event_kinds(ci_root)
        used |= ci_used

    unregistered = {k for k in used if not _known_level(k)}
    assert not unregistered, f"unregistered event kinds: {sorted(unregistered)}"

    allowed_keys = {site.key for site in _ALLOWED_UNRESOLVED_KIND_SITES}
    found_keys = {site.key for site in unresolved}

    unaccounted = [site for site in unresolved if site.key not in allowed_keys]
    assert not unaccounted, (
        "unresolved event-kind expression(s) not in _ALLOWED_UNRESOLVED_KIND_SITES "
        "(either make the expression statically resolvable, pass an explicit "
        "level= at the call site, or add a reasoned allow-list entry): "
        + "; ".join(
            f"{site.path}:{site.lineno} in {site.scope}(): `{site.source}`" for site in unaccounted
        )
    )

    stale = [site for site in _ALLOWED_UNRESOLVED_KIND_SITES if site.key not in found_keys]
    assert not stale, (
        "_ALLOWED_UNRESOLVED_KIND_SITES entry no longer matches any unresolved "
        "site -- remove it or update it to match the current source: "
        + "; ".join(f"{site.path} in {site.scope}(): `{site.source}`" for site in stale)
    )


def test_review_dispatch_skipped_ci_red_kind_registered_matches_family() -> None:
    """Issue #1258: the janitor's CI-red short-circuit (sole-failure and the
    new co-occurring-failure branch alike) must have a dedicated, registered
    provenance kind -- previously it only produced whatever generic
    ``record_review`` itself logs, with nothing naming the deterministic
    gate as the decision's source.

    Pinned to the ``review_dispatch_*`` family per the issue's binding
    comment (which corrects the plan body's originally-proposed
    ``review_skipped_ci_red`` naming) so it groups with
    ``review_dispatch_claim``/``review_dispatch`` for ``event_counts_by_kind``
    roll-ups, and pinned to level ``info``: this is the deterministic gate
    doing its routine job (routing to rework without ever starting a paid
    reviewer session), not a condition that ended a lane or lost work.
    """
    assert "review_dispatch_skipped_ci_red" in _LEVEL_BY_KIND
    assert _LEVEL_BY_KIND["review_dispatch_skipped_ci_red"] == "info"
    assert "review_dispatch_skipped_ci_red".startswith("review_dispatch_")

    # Deferral (d): the stale/absent-checks auto-retrigger is W17's, landing
    # after this item in the lane -- no retrigger emitter exists yet, so no
    # retrigger-family kind may be registered here. A registered-but-unused
    # kind would be exactly as misleading as an emitted-but-unregistered one:
    # it would claim a mechanism exists that this diff never builds.
    retrigger_kinds = {
        kind
        for kind in _LEVEL_BY_KIND
        if kind.startswith("review_dispatch_") and "retrigger" in kind
    }
    assert not retrigger_kinds, (
        f"no retrigger-family kind may be registered by this item (W17's job): {retrigger_kinds}"
    )


def test_expected_operational_kinds_are_all_registered_warnings() -> None:
    """#1271: bucketing only makes sense for warnings.

    Every member of ``EXPECTED_OPERATIONAL_KINDS`` must be registered in
    ``_LEVEL_BY_KIND`` at ``"warning"`` -- an info or error kind (or an
    unregistered one) accidentally added to the set would silently vanish
    from ``check_error_events``'s coverage or from the info stream, since
    ``check_warning_events`` only ever queries ``level = 'warning'`` rows.
    """
    assert EXPECTED_OPERATIONAL_KINDS, "the set must not be empty"
    for kind in EXPECTED_OPERATIONAL_KINDS:
        assert kind in _LEVEL_BY_KIND, f"{kind} is not registered in _LEVEL_BY_KIND"
        assert _LEVEL_BY_KIND[kind] == "warning", (
            f"{kind} is registered at level {_LEVEL_BY_KIND[kind]!r}, not 'warning' -- "
            "bucketing only makes sense for warning-level kinds"
        )


def test_self_deploy_event_kind_only_returns_registered_kinds() -> None:
    """#995: independently verify the claim behind the supervise.py allow-list entry.

    ``_self_deploy_event_kind`` is call-based, so the static scanner cannot
    resolve it and it is allow-listed above on the strength of this test.
    This enumerates every branch of the function (by constructing a
    ``SelfDeployResult`` for each) and checks each returned kind against the
    registry, so the allow-list's claim is enforced on every run rather than
    trusted from a comment.
    """
    from charlie_work.supervise import SelfDeployResult, _self_deploy_event_kind

    failed = SelfDeployResult(ok=False, pulled=False, changed=False, synced=False)
    succeeded_changed = SelfDeployResult(ok=True, pulled=True, changed=True, synced=True)
    succeeded_venv_repaired = SelfDeployResult(
        ok=True, pulled=True, changed=False, synced=False, venv_repaired=True
    )
    skipped = SelfDeployResult(ok=True, pulled=True, changed=False, synced=False)

    for result in (failed, succeeded_changed, succeeded_venv_repaired, skipped):
        kind = _self_deploy_event_kind(result)
        assert _known_level(kind), f"{kind} (from {result}) is not registered"

    assert _self_deploy_event_kind(failed) == "self_deploy_failed"
    assert _self_deploy_event_kind(succeeded_changed) == "self_deploy_succeeded"
    assert _self_deploy_event_kind(succeeded_venv_repaired) == "self_deploy_succeeded"
    assert _self_deploy_event_kind(skipped) == "self_deploy_skipped"


def test_sweep_event_append_kinds_are_registered() -> None:
    """#995: independently verify the claim behind the two `_append_sweep_events` entries.

    `_append_sweep_events`'s `kind` loop variable is allow-listed above
    because the real literal is chosen at each `sweep_events.append((kind,
    payload))` call site, not in the loop. This scans for exactly those call
    sites directly and checks every literal they contribute against the
    registry, so that claim is enforced rather than trusted.
    """
    src_root = Path(__file__).parents[1] / "src" / "charlie_work"
    found, unresolved = _scan_sweep_append_kinds(src_root)

    assert not unresolved, (
        "sweep_events.append((kind, payload)) site(s) with an unresolvable "
        "kind element -- make it a literal or trace it manually: " + "; ".join(unresolved)
    )
    assert found, "expected at least one sweep_events.append((kind, payload)) call site"
    unregistered = {k for k in found if not _known_level(k)}
    assert not unregistered, f"unregistered sweep_events kinds: {sorted(unregistered)}"


def test_issue_910_active_kinds_are_error_or_warning(tmp_path: Path) -> None:
    """#910: the 11 production-missed active kinds are now classified.

    The table from the issue body; two rows (review_dispatch_escalated and
    review_verdict_missed) were also discussed in the co-occurrence comment,
    which did not change their enrollment in the error stream. Their levels are
    the same as the issue's proposed table.
    """
    expected = {
        "review_verdict_missed": "error",
        "review_dispatch_escalated": "error",
        "merge_failed_attempt_alarm": "error",
        "dispatch_blocked_chain_dead": "error",
        "flake_rerun_failed": "warning",
        "quota_probe_failed": "warning",
        "janitor_rework_escalated": "error",
        "merge_deferred_stale_base_alarm": "error",
        "janitor_rework_stalled": "warning",
        "supervise_relaunch_cap_reached": "warning",
    }
    for kind, level in expected.items():
        assert _LEVEL_BY_KIND[kind] == level, f"{kind} should be {level!r}"


def test_issue_910_latent_kinds_are_classified(tmp_path: Path) -> None:
    """#910: the additional unclassified but zero-event kinds are enrolled."""
    expected = {
        "infra_rerun_failed": "warning",
        "infra_rerun_escalated": "error",
        "reconcile_pass_failed": "error",
        "session_budget_exceeded": "warning",
        "deescalation_cap_exhausted": "warning",
        "required_changes_vacuous": "warning",
        "rescue_review_escalated": "error",
        "janitor_rework_cycle_failed": "error",
        "worktree_foreign_writer": "warning",
    }
    for kind, level in expected.items():
        assert _LEVEL_BY_KIND[kind] == level, f"{kind} should be {level!r}"


def test_sweep_inherits_base_level(tmp_path: Path) -> None:
    """Sweep-aggregated kinds (``{base}_sweep``) inherit the base kind's level."""
    state_path = tmp_path / "state.json"
    log_event(state_path, "review_dispatch_stalled_sweep", {"count": 3})
    log_event(state_path, "orphaned_worker_drift_sweep", {"count": 5})
    log_event(state_path, "unknown_kind_sweep", {"count": 1})

    events = read_event_log(state_path)
    levels = {e["kind"]: e["level"] for e in events}
    assert levels["review_dispatch_stalled_sweep"] == "error"
    assert levels["orphaned_worker_drift_sweep"] == "info"
    assert levels["unknown_kind_sweep"] == "info"
