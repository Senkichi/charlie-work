"""Stale ``issues[N].pr_number`` cache-pointer repair tests (issue #1498).

Split out of ``test_fix_reconcile_closed_unmerged.py`` to keep that module
under the 800-line file-size ratchet cap (issue #1442) — and because the
``stale_issue_pr_number`` drift kind is not one of the ``closed_unmerged_*``
kinds that module's docstring scopes it to. Every test here exercises the
write-side repair that repoints a tracked issue's cached ``pr_number`` at its
current open linked PR.

Reuses the lightweight ``FakeGitHub``/``_pr``/``_issue`` fixtures defined in
``test_reconcile.py``'s shared fixture module ``_reconcile_fixtures.py``
(pytest's rootless import mode makes this a plain top-level import, the same
pattern ``test_fix_reconcile.py`` itself uses).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from charlie_work.config import OrchestratorConfig
from charlie_work.reconcile import apply_fixes, detect_drift
from charlie_work.state import empty_state

from _reconcile_fixtures import FakeGitHub, _issue, _pr


# ---------------------------------------------------------------------------
# Issue #1498: the #1398 guard's uncovered variant. ``state["issues"][N]
# ["pr_number"]`` is only written by the orchestrator's own PR-opening paths
# (dispatch salvage / orphaned-branch recovery). A PR that links itself to
# an issue via its own closing keyword -- opened outside those paths --
# never updates the cache, so it can keep pointing at a dead closed-unmerged
# PR while a live open PR also links the issue. That staleness defeats
# ``_closed_pr_superseded_by_newer_session``'s signal 1 (cached ==
# closed PR being evaluated, so "issue moved on" cannot be proven) and the
# two rules then flap ``agent:pr-open`` off/on every other pass (issue
# #1068: 23 add/remove transitions in 16h). The ``stale_issue_pr_number``
# drift kind repoints the cache at the issue's current open PR -- the same
# ``min(open PR number)`` pick ``issue_active_label_with_open_pr`` uses --
# so the supersession guard sees the live PR on the next pass and both
# rules converge.
# ---------------------------------------------------------------------------


def test_stale_issue_pr_number_repoints_cache_to_open_linked_pr() -> None:
    """Detection: cached ``pr_number`` names a CLOSED-unmerged PR while a
    different OPEN PR links the issue via its own closing keyword. The drift
    item must carry the open PR's number and ``apply_fixes`` must rewrite
    the cached pointer, preserving the entry's other fields. A second pass
    over the corrected state must emit nothing (self-heals)."""
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[
            # Closing-keyword bodies -- not branch names -- are what binds
            # both PRs to issue #1068, mirroring the real incident's
            # closingIssuesReferences linkage.
            _pr(1214, "CLOSED", head_ref="fix/dead-attempt", body="Closes #1068\n\nstale"),
            _pr(1405, "OPEN", head_ref="fix/janitor-gate", body="Closes #1068\n\nlive"),
        ],
        # Labels already converged (agent:pr-open present) so the sibling
        # label repair does not fire and the pointer repair is isolated.
        issues=[_issue(1068, [config.labels.ready, config.labels.pr_open])],
    )
    state = empty_state()
    state["issues"]["1068"] = {
        "number": 1068,
        "status": "open_passive",
        "pr_number": 1214,
        "title": "cached title",
    }

    drift = [
        item for item in detect_drift(gh, state, config) if item.kind == "stale_issue_pr_number"
    ]
    assert len(drift) == 1
    assert drift[0].issue_number == 1068
    assert drift[0].pr_number == 1405

    new_state = apply_fixes(gh, state, drift, config)
    assert new_state["issues"]["1068"]["pr_number"] == 1405
    # Other fields are preserved by the overlay write.
    assert new_state["issues"]["1068"]["status"] == "open_passive"
    assert new_state["issues"]["1068"]["title"] == "cached title"

    second = [
        item
        for item in detect_drift(gh, new_state, config)
        if item.kind == "stale_issue_pr_number"
    ]
    assert second == []


def test_stale_issue_pr_number_picks_lowest_open_linked_pr() -> None:
    """Tie-break: when several open PRs link the same issue, the cache
    repoints to the LOWEST PR number -- the same
    ``min(int(pr["number"]) for pr in open_prs)`` pick
    ``issue_active_label_with_open_pr`` reports."""
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[
            _pr(1501, "OPEN", head_ref="fix/b", body="Closes #1068"),
            _pr(1405, "OPEN", head_ref="fix/a", body="Closes #1068"),
        ],
        issues=[_issue(1068, [config.labels.pr_open])],
    )
    state = empty_state()
    state["issues"]["1068"] = {
        "number": 1068,
        "status": "open_passive",
        "pr_number": 1501,
    }

    drift = [
        item for item in detect_drift(gh, state, config) if item.kind == "stale_issue_pr_number"
    ]
    assert len(drift) == 1
    assert drift[0].pr_number == 1405

    new_state = apply_fixes(gh, state, drift, config)
    assert new_state["issues"]["1068"]["pr_number"] == 1405


def test_stale_issue_pr_number_populates_missing_key() -> None:
    """An issue state entry with no ``pr_number`` key at all (e.g. the
    externally-linked PR was never recorded) still gets the pointer
    written -- ``None`` counts as differing from the open PR's number."""
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[_pr(1405, "OPEN", head_ref="fix/janitor-gate", body="Closes #1068")],
        issues=[_issue(1068, [config.labels.pr_open])],
    )
    state = empty_state()
    state["issues"]["1068"] = {"number": 1068, "status": "open_passive"}

    drift = [
        item for item in detect_drift(gh, state, config) if item.kind == "stale_issue_pr_number"
    ]
    assert len(drift) == 1

    new_state = apply_fixes(gh, state, drift, config)
    assert new_state["issues"]["1068"]["pr_number"] == 1405


def test_stale_issue_pr_number_does_not_fire_when_cache_matches_open_pr() -> None:
    """Idempotency guard: cached ``pr_number`` already naming the open
    linked PR is not drift -- the kind must not re-fire every pass."""
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[_pr(1405, "OPEN", head_ref="fix/janitor-gate", body="Closes #1068")],
        issues=[_issue(1068, [config.labels.pr_open])],
    )
    state = empty_state()
    state["issues"]["1068"] = {
        "number": 1068,
        "status": "open_passive",
        "pr_number": 1405,
    }

    drift = [
        item for item in detect_drift(gh, state, config) if item.kind == "stale_issue_pr_number"
    ]
    assert drift == []


def test_stale_issue_pr_number_does_not_fire_without_open_linked_pr() -> None:
    """No open linked PR means there is nothing truthful to repoint to --
    a closed-unmerged-only linkage is the closed-unmerged rules' shape,
    not this one's."""
    config = OrchestratorConfig()
    now = datetime.now(UTC)
    closed_at = (now - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    gh = FakeGitHub(
        prs=[
            _pr(
                1214,
                "CLOSED",
                head_ref="fix/dead-attempt",
                body="Closes #1068",
                closed_at=closed_at,
            )
        ],
        issues=[_issue(1068, [config.labels.ready])],
    )
    state = empty_state()
    state["issues"]["1068"] = {"number": 1068, "pr_number": 1214}

    drift = [
        item for item in detect_drift(gh, state, config) if item.kind == "stale_issue_pr_number"
    ]
    assert drift == []


def test_stale_issue_pr_number_does_not_fire_for_closed_issue() -> None:
    """A CLOSED GitHub issue's ``pr_number`` is a historical record of
    which PR resolved it (``_merged_issue_fields`` preserves it verbatim),
    not a live pointer to repair -- an open PR linking it later must not
    overwrite that record."""
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[_pr(1405, "OPEN", head_ref="fix/janitor-gate", body="Closes #1068")],
        issues=[_issue(1068, [config.labels.done], state="CLOSED")],
    )
    state = empty_state()
    state["issues"]["1068"] = {
        "number": 1068,
        "status": "closed",
        "pr_number": 1214,
    }

    drift = [
        item for item in detect_drift(gh, state, config) if item.kind == "stale_issue_pr_number"
    ]
    assert drift == []


def test_stale_issue_pr_number_self_heal_stops_pr_open_label_flap() -> None:
    """The #1068 regression shape, end to end: cached ``pr_number`` names
    the closed-unmerged PR #1214 while the open PR #1405 also links issue
    #1068, and the issue carries ``agent:pr-open``.

    Without the repoint, ``closed_unmerged_pr_active_labels`` (strips
    ``agent:pr-open``) and ``issue_active_label_with_open_pr`` (re-adds it)
    alternate forever -- 23 transitions over 16h in production. Once the
    cache heals to #1405, the #1398 supersession guard sees the live PR on
    the next pass and the label state reaches a fixed point: two
    consecutive reconcile passes must leave the label set untouched.
    """
    config = OrchestratorConfig()
    now = datetime.now(UTC)
    closed_at = (now - timedelta(days=5)).isoformat().replace("+00:00", "Z")
    issue = _issue(1068, [config.labels.ready, config.labels.pr_open])
    gh = FakeGitHub(
        prs=[
            _pr(
                1214,
                "CLOSED",
                head_ref="fix/dead-attempt",
                body="Closes #1068\n\nsuperseded attempt",
                closed_at=closed_at,
            ),
            _pr(
                1405,
                "OPEN",
                head_ref="fix/janitor-cross-pr-revert",
                body="Closes #1068\n\nfix(janitor): detect_cross_pr_revert fails closed",
            ),
        ],
        issues=[issue],
    )
    state = empty_state()
    # The live #1068 entry shape: pr_number still names the dead PR, and
    # there is no ``dispatched_at`` for the guard's signal 2 (only
    # dispatch_pending_at / terminal_since from the long-terminated
    # pending session).
    state["issues"]["1068"] = {
        "number": 1068,
        "status": "open_passive",
        "pr_number": 1214,
        "dispatch_pending_at": (now - timedelta(days=6)).isoformat().replace("+00:00", "Z"),
        "terminal_since": (now - timedelta(days=6)).isoformat().replace("+00:00", "Z"),
    }

    flap_kinds = {
        "closed_unmerged_pr_active_labels",
        "closed_unmerged_pr_issue_state_converged",
        "issue_active_label_with_open_pr",
        "stale_issue_pr_number",
    }
    label_history: list[frozenset[str]] = []
    kinds_by_pass: list[set[str]] = []
    for _ in range(4):
        drift = detect_drift(gh, state, config)
        state = apply_fixes(gh, state, drift, config)
        # Mirror the applied label writes onto the fake's served payload the
        # way the real API's next ``issue list`` reflects prior mutations --
        # FakeGitHub only records the calls.
        names = {entry["name"] for entry in issue["labels"]}
        names.difference_update(label for n, label in gh.labels_removed if n == 1068)
        names.update(label for n, label in gh.labels_added if n == 1068)
        issue["labels"] = [{"name": name} for name in sorted(names)]
        gh.labels_added.clear()
        gh.labels_removed.clear()
        label_history.append(frozenset(names))
        kinds_by_pass.append({item.kind for item in drift if item.issue_number == 1068})

    # The cache repointed to the live open PR on the first pass and the
    # drift kind fired exactly once -- never again.
    assert state["issues"]["1068"]["pr_number"] == 1405
    assert "stale_issue_pr_number" in kinds_by_pass[0]
    assert all("stale_issue_pr_number" not in kinds for kinds in kinds_by_pass[1:])

    # Convergence: the label set is a fixed point across the last two
    # consecutive passes (the remove/add oscillation is over).
    assert (
        label_history[-1]
        == label_history[-2]
        == frozenset({config.labels.ready, config.labels.pr_open})
    )

    # The final pass emitted no flap-class drift for the issue at all:
    # the #1398 supersession guard now sees the live PR and both
    # closed-unmerged rules skip it.
    assert kinds_by_pass[-1] & flap_kinds == set()
