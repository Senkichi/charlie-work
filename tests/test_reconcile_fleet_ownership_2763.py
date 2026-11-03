"""Issue #2763: ``detect_drift`` only treats a PR the fleet OWNS as an issue's open PR.

Sibling of the ``issue_active_label_with_open_pr`` tests in ``test_reconcile.py``
(split out to keep that module under its file-size ratchet mark). The ownership
predicate itself is covered in ``test_issue_2763_fleet_ownership.py``; these
tests pin its two reconcile call sites, with a positive control for each
negative so an empty result cannot be a fixture that never armed the rule.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest
from _reconcile_fixtures import FakeGitHub, _issue, _pr

from charlie_work.config import OrchestratorConfig
from charlie_work.reconcile import DriftItem, detect_drift
from charlie_work.state import empty_state


_OWNERSHIP_DRIFT_KINDS = {"issue_active_label_with_open_pr", "stale_issue_pr_number"}


def _ownership_drift(
    *,
    labels: list[str],
    head_ref: str,
    issue_entry: dict[str, Any] | None,
    config: OrchestratorConfig,
) -> list[DriftItem]:
    gh = FakeGitHub(
        prs=[_pr(7, "OPEN", head_ref=head_ref, body="Closes #40")],
        issues=[_issue(40, labels)],
    )
    state = empty_state()
    if issue_entry is not None:
        state["issues"]["40"] = issue_entry
    return [
        item for item in detect_drift(gh, state, config) if item.kind in _OWNERSHIP_DRIFT_KINDS
    ]


def test_unowned_keyword_pr_yields_no_open_pr_drift_item() -> None:
    """The headline: a same-repo, off-prefix PR naming an undispatched,
    non-ready issue produces NO drift item -- and the byte-identical fixture
    made owned (the parametrized control below) produces one."""
    config = OrchestratorConfig()

    items = _ownership_drift(
        labels=[], head_ref="session/own-work", issue_entry=None, config=config
    )

    assert items == []


def test_bare_open_passive_entry_and_pr_open_label_do_not_make_a_pr_owned() -> None:
    """The self-legitimising trap: reconcile itself writes ``agent:pr-open`` and
    an ``open_passive`` state entry for whatever PR it adopted, so neither may
    count as ownership -- or adopting a PR once would keep it adopted forever."""
    config = OrchestratorConfig()

    items = _ownership_drift(
        labels=[config.labels.pr_open],
        head_ref="session/own-work",
        issue_entry={"number": 40, "status": "open_passive", "pr_number": 3},
        config=config,
    )

    assert items == []


@pytest.mark.parametrize(
    ("labels", "head_ref", "issue_entry"),
    [
        pytest.param(["READY"], "session/own-work", None, id="ready-label"),
        pytest.param([], "agent/issue-40-x", None, id="dispatch-branch-prefix"),
        pytest.param(
            [],
            "session/own-work",
            {"number": 40, "status": "open_passive", "dispatched_at": "x"},
            id="dispatch-provenance-dispatched_at",
        ),
        pytest.param(
            [],
            "session/own-work",
            {"number": 40, "status": "open_passive", "branch_name": "agent/issue-40-x"},
            id="dispatch-provenance-branch_name",
        ),
    ],
)
def test_owned_pr_still_yields_the_open_pr_drift_item(
    labels: list[str], head_ref: str, issue_entry: dict[str, Any] | None
) -> None:
    """Positive control for the test above: each ownership signal alone
    restores the pre-#2763 behaviour, so the empty result above is the gate's
    doing and not a fixture that never armed the rule."""
    config = OrchestratorConfig()
    labels = [config.labels.ready if label == "READY" else label for label in labels]

    items = _ownership_drift(
        labels=labels, head_ref=head_ref, issue_entry=issue_entry, config=config
    )

    assert [item.kind for item in items] == ["issue_active_label_with_open_pr"]
    assert items[0].add_labels == (config.labels.pr_open,)


def test_unowned_pr_does_not_repoint_the_cached_pr_number() -> None:
    """``stale_issue_pr_number`` must not repoint a tracked issue's cache at a
    human's PR either; the same fixture owned (ready label) does repoint."""
    config = OrchestratorConfig()
    entry = {"number": 40, "status": "open_passive", "pr_number": 3}

    unowned = _ownership_drift(
        labels=[config.labels.pr_open],
        head_ref="session/own-work",
        issue_entry=entry,
        config=config,
    )
    owned = _ownership_drift(
        labels=[config.labels.ready, config.labels.pr_open],
        head_ref="session/own-work",
        issue_entry=entry,
        config=config,
    )

    assert unowned == []
    assert [(item.kind, item.pr_number) for item in owned] == [("stale_issue_pr_number", 7)]


def test_require_fleet_ownership_kill_switch_restores_adoption() -> None:
    """``review.require_fleet_ownership: false`` is the documented kill switch:
    the un-owned fixture is adopted exactly as before the gate existed."""
    config = OrchestratorConfig()
    off = dataclasses.replace(
        config, review=dataclasses.replace(config.review, require_fleet_ownership=False)
    )

    items = _ownership_drift(labels=[], head_ref="session/own-work", issue_entry=None, config=off)

    assert [item.kind for item in items] == ["issue_active_label_with_open_pr"]
