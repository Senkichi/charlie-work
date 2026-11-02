"""Issue #2005: cross-repo blockers must stay repo-qualified in state, events, dead-blocker checks."""

from __future__ import annotations

import json

from charlie_work.github_capabilities.cross_repo_blockers import (
    CrossRepoBlocker,
    blocker_ref,
    blocker_refs,
)
from charlie_work.orchestration.adapters import _is_dead_blocker

FOREIGN = "Senkichi/fresh-eyes"


def test_blocker_refs_survive_json_round_trip_equal_to_themselves():
    refs = blocker_refs([CrossRepoBlocker(60, FOREIGN), 12])
    assert refs == ["senkichi/fresh-eyes#60", 12]
    assert json.loads(json.dumps(refs)) == refs


def test_raw_cross_repo_blocker_would_round_trip_to_a_bare_int():
    # Documents why blocker_ref exists: the int subclass serialises bare.
    raw = [CrossRepoBlocker(60, FOREIGN)]
    assert json.loads(json.dumps(raw)) != raw
    assert blocker_ref(raw[0]) == "senkichi/fresh-eyes#60"


def test_foreign_blocker_is_never_dead_via_same_numbered_local_issue():
    state = {"issues": {"60": {"status": "escalated"}}, "prs": {}}
    assert _is_dead_blocker(60, state, {}) is True
    assert _is_dead_blocker(CrossRepoBlocker(60, FOREIGN), state, {}) is False
