"""Shared fixtures for the #502 unauthorized-merge tripwire tests.

Hoisted out of ``test_charlie_work.py`` (issue #1284): arming/acking helpers
for the tripwire's baseline and acknowledgement state, a merged
worker-branch PR shape, a minimal merge-check app builder, and a
review-decision.json writer, all imported by the dedicated
``test_tripwire_*``/``test_merge_authorize.py``/
``test_verdict_provenance_enforcement.py`` modules.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from _fakes_github import FakeGitHub
from _review_fixtures import _required_checks_config
from charlie_work.paths import runtime_paths
from charlie_work.workflow import OrchestratorApp


def _arm_unauthorized_merge_tripwire(paths, pre_existing: tuple[int, ...] = ()) -> None:
    """Declare the #502 tripwire already armed, so a test sees its steady state.

    The tripwire's first pass over a fresh state records a baseline of the merges
    that predate it and reports nothing (see
    ``OrchestratorApp._apply_unauthorized_merge_baseline``). Any test asserting on
    findings must therefore say whether it is exercising the arming pass or the
    steady state. Calling this with no arguments means "armed, and there was no
    pre-existing backlog", which is the condition every pre-baseline tripwire test
    was implicitly written against.
    """
    from charlie_work.state import load_state, save_state
    from charlie_work.workflow import UNAUTHORIZED_MERGE_BASELINE_KEY

    state = load_state(paths.state_file)
    state[UNAUTHORIZED_MERGE_BASELINE_KEY] = {
        "armed_at": "2026-07-26T00:00:00Z",
        "pre_existing_prs": list(pre_existing),
    }
    save_state(paths.state_file, state)


def _merged_worker_pr(number: int, issue: int, sha: str) -> dict[str, Any]:
    """A merged worker-branch PR shaped like merged_pr_list() output."""
    return {
        "number": number,
        "title": f"fix: work for #{issue}",
        "url": f"https://example.test/pull/{number}",
        "headRefName": f"agent/issue-{issue}-fix",
        "baseRefName": "main",
        "headRefOid": sha,
        "state": "MERGED",
        "isCrossRepository": False,
        "body": f"Closes #{issue}",
        "labels": [],
    }


def _ack_unauthorized_merge(paths, pr_number: int, reason: str = "triaged") -> None:
    """Mark a post-arming unauthorized-merge finding as acknowledged in state.json.

    Mirrors ``_arm_unauthorized_merge_tripwire`` for the ack half of the
    tripwire: tests asserting on post-arming findings use this to declare the
    steady state where a finding has already been triaged and must no longer
    pin ``ok=False`` (issue #673).
    """
    from charlie_work.state import load_state, save_state
    from charlie_work.workflow import UNAUTHORIZED_MERGE_ACK_KEY

    state = load_state(paths.state_file)
    acks = state.get(UNAUTHORIZED_MERGE_ACK_KEY)
    if not isinstance(acks, dict):
        acks = {}
    acks[str(pr_number)] = {
        "acknowledged_at": "2026-07-27T00:00:00Z",
        "reason": reason,
    }
    state[UNAUTHORIZED_MERGE_ACK_KEY] = acks
    save_state(paths.state_file, state)


def _merge_check_app(tmp_path: Path):
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    return OrchestratorApp(tmp_path, paths, config, fake_gh), paths, fake_gh


def _write_decision(tmp_path: Path, pr: int, payload: dict) -> None:
    decision_dir = tmp_path / ".var" / "charlie-work" / "prs" / f"pr-{pr}"
    decision_dir.mkdir(parents=True, exist_ok=True)
    (decision_dir / "review-decision.json").write_text(json.dumps(payload), encoding="utf-8")
