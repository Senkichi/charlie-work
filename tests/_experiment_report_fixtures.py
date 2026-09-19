"""Shared fixtures for the ``charlie experiment-report`` tests (issue #1701).

Hoisted out of ``test_experiment_report.py`` under the repo's 800-line
file-size cap (issue #1442), the same pattern ``_cli_fixtures.py``
established: the event builders every report-level test uses, plus the
fake-repo helpers the CLI-driving tests share.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from _cli_fixtures import _FakeGitHub, _make_repo
from charlie_work import cli, instrumentation

KEY = "review_effort_arm"


def _evt(ts: str, kind: str, payload: dict) -> dict:
    return {
        "ts": ts,
        "kind": kind,
        "payload": payload,
        "pr_number": payload.get("pr_number"),
        "issue_number": payload.get("issue_number"),
        "repo": None,
        "correlation_id": None,
        "level": "info",
    }


def _round(pr: int, decision: str, arm: str, *, ts: str, cost=None, issue: int = 1) -> dict:
    sm: dict = {KEY: arm}
    if cost is not None:
        sm["cost_usd"] = cost
    return _evt(
        ts,
        "record_review",
        {
            "pr_number": pr,
            "issue_number": issue,
            "decision": decision,
            "session_metrics": sm,
        },
    )


def _state_path(repo: Path) -> Path:
    return repo / ".var" / "charlie-work" / "state.json"


def _repo_with_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """A fake repo with one recorded event (and thus a live events.db)."""
    monkeypatch.setattr(cli, "GitHub", _FakeGitHub)
    repo = _make_repo(tmp_path)
    state_path = _state_path(repo)
    instrumentation.log_event(
        state_path,
        "record_review",
        {
            "pr_number": 1,
            "issue_number": 1,
            "decision": "approved",
            "session_metrics": {KEY: "deep", "cost_usd": 1.25},
        },
    )
    return repo, state_path


def _state_dir_snapshot(state_path: Path) -> dict[str, bytes]:
    """Every file in the state dir mapped to its bytes -- catches a WAL
    sidecar, a .migrated rename, or any stray file, not just the two
    byte-compared files."""
    return {p.name: p.read_bytes() for p in sorted(state_path.parent.iterdir()) if p.is_file()}
