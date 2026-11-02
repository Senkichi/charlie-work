"""Issue #2837 on the local (no-remote) review lane.

The local lane builds its own packets (``_local_build_packet``), records its
own verdicts (``record_local_review``) and claims its own reviewers
(``_local_dispatch_reviewers``), so each of the three #2837 seams needs a local
pin of its own: the round's ``diff.patch`` is archived, the packet carries the
round facts (and a never-reviewed packet carries NO prior-review section and
NO fallback measurement), and the dispatch claim pins the incremental turn
budget from the stamped round kind.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from _incremental_review_fixtures import capture_real_review_launches, max_turns_in
from _local_lane_fixtures import (
    _app,
    _commit_file,
    _git,
    _init_repo,
    _local_config,
    _make_branch,
    _parked_issue,
    new_repo_root,
)
from charlie_work.instrumentation import query_events
from charlie_work.state import load_state_locked
from charlie_work.workflow import OrchestratorApp

_ISSUE = 7
_BRANCH = "agent/issue-7-x"


def _local_app(monkeypatch: pytest.MonkeyPatch) -> tuple[OrchestratorApp, str, list[Any]]:
    """A local-lane app with issue 7 parked and its round-1 packet built.

    Returns ``(app, branch head, prior_review_section calls)``; the spy is
    installed BEFORE the first packet so it sees every builder entry.
    """
    calls: list[Any] = []
    original = OrchestratorApp._build_prior_review_section

    def spy(self: OrchestratorApp, *args: Any, **kwargs: Any) -> str:
        calls.append(args)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(OrchestratorApp, "_build_prior_review_section", spy)

    repo = new_repo_root()
    _init_repo(repo)
    issues_dir = repo / "docs" / "issues"
    config = _local_config(
        repo,
        issues_dir,
        fleet={"global_max_concurrent_reviews": 0, "launch_lock_wait_seconds": 0},
        review_dispatch={"max_local_review_processes": 0},
    )
    app = _app(repo, issues_dir, config)
    head = _make_branch(repo, _BRANCH, "a.py", "a = 1\n")
    _parked_issue(app, issues_dir, _ISSUE, _BRANCH)
    packets = app._local_review_packets()
    assert [p["ok"] for p in packets["packets"]] == [True], packets
    return app, head, calls


def _pr_dir(app: OrchestratorApp) -> Path:
    return app.paths.prs / f"pr-{_ISSUE}"


def _record(app: OrchestratorApp) -> dict[str, Any]:
    return load_state_locked(app.paths.state_file)["prs"][str(_ISSUE)]


def _local_packet_events(app: OrchestratorApp) -> list[dict[str, Any]]:
    return [
        event["payload"]
        for event in query_events(app.paths.state_file, kind="review_packet")
        if event["payload"].get("local")
    ]


def test_local_first_round_packet_has_no_prior_review_and_no_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A never-reviewed local packet is round "first": it must not enter the
    prior-review builder (which used to render a bogus "Round 1 (decision:
    missing)" section and emit a false ``review_interdiff_fallback``), and the
    packet event + PR record carry the round facts."""
    app, _head, builder_calls = _local_app(monkeypatch)

    assert builder_calls == []
    assert query_events(app.paths.state_file, kind="review_interdiff_fallback") == []
    packet = (_pr_dir(app) / "review-prompt.md").read_text(encoding="utf-8")
    assert "## Prior review" not in packet
    assert "Incremental review procedure" not in packet
    assert not (_pr_dir(app) / "interdiff.patch").exists()
    record = _record(app)
    assert record["review_round_kind"] == "first"
    assert (record["review_interdiff_files"], record["review_interdiff_lines"]) == (0, 0)
    assert _local_packet_events(app)[-1]["review_round_kind"] == "first"
    assert _local_packet_events(app)[-1]["review_interdiff_files"] == 0


def test_local_first_round_dispatch_pins_the_flat_turn_budget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app, _head, _calls = _local_app(monkeypatch)
    launches = capture_real_review_launches(monkeypatch, tmp_path)

    result = app._local_dispatch_reviewers()

    assert len(launches) == 1, result
    assert max_turns_in(launches[0]["argv"]) == "40"
    claims = query_events(app.paths.state_file, kind="review_dispatch_claim")
    assert claims[-1]["payload"]["review_round_kinds"] == {str(_ISSUE): "first"}


def test_local_rework_round_is_incremental_from_the_archived_round_diff(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Round 1 verdict archives the packet's diff.patch; the rework's new
    commit then builds a PR-relative interdiff from it (only the new file),
    stamps the round "incremental", and the dispatch launches the reviewer
    with the 20-turn incremental budget."""
    app, head1, builder_calls = _local_app(monkeypatch)
    pr_dir = _pr_dir(app)
    round1_diff = (pr_dir / "diff.patch").read_text(encoding="utf-8")
    assert "a.py" in round1_diff

    verdict = app.record_local_review(
        _ISSUE,
        "request_changes",
        summary="Add b.py.",
        reviewed_head=head1,
        verdict_provenance="fresh_llm_review",
    )
    assert verdict.ok, verdict.message

    # The verdict archived the diff the reviewer saw, beside the decision.
    archived = pr_dir / "rounds" / "round-1" / "diff.patch"
    assert archived.read_text(encoding="utf-8") == round1_diff

    # The rework lands a new commit on the branch -> the packet is rebuilt.
    _git(app.repo_root, "checkout", _BRANCH)
    _commit_file(app.repo_root, "b.py", "b = 2\n", "feat: b")
    _git(app.repo_root, "checkout", "main")
    packets = app._local_review_packets()
    assert [p["ok"] for p in packets["packets"]] == [True], packets

    assert len(builder_calls) == 1  # round 2 enters the builder; round 1 did not
    interdiff = (pr_dir / "interdiff.patch").read_text(encoding="utf-8")
    assert "b/b.py" in interdiff
    assert "a.py" not in interdiff  # a.py's patch is unchanged since round 1
    packet = (pr_dir / "review-prompt.md").read_text(encoding="utf-8")
    assert "## Incremental review procedure" in packet
    assert "Files unchanged since round 1" in packet
    assert "`a.py`" in packet
    assert query_events(app.paths.state_file, kind="review_interdiff_fallback") == []
    record = _record(app)
    assert record["review_round_kind"] == "incremental"
    assert record["review_interdiff_files"] == 1
    packet_event = _local_packet_events(app)[-1]
    assert packet_event["review_round_kind"] == "incremental"
    assert packet_event["review_interdiff_files"] == 1
    assert packet_event["review_interdiff_lines"] == record["review_interdiff_lines"] > 0

    launches = capture_real_review_launches(monkeypatch, tmp_path)
    app._local_dispatch_reviewers()

    assert len(launches) == 1
    assert launches[0]["kwargs"]["max_turns_override"] == 20
    assert max_turns_in(launches[0]["argv"]) == "20"
    claims = query_events(app.paths.state_file, kind="review_dispatch_claim")
    assert claims[-1]["payload"]["review_round_kinds"] == {str(_ISSUE): "incremental"}
