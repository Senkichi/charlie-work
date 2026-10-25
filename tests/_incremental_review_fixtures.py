"""Shared fixtures for the issue #2837 incremental-review tests.

The round-history acceptance tests (``test_charlie_work_review_round_history``)
and the incremental-review suites (``test_issue_2837_incremental_review``,
``test_review_interdiff``, ``test_issue_2837_local_lane``) plant the same
round-1/round-2 PR diffs and assert on the same launcher argv. Hoisted here
(the ``tests/_*.py`` convention sanctioned by
``tests/test_zero_cross_test_import_guard.py``) so the suites share one
definition instead of importing from each other.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from _claude_adapter_fixtures import _fake_worktree
from charlie_work import claude_code
from charlie_work.tee_launch import TeeLaunchError

ROUND1_HEAD = "sha-r1"

ROUND1_PR_DIFF = (
    "diff --git a/one.py b/one.py\n"
    "--- a/one.py\n"
    "+++ b/one.py\n"
    "@@ -10,3 +10,3 @@\n"
    " line\n"
    "+added line\n"
    "-context\n"
    "diff --git a/two.py b/two.py\n"
    "--- a/two.py\n"
    "+++ b/two.py\n"
    "@@ -20,2 +20,2 @@\n"
    " alpha\n"
    " beta\n"
)

# The planted rework's PR diff. one.py is byte-identical above everything but
# the hunk header's line numbers (the base merge shifted them -- a pure
# base-merge re-chunking is NOT a content change); two.py's only delta is
# shifted line numbers too; three.py is the one genuinely changed (new) file.
# base_noise.txt is deliberately ABSENT from both PR diffs: a rework that also
# merges base commits must not surface those files in ANY list (the old
# three-dot interdiff included them; the PR-relative one cannot).
ROUND2_PR_DIFF = (
    "diff --git a/one.py b/one.py\n"
    "--- a/one.py\n"
    "+++ b/one.py\n"
    "@@ -40,3 +40,3 @@\n"
    " line\n"
    "+added line\n"
    "-context\n"
    "diff --git a/two.py b/two.py\n"
    "--- a/two.py\n"
    "+++ b/two.py\n"
    "@@ -30,2 +30,2 @@\n"
    " alpha\n"
    " beta\n"
    "diff --git a/three.py b/three.py\n"
    "--- /dev/null\n"
    "+++ b/three.py\n"
    "@@ -0,0 +1,1 @@\n"
    "+hello\n"
)

# What GitHub's three-dot ``compare_diff(prior_head, new_head)`` returns for a
# rework that merged the base branch: the base's own commits ride along. Tests
# plant this as ``fake_gh.compare_diff_overrides[(ROUND1_HEAD, <live head>)]``
# so any "base_noise absent" assertion is non-vacuous -- were the packet built
# from ``compare_diff`` instead of the archived PR diffs, this file would
# appear in the interdiff.
BASE_NOISE_COMPARE_DIFF = (
    "diff --git a/base_noise.txt b/base_noise.txt\n"
    "--- a/base_noise.txt\n"
    "+++ b/base_noise.txt\n"
    "@@ -1,1 +1,1 @@\n"
    "-old base line\n"
    "+new base line\n"
)


def round1_decision(*, head: str = ROUND1_HEAD) -> dict[str, object]:
    return {
        "decision": "request_changes",
        "summary": "round-1 summary text",
        "required_changes": ["round-1 finding: add null check"],
        "reviewed_head_sha": head,
    }


def plant_round1(
    decision_dir: Path,
    *,
    diff: str | None = ROUND1_PR_DIFF,
    head: str = ROUND1_HEAD,
) -> None:
    """Round-1 archive plus the flat mirror, as ``record_review`` leaves them.

    ``diff=None`` plants a legacy round: the decision is archived but there is
    no ``diff.patch`` beside it (recorded before #2837).
    """
    round_dir = decision_dir / "rounds" / "round-1"
    round_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(round1_decision(head=head))
    (round_dir / "review-decision.json").write_text(payload, encoding="utf-8")
    (decision_dir / "review-decision.json").write_text(payload, encoding="utf-8")
    if diff is not None:
        (round_dir / "diff.patch").write_text(diff, encoding="utf-8")


def capture_real_review_launches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> list[dict[str, Any]]:
    """Route review launches through the REAL ``launch_claude_worker``, stopping
    short of spawning a process, and return one capture per launch.

    Each capture holds the kwargs the dispatch passed (``max_turns_override``)
    and ``argv`` -- the command the launcher actually rendered, where the
    ``--max-turns`` pin lives. The launcher records its rendered command on
    the error record it returns when the spawn step fails, so both spawn seams
    (the direct ``popen_worker`` and the stream-json tee) are stubbed to fail:
    no ``claude`` binary and no splitter child is ever started.
    """
    captures: list[dict[str, Any]] = []
    real_launch = claude_code.launch_claude_worker

    def launch(*args: Any, **kwargs: Any) -> Any:
        record = real_launch(*args, **kwargs)
        captures.append({"kwargs": kwargs, "argv": tuple(record.command)})
        return record

    def no_spawn(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("test double: no reviewer process is spawned")

    monkeypatch.setattr("charlie_work.workflow.launch_claude_worker", launch)
    monkeypatch.setattr(
        claude_code,
        "create_review_checkout",
        lambda *_a, **_k: _fake_worktree(tmp_path, "review-checkout"),
    )
    monkeypatch.setattr(claude_code, "popen_worker", no_spawn)
    monkeypatch.setattr(
        claude_code,
        "launch_tee",
        lambda **_k: TeeLaunchError(error="test double: no tee splitter is spawned"),
    )
    return captures


def max_turns_in(argv: tuple[str, ...]) -> str:
    """The value following the single ``--max-turns`` flag in ``argv``."""
    assert argv.count("--max-turns") == 1, argv
    return argv[argv.index("--max-turns") + 1]
