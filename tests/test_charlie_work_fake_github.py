"""The FakeGitHub test double itself: default PR head indexing, merge-base determinism, typed results.

Split out of ``tests/test_charlie_work.py`` (issue #1554,
Track-1 wave 8/8).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from _fakes_github import FakeGitHub
from charlie_work import github as github_module


def test_fake_github_default_pr_head_is_indexed() -> None:
    """Issue #347: the default FakeGitHub fixture must index the PR head in commits.

    The default fixture assigns ``self.prs`` before ``self.commits`` exists, so
    the ``__setattr__`` hook's call to ``_record_pr_heads`` silently no-ops.
    This test ensures the PR head is indexed and that ``compare()`` can derive
    the merge-base from the commit graph.
    """
    gh = FakeGitHub()
    assert "sha-abc123" in gh.commits
    assert gh.commits["sha-abc123"]["parents"] == [{"sha": "base-sha"}]
    assert gh.base_head_sha in gh.commits
    result = gh.compare("main", "sha-abc123")
    assert result is not None
    assert result["merge_base_commit"]["sha"] == gh.base_head_sha


def test_fake_github_merge_base_criss_cross_is_deterministic() -> None:
    """Issue #347: _merge_base must be deterministic across PYTHONHASHSEED.

    In a criss-cross graph, the two best common ancestors are both minimal.
    A correct BFS distance and a deterministic tie-break must produce the same
    merge base regardless of hash randomization.

    Issue #1292: the harness was previously flaky because (a) it imported
    ``FakeGitHub`` via ``test_charlie_work`` -- dragging the entire 50k-line
    test module (pytest, yaml, every helper fixture) into a subprocess whose
    import can fail transiently under full-suite parallel-run contention --
    and (b) it collected results into a ``set``, discarding the seed-to-result
    mapping so a failure was unactionable. The harness now imports
    ``FakeGitHub`` directly from ``_fakes_github`` (the lightweight module that
    defines it) and records the seed that produced each result so any future
    non-determinism is immediately attributable.
    """
    tests_dir = Path(__file__).parent
    repo_root = tests_dir.parent
    code = "\n".join(
        [
            "import os, sys",
            "sys.path.insert(0, sys.argv[1])",
            "from _fakes_github import FakeGitHub",
            "gh = FakeGitHub()",
            "gh.commits = {",
            '    "R": {"parents": []},',
            '    "A1": {"parents": [{"sha": "R"}]},',
            '    "B1": {"parents": [{"sha": "R"}]},',
            '    "A2": {"parents": [{"sha": "A1"}, {"sha": "B1"}]},',
            '    "B2": {"parents": [{"sha": "B1"}, {"sha": "A1"}]},',
            "}",
            'gh.base_head_sha = "A2"',
            'print(gh._merge_base("A2", "B2"))',
        ]
    )
    # Record the seed that produced each result so a failure is actionable.
    results: dict[str, str] = {}
    for seed in range(5):
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = str(seed)
        proc = subprocess.run(
            [sys.executable, "-c", code, str(tests_dir)],
            env=env,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise AssertionError(
                f"subprocess for PYTHONHASHSEED={seed} exited {proc.returncode}.\n"
                f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )
        results[str(seed)] = proc.stdout.strip()
    distinct = set(results.values())
    assert len(distinct) == 1, f"merge_base varied across hash seeds (seed -> result): {results}"


def test_base_fake_github_merged_prs_for_issue_returns_typed_result() -> None:
    """Issue #882 regression guard.

    Production ``GitHubCLI.merged_prs_for_issue`` always returns a
    ``MergedPRSearchResult`` carrying ``.ok``. The base ``FakeGitHub`` used to
    return a plain ``list`` with no ``.ok`` attribute, so it silently disagreed
    with the real thing -- only inert because the sole consumer reads
    defensively via ``getattr(merged_prs, "ok", True)``. A future caller that
    reads ``.ok`` directly would pass tests against this fake and raise
    ``AttributeError`` in production.

    Pin the typed shape on the base fake so the fake and the real thing agree.
    """
    fake_gh = FakeGitHub()
    # The default PR ships OPEN; flip it to MERGED so it matches.
    fake_gh.prs[0]["state"] = "MERGED"

    result = fake_gh.merged_prs_for_issue(123, "agent/issue-")

    assert isinstance(result, github_module.MergedPRSearchResult)
    assert result.ok is True
    assert [pr["number"] for pr in result] == [456]
    # An empty search must still carry the typed shape (ok=True, not a bare []).
    empty = fake_gh.merged_prs_for_issue(999, "agent/issue-")
    assert isinstance(empty, github_module.MergedPRSearchResult)
    assert empty.ok is True
    assert list(empty) == []
