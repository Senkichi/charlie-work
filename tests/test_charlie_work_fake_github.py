"""The FakeGitHub test double itself: default PR head indexing, merge-base determinism, typed results.

Split out of ``tests/test_charlie_work.py`` (issue #1554,
Track-1 wave 8/8).
"""

from __future__ import annotations

import ast
import base64
import os
import subprocess
import sys
import textwrap
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
    mapping so a failure was unactionable. The harness now records the seed
    that produced each result so any future non-determinism is immediately
    attributable.

    Issue #2860: importing ``_fakes_github`` in each seed subprocess drags in
    ``charlie_work.github`` and its whole capability-collaborator chain,
    which grew past ~200ms of import per subprocess (x5 seeds). The harness
    instead extracts the real ``_ancestors``/``_merge_base`` source from
    ``_fakes_github.py`` with ``ast`` (once, in the parent) and ships it to
    each child base64-encoded, executed on a stub class under ``python -S``
    (no venv/site processing -- the child needs only stdlib). The
    determinism question is still asked of the production fake's actual
    implementation; the expected answer is pinned literally below, and parity
    with the real ``FakeGitHub`` is pinned by an in-process check.
    """
    tests_dir = Path(__file__).parent
    source = (tests_dir / "_fakes_github.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    klass = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FakeGitHub"
    )
    wanted = {"_ancestors", "_merge_base"}
    found = {
        node.name: ast.get_source_segment(source, node)
        for node in klass.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    }
    missing = wanted - found.keys()
    assert not missing, (
        f"merge-base harness methods missing from _fakes_github.py: {sorted(missing)}"
    )
    methods = "\n".join(textwrap.dedent(seg) for seg in found.values())
    payload = base64.b64encode(methods.encode("utf-8")).decode("ascii")
    graph = (
        '{"R": {"parents": []},'
        '"A1": {"parents": [{"sha": "R"}]},'
        '"B1": {"parents": [{"sha": "R"}]},'
        '"A2": {"parents": [{"sha": "A1"}, {"sha": "B1"}]},'
        '"B2": {"parents": [{"sha": "B1"}, {"sha": "A1"}]}}'
    )
    code = "\n".join(
        [
            "import ast, base64, sys, textwrap",
            "methods = base64.b64decode(sys.argv[1]).decode('utf-8')",
            "prog = 'class FakeMergeBase:\\n' + textwrap.indent(methods, '    ')",
            "ns = {}",
            "exec(compile(prog, '<test_fake_github_merge_base>', 'exec'), ns)",
            "gh = ns['FakeMergeBase']()",
            "gh.commits = " + graph,
            'print(gh._merge_base("A2", "B2"))',
        ]
    )
    # Record the seed that produced each result so a failure is actionable.
    results: dict[str, str] = {}
    for seed in range(5):
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = str(seed)
        proc = subprocess.run(
            [sys.executable, "-S", "-c", code, payload],
            env=env,
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

    # Parity pins (issue #2860): the extracted-and-exec'd stub must return the
    # literal expected answer -- A1 and B1 tie on BFS distance from both tips,
    # so the sha tie-break must pick A1 -- and must agree with the real
    # FakeGitHub on the same graph.
    stub_result = next(iter(distinct))
    assert stub_result == "A1", f"expected merge base A1, got {stub_result!r}"
    real_gh = FakeGitHub()
    real_gh.commits = {
        "R": {"parents": []},
        "A1": {"parents": [{"sha": "R"}]},
        "B1": {"parents": [{"sha": "R"}]},
        "A2": {"parents": [{"sha": "A1"}, {"sha": "B1"}]},
        "B2": {"parents": [{"sha": "B1"}, {"sha": "A1"}]},
    }
    real_gh.base_head_sha = "A2"
    assert stub_result == real_gh._merge_base("A2", "B2")


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
