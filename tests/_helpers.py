"""Miscellaneous shared test helpers and constants.

Hoisted out of ``test_charlie_work.py`` (issue #1284): a real-git-repo
fixture builder, a cross-family review app builder, a second
mergequeue-candidate PR seeder, and a handful of read-only module
constants (the examples directory, a stale-CI check fixture triad, and a
minimal valid cross-family report body), all imported by other test
modules that need the same shapes.
"""

from __future__ import annotations

from pathlib import Path

from _fakes_github import FakeGitHub
from _worktree_fixtures import _init_repo
from charlie_work.config import OrchestratorConfig
from charlie_work.paths import runtime_paths
from charlie_work.workflow import OrchestratorApp

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"

_STALE_CI_REQUIRED = ("Tests passed", "Pre-commit")

# NOTE: the three lists below are shared, mutable fixture objects. Do not
# mutate them in place (e.g. ``.append``/item assignment) -- they are
# imported and reused across many tests. Nothing mutates them today; keep
# it that way and copy before mutating if a future test needs a variant.
_STALE_CI_GREEN_CHECKS = [
    {"name": "Tests passed", "state": "SUCCESS"},
    {"name": "Pre-commit", "state": "SUCCESS"},
]

_STALE_CI_RED_CHECKS = [
    {"name": "Tests passed", "state": "FAILURE"},
    {"name": "Pre-commit", "state": "SUCCESS"},
]

_STALE_CI_CONTAMINATED_REQUIRED_CHANGES = [
    "Tests passed: .github:18 — Process completed with exit code 1."
]


def _init_git_repo(repo_root: Path) -> None:
    """Create a real non-bare git repo with one commit on ``main``.

    Delegates to ``_worktree_fixtures._init_repo`` so the repo materializes
    from the per-process ``plain`` git template (HS-CW-4,
    ``tests/_git_templates.py``) instead of re-running the five-process init
    sequence per call — the spawn cost is what the test ledger flagged at
    ~2.9x baseline on Windows CI (issue #2387). A test that builds two
    repos that must not share a commit SHA needs a ``fresh=True`` builder,
    which this helper deliberately does not expose.
    """
    _init_repo(repo_root)


def _cross_family_app(tmp_path: Path) -> OrchestratorApp:
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    return OrchestratorApp(tmp_path, paths, config, FakeGitHub())


VALID_CROSS_FAMILY_REPORT = "**MAJOR**\nissue\n\nVerdict: safe"


def _second_mergequeue_pr(fake_gh) -> None:
    """Add a second approved-candidate issue/PR pair (124/789) to a FakeGitHub
    fixture, reviewed after the default 123/456 pair."""
    fake_gh.issues.append(
        {
            "number": 124,
            "title": "Fix parsing",
            "url": "https://example.test/issues/124",
            "body": "Parsing is broken",
            "labels": [{"name": "automated-ready"}],
            "state": "OPEN",
        }
    )
    fake_gh.prs.append(
        {
            "number": 789,
            "title": "Fix #124: parsing",
            "url": "https://example.test/pull/789",
            "headRefName": "agent/issue-124-fix-parsing",
            "baseRefName": "main",
            "headRefOid": "sha-def789",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #124\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        }
    )
