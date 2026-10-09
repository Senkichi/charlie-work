"""Repo-pair builders and the shared ``deployment_gate_repos`` fixture for
``tests/test_backfill_stale_rework_briefs.py`` (issue #2663).

Extracted from the test module for two reasons:

* The test module sits at the 800-line file-size ratchet cap (issue #1442);
  the shared fixture could not land in-file.
* Sharing one (with-fix, without-fix) pair across the five deployment-gate
  consumers moves each test's two ``shutil.copytree`` materializations and
  its ``git fast-import`` spawn out of the ``call`` phase and into module
  setup -- the cost the ledger flagged at 2.1x baseline when the runner
  pool's per-spawn price inflated. Same mechanism the module's own
  ``_warm_plain_git_template`` fixture (#2577) charges the ``plain``
  template build to module setup for.

Every consumer treats the pair as read-only: ``check_deployment_gate`` runs
``merge-base --is-ancestor`` / ``rev-parse`` queries only, and ``main()``'s
dry-run and refused-``--apply`` paths never write into either repo. A test
that needs to mutate a repo must build its own (``_init_repo``) -- do not
mutate these.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import _git_templates
from _worktree_fixtures import _init_repo


def _make_repo_with_fix(tmp_path: Path, name: str) -> tuple[Path, str]:
    """Create a git repo whose HEAD contains a 'fix' commit; return (repo, fix_sha).

    The base repo materializes from the per-process ``plain`` git template
    (``_worktree_fixtures._init_repo`` -> ``tests/_git_templates.py``,
    HS-CW-4: ``shutil.copytree``, no subprocesses); the fix commit is one
    ``git fast-import`` rather than an add/commit/rev-parse trio (issue
    #2605, the same runner-pool spawn slowdown #2592 fixed elsewhere), and
    ``--export-marks`` returns the new SHA without a ``rev-parse``. Callers
    only ever resolve the SHA, so the stream adds ``fix.txt`` to the commit
    without touching the worktree or index. The commit is created after
    the copy, so no other template copy's object store contains it -- what
    the renderer-vs-state-repo tests rely on."""
    repo = tmp_path / name
    _init_repo(repo)
    marks_path = tmp_path / f"{name}.marks"
    # ``from refs/heads/main`` on the branch being committed is a hard error
    # ("Can't create a branch from itself"), so the stream commits on a
    # scratch branch, fast-forwards main to it, and deletes the scratch ref.
    subprocess.run(
        ["git", "fast-import", "--quiet", f"--export-marks={marks_path}"],
        cwd=repo,
        # input as bytes: text=True would translate \n to \r\n on Windows
        # stdin, and git parses the \r as part of each command line.
        input=b"""\
commit refs/heads/fix-tip
mark :1
committer Test User <test@example.test> 0 +0000
data <<MSG
the renderer fix
MSG
from refs/heads/main
M 100644 inline fix.txt
data <<EOT
fix
EOT
reset refs/heads/main
from :1
reset refs/heads/fix-tip
from 0000000000000000000000000000000000000000
""",
        check=True,
        capture_output=True,
    )
    fix_sha = marks_path.read_text(encoding="utf-8").split()[1]  # ":1 <sha>"
    return repo, fix_sha


def _make_repo_without_fix(tmp_path: Path, name: str) -> Path:
    """Create a git repo whose HEAD does NOT contain the fix (simulates a
    state root / different repo that cannot resolve the fix SHA).

    A ``plain`` template copy is sufficient here: the gate only requires a
    real git work tree whose object store lacks the fix SHA, and the fix
    commit is only ever created inside ``_make_repo_with_fix``'s copy."""
    repo = tmp_path / name
    _init_repo(repo)
    return repo


@pytest.fixture(scope="module", name="deployment_gate_repos")
def _deployment_gate_repos(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, str, Path]:
    """``(repo_with_fix, fix_sha, repo_without_fix)`` built once per worker.

    The ``GIT_CONFIG_ENV`` trio is re-applied so ``_init_repo`` copies the
    ``plain`` template under the same ``env_fingerprint()`` the call phase
    looks up -- ``conftest._isolate_git_env`` is function-scoped, so the
    trio is absent at module setup and a bare ``_init_repo`` would build a
    second template under a fingerprint the call phase never sees (the
    issue-#2565 lesson ``_warm_plain_git_template`` documents).
    """
    base = tmp_path_factory.mktemp("deployment-gate-repos")
    with pytest.MonkeyPatch.context() as mp:
        for key, value in _git_templates.GIT_CONFIG_ENV.items():
            mp.setenv(key, value)
        with_fix, fix_sha = _make_repo_with_fix(base, "with-fix")
        without_fix = _make_repo_without_fix(base, "without-fix")
    return with_fix, fix_sha, without_fix
