"""Doctor checks for a backend that does not publish pull requests
(issue #1706).

``run_doctor`` calls ``_check_local_issue_backend`` in place of the gh-shaped
checks when ``publishes_pull_requests(gh)`` is False.

Lives outside ``doctor`` on purpose: ``doctor.py`` is over the 800-line
module cap and pinned by the file-size high-water-mark ratchet
(``tests/test_file_size_ratchet.py``), which never allows an over-cap file to
grow past its recorded mark -- new code lands in a domain module instead.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .config import OrchestratorConfig
from .github import GitHubLike
from .local_issue_files import scan_issues
from .paths import RuntimePaths
from .subprocess_runner import run_captured


def _check_local_issue_backend(
    add: Any,
    gh: GitHubLike,
    repo_root: Path,
    paths: RuntimePaths,
    config: OrchestratorConfig,
) -> None:
    """Health checks specific to a backend that does not publish pull requests
    (issue #1706).

    Runs only when ``publishes_pull_requests(gh)`` is False, in place of the
    gh-shaped checks that cannot apply. Surfaces the failure modes that ARE
    load-bearing on a file-backed issue source:

    * ``issues_dir`` must exist -- a missing one makes every loop pass defer
      on ``GitHubError`` (``LocalFileGitHub._handle_scan_problems`` raises on
      the missing-directory sentinel), so it is a blocking finding here, the
      same as a ``gh`` outage would be.
    * Per-file scan problems (bad frontmatter, duplicate numbers) drop the
      file from the dispatch set entirely; the backend warns-and-continues,
      so doctor mirrors that severity: a warning, not a block.
    * The orchestrator state dir must be gitignored inside the consumer repo,
      or every pass leaves ``git status`` noise and a worker ``git add -A``
      can commit orchestrator bookkeeping into the consumer's history.
      ``git check-ignore`` is the authoritative answer -- it honours nested
      ``.gitignore`` files, negations, and ``.git/info/exclude``, which a
      hand-rolled ``.gitignore`` text search would not. The probe targets
      ``paths.state_file`` (a file INSIDE the state dir), not the bare dir
      path: check-ignore cannot classify a path that does not exist yet as a
      directory, so a dir-only pattern like ``.var/charlie-work/`` matches the
      state file but not the state dir itself -- probing the dir would
      false-positive a healthy fresh repo whose state dir has not been
      created.

    ``local_merge_queue`` does not exist as a config section yet; the getattr
    guard lets this light up with the section rather than needing a follow-up
    doctor patch. When it lands and is enabled, the merge side runs in the
    main checkout -- a detached HEAD gives it no branch to land worker
    branches on, and each verify command's binary must resolve on PATH before
    the queue tries to run it.
    """
    issues_dir = getattr(gh, "issues_dir", None)
    if isinstance(issues_dir, Path):
        scan = scan_issues(issues_dir)
        add(
            "local issues dir",
            issues_dir.is_dir(),
            f"{issues_dir} ({len(scan.issues)} issue file(s))"
            if issues_dir.is_dir()
            else f"local_issues.issues_dir does not exist: {issues_dir} — "
            "every loop pass defers on this",
        )
        # The missing-dir sentinel problem (path == issues_dir itself) is
        # already reported by the check above; the rest are per-file problems.
        problems = [p for p in scan.problems if p.path != issues_dir]
        if problems:
            detail = "; ".join(f"{p.path.name}: {p.reason}" for p in problems)
            add(
                "local issue files",
                False,
                f"{len(problems)} file(s) skipped by the scan — invisible to "
                f"dispatch until fixed: {detail}",
                severity="warning",
            )
        else:
            add("local issue files", True, "no scan problems")
    else:
        add(
            "local issues dir",
            True,
            "skipped — this no-PR backend exposes no issues_dir",
            severity="warning",
        )

    repo_resolved = repo_root.resolve()
    try:
        state_rel = paths.state_file.relative_to(repo_resolved)
    except ValueError:
        add(
            "state dir gitignored",
            True,
            f"state dir {paths.root} is outside the repo root — nothing to ignore",
        )
    else:
        result = run_captured(
            ["git", "check-ignore", "-q", state_rel.as_posix()],
            cwd=repo_root,
            timeout_seconds=30,
        )
        if result.returncode == 0:
            add("state dir gitignored", True, f"{state_rel.as_posix()} is ignored")
        elif result.returncode == 1:
            add(
                "state dir gitignored",
                False,
                f"{state_rel.as_posix()} is NOT ignored — add the state dir to the "
                "consumer repo's .gitignore so orchestrator state stays out of "
                "`git status` and out of any worker `git add -A`",
            )
        else:
            add(
                "state dir gitignored",
                False,
                "could not determine — git check-ignore failed: "
                f"{result.error or result.stderr.strip() or 'unknown'}",
                severity="warning",
            )

    merge_queue = getattr(config, "local_merge_queue", None)
    if merge_queue is not None and getattr(merge_queue, "enabled", False):
        head = run_captured(
            ["git", "symbolic-ref", "-q", "--short", "HEAD"],
            cwd=repo_root,
            timeout_seconds=30,
        )
        branch = head.stdout.strip() if head.ok else ""
        add(
            "local merge queue branch",
            bool(branch),
            f"main checkout is on branch {branch!r}"
            if branch
            else "main checkout HEAD is detached or unreadable — the merge "
            "queue has no branch to land worker branches on",
        )
        binaries: set[str] = set()
        for cmd in getattr(merge_queue, "verify_commands", ()) or ():
            if isinstance(cmd, str):
                binary = cmd.split()[0] if cmd.split() else ""
            elif cmd:
                binary = str(cmd[0])
            else:
                continue
            if binary:
                binaries.add(binary)
        missing = sorted(b for b in binaries if shutil.which(b) is None)
        add(
            "local merge queue verify commands",
            not missing,
            f"{len(binaries)} verify command binaries, all resolve on PATH"
            if not missing
            else f"verify command binaries not on PATH: {missing}",
        )
