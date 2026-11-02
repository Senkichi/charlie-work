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

import json
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .config import OrchestratorConfig
from .github import GitHubError, GitHubLike
from .instrumentation import query_events
from .local_issue_commits import dirty_tracker_files
from .local_issue_files import scan_issues
from .local_lane import disabled_lane_switches, kill_switch_stall_payloads
from .paths import RuntimePaths
from .subprocess_runner import run_captured

# How far back the deferred-flush events doctor counts belong (issue #2434):
# the pass-end flush runs every pass, so two -- or more -- recorded deferrals
# inside this window mean the consumer's issues dir has stayed dirty across
# more than one pass, which is exactly the accumulation the tracker-write
# commits exist to prevent.
_DEFERRAL_LOOKBACK_HOURS = 24.0

# Doctor warns once the deferral has been observed on this many passes within
# the lookback window -- the issue text's "non-empty for more than one pass".
_DEFERRAL_WARN_THRESHOLD = 2


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

    _check_local_lane_kill_switch(add, gh, paths, config)
    _check_local_tracker_writes(add, gh, repo_root, paths, config)


def _check_local_tracker_writes(
    add: Any,
    gh: GitHubLike,
    repo_root: Path,
    paths: RuntimePaths,
    config: OrchestratorConfig,
) -> None:
    """Warn when tracker writes accumulate uncommitted across passes (#2434).

    The backend's tracker writes are committed once per loop pass
    (``orchestration.local_tracker_flush``), so the tracked issue files under
    ``issues_dir`` are normally clean at any doctor run. Dirt at check time is
    one of: the flush being repeatedly skipped (HEAD detached, a merge/rebase
    in progress, a missing committer identity) -- each skip leaves a
    ``local_tracker_writes_deferred`` event -- or the ``commit_writes`` kill
    switch, which is configuration and only worth a note.

    The warning is deliberately threshold-gated: ``>= 2`` recorded deferrals
    within the lookback window means the directory stayed dirty across more
    than one pass. A single deferral (typically yesterday's mid-merge, since
    committed by the next pass's flush) gets an informational note instead,
    because the accumulator's real failure mode is repetition, not a one-off.
    """
    if not config.local_issues.enabled:
        return
    issues_dir = getattr(gh, "issues_dir", None)
    if not isinstance(issues_dir, Path):
        return
    cutoff = (
        (datetime.now(UTC) - timedelta(hours=_DEFERRAL_LOOKBACK_HOURS))
        .isoformat()
        .replace("+00:00", "Z")
    )
    dirty = dirty_tracker_files(repo_root, issues_dir)
    if not dirty:
        add(
            "local tracker writes",
            True,
            "issues dir clean — tracker writes are committed at each pass flush",
        )
        return
    if not config.local_issues.commit_writes:
        add(
            "local tracker writes",
            True,
            f"{len(dirty)} issue file(s) uncommitted — local_issues.commit_writes "
            "is false (kill switch); the flush is disabled by config",
            severity="warning",
        )
        return
    deferred = query_events(paths.state_file, kind="local_tracker_writes_deferred", since=cutoff)
    reason_values: list[str] = []
    for event in deferred:
        payload = event.get("payload")
        reason = payload.get("reason") if isinstance(payload, dict) else None
        if isinstance(reason, str) and reason:
            reason_values.append(reason)
    reasons = "; ".join(dict.fromkeys(reason_values))
    if len(deferred) >= _DEFERRAL_WARN_THRESHOLD:
        add(
            "local tracker writes",
            False,
            f"{len(dirty)} issue file(s) left dirty across more than one pass — "
            f"the pass-end commit flush keeps failing/skipping ({reasons or 'no reason recorded'}): "
            "resolve the stated cause or run `charlie doctor` after the next pass; "
            "the tracker's source of truth is slipping out of git history",
            severity="warning",
        )
        return
    add(
        "local tracker writes",
        True,
        f"{len(dirty)} issue file(s) uncommitted at check time — the end-of-pass "
        "flush commits these once passes run (or the next pass sweeps them); "
        f"{len(deferred)} recent pass-flush deferral(s) recorded",
        severity="warning",
    )


def _check_local_lane_kill_switch(
    add: Any,
    gh: GitHubLike,
    paths: RuntimePaths,
    config: OrchestratorConfig,
) -> None:
    """Warn when a local-lane kill switch strands review-ready issues (#1968).

    An explicit ``review_dispatch.enabled: false`` or
    ``auto_merge.enabled: false`` on a ``local_issues`` repo is a deliberate
    gate and stays honored -- but it dead-ends every finished ticket at
    ``agent:review-ready`` with no signal. This is the same condition the
    loop pass's ``local_lane_kill_switch_stalled`` event reports, evaluated
    live (via ``kill_switch_stall_payloads``) so doctor shows the current
    truth rather than the last emitted row. Warning severity: the config is
    honored as written; the finding names the key to flip.
    """
    if not config.local_issues.enabled:
        return
    stall_hours = config.local_lane.kill_switch_stall_hours
    if stall_hours <= 0:
        add(
            "local lane kill switch",
            True,
            "stall alarm muted — local_lane.kill_switch_stall_hours is 0",
            severity="warning",
        )
        return
    try:
        parked = gh.issue_list(labels=[config.labels.review_ready], state="open")
    except GitHubError as exc:
        add(
            "local lane kill switch",
            False,
            f"could not evaluate — issue scan failed: {exc} "
            "(see 'local issues dir'/'local issue files')",
            severity="warning",
        )
        return
    issues_state: Any = {}
    if paths.state_file.exists():
        try:
            raw = json.loads(paths.state_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raw = None
        if isinstance(raw, dict) and isinstance(raw.get("issues"), dict):
            issues_state = raw["issues"]
    disabled = disabled_lane_switches(
        review_dispatch_enabled=config.review_dispatch.enabled,
        auto_merge_enabled=config.auto_merge.enabled,
    )
    if not disabled:
        add(
            "local lane kill switch",
            True,
            "review_dispatch.enabled and auto_merge.enabled are on — the "
            "local review/merge lane is armed",
        )
        return
    payloads = kill_switch_stall_payloads(
        review_dispatch_enabled=config.review_dispatch.enabled,
        auto_merge_enabled=config.auto_merge.enabled,
        parked=parked,
        issues_state=issues_state,
        stall_hours=stall_hours,
    )
    if not payloads:
        add(
            "local lane kill switch",
            True,
            f"{', '.join(disabled)} is off but no review-ready issue has "
            f"waited past {stall_hours}h — the gate is honored and nothing "
            "is stranded yet",
            severity="warning",
        )
        return
    detail = "; ".join(
        f"{p['switch']} stranding {p['issue_numbers']} (oldest {p['oldest_age_hours']}h)"
        for p in payloads
    )
    add(
        "local lane kill switch",
        False,
        f"{detail} — review-ready issue(s) are dead-ending at "
        f"{config.labels.review_ready}; set the named config key(s) to true "
        "(or remove the explicit false) in orchestrator.config.yaml, or "
        "expect to review/merge the parked worker branches by hand",
        severity="warning",
    )
