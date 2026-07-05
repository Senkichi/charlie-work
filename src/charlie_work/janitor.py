"""Deterministic, non-LLM pre-review gate ("janitor" pattern).

Every LLM review costs real money. The janitor runs BEFORE review-packet
generation and short-circuits obviously-not-ready PRs (draft, closed,
conflicting, failing required checks, missing linked issue, empty body, no
tests/rationale mention) so no review tokens are spent on them. Research
consensus (see docs/design/extraction-dossier.md, "Deterministic, non-LLM
verification before spending review budget") is to verify cheap, concrete
signals before ever routing to the adversarial LLM reviewer.

This module is pure: no I/O, no `gh` calls. The caller (``workflow.review``)
already fetches ``pr`` (``gh pr view`` JSON) and ``checks`` (``gh pr checks``
JSON) for packet generation, so it feeds that same data in here first.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from charlie_work.checks import summarize_checks
from charlie_work.github import linked_issue_number

if TYPE_CHECKING:
    from charlie_work.config import OrchestratorConfig


# Case-insensitive word-boundary regex for tests/rationale markers.
# Matches whole words only to avoid false positives like "test" in "latest".
_TESTS_OR_RATIONALE_RE = re.compile(
    r"\b(?:tests?|tested?|testing|verified?|verification|rationale|no tests because)\b",
    flags=re.IGNORECASE,
)

_CONVENTIONAL_COMMIT_RE = re.compile(r"^(feat|fix|refactor|docs|test|chore|perf|ci)(\(|:|!)")

# Oversized-diff warning threshold: additions + deletions above this line
# count flags the PR as a warning (not a block) for reviewer awareness.
_OVERSIZED_DIFF_THRESHOLD = 1500

# PR dict keys read by janitor gate functions.
# This is the single source of truth for what fields the janitor needs from PR data.
# All keys here must be present in github.PR_VIEW_FIELDS or the corresponding gate will be silently disabled.
JANITOR_PR_KEYS = frozenset(
    {
        "isDraft",  # _check_draft
        "state",  # _check_state
        "mergeable",  # _check_mergeable
        "isCrossRepository",  # _check_linked_issue, _check_base_movement
        "headRefName",  # _check_linked_issue, _check_base_movement, _check_no_op_rework
        "baseRefName",  # _check_no_op_rework
        "body",  # _check_body
        "title",  # _check_title_conventional
        "additions",  # _check_diff_size
        "deletions",  # _check_diff_size
        "mergeStateStatus",  # _check_base_movement
    }
)


@dataclass(frozen=True)
class JanitorVerdict:
    ok: bool
    failures: tuple[str, ...]
    warnings: tuple[str, ...]


def run_janitor(
    pr: dict[str, Any],
    checks: list[dict[str, Any]],
    config: OrchestratorConfig,
    *,
    pr_state: dict[str, Any] | None = None,
    repo_root: Path | None = None,
) -> JanitorVerdict:
    """Run deterministic pre-review checks over ``pr``/``checks`` data.

    Missing keys in ``pr`` never raise: `gh` omits fields depending on the
    flags used to fetch it, so an absent key is treated as "unknown" and the
    check that depends on it is skipped rather than failed.
    """
    failures: list[str] = []
    warnings: list[str] = []

    _check_draft(pr, failures)
    _check_state(pr, failures)
    _check_mergeable(pr, failures)
    _check_required_checks(checks, config, failures, warnings)
    _check_linked_issue(pr, config, failures)
    _check_body(pr, config, failures)
    _check_title_conventional(pr, warnings)
    _check_diff_size(pr, warnings)
    _check_base_movement(pr, config, warnings)

    # Only run no-op rework check if repo_root is provided (needed for worktree enrichment)
    if repo_root is not None:
        _check_no_op_rework(pr, pr_state, failures, warnings, repo_root)

    return JanitorVerdict(ok=not failures, failures=tuple(failures), warnings=tuple(warnings))


def _check_draft(pr: dict[str, Any], failures: list[str]) -> None:
    if "isDraft" not in pr:
        return
    if bool(pr.get("isDraft")):
        failures.append("PR is a draft")


def _check_state(pr: dict[str, Any], failures: list[str]) -> None:
    if "state" not in pr:
        return
    state = str(pr.get("state") or "").upper()
    if state and state != "OPEN":
        failures.append(f"PR state is {state}, not OPEN")


def _check_mergeable(pr: dict[str, Any], failures: list[str]) -> None:
    if "mergeable" not in pr:
        return
    if str(pr.get("mergeable") or "").upper() == "CONFLICTING":
        failures.append("PR has merge conflicts (mergeable=CONFLICTING)")


def _check_required_checks(
    checks: list[dict[str, Any]],
    config: OrchestratorConfig,
    failures: list[str],
    warnings: list[str],
) -> None:
    required = config.auto_merge.required_checks
    if not required:
        return
    summary = summarize_checks(checks, required)
    if summary.failed:
        failures.append(f"Required check(s) failed: {', '.join(summary.failed)}")
    if summary.missing:
        failures.append(f"Required check(s) missing: {', '.join(summary.missing)}")
    if summary.pending:
        warnings.append(f"Required check(s) still pending: {', '.join(summary.pending)}")


def _check_linked_issue(
    pr: dict[str, Any], config: OrchestratorConfig, failures: list[str]
) -> None:
    if not config.review.require_issue_link:
        return
    if (
        linked_issue_number(
            pr,
            is_cross_repository=pr.get("isCrossRepository"),
            branch_prefix=config.dispatch.branch_prefix,
        )
        is None
    ):
        failures.append("No linked issue found (branch name, title, or body)")


def _check_body(pr: dict[str, Any], config: OrchestratorConfig, failures: list[str]) -> None:
    if "body" not in pr:
        return
    body = str(pr.get("body") or "").strip()
    if not body:
        failures.append("PR body is empty")
        return
    if config.review.require_tests_or_rationale:
        if not _TESTS_OR_RATIONALE_RE.search(body):
            failures.append("PR body has no tests/verification/rationale mention")


def _check_title_conventional(pr: dict[str, Any], warnings: list[str]) -> None:
    if "title" not in pr:
        return
    title = str(pr.get("title") or "")
    if title and not _CONVENTIONAL_COMMIT_RE.match(title):
        warnings.append("PR title is not conventional-commit shaped")


def _check_diff_size(pr: dict[str, Any], warnings: list[str]) -> None:
    if "additions" not in pr or "deletions" not in pr:
        return
    additions = pr.get("additions")
    deletions = pr.get("deletions")
    if not isinstance(additions, int) or not isinstance(deletions, int):
        return
    total = additions + deletions
    if total > _OVERSIZED_DIFF_THRESHOLD:
        warnings.append(f"Oversized diff: {total} lines changed (additions+deletions)")


def _check_base_movement(
    pr: dict[str, Any], config: OrchestratorConfig, warnings: list[str]
) -> None:
    """Check if the PR's base has moved since the branch was created.

    Only applies to same-repo PRs with the configured branch prefix (agent PRs).
    Fork PRs and non-prefix branches are excluded to avoid false positives on
    external contributions or unrelated branches.
    """
    # Skip fork PRs entirely
    if pr.get("isCrossRepository"):
        return

    # Only check PRs with the configured branch prefix
    head = str(pr.get("headRefName") or "")
    if not head.startswith(config.dispatch.branch_prefix):
        return

    # Check mergeStateStatus for BEHIND indication
    merge_status = pr.get("mergeStateStatus")
    if merge_status == "BEHIND":
        warnings.append("Base branch has moved since branch (mergeStateStatus=BEHIND)")


def _check_no_op_rework(
    pr: dict[str, Any],
    pr_state: dict[str, Any] | None,
    failures: list[str],
    warnings: list[str],
    repo_root: Path,
) -> None:
    """Check if the PR head is unchanged since a request_changes verdict.

    When a PR has a request_changes verdict in its state, compare the current
    headRefOid against the reviewed_head_sha from that verdict. If they match,
    the rework produced no pushed commits (no-op rework).

    GitHub update-branch merges (base-update commits) are excluded: only non-merge
    commits since the verdict are considered real work.
    """
    if not pr_state:
        return

    # Check if the most recent verdict was request_changes
    decision = pr_state.get("decision")
    if decision != "request_changes":
        return

    reviewed_head_sha = pr_state.get("reviewed_head_sha")
    if not reviewed_head_sha:
        return

    current_head_sha = pr.get("headRefOid")
    if not current_head_sha:
        return

    # If heads match exactly, it's a no-op rework
    if current_head_sha == reviewed_head_sha:
        failure_msg = (
            f"PR head unchanged since request_changes verdict ({reviewed_head_sha}) — "
            f"the rework produced no pushed commits"
        )

        # Enrich with unpushed-commit count if worktree exists
        head_ref = pr.get("headRefName")
        if head_ref:
            unpushed_info = _get_unpushed_commit_info(head_ref, repo_root)
            if unpushed_info:
                failure_msg += f"; {unpushed_info}"
            else:
                failure_msg += "; check the branch worktree for unpushed work before re-reviewing"
        else:
            failure_msg += "; check the branch worktree for unpushed work before re-reviewing"

        failures.append(failure_msg)
        return

    # Criterion 2: detect merge-only advances (e.g., from ship-it's update_open_prs)
    # Fetch the PR head ref and check if any non-merge commits exist since the verdict
    head_ref = pr.get("headRefName")
    base_ref = pr.get("baseRefName")
    if not head_ref:
        # Can't check merge-only case without ref name; fall back to SHA equality check
        # (which already passed above, so no failure here)
        return

    try:
        # Fetch both the PR head ref and base ref from origin
        # We need the base ref to exclude base-reachable commits from the count
        fetch_refs = [head_ref]
        if base_ref:
            fetch_refs.append(base_ref)
        subprocess.run(
            ["git", "fetch", "origin"] + fetch_refs,
            cwd=repo_root,
            capture_output=True,
            check=True,
            text=True,
        )
        # Count non-merge commits since the reviewed head, excluding base-reachable commits
        # The ^ syntax excludes commits reachable from the given refs
        # This counts commits that are:
        # - Not merge commits (--no-merges)
        # - Not in reviewed_head_sha (^reviewed_head_sha)
        # - Not in origin/baseRefName (^origin/baseRefName if base_ref exists)
        # - Reachable from current_head_sha (implicit in the range syntax)
        rev_list_args = ["git", "rev-list", "--no-merges", "--count", current_head_sha]
        rev_list_args.append(f"^{reviewed_head_sha}")
        if base_ref:
            rev_list_args.append(f"^origin/{base_ref}")
        result = subprocess.run(
            rev_list_args,
            cwd=repo_root,
            capture_output=True,
            check=True,
            text=True,
        )
        non_merge_count = int(result.stdout.strip())

        if non_merge_count == 0:
            # Head advanced only by merge commits (base-update) — still a no-op
            failure_msg = (
                f"PR head advanced only by merge commits since request_changes verdict ({reviewed_head_sha} → {current_head_sha}) — "
                f"the rework produced no real work (only base-update merges)"
            )

            # Enrich with unpushed-commit count if worktree exists
            unpushed_info = _get_unpushed_commit_info(head_ref, repo_root)
            if unpushed_info:
                failure_msg += f"; {unpushed_info}"
            else:
                failure_msg += "; check the branch worktree for unpushed work before re-reviewing"

            failures.append(failure_msg)
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        # Git failed (no network, unknown ref, shallow history, or parse error)
        # Fall back to SHA equality result and append a warning
        warnings.append(
            f"Could not verify whether PR head advance ({reviewed_head_sha} → {current_head_sha}) "
            f"included non-merge commits; git fetch/rev-list failed. "
            f"If the advance was only base-update merges, this may be a no-op rework."
        )


def _get_unpushed_commit_info(
    branch: str,
    repo_root: Path,
) -> str | None:
    """Check if the branch has unpushed commits in its local worktree.

    Returns a message with the unpushed commit count and push remediation if
    unpushed commits exist, None otherwise.
    """
    # Try to find the worktree for this branch
    try:
        result = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            cwd=repo_root,
            capture_output=True,
            check=True,
            text=True,
        )

        # Parse worktree list to find the worktree for this branch
        worktree_path = None
        current_worktree = None
        for line in result.stdout.splitlines():
            if line.startswith("worktree "):
                current_worktree = Path(line[len("worktree ") :].strip())
            elif line.startswith("branch ") and current_worktree:
                branch_line = line[len("branch ") :].strip()
                # Branch names may have refs/heads/ prefix
                if branch_line.endswith(f"/{branch}") or branch_line == f"refs/heads/{branch}":
                    worktree_path = current_worktree
                    break

        if not worktree_path or not worktree_path.exists():
            return None

        # Check for unpushed commits in the worktree
        result = subprocess.run(
            ["git", "rev-list", "--count", f"origin/{branch}..HEAD"],
            cwd=worktree_path,
            capture_output=True,
            check=True,
            text=True,
        )

        unpushed_count = int(result.stdout.strip())
        if unpushed_count > 0:
            return (
                f"worktree has {unpushed_count} unpushed commit(s); "
                f"run 'git push origin {branch}' from the worktree to push them"
            )

        return None
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        # Git failed or worktree not found; skip enrichment
        return None


def check_operator_containment(repo_root: Path, pr_diff: str, pr_number: int) -> tuple[str, ...]:
    """Check for worker edits leaked into the operator checkout.

    Runs ``git status --porcelain --untracked-files=no`` at ``repo_root`` to detect
    uncommitted changes (excluding untracked files). When dirty in the context of a
    candidate PR, diffs dirty files against the PR's patch to distinguish:
    - **Leaked worker edits**: files whose working-tree diff against HEAD is byte-identical
      to the PR's diff (provably redundant leaks)
    - **Generic dirty tree**: other dirty files (could be legitimate operator work)

    Untracked files are ignored unless they are introduced by the PR diff as new files.

    NOTE: If main has advanced past the PR base, hunk line numbers can shift and a genuine
    leak may read as unrelated-dirty (false negative). This is a report-only feature and
    acceptable degradation.

    Returns a tuple of warning messages. Never modifies or deletes files.
    """
    warnings: list[str] = []

    # Check for uncommitted changes in the operator checkout (excluding untracked files)
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        dirty_output = result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        # If git fails, skip the check rather than blocking
        return ()

    # Parse dirty files from git status --porcelain -z (NUL-separated, robust parsing)
    # Also check for untracked files that might be in the PR diff
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "-z"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        all_status_output = result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        all_status_output = ""

    if not dirty_output and not all_status_output:
        # Clean tree — no containment issues
        return ()

    # Parse NUL-separated records from git status --porcelain -z
    # Format: XY filename\0 (or XY newname\0oldname\0 for renames/copies with -z)
    dirty_files: set[str] = set()
    untracked_files: set[str] = set()

    if all_status_output:
        records = all_status_output.split("\0")
        i = 0
        while i < len(records):
            record = records[i]
            if not record:
                i += 1
                continue
            # Extract status (first 2 characters)
            if len(record) < 3:
                i += 1
                continue
            status = record[:2]
            # Extract filename (skip the 2-character status and the space after it)
            # The record format is: XY<space>filename
            rest_of_record = record[2:]
            if rest_of_record.startswith(" "):
                rest_of_record = rest_of_record[1:]

            # For renames (R) and copies (C) with -z, the next field is the old path
            # Format: XY newname\0oldname\0
            if status.startswith(("R", "C")):
                # Skip the next record (old path) and use the new name
                filename = rest_of_record
                i += 2  # Skip both current and next record
            else:
                filename = rest_of_record
                i += 1

            if not filename:
                continue

            # Track untracked files separately (status starts with ?)
            if status.startswith("?"):
                untracked_files.add(filename)
            # Track dirty files (modified, added, deleted, etc.)
            else:
                dirty_files.add(filename)

    if not dirty_files and not untracked_files:
        return ()

    # Parse the PR diff to extract hunks for each file
    # We compare hunks directly instead of reconstructing content
    # Split on diff --git section boundaries to avoid multi-file clobbering
    pr_file_hunks: dict[str, str] = {}
    pr_new_files: set[str] = set()

    # Split the diff into sections (each file starts with "diff --git")
    sections = pr_diff.split("\ndiff --git")
    for section in sections:
        if not section.strip():
            continue

        # Re-add the "diff --git" prefix that was stripped by split
        if not section.startswith("diff --git"):
            section = "diff --git" + section

        # Extract the file path from the +++ line
        current_file: str | None = None
        current_hunks: list[str] = []
        is_new_file = False

        for line in section.splitlines():
            if line.startswith("+++ b/"):
                current_file = line[6:]  # Strip "+++ b/" prefix
            elif line.startswith("new file mode"):
                is_new_file = True
            elif line.startswith("@@"):
                # Start of a hunk - include the hunk header
                current_hunks.append(line)
            elif current_hunks:
                # Include all hunk lines (context, additions, deletions)
                # Skip diff metadata lines (starting with \)
                if not line.startswith("\\"):
                    current_hunks.append(line)

        # Save the file's hunks
        if current_file is not None:
            pr_file_hunks[current_file] = "\n".join(current_hunks)
            if is_new_file:
                pr_new_files.add(current_file)

    # Check each dirty file against the PR's hunks
    leaked_files: list[str] = []
    unrelated_dirty_files: list[str] = []

    for dirty_file in dirty_files:
        file_path = repo_root / dirty_file
        if not file_path.exists():
            unrelated_dirty_files.append(dirty_file)
            continue

        # Check if this file is in the PR diff
        if dirty_file not in pr_file_hunks:
            unrelated_dirty_files.append(dirty_file)
            continue

        # Get the working-tree diff against HEAD for this file
        try:
            result = subprocess.run(
                ["git", "diff", "HEAD", "--", dirty_file],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=True,
            )
            working_tree_diff = result.stdout
        except (subprocess.CalledProcessError, FileNotFoundError):
            unrelated_dirty_files.append(dirty_file)
            continue

        # Strip index/hash header lines from both diffs for comparison
        # The header lines vary (hashes, timestamps) but the hunks should match
        def normalize_diff(diff: str) -> str:
            lines = diff.splitlines()
            normalized = []
            for line in lines:
                # Skip header lines that vary between runs
                if (
                    line.startswith("diff --git")
                    or line.startswith("index ")
                    or line.startswith("--- ")
                    or line.startswith("+++ ")
                ):
                    continue
                # Skip diff metadata lines
                if line.startswith("\\"):
                    continue
                normalized.append(line)
            return "\n".join(normalized)

        pr_hunks_normalized = normalize_diff(pr_file_hunks[dirty_file])
        working_tree_hunks_normalized = normalize_diff(working_tree_diff)

        # Compare hunks - byte-identical hunks mean leak
        if working_tree_hunks_normalized == pr_hunks_normalized:
            leaked_files.append(dirty_file)
        else:
            unrelated_dirty_files.append(dirty_file)

    # Check untracked files - only warn if they're introduced by the PR diff
    for untracked_file in untracked_files:
        if untracked_file in pr_new_files:
            # Untracked file that matches a new file in the PR - likely a leak
            leaked_files.append(untracked_file)
        # Otherwise, ignore untracked files entirely (no warning)

    # Generate warnings
    if leaked_files:
        remediation = " ".join(f"git checkout -- {f}" for f in leaked_files)
        warnings.append(
            f"Containment leak detected: PR #{pr_number} edits leaked into operator checkout. "
            f"Files with byte-identical diff to PR: {', '.join(leaked_files)}. "
            f"Remediation: {remediation}"
        )

    if unrelated_dirty_files:
        warnings.append(
            f"Operator checkout has uncommitted changes (not a leak): {', '.join(unrelated_dirty_files)}. "
            f"This may be legitimate operator work or requires manual cleanup."
        )

    return tuple(warnings)
