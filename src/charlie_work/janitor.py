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


# Case-insensitive markers scanned for in the PR body when
# `config.review.require_tests_or_rationale` is set. Presence of any one of
# these substrings is treated as evidence the author addressed testing or
# gave a rationale for omitting it (e.g. "no tests because ...").
TESTS_OR_RATIONALE_MARKERS = frozenset(
    {
        "test",
        "tests",
        "tested",
        "testing",
        "verified",
        "verification",
        "rationale",
        "no tests because",
    }
)

_CONVENTIONAL_COMMIT_RE = re.compile(r"^(feat|fix|refactor|docs|test|chore|perf|ci)(\(|:|!)")

# Oversized-diff warning threshold: additions + deletions above this line
# count flags the PR as a warning (not a block) for reviewer awareness.
_OVERSIZED_DIFF_THRESHOLD = 1500


@dataclass(frozen=True)
class JanitorVerdict:
    ok: bool
    failures: tuple[str, ...]
    warnings: tuple[str, ...]


def run_janitor(
    pr: dict[str, Any], checks: list[dict[str, Any]], config: OrchestratorConfig
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
        lowered = body.lower()
        if not any(marker in lowered for marker in TESTS_OR_RATIONALE_MARKERS):
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


def check_operator_containment(
    repo_root: Path, pr_diff: str, pr_number: int
) -> tuple[str, ...]:
    """Check for worker edits leaked into the operator checkout.

    Runs ``git status --porcelain`` at ``repo_root`` to detect uncommitted changes.
    When dirty in the context of a candidate PR, diffs dirty files against the PR's
    patch to distinguish:
    - **Leaked worker edits**: files whose working-tree content is byte-identical to
      the PR's post-image (provably redundant leaks)
    - **Generic dirty tree**: other dirty files (could be legitimate operator work)

    Returns a tuple of warning messages. Never modifies or deletes files.
    """
    warnings: list[str] = []

    # Check for uncommitted changes in the operator checkout
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        dirty_output = result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        # If git fails, skip the check rather than blocking
        return ()

    if not dirty_output:
        # Clean tree — no containment issues
        return ()

    # Parse dirty files from git status --porcelain
    # Format: XY filename (X=staging, Y=worktree) or " XY filename" (with leading space)
    dirty_files: set[str] = set()
    for line in dirty_output.splitlines():
        if not line:
            continue
        # Strip leading whitespace (git status --porcelain can have leading space)
        line = line.lstrip()
        # Extract filename (skip the 2-character status prefix and the space after it)
        # Handle both "XY filename" and "XY filename -> newname" (renames)
        if len(line) > 2:
            # Skip the first 2 characters (XY status)
            rest_of_line = line[2:]
            # Skip the space after XY
            if rest_of_line.startswith(" "):
                rest_of_line = rest_of_line[1:]
            # Split on whitespace and take the first part (the filename)
            parts = rest_of_line.split()
            filename = parts[0] if parts else ""
            if filename:
                dirty_files.add(filename)

    if not dirty_files:
        return ()

    # Parse the PR diff to extract post-image content for each file
    # The post-image is what the file looks like after applying the PR
    pr_file_contents: dict[str, str] = {}
    current_file: str | None = None
    current_content: list[str] = []
    in_hunk = False

    for line in pr_diff.splitlines():
        if line.startswith("+++ b/"):
            # New file in the diff - save previous file's content
            if current_file is not None:
                pr_file_contents[current_file] = "\n".join(current_content)
            current_file = line[6:]  # Strip "+++ b/" prefix
            current_content = []
            in_hunk = False
        elif line.startswith("@@"):
            # Start of a hunk
            in_hunk = True
        elif in_hunk:
            if line.startswith(" "):
                # Context line - present in both pre and post image
                current_content.append(line[1:])
            elif line.startswith("+") and not line.startswith("++"):
                # Added line - only in post image
                current_content.append(line[1:])
            elif line.startswith("-"):
                # Removed line - only in pre image, skip for post-image
                continue
            elif line.startswith("\\"):
                # Diff metadata (e.g., " No newline at end of file")
                continue

    # Don't forget the last file
    if current_file is not None:
        pr_file_contents[current_file] = "\n".join(current_content)

    # Check each dirty file against the PR's post-image
    leaked_files: list[str] = []
    unrelated_dirty_files: list[str] = []

    for dirty_file in dirty_files:
        file_path = repo_root / dirty_file
        if not file_path.exists():
            unrelated_dirty_files.append(dirty_file)
            continue

        # Read the working-tree content
        try:
            working_tree_content = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            unrelated_dirty_files.append(dirty_file)
            continue

        # Check if this file is in the PR diff
        if dirty_file not in pr_file_contents:
            unrelated_dirty_files.append(dirty_file)
            continue

        # Compare byte-identical content
        pr_content = pr_file_contents[dirty_file]
        if working_tree_content == pr_content:
            leaked_files.append(dirty_file)
        else:
            unrelated_dirty_files.append(dirty_file)

    # Generate warnings
    if leaked_files:
        remediation = " ".join(f"git checkout -- {f}" for f in leaked_files)
        warnings.append(
            f"Containment leak detected: PR #{pr_number} edits leaked into operator checkout. "
            f"Files with byte-identical content to PR post-image: {', '.join(leaked_files)}. "
            f"Remediation: {remediation}"
        )

    if unrelated_dirty_files:
        warnings.append(
            f"Operator checkout has uncommitted changes (not a leak): {', '.join(unrelated_dirty_files)}. "
            f"This may be legitimate operator work or requires manual cleanup."
        )

    return tuple(warnings)
