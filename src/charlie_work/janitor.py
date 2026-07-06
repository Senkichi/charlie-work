"""Deterministic, non-LLM pre-review gate ("janitor" pattern).

Every LLM review costs real money. The janitor runs BEFORE review-packet
generation and short-circuits obviously-not-ready PRs (draft, closed,
conflicting, failing required checks, missing linked issue, empty body, no
tests/rationale mention) so no review tokens are spent on them. Research
consensus (see docs/design/extraction-dossier.md, "Deterministic, non-LLM
verification before spending review budget") is to verify cheap, concrete
signals before ever routing to the adversarial LLM reviewer.

The `run_janitor` gate functions themselves are pure: no I/O, no ``gh``
calls. The caller (``workflow.review``) already fetches ``pr`` (``gh pr
view`` JSON) and ``checks`` (``gh pr checks`` JSON) for packet generation, so
it feeds that same data in here first. The module as a whole is not pure,
however: `_check_no_op_rework`, `_get_unpushed_commit_info`, and
`check_operator_containment` shell out to `git` via `subprocess.run` to
compare worktree/branch state against the PR diff.
"""

from __future__ import annotations

import fnmatch
import re
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from charlie_work.checks import summarize_checks
from charlie_work.github import linked_issue_number

if TYPE_CHECKING:
    from charlie_work.config import OrchestratorConfig, TestAdequacyConfig


# Case-insensitive word-boundary regex for tests/rationale markers.
# Matches whole words only to avoid false positives like "test" in "latest".
_TESTS_OR_RATIONALE_RE = re.compile(
    r"\b(?:tests?|tested?|testing|verified?|verification|rationale|no tests because)\b",
    flags=re.IGNORECASE,
)

# Single source of truth for conventional-commit types.
# This tuple must match the documented types in CONTRIBUTING.md and prompts/worker.md.
CONVENTIONAL_COMMIT_TYPES = frozenset(
    {"feat", "fix", "refactor", "docs", "test", "chore", "perf", "ci"}
)

_CONVENTIONAL_COMMIT_RE = re.compile(
    r"^(" + "|".join(sorted(CONVENTIONAL_COMMIT_TYPES)) + r")(\(|:|!)"
)

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


@dataclass(frozen=True)
class TestAdequacyFacts:
    __test__ = False  # Prevent pytest from collecting this as a test class

    added_product_loc: int
    added_test_loc: int
    assertion_count: int
    test_files_changed: int
    untested_product_files: tuple[str, ...]
    exempt: bool
    exempt_reason: str


@dataclass(frozen=True)
class TestAdequacyVerdict:
    __test__ = False  # Prevent pytest from collecting this as a test class

    ok: bool
    failures: tuple[str, ...]
    warnings: tuple[str, ...]
    facts: TestAdequacyFacts


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
        warnings.append(
            "PR title is not conventional-commit shaped (see prompts/worker.md PR requirements)"
        )


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


def iter_diff_files(diff: str) -> Iterator[tuple[str, bool, list[str]]]:
    """Split a unified diff into per-file hunk bodies.

    Yields ``(filename, is_new_file, hunk_lines)`` for each file section in
    ``diff``, where ``hunk_lines`` is every ``@@``-header and hunk-body line
    for that file (diff-metadata lines starting with ``\\`` are dropped).
    Sections with no discoverable ``+++ b/`` path are skipped. This performs
    structural splitting only — it does not tally added/removed lines or
    inspect hunk content beyond locating file/hunk boundaries; line counting
    is the caller's responsibility (see ``check_test_adequacy`` in a later
    module addition).
    """
    sections = diff.split("\ndiff --git")
    for section in sections:
        if not section.strip():
            continue
        if not section.startswith("diff --git"):
            section = "diff --git" + section

        current_file: str | None = None
        current_hunks: list[str] = []
        is_new_file = False

        for line in section.splitlines():
            if line.startswith("+++ b/"):
                current_file = line[6:]
            elif line.startswith("new file mode"):
                is_new_file = True
            elif line.startswith("@@"):
                current_hunks.append(line)
            elif current_hunks:
                if not line.startswith("\\"):
                    current_hunks.append(line)

        if current_file is not None:
            yield current_file, is_new_file, current_hunks


def check_test_adequacy(
    diff: str, pr: dict[str, Any], config: TestAdequacyConfig
) -> TestAdequacyVerdict:
    """Check test adequacy of a PR diff.

    Pure function: no I/O, no ``gh`` calls, no subprocess. Never raises —
    any exception returns an ``ok=True`` verdict with a warning.

    Detects the "pure skip" failure mode (product code changed, zero test files
    touched) deterministically, and routes the fuzzier "tests present but zero
    recognized assertions" case to a warning for Tier 2 to judge.
    """
    # Default facts (all zero/empty) for exemption or parse failure
    default_facts = TestAdequacyFacts(
        added_product_loc=0,
        added_test_loc=0,
        assertion_count=0,
        test_files_changed=0,
        untested_product_files=(),
        exempt=False,
        exempt_reason="",
    )

    try:
        # Step 1: Parse the diff using the shared hunk parser
        added_product_loc = 0
        added_test_loc = 0
        assertion_count = 0
        test_files_changed = 0
        untested_product_files: list[str] = []

        files_parsed = False
        for filename, is_new_file, hunk_lines in iter_diff_files(diff):
            files_parsed = True
            # Step 2: Partition file into test/product/exempt
            # test_path_globs wins over exempt_path_globs on overlap
            is_test = any(fnmatch.fnmatch(filename, glob) for glob in config.test_path_globs)
            is_exempt = any(fnmatch.fnmatch(filename, glob) for glob in config.exempt_path_globs)

            if is_test:
                test_files_changed += 1
            elif is_exempt:
                continue  # Skip exempt files entirely
            else:
                # Product file
                pass

            # Step 3: Count added lines
            file_added_loc = 0
            for line in hunk_lines:
                # Added line: starts with '+' and not '+++'
                if line.startswith("+") and not line.startswith("+++"):
                    # Check if blank/comment
                    stripped = line[1:].strip()  # Remove the '+' prefix
                    if stripped and not any(
                        stripped.startswith(prefix) for prefix in config.comment_prefixes
                    ):
                        file_added_loc += 1
                        # Step 4: Count assertions in test files
                        if is_test:
                            if any(marker in line for marker in config.assertion_markers):
                                assertion_count += 1

            if is_test:
                added_test_loc += file_added_loc
            else:
                added_product_loc += file_added_loc
                if file_added_loc > 0:
                    untested_product_files.append(filename)

        # If no files were parsed, the diff is malformed or binary
        # Exception: non-empty diff with no files parsed could be a rename-only diff
        # (100% similarity, no hunk body), which should pass with 0 added lines
        if not files_parsed:
            if diff.strip() and "diff --git" in diff:
                # Valid diff format but no hunks (e.g., rename-only)
                return TestAdequacyVerdict(ok=True, failures=(), warnings=(), facts=default_facts)
            else:
                # Malformed or binary diff
                return TestAdequacyVerdict(
                    ok=True,
                    failures=(),
                    warnings=("diff unparseable — test-adequacy skipped",),
                    facts=default_facts,
                )

        # Step 5: Exemption check
        exempt = False
        exempt_reason = ""
        body = pr.get("body") or ""
        exempt_re = re.compile(rf"^{re.escape(config.exempt_marker)}\s*(?P<reason>.+)$", re.M)
        match = exempt_re.search(body)
        if match:
            reason = match.group("reason").strip()
            if reason:  # Non-empty reason required
                exempt = True
                exempt_reason = reason

        facts = TestAdequacyFacts(
            added_product_loc=added_product_loc,
            added_test_loc=added_test_loc,
            assertion_count=assertion_count,
            test_files_changed=test_files_changed,
            untested_product_files=tuple(untested_product_files),
            exempt=exempt,
            exempt_reason=exempt_reason,
        )

        # Step 6: Verdict
        if exempt:
            return TestAdequacyVerdict(ok=True, failures=(), warnings=(), facts=facts)

        failures: list[str] = []
        warnings: list[str] = []

        if added_product_loc >= config.min_product_lines and test_files_changed == 0:
            # Hard fail: pure skip
            failures.append(
                f"Product code changed ({added_product_loc} LOC added) but no test files changed. "
                f"Untested product files: {', '.join(untested_product_files)}. "
                f"Add tests or use '{config.exempt_marker} <reason>' in the PR body to exempt."
            )
            return TestAdequacyVerdict(
                ok=False, failures=tuple(failures), warnings=(), facts=facts
            )

        if (
            added_product_loc >= config.min_product_lines
            and test_files_changed > 0
            and assertion_count == 0
        ):
            if config.require_assertions:
                # Hard fail: zero assertions when required
                failures.append(
                    f"Test files changed ({test_files_changed}) but zero recognized assertions found. "
                    f"Configure assertion_markers for your assertion style or use '{config.exempt_marker} <reason>' to exempt."
                )
                return TestAdequacyVerdict(
                    ok=False, failures=tuple(failures), warnings=(), facts=facts
                )
            else:
                # Warn: possibly hollow tests
                warnings.append(
                    f"Test files changed ({test_files_changed}) but zero recognized assertions found. "
                    f"Tests may be hollow (over-mocked, tautological, or using custom assertion helpers). "
                    f"Configure assertion_markers or set require_assertions=True to hard-fail this case."
                )

        return TestAdequacyVerdict(
            ok=True, failures=tuple(failures), warnings=tuple(warnings), facts=facts
        )

    except Exception:
        # Never raise — return a safe default on any error
        return TestAdequacyVerdict(
            ok=True,
            failures=(),
            warnings=("diff unparseable — test-adequacy skipped",),
            facts=default_facts,
        )


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

    for filename, is_new_file, hunk_lines in iter_diff_files(pr_diff):
        pr_file_hunks[filename] = "\n".join(hunk_lines)
        if is_new_file:
            pr_new_files.add(filename)

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
