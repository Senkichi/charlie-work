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

import ast
import builtins
import fnmatch
import logging
import re
import subprocess
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from charlie_work.checks import (
    CheckSummary,
    classify_check_failures,
    classify_infra_failures,
    summarize_checks,
)
from charlie_work.github import linked_issue_number
from charlie_work.safe_ref import require_valid_ref_name, require_valid_sha
from charlie_work.subprocess_runner import (
    hidden_console_kwargs,
    no_console_window_kwargs,
    run_captured,
)

if TYPE_CHECKING:
    from charlie_work.config import OrchestratorConfig, TestAdequacyConfig

# Builtin names that should not be treated as external constants in assertions.
_BUILTIN_NAMES = frozenset(name for name in dir(builtins) if not name.startswith("_"))

logger = logging.getLogger(__name__)


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

# External API/library call patterns. These are heuristic, not AST, and intentionally
# broad: the gate is a warning, not a block, and the goal is to catch invented shapes.
_LINE_EXTERNAL_API_RE = re.compile(
    r"""
    \bgh\s+api\b
    |
    \bcurl\b
    |
    \bwget\b
    |
    \brequests\.(?:get|post|put|delete|patch|head|options)\b
    |
    \bhttpx\.(?:get|post|put|delete|patch|head|options)\b
    |
    \burllib\.request\.(?:urlopen|Request)\b
    |
    \bhttp\.client\.
    |
    \baiohttp\.(?:ClientSession|request|get|post)\b
    |
    ["']gh["']\s*,\s*["']api["']
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Multi-line gh.run / self.run / subprocess.run(["gh", "api", ...) calls.
_MULTILINE_API_CALL_RE = re.compile(
    r"""
    # gh.run(["api", ...]) or self.run(["api", ...]) or any <x>.run(["api", ...])
    # except subprocess.run, which is handled separately.
    (?:\w+)?(?<!subprocess)\.run\s*\([^)]*?\[\s*["']api["'][^)]*\)
    |
    # subprocess.run(["gh", "api", ...])
    subprocess\.run\s*\([^)]*?\[\s*["']gh["']\s*,\s*["']api["'][^)]*\)
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Path segment that identifies a vendored test fixture directory.
_FIXTURE_PATH_RE = re.compile(r"fixtures?[/\\]", re.IGNORECASE)

# Evidence in the PR body that the real external API/library shape was verified.
_LIVE_PAYLOAD_EVIDENCE_RE = re.compile(
    r"(?:"
    r"\b(?:live payload|live call|transcript|fixture|captured|doc link|gh api|curl|wget)\b|"
    r"docs\.github\.com|api\.github\.com|https://docs\.|https://api\."
    r")",
    re.IGNORECASE,
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
    failed_required_checks: tuple[str, ...] = ()
    is_check_failure_block: bool = False
    rerun_run_ids: tuple[int, ...] = ()
    check_rerun_attempts: dict[str, Any] = field(default_factory=dict)
    # Infra-failed (CANCELLED/INFRA_FAILURE/TIMED_OUT) required-check auto-rerun
    # and escalation (issue #841). is_infra_failure_block mirrors
    # is_check_failure_block: True only when an infra-failed required check is
    # the SOLE janitor blocker this pass. Consumers must branch on this
    # structured flag, never on the failure-message text (same rule as the
    # is_draft_only_block/is_no_op_rework flags below).
    is_infra_failure_block: bool = False
    infra_rerun_run_ids: tuple[int, ...] = ()
    infra_rerun_attempts: dict[str, Any] = field(default_factory=dict)
    # Infra-failed required checks that will NOT be retried this pass, either
    # because every failing run id has exhausted auto_merge.infra_rerun_attempt_cap
    # or because no run id could be parsed at all. Non-empty here together with
    # is_infra_failure_block is the caller's signal to escalate to a human --
    # there is no code-fix rework path for an infra failure.
    infra_definitive_failed: tuple[str, ...] = ()
    # Structured flag for _check_no_op_rework's finding (either variant:
    # patch-id or head-SHA unchanged since the last request_changes verdict).
    # Consumers must branch on this, never on the failure-message text.
    is_no_op_rework: bool = False
    # Issue #1116: the no-op rework check was deliberately skipped because the
    # recorded request_changes verdict is stale-CI (its only findings cite
    # required checks that are all green now — is_stale_ci_verdict). An
    # unchanged diff is EXPECTED under a contaminated verdict, and failing the
    # gate on it would permanently block the packet rebuild the fresh review
    # needs. Consumers must branch on this flag, never on the warning text.
    no_op_check_skipped_stale_ci: bool = False
    # Structured flags for _check_draft's finding (issue #818). is_draft is
    # true whenever GitHub reports the PR as a draft, regardless of any other
    # failure. is_draft_only_block is true only when draft is the SOLE janitor
    # failure -- i.e. every other gate (checks, mergeable, linked issue, body,
    # ...) already passed -- which is the "otherwise ready" condition workflow
    # review() uses to decide whether auto-readying the PR is safe. Consumers
    # must branch on these flags, never on the failure-message text.
    is_draft: bool = False
    is_draft_only_block: bool = False
    # Required check names GitHub has reported as missing from the check list
    # (CheckSummary.missing) -- distinct from failed_required_checks (a check
    # that ran and failed). Consumers must branch on this structured field,
    # never on the "Required check(s) missing" failure-message text, to
    # decide whether a gh Actions query for the head SHA is warranted.
    # job-cannon 2026-08-06/07: GitHub Actions silently created no workflow
    # run for pushed heads; detection added so the janitor gate distinguishes
    # "CI never started" from "CI failed".
    missing_required_checks: tuple[str, ...] = ()


def _calculate_patch_id(diff: str) -> str:
    """Calculate a stable patch-id from a diff.

    Feeds the unified diff to ``git patch-id --stable`` and returns the
    resulting SHA-1. This is the canonical plumbing for content-identical
    detection: it ignores metadata (index hashes, file modes) and hunk
    line-number offsets that change on base-update merges but do not affect
    the actual content.

    Args:
        diff: The unified diff string (e.g., from `git diff` or `gh pr diff`)

    Returns:
        A stable patch-id string, or empty string when the diff contains no
        real hunks, is empty/whitespace-only, or when ``git patch-id`` fails.
    """
    if not diff or not diff.strip() or "@@" not in diff:
        return ""

    result = run_captured(
        ["git", "patch-id", "--stable"],
        cwd=Path.cwd(),
        timeout_seconds=30,
        stdin=diff,
    )
    if not result.ok:
        return ""

    for line in result.stdout.splitlines():
        line = line.strip()
        if line:
            return line.split()[0]
    return ""


@dataclass(frozen=True)
class DiffContentSignature:
    """Normalized content signature of a unified diff (issue #414, tier 2).

    ``git patch-id --stable`` hashes hunk content INCLUDING context lines, so
    it is unstable whenever the merge-base moves — which happens on every
    ordinary main advance, not just sibling-PR hunk drift. This signature is
    the tier-2 fallback: it captures only the ordered ``+``/``-`` content
    lines (excluding the ``+++``/``---`` file-marker lines and the
    ``@@ ... @@`` hunk headers, both of which carry line-number offsets that
    shift on every rebase without the actual change moving) plus the set of
    changed file paths (from ``diff --git`` headers, which never carry line
    numbers). Two diffs with an identical signature carried the same actual
    change regardless of where main had advanced to when either was taken.

    ``changed_lines`` is ORDERED and compared with ``==``: a reordering of
    hunks/lines is a real semantic change and must not compare equal.
    ``changed_files`` is a set: file order in the diff is not meaningful.

    ``has_binary`` flags whether any file's diff body was a binary section
    (``Binary files ... differ`` or ``GIT binary patch``). A binary payload
    emits no ``+``/``-`` content lines at all, so two diffs touching the
    SAME path with genuinely DIFFERENT binary content would otherwise
    produce an identical ``changed_lines``/``changed_files`` signature —
    the signature is blind to content it never saw. Callers must treat
    ``has_binary`` as an eligibility gate (fail closed to stale whenever
    set on either side of a comparison), not fold it into the ``==``
    content check.
    """

    changed_lines: tuple[str, ...]
    changed_files: frozenset[str]
    has_binary: bool = False


def _diff_content_signature(diff: str) -> DiffContentSignature:
    """Derive a :class:`DiffContentSignature` from a unified diff string.

    Pure string parsing — no subprocess, no git required — so this never
    fails; an empty or unparseable diff simply yields an empty signature
    (``changed_lines=()``, ``changed_files=frozenset()``, ``has_binary=False``),
    which compares equal only to another empty signature.
    """
    changed_lines: list[str] = []
    changed_files: set[str] = set()
    has_binary = False
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            _, sep, b_path = line[len("diff --git ") :].partition(" b/")
            if sep:
                changed_files.add(b_path)
            continue
        if line.startswith("Binary files ") or line.startswith("GIT binary patch"):
            has_binary = True
            continue
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("@@"):
            continue
        if line.startswith("+") or line.startswith("-"):
            changed_lines.append(line)
    return DiffContentSignature(
        changed_lines=tuple(changed_lines),
        changed_files=frozenset(changed_files),
        has_binary=has_binary,
    )


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
    checks: list[dict[str, Any]] | None,
    config: OrchestratorConfig,
    *,
    pr_state: dict[str, Any] | None = None,
    repo_root: Path | None = None,
    pr_diff: str | None = None,
    review_decision: Mapping[str, Any] | None = None,
) -> JanitorVerdict:
    """Run deterministic pre-review checks over ``pr``/``checks`` data.

    Missing keys in ``pr`` never raise: `gh` omits fields depending on the
    flags used to fetch it, so an absent key is treated as "unknown" and the
    check that depends on it is skipped rather than failed.

    ``review_decision`` is the recorded review-decision mapping for this PR
    (``prs/pr-N/review-decision.json`` content), used only by the issue #1116
    stale-CI skip: when the verdict is a non-escalated request_changes whose
    findings all cite required checks that are green now
    (``is_stale_ci_verdict``), the no-op rework check is skipped so the gate
    can pass and the packet/fresh-review machinery can run. ``None`` (or any
    non-stale decision) preserves the existing behavior — the predicate fails
    closed on red, pending, missing, or unavailable checks.
    """
    failures: list[str] = []
    warnings: list[str] = []
    failed_required_checks: tuple[str, ...] = ()

    required = config.auto_merge.required_checks
    summary: CheckSummary | None = summarize_checks(checks, required) if required else None
    missing_required_checks: tuple[str, ...] = ()
    if summary is not None:
        failed_required_checks = summary.failed
        missing_required_checks = summary.missing

    is_draft = _check_draft(pr, failures)
    _check_state(pr, failures)
    _check_mergeable(pr, failures)
    # Marker to isolate exactly what _check_required_checks contributes below,
    # so is_infra_failure_block (issue #841) can tell "infra_failed is the only
    # required-checks-derived failure" apart from a co-occurring missing/
    # unavailable check, without parsing failure-message text.
    pre_required_checks_len = len(failures)
    _check_required_checks(summary, failures, warnings)
    required_checks_added = len(failures) - pre_required_checks_len
    _check_linked_issue(pr, config, failures)
    _check_body(pr, config, failures)
    _check_title_conventional(pr, warnings)
    _check_diff_size(pr, warnings)
    _check_base_movement(pr, config, warnings)

    if pr_diff is not None:
        _check_external_api_fixtures(pr, pr_diff, config, warnings)

    if pr_diff is not None and pr_diff.strip():
        warnings.extend(check_stub_tests(pr_diff, config.test_adequacy))

    # Only run no-op rework check if repo_root is provided (needed for worktree enrichment)
    is_no_op_rework = False
    # Issue #1116: under a stale-CI verdict an unchanged diff is the expected
    # outcome of rework, not a worker no-op — the recorded findings only cite
    # required checks that are green now, so there was nothing to change.
    # Failing the gate here would permanently block the packet rebuild (and
    # therefore the fresh review) for this PR. is_stale_ci_verdict fails
    # closed: red/pending/missing checks, an escalated or prose-bearing
    # verdict, or a missing summary all leave the no-op check active.
    no_op_check_skipped_stale_ci = (
        review_decision is not None
        and summary is not None
        and is_stale_ci_verdict(review_decision, summary)
    )
    if no_op_check_skipped_stale_ci:
        warnings.append(
            "No-op rework check skipped: recorded request_changes verdict is "
            "stale-CI (all findings cite required checks that are green now)"
        )
    elif repo_root is not None:
        is_no_op_rework = _check_no_op_rework(pr, pr_state, failures, warnings, repo_root, pr_diff)

    is_check_failure_block = bool(failed_required_checks) and not failures

    # issue #841: mirror is_check_failure_block for infra-failed required
    # checks. `failures` at this point still excludes the "Required check(s)
    # failed" message (appended below) but DOES include
    # _check_required_checks' own infra_failed/missing/unavailable messages --
    # `required_checks_added` isolates exactly that contribution so this is
    # True only when infra_failed is non-empty, no missing/unavailable check
    # co-occurs (which would inflate required_checks_added past 1), no
    # genuine code failure co-occurs, and nothing else in `failures` (draft,
    # state, mergeable, linked issue, body, no-op-rework) is set.
    non_required_checks_failures = len(failures) - required_checks_added
    is_infra_failure_block = (
        summary is not None
        and bool(summary.infra_failed)
        and not failed_required_checks
        and not summary.missing
        and not summary.unavailable
        and non_required_checks_failures == 0
    )

    # Flake-aware debounce (issue #391): if the only blocker is failed required
    # checks, decide which runs get a one-time auto-rerun vs which are already
    # retried and therefore definitive. The actual gh run rerun call lives in
    # workflow.review; run_janitor stays pure and just returns the classification.
    rerun_run_ids: tuple[int, ...] = ()
    check_rerun_attempts: dict[str, Any] = {}
    infra_rerun_run_ids: tuple[int, ...] = ()
    infra_rerun_attempts: dict[str, Any] = {}
    infra_definitive_failed: tuple[str, ...] = ()
    if summary is not None:
        head_sha = str(pr.get("headRefOid") or "") or None
        debounce = classify_check_failures(
            checks,
            required,
            pr_state,
            head_sha,
            record_attempts=is_check_failure_block,
        )
        rerun_run_ids = debounce.rerun_run_ids
        check_rerun_attempts = debounce.check_rerun_attempts

        # Infra-failure auto-rerun + escalation (issue #841): same debounce
        # shape as above, but over CANCELLED/INFRA_FAILURE/TIMED_OUT checks,
        # which have no code-fix rework path -- see classify_infra_failures.
        infra_debounce = classify_infra_failures(
            checks,
            required,
            pr_state,
            head_sha,
            record_attempts=is_infra_failure_block,
            attempt_cap=config.auto_merge.infra_rerun_attempt_cap,
        )
        infra_rerun_run_ids = infra_debounce.rerun_run_ids
        infra_rerun_attempts = infra_debounce.infra_rerun_attempts
        infra_definitive_failed = infra_debounce.definitive_failed

    if failed_required_checks:
        failures.append(f"Required check(s) failed: {', '.join(failed_required_checks)}")

    # "Otherwise ready" (issue #818): draft is the ONLY failure once every
    # other gate above (state, mergeable, required checks, linked issue,
    # body, no-op-rework, ...) has had its say. A real failing check or any
    # other janitor failure lengthens `failures` and disqualifies auto-ready.
    is_draft_only_block = is_draft and len(failures) == 1

    return JanitorVerdict(
        ok=not failures,
        failures=tuple(failures),
        warnings=tuple(warnings),
        failed_required_checks=failed_required_checks,
        is_check_failure_block=is_check_failure_block,
        rerun_run_ids=rerun_run_ids,
        check_rerun_attempts=check_rerun_attempts,
        is_infra_failure_block=is_infra_failure_block,
        infra_rerun_run_ids=infra_rerun_run_ids,
        infra_rerun_attempts=infra_rerun_attempts,
        infra_definitive_failed=infra_definitive_failed,
        is_no_op_rework=is_no_op_rework,
        no_op_check_skipped_stale_ci=no_op_check_skipped_stale_ci,
        is_draft=is_draft,
        is_draft_only_block=is_draft_only_block,
        missing_required_checks=missing_required_checks,
    )


def _check_draft(pr: dict[str, Any], failures: list[str]) -> bool:
    """Detect a draft PR. Returns True iff the draft failure was appended."""
    if "isDraft" not in pr:
        return False
    if bool(pr.get("isDraft")):
        failures.append("PR is a draft")
        return True
    return False


def _check_state(pr: dict[str, Any], failures: list[str]) -> None:
    if "state" not in pr:
        return
    state = str(pr.get("state") or "").upper()
    if state and state != "OPEN":
        failures.append(f"PR state is {state}, not OPEN")


def _check_mergeable(pr: dict[str, Any], failures: list[str]) -> None:
    if "mergeable" in pr and str(pr.get("mergeable") or "").upper() == "CONFLICTING":
        failures.append("PR has merge conflicts (mergeable=CONFLICTING)")
    if "mergeStateStatus" in pr and str(pr.get("mergeStateStatus") or "").upper() == "DIRTY":
        failures.append("PR has merge conflicts (mergeStateStatus=DIRTY)")


def _check_required_checks(
    summary: CheckSummary | None,
    failures: list[str],
    warnings: list[str],
) -> None:
    if summary is None or not summary.required:
        return
    if summary.unavailable:
        failures.append(f"Checks unavailable (gh failure): {', '.join(summary.unavailable)}")
    if summary.infra_failed:
        failures.append(
            f"CI never ran (infrastructure failure): {', '.join(summary.infra_failed)}"
        )
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


def _check_external_api_fixtures(
    pr: dict[str, Any],
    pr_diff: str,
    config: OrchestratorConfig,
    warnings: list[str],
) -> None:
    """Warn when the diff adds an external API/library call without live-payload evidence.

    The check is heuristic: it looks for added lines exercising ``gh api`` or HTTP
    client libraries in product files, then requires either a vendored fixture file in
    the diff or explicit evidence in the PR body. Test files and exempt files are
    excluded from call detection; fixture files are detected by a ``fixtures/`` path
    segment.
    """
    body = str(pr.get("body") or "")
    if _LIVE_PAYLOAD_EVIDENCE_RE.search(body):
        return

    added_calls: list[str] = []
    fixture_added = False
    # Reuses test_adequacy path globs even when test_adequacy.enabled is False.
    test_globs = config.test_adequacy.test_path_globs
    exempt_globs = config.test_adequacy.exempt_path_globs

    for filename, _is_new_file, hunk_lines in iter_diff_files(pr_diff):
        if any(fnmatch.fnmatch(filename, glob) for glob in exempt_globs):
            continue
        if _FIXTURE_PATH_RE.search(filename):
            fixture_added = True
            continue
        if any(fnmatch.fnmatch(filename, glob) for glob in test_globs):
            continue

        added_lines = [
            line[1:] for line in hunk_lines if line.startswith("+") and not line.startswith("+++")
        ]
        added_content = "\n".join(added_lines)

        multi_match = _MULTILINE_API_CALL_RE.search(added_content)
        if multi_match:
            added_calls.append(f"{_call_match_repr(multi_match)} in {filename}")
            continue

        for line in added_lines:
            line_match = _LINE_EXTERNAL_API_RE.search(line)
            if line_match:
                added_calls.append(f"{_call_match_repr(line_match)} in {filename}")
                break

    if not added_calls:
        return

    if fixture_added:
        return

    warnings.append(
        "Diff adds external API/library call(s): "
        + ", ".join(added_calls[:3])
        + " with no test fixture sourced from a live payload or PR-body evidence. "
        "Add a fixture under tests/fixtures/ or include a live call transcript / doc link in the PR body."
    )


def _call_match_repr(match: re.Match) -> str:
    """Return a concise, single-line representation of an API-call regex match."""
    text = match.group(0)
    if "\n" in text:
        text = text.splitlines()[0].strip() + "..."
    text = text.strip()
    if len(text) > 80:
        text = text[:77] + "..."
    return text


def _check_no_op_rework(
    pr: dict[str, Any],
    pr_state: dict[str, Any] | None,
    failures: list[str],
    warnings: list[str],
    repo_root: Path,
    pr_diff: str | None = None,
) -> bool:
    """Check if the PR has actual content changes since a request_changes verdict.

    When a PR has a request_changes verdict in its state, compare the current
    patch-id against the reviewed_patch_id from that verdict. If they match,
    the rework produced no actual content changes (no-op rework).

    This is superior to head SHA comparison because base-update merges can
    advance the head SHA without changing the actual diff content (issue #222).

    Falls back to head SHA comparison if patch-id is not available (for backwards
    compatibility with old verdicts).

    Returns True when a no-op-rework failure was appended (any variant), so
    ``run_janitor`` can expose the finding as the structured
    ``JanitorVerdict.is_no_op_rework`` flag instead of consumers matching on
    failure-message text.
    """
    if not pr_state:
        return False

    # Check if the most recent verdict was request_changes
    decision = pr_state.get("decision")
    if decision != "request_changes":
        return False

    # Primary check: compare patch-ids when both are available
    reviewed_patch_id = pr_state.get("reviewed_patch_id")
    if reviewed_patch_id and pr_diff is not None:
        current_patch_id = _calculate_patch_id(pr_diff)
        if current_patch_id == reviewed_patch_id:
            failure_msg = (
                f"PR diff unchanged since request_changes verdict (patch-id {current_patch_id[:12]}...) — "
                f"the rework produced no actual content changes"
            )

            # Enrich with unpushed-commit count if worktree exists
            head_ref = pr.get("headRefName")
            if head_ref:
                try:
                    validated_head_ref = require_valid_ref_name(
                        head_ref, context="_check_no_op_rework head_ref (patch-id)"
                    )
                    validated_base_ref = None
                    base_ref_raw = pr.get("baseRefName")
                    if base_ref_raw:
                        validated_base_ref = require_valid_ref_name(
                            base_ref_raw, context="_check_no_op_rework base_ref (patch-id)"
                        )
                    unpushed_info = _get_unpushed_commit_info(
                        validated_head_ref, repo_root, base_ref=validated_base_ref
                    )
                    if unpushed_info:
                        failure_msg += f"; {unpushed_info}"
                    else:
                        failure_msg += (
                            "; check the branch worktree for unpushed work before re-reviewing"
                        )
                except ValueError as exc:
                    warnings.append(str(exc))
                    failure_msg += (
                        "; check the branch worktree for unpushed work before re-reviewing"
                    )
            else:
                failure_msg += "; check the branch worktree for unpushed work before re-reviewing"

            failures.append(failure_msg)
            return True
        # Patch-id check ran (no match) — skip SHA fallback
        return False

    # Fallback: head SHA comparison (only when patch-id check could not run)

    reviewed_head_sha = pr_state.get("reviewed_head_sha")
    if not reviewed_head_sha:
        return False

    current_head_sha = pr.get("headRefOid")
    if not current_head_sha:
        return False

    try:
        # Validate before the values reach any git argv: reviewed_head_sha comes
        # from persisted state.json, which can diverge or be hand-edited over a
        # long lifetime (issue #659). The ^ prefix in the rev-list arg currently
        # prevents flag parsing, but this guard keeps that true after refactors.
        reviewed_head_sha = require_valid_sha(
            reviewed_head_sha, context="_check_no_op_rework reviewed_head_sha"
        )
        current_head_sha = require_valid_sha(
            current_head_sha, context="_check_no_op_rework current_head_sha"
        )
    except ValueError as exc:
        warnings.append(str(exc))
        return False

    # If heads match exactly, it's a no-op rework
    if current_head_sha == reviewed_head_sha:
        failure_msg = (
            f"PR head unchanged since request_changes verdict ({reviewed_head_sha}) — "
            f"the rework produced no pushed commits"
        )

        # Enrich with unpushed-commit count if worktree exists
        head_ref = pr.get("headRefName")
        if head_ref:
            try:
                validated_head_ref = require_valid_ref_name(
                    head_ref, context="_check_no_op_rework head_ref (sha-match)"
                )
                validated_base_ref = None
                base_ref_raw = pr.get("baseRefName")
                if base_ref_raw:
                    validated_base_ref = require_valid_ref_name(
                        base_ref_raw, context="_check_no_op_rework base_ref (sha-match)"
                    )
                unpushed_info = _get_unpushed_commit_info(
                    validated_head_ref, repo_root, base_ref=validated_base_ref
                )
                if unpushed_info:
                    failure_msg += f"; {unpushed_info}"
                else:
                    failure_msg += (
                        "; check the branch worktree for unpushed work before re-reviewing"
                    )
            except ValueError as exc:
                warnings.append(str(exc))
                failure_msg += "; check the branch worktree for unpushed work before re-reviewing"
        else:
            failure_msg += "; check the branch worktree for unpushed work before re-reviewing"

        failures.append(failure_msg)
        return True

    # Criterion 2: detect merge-only advances (e.g., from ship-it's update_open_prs)
    # Fetch the PR head ref and check if any non-merge commits exist since the verdict
    head_ref = pr.get("headRefName")
    base_ref = pr.get("baseRefName")
    if not head_ref:
        # Can't check merge-only case without ref name; fall back to SHA equality check
        # (which already passed above, so no failure here)
        return False

    try:
        # Validate ref names before they reach git argv (issue #659): head_ref is
        # passed as a plain positional to ``git fetch origin``, so a flag-like
        # value would be parsed as an option without this guard.
        head_ref = require_valid_ref_name(
            head_ref, context="_check_no_op_rework head_ref (merge-only)"
        )
        if base_ref:
            base_ref = require_valid_ref_name(
                base_ref, context="_check_no_op_rework base_ref (merge-only)"
            )

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
            **hidden_console_kwargs(),
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
            **no_console_window_kwargs(),
        )
        non_merge_count = int(result.stdout.strip())

        if non_merge_count == 0:
            # Head advanced only by merge commits (base-update) — still a no-op
            failure_msg = (
                f"PR head advanced only by merge commits since request_changes verdict ({reviewed_head_sha} → {current_head_sha}) — "
                f"the rework produced no real work (only base-update merges)"
            )

            # Enrich with unpushed-commit count if worktree exists
            unpushed_info = _get_unpushed_commit_info(head_ref, repo_root, base_ref=base_ref)
            if unpushed_info:
                failure_msg += f"; {unpushed_info}"
            else:
                failure_msg += "; check the branch worktree for unpushed work before re-reviewing"

            failures.append(failure_msg)
            return True
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as exc:
        # Git failed (no network, unknown ref, shallow history, parse error) or
        # a ref/SHA value failed format validation. Fall back to the SHA equality
        # result and append a warning; include the validation message if that is
        # what failed so callers can tell the difference.
        if isinstance(exc, ValueError):
            warnings.append(str(exc))
        warnings.append(
            f"Could not verify whether PR head advance ({reviewed_head_sha} → {current_head_sha}) "
            f"included non-merge commits; git fetch/rev-list failed. "
            f"If the advance was only base-update merges, this may be a no-op rework."
        )
    return False


def required_check_citation_names(
    decision: Mapping[str, Any] | None,
    required: Sequence[str],
) -> tuple[str, ...] | None:
    """Return the required-check names a request_changes verdict cites, or None.

    A verdict "cites only required checks" when it is a non-escalated
    ``request_changes`` whose ``required_changes`` list is non-empty and EVERY
    entry begins with a configured required-check name followed by ``:`` —
    the shape record_review's #792 derivation produces when a reviewer's sole
    findings were check-status observations (issue #1111: e.g.
    ``"Tests passed: .github:18 — Process completed with exit code 1."``).
    Check names come from ``config.auto_merge.required_checks``, never from a
    hard-coded list. Any entry that does not match — a real code finding, free
    prose, an empty string — makes the whole verdict non-citation (None), so
    mixed verdicts keep their normal lifecycle. Fail-closed by construction:
    None means "treat the verdict as substantive".
    """
    if not decision or not required:
        return None
    if decision.get("decision") != "request_changes" or decision.get("escalated"):
        return None
    required_changes = decision.get("required_changes")
    if not isinstance(required_changes, list) or not required_changes:
        return None
    cited: list[str] = []
    for entry in required_changes:
        if not isinstance(entry, str):
            return None
        text = entry.lstrip()
        match = next((name for name in required if text.startswith(f"{name}:")), None)
        if match is None:
            return None
        cited.append(match)
    return tuple(cited)


def is_stale_ci_verdict(
    decision: Mapping[str, Any] | None,
    summary: CheckSummary | None,
) -> bool:
    """True when a request_changes verdict's only findings are required-check
    citations and every required check is green right now.

    This is the issue #1111 staleness predicate: a reviewer recorded
    ``request_changes`` citing a required CI check (typically a transient
    infra/timeout failure that flipped mid-review), the check has since
    recovered on the same content, and the verdict therefore describes a
    failure that no longer exists. A stale verdict must be superseded by a
    FRESH review — never auto-approved — so callers use this only to (a)
    re-queue the PR for review despite an unchanged patch-id and (b) suppress
    no-op-rework counter burn while that re-review is pending. ``summary``
    must be a live :class:`CheckSummary` over the configured required checks;
    ``None`` (checks unavailable) fails closed to False.
    """
    if summary is None:
        return False
    if required_check_citation_names(decision, summary.required) is None:
        return False
    return summary.ready


def _get_unpushed_commit_info(
    branch: str,
    repo_root: Path,
    base_ref: str | None = None,
) -> str | None:
    """Check if the branch has genuinely unpushed content commits in its local worktree.

    Counts only non-merge commits that are reachable from the worktree's local
    HEAD but not from ``origin/{branch}`` (i.e. not yet pushed) and, when
    ``base_ref`` is available, not already reachable from ``origin/{base_ref}``
    either. Without the base-ref exclusion, a local ``git merge origin/{base_ref}``
    (e.g. to resolve conflicts before a re-review) drags in every already-landed
    squash-merge commit from the base branch's history and reports them as
    "unpushed commits" even though the remote is fully caught up on real content.

    Returns a message with the unpushed commit count and push remediation if
    genuine unpushed content commits exist, None otherwise (including when the
    only local-not-remote commits are base-update merges).
    """
    # Try to find the worktree for this branch
    try:
        result = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            cwd=repo_root,
            capture_output=True,
            check=True,
            text=True,
            **no_console_window_kwargs(),
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

        # Count non-merge commits since origin/{branch}, excluding base-reachable
        # commits. Mirrors the merge-only-advance check above: --no-merges drops
        # base-update merge commits themselves, and ^origin/{base_ref} drops any
        # commits those merges transitively pulled in that are already on the
        # base branch (e.g. other PRs' squash-merge commits).
        rev_list_args = ["git", "rev-list", "--no-merges", "--count", "HEAD"]
        rev_list_args.append(f"^origin/{branch}")
        if base_ref:
            rev_list_args.append(f"^origin/{base_ref}")
        result = subprocess.run(
            rev_list_args,
            cwd=worktree_path,
            capture_output=True,
            check=True,
            text=True,
            **no_console_window_kwargs(),
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


def detect_cross_pr_revert(
    pr: dict[str, Any],
    repo_root: Path | None,
    allow_marker: str = "allow-revert",
) -> str | None:
    """Return a blocking reason if the PR branch would silently revert a base commit.

    Enumerates non-merge commits on the PR branch that are not reachable from
    the base. A commit whose subject is ``Revert "<original>"`` and whose
    inner ``<original>`` subject matches a commit reachable from the base
    indicates that squashing the branch would undo a change already on the base
    (the cross-PR revert incident, issue #390). An explicit ``allow-revert:``
    marker line in the PR body suppresses the block so legitimate intentional
    reverts can be merged.

    Returns ``None`` when no revert is detected, an ``allow-revert:`` marker
    line is present, or the local git history required to decide is unavailable.
    """
    if not repo_root:
        return None

    repo_root_path = Path(repo_root)
    if not repo_root_path.is_dir() or not (repo_root_path / ".git").exists():
        return None

    body = str(pr.get("body") or "")
    # Require a structural marker line: "allow-revert:" followed by a reason.
    # A bare word/substring (e.g. quoting this guidance) must not bypass the gate.
    marker_re = re.compile(
        rf"^{re.escape(allow_marker)}:\s*\S",
        re.IGNORECASE | re.MULTILINE,
    )
    if marker_re.search(body):
        return None

    head_ref = pr.get("headRefName")
    base_ref = pr.get("baseRefName")
    if not head_ref or not base_ref:
        return None

    try:
        # Validate ref names before they reach git argv (issue #659). Both
        # values are passed as plain positionals to ``git fetch origin``, so a
        # flag-like value would be parsed as an option without this guard.
        head_ref = require_valid_ref_name(head_ref, context="detect_cross_pr_revert head_ref")
        base_ref = require_valid_ref_name(base_ref, context="detect_cross_pr_revert base_ref")

        fetch = subprocess.run(
            ["git", "fetch", "origin", str(head_ref), str(base_ref)],
            cwd=repo_root_path,
            capture_output=True,
            text=True,
            check=False,
            **hidden_console_kwargs(),
        )
        if fetch.returncode != 0:
            return None

        commits = subprocess.run(
            [
                "git",
                "rev-list",
                "--no-merges",
                f"origin/{head_ref}",
                f"^origin/{base_ref}",
            ],
            cwd=repo_root_path,
            capture_output=True,
            text=True,
            check=False,
            **no_console_window_kwargs(),
        )
        if commits.returncode != 0:
            return None

        for sha in commits.stdout.strip().splitlines():
            if not sha:
                continue
            subject_proc = subprocess.run(
                ["git", "log", "-1", "--format=%s", sha],
                cwd=repo_root_path,
                capture_output=True,
                text=True,
                check=False,
                **no_console_window_kwargs(),
            )
            if subject_proc.returncode != 0:
                continue
            subject = subject_proc.stdout.strip()
            if subject.startswith('Revert "') and subject.endswith('"'):
                original = subject[len('Revert "') : -1]
                match_proc = subprocess.run(
                    [
                        "git",
                        "log",
                        f"origin/{base_ref}",
                        "--format=%H",
                        "--fixed-strings",
                        "--grep",
                        original,
                    ],
                    cwd=repo_root_path,
                    capture_output=True,
                    text=True,
                    check=False,
                    **no_console_window_kwargs(),
                )
                if match_proc.returncode != 0:
                    continue
                for base_sha in match_proc.stdout.strip().splitlines():
                    if not base_sha:
                        continue
                    base_subject_proc = subprocess.run(
                        ["git", "log", "-1", "--format=%s", base_sha],
                        cwd=repo_root_path,
                        capture_output=True,
                        text=True,
                        check=False,
                        **no_console_window_kwargs(),
                    )
                    if base_subject_proc.returncode != 0:
                        continue
                    if base_subject_proc.stdout.strip() == original:
                        return (
                            f"PR branch contains revert commit {sha[:12]} ({subject}) which "
                            f"would silently undo base commit {base_sha[:12]}; add an explicit "
                            f"'{allow_marker}: <reason>' line to the PR body to proceed"
                        )
    except ValueError as exc:
        # Ref validation failed (issue #659). The function returns None for a
        # false negative, so a diagnostic is required to distinguish this from
        # "no revert detected".
        logger.warning("detect_cross_pr_revert ref validation failed: %s", exc)
        return None
    except OSError:
        return None

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


def check_stub_tests(diff: str, config: TestAdequacyConfig) -> tuple[str, ...]:
    """Flag stub tests in the PR diff.

    Heuristic warnings only — they do not block the janitor gate. The three checks are:
    1. body is only pass/.../docstring
    2. assertion does not reference a product module from the diff
    3. test name contains a seam keyword but the body never calls/mentions it
    """
    warnings: list[str] = []
    try:
        product_modules: set[str] = set()
        test_files: list[tuple[str, list[str]]] = []
        for filename, _is_new_file, hunk_lines in iter_diff_files(diff):
            is_test = any(fnmatch.fnmatch(filename, glob) for glob in config.test_path_globs)
            is_exempt = any(fnmatch.fnmatch(filename, glob) for glob in config.exempt_path_globs)
            if is_test and not is_exempt:
                test_files.append((filename, hunk_lines))
            elif not is_exempt:
                product_modules.add(_module_from_path(filename))

        for filename, hunk_lines in test_files:
            source_lines, touched = _reconstruct_post_image(hunk_lines)
            source = "\n".join(source_lines)
            try:
                tree = ast.parse(source, filename=filename)
            except SyntaxError:
                continue
            test_defined_names = _collect_test_defined_names(tree)
            alias_map = _collect_import_aliases(tree)
            for node in ast.walk(tree):
                if isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef)
                ) and node.name.startswith("test_"):
                    if node.lineno is None:
                        continue
                    end_lineno = node.end_lineno if node.end_lineno is not None else node.lineno
                    if not any(
                        0 < i <= len(touched) and touched[i - 1]
                        for i in range(node.lineno, end_lineno + 1)
                    ):
                        continue
                    if _body_is_only_pass_ellipsis_docstring(node):
                        warnings.append(
                            f"Stub test (FAIL): {filename}:{node.name} body is only pass/.../docstring"
                        )
                    if _assertion_is_constant(
                        node, alias_map, product_modules, test_defined_names
                    ):
                        warnings.append(
                            f"Stub test (assert-constant): {filename}:{node.name} assertion does not reference the module under test"
                        )
                    for keyword in _function_seam_keywords(node, config.stub_test_seam_keywords):
                        if not _body_calls_seam(node, keyword):
                            warnings.append(
                                f"Stub test (seam-name): {filename}:{node.name} name contains '{keyword}' but body never calls it"
                            )
    except Exception:
        warnings.append("diff unparseable — stub test checks skipped")
    return tuple(warnings)


_HUNK_HEADER_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def _module_from_path(filename: str) -> str:
    """Convert a file path to its import-style module name, ignoring src/ layout."""
    path = filename
    if path.startswith("src/"):
        path = path[4:]
    if path.endswith(".py"):
        path = path[:-3]
    return path.replace("/", ".")


def _reconstruct_post_image(hunk_lines: list[str]) -> tuple[list[str], list[bool]]:
    """Build the new-file source from unified-diff hunk lines.

    Returns a list of source lines and a parallel list of booleans indicating
    whether each line was touched by the diff: added (``+``), or adjacent to a
    deletion (``-``). Marking the post-image line just before a deletion is what
    lets deletion-only gutting (removing assertions without adding replacement
    lines) attribute the change to the enclosing function. The reconstruction is
    best-effort; missing context is represented as blank lines.
    """
    source: list[str] = []
    touched: list[bool] = []
    new_index: int | None = None
    for line in hunk_lines:
        header = _HUNK_HEADER_RE.match(line)
        if header:
            new_index = int(header.group(1)) - 1
            continue
        if line.startswith("-"):
            # A deletion sits between post-image lines new_index-1 and
            # new_index; mark the preceding line so the function that lost
            # body lines is inspected even when nothing was added in it.
            if new_index is not None and new_index > 0:
                mark = new_index - 1
                if mark >= len(source):
                    source.extend([""] * (mark + 1 - len(source)))
                    touched.extend([False] * (mark + 1 - len(touched)))
                touched[mark] = True
            continue
        if line.startswith("+"):
            content = line[1:]
            is_touched = True
        elif line.startswith(" "):
            content = line[1:]
            is_touched = False
        else:
            continue
        if new_index is None:
            continue
        if new_index > len(source):
            source.extend([""] * (new_index - len(source)))
            touched.extend([False] * (new_index - len(touched)))
        if new_index < len(source):
            source[new_index] = content
            touched[new_index] = touched[new_index] or is_touched
        else:
            source.append(content)
            touched.append(is_touched)
        new_index += 1
    while source and source[-1] == "" and not touched[-1]:
        source.pop()
        touched.pop()
    return source, touched


def _collect_test_defined_names(tree: ast.AST) -> frozenset[str]:
    """Collect top-level function, class, and assignment names in the test file."""
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                names.update(_names_in_target(target))
        elif isinstance(node, ast.AnnAssign):
            names.update(_names_in_target(node.target))
    return frozenset(names)


def _names_in_target(target: ast.AST) -> list[str]:
    """Extract the ``Name`` ids from an assignment target."""
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        return [name for elt in target.elts for name in _names_in_target(elt)]
    return []


def _collect_import_aliases(tree: ast.AST) -> dict[str, str]:
    """Map top-level import aliases to their fully qualified module names."""
    aliases: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname if alias.asname else alias.name.split(".")[0]
                aliases[local] = alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                continue
            module = node.module or ""
            for alias in node.names:
                if alias.name == "*":
                    continue
                local = alias.asname if alias.asname else alias.name
                aliases[local] = f"{module}.{alias.name}" if module else alias.name
    return aliases


def _collect_local_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> frozenset[str]:
    """Collect argument names and locally assigned names for a function."""
    names: set[str] = set()
    for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
        names.add(arg.arg)
    if node.args.vararg:
        names.add(node.args.vararg.arg)
    if node.args.kwarg:
        names.add(node.args.kwarg.arg)
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
            names.add(child.id)
    return frozenset(names)


def _body_is_only_pass_ellipsis_docstring(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return True if the function body is only pass/.../docstring."""
    for stmt in node.body:
        if isinstance(stmt, ast.Pass):
            continue
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            if isinstance(stmt.value.value, str) or stmt.value.value is ...:
                continue
        return False
    return True


def _assertion_is_constant(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    alias_map: dict[str, str],
    product_modules: set[str],
    test_defined_names: frozenset[str],
) -> bool:
    """Return True if any assertion in the function only references constants."""
    local_names = _collect_local_names(node)
    for child in ast.walk(node):
        if isinstance(child, ast.Assert):
            if _assert_expr_is_constant(
                child, alias_map, product_modules, local_names, test_defined_names
            ):
                return True
    return False


def _assert_expr_is_constant(
    assert_node: ast.Assert,
    alias_map: dict[str, str],
    product_modules: set[str],
    local_names: frozenset[str],
    test_defined_names: frozenset[str],
) -> bool:
    """Return True if the assertion expression does not reference a product module."""
    if any(isinstance(n, ast.Call) for n in ast.walk(assert_node.test)):
        return False
    parent_map: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(assert_node.test):
        for child in ast.iter_child_nodes(node):
            parent_map[child] = node
    has_name = False
    has_product = False
    has_non_trivial = False
    for node in ast.walk(assert_node.test):
        if isinstance(node, ast.Attribute):
            root, attrs = _attribute_root_and_attrs(node)
            if root is None or not isinstance(root, ast.Name):
                has_non_trivial = True
                continue
            if (
                root.id in local_names
                or root.id in test_defined_names
                or root.id in _BUILTIN_NAMES
            ):
                continue
            if root.id in alias_map:
                module = alias_map[root.id]
                for prefix in _attribute_prefixes(module, attrs):
                    if prefix in product_modules:
                        has_product = True
                        break
                else:
                    has_non_trivial = True
                continue
            has_non_trivial = True
            continue
        if isinstance(node, ast.Name):
            has_name = True
            parent = parent_map.get(node)
            if isinstance(parent, ast.Attribute) and parent.value is node:
                continue
            if (
                node.id in local_names
                or node.id in test_defined_names
                or node.id in _BUILTIN_NAMES
            ):
                continue
            if node.id in alias_map:
                if alias_map[node.id] in product_modules:
                    has_product = True
                else:
                    has_non_trivial = True
                continue
            has_non_trivial = True
    if has_product:
        return False
    if has_non_trivial:
        return True
    return not has_name


def _attribute_root_and_attrs(node: ast.Attribute) -> tuple[ast.Name | None, list[str]]:
    """Return the root Name of an attribute chain and the list of attributes in order."""
    attrs: list[str] = []
    current: ast.AST = node
    while isinstance(current, ast.Attribute):
        attrs.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        return current, attrs[::-1]
    return None, []


def _attribute_prefixes(alias_module: str, attrs: list[str]) -> Iterator[str]:
    """Yield the module and each attribute prefix of an attribute chain."""
    module = alias_module
    yield module
    for attr in attrs:
        module = f"{module}.{attr}"
        yield module


def _function_seam_keywords(
    node: ast.FunctionDef | ast.AsyncFunctionDef, keywords: tuple[str, ...]
) -> tuple[str, ...]:
    """Return the configured seam keywords that appear in the function name."""
    funcname = node.name.lower()
    return tuple(kw for kw in keywords if kw in funcname)


def _body_calls_seam(node: ast.FunctionDef | ast.AsyncFunctionDef, keyword: str) -> bool:
    """Return True if the function body contains the seam keyword in an identifier or string."""
    keyword = keyword.lower()
    for stmt in node.body:
        if (
            stmt is node.body[0]
            and isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        ):
            continue
        for child in ast.walk(stmt):
            if isinstance(child, ast.Name):
                if keyword in child.id.lower():
                    return True
            elif isinstance(child, ast.Attribute):
                if keyword in child.attr.lower():
                    return True
            elif isinstance(child, ast.Constant) and isinstance(child.value, str):
                if keyword in child.value.lower():
                    return True
    return False


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
            **no_console_window_kwargs(),
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
            **no_console_window_kwargs(),
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
                **no_console_window_kwargs(),
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
