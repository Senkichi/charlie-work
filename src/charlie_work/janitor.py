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
from dataclasses import dataclass
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
    if linked_issue_number(pr) is None:
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
