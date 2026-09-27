"""CLI command layer for the closing-keyword gate (issue #790).

This module is the command wrapper for
:mod:`charlie_work.closing_keyword_gate`, following the same split as
:mod:`charlie_work.private_slug_check_command` and
:mod:`charlie_work.collect_only_gate_command` (subparser registration +
GitHub I/O + exit-code decision here; pure scanning logic in the gate
module). ``run_closing_keyword_check_command`` was extracted out of
``cli.py`` under issue #1872's rework: the #1872 change pushed ``cli.py``
past its file-size high-water mark (issue #1442's ratchet), so the command
moved here byte-identically rather than growing the monolith further.

``cli`` is imported lazily *inside* the functions that need it, for the same
circular-import / ``-m`` guard reasons documented in
:mod:`charlie_work.private_slug_check_command`.
"""

from __future__ import annotations

import argparse

from .closing_keyword_gate import (
    exclude_base_reachable_commits,
    find_unexpected_closing_references,
)
from .github import CLOSING_KEYWORD_PR_FIELDS, defang_closing_keywords
from .issue_linking import linked_issue_number
from .workflow import CommandResult


def register_closing_keyword_check_subparser(
    subparsers: argparse._SubParsersAction,
) -> None:
    """Register the ``closing-keyword-check`` subcommand on *subparsers*."""
    parser = subparsers.add_parser(
        "closing-keyword-check",
        help=(
            "CI gate (issue #790): fail if the PR body or any commit message "
            "contains an unnegated closing keyword (Closes/Fixes/Resolves #N) "
            "referencing an issue other than this PR's own declared target. "
            "GitHub's native auto-close-on-merge scans both surfaces with no "
            "negation awareness at all; this is a required PR check, not a "
            "label-transition helper."
        ),
    )
    parser.add_argument("--pr", type=int, required=True)


def run_closing_keyword_check_command(args: argparse.Namespace) -> CommandResult:
    """CI gate (issue #790): fail on any unnegated closing keyword pointing off-target.

    Fetches the PR's title/body/branch (`GitHub.pr_view`, deliberately scoped
    to `CLOSING_KEYWORD_PR_FIELDS` rather than the general-purpose
    `PR_VIEW_FIELDS` — this gate never touches CI/review/label state, and
    `PR_VIEW_FIELDS`'s `statusCheckRollup` triggers a nested GraphQL
    connection the default Actions `GITHUB_TOKEN` cannot read without
    additional scope grants; see `CLOSING_KEYWORD_PR_FIELDS`'s docstring for
    the two live failures this caused) and every commit's raw message
    (`GitHub.pr_commits` — the REST endpoint, not `gh pr view --json commits`, whose GraphQL fields
    truncate/corrupt long commit messages; see `GitHub.pr_commits`'s
    docstring). The PR's own declared target issue is resolved the same way
    charlie-work's own label-transition binding resolves it
    (`linked_issue_number`: same-repo branch-prefix first, then an unnegated
    closing keyword in the PR's own title/body) — that single number is the
    only exemption `find_unexpected_closing_references` allows. Everything
    else it finds is a reference GitHub's native auto-close-on-merge will act
    on regardless of what this codebase intends, because that GitHub feature
    scans PR body + every commit message with no negation awareness (issue
    #790; PR #788's own commit text is the regression fixture proving this).

    Issue #1872: the ``pulls/{n}/commits`` surface is computed against the
    PR's *recorded* ``base.sha`` (effectively ``rev-list base.sha..head``),
    which lags when a push merges a newer ``main`` — a foreign squash-merge
    commit already on ``main`` then lists as one of this PR's commits and
    false-positives the gate. The merge base is therefore re-resolved against
    the *live* base ref (``GitHub.compare``'s ``merge_base_commit``) and every
    listed commit already reachable from it is excluded before scanning.
    Like the two fetch legs above, an unresolvable live merge base fails
    closed rather than falling back to the stale recorded surface.
    """
    from . import cli  # deferred: see module docstring (circular-import / -m guard)

    ctx = cli.bootstrap_command(args)

    pr = ctx.gh.pr_view(args.pr, fields=CLOSING_KEYWORD_PR_FIELDS)
    if not pr:
        return CommandResult(False, f"closing-keyword-check: could not fetch PR #{args.pr}", {})

    commits = ctx.gh.pr_commits(args.pr)
    if commits is None:
        return CommandResult(
            False, f"closing-keyword-check: could not fetch commits for PR #{args.pr}", {}
        )

    base_ref = pr.get("baseRefName")
    head_sha = pr.get("headRefOid")
    comparison = ctx.gh.compare(str(base_ref), str(head_sha)) if base_ref and head_sha else None
    merge_base_commit = comparison.get("merge_base_commit") if comparison else None
    merge_base_sha = merge_base_commit.get("sha") if isinstance(merge_base_commit, dict) else None
    if not isinstance(merge_base_sha, str) or not merge_base_sha:
        return CommandResult(
            False,
            f"closing-keyword-check: could not resolve live merge base for PR #{args.pr} "
            f"(baseRefName={base_ref!r}, headRefOid={head_sha!r})",
            {},
        )
    scanned_commits = exclude_base_reachable_commits(commits, merge_base_sha=merge_base_sha)
    commit_messages = [str((c.get("commit") or {}).get("message") or "") for c in scanned_commits]

    # Issue #1229 scoping decision: this call site is deliberately NOT
    # threaded through branch_issue_validator. ``intended`` is the single
    # issue number ``find_unexpected_closing_references`` exempts from its
    # unexpected-closing-reference scan; it is a diagnostic/reporting value
    # (surfaced as ``intended_issue_number`` in the command's JSON output),
    # not a key for any issue-label transition or state write. A stale
    # branch-name binding would set ``intended`` to the wrong number, causing
    # the real intended issue's closing keyword to be flagged as an
    # unexpected reference -- a conservative false-positive failure direction
    # (the check blocks rather than corrupts), and one an operator can
    # resolve by rewording the PR body. Threading the validator would also
    # add an ``issue_list(state="open")`` call to a one-shot CLI command that
    # otherwise makes only the two ``pr_view``/``pr_commits`` calls above.
    intended = linked_issue_number(
        pr,
        is_cross_repository=pr.get("isCrossRepository"),
        branch_prefix=ctx.config.dispatch.branch_prefix,
    )

    findings = find_unexpected_closing_references(
        pr_body=str(pr.get("body") or ""),
        commit_messages=commit_messages,
        intended_issue_number=intended,
    )

    data = {
        "pr": args.pr,
        "intended_issue_number": intended,
        "excluded_base_reachable_count": len(commits) - len(scanned_commits),
        "findings": [
            {
                "issue_number": finding.issue_number,
                "source": finding.source,
                "matched_text": finding.matched_text,
            }
            for finding in findings
        ],
    }

    if findings:
        lines = [
            f"  issue #{finding.issue_number} via {finding.source}: "
            f"{finding.matched_text!r} -> reword to {defang_closing_keywords(finding.matched_text)!r}"
            for finding in findings
        ]
        message = (
            f"closing-keyword-check: {len(findings)} unexpected closing reference(s) on "
            f"PR #{args.pr} (declared target: "
            f"{'#' + str(intended) if intended is not None else 'none resolved'})\n"
            + "\n".join(lines)
            + "\nGitHub will auto-close these issues on merge unless the wording above is "
            "changed to the suggested rewrite (or the reference is dropped entirely)."
        )
        return CommandResult(False, message, data)

    return CommandResult(
        True,
        f"closing-keyword-check: clean (PR #{args.pr}, declared target: "
        f"{'#' + str(intended) if intended is not None else 'none resolved'})",
        data,
    )
