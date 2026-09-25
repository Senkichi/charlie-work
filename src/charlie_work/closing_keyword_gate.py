from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .issue_linking import iter_unnegated_closing_keyword_matches


@dataclass(frozen=True)
class UnexpectedClosingReference:
    """A closing-keyword reference to an issue this PR does not declare as its target.

    GitHub's native auto-close-on-merge scans *both* a PR's body and every
    commit message in the PR for ``close(s/d)``/``fix(es/ed)``/``resolve(s/d)``
    followed by ``#N``, and acts on every match — with no regard for
    negation, quoting, backticks, or authorial intent. That is a different
    (and wider) surface than `charlie_work.github.linked_issue_number`, which
    only ever inspects the PR's own title/body for charlie-work's internal
    label-transition binding and never looks at commit messages at all.

    Issue #790: PR #788's own commit message demonstrated the gap it was
    fixing — its commit body illustrated the negated-phrase bug with the
    literal, quoted example text ``"Fixes #649"`` inside a sentence
    ("...the same as \"Fixes #649\""). That match is *not* preceded by a
    negation word within the lookback window, so GitHub auto-closed issue
    #649 on merge even though `closingIssuesReferences` on the PR itself
    (body-only) came back empty. One of these records that exact class of
    finding: a live, unnegated closing-keyword match pointing somewhere other
    than the PR's declared target.
    """

    issue_number: int
    source: str
    matched_text: str


def exclude_base_reachable_commits(
    commits: Sequence[Mapping[str, Any]],
    *,
    merge_base_sha: str,
) -> list[Mapping[str, Any]]:
    """Drop listed commits already reachable from the live base tip (issue #1872).

    The commit surface this gate scans — ``gh api
    repos/{owner}/{repo}/pulls/{n}/commits`` — is computed by GitHub against
    the PR's *recorded* ``base.sha`` (effectively ``rev-list base.sha..head``),
    and that recorded SHA lags: when a PR head merges a newer ``main``, the
    recorded ``base.sha`` can still predate main commits the merge brought in.
    A foreign squash-merge commit already on ``main`` — e.g. another PR's
    ``Closes #N`` — then appears in the commit list and false-positives this
    gate until the recorded base re-syncs.

    ``merge_base_sha`` must be the merge base of the PR head against the
    *live* base ref (``compare``'s ``merge_base_commit.sha``), re-resolved at
    gate time — never the recorded ``base.sha``. Every commit in ``commits``
    that is an ancestor-or-self of that merge base is already on the live
    base and contributes no diff to this PR, so it is excluded from the scan.

    Reachability is derived from each commit object's own ``parents`` array
    (part of the ``pulls/{n}/commits`` REST response schema), walking parent
    links from ``merge_base_sha`` across only the listed commits — no extra
    API calls. The walk stays inside the listed set by construction: any
    ancestor of the merge base that appears in the list is reachable from the
    merge base through intermediate list entries (an intermediate node
    reachable from the recorded base would make the descendant reachable too,
    contradicting its presence in the list).

    Fail-closed on unprovable ancestry: a commit with no ``sha``/``parents``
    in its payload is kept (scanned), and when ``merge_base_sha`` is not
    itself in the list — e.g. the recorded base already equals the live merge
    base — nothing is excluded at all. Under exclusion, ``find_unexpected_
    closing_references``'s ``commit #N`` source labels index the filtered
    list (this PR's own commits), not the raw endpoint ordering.
    """
    by_sha: dict[str, Mapping[str, Any]] = {}
    for commit in commits:
        sha = commit.get("sha")
        if isinstance(sha, str) and sha:
            by_sha[sha] = commit

    reachable_from_base: set[str] = set()
    stack = [merge_base_sha]
    while stack:
        sha = stack.pop()
        if sha in reachable_from_base:
            continue
        commit = by_sha.get(sha)
        if commit is None:
            continue
        reachable_from_base.add(sha)
        for parent in commit.get("parents") or []:
            if isinstance(parent, Mapping):
                parent_sha = parent.get("sha")
                if isinstance(parent_sha, str) and parent_sha:
                    stack.append(parent_sha)

    return [c for c in commits if c.get("sha") not in reachable_from_base]


def find_unexpected_closing_references(
    *,
    pr_body: str,
    commit_messages: Sequence[str],
    intended_issue_number: int | None,
) -> list[UnexpectedClosingReference]:
    """Return every closing-keyword reference GitHub would act on that isn't the intended issue.

    Scans ``pr_body`` and every entry in ``commit_messages`` with
    `iter_unnegated_closing_keyword_matches` — the same negation-aware
    `finditer` + lookback primitive `linked_issue_number` uses for its own
    binding decision, refactored out of `_first_unnegated_closing_keyword_match`
    specifically so this scan cannot drift from that one onto a second,
    hand-rolled regex. GitHub's native auto-close scans the identical two
    surfaces (PR body, every commit message) for the identical keyword set,
    with *no* negation awareness at all — so any unnegated match this
    function finds is a match GitHub itself will act on the moment the PR
    merges.

    ``intended_issue_number`` is the one exemption: the PR's own declared
    target, as resolved by `linked_issue_number` (same-repo branch name
    convention first, then an unnegated closing keyword in the PR's own
    title/body). Every other closing-keyword issue reference found anywhere
    in ``pr_body`` or ``commit_messages`` is a live hazard — an unrelated
    issue GitHub will silently close on merge — and is returned as a finding.

    When ``intended_issue_number`` is ``None`` (e.g. a cross-repository/fork
    PR, or a same-repo PR with no resolvable branch/keyword binding at all),
    every unnegated closing-keyword match found is flagged: there is no
    trusted target to exempt anything against, so nothing is exempt. This
    mirrors `linked_issue_number`'s own fail-closed posture for unknown
    provenance.

    Returns an empty list when clean. Never raises — callers own I/O and
    error handling; this function is a pure scan over already-fetched text.
    """
    findings: list[UnexpectedClosingReference] = []

    for match in iter_unnegated_closing_keyword_matches(pr_body):
        issue_number = int(match.group(1))
        if issue_number != intended_issue_number:
            findings.append(
                UnexpectedClosingReference(
                    issue_number=issue_number,
                    source="pr body",
                    matched_text=match.group(0),
                )
            )

    for index, message in enumerate(commit_messages):
        for match in iter_unnegated_closing_keyword_matches(message):
            issue_number = int(match.group(1))
            if issue_number != intended_issue_number:
                findings.append(
                    UnexpectedClosingReference(
                        issue_number=issue_number,
                        source=f"commit #{index + 1}",
                        matched_text=match.group(0),
                    )
                )

    return findings
