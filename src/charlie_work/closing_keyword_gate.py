from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

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
