"""Validate and canonicalize the ``Closes #N`` line in orchestrator-written PR bodies.

Scope (cw#1263, binding design from the issue's phase-0-recon comment):
covers only the closing line the *orchestrator itself* writes when it builds
a salvage PR body -- it already knows the true target issue number, so this
module never re-derives the target via ``linked_issue_number``'s
branch-prefix/keyword resolution (that machinery answers a different
question: "which issue does an *arbitrary* PR appear to target", used for
worker-authored PRs and hijack-safety).

Two independent event kinds cover two independent gaps:

- ``pr_closing_ref_rewritten`` (this module, write-time): the orchestrator's
  own salvage-body builders (``workflow.py::_open_salvage_pr`` and
  ``reconcile.py``'s ``session_unpublished_work_salvaged`` branch) are the
  only PR-creation call sites this module can reach -- roughly a third of
  PRs. Both route through :func:`validate_closing_reference` so they cannot
  drift onto their own hand-rolled ``Closes #N`` string apart from each
  other.
- ``pr_closing_ref_unlinked`` (logged by the callers, not this module,
  post-create): compares GitHub's own ``closingIssuesReferences``
  resolution against the intended issue for *every* orchestrator-created PR
  (salvage and worker-authored alike, wherever the caller can reach a fresh
  PR number) -- this is what actually covers the worker-authored majority,
  since workers write their own bodies with no orchestrator helper in the
  loop at all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol

# Mirrors `github._CLOSING_KEYWORDS_ALT` -- deliberately not imported from
# there. That module's regex is scoped to bare `#N` (it exists to detect
# GitHub's own auto-close scanning, which never accepts an owner/repo
# qualifier from a same-repo PR body the way *this* module's canonical
# rewrite needs to preserve one). Duplicating the keyword vocabulary as a
# short, self-contained pattern keeps this module free of a github.py import
# it does not otherwise need.
# (`_CLOSING_KEYWORDS_ALT` now lives in `issue_linking.py`, re-exported through `github.py` -- Track 2, issue #1613 -- but the reasoning above is unchanged.)
_CLOSING_KEYWORDS_ALT = r"(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)"
# A whole closing-reference line: keyword, then an optional `owner/repo`
# qualifier, then `#N`. Anchored to the whole line (MULTILINE + $) so a
# keyword appearing mid-sentence in prose ("this closes #4 loose ends") is
# not mistaken for a structured closing-reference line -- the orchestrator
# only ever writes (and only ever needs to recognize) a line consisting of
# nothing else.
_CLOSING_LINE_RE = re.compile(
    r"^[ \t]*" + _CLOSING_KEYWORDS_ALT + r"[ \t]+(?:([\w.-]+/[\w.-]+))?#(\d+)[ \t]*$",
    flags=re.IGNORECASE | re.MULTILINE,
)


@dataclass(frozen=True)
class ClosingLineMatch:
    """One recognized ``Closes #N`` / ``Closes owner/repo#N`` line.

    ``start``/``end`` are the match's character offsets in the body it was
    found in. Corrections below splice by these offsets, never by searching
    for ``line`` as a literal substring -- the exact line text can (and, in
    the field, does) also occur as an unstructured substring elsewhere in
    the body, e.g. inside prose pulled from a commit-message summary
    (``summarize_branch_work``). A content-based ``str.replace`` would then
    patch that earlier, unrelated occurrence instead of the actual
    structured line at ``start``:``end``, leaving the real defect (wrong or
    duplicate closing line) untouched in the output.
    """

    line: str
    repo: str | None
    issue_number: int
    start: int
    end: int


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of validating/canonicalizing an orchestrator-written PR body.

    ``body`` is always a body safe to pass to ``gh.pr_create`` -- this
    function never returns something that would block PR creation, and
    never raises (all ``gh`` interaction, when a bounded liveness probe is
    requested, is wrapped so a probe failure degrades to
    ``target_issue_open=None`` rather than propagating).

    ``changed`` is True whenever ``body`` differs from the input body.
    ``findings`` names each correction applied (empty when unchanged).
    ``target_issue_open`` is ``True``/``False`` when a ``gh`` probe was
    supplied and answered; ``None`` when no probe was requested or the probe
    itself failed/returned nothing -- the closed-issue condition is
    informational only and never withholds or blocks the body.
    """

    body: str
    changed: bool
    findings: tuple[str, ...]
    target_issue_open: bool | None


class _IssueViewer(Protocol):
    """Narrow protocol for the optional liveness probe.

    Matches the single method this module calls on ``github.GitHubLike``
    (``issue_view(number) -> dict``) without importing the wider protocol --
    tests pass a bare stub with just this method.
    """

    def issue_view(self, number: int) -> dict[str, Any]: ...


def validate_closing_reference(
    body: str,
    issue_number: int,
    repo: str,
    gh: _IssueViewer | None = None,
) -> ValidationResult:
    """Ensure ``body`` contains exactly one correct ``Closes #issue_number`` line.

    ``repo`` is the ``owner/repo`` the target issue lives in. It is used two
    ways: (1) an existing, already-correct ``owner/repo#N`` line is left
    byte-for-byte untouched only when its qualifier matches ``repo`` --
    dropping a correct qualifier down to a bare ``#N`` would silently
    re-target the reference at whatever repo the PR happens to be opened
    against; (2) it is the ``gh issue view`` scope for the optional
    liveness probe.

    Correction rules, in priority order:

    1. Zero closing lines found -> append a canonical ``Closes #N`` line
       (finding: "missing closing line added").
    2. Exactly one closing line, but its issue number is wrong (or its
       ``owner/repo`` qualifier doesn't match ``repo``) -> rewrite just that
       line to canonical form, preserving an existing qualifier style (if
       the line was already qualified, rewrite with a qualifier; otherwise
       bare) (finding: "closing reference rewritten").
    3. Exactly one closing line and it is already correct -> untouched,
       ``changed=False`` (this is the path that proves a correct
       ``owner/repo#N`` form survives validation).
    4. More than one closing line -> collapse to a single canonical line at
       the first match's position, dropping the rest (finding: "multiple
       closing lines collapsed to one").

    Never raises. Never returns a body without a closing line -- rule 1
    guarantees at least one is always present in the output.
    """
    matches = [
        ClosingLineMatch(
            line=m.group(0),
            repo=m.group(1),
            issue_number=int(m.group(2)),
            start=m.start(),
            end=m.end(),
        )
        for m in _CLOSING_LINE_RE.finditer(body)
    ]

    findings: list[str] = []
    new_body = body

    if not matches:
        canonical = f"Closes #{issue_number}"
        # Prepend with a blank-line separator so the closing reference reads
        # as its own line, matching the shape `_open_salvage_pr` already
        # writes (`Closes #N\n\n<rest>`); no separator needed for an empty
        # body.
        new_body = f"{canonical}\n\n{body}" if body.strip() else canonical
        findings.append("missing closing line added")
    elif len(matches) == 1:
        match = matches[0]
        # A correct reference resolves to the right issue AND, if it names a
        # repo at all, names the right repo. An unqualified `#N` is correct
        # regardless of `repo` (bare form is always same-repo-relative and
        # this module's callers only ever build bodies for PRs opened in
        # `repo`).
        is_correct = match.issue_number == issue_number and (
            match.repo is None or match.repo == repo
        )
        if is_correct:
            findings.clear()
        else:
            # Preserve the existing line's qualified/bare STYLE, but always
            # substitute the correct `repo` -- the whole point of this branch
            # is that `match.repo` (when present) is untrusted: it is either
            # absent, or it names the wrong repo. Reusing `match.repo` here
            # would silently keep pointing the rewritten line at the wrong
            # repo whenever the original qualifier was itself the defect.
            qualifier = f"{repo}#" if match.repo else "#"
            canonical_line = f"Closes {qualifier}{issue_number}"
            # Splice by offset, not by content -- see `ClosingLineMatch`.
            new_body = body[: match.start] + canonical_line + body[match.end :]
            findings.append("closing reference rewritten")
    else:
        canonical = f"Closes #{issue_number}"
        # Splice every match by offset in a single left-to-right pass so
        # matches never shift each other's remaining offsets and the same
        # literal text occurring elsewhere in the body (e.g. in prose) is
        # never touched.
        pieces: list[str] = []
        cursor = 0
        for index, match in enumerate(matches):
            pieces.append(body[cursor : match.start])
            pieces.append(canonical if index == 0 else "")
            cursor = match.end
        pieces.append(body[cursor:])
        new_body = "".join(pieces)
        findings.append("multiple closing lines collapsed to one")

    target_issue_open: bool | None = None
    if gh is not None:
        try:
            issue = gh.issue_view(issue_number)
        except Exception:
            issue = None
        if isinstance(issue, dict) and issue:
            state = str(issue.get("state") or "").upper()
            if state in ("OPEN", "CLOSED"):
                target_issue_open = state == "OPEN"

    return ValidationResult(
        body=new_body,
        changed=new_body != body,
        findings=tuple(findings),
        target_issue_open=target_issue_open,
    )


def closing_issues_referenced_numbers(pr_view: dict[str, Any]) -> set[int]:
    """Parse the GraphQL ``closingIssuesReferences`` field from a ``pr_view`` result.

    Pure parsing, no I/O -- callers fetch ``pr_view`` themselves (typically
    via ``gh.pr_view(pr_number, fields=github.PR_CLOSING_ISSUES_FIELDS)``)
    and pass the result here. Returns the empty set for a missing/malformed
    field rather than raising, matching this module's errors-as-values
    posture; a caller comparing against an intended issue number then simply
    checks membership.
    """
    refs = pr_view.get("closingIssuesReferences")
    if not isinstance(refs, list):
        return set()
    numbers: set[int] = set()
    for ref in refs:
        if isinstance(ref, dict) and isinstance(ref.get("number"), int):
            numbers.add(ref["number"])
    return numbers
