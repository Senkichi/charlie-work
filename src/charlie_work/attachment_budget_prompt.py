"""Attachment-budget prompt/packet text (issue #1460).

Pure text-formatting only, extracted out of ``workflow.py`` per the #1442
file-size ratchet's prescribed remedy for over-cap files: new code must land
in a domain module and be re-exported through ``workflow.py``'s facade
import block, not grow the monolith directly. See the
``.dispatch_selection`` / ``.escalation`` / ``.verdict_parsing`` /
``.rework_prompts`` / ``.ci_findings`` / ``.backlog_reachability`` /
``.stalled_review_reap`` re-export blocks at the top of ``workflow.py`` for
the established lineage this extraction follows.

Both symbols here are pure and free of ``self``/``OrchestratorApp`` state:
the stateful gating and I/O live in
``OrchestratorApp._build_attachment_budget_value`` and
``OrchestratorApp._build_attachment_budget_section`` in ``workflow.py``,
which import from this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .attachment_contracts.review_delta import BudgetSection

# Issue #1460: the static dispatch-clause prose emitted by
# ``_build_attachment_budget_value`` when `.attachment-budgets/` is
# present and structurally valid. A module-level constant (not inlined in
# the method) so it renders identically regardless of call site and stays
# trivially diffable against the plan's exact prose.
ATTACHMENT_BUDGET_CLAUSE = """\
## Attachment-point placement contract

This repository enforces attachment-point contracts (member counts on
attachment points, never line counts). Before adding code that binds a new
member to an existing attachment point -- a new command on a CLI app, a new
route on a blueprint, a new method on a class, a new test in a test module,
a new migration -- check whether the object you are extending is already the
saturated owner for its archetype:

    python -m charlie_work.attachment_contracts check-file <path-you-will-edit>

If that path's attachment point is saturated (the command reports a `block`
finding with a redirect), do NOT add the new member there and do NOT raise
the baseline. Place the new member in the suggested sibling/new module from
the `redirect` field instead. Scaffold the redirect destination if it does
not exist yet.

If a placement advisory fires mid-session while you are editing, take the
redirect it names rather than bumping the baseline. Bumping the baseline is
reserved for cases with an external, cited justification (an issue or PR
reference) supplied by the dispatch prompt or a human -- a worker may not
author its own bump justification, and a review-time gate will block any
bump whose acknowledgement you invented.

If ANY placement advisory fired during this session (the hook appends each
to `.var/attachment-contracts/advisories.jsonl` in your worktree), publish
a single PR comment surfacing them so the review packet can read it: the
comment body MUST start with the marker line `<!-- attachment-advisories v1 -->`
followed by a fenced `json` block containing a JSON array of your advisory
records, each an object with the fields `severity`, `file`, `identity`,
`message`, `redirect`, `timestamp` (the same schema the hook logs). Post it
once at PR-open time and again on any subsequent push that fires new
advisories, replacing the prior comment's content. The review-packet
builder reads this PR-comment channel in preference to your worktree-local
log (which it generally cannot see); without it, the redirects-not-taken
section of your review packet renders a "log not available" NOTE.
"""


def render_attachment_budget_section(section: BudgetSection | None) -> str:
    """Render the ``$attachment_budget_section`` packet block (issue #1460).

    Returns ``""`` when ``section`` is ``None`` (the review() cheap gate
    decided this PR neither touches `.attachment-budgets/` nor a
    baselined host file, or the marker is absent), mirroring
    ``render_over_cap_section``'s disabled contract. When gated in, this
    ALWAYS renders visible text -- even with zero findings -- rather than
    ``""`` for a clean pass, the same never-silent contract: an advisory
    section that goes silent on a clean run is indistinguishable, from the
    packet alone, from one that never ran.

    Row order: BLOCKING rows first (the ones a reviewer must act on), then
    every new bump, then saturated-but-touched hosts, then redirects not
    taken, then never-silent NOTE rows for anything the section could not
    evaluate.
    """
    if section is None:
        return ""

    lines = ["## Attachment-budget diff"]

    for entry, bump in section.blocking_bumps:
        lines.append(
            "- BLOCKING -- worker-authored baseline bump without external "
            f"acknowledgement: {entry.identity} ({entry.file}): member ceiling "
            f"bumped to {bump.to}, actor=worker, ack={bump.ack or '(empty)'}. "
            "A worker may not justify its own bump. Require an external "
            "citation (issue/PR reference) or reject this bump."
        )

    for entry, bump in section.bumps:
        lines.append(
            f"- {entry.identity} ({entry.file}): -> {bump.to}  reason={bump.reason}  "
            f"actor={bump.actor}  ack={bump.ack or '(empty)'}"
        )

    for entry in section.saturated_touched:
        lines.append(
            f"- {entry.identity} ({entry.file}): frozen at {entry.member_count} "
            f"members [{entry.kind}] -- verify no new members were bound; "
            "growth is enforced at generation time / CI"
        )

    for record in section.redirects_not_taken:
        lines.append(
            f"- redirect not taken: {record.identity} ({record.file}) advised "
            f"redirect to `{record.redirect}`, which this diff does not touch: "
            f"{record.message}"
        )

    for point in section.ratchetable:
        lines.append(
            f"- {point.identity} ({point.file}): live member count "
            f"{point.live_count} is below baseline {point.baseline_members} -- "
            "a ratchet, not a bump. Run `python -m charlie_work.attachment_contracts "
            "baseline --ratchet` and commit the resulting `.attachment-budgets/` "
            "tightening in this PR -- the command is lower-only (it never raises a "
            "baseline entry), so it is safe to run mid-PR. A lowered count is a "
            "ratchet, not tamper: G4 (workers may not self-ack bumps) governs raises "
            "only -- CI re-verifies `actual <= baseline` deterministically from the "
            "scan, so there is nothing for a worker to launder by self-committing a "
            "decrease."
        )

    if section.head_unreadable:
        lines.append(
            "- NOTE: could not evaluate .attachment-budgets/ at PR head; "
            "bump and G4 checks skipped"
        )
    if section.advisories_unavailable:
        lines.append(
            "- NOTE: advisories log not available for this PR; "
            "redirects-not-taken could not be computed"
        )

    if len(lines) == 1:
        lines.append("No saturated-point growth or baseline bumps detected in this PR's diff.")

    return "\n".join(lines) + "\n"


def _diff_file_summary(diff: str) -> tuple[int, list[tuple[str, int, int]]]:
    """Return (total_lines, per_file_stats) from a unified diff.

    ``per_file_stats`` is a list of ``(filename, added, deleted)`` tuples.
    ``total_lines`` counts content lines (not diff headers/meta lines).
    """
    files: list[tuple[str, int, int]] = []
    current_file = ""
    added = 0
    deleted = 0
    total = 0
    for line in diff.splitlines():
        if line.startswith("diff --git"):
            if current_file:
                files.append((current_file, added, deleted))
            current_file = ""
            added = 0
            deleted = 0
        elif line.startswith("+++ "):
            current_file = line[4:].strip()
            if current_file == "/dev/null":
                current_file = ""
        elif line.startswith("--- "):
            # Use the source file if the dest is /dev/null (deletion)
            if not current_file:
                current_file = line[4:].strip()
                if current_file == "/dev/null":
                    current_file = ""
        elif line.startswith("+") and not line.startswith("+++"):
            added += 1
            total += 1
        elif line.startswith("-") and not line.startswith("---"):
            deleted += 1
            total += 1
    if current_file:
        files.append((current_file, added, deleted))
    return total, files


@dataclass(frozen=True)
class OverCapFileFinding:
    """One file whose post-diff line count exceeds the repo size cap and whose
    diff adds code to it (issue #1445).

    ``line_count`` is the post-diff line count (base file at ``repo_root`` plus
    the diff's net added lines, or the added-line count for a new file).
    ``added_lines`` is the diff's gross added content lines for this file --
    the quantity the rubric flags as "new code added to an over-cap file".
    """

    filename: str
    line_count: int
    cap: int
    added_lines: int


def _over_cap_file_findings(
    diff: str, repo_root: Path, cap: int
) -> tuple[OverCapFileFinding, ...]:
    """Detect files this diff adds code to that are over the repo size cap.

    Returns findings only for files that (a) have at least one added content
    line in the diff and (b) whose post-diff line count exceeds ``cap``. A
    ``cap`` of 0 disables the check (returns ``()``) -- mirrors
    ``turn_cap_large_file_threshold``'s 0-disables convention.

    File sizes are read from ``repo_root`` (the orchestrator's checkout) plus
    the diff's net added lines, mirroring ``_max_touched_file_line_count``; a
    new file's size is its added lines. Best-effort: a file that cannot be read
    is skipped, never raises -- the same posture as the static probe and
    ``check_operator_containment``.
    """
    if cap <= 0:
        return ()
    _total, files = _diff_file_summary(diff)
    findings: list[OverCapFileFinding] = []
    for name, added, deleted in files:
        if not name or added <= 0:
            continue
        # Unified diff ``+++`` paths are prefixed with ``b/``; strip it so the
        # path resolves under ``repo_root`` and the reported filename is the
        # repo-relative path (not the diff's ``b/``-prefixed form). A literal
        # ``b/...`` file would be vanishingly rare and only makes the lookup
        # miss (falling back to the added-line count), never reads the wrong
        # file -- mirrors ``_max_touched_file_line_count``'s stance.
        filename = name[2:] if name.startswith("b/") else name
        path = repo_root / filename
        if path.exists() and path.is_file():
            try:
                with path.open("r", encoding="utf-8", errors="replace") as handle:
                    base = sum(1 for _ in handle)
            except OSError:
                continue
            line_count = base + added - deleted
        else:
            # New file (not present at repo_root): its size is the added lines.
            line_count = added
        if line_count > cap:
            findings.append(OverCapFileFinding(filename, line_count, cap, added))
    return tuple(findings)


def render_over_cap_section(findings: tuple[OverCapFileFinding, ...] | None) -> str:
    """Render the ``$over_cap_section`` packet block (issue #1445).

    Returns ``""`` when ``findings`` is ``None`` (cap disabled -- the caller in
    ``review()`` passes ``None`` when ``file_size_cap_lines`` is 0), mirroring
    ``render_static_probe_section``'s disabled contract. When enabled, this
    ALWAYS renders visible text -- even with zero findings -- rather than ``"``
    for a clean pass, mirroring ``render_static_probe_section``'s never-silent
    contract: an advisory probe that goes silent on a clean run is
    indistinguishable, from the rendered packet alone, from one that never ran.
    """
    if findings is None:
        return ""
    if not findings:
        return "File-size cap: no over-cap additions in this diff.\n"
    lines = ["**Over-cap file additions (issue #1445):**"]
    for finding in findings:
        lines.append(
            f"- `{finding.filename}`: {finding.line_count} lines "
            f"(cap {finding.cap}), +{finding.added_lines} added -- "
            "REPORTABLE FINDING. Suggested remedy: extract the new code to a "
            "domain module (facade re-export block in the monolith, "
            "implementation in the module), matching the #1283-era extractions. "
            "Then run `python scripts/refresh_file_size_ratchet.py` and commit "
            "the resulting `file_size_ratchet_baseline/` entry tightening in this "
            "PR -- the script's default mode is lower-only (never raises a "
            "mark), so it is safe to run mid-PR (#1495)."
        )
    return "\n".join(lines) + "\n"
