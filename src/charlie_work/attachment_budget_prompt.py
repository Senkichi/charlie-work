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

from .attachment_contracts.review_delta import BudgetSection

# Issue #1460: the static dispatch-clause prose emitted by
# ``_build_attachment_budget_value`` when `.attachment-budgets.json` is
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
"""


def render_attachment_budget_section(section: BudgetSection | None) -> str:
    """Render the ``$attachment_budget_section`` packet block (issue #1460).

    Returns ``""`` when ``section`` is ``None`` (the review() cheap gate
    decided this PR neither touches `.attachment-budgets.json` nor a
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

    if section.head_unreadable:
        lines.append(
            "- NOTE: could not evaluate .attachment-budgets.json at PR head; "
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
