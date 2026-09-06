"""Rework-packet / PR-comment I/O delegates for ``OrchestratorApp``.

Track 2 Phase B leaf L02 batch 2 (issue #1653, parent #1633, umbrella #1582).
Method bodies moved verbatim from ``OrchestratorApp`` in ``charlie_work.workflow``;
the ``workflow_delegation`` installer re-attaches each ``def`` onto the class.
"""

from __future__ import annotations

import json

import charlie_work.workflow as _wf
from charlie_work.attachment_contracts import hook_entry as attachment_hook_entry
from charlie_work.attachment_contracts.model import AdvisoryRecord


def _read_advisories_from_pr_comment(self, pr_number: int) -> tuple[AdvisoryRecord, ...] | None:
    """Read worker-published advisories from the PR-comment channel (#1466).

    Scans the PR's issue-level comments (a PR is also an issue, so its
    top-level comments live at ``repos/{owner}/{repo}/issues/<n>/comments``)
    for the most recent one whose body starts with
    ``ADVISORY_COMMENT_MARKER`` and parses it via
    ``attachment_hook_entry.parse_advisories_comment``.

    Returns ``None`` when no marker comment is present (the caller falls
    back to the local advisories log), or a tuple (possibly ``()``) when
    one is -- a present marker with no parseable records is still a
    present channel, so the caller does NOT fall back and the review
    packet renders an empty redirects-not-taken list rather than the
    "log not available" NOTE.

    The most-recent marker comment wins: a worker re-posts the comment on
    each push, so the latest one reflects the current head's advisories
    and any older marker comment is stale. ``_gh_api_list`` returns
    comments in chronological order (GitHub's default for that endpoint),
    so the last match in the list is the newest.

    Fail-soft: any GitHub API error (``_gh_api_list`` swallows them and
    returns ``[]``) yields ``None`` -- a transient API failure degrades to
    the local-log fallback rather than blocking packet generation, since
    this section is advisory-only.
    """
    owner_repo = "{owner}/{repo}"
    comments = _wf._gh_api_list(self.gh, f"repos/{owner_repo}/issues/{pr_number}/comments")
    marker_body: str | None = None
    for item in comments:
        body = item.get("body")
        if isinstance(body, str) and body.lstrip().startswith(
            attachment_hook_entry.ADVISORY_COMMENT_MARKER
        ):
            marker_body = body
    if marker_body is None:
        return None
    parsed = attachment_hook_entry.parse_advisories_comment(marker_body)
    # ``parse_advisories_comment`` returns ``None`` only for a body that
    # does not start with the marker -- which the loop above already
    # guaranteed -- so this is always a tuple here. Guard anyway so a
    # future change to the parser's contract cannot make this method
    # silently return ``None`` for a present-but-malformed marker comment
    # (which would wrongly trigger the local-log fallback).
    return parsed if parsed is not None else ()


def _read_packet_head_oid(self, pr_number: int) -> str | None:
    """Return the ``headRefOid`` stored in the existing review packet for
    ``pr_number``, or ``None`` if no packet exists or it cannot be read.

    Used by ``loop()`` to detect same-head PRs whose review packet is already
    current, so repeated supervised passes don't regenerate the packet or
    re-fire ``review_started`` label transitions while the operator is still
    reading the packet.
    """
    pr_json_path = self.paths.prs / f"pr-{pr_number}" / "pr.json"
    if not pr_json_path.exists():
        return None
    try:
        with pr_json_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    value = data.get("headRefOid")
    return str(value) if value is not None else None


def _read_packet_turn_cap_multiplier(self, pr_number: int) -> int:
    """Return the structure-aware turn-cap multiplier stamped into the
    review packet for ``pr_number`` (issue #1439), or 1 if no packet
    exists, the packet predates the field, or it cannot be read.

    A missing/old packet yielding 1 preserves the pre-#1439 flat
    ``review_max_turns`` budget -- the base cap, no structure bonus.
    """
    pr_json_path = self.paths.prs / f"pr-{pr_number}" / "pr.json"
    if not pr_json_path.exists():
        return 1
    try:
        with pr_json_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return 1
    if not isinstance(data, dict):
        return 1
    value = data.get("review_turn_cap_structure_multiplier", 1)
    if not isinstance(value, int) or isinstance(value, bool):
        return 1
    return max(value, 1)


def _read_packet_template_sha(self, pr_number: int) -> str | None:
    """Return the ``prompt_template_sha`` stamped in the existing review
    packet for ``pr_number``, or ``None`` if no packet exists, it cannot
    be read, or it predates issue #592 (when the field started being
    stamped).

    A ``None`` result is a legacy packet, not a staleness signal: callers
    conservatively treat a missing digest as "current" so a one-time
    fleet-wide burst of regenerations is not forced on upgrade. Packets
    rendered after this fix always carry the digest, so every *future*
    template change reaches static-head PRs.
    """
    pr_json_path = self.paths.prs / f"pr-{pr_number}" / "pr.json"
    if not pr_json_path.exists():
        return None
    try:
        with pr_json_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    value = data.get("prompt_template_sha")
    return str(value) if value is not None else None


def _packet_template_current(self, pr_number: int) -> bool:
    """True if the packet's stamped template digest matches the current
    review template.

    A packet that predates issue #592 (no ``prompt_template_sha`` field)
    is treated as current so the upgrade does not force a one-time
    fleet-wide regeneration burst; packets rendered after this fix always
    carry the digest, so future template edits are caught.
    """
    packet_template_sha = self._read_packet_template_sha(pr_number)
    if packet_template_sha is None:
        return True
    return packet_template_sha == self._review_template_sha()


def _read_packet_diff(self, pr_number: int) -> str | None:
    """Return the diff text stored in the existing review packet for
    ``pr_number``, or ``None`` if no packet exists or it cannot be read.

    Mirrors ``_read_packet_head_oid``: keeps ``reviewed_patch_id`` derived
    from the diff the reviewer actually saw rather than a live re-fetch.
    """
    diff_path = self.paths.prs / f"pr-{pr_number}" / "diff.patch"
    if not diff_path.exists():
        return None
    try:
        return diff_path.read_text(encoding="utf-8")
    except OSError:
        return None


def _comment_pr(self, pr_number: int, summary: str) -> None:
    """Post a comment on ``pr_number``, stamped with the orchestrator's marker.

    The marker is what lets #950's external-findings ingestion recognise this
    comment as its own output rather than a human finding -- author identity
    cannot make that distinction, because the orchestrator posts under a user
    token. Stamping happens here, at the single point every orchestrator
    comment passes through, so no call site can forget it.
    """
    pr_dir = self.paths.prs / f"pr-{pr_number}"
    body_path = pr_dir / "review-comment.md"
    body_path.write_text(f"{_wf.ORCHESTRATOR_COMMENT_MARKER}\n{summary}", encoding="utf-8")
    self.gh.pr_comment(pr_number, body_path)
