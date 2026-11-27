"""The guard layer every Apply effect passes first (issue #3119).

Split out of ``apply.py``: the shared checks the per-type handlers in
``apply_handlers.py`` run before touching anything — charter repo
membership, the private-slug and outbound-body refusals, authored-issue
authority (ownership label, marker, open, not in an active/terminal
lifecycle state), the authored-issue body-hash cache, arming, and
exact-body comment dedupe.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..atomic_write import write_text_atomic
from ..github import GitHubError, label_names
from ..issue_contract_lint import render_blocker_section
from ..labels import apply_issue_labels
from ..outbound_body_guard import (
    OutboundBodyGuardError,
    check_outbound_write,
    refusal_summary,
    scan_outbound_text,
)
from ..private_slug_gate import count_slug_mentions_in_text
from ..ratchet_baseline import BaselineFormatError, load_set_baseline
from .apply_context import (
    _SLUG_BASELINE_DIRNAME,
    _SLUG_SETS_DIRNAME,
    _body_hash,
    _marker_pattern,
    _Run,
)
from .charter import ARMING_AUTO
from .items import create_item
from .paths import state_path, turn_dir
from .state import AuthoredIssue, read_state_snapshot, save_state, with_updates


def _repo_check(run: _Run, repo: str) -> str | None:
    if repo not in run.charter.repos:
        return f"repo {repo!r}: not in the charter ({list(run.charter.repos)})"
    return None


def _slug_baseline(repo_root: Path | None) -> frozenset[str] | None:
    """Listed slugs, or None when the repo carries no baseline (check skipped)."""
    if repo_root is None:
        return None
    directory = repo_root / _SLUG_BASELINE_DIRNAME / _SLUG_SETS_DIRNAME
    try:
        return load_set_baseline(directory)
    except BaselineFormatError:
        return None


def _slug_refusal(run: _Run, parts: tuple[tuple[str, str], ...]) -> str | None:
    """Refusal reason when any part names a baseline slug, else None."""
    slugs = _slug_baseline(run.repo_root)
    if not slugs:
        return None
    for part_name, text in parts:
        if text and count_slug_mentions_in_text(text, sorted(slugs)):
            return (
                f"{part_name} names a private slug from "
                f"{_SLUG_BASELINE_DIRNAME}/{_SLUG_SETS_DIRNAME}/ -- private repo "
                "names never reach a public repo"
            )
    return None


def _outbound_refusal(
    run: _Run, surface: str, parts: tuple[tuple[str, str], ...], issue_number: int | None
) -> str | None:
    """Refusal reason when the secret guard matches, else None.

    On a live run the guarded chokepoint itself scans and emits its refusal
    event; on dry-run the same scan runs without the event. A guard-machinery
    failure (``OutboundBodyGuardError``) fails closed as a refusal — the
    write is the thing being protected, so it can never proceed unchecked.
    """
    try:
        if run.dry_run:
            matches = tuple(
                match
                for part_name, text in parts
                for match in scan_outbound_text(text, part=part_name)
            )
        else:
            matches = check_outbound_write(
                surface=surface,
                parts=parts,
                repo_root=run.repo_root,
                state_dir=run.state_dir,
                issue_number=issue_number,
            )
    except OutboundBodyGuardError as exc:
        return f"outbound guard failure (refusing closed): {exc}"
    if matches:
        return refusal_summary(surface, matches)
    return None


def _live_authored(run: _Run, number: int) -> tuple[dict[str, Any], str | None]:
    """Live-fetch an issue and check authored-issue authority.

    Returns ``(issue, None)`` when the issue carries the ownership label, the
    foreman's marker, no active or terminal label, and is still open — or
    ``({}, reason)`` naming the first authority miss.
    """
    issue = run.gh.issue_view(number)
    if not issue:
        return {}, f"issue #{number}: not found in {run.charter.repos[0]}"
    names = label_names(issue)
    if run.owner_label not in names:
        return {}, f"issue #{number}: lacks ownership label {run.owner_label!r}"
    if str(issue.get("state", "")).upper() != "OPEN":
        return {}, f"issue #{number}: not open (state {issue.get('state')!r})"
    if not _marker_pattern(run.charter.name).search(issue.get("body") or ""):
        return {}, f"issue #{number}: tracked, not authored (no foreman marker)"
    blocked = names & (run.labels.active | run.labels.terminal)
    if blocked:
        return {}, f"issue #{number}: in active/terminal state {sorted(blocked)}"
    return issue, None


def _check_outbound_and_slug(
    run: _Run, surface: str, parts: tuple[tuple[str, str], ...], issue_number: int | None
) -> str | None:
    refusal = _slug_refusal(run, parts)
    if refusal is not None:
        return refusal
    return _outbound_refusal(run, surface, parts, issue_number)


def _record_authored(run: _Run, number: int, body: str) -> None:
    """Cache the issue's body hash in ``state.json`` (rebuildable; best-effort)."""
    state = read_state_snapshot(run.foreman_root)
    issues = dict(state.issues)
    issues[number] = AuthoredIssue(number=number, body_hash=_body_hash(body))
    # write-gate-exempt(issue=3119): name-matched only — foreman.state.save_state is the foreman's own store, not the orchestrator primitive; no write_gate receiver here.
    save_state(run.foreman_root, with_updates(state, issues=issues))


def _blocker_tail(blockers: frozenset[int], marker: str) -> str:
    """The ``## Blocked by`` section with the hidden marker on its last line.

    The marker must end the body (design §8.3), but a bare trailing line
    under the section heading is scanned as a section *item* — neither a
    ``#N`` ref, a ``None`` sentinel, nor an inline declaration — and parks
    the issue as unreadable (``_scan_blocker_sections``). Riding the last
    line instead keeps every parse: ``Blocked by #N <!-- marker -->`` reads
    as the inline declaration the line already is, and ``None <!-- ... -->``
    still prefix-matches the sentinel.
    """
    return render_blocker_section(blockers).rstrip() + " " + marker


def _arm_or_raise(run: _Run, number: int, key: str, title: str) -> str:
    """The arming step: add Ready, or leave unarmed behind a decision item.

    Runs *after* the issue exists, per the design's "separate, final step":
    Ready only when the charter says ``arming: auto``, the foreman is not
    ``degraded``, and the Turn is not signal-marked. Every other path leaves
    the issue unarmed and raises a decision item whose ``command`` arms it.
    Returns the detail fragment for the applied outcome.
    """
    state = read_state_snapshot(run.foreman_root)
    why_unarmed: str | None = None
    if run.signal_marked:
        why_unarmed = "turn is signal-marked"
    elif state.forced_arming is not None:
        # Issue #3123: a failed §16 verdict forces propose arming until the
        # operator answers the retire/retune item — whatever the charter says.
        why_unarmed = "verdict forced arming: propose"
    elif run.charter.arming != ARMING_AUTO:
        why_unarmed = f"arming: {run.charter.arming}"
    elif state.state == "degraded":
        why_unarmed = "foreman is degraded"
    if why_unarmed is None:
        if run.dry_run:
            return "armed"
        # write-gate-exempt(issue=3119): the foreman Apply engine has no write_gate receiver; apply_issue_labels is itself the issue-#2226 label-write seam, and the emit lands on the foreman's own events.db.
        result = apply_issue_labels(
            run.gh,
            run.labels,
            number,
            add=(run.labels.ready,),
            state_path=state_path(run.foreman_root),
            repo=run.charter.repos[0] if run.charter.repos else None,
            cause="foreman_arm",
        )
        if result.ok:
            return "armed"
        why_unarmed = "ready label add failed"

    command = f'gh issue edit {number} --add-label "{run.labels.ready}"'
    item_id = f"{run.turn_id}:{key}:arm"
    if run.dry_run:
        return f"unarmed ({why_unarmed}); would raise arming item {item_id}"
    item = create_item(
        run.foreman_root,
        kind="decision",
        question=f"Arm issue #{number} ({title}) for dispatch?",
        options=(f"Arm it — add {run.labels.ready}", "Leave it unarmed"),
        recommended=0,
        command=command,
        item_id=item_id,
    )
    if item.ok:
        return f"unarmed ({why_unarmed}); arming item {item.value.id}"
    return f"unarmed ({why_unarmed}); arming item failed: {item.error}"


def _comment_body_file(run: _Run, index: int) -> Path:
    """The per-action comment body file ``issue_comment`` reads from."""
    return turn_dir(run.foreman_root, run.turn_id) / f"comment-{index}.md"


def _post_comment(run: _Run, number: int, index: int, text: str) -> str | None:
    """Append *text* as a comment, deduped on exact body; error string or None."""
    if _has_comment(run, number, text):
        return None
    path = _comment_body_file(run, index)
    write_text_atomic(path, text)
    try:
        run.gh.issue_comment(number, path)
    except GitHubError as exc:
        return f"comment on #{number} failed: {exc}"
    return None


def _comment_matches(stored: str, text: str) -> bool:
    """True when *stored* is the comment ``text`` — or carries it as the body.

    The local backend stores one comments-block chunk per entry: the
    ``**charlie-work** (<ts>)`` author line, a blank line, then the body —
    so an exact-body dedupe must accept the stored form ``author + "\\n\\n"
    + text``. On the ``gh`` backend ``comments[].body`` is the text alone.
    """
    stored = stored.strip().replace("\r\n", "\n")
    text = text.strip().replace("\r\n", "\n")
    return stored == text or stored.endswith("\n\n" + text)


def _has_comment(run: _Run, number: int, text: str) -> bool:
    """True when the exact comment body already sits on the issue."""
    live = run.gh.issue_view(number)
    existing = [str(c.get("body", "")) for c in live.get("comments", []) if isinstance(c, dict)]
    return any(_comment_matches(body, text) for body in existing)
