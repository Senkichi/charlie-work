"""Single reader for a PR's review decision (issue #1362 Stage 1).

A PR's review decision has historically lived in three places that can
disagree: ``state.json``'s ``prs[N].decision`` field, a flat
``<state_dir>/prs/pr-N/review-decision.json`` file, and per-round archives
under ``rounds/round-K/review-decision.json`` (issue #1268/W11,
issue #1270/W13). Every control-flow *read* of "what did the reviewer say"
must go through :func:`review_decision` so staleness/missing/fail-safe
semantics are enforced in exactly one place instead of re-derived at each
call site.

Stage 1 (this module) introduced the reader only. Stage 2 adds
:func:`record_decision`, the single writer: every direct
``review-decision.json`` writer in ``workflow.py`` (``record_review``, the
packet-build placeholder, ``merge_authorize``, and the carry-forward path in
``_update_approval_head``) is converted to call it instead of writing the
file itself. ``state.json``'s decision fields are still WRITTEN everywhere
they are today by those same call sites -- ``record_decision`` does not
touch ``state.json``; folding the two stores together is Stage 3
(state-as-cache) and is out of scope here. Nothing in ``src/`` should read
those fields for control flow after Stage 1 lands; this module is the
replacement.

Fail-safe contract: a head mismatch or unparseable/corrupt file must never
be reported as an approval. ``ReviewDecision.stale`` and
``ReviewDecision.missing`` exist precisely so a caller doing
``decision == "approved" and not stale`` can't be fooled by a torn file or
a verdict pinned to an old head.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ReviewDecision:
    """The resolved review decision for one PR, from a single read.

    ``decision`` is ``None`` when no decision could be established at all
    (``missing=True``) -- never a string that could be mistaken for a real
    verdict. ``source_round`` is ``None`` when the decision came from the
    flat file (or nowhere); it is the integer round number when the flat
    file was missing/corrupt and a round archive supplied the fallback.
    """

    decision: str | None
    reviewed_head_sha: str | None
    recorded_at: str | None
    source_round: int | None
    stale: bool
    missing: bool


def _read_review_decision_payload(path: Path) -> dict[str, Any] | None:
    """Best-effort read of one ``review-decision.json``-shaped file.

    Returns ``None`` for a missing file, an OS error, invalid JSON (a torn
    write), or JSON that does not decode to a dict -- callers treat all of
    these as "nothing usable here" and fall through, never as a decision.

    A private copy of ``rework_prompts._read_review_decision``'s shape
    rather than an import: this module must not import from
    ``rework_prompts`` (that module imports the round-fallback logic FROM
    here -- see :func:`_round_history_entries` -- and a reverse import would
    be circular).
    """
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _existing_round_numbers(rounds_dir: Path) -> list[int]:
    """Return the round numbers archived under ``rounds_dir``.

    Mirrors ``rework_prompts._existing_round_numbers`` (private copy, same
    reasoning as :func:`_read_review_decision_payload` above: no import
    back into ``rework_prompts`` to avoid a cycle).
    """
    if not rounds_dir.is_dir():
        return []
    numbers: list[int] = []
    for entry in rounds_dir.iterdir():
        if entry.is_dir() and entry.name.startswith("round-"):
            suffix = entry.name[len("round-") :]
            if suffix.isdigit():
                numbers.append(int(suffix))
    return numbers


def _round_history_entries(
    rounds_dir: Path,
    fallback_decision: Mapping[str, Any] | None = None,
) -> list[tuple[int, dict[str, Any]]]:
    """Return every archived review round under ``rounds_dir``, oldest first.

    Hoisted verbatim from ``rework_prompts._round_history_entries`` (issue
    #1362 Stage 1) -- ``rework_prompts.py`` now imports this function rather
    than defining it, and ``workflow.py``'s existing facade import keeps
    working unchanged through that re-export chain.

    Issue #1270 (W13): reads exclusively from the ``rounds/round-K`` layout
    W11 built (issue #1268) -- never events.db, never
    ``request_changes_count``, per the binding decision on #1270. Each
    element is ``(round_number, decision_payload)``; the payload is read
    fresh from ``rounds/round-K/review-decision.json``, not derived from
    ``fallback_decision``, so a hand-edited round archive (an operator
    correction) is reflected exactly.

    ``fallback_decision`` exists for two cases, both gated on ``not entries``
    (i.e. checked AFTER attempting to read every archived round, not merely
    on ``rounds_dir`` having no subdirectories -- issue #1270 review round 1
    found the original ``not numbers`` gate left a PR's prior review
    silently invisible whenever a round directory existed but its decision
    file did not parse):

    * The transition window around the W11 deploy: a PR whose only recorded
      verdict predates W11 has a flat ``review-decision.json`` but no
      ``rounds/`` directory at all (``numbers`` is empty).
    * A round directory exists (``numbers`` is non-empty) but every
      ``round-K/review-decision.json`` under it is missing or fails to
      parse -- e.g. ``OrchestratorApp._write_json`` (workflow.py), the method
      ``record_review`` uses to archive each round, creates the round
      directory via ``mkdir`` strictly before its atomic ``tmp_path.replace()``,
      so a crash in that window leaves an empty, decision-file-less
      ``round-K/`` that ``_existing_round_numbers`` still counts.

    Without a fallback in either case, such a PR would silently lose its
    prior-round findings even though the caller's own round-2 gate
    (``is_round2_review`` in ``workflow.py``, unchanged by this issue)
    already says a prior verdict exists. The fallback is surfaced as a
    synthetic round 1 -- ``fallback_decision`` itself, not a disk read --
    and is consulted ONLY when every attempted read came back empty; once at
    least one round is actually read successfully, the archive is
    authoritative and the fallback is never consulted, matching "exclusively
    from that layout" for every PR reviewed since W11 shipped.
    """
    numbers = sorted(_existing_round_numbers(rounds_dir))
    entries: list[tuple[int, dict[str, Any]]] = []
    for number in numbers:
        payload = _read_review_decision_payload(
            rounds_dir / f"round-{number}" / "review-decision.json"
        )
        if payload is not None:
            entries.append((number, payload))
    if not entries:
        if fallback_decision is not None:
            return [(1, dict(fallback_decision))]
        return []
    return entries


def _write_json_atomic(path: Path, value: Any) -> None:
    """Atomic temp-file + ``replace()`` write, matching the repo's canonical
    pattern (``OrchestratorApp._write_json`` in ``workflow.py``,
    ``adapters._write_json``, ``devin_shell._write_json`` -- see CLAUDE.md's
    "All JSON state writes are atomic" invariant).

    Hoisted here (issue #1362 Stage 2) as the one atomic-write primitive
    :func:`record_decision` uses for both the round-file and flat-file
    writes, rather than inventing a second shape or requiring an
    ``OrchestratorApp`` instance just to reach the static method.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


# Issue #1268 (W11): the field set that identifies a review round. Two writes
# that agree on all four are the same round (a retry); any difference --
# including on an unchanged head -- is a distinct verdict and must never
# overwrite a prior round's archived text. Verbatim copy of
# ``rework_prompts._ROUND_COMPARE_KEYS`` -- see :func:`_next_round_number`
# below for why this is a private mirror rather than an import.
_ROUND_COMPARE_KEYS = ("decision", "summary", "required_changes", "reviewed_head_sha")


def _next_round_number(rounds_dir: Path, decision_payload: Mapping[str, Any]) -> int:
    """Return the round-K under which ``decision_payload`` should be archived.

    Private mirror of ``rework_prompts._next_round_number`` (same retry/
    distinct-verdict reasoning documented there in full) -- not an import,
    for the same reason ``_read_review_decision_payload`` and
    ``_existing_round_numbers`` above are private copies rather than
    imports: ``rework_prompts.py`` already imports ``_round_history_entries``
    FROM this module, so importing ``_next_round_number`` the other way
    would be circular. :func:`record_decision` must reuse this exact
    dedup logic (never fork a second numbering scheme), so any future change
    to ``rework_prompts._next_round_number`` must be mirrored here too.
    """
    highest = max(_existing_round_numbers(rounds_dir), default=0)
    if highest == 0:
        return 1
    prior_decision = _read_review_decision_payload(
        rounds_dir / f"round-{highest}" / "review-decision.json"
    )
    is_retry = prior_decision is not None and all(
        prior_decision.get(key) == decision_payload.get(key) for key in _ROUND_COMPARE_KEYS
    )
    return highest if is_retry else highest + 1


def record_decision(
    pr_dir: Path,
    verdict_payload: Mapping[str, Any],
    head_sha: str | None,
    *,
    archive_round: bool = True,
) -> ReviewDecision:
    """Single writer for a PR's review decision (issue #1362 Stage 2).

    Writes, in this order:

    1. The per-round archive, ``rounds/round-K/review-decision.json`` -- K
       derived from :func:`_next_round_number` against whatever is already
       archived on disk, exactly as ``record_review`` derives it today.
       Skipped entirely when ``archive_round=False`` (see below).
    2. The flat ``review-decision.json``, atomically.

    ``archive_round`` (default ``True``) exists for callers whose write is a
    mechanical patch onto an existing verdict rather than a new reviewer
    round -- currently only the carry-forward path in
    ``_update_approval_head``. That write changes ``reviewed_head_sha`` (one
    of :data:`_ROUND_COMPARE_KEYS`) while leaving ``decision``/``summary``/
    ``required_changes`` untouched, so routing it through the default
    round-mint logic would archive a content-free "round" whose only
    difference from the prior one is the head it is pinned to --
    ``_round_history_entries``/``prior_review_section`` would then render an
    extra, duplicate-looking entry in the rendered prior-review history for
    every carry-forward, even though no reviewer produced a new verdict.
    ``archive_round=False`` skips :func:`_next_round_number` and the round
    write entirely, performing only the flat-file write -- the carry-forward
    still updates the single durable record, it just never masquerades as a
    new round. ``merge_authorize``'s override patch does not need this flag:
    it only adds ``authorized_override``, which is not one of
    ``_ROUND_COMPARE_KEYS``, so it already dedupes as a retry onto the
    existing highest round (see :func:`_next_round_number`) with the default.

    This is round-file-*then*-flat -- the inverse of the historical
    ``record_review`` ordering, which wrote flat first and archived the
    round second. Round-first means a crash (or any exception) between the
    two writes leaves the round archive as the durable record and the flat
    file either absent (a PR's first-ever verdict) or still holding the
    prior round's content; either way, :func:`review_decision`'s
    flat-then-round-fallback read order still resolves to a real verdict --
    the crash never silently loses the verdict this call was recording. See
    ``tests/test_review_decision.py`` for the regression test (issue #1362
    AC4). Neither write is skipped or reordered on any path; if the round
    write raises, the flat write is never attempted and the exception
    propagates to the caller (errors are not swallowed here -- callers that
    need errors-as-values wrap this call, per the adapters' convention).

    ``head_sha`` stamps ``verdict_payload["reviewed_head_sha"]`` (overwriting
    any value already present in the payload) before either write, so every
    writer -- including a caller recording only a head-stamped placeholder,
    e.g. ``{"decision": "pending"}`` -- goes through the same single point
    of truth for "what head was this decision recorded against", making a
    pending-for-a-dead-head verdict detectable downstream. Pass ``None`` to
    leave the payload's own ``reviewed_head_sha`` (if any) untouched -- for
    a caller that has already resolved and embedded the correct value and
    has no independent head argument to assert.

    Every other key in ``verdict_payload`` passes through completely
    unchanged -- in particular ``verdict_provenance`` (issue #1265: every
    verdict must record where it came from). This function does not
    interpret, validate, rename, or drop any field it does not itself
    write; ``verdict_provenance`` enforcement (e.g. against
    ``VERDICT_PROVENANCE_VALUES``) remains the caller's responsibility, as
    it is today at the ``record_review`` call boundary.

    Returns the :class:`ReviewDecision` now readable for ``pr_dir`` --
    obtained by calling :func:`review_decision` itself, against
    ``head_sha``, immediately after both writes complete, so the writer and
    the reader can never disagree about what a freshly-recorded decision
    looks like.
    """
    payload = dict(verdict_payload)
    if head_sha is not None:
        payload["reviewed_head_sha"] = head_sha

    if archive_round:
        rounds_dir = pr_dir / "rounds"
        round_number = _next_round_number(rounds_dir, payload)
        round_path = rounds_dir / f"round-{round_number}" / "review-decision.json"
        _write_json_atomic(round_path, payload)

    flat_path = pr_dir / "review-decision.json"
    _write_json_atomic(flat_path, payload)

    return review_decision(pr_dir, pr_state=None, current_head_sha=head_sha)


def resolve_decision_payload(pr_dir: Path) -> dict[str, Any]:
    """Return the full recorded decision payload for ``pr_dir``, or ``{"decision": "missing"}``.

    Same read order as :func:`review_decision` (flat file first, falling
    back to the highest-numbered round archive) but returns the raw dict
    instead of the narrow :class:`ReviewDecision` -- for callers that need
    fields the dataclass deliberately does not carry (``required_changes``,
    ``summary``, ``escalated``, ``reviewed_patch_id``, ...), such as
    rendering a rework prompt's required-changes section.

    Hoisted out of ``OrchestratorApp._review_decision`` (issue #1362 Stage
    1) so both that method and any standalone caller (e.g.
    ``rework_prompts._render_rework_prompt``) share one resolution instead
    of ``rework_prompts.py`` re-deriving a flat-file-only read that silently
    dropped the round fallback ``review_decision()``/``_review_decision``
    both apply.
    """
    payload = _read_review_decision_payload(pr_dir / "review-decision.json")
    if payload is None:
        entries = _round_history_entries(pr_dir / "rounds")
        if entries:
            payload = entries[-1][1]
    if payload is None:
        return {"decision": "missing"}
    return payload


def review_decision(
    pr_dir: Path,
    pr_state: Mapping[str, Any] | None,
    current_head_sha: str | None,
) -> ReviewDecision:
    """Resolve the single review decision for a PR.

    Read order: the flat ``pr_dir/review-decision.json`` file first; when
    it is missing or unparseable (torn write), fall back to the
    highest-numbered ``pr_dir/rounds/round-K/review-decision.json``. Neither
    source ever raises -- a missing/corrupt file at either layer degrades to
    the next fallback, and exhausting all of them yields ``missing=True``.

    ``pr_state`` (``state.json``'s ``prs[str(pr_number)]`` entry, or
    ``None``/``{}`` when unavailable) is accepted for forward compatibility
    with later stages but is NOT consulted for the decision itself in Stage
    1 -- the file layer is authoritative for control flow, per the spec's
    file-first read order. It is reserved for Stage 2 (writer unification),
    where the two stores are made to agree; keeping the parameter now avoids
    re-threading every call site's signature twice.

    ``stale`` is True whenever the resolved decision's ``reviewed_head_sha``
    does not equal ``current_head_sha`` (including when it is absent from
    the payload) -- a head mismatch must never be reported as approved, so
    callers should always check ``not stale`` alongside ``decision ==
    "approved"``. ``stale`` is False when ``missing`` is True: there is no
    verdict to be stale, so the caller's ``decision == "approved"`` check
    already fails safe without help from this flag.
    """
    del pr_state  # reserved for Stage 2; see docstring.

    flat_path = pr_dir / "review-decision.json"
    payload = _read_review_decision_payload(flat_path)
    source_round: int | None = None

    if payload is None:
        rounds_dir = pr_dir / "rounds"
        entries = _round_history_entries(rounds_dir)
        if entries:
            source_round, payload = entries[-1]

    if payload is None:
        return ReviewDecision(
            decision=None,
            reviewed_head_sha=None,
            recorded_at=None,
            source_round=None,
            stale=False,
            missing=True,
        )

    decision_value = payload.get("decision")
    reviewed_head_sha = payload.get("reviewed_head_sha")
    recorded_at = payload.get("recorded_at")
    stale = reviewed_head_sha is None or reviewed_head_sha != current_head_sha

    return ReviewDecision(
        decision=decision_value if isinstance(decision_value, str) else None,
        reviewed_head_sha=reviewed_head_sha if isinstance(reviewed_head_sha, str) else None,
        recorded_at=recorded_at if isinstance(recorded_at, str) else None,
        source_round=source_round,
        stale=stale,
        missing=False,
    )
