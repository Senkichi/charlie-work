"""Single reader for a PR's review decision (issue #1362 Stage 1).

A PR's review decision has historically lived in three places that can
disagree: ``state.json``'s ``prs[N].decision`` field, a flat
``<state_dir>/prs/pr-N/review-decision.json`` file, and per-round archives
under ``rounds/round-K/review-decision.json`` (issue #1268/W11,
issue #1270/W13). Every control-flow *read* of "what did the reviewer say"
must go through :func:`review_decision` so staleness/missing/fail-safe
semantics are enforced in exactly one place instead of re-derived at each
call site.

Stage 1 (this module) introduces the reader only. ``state.json``'s decision
fields are still WRITTEN everywhere they are today -- writer unification is
a later stage and is out of scope here. Nothing in ``src/`` should read
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
