"""Baseline: freeze-on-adopt, ratchet-down, bump validation, tamper guard.

The committed baseline lives in the per-entry ``.attachment-budgets/``
directory (issue #1839 -- the single ``.attachment-budgets.json`` document
was a shared append point; see ``baseline_dir.py`` for the on-disk layout).
This module owns the layout-independent DOCUMENT contract: the dict shape
(version/generated_by/generated_at/floor/entries[/kind_stats]), generation,
comparison, ratcheting, bump validation, and the tamper guards. An entry may
carry ``"pinned": true`` (issue #1620): an operator-authored sub-saturation
contract on a de-godded point -- enforced below the fence, never dropped by
the ratchet (see ``BaselineEntry`` and ``compare()``). ``dumps``/
``loads``/``dump``/``load`` remain the legacy single-file codec -- used for
pre-#1839 checkouts via ``baseline_dir``'s layout dispatch, and for any
caller that needs a document serialized as one JSON text.

Serialization is fully deterministic: sorted entries, indent=1, sorted keys,
trailing newline — so two runs against the same scan produce byte-identical
output.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from charlie_work.attachment_contracts.model import (
    BaselineEntry,
    Bump,
    Finding,
    Kind,
    KindStats,
    SaturationVerdict,
)

SCHEMA_VERSION = 1
BASELINE_FILENAME = ".attachment-budgets.json"
# Issue #1839: the committed baseline's current on-disk layout. The constant
# lives here (next to the legacy filename it replaces) so both the document
# layer and ``baseline_dir`` can name it without an import cycle.
BASELINE_DIRNAME = ".attachment-budgets"
# Top-level key under which frozen per-kind Tukey fences are persisted
# (issue #1614). Optional: baselines written before #1614 lack it and load
# fine (callers fall back to live recomputation), so SCHEMA_VERSION stays 1.
KIND_STATS_KEY = "kind_stats"


class TamperError(ValueError):
    """A baseline file failed structural or referential validation."""


def _bump_to_dict(bump: Bump) -> dict[str, object]:
    return {"to": bump.to, "reason": bump.reason, "actor": bump.actor, "ack": bump.ack}


def _bump_from_dict(raw: dict[str, object]) -> Bump:
    try:
        return Bump(
            to=int(raw["to"]),  # type: ignore[arg-type]
            reason=str(raw["reason"]),
            actor=str(raw["actor"]),  # type: ignore[arg-type]
            ack=str(raw.get("ack", "")),
        )
    except (KeyError, ValueError, TypeError) as exc:
        # Finding #12: a missing or non-numeric field must surface as a
        # structured TamperError -- never as a bare KeyError/ValueError that
        # escapes check_tree's `except TamperError` and crashes the CI step,
        # defeating the report-only "can never fail the job" contract for
        # exactly the tamper vector it exists to catch.
        raise TamperError(f"malformed bump entry: {exc}") from exc


def _entry_to_dict(entry: BaselineEntry) -> dict[str, object]:
    data: dict[str, object] = {
        "kind": entry.kind,
        "identity": entry.identity,
        "file": entry.file,
        "member_count": entry.member_count,
        "boundary": entry.boundary,
        "bumps": [_bump_to_dict(b) for b in entry.bumps],
    }
    # Issue #1620: emitted only when True -- every pre-#1620 entry file lacks
    # the key, and a routine --ratchet must not churn all of their bytes just
    # to write ``"pinned": false`` into each one.
    if entry.pinned:
        data["pinned"] = True
    return data


def _entry_from_dict(raw: dict[str, object]) -> BaselineEntry:
    bumps_raw = raw.get("bumps", [])
    if not isinstance(bumps_raw, list):
        raise TamperError(f"entries[].bumps must be a list, got {type(bumps_raw)!r}")
    pinned = raw.get("pinned", False)
    if not isinstance(pinned, bool):
        raise TamperError(f"entries[].pinned must be a bool, got {type(pinned)!r}")
    try:
        return BaselineEntry(
            kind=str(raw["kind"]),  # type: ignore[arg-type]
            identity=str(raw["identity"]),
            file=str(raw["file"]),
            member_count=int(raw["member_count"]),  # type: ignore[arg-type]
            boundary=float(raw["boundary"]),  # type: ignore[arg-type]
            bumps=tuple(_bump_from_dict(b) for b in bumps_raw),
            pinned=pinned,
        )
    except (KeyError, ValueError, TypeError) as exc:
        # Finding #12: same rationale as _bump_from_dict -- a missing key or a
        # non-numeric member_count/boundary must become a structured Finding
        # via TamperError, not an uncaught crash that bypasses --report-only.
        raise TamperError(f"malformed baseline entry: {exc}") from exc


def _entry_sort_key(entry: BaselineEntry) -> tuple[str, str, str]:
    return (entry.kind, entry.file, entry.identity)


def _kind_stats_to_dict(stats: KindStats) -> dict[str, object]:
    return {
        "q3": stats.q3,
        "iqr": stats.iqr,
        "boundary": stats.boundary,
        "population": stats.population,
    }


def _kind_stats_from_dict(kind: str, raw: dict[str, object]) -> KindStats:
    try:
        return KindStats(
            kind=kind,  # type: ignore[arg-type]
            q3=float(raw["q3"]),  # type: ignore[arg-type]
            iqr=float(raw["iqr"]),  # type: ignore[arg-type]
            boundary=float(raw["boundary"]),  # type: ignore[arg-type]
            population=int(raw["population"]),  # type: ignore[arg-type]
        )
    except (KeyError, ValueError, TypeError) as exc:
        # Same rationale as _entry_from_dict / _bump_from_dict (finding #12):
        # a missing or non-numeric kind_stats field must surface as a
        # structured TamperError, never as a bare KeyError/ValueError that
        # escapes check_tree's `except TamperError` and crashes the CI step.
        raise TamperError(f"malformed kind_stats entry for {kind!r}: {exc}") from exc


def kind_stats_of(document: dict[str, object]) -> dict[Kind, KindStats]:
    """Return the frozen per-kind fence stats persisted in ``document``.

    Empty dict when the document predates issue #1614 (no ``kind_stats`` key)
    -- callers treat absence as "no frozen fence, fall back to live
    recomputation". Raises ``TamperError`` if the key is present but
    malformed (non-object value, missing/ non-numeric fields).
    """
    raw = document.get(KIND_STATS_KEY)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise TamperError(f"baseline {KIND_STATS_KEY!r} must be an object, got {type(raw)!r}")
    result: dict[Kind, KindStats] = {}
    for kind, stats_raw in raw.items():
        if not isinstance(stats_raw, dict):
            raise TamperError(f"kind_stats[{kind!r}] must be an object, got {type(stats_raw)!r}")
        result[kind] = _kind_stats_from_dict(kind, stats_raw)  # type: ignore[index]
    return result


def _kind_stats_from_verdicts(
    verdicts: tuple[SaturationVerdict, ...],
) -> dict[Kind, KindStats]:
    """Derive frozen per-kind stats from a live ``saturate_all`` result.

    Every verdict of a kind carries that kind's q3/iqr/boundary/population
    (they are identical across a kind's verdicts by construction), so the
    first verdict seen per kind is authoritative. Kinds with no eligible
    points produce no verdicts and therefore no entry -- nothing of that
    kind can saturate anyway.
    """
    stats: dict[Kind, KindStats] = {}
    for v in verdicts:
        if v.point.kind in stats:
            continue
        stats[v.point.kind] = KindStats(
            kind=v.point.kind,
            q3=v.q3,
            iqr=v.iqr,
            boundary=v.boundary,
            population=v.population,
        )
    return stats


def generate(
    verdicts: tuple[SaturationVerdict, ...],
    *,
    generated_by: str,
    generated_at: str,
    floor: int,
) -> dict[str, object]:
    """Build the baseline document (as a plain dict, ready for dump()) from verdicts.

    Only saturated points are entered; freshly generated entries carry no bumps.
    The per-kind Tukey fence statistics (issue #1614) are persisted under
    ``kind_stats`` so later ``check_tree`` / ``--ratchet`` runs saturate
    against the frozen fence instead of recomputing it live.
    """
    entries = [
        BaselineEntry(
            kind=v.point.kind,
            identity=v.point.identity,
            file=v.point.file,
            member_count=v.point.member_count,
            boundary=v.boundary,
        )
        for v in verdicts
        if v.saturated
    ]
    entries.sort(key=_entry_sort_key)
    kind_stats = _kind_stats_from_verdicts(verdicts)
    return {
        "version": SCHEMA_VERSION,
        "generated_by": generated_by,
        "generated_at": generated_at,
        "floor": floor,
        "entries": [_entry_to_dict(e) for e in entries],
        KIND_STATS_KEY: {
            kind: _kind_stats_to_dict(stats) for kind, stats in sorted(kind_stats.items())
        },
    }


def dumps(document: dict[str, object]) -> str:
    """Deterministic JSON: sorted entries, indent=1, sorted keys, trailing newline."""
    entries_raw = document.get("entries", [])
    ordered_entries = sorted(
        entries_raw,  # type: ignore[arg-type]
        key=lambda e: (e["kind"], e["file"], e["identity"]),
    )
    normalized: dict[str, object] = {**document, "entries": ordered_entries}
    # Normalize kind_stats key order for byte-stable output across runs when
    # present. Only touch the key if the document already carries it -- a
    # pre-#1614 baseline has no kind_stats and must not gain a ``null`` entry
    # on a round-trip dump.
    kind_stats_raw = document.get(KIND_STATS_KEY)
    if isinstance(kind_stats_raw, dict):
        normalized[KIND_STATS_KEY] = {k: kind_stats_raw[k] for k in sorted(kind_stats_raw)}
    return json.dumps(normalized, indent=1, sort_keys=True) + "\n"


def with_kind_stats(
    document: dict[str, object],
    verdicts: tuple[SaturationVerdict, ...],
    *,
    generated_at: str,
) -> dict[str, object]:
    """Return a copy of ``document`` with ``kind_stats`` recomputed from ``verdicts``.

    The ``--refreeze`` path (issue #1614): ratchet the entries against a
    LIVE fence (the caller passes live ``saturate_all`` verdicts to
    ``compare``), then overwrite the frozen per-kind stats with the
    recomputed values via this helper. ``generated_at`` is re-stamped for
    the same reason ``generate()`` stamps it: recomputing the frozen fence
    IS a generation event, and ``check_ratchet_tamper`` requires BOTH the
    re-stamp AND statistics equal to a live recompute of the checked tree
    before it treats a ``kind_stats`` change as sanctioned -- a refreeze
    that kept the old ``generated_at`` would read as a ratchet-to-ratchet
    boundary raise and be flagged as tamper even though its stats are
    honest. The input document is not mutated.
    """
    kind_stats = _kind_stats_from_verdicts(verdicts)
    return {
        **document,
        "generated_at": generated_at,
        KIND_STATS_KEY: {
            kind: _kind_stats_to_dict(stats) for kind, stats in sorted(kind_stats.items())
        },
    }


def dump(document: dict[str, object], path: Path) -> None:
    # Temp-file + replace(): the baseline is read by other processes (the
    # PreToolUse hook, review-packet builder) and must never be observed
    # half-written.
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(dumps(document), encoding="utf-8")
    tmp.replace(path)


def loads(text: str) -> dict[str, object]:
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        # Same finding-#12 contract as _entry_from_dict: a corrupt baseline
        # must surface as TamperError so check_tree's `except TamperError`
        # turns it into a structured Finding instead of crashing the
        # report-only CI step.
        raise TamperError(f"baseline is not valid JSON: {exc}") from exc
    return _validate_document(document)


def _validate_document(document: object) -> dict[str, object]:
    """Validate a decoded baseline document, returning it unchanged.

    Shared by ``loads`` (the single-file codec) and ``baseline_dir`` (the
    per-entry directory layout reassembles the same document dict before
    validation), so both on-disk forms enforce an identical schema.
    """
    if not isinstance(document, dict):
        raise TamperError("baseline root must be a JSON object")
    if document.get("version") != SCHEMA_VERSION:
        raise TamperError(f"unsupported baseline version: {document.get('version')!r}")
    entries_raw = document.get("entries")
    if not isinstance(entries_raw, list):
        raise TamperError("baseline 'entries' must be a list")
    # Validate every entry parses; surfaces structural tamper immediately.
    seen_keys: set[tuple[str, str, str]] = set()
    for raw in entries_raw:
        if not isinstance(raw, dict):
            raise TamperError(f"baseline entry must be an object, got {type(raw)!r}")
        entry = _entry_from_dict(raw)
        key = _entry_key(entry)
        if key in seen_keys:
            # Duplicate (kind, file, identity) collides in every identity-keyed
            # map this module builds (compare(), check_tamper(), ratchet
            # writeback) -- last-write-wins would silently DROP one entry the
            # moment `--ratchet` rewrites the file. Reject at load time
            # instead of losing a frozen entry to a silent overwrite.
            raise TamperError(
                f"duplicate baseline entry for kind={key[0]!r} file={key[1]!r} identity={key[2]!r}"
            )
        seen_keys.add(key)
    # Validate frozen per-kind stats if present (issue #1614). Absent is
    # valid -- baselines written before #1614 have no kind_stats and callers
    # fall back to live recomputation. Present-but-malformed is tamper.
    kind_stats_of(document)
    return document


def load(path: Path) -> dict[str, object]:
    return loads(path.read_text(encoding="utf-8"))


def entries_of(document: dict[str, object]) -> tuple[BaselineEntry, ...]:
    return tuple(_entry_from_dict(raw) for raw in document["entries"])  # type: ignore[union-attr,arg-type]


# Finding #10 (at minimum: validate ack SHAPE, not just non-emptiness).
# Accepts an http(s) URL, a bare or "owner/repo"-qualified issue/PR reference
# (`#123`, `owner/repo#123`), or an explicit "source:id" handle (dispatch-
# prompt id / human handle, e.g. "dispatch:abc123" or "handle:operator").
# A one-character junk ack like "ack: 'x'" (round-2 review's example) does
# not match any of these and is now rejected instead of merely non-empty.
#
# The alternatives are named groups (not just grouped) so that
# ``bump_ack_is_external`` (issue #1460) can single-source off THIS regex --
# telling a URL/#123 ack apart from the self-declared "source:id" handle form
# by which named group matched, instead of duplicating a second ack regex.
_ACK_SHAPE = re.compile(r"^(?:https?://\S+|[\w./-]*#\d+|(?P<source_id>[A-Za-z][\w.-]*:[\w./-]+))$")


def validate_bump(bump: Bump) -> str | None:
    """Return an error message if `bump` is invalid, else None.

    G4 (round-2 review finding #10): a shape-checked, non-empty `ack` is
    REQUIRED on every bump, regardless of `actor`. Round-1 hardened only the
    worker branch (a junk ack like "x" on a worker bump is rejected), but the
    discriminating vector was untouched: `actor` is a self-declared field
    unbound to execution context, so `Bump(actor="interactive", ack="")` --
    a worker that mislabels itself -- sailed through the ack requirement
    entirely. Requiring the same shape-checked ack for BOTH actors closes
    that mislabel vector outright: there is no longer anything to gain by
    claiming "interactive". The `actor` field itself is kept (the spec's
    baseline schema pins it, and it remains useful provenance/audit data,
    plus the real backstop for `actor` truthfulness is out-of-band review of
    the baseline diff, e.g. CODEOWNERS on `.attachment-budgets/` -- no
    comparison-only validator can bind a self-declared field to the actual
    execution context from the JSON alone); "interactive bumps self-ack"
    now means an interactive actor's own handle satisfies the same shape
    check (e.g. "handle:operator"), not that the ack requirement is waived.
    """
    if not bump.reason.strip():
        return "bump.reason must be non-empty"
    if bump.actor not in ("interactive", "worker"):
        return f"bump.actor must be 'interactive' or 'worker', got {bump.actor!r}"
    ack = bump.ack.strip()
    if not ack:
        return (
            "G4: bump requires a non-empty ack "
            "(issue URL / dispatch-prompt id / human handle) regardless of actor"
        )
    if not _ACK_SHAPE.match(ack):
        return (
            f"G4: bump ack {ack!r} does not look like an external reference "
            "(expected an issue URL, '#123' / 'owner/repo#123', or 'source:id')"
        )
    return None


def _entry_key(entry: BaselineEntry) -> tuple[str, str, str]:
    return (entry.kind, entry.file, entry.identity)


def _verdict_key(verdict: SaturationVerdict) -> tuple[str, str, str]:
    return (verdict.point.kind, verdict.point.file, verdict.point.identity)


def effective_ceiling(entry: BaselineEntry) -> int:
    """The highest member_count this entry currently permits without a Finding.

    That is the baselined member_count itself, or the highest bump.to if bumps
    exist (bumps only ever raise the ceiling; validity of each bump is checked
    separately by validate_bump at bump-authoring time).

    Public: issue #1460's review-packet section (``review_delta.py``) needs
    this to compute the saturation ceiling for the review-side G4 check
    without duplicating the max-of-member_count-and-bumps logic.
    """
    if not entry.bumps:
        return entry.member_count
    return max(entry.member_count, max(b.to for b in entry.bumps))


def bump_ack_is_external(bump: Bump) -> bool:
    """True only for a URL / ``#123`` / ``owner/repo#123`` ack form.

    False for the ``source:id`` handle form and for an empty ack. Single-
    sourced off ``_ACK_SHAPE`` (the same regex ``validate_bump`` uses): the
    ``source:id`` alternative is its only named group, so "external" is
    simply "matched, and not via that named group" -- no second ack regex.
    Used by the review packet (issue #1460) to flag a worker-authored bump
    whose ack is merely shape-valid (passes ``validate_bump``) but not an
    external, citable justification -- a worker may not author its own bump
    acknowledgement.
    """
    ack = bump.ack.strip()
    if not ack:
        return False
    match = _ACK_SHAPE.match(ack)
    if match is None:
        return False
    return match.group("source_id") is None


def new_bumps(
    base_document: dict[str, object],
    head_document: dict[str, object],
) -> tuple[tuple[BaselineEntry, Bump], ...]:
    """Bumps present in ``head_document`` but not in ``base_document``.

    Pure, no I/O: both documents are already-loaded baseline dicts (typically
    from ``loads()``). A bump is identified by
    ``(entry.identity, bump.to, bump.actor, bump.ack)`` -- keyed on the
    entry's identity rather than the full ``(kind, file, identity)`` tuple
    because a bump is meaningful per-identity regardless of a coincidental
    kind/file rename; two entries can only collide on identity if they are
    the same conceptual attachment point. Order is stable: entries sorted by
    ``_entry_sort_key``, bumps in each entry's own list order.
    """
    base_keys: set[tuple[str, int, str, str]] = set()
    for entry in entries_of(base_document):
        for bump in entry.bumps:
            base_keys.add((entry.identity, bump.to, bump.actor, bump.ack))

    result: list[tuple[BaselineEntry, Bump]] = []
    for entry in sorted(entries_of(head_document), key=_entry_sort_key):
        for bump in entry.bumps:
            key = (entry.identity, bump.to, bump.actor, bump.ack)
            if key not in base_keys:
                result.append((entry, bump))
    return tuple(result)


def compare(
    current: tuple[SaturationVerdict, ...],
    baseline_document: dict[str, object],
) -> tuple[list[Finding], dict[str, object]]:
    """Compare current saturation verdicts against the baseline.

    Returns (findings, ratcheted_document):
    - A currently-saturated point above its baseline's effective ceiling with
      no covering bump -> Finding(block).
    - A baselined point now saturated at or below its ceiling -> clean.
    - A baselined point no longer saturated, or saturated but with a strictly
      lower member_count than the baseline -> ratchet down (entry rewritten to
      the lower count; bumps for a point are dropped once ratcheted, since a
      bump raising an old, higher ceiling no longer applies to a lower one).
    - A currently-saturated point with no baseline entry at all -> Finding
      (block), because a committed baseline document already exists whenever
      `compare()` is called (both call sites -- check_tree and `baseline
      --ratchet` -- guard on the baseline file existing before calling this;
      the genuine "no baseline anywhere yet" freeze-on-adopt case never
      reaches compare() at all, it is handled entirely by `generate()`). A
      new AP appearing already saturated is therefore a NEW god-object, not
      an adoption artifact (round-2 review finding #13: this branch used to
      freeze it silently with no Finding, so a brand-new 50-method class
      entered completely unchecked). It IS still added to the ratcheted
      document so the baseline stays a complete snapshot of the tree's
      current state -- the enforcement is the Finding, not the omission.
    - A ``pinned`` entry (issue #1620 -- an operator-authored sub-saturation
      contract on a deliberately de-godded point) is enforced whether or not
      its point is saturated: growth past ``min(effective_ceiling,
      boundary)`` blocks exactly like over-ceiling saturation, a strict
      shrink ratchets the pin down like any other row, and the row is never
      dropped by omission on de-saturation. Capping the ceiling at the fence
      means a pin can only tighten a point's budget -- a hand-authored pin
      with a member_count above the boundary cannot act as a blanket
      exemption. `--ratchet` therefore never deletes a pinned row, and never
      creates one either: the operator authors it explicitly.
    The input document is never mutated; a new document dict is returned.
    """
    baseline_entries = {_entry_key(e): e for e in entries_of(baseline_document)}
    current_by_key = {_verdict_key(v): v for v in current if v.saturated}
    all_verdicts = {_verdict_key(v): v for v in current}

    findings: list[Finding] = []
    new_entries: list[BaselineEntry] = []

    for key, verdict in current_by_key.items():
        point = verdict.point
        baseline_entry = baseline_entries.get(key)
        if baseline_entry is None:
            # Finding #13: a baseline document already exists at every call
            # site of compare() (see docstring) -- this is a brand-new AP
            # appearing already saturated, not adoption. Block it the same
            # way growth past an existing ceiling is blocked, and still
            # snapshot it into the ratcheted document.
            findings.append(
                Finding(
                    severity="block",
                    file=point.file,
                    identity=point.identity,
                    message=(
                        f"{point.identity} ({point.kind}) is a new attachment point, "
                        f"already saturated at {point.member_count} members, with no "
                        "baseline entry. Add a bump or move new members to a redirect "
                        "destination, or run `baseline --ratchet` after review."
                    ),
                    redirect=None,
                )
            )
            new_entries.append(
                BaselineEntry(
                    kind=point.kind,
                    identity=point.identity,
                    file=point.file,
                    member_count=point.member_count,
                    boundary=verdict.boundary,
                )
            )
            continue

        ceiling = effective_ceiling(baseline_entry)
        if baseline_entry.pinned:
            # A pin can only TIGHTEN the contract (issue #1620): cap the
            # ceiling at the fence so a pin row whose member_count sits above
            # the boundary cannot silently exempt its point from saturation.
            ceiling = min(ceiling, verdict.boundary)
        if point.member_count > ceiling:
            qualifier = "pinned ceiling" if baseline_entry.pinned else "baselined ceiling"
            findings.append(
                Finding(
                    severity="block",
                    file=point.file,
                    identity=point.identity,
                    message=(
                        f"{point.identity} ({point.kind}) has {point.member_count} "
                        f"members, exceeding {qualifier} {ceiling}. Add a bump "
                        "or move new members to a redirect destination."
                    ),
                    redirect=None,
                )
            )
            new_entries.append(baseline_entry)
        elif point.member_count < baseline_entry.member_count:
            # Ratchet down: strictly improved, drop stale bumps for this point.
            new_entries.append(
                BaselineEntry(
                    kind=point.kind,
                    identity=point.identity,
                    file=point.file,
                    member_count=point.member_count,
                    boundary=verdict.boundary,
                    pinned=baseline_entry.pinned,
                )
            )
        else:
            new_entries.append(baseline_entry)

    # Points baselined before but no longer saturated at all are simply absent
    # from `new_entries` (the loop above only ever visits currently-saturated
    # points) — that is the "ratchet down to not tracked" case, handled by
    # omission rather than an explicit branch. Pinned rows are the exception
    # (issue #1620): their point is deliberately BELOW the fence, so the
    # saturation loop never visits them — evaluate them separately and keep
    # the row even when nothing about it changes.
    for key, entry in baseline_entries.items():
        if not entry.pinned or key in current_by_key:
            continue
        verdict = all_verdicts.get(key)
        if verdict is None:
            # The point left the eligible population entirely (class deleted,
            # renamed, ledger/trivial, or zero members): the pin can neither
            # block nor ratchet, so the row is carried verbatim -- dropping
            # it would silently end the contract, and a re-eligible point is
            # checked against it again on the next pass.
            new_entries.append(entry)
            continue
        point = verdict.point
        ceiling = min(effective_ceiling(entry), verdict.boundary)
        if point.member_count > ceiling:
            findings.append(
                Finding(
                    severity="block",
                    file=point.file,
                    identity=point.identity,
                    message=(
                        f"{point.identity} ({point.kind}) has {point.member_count} "
                        f"members, exceeding pinned ceiling {ceiling} below the "
                        "saturation fence. The row is an explicit sub-saturation "
                        "contract on a de-godded point (issue #1620) -- add a "
                        "bump or move new members to a redirect destination."
                    ),
                    redirect=None,
                )
            )
            new_entries.append(entry)
        elif point.member_count < entry.member_count:
            # The pin ratchets down with the class like any other row: a
            # shrink tightens the contract (stale bumps are dropped the same
            # way -- a raise on a now-lower ceiling no longer applies).
            new_entries.append(
                BaselineEntry(
                    kind=point.kind,
                    identity=point.identity,
                    file=point.file,
                    member_count=point.member_count,
                    boundary=verdict.boundary,
                    pinned=True,
                )
            )
        else:
            new_entries.append(entry)
    sorted_entries = sorted(new_entries, key=_entry_sort_key)
    # Finding #11: preserve every top-level key already in the document
    # (e.g. an operator-set "mode": "enforce") instead of rebuilding from a
    # fixed key allowlist. A routine `baseline --ratchet` must never silently
    # strip a key it doesn't know about -- that previously reverted the
    # PreToolUse hook's enforce mode back to "advise" with no finding and a
    # diff that reads as a normal ratchet.
    ratcheted = {
        **baseline_document,
        "version": SCHEMA_VERSION,
        "entries": [_entry_to_dict(e) for e in sorted_entries],
    }
    # Issue #1614: a ratchet must NOT recompute or raise the frozen per-kind
    # fence. ``kind_stats`` is preserved verbatim from ``baseline_document``
    # by the spread above (compare never writes KIND_STATS_KEY itself -- only
    # generate() and with_kind_stats() do, on the explicit re-baseline /
    # --refreeze paths). The boundary-raise tamper guard lives in
    # check_ratchet_tamper, which diffs the frozen fence against the previous
    # committed baseline.
    return findings, ratcheted
