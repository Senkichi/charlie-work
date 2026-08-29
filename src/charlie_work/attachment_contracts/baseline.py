"""Baseline: freeze-on-adopt, ratchet-down, bump validation, tamper guard.

`.attachment-budgets.json` is GENERATED only (via the `baseline` CLI command).
Entries are saturated points only, sorted by (kind, file, identity) for stable
diffs. Serialization is fully deterministic: sorted entries, indent=1, sorted
keys, trailing newline — so two runs against the same scan produce byte-
identical output.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from charlie_work.attachment_contracts.model import (
    BaselineEntry,
    Bump,
    Finding,
    SaturationVerdict,
)

SCHEMA_VERSION = 1
BASELINE_FILENAME = ".attachment-budgets.json"


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
    return {
        "kind": entry.kind,
        "identity": entry.identity,
        "file": entry.file,
        "member_count": entry.member_count,
        "boundary": entry.boundary,
        "bumps": [_bump_to_dict(b) for b in entry.bumps],
    }


def _entry_from_dict(raw: dict[str, object]) -> BaselineEntry:
    bumps_raw = raw.get("bumps", [])
    if not isinstance(bumps_raw, list):
        raise TamperError(f"entries[].bumps must be a list, got {type(bumps_raw)!r}")
    try:
        return BaselineEntry(
            kind=str(raw["kind"]),  # type: ignore[arg-type]
            identity=str(raw["identity"]),
            file=str(raw["file"]),
            member_count=int(raw["member_count"]),  # type: ignore[arg-type]
            boundary=float(raw["boundary"]),  # type: ignore[arg-type]
            bumps=tuple(_bump_from_dict(b) for b in bumps_raw),
        )
    except (KeyError, ValueError, TypeError) as exc:
        # Finding #12: same rationale as _bump_from_dict -- a missing key or a
        # non-numeric member_count/boundary must become a structured Finding
        # via TamperError, not an uncaught crash that bypasses --report-only.
        raise TamperError(f"malformed baseline entry: {exc}") from exc


def _entry_sort_key(entry: BaselineEntry) -> tuple[str, str, str]:
    return (entry.kind, entry.file, entry.identity)


def generate(
    verdicts: tuple[SaturationVerdict, ...],
    *,
    generated_by: str,
    generated_at: str,
    floor: int,
) -> dict[str, object]:
    """Build the baseline document (as a plain dict, ready for dump()) from verdicts.

    Only saturated points are entered; freshly generated entries carry no bumps.
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
    return {
        "version": SCHEMA_VERSION,
        "generated_by": generated_by,
        "generated_at": generated_at,
        "floor": floor,
        "entries": [_entry_to_dict(e) for e in entries],
    }


def dumps(document: dict[str, object]) -> str:
    """Deterministic JSON: sorted entries, indent=1, sorted keys, trailing newline."""
    entries_raw = document.get("entries", [])
    ordered_entries = sorted(
        entries_raw,  # type: ignore[arg-type]
        key=lambda e: (e["kind"], e["file"], e["identity"]),
    )
    normalized = {**document, "entries": ordered_entries}
    return json.dumps(normalized, indent=1, sort_keys=True) + "\n"


def dump(document: dict[str, object], path: Path) -> None:
    path.write_text(dumps(document), encoding="utf-8")


def loads(text: str) -> dict[str, object]:
    document = json.loads(text)
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
    return document  # type: ignore[return-value]


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
    the baseline diff, e.g. CODEOWNERS on `.attachment-budgets.json` -- no
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
    The input document is never mutated; a new document dict is returned.
    """
    baseline_entries = {_entry_key(e): e for e in entries_of(baseline_document)}
    current_by_key = {_verdict_key(v): v for v in current if v.saturated}

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
        if point.member_count > ceiling:
            findings.append(
                Finding(
                    severity="block",
                    file=point.file,
                    identity=point.identity,
                    message=(
                        f"{point.identity} ({point.kind}) has {point.member_count} "
                        f"members, exceeding baselined ceiling {ceiling}. Add a bump "
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
                )
            )
        else:
            new_entries.append(baseline_entry)

    # Points baselined before but no longer saturated at all are simply absent
    # from `new_entries` (the loop above only ever visits currently-saturated
    # points) — that is the "ratchet down to not tracked" case, handled by
    # omission rather than an explicit branch.
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
    return findings, ratcheted


def check_ratchet_tamper(
    previous_document: dict[str, object] | None,
    current_document: dict[str, object],
) -> list[Finding]:
    """Diff-based tamper guard: closes the raise-to-match laundering gap.

    `check_tamper` (below) compares the baseline against the CURRENT actual
    scan only. It is blind when an attacker raises a frozen entry's
    `member_count` in lockstep with real growth: both numbers end up equal,
    so nothing looks anomalous within a single snapshot. Detecting that
    requires an independent reference point -- the previous commit's
    baseline -- which is why this is a separate function taking it
    explicitly, rather than something `check_tamper` could re-derive from
    `current_document` alone.

    Under every legitimate write path (`generate()` for a fresh entry,
    `compare()`'s ratchet), an EXISTING entry's `member_count` field is only
    ever left unchanged or lowered -- raising the effective ceiling for real
    growth is expressed purely through `bumps`, never by rewriting
    `member_count` itself (see `compare()`'s ratchet branches). So for any
    identity present in both documents, ANY rise in `member_count` did not
    come from this package's own tooling -- it is tamper, full stop.

    `previous_document` is None when there is nothing to diff against yet
    (e.g. the very first committed baseline) -- no findings are possible.
    """
    if previous_document is None:
        return []
    previous_entries = {_entry_key(e): e for e in entries_of(previous_document)}
    findings: list[Finding] = []
    for entry in entries_of(current_document):
        prev = previous_entries.get(_entry_key(entry))
        if prev is None or entry.member_count <= prev.member_count:
            continue
        findings.append(
            Finding(
                severity="error",
                file=entry.file,
                identity=entry.identity,
                message=(
                    f"tamper: baseline member_count for {entry.identity} rose "
                    f"{prev.member_count} -> {entry.member_count} since the previous "
                    "committed baseline. member_count is immutable-once-frozen except "
                    "via a strictly-lower ratchet; a rise can only be a hand-edit "
                    "(raise legitimate growth via a validly-acked bump instead)."
                ),
                redirect=None,
            )
        )
    return findings


def check_tamper(
    current: tuple[SaturationVerdict, ...],
    baseline_document: dict[str, object],
) -> list[Finding]:
    """Tamper guard: a baseline entry raised without a covering bump record.

    Recomputes what each unchanged point's baseline SHOULD show. If the
    on-disk baseline's member_count for a point is HIGHER than what the point
    itself currently reports, and no bump on that entry accounts for the
    difference, that is tamper (someone hand-edited the JSON to raise a
    ceiling) -> Finding(error).
    """
    current_by_key = {_verdict_key(v): v for v in current}
    findings: list[Finding] = []

    for entry in entries_of(baseline_document):
        key = (entry.kind, entry.file, entry.identity)
        verdict = current_by_key.get(key)
        actual_count = verdict.point.member_count if verdict is not None else None

        for bump in entry.bumps:
            error = validate_bump(bump)
            if error is not None:
                findings.append(
                    Finding(
                        severity="error",
                        file=entry.file,
                        identity=entry.identity,
                        message=f"invalid bump on {entry.identity}: {error}",
                        redirect=None,
                    )
                )

        if actual_count is not None and entry.member_count > actual_count:
            # Baseline claims more members than the point actually has, and
            # there is no bump whose `to` matches the inflated member_count —
            # the entry itself was hand-raised.
            covered = any(b.to == entry.member_count for b in entry.bumps)
            if not covered:
                findings.append(
                    Finding(
                        severity="error",
                        file=entry.file,
                        identity=entry.identity,
                        message=(
                            f"tamper: baseline member_count {entry.member_count} for "
                            f"{entry.identity} exceeds actual {actual_count} with no "
                            "covering bump"
                        ),
                        redirect=None,
                    )
                )

    return findings
