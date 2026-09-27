"""Tamper guards: hand-edit detection for the attachment-budget baseline.

Extracted from ``baseline.py`` (which owns the document contract) so that
module stays under the file-size ratchet cap after the issue-#1839 layout
work. Two guards live here:

- ``check_tamper`` -- single-snapshot check: a baseline entry raised above
  its point's live member count with no covering bump record.
- ``check_ratchet_tamper`` -- diff-based check: compares the current
  baseline against the previous commit's to catch raise-to-match
  laundering and frozen ``kind_stats`` fence edits.

Both emit ``Finding``s consumed by ``check_tree``.
"""

from __future__ import annotations

from charlie_work.attachment_contracts import baseline
from charlie_work.attachment_contracts.baseline import BASELINE_FILENAME, KIND_STATS_KEY
from charlie_work.attachment_contracts.model import (
    Finding,
    KindStats,
    SaturationVerdict,
)


def _kind_stats_delta(prev: KindStats, current: KindStats) -> str:
    """One-line summary of how a frozen kind's stats changed, for tamper messages."""
    if current.boundary != prev.boundary:
        direction = "rose" if current.boundary > prev.boundary else "lowered"
        return f"boundary {direction} {prev.boundary} -> {current.boundary}"
    return (
        "boundary unchanged but stats modified "
        f"(q3 {prev.q3} -> {current.q3}, iqr {prev.iqr} -> {current.iqr}, "
        f"population {prev.population} -> {current.population})"
    )


def check_ratchet_tamper(
    previous_document: dict[str, object] | None,
    current_document: dict[str, object],
    *,
    live_verdicts: tuple[SaturationVerdict, ...] | None,
    baseline_file: str = BASELINE_FILENAME,
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

    Issue #1614 extends the same logic to the frozen per-kind Tukey fence:
    under every legitimate write path a ratchet preserves ``kind_stats``
    verbatim (``compare`` spreads ``baseline_document`` without rewriting
    ``KIND_STATS_KEY``), and an explicit re-baseline / ``--refreeze`` is the
    only path that recomputes it. A generation event is recognized by TWO
    conditions that must BOTH hold: a fresh ``generated_at`` (``generate()``
    and ``with_kind_stats()`` always re-stamp) AND ``kind_stats`` equal to
    a live recompute of the tree under check. Neither half suffices alone:
    ``generated_at`` is a self-declared field in the same document an
    attacker is hand-editing, so a bare timestamp difference cannot be the
    discriminator -- a one-line forgery that bumps it alongside a raised
    boundary would read as a sanctioned regen and skip every per-kind check
    (round-4 review). The live-recompute half is the part that cannot be
    forged: the only way to write statistics equal to the honest recompute
    is to produce exactly what ``generate()`` / ``with_kind_stats()`` would
    emit, i.e. the sanctioned output itself. On a transition that is not a
    verified generation event ANY difference in ``kind_stats`` is tamper:
    a raised or lowered boundary, a tweaked q3/iqr/population (``iqr: 0`` or
    ``population < FLOOR`` disables a kind's fence outright inside
    ``saturate_with_fence``), a removed kind (a kind with no frozen fence
    gets no verdicts at all -- the whole kind is silently exempted), or an
    added kind. Removing the ``kind_stats`` key outright is flagged
    unconditionally: every baseline writer emits the key (``compare``
    preserves it; ``generate()`` / ``with_kind_stats()`` always emit it), so
    its absence can only be a hand-deletion silently reverting to the
    pre-#1614 live-recomputation fallback.

    ``live_verdicts`` is the checked tree's LIVE ``saturate_all`` output --
    the same verdict stream ``generate()`` / ``--refreeze`` consume, NOT
    the frozen-fence verdicts ``compare()`` uses for entry ratcheting.
    ``None`` means the caller had no scan context and fails the guard
    closed: no ``kind_stats`` change is excused. ``check_tree`` passes the
    live verdicts whenever it passes ``previous_baseline_document``.

    `previous_document` is None when there is nothing to diff against yet
    (e.g. the very first committed baseline) -- no findings are possible.

    ``baseline_file`` labels document-level findings (the ``kind_stats``
    guards below) with the path where the metadata actually lives:
    ``.attachment-budgets/meta.json`` under the per-entry directory layout
    (issue #1839), or ``.attachment-budgets.json`` for the legacy single
    file. ``check_tree`` supplies it; the default keeps direct callers on
    the legacy name.
    """
    if previous_document is None:
        return []
    previous_entries = {baseline._entry_key(e): e for e in baseline.entries_of(previous_document)}
    findings: list[Finding] = []
    for entry in baseline.entries_of(current_document):
        prev = previous_entries.get(baseline._entry_key(entry))
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
    # Issue #1614: a ratchet preserves ``kind_stats`` verbatim, so on a
    # ratchet-to-ratchet transition ANY difference in the frozen fence is a
    # hand-edit. The sanctioned exception is a VERIFIED generation event: a
    # fresh ``generated_at`` AND statistics equal to a live recompute of the
    # tree under check. The stamp alone is forgeable by the same hand-edit
    # it is meant to distinguish (round-4 review) -- the live-recompute
    # equality is the binding a forgery cannot satisfy without writing the
    # honest values. Malformed kind_stats already raised at loads() time,
    # so kind_stats_of here cannot raise on a document that passed loads().
    previous_stats = baseline.kind_stats_of(previous_document)
    current_stats = baseline.kind_stats_of(current_document)
    regenerated = (
        current_document.get("generated_at") is not None
        and current_document.get("generated_at") != previous_document.get("generated_at")
        and live_verdicts is not None
        and current_stats == baseline._kind_stats_from_verdicts(live_verdicts)
    )

    if KIND_STATS_KEY in previous_document and KIND_STATS_KEY not in current_document:
        # No sanctioned writer ever drops the key, so its absence can only be
        # a hand-deletion -- silently reverting check_tree / --ratchet to the
        # pre-#1614 live-recomputation fallback this issue exists to close.
        # Flag it even when ``generated_at`` changed: a real regen /
        # --refreeze still writes the key.
        findings.append(
            Finding(
                severity="error",
                file=baseline_file,
                identity=KIND_STATS_KEY,
                message=(
                    f"tamper: baseline {KIND_STATS_KEY!r} was removed since the "
                    "previous committed baseline. Removing the frozen per-kind "
                    "fence silently reverts saturation to live recomputation; "
                    "restore it via `baseline --refreeze` or a full `baseline` "
                    "run."
                ),
                redirect=None,
            )
        )
    elif not regenerated:
        for kind, prev in previous_stats.items():
            current = current_stats.get(kind)
            if current is None:
                findings.append(
                    Finding(
                        severity="error",
                        file=baseline_file,
                        identity=f"kind_stats:{kind}",
                        message=(
                            f"tamper: frozen {kind} fence was removed since the "
                            "previous committed baseline. A kind with no frozen "
                            "fence gets no verdicts at all, silently exempting "
                            "the whole kind; restore it via `baseline --refreeze` "
                            "or a full `baseline` run."
                        ),
                        redirect=None,
                    )
                )
            elif current != prev:
                findings.append(
                    Finding(
                        severity="error",
                        file=baseline_file,
                        identity=f"kind_stats:{kind}",
                        message=(
                            f"tamper: frozen {kind} fence was modified since the "
                            "previous committed baseline "
                            f"({_kind_stats_delta(prev, current)}) with no "
                            "verifiable re-baseline. A ratchet preserves "
                            "kind_stats verbatim; it may only change through a "
                            "generation event (`baseline --refreeze` or a full "
                            "`baseline` run) that re-stamps generated_at AND "
                            "writes the exact statistics a live recompute of "
                            "the current tree produces."
                        ),
                        redirect=None,
                    )
                )
        for kind in current_stats:
            if kind in previous_stats:
                continue
            findings.append(
                Finding(
                    severity="error",
                    file=baseline_file,
                    identity=f"kind_stats:{kind}",
                    message=(
                        f"tamper: frozen {kind} fence appeared since the "
                        "previous committed baseline with no verifiable "
                        "re-baseline. kind_stats may only change through a "
                        "generation event (`baseline --refreeze` or a full "
                        "`baseline` run) that re-stamps generated_at AND "
                        "writes the exact statistics a live recompute of "
                        "the current tree produces."
                    ),
                    redirect=None,
                )
            )

    # Issue #1620: a pinned row is a sub-saturation contract that no write
    # path in this package ever drops -- compare() carries it through every
    # ratchet (ratchet-down preserves the flag, de-saturation keeps the row),
    # and a verified generation event (a full `baseline` regen re-derives
    # entries from saturated verdicts alone) is the only sanctioned clean
    # slate. Outside that event, a pinned entry that vanished or lost its
    # flag is a hand-edit: deleting the row silently re-opens the regrowth
    # window the pin exists to close, and -- unlike removing an ordinary
    # entry, which finding #13 re-blocks the moment the point re-saturates --
    # an un-saturated point produces no finding to catch it.
    if not regenerated:
        current_entries = {
            baseline._entry_key(e): e for e in baseline.entries_of(current_document)
        }
        for prev in previous_entries.values():
            if not prev.pinned:
                continue
            current_entry = current_entries.get(baseline._entry_key(prev))
            if current_entry is not None and current_entry.pinned:
                continue
            findings.append(
                Finding(
                    severity="error",
                    file=prev.file,
                    identity=prev.identity,
                    message=(
                        f"tamper: pinned baseline entry for {prev.identity} was "
                        "removed or unpinned since the previous committed "
                        "baseline. Tooling never drops a pinned row; retire it "
                        "through a full `baseline` regeneration or restore the "
                        "entry."
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
    ceiling) -> Finding(error). Pinned entries (issue #1620) are exempt from
    that specific check: their member_count is an operator-authored ceiling
    that legitimately sits above the live count by design.
    """
    current_by_key = {baseline._verdict_key(v): v for v in current}
    findings: list[Finding] = []

    for entry in baseline.entries_of(baseline_document):
        key = (entry.kind, entry.file, entry.identity)
        verdict = current_by_key.get(key)
        actual_count = verdict.point.member_count if verdict is not None else None

        for bump in entry.bumps:
            error = baseline.validate_bump(bump)
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

        if actual_count is not None and entry.member_count > actual_count and not entry.pinned:
            # Baseline claims more members than the point actually has, and
            # there is no bump whose `to` matches the inflated member_count —
            # the entry itself was hand-raised. Pinned rows (issue #1620) are
            # exempt: their member_count is an operator-authored ceiling that
            # legitimately sits ABOVE the live count -- the headroom IS the
            # contract -- and a cross-commit rise is still caught by
            # check_ratchet_tamper's diff guard.
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
