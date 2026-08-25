"""check_file / check_tree: the two orchestration entry points (hook + CI).

`check_tree` is the single point of enforcement: it runs the full scan,
compares saturation verdicts against the committed baseline, runs the tamper
guard, and turns every G6 parse failure into an `error` Finding (never
silently dropped). `check_file` is a thin filter over `check_tree` for a
single path, so the hook's fast per-file check and CI's full-tree check can
never diverge in behavior — there is exactly one comparison algorithm.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from charlie_work.attachment_contracts import baseline as baseline_mod
from charlie_work.attachment_contracts.archetypes import scan_tree
from charlie_work.attachment_contracts.excludes import load_excludes
from charlie_work.attachment_contracts.model import Finding, ScanResult
from charlie_work.attachment_contracts.outliers import saturate_all
from charlie_work.attachment_contracts.redirect import suggest

_SEVERITY_RANK: dict[str, int] = {"error": 0, "block": 1, "advise": 2}


def _parse_failure_finding(path: str) -> Finding:
    return Finding(
        severity="error",
        file=path,
        identity=path,
        message=f"G6: {path} failed to parse; findings for this file cannot be trusted.",
        redirect=None,
    )


def _enrich_with_redirect(finding: Finding, scan: ScanResult) -> Finding:
    """Attach a G2 redirect suggestion to a block Finding, if we can locate
    the offending attachment point in the current scan."""
    point = next(
        (p for p in scan.points if p.file == finding.file and p.identity == finding.identity),
        None,
    )
    if point is None:
        return finding
    redirect = suggest(point, scan)
    message = (
        f"{finding.message} Redirect new growth to {redirect.destination}: {redirect.rationale}"
    )
    return Finding(
        severity=finding.severity,
        file=finding.file,
        identity=finding.identity,
        message=message,
        redirect=redirect.destination,
    )


def check_tree(
    root: Path,
    *,
    content_overrides: Mapping[str, str] | None = None,
    previous_baseline_document: dict[str, object] | None = None,
) -> list[Finding]:
    """Full-tree scan + baseline compare + tamper guard + G6.

    `content_overrides` is forwarded to `scan_tree` so a caller (the
    PreToolUse hook) can evaluate a pending edit's proposed content instead
    of the stale on-disk file. `previous_baseline_document`, when given,
    enables the diff-based ratchet-tamper guard (raise-to-match laundering,
    finding #1) -- it is optional and CI-only (it needs a prior commit's
    baseline as an independent reference point) because the per-edit hook
    path has no git context and should stay cheap.

    Returns an empty list when the tree is clean. Absence of a committed
    baseline is not itself a Finding (freeze-on-adopt has not happened yet in
    this repo) — only parse failures and, once a baseline exists, block/tamper
    conditions produce Findings.
    """
    excludes = load_excludes(root)
    scan = scan_tree(root, excludes, content_overrides=content_overrides)

    findings: list[Finding] = [_parse_failure_finding(pf) for pf in scan.parse_failures]

    kinds = sorted({p.kind for p in scan.points})
    verdicts = saturate_all(scan.points, kinds)

    baseline_path = root / baseline_mod.BASELINE_FILENAME
    if baseline_path.is_file():
        try:
            document = baseline_mod.load(baseline_path)
        except baseline_mod.TamperError as exc:
            findings.append(
                Finding(
                    severity="error",
                    file=baseline_mod.BASELINE_FILENAME,
                    identity=baseline_mod.BASELINE_FILENAME,
                    message=f"tamper: baseline file is structurally invalid: {exc}",
                    redirect=None,
                )
            )
        else:
            compare_findings, _ratcheted = baseline_mod.compare(verdicts, document)
            findings.extend(
                _enrich_with_redirect(f, scan) if f.severity == "block" else f
                for f in compare_findings
            )
            findings.extend(baseline_mod.check_tamper(verdicts, document))
            findings.extend(
                baseline_mod.check_ratchet_tamper(previous_baseline_document, document)
            )

    findings.sort(key=lambda f: (_SEVERITY_RANK.get(f.severity, 99), f.file, f.identity))
    return findings


def check_file(
    path: str,
    root: Path,
    *,
    content_overrides: Mapping[str, str] | None = None,
) -> list[Finding]:
    """Single-file view of `check_tree`: every Finding whose `file` is `path`.

    Delegating to `check_tree` keeps hook and CI behavior identical by
    construction — there is no second comparison algorithm to drift out of
    sync with the first.
    """
    return [
        f
        for f in check_tree(root, content_overrides=content_overrides)
        if f.file == path
    ]
