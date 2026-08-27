"""Backfill rework briefs that will not regenerate on their own (F6).

Implements F6 of ``docs/plans/rework-findings-channel.md`` section 6 (read that
section before touching this file - it has the full mechanism, the measured
evidence, and the ordering constraint this script exists to respect).

One-sentence summary: ``dispatch_rework`` (``workflow.py``) reuses an
existing rework brief (``rework-prompt.md``) *verbatim* unless
``review-decision.json`` is strictly newer than it
(``_is_verdict_newer_than_brief``, ``workflow.py`` - a strict ``>``
comparison). A fix to the brief *renderer* therefore never reaches a PR whose
brief's mtime is already >= its verdict's. This script makes the verdict
strictly newer for exactly that population, so the next dispatch_rework pass
regenerates the brief through the fixed renderer.

Terminology - read carefully, this is a corrected framing:
An equal verdict/brief mtime is the NORMAL case (the verdict is written
immediately before the brief, by the same code, from the same data) and is
NOT itself evidence of a problem. The only fact that matters here is
mechanical: will ``_is_verdict_newer_than_brief`` return True or False for
this pair, right now? "will-not-regenerate" (brief mtime >= verdict mtime)
and "will-self-heal" (verdict already strictly newer -- a future
dispatch_rework pass regenerates it with no help from this script) are the
two buckets that matter. This is a *different* question from whether a
verdict's *content* has drifted from its brief (the #632 hand-corrected-
verdict case) -- do not conflate the two.

Selection (all must hold, re-derived at run time on every invocation - never
hardcoded, see CLAUDE.md global rule #9 and the plan's section 6/F6 note):
  - a ``review-decision.json`` exists under ``<state_dir>/prs/pr-<n>/``
  - its ``decision`` is ``request_changes``
  - ``rework-prompt.md`` exists alongside it (the brief-absent bucket is a
    different, disjoint population: dispatch_rework SKIPS a missing brief
    rather than regenerating it, per issue #116, so touching the verdict
    there would accomplish nothing -- this script does not attempt to
    populate a missing brief)
  - the brief's mtime is >= the verdict's mtime, i.e. "will-not-regenerate"
  - the PR is currently OPEN per GitHub (``gh``, never local state, which
    goes stale as PRs merge and close continuously)
  - the PR number is not passed via --exclude

The target set is time-varying by construction: a PR in the "will-self-heal"
bucket today migrates into "will-not-regenerate" if a dispatch_rework pass
regenerates its brief before the renderer fix deploys (the regenerated brief
becomes newer than the verdict again, through the OLD code). There is no way
to freeze this set in advance; re-derive it every run and trust what the
current run prints over any previously-cited numbers.

SAFETY - read before ever passing --apply:
Bumping a verdict's mtime while the OLD (pre-fix) renderer is still deployed
regenerates the brief through the OLD code. The freshly-written brief is then
newer than the verdict again, so the PR falls straight back into
"will-not-regenerate" - silently burning the one lever this script has, with
no signal that anything went wrong. --apply therefore refuses to run unless
--require-commit proves the renderer fix is an ancestor of the RENDERER
checkout's live HEAD -- the checkout whose code actually performs
dispatch_rework's render. In the live fleet topology that is the daemon
deployment (C:\\Users\\senki\\srv\\charlie-work-daemon), NOT the state root
selected by --repo: the fleet daemon runs all lane code (including the brief
renderer) from there for every fleet, so the gate must anchor there
regardless of which state root is being backfilled. Point --renderer-repo at
that checkout; when omitted it defaults to --repo (correct only for a
single-checkout layout where the state repo IS the renderer). This check is
not bypassable by a flag; if you need to skip it, you are doing something
this script was deliberately built to prevent.

--dry-run is the default and is always safe to run (it opens files read-only
and never calls os.utime): it still evaluates and reports the gate, since an
operator planning ahead of deployment benefits from seeing whether --apply
would currently be refused, but a failing gate never blocks the informational
preview - only --apply is refused.

Post-apply verification: --apply writes a manifest of each touched brief's
SHA-256 hash *before* the bump (bumping never touches the brief itself, so
this is also its hash at apply time). Run this script again later with
--check-regenerated to re-hash those same briefs and confirm at least one
changed -- a run that reports success while changing no brief is exactly the
silent-no-op failure mode this script exists to catch.

Usage:
    # Preview (default; never writes anything). --require-commit is still
    # mandatory so the printed report also shows whether --apply is
    # currently allowed.
    python scripts/backfill_stale_rework_briefs.py --require-commit <sha>

    # Apply, once the renderer fix has actually merged and deployed to the
    # repo this script targets:
    python scripts/backfill_stale_rework_briefs.py --require-commit <sha> --apply

    # Hold specific PRs back from this apply (e.g. one is live evidence for
    # an unrelated, still-open defect):
    python scripts/backfill_stale_rework_briefs.py --require-commit <sha> --apply \\
        --exclude 696,700

    # Later: confirm the held-back-from-old-renderer briefs actually
    # regenerated once the fix was live:
    python scripts/backfill_stale_rework_briefs.py --check-regenerated

    # Target a different checkout than the one this script lives in (e.g.
    # running from a worktree against the main checkout's live state):
    python scripts/backfill_stale_rework_briefs.py --repo C:/path/to/main-checkout \\
        --require-commit <sha>

    # Fleet topology: the state root (--repo) and the renderer checkout
    # (the daemon deployment) are different checkouts, and for the
    # job-cannon lane they are different repos entirely. The gate must
    # anchor to the renderer checkout, where the renderer-fix SHA lives:
    python scripts/backfill_stale_rework_briefs.py \\
        --repo C:/Users/senki/repos/job-cannon \\
        --renderer-repo C:/Users/senki/srv/charlie-work-daemon \\
        --require-commit <charlie-work-fix-sha> --apply
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from charlie_work.github import GitHub, GitHubError
from charlie_work.global_config import load_layered_config
from charlie_work.paths import find_repo_root, runtime_paths
from charlie_work.subprocess_runner import no_console_window_kwargs
from charlie_work.verdict_parsing import body_has_crash_signature
from charlie_work.workflow import _is_verdict_newer_than_brief, _read_review_decision

# Nanoseconds of margin added on top of the brief's mtime when bumping the
# verdict. A bare +1ns satisfies the strict ">" comparison in
# _is_verdict_newer_than_brief, but NTFS timestamps have 100ns granularity
# and other filesystems can be coarser still (FAT: 2s); a 1ms margin is
# comfortably above any of those and still reads as "just now" to a human.
_BUMP_MARGIN_NS = 1_000_000

_MANIFEST_FILENAME = "backfill-verification-manifest.json"

# The three mechanical buckets every request_changes verdict/brief pair falls
# into. See the module docstring's "Terminology" section before renaming
# these -- "stale"/"drifted" describe a different (content-divergence)
# concept and must not be reused here.
BUCKET_BRIEF_ABSENT = "brief_absent"
BUCKET_WILL_SELF_HEAL = "will_self_heal"
BUCKET_WILL_NOT_REGENERATE = "will_not_regenerate"


@dataclass(frozen=True)
class RequestChangesEntry:
    """One ``request_changes`` verdict, classified by mechanical bucket and
    live GitHub open state. Covers all three buckets -- not just the
    backfill-candidate one -- so the funnel is auditable end to end."""

    pr_number: int
    bucket: str
    decision_path: Path
    brief_path: Path | None
    verdict_mtime_ns: int
    brief_mtime_ns: int | None
    is_open: bool | None  # None => gh lookup failed; never treat as closed


@dataclass(frozen=True)
class Candidate:
    """One PR selected for backfill: will-not-regenerate, request_changes,
    open, and not excluded."""

    pr_number: int
    decision_path: Path
    brief_path: Path
    verdict_mtime_ns: int
    brief_mtime_ns: int
    decision: str


@dataclass(frozen=True)
class FunnelCounts:
    """Intermediate counts through the selection funnel (verification-ladder
    discipline: an empty final set is a claim about the query, not the
    world, until these intermediate counts show where it narrowed)."""

    verdict_dirs_total: int = 0
    unreadable_decision: int = 0
    brief_absent: int = 0
    will_self_heal: int = 0
    will_not_regenerate: int = 0
    # Cross-section restricted to decision == request_changes, spanning all
    # three buckets above -- this is what answers "of the open
    # request_changes PRs, how many need this script vs. will fix
    # themselves vs. are stuck behind a missing brief entirely?".
    request_changes_total: int = 0
    request_changes_brief_absent_total: int = 0
    request_changes_brief_absent_open: int = 0
    request_changes_will_self_heal_total: int = 0
    request_changes_will_self_heal_open: int = 0
    request_changes_will_not_regenerate_total: int = 0
    request_changes_will_not_regenerate_open: int = 0


def _iso(mtime_ns: int) -> str:
    return datetime.fromtimestamp(mtime_ns / 1_000_000_000, tz=UTC).isoformat(
        timespec="milliseconds"
    )


def _pr_dirs(prs_root: Path) -> list[Path]:
    if not prs_root.exists():
        return []
    return sorted(
        (d for d in prs_root.iterdir() if d.is_dir() and d.name.startswith("pr-")),
        key=lambda d: d.name,
    )


def _pr_number_from_dir(pr_dir: Path) -> int | None:
    suffix = pr_dir.name.removeprefix("pr-")
    try:
        return int(suffix)
    except ValueError:
        return None


def _is_pr_open(gh: GitHub, pr_number: int) -> bool | None:
    """Return True/False for the PR's open state, or None if it could not be
    determined (network/gh failure) - callers must not treat None as closed."""
    try:
        pr = gh.pr_view(pr_number)
    except GitHubError as exc:
        print(f"  WARNING: gh pr view {pr_number} failed: {exc}", file=sys.stderr)
        return None
    if not pr:
        print(f"  WARNING: gh pr view {pr_number} returned no data", file=sys.stderr)
        return None
    state = str(pr.get("state") or "").upper()
    return state == "OPEN"


def derive_entries(prs_root: Path, gh: GitHub) -> tuple[list[RequestChangesEntry], FunnelCounts]:
    """Walk every pr-* directory, bucket every request_changes verdict, and
    resolve GitHub open state for each. Never restricts to a hardcoded PR
    list -- every count here is read fresh from disk and GitHub on every
    call.

    Reuses the real ``_is_verdict_newer_than_brief`` from workflow.py for the
    bucket classification -- the same function ``dispatch_rework`` itself
    calls -- so this script's notion of "will not regenerate" can never
    drift from the orchestrator's own gate.
    """
    total = 0
    unreadable_decision = 0
    brief_absent = 0
    will_self_heal = 0
    will_not_regenerate = 0
    rc_entries: list[RequestChangesEntry] = []

    for pr_dir in _pr_dirs(prs_root):
        decision_path = pr_dir / "review-decision.json"
        if not decision_path.exists():
            continue
        total += 1

        brief_path = pr_dir / "rework-prompt.md"
        brief_exists = brief_path.exists()

        if not brief_exists:
            bucket = BUCKET_BRIEF_ABSENT
            brief_absent += 1
        elif _is_verdict_newer_than_brief(decision_path, brief_path):
            bucket = BUCKET_WILL_SELF_HEAL
            will_self_heal += 1
        else:
            bucket = BUCKET_WILL_NOT_REGENERATE
            will_not_regenerate += 1

        decision = _read_review_decision(decision_path)
        if decision is None:
            unreadable_decision += 1
            continue
        if decision.get("decision") != "request_changes":
            continue

        pr_number = _pr_number_from_dir(pr_dir)
        if pr_number is None:
            print(
                f"  WARNING: could not parse PR number from {pr_dir.name}, skipping",
                file=sys.stderr,
            )
            continue

        rc_entries.append(
            RequestChangesEntry(
                pr_number=pr_number,
                bucket=bucket,
                decision_path=decision_path,
                brief_path=brief_path if brief_exists else None,
                verdict_mtime_ns=decision_path.stat().st_mtime_ns,
                brief_mtime_ns=brief_path.stat().st_mtime_ns if brief_exists else None,
                is_open=None,  # resolved below
            )
        )

    # Resolve GitHub open state only for request_changes verdicts (bounded,
    # small set) rather than every one of the 295 verdict dirs.
    resolved: list[RequestChangesEntry] = []
    for entry in rc_entries:
        is_open = _is_pr_open(gh, entry.pr_number)
        resolved.append(
            RequestChangesEntry(
                pr_number=entry.pr_number,
                bucket=entry.bucket,
                decision_path=entry.decision_path,
                brief_path=entry.brief_path,
                verdict_mtime_ns=entry.verdict_mtime_ns,
                brief_mtime_ns=entry.brief_mtime_ns,
                is_open=is_open,
            )
        )

    def _count(bucket: str, *, open_only: bool) -> int:
        return sum(
            1 for e in resolved if e.bucket == bucket and (not open_only or e.is_open is True)
        )

    counts = FunnelCounts(
        verdict_dirs_total=total,
        unreadable_decision=unreadable_decision,
        brief_absent=brief_absent,
        will_self_heal=will_self_heal,
        will_not_regenerate=will_not_regenerate,
        request_changes_total=len(resolved),
        request_changes_brief_absent_total=_count(BUCKET_BRIEF_ABSENT, open_only=False),
        request_changes_brief_absent_open=_count(BUCKET_BRIEF_ABSENT, open_only=True),
        request_changes_will_self_heal_total=_count(BUCKET_WILL_SELF_HEAL, open_only=False),
        request_changes_will_self_heal_open=_count(BUCKET_WILL_SELF_HEAL, open_only=True),
        request_changes_will_not_regenerate_total=_count(
            BUCKET_WILL_NOT_REGENERATE, open_only=False
        ),
        request_changes_will_not_regenerate_open=_count(
            BUCKET_WILL_NOT_REGENERATE, open_only=True
        ),
    )
    return resolved, counts


def _decision_has_crash_signature(decision: dict[str, Any]) -> bool:
    """True when a persisted decision's findings channel carries a
    reviewer-session crash summary (issue #1269, W12).

    Checks both shapes the render path
    (``rework_prompts._render_required_changes_section``) understands:
    old-shape ``required_changes`` (only when ``findings_channel ==
    "external"`` -- mirroring that function's own guard, since an
    ``"external"``-tagged ``required_changes`` is the only shape known to
    ever contain merged-in external comment bodies) and new-shape
    ``external_findings`` (checked unconditionally, also mirroring the
    render guard -- a new-shape record can carry a crash comment too, since
    the collector-side fix only stops future ingestion).
    """
    findings_channel = decision.get("findings_channel")
    if findings_channel == "external":
        raw_required_changes = decision.get("required_changes")
        if isinstance(raw_required_changes, list) and any(
            body_has_crash_signature(str(item))
            for item in raw_required_changes
            if str(item).strip()
        ):
            return True
    raw_external = decision.get("external_findings")
    if isinstance(raw_external, list) and any(
        body_has_crash_signature(str(item)) for item in raw_external if str(item).strip()
    ):
        return True
    return False


def select_candidates(
    entries: list[RequestChangesEntry], *, exclude: set[int], crash_signature_only: bool = False
) -> tuple[list[Candidate], list[int]]:
    """Narrow request_changes entries to the backfill-candidate set:
    will-not-regenerate, currently open, and not excluded.

    ``crash_signature_only``, when True, additionally narrows the result to
    candidates whose persisted decision carries a reviewer-session crash
    summary (issue #1269, W12 -- see ``_decision_has_crash_signature``).
    This is an ADDITIVE, opt-in narrowing, off by default: the original F6
    selection (every will-not-regenerate, open, request_changes PR) is
    unchanged unless an operator explicitly asks to target only the
    crash-noise population with ``--crash-signature-only``. Neither
    selection needs to become crash-content-aware to make the render-side
    fix reach already-persisted records -- the #800 drift reconciler
    already re-renders and diffs every will-not-regenerate brief once the
    fix ships, with no help from this script -- but this flag lets an
    operator run a narrower, faster, auditable pass over just the known-
    poisoned population when that is preferred over the broad run.

    Returns (candidates, excluded_pr_numbers_actually_present) so the caller
    can report which --exclude entries actually mattered.
    """
    candidates: list[Candidate] = []
    excluded_present: list[int] = []
    for e in entries:
        if e.bucket != BUCKET_WILL_NOT_REGENERATE or e.is_open is not True:
            continue
        assert e.brief_path is not None and e.brief_mtime_ns is not None
        if e.pr_number in exclude:
            excluded_present.append(e.pr_number)
            continue
        decision = _read_review_decision(e.decision_path) or {}
        if crash_signature_only and not _decision_has_crash_signature(decision):
            continue
        candidates.append(
            Candidate(
                pr_number=e.pr_number,
                decision_path=e.decision_path,
                brief_path=e.brief_path,
                verdict_mtime_ns=e.verdict_mtime_ns,
                brief_mtime_ns=e.brief_mtime_ns,
                decision=str(decision.get("decision")),
            )
        )
    return candidates, excluded_present


def _current_head(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
        **no_console_window_kwargs(),
    )
    return result.stdout.strip()


def check_deployment_gate(
    renderer_repo: Path, require_commits: list[str]
) -> tuple[bool, list[str]]:
    """Verify every ref in *require_commits* is an ancestor of *renderer_repo*'s HEAD.

    *renderer_repo* is the checkout whose code actually performs
    ``dispatch_rework``'s render -- in the live fleet topology that is the
    daemon deployment (``C:\\Users\\senki\\srv\\charlie-work-daemon``), NOT
    the state root selected by ``--repo``. The gate exists to prove the
    renderer fix is deployed in the checkout that will regenerate the brief,
    so the anchor must be that checkout's HEAD, independent of which state
    root the operator targets (issue #1332). When the state root and the
    renderer are the same checkout (single-repo dev layout) the caller passes
    the same path for both.

    Returns (all_pass, failure_messages). Never raises for an ordinary
    "not an ancestor" result - only genuine command failure (git missing,
    not a repo) propagates, since that is a setup error the operator must
    see, not a routine gate failure.

    CAUTION for callers constructing *require_commits*: a ref that resolves
    relative to *renderer_repo* itself (e.g. the bare string "HEAD") is a
    tautology here -- "is renderer_repo's HEAD an ancestor of
    renderer_repo's HEAD" is always true. --require-commit must always be a
    fixed, specific commit SHA that exists in *renderer_repo*'s object
    store. A SHA from a *different* repository (e.g. a charlie-work fix SHA
    evaluated against a job-cannon checkout) is NOT supported: the renderer
    checkout cannot resolve it and git exits 128 ("Not a valid commit
    name"). This is exactly why the gate anchors to the renderer checkout
    rather than to ``--repo`` -- the renderer fix SHA lives in
    charlie-work's history, so the gate must run in a charlie-work checkout
    (the daemon deployment), never in the state root's checkout when that
    is a different repo.
    """
    head = _current_head(renderer_repo)
    failures: list[str] = []
    for ref in require_commits:
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", ref, "HEAD"],
            cwd=renderer_repo,
            text=True,
            capture_output=True,
            **no_console_window_kwargs(),
        )
        if result.returncode == 0:
            continue
        if result.returncode == 1:
            failures.append(
                f"{ref} is NOT an ancestor of {renderer_repo}'s HEAD ({head}) - "
                "the renderer fix is not deployed here yet"
            )
        else:
            failures.append(
                f"could not evaluate {ref} against HEAD ({head}): "
                f"git exited {result.returncode}: {result.stderr.strip()}"
            )
    return (not failures, failures)


def bump_verdict_mtime(candidate: Candidate) -> int:
    """Bump review-decision.json's mtime strictly past the brief's mtime.

    Only ``os.utime`` - the verdict's contents (the evidentiary record) are
    never rewritten. Returns the new mtime in nanoseconds.
    """
    new_mtime_ns = candidate.brief_mtime_ns + _BUMP_MARGIN_NS
    os.utime(candidate.decision_path, ns=(new_mtime_ns, new_mtime_ns))
    return new_mtime_ns


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _default_manifest_path(paths_root: Path) -> Path:
    return paths_root / _MANIFEST_FILENAME


def write_verification_manifest(manifest_path: Path, candidates: list[Candidate]) -> None:
    """Record each touched brief's pre-bump SHA-256 so --check-regenerated
    can later prove the backfill actually caused a regeneration, not just a
    mtime change that nothing ever consumed."""
    manifest = {
        "captured_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "briefs": {
            str(c.pr_number): {
                "brief_path": str(c.brief_path),
                "brief_sha256": _hash_file(c.brief_path),
            }
            for c in candidates
        },
    }
    tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    tmp.replace(manifest_path)


def check_regenerated(manifest_path: Path) -> int:
    """Re-hash every brief recorded in *manifest_path* and report which
    changed. Exits non-zero if the manifest is missing/empty, or if it is
    present but not a single brief changed -- both are the "reports success
    but nothing happened" failure this exists to catch."""
    if not manifest_path.exists():
        print(f"ERROR: no manifest at {manifest_path} (run --apply first).", file=sys.stderr)
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    briefs: dict[str, dict[str, str]] = manifest.get("briefs", {})
    if not briefs:
        print(f"ERROR: manifest at {manifest_path} records zero briefs.", file=sys.stderr)
        return 1

    print(f"Manifest: {manifest_path} (captured {manifest.get('captured_at')})")
    changed = 0
    for pr_number, record in briefs.items():
        brief_path = Path(record["brief_path"])
        before = record["brief_sha256"]
        if not brief_path.exists():
            print(f"PR #{pr_number}: brief no longer exists at {brief_path} - SKIP")
            continue
        after = _hash_file(brief_path)
        did_change = after != before
        changed += did_change
        status = "REGENERATED" if did_change else "unchanged"
        print(f"PR #{pr_number}: {status}  before={before[:12]} after={after[:12]}")

    print()
    if changed == 0:
        print(
            f"FAIL: 0 of {len(briefs)} briefs regenerated. This is the silent "
            "no-op failure mode -- either the deploy hasn't reached these PRs "
            "yet, or dispatch_rework has not run for them. Do not report the "
            "backfill as complete.",
            file=sys.stderr,
        )
        return 1
    print(f"PASS: {changed} of {len(briefs)} briefs regenerated.")
    return 0


def _print_funnel(counts: FunnelCounts, prs_root: Path) -> None:
    print(f"Verdict directories root: {prs_root}")
    print("Funnel (all decisions):")
    print(
        f"  verdict directories total (review-decision.json present): {counts.verdict_dirs_total}"
    )
    print(
        f"  unreadable/corrupt decision json:                         {counts.unreadable_decision}"
    )
    print(f"  brief absent (dispatch_rework SKIPS, never regenerates):   {counts.brief_absent}")
    print(f"  will self-heal (verdict already newer than brief):        {counts.will_self_heal}")
    print(
        f"  will NOT regenerate (brief mtime >= verdict mtime):       {counts.will_not_regenerate}"
    )
    print()
    print(
        f"Cross-section: decision == request_changes ({counts.request_changes_total} total), by bucket (total / currently OPEN):"
    )
    print(
        f"  brief absent:          {counts.request_changes_brief_absent_total} / {counts.request_changes_brief_absent_open}"
    )
    print(
        f"  will self-heal:        {counts.request_changes_will_self_heal_total} / {counts.request_changes_will_self_heal_open}"
    )
    print(
        f"  will NOT regenerate:   {counts.request_changes_will_not_regenerate_total} / {counts.request_changes_will_not_regenerate_open}  <- pre-exclude candidate pool"
    )


def _print_candidates(
    candidates: list[Candidate], excluded_present: list[int], *, apply_mode: bool
) -> None:
    if excluded_present:
        print(f"Excluded by --exclude (present in candidate pool): {sorted(excluded_present)}")
    if not candidates:
        print("No candidates - nothing to back-fill.")
        return
    verb = "Bumped" if apply_mode else "Would bump"
    for c in candidates:
        new_mtime_ns = c.brief_mtime_ns + _BUMP_MARGIN_NS
        print(f"PR #{c.pr_number} (decision={c.decision})")
        print(f"  verdict path:        {c.decision_path}")
        print(f"  brief path:          {c.brief_path}")
        print(f"  verdict mtime (now): {_iso(c.verdict_mtime_ns)}")
        print(f"  brief mtime:         {_iso(c.brief_mtime_ns)}")
        print(f"  {verb} verdict mtime to: {_iso(new_mtime_ns)}")


def _parse_exclude(raw: str | None) -> set[int]:
    if not raw:
        return set()
    result: set[int] = set()
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        result.add(int(chunk))
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill rework briefs that will not regenerate on their own "
            "(F6) so they pick up the fixed renderer on the next "
            "dispatch_rework pass. Dry-run by default; see the module "
            "docstring before using --apply."
        ),
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=None,
        help=(
            "Path to the git checkout holding the .var/charlie-work state to "
            "operate on (default: discovered from the current directory). "
            "Point this at the live/main checkout when running from a "
            "worktree that has no state of its own."
        ),
    )
    parser.add_argument(
        "--require-commit",
        action="append",
        metavar="SHA_OR_REF",
        help=(
            "A fixed commit SHA (not a floating ref like plain 'HEAD') that "
            "must be an ancestor of the renderer checkout's HEAD for --apply "
            "to proceed - this is the renderer-fix deployment gate. The "
            "renderer checkout is --renderer-repo if given, else --repo. "
            "The SHA must exist in that checkout's object store (a SHA from "
            "a different repository exits 128). May be given more than once "
            "(e.g. F1 and F5) to require all of them. Required for every "
            "invocation except --check-regenerated, so the report always "
            "shows whether --apply is currently allowed."
        ),
    )
    parser.add_argument(
        "--renderer-repo",
        type=Path,
        default=None,
        help=(
            "Path to the git checkout whose code actually performs the "
            "rework-brief render -- the checkout the deployment gate "
            "evaluates --require-commit against. In the live fleet topology "
            "this is the daemon deployment "
            "(C:/Users/senki/srv/charlie-work-daemon), NOT the state root "
            "selected by --repo: the gate must prove the renderer fix is "
            "deployed in the checkout that will regenerate the brief, and "
            "for the job-cannon lane that checkout is a different repo than "
            "the state root (issue #1332). Default: same as --repo (correct "
            "for a single-checkout layout where the state repo IS the "
            "renderer)."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform the mtime bump. Refused unless the deployment gate passes.",
    )
    parser.add_argument(
        "--exclude",
        default=None,
        metavar="PR,PR,...",
        help=(
            "Comma-separated PR numbers to hold back from this run's "
            "candidate set (e.g. a PR that is live evidence for a separate "
            "open defect). Default: exclude nothing."
        ),
    )
    parser.add_argument(
        "--crash-signature-only",
        action="store_true",
        help=(
            "Additionally narrow the candidate set to PRs whose persisted "
            "verdict carries a reviewer-session crash summary (issue #1269, "
            "W12) in required_changes/external_findings. Opt-in; the "
            "default selection is unchanged (every will-not-regenerate, "
            "open, request_changes PR, this script's original F6 "
            "population)."
        ),
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help=(
            "Where --apply writes (and --check-regenerated reads) the "
            "pre-bump brief-hash manifest. Default: "
            f"<state_dir>/prs/{_MANIFEST_FILENAME}."
        ),
    )
    parser.add_argument(
        "--check-regenerated",
        action="store_true",
        help=(
            "Skip derivation/apply entirely. Re-hash every brief recorded in "
            "the manifest from a prior --apply and report which actually "
            "regenerated. Exits non-zero if none did."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    explicit_repo = args.repo is not None
    repo_root = find_repo_root(cwd=args.repo, explicit=explicit_repo)
    config = load_layered_config(repo_root)
    paths = runtime_paths(repo_root, config.runtime.state_dir)
    prs_root = paths.prs

    # The deployment gate anchors to the RENDERER checkout -- the checkout
    # whose code performs dispatch_rework's render -- not the state root
    # selected by --repo. In the live fleet topology these are different
    # checkouts (the renderer is the daemon deployment; --repo may be a
    # different repo's state root, e.g. job-cannon). Default to repo_root
    # for the single-checkout layout where the state repo IS the renderer
    # (issue #1332).
    if args.renderer_repo is not None:
        renderer_repo = find_repo_root(cwd=args.renderer_repo, explicit=True)
    else:
        renderer_repo = repo_root

    print(f"Repo root: {repo_root}")
    if renderer_repo != repo_root:
        print(f"Renderer repo (deployment gate anchor): {renderer_repo}")

    if args.check_regenerated:
        manifest_path = args.manifest_path or _default_manifest_path(prs_root)
        return check_regenerated(manifest_path)

    if not args.require_commit:
        print(
            "ERROR: --require-commit is required (unless --check-regenerated is given).",
            file=sys.stderr,
        )
        return 2

    gate_ok, gate_failures = check_deployment_gate(renderer_repo, args.require_commit)
    if gate_ok:
        print(
            f"Deployment gate: PASS - all of {args.require_commit} are ancestors of "
            f"{renderer_repo}'s HEAD."
        )
    else:
        print("Deployment gate: FAIL")
        for msg in gate_failures:
            print(f"  - {msg}")

    if args.apply and not gate_ok:
        print(
            "\nABORT: --apply refused. The renderer fix is not proven deployed "
            "on this checkout's HEAD. Applying now would regenerate briefs "
            "through the OLD renderer and burn the only available lever "
            "(see this script's module docstring). Nothing was touched.",
            file=sys.stderr,
        )
        return 1

    exclude = _parse_exclude(args.exclude)

    gh = GitHub(repo_root=repo_root)
    entries, counts = derive_entries(prs_root, gh)
    candidates, excluded_present = select_candidates(
        entries, exclude=exclude, crash_signature_only=args.crash_signature_only
    )

    print()
    _print_funnel(counts, prs_root)
    print()
    _print_candidates(candidates, excluded_present, apply_mode=args.apply)

    if args.apply:
        manifest_path = args.manifest_path or _default_manifest_path(prs_root)
        # Capture pre-bump brief hashes before touching anything, so the
        # manifest reflects the state --check-regenerated should diff against.
        write_verification_manifest(manifest_path, candidates)

        bumped = 0
        postcondition_failures: list[int] = []
        for c in candidates:
            bump_verdict_mtime(c)
            bumped += 1
            if not _is_verdict_newer_than_brief(c.decision_path, c.brief_path):
                # Should be unreachable given the margin above, but this is
                # exactly the class of silent-no-op this script exists to
                # prevent, so verify rather than assume.
                postcondition_failures.append(c.pr_number)
        print()
        print(f"SUMMARY: applied - bumped {bumped} verdict mtime(s).")
        print(f"  Verification manifest written: {manifest_path}")
        print(
            "  Re-run with --check-regenerated once dispatch_rework has had a pass at these PRs."
        )
        if postcondition_failures:
            print(
                f"  ERROR: {len(postcondition_failures)} PR(s) did NOT flip the "
                f"gate after bumping: {postcondition_failures}. Investigate "
                "before trusting the next dispatch_rework pass to regenerate "
                "these.",
                file=sys.stderr,
            )
            return 1
        print(
            "  Postcondition verified: every bumped verdict now reads newer "
            "than its brief (_is_verdict_newer_than_brief == True). Each "
            "should drop out of this script's candidate set on the next run."
        )
    else:
        print()
        print(
            f"SUMMARY: dry-run - {len(candidates)} candidate(s) found, 0 changed "
            "(pass --apply, after confirming the deployment gate above passes, "
            "to perform the bump)."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
