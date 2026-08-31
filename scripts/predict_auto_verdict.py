"""Predict what enabling ``cross_family.auto_verdict`` would convert, before enabling it.

Answers one question: for every PR whose review decision is still ``pending``, would
the auto-verdict path record a verdict on the next pass, and if so, *which* verdict?

It reproduces the real code path from ``workflow._record_cross_family_verdicts`` —
the staleness comparison at ``workflow.py:10235`` and the parser it feeds — rather
than approximating it, so the counts it reports are the counts you should expect to
observe. Approximating this is how you get a plan that predicts the wrong outcome and
then reads a correct result as a failure.

Usage::

    python scripts/predict_auto_verdict.py --repo ../other-repo --gh-repo owner/other-repo

Live PR heads are fetched with ``gh``; pass ``--heads-json`` to reuse a saved
``gh pr list --json number,headRefOid`` payload instead.
"""

from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from charlie_work.cross_family import parse_cross_family_verdict  # noqa: E402
from charlie_work.rescue_review import extract_head_ref_oid  # noqa: E402

# Decisions the auto-verdict path treats as unresolved (workflow.py:10221).
UNRESOLVED = ("pending", "vacuous")


def _fetch_heads(gh_repo: str, timeout: int) -> dict[str, str]:
    proc = subprocess.run(
        [
            "gh",
            "pr",
            "list",
            "--repo",
            gh_repo,
            "--state",
            "open",
            "--limit",
            "500",
            "--json",
            "number,headRefOid",
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise SystemExit(f"gh pr list failed ({proc.returncode}): {proc.stderr.strip()}")
    return {str(d["number"]): d["headRefOid"] for d in json.loads(proc.stdout)}


def predict(state_root: Path, heads: dict[str, str]) -> tuple[collections.Counter, dict]:
    counts: collections.Counter = collections.Counter()
    detail: dict[str, list[str]] = collections.defaultdict(list)

    for decision_path in sorted((state_root / "prs").glob("pr-*/review-decision.json")):
        try:
            decision = json.loads(decision_path.read_text(encoding="utf-8")).get("decision")
        except (OSError, ValueError):
            continue
        if decision not in UNRESOLVED:
            continue

        pr_dir = decision_path.parent
        number = pr_dir.name.removeprefix("pr-")
        report_path = pr_dir / "cross-family-review.md"
        if not report_path.exists():
            counts["no report yet (waiting on cross-family)"] += 1
            continue

        report_text = report_path.read_text(encoding="utf-8", errors="replace")
        live_head = heads.get(number)
        if live_head is None:
            counts["PR not open (closed or merged)"] += 1
            continue

        # workflow.py:10235 — skip only when BOTH heads are known and differ.
        report_head = extract_head_ref_oid(report_text)
        if report_head is not None and report_head != live_head:
            counts["skipped: head moved (guard working as intended)"] += 1
            detail["stale"].append(number)
            continue

        parsed = parse_cross_family_verdict(report_text)
        if parsed is None:
            counts["no verdict: report does not parse"] += 1
            detail["unparseable"].append(number)
            continue
        verdict = getattr(parsed, "decision", type(parsed).__name__)
        counts[f"WOULD RECORD: {verdict}"] += 1
        detail[str(verdict)].append(number)
        if report_head is None:
            # Reaches record_review with nothing having validated the head — the
            # fail-open this plan's Gate 3 closes. Surfaced, not silently counted.
            detail["recorded_without_head_validation"].append(number)

    return counts, detail


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--repo",
        required=True,
        help="path to the target repo checkout",
    )
    ap.add_argument(
        "--gh-repo",
        required=True,
        help="owner/name for the gh query",
    )
    ap.add_argument(
        "--heads-json",
        type=Path,
        help="reuse a saved 'gh pr list --json number,headRefOid' payload",
    )
    ap.add_argument("--timeout", type=int, default=120, help="gh subprocess timeout (s)")
    args = ap.parse_args()

    state_root = Path(args.repo).expanduser().resolve() / ".var" / "charlie-work"
    if not (state_root / "prs").is_dir():
        raise SystemExit(f"no PR state under {state_root} — wrong --repo?")

    if args.heads_json:
        heads = {
            str(d["number"]): d["headRefOid"]
            for d in json.loads(args.heads_json.read_text(encoding="utf-8"))
        }
    else:
        heads = _fetch_heads(args.gh_repo, args.timeout)

    if not heads:
        # An empty head map would make every PR look closed and report a confident
        # zero. That is a broken query, not a finding.
        raise SystemExit("gh returned zero open PRs — treat as a failed query, not a result")

    counts, detail = predict(state_root, heads)

    print(f"live open PRs: {len(heads)}   state root: {state_root}")
    print()
    print("=== what enabling cross_family.auto_verdict would do on the next pass ===")
    for label, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {n:4d}  {label}")

    recorded = sum(n for label, n in counts.items() if label.startswith("WOULD RECORD"))
    print()
    print(f"  total verdicts recorded on the first pass: {recorded}")
    approved = counts.get("WOULD RECORD: approved", 0)
    print(f"  of which approvals (i.e. merge candidates): {approved}")
    print("  the rest route to REWORK, not merge — a low approval count is not a failure")

    for key, header in (
        ("approved", "approvals (expect these to merge via app/aviator-app)"),
        (
            "recorded_without_head_validation",
            "!! recorded with NO head validation (workflow.py:10235 fail-open)",
        ),
        ("unparseable", "reports that do not parse (no verdict, no harm)"),
        ("stale", "correctly skipped, head moved"),
    ):
        if detail.get(key):
            print()
            print(f"{header}:")
            print("  " + " ".join(sorted(detail[key], key=int)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
