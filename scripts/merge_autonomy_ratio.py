"""Report the merge-autonomy ratio, per repo, over a time window.

charlie-work's whole purpose is that PRs reach `MERGED` without a human at
the keyboard. Measuring that sounds trivial -- "who merged it?" -- but the
obvious fields all lie: the orchestrator authenticates AS the operator's
human GitHub account (a `gh` token, not a bot identity) so it can act on
issues and PRs the same way a person would. PR author, committer, and
`mergedBy` in the common case are therefore IDENTICAL whether a human
clicked "Merge" or the orchestrator's dispatched worker did every bit of the
work and the PR simply got merged by hand at the end. `mergedBy.login`
matching the operator's account is consistent with both stories and proves
neither.

The one field that is not authenticated as the human account is `mergedBy`
when the merge went through the Aviator merge queue: those merges are
performed by the `app/aviator-app` GitHub App, a distinct actor from the
operator's account. `mergedBy.login == "app/aviator-app"` is therefore the only
honest signal that a merge happened autonomously. See
docs/plans/STATUS-AND-EXECUTION-PLAN-2026-08-06.md (AC-7) and
docs/plans/merge-lane-recovery.md for the history of this discriminator, and
do not substitute any other field (author, committer, PR labels, etc.) for
it -- they were tried and do not discriminate.

A `mergedBy` that is absent or null is its own bucket ("unknown"), never
folded into either autonomous or human: GitHub omits `mergedBy` for some
merge commits (e.g. merges performed outside the API the record was written
through), and treating that silence as evidence in either direction would be
exactly the false-confidence failure mode this repo keeps re-discovering.

WHY THE FETCH IS SERVER-SIDE WINDOWED, NOT POST-FILTERED
-----------------------------------------------------------
This used to fetch a `--state merged` page (sorted by CREATION date, gh's
default) and filter by `mergedAt` in Python. That is unsound as a window
boundary: the fetched page is a creation-date-ordered PREFIX of all-time
merges, so a PR created long ago but merged inside the window can fall
outside that prefix -- silently, with nothing in the result signaling the
gap. This was not hypothetical: it under-counted on a real run against this
repo before the fix (see git history / PR description for the before/after
numbers).

The fetch now uses `gh pr list --search "merged:>=<since>"`, which asks
GitHub's own search index to scope the result set to the window, rather
than truncating a differently-ordered list and hoping the window fit inside
it. `gh search prs` (the other candidate) cannot be used for this: its
`--json` field set (verified against `gh search prs --help`) has no
`mergedBy`/`mergedAt` at all -- it goes through the REST search API, whose
schema doesn't carry them. `gh pr list --search` goes through the same
GraphQL-backed listing as the unsearched form and keeps the full field set,
`mergedBy` included -- confirmed empirically, not assumed.

Server-side windowing does not, by itself, prove the repo/query is valid:
`gh pr list --search` on a repo `gh` cannot resolve returns `[]` with exit
0 (GraphQL search treats an unmatched `repo:` scope as "zero results", not
a 404) -- so an empty windowed result is ambiguous between "broken query"
and "legitimately no merges in range" on its own. `fetch_merged_prs_in_window`
resolves that by probing the repo first through the unsearched path (which
does error loudly on a bad repo -- see `fetch_merged_prs`) before trusting
any empty windowed result as real. It also keeps the truncation guard: a
windowed result that hits `--limit` is still only a possibly-incomplete
page of the window, not proof the window is fully covered.

Usage::

    python scripts/merge_autonomy_ratio.py --repo owner/repo
    python scripts/merge_autonomy_ratio.py --repo owner/repo-a --repo owner/repo-b \\
        --since 2026-08-01T00:00:00Z --json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

#: The only honest autonomous-merge discriminator (see module docstring).
AVIATOR_LOGIN = "app/aviator-app"

DEFAULT_LIMIT = 200
DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_WINDOW_DAYS = 7


# --------------------------------------------------------------------------
# Pure counting logic -- no gh, no network, no filesystem. Takes a list of
# PR dicts (already filtered to the window by the caller) and classifies
# each one by its mergedBy field.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AutonomyStats:
    """Classification of a set of merged PRs by who merged them."""

    total: int
    autonomous: int
    human_counts: dict[str, int]
    unknown: int

    @property
    def human_total(self) -> int:
        return sum(self.human_counts.values())

    @property
    def ratio(self) -> float | None:
        """autonomous / total, or None when total == 0 (undefined, not 0.0)."""
        if self.total == 0:
            return None
        return self.autonomous / self.total


def compute_autonomy_stats(prs: list[dict[str, Any]]) -> AutonomyStats:
    """Classify each PR's ``mergedBy`` into autonomous / human / unknown.

    A `None`/absent `mergedBy`, or a `mergedBy` with a `None`/absent
    `login`, counts as "unknown" -- never folded into either of the other
    two buckets. This is the discriminator the whole script exists to
    apply; see the module docstring for why no other field substitutes.
    """
    autonomous = 0
    unknown = 0
    human_counts: dict[str, int] = {}

    for pr in prs:
        merged_by = pr.get("mergedBy")
        login = merged_by.get("login") if isinstance(merged_by, dict) else None
        if not login:
            unknown += 1
        elif login == AVIATOR_LOGIN:
            autonomous += 1
        else:
            human_counts[login] = human_counts.get(login, 0) + 1

    return AutonomyStats(
        total=len(prs), autonomous=autonomous, human_counts=human_counts, unknown=unknown
    )


def filter_since(prs: list[dict[str, Any]], since: datetime) -> list[dict[str, Any]]:
    """Return only the PRs whose ``mergedAt`` is >= *since*.

    `gh pr list` has no reliable server-side filter for this (it filters on
    PR *state* changes, not a merge timestamp comparison), so this is done
    in Python against the raw fetched page. A PR with a missing/unparseable
    `mergedAt` is dropped -- it cannot be placed in the window at all, and
    silently keeping it would be its own kind of false confidence.
    """
    kept: list[dict[str, Any]] = []
    for pr in prs:
        merged_at = pr.get("mergedAt")
        if not merged_at:
            continue
        try:
            merged_dt = datetime.fromisoformat(str(merged_at).replace("Z", "+00:00"))
        except ValueError:
            continue
        if merged_dt.tzinfo is None:
            merged_dt = merged_dt.replace(tzinfo=timezone.utc)
        if merged_dt >= since:
            kept.append(pr)
    return kept


# --------------------------------------------------------------------------
# Per-repo report: raw fetch metadata (for the truncation/broken-query
# guards) plus the AutonomyStats computed over the windowed subset.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RepoReport:
    repo: str
    since: str  # ISO8601, as passed to the window filter
    raw_fetched: int  # size of the page gh actually returned, pre-filter
    truncated: bool  # raw_fetched >= the page limit requested
    stats: AutonomyStats

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "repo": self.repo,
            "since": self.since,
            "raw_fetched": self.raw_fetched,
            "truncated": self.truncated,
            "total": self.stats.total,
            "autonomous": self.stats.autonomous,
            "human_total": self.stats.human_total,
            "human_counts": dict(self.stats.human_counts),
            "unknown": self.stats.unknown,
            "ratio": self.stats.ratio,
            "note": None if self.stats.total else "no merges in window - ratio undefined",
        }


def build_repo_report(
    repo: str, since: datetime, raw_prs: list[dict[str, Any]], limit: int
) -> RepoReport:
    """Combine a raw `gh pr list` page into a `RepoReport`.

    ``raw_prs`` may legitimately be empty here: when it comes from
    `fetch_merged_prs_in_window`, empty means "the window has zero merges",
    already distinguished from a broken query by that function's repo probe
    (see its docstring). `filter_since` is applied again below regardless --
    a cheap, idempotent second pass, not a correctness dependency for the
    server-side-windowed path, but real defense-in-depth if ``raw_prs`` ever
    comes from somewhere that isn't already windowed (as several tests below
    deliberately do, to exercise this function in isolation).
    """
    truncated = len(raw_prs) >= limit
    windowed = filter_since(raw_prs, since)
    stats = compute_autonomy_stats(windowed)
    return RepoReport(
        repo=repo,
        since=since.isoformat().replace("+00:00", "Z"),
        raw_fetched=len(raw_prs),
        truncated=truncated,
        stats=stats,
    )


# --------------------------------------------------------------------------
# gh fetch -- the only part of this script that touches the network. Every
# failure mode here fails loudly: non-zero exit, unparseable JSON, and an
# empty result are each a BROKEN QUERY, not "zero merges" -- conflating them
# is the false-zero this repo's tooling keeps re-discovering.
# --------------------------------------------------------------------------


def fetch_merged_prs(repo: str, limit: int, timeout: int) -> list[dict[str, Any]]:
    try:
        proc = subprocess.run(
            [
                "gh",
                "pr",
                "list",
                "--repo",
                repo,
                "--state",
                "merged",
                "--limit",
                str(limit),
                "--json",
                "number,mergedAt,mergedBy",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise SystemExit(f"gh pr list timed out after {timeout}s for {repo}") from exc
    except FileNotFoundError as exc:
        raise SystemExit("gh CLI is not installed or not on PATH") from exc

    if proc.returncode != 0:
        raise SystemExit(
            f"gh pr list failed for {repo} ({proc.returncode}): {proc.stderr.strip()}"
        )

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"gh pr list returned unparseable JSON for {repo}: {exc}") from exc

    if not isinstance(data, list):
        raise SystemExit(
            f"gh pr list returned unexpected JSON shape for {repo}: {type(data).__name__}"
        )

    if not data:
        # An empty page is a broken query (wrong repo, no `gh` auth, or a
        # transient API hiccup), not a finding of "zero merges ever". If the
        # window itself is legitimately empty that is caught downstream, on
        # the WINDOWED subset, and reported as "undefined" -- never as this.
        raise SystemExit(
            f"gh returned zero merged PRs for {repo} -- treat as a failed query, not a result"
        )

    return data


def fetch_merged_prs_in_window(
    repo: str, since: datetime, limit: int, timeout: int
) -> list[dict[str, Any]]:
    """Fetch merged PRs for *repo* with `mergedAt` >= *since*, filtered
    SERVER-SIDE by GitHub's search index (`gh pr list --search
    "merged:>=..."`) -- see the module docstring's "WHY THE FETCH IS
    SERVER-SIDE WINDOWED" section for why this replaced fetch-then-filter.

    Runs `fetch_merged_prs(repo, limit=1, timeout)` first, discarding its
    result, purely to validate the repo/gh pathway using guards already
    proven to fail loudly on a bad repo or broken auth. This is required,
    not defensive-programming boilerplate: `gh pr list --search` returns
    `[]` with exit code 0 for a repo `gh` cannot resolve (verified
    empirically), so the windowed call below cannot tell "broken query"
    apart from "genuinely nothing merged in range" on its own. Once the
    probe has passed, an empty windowed result is trusted as the latter --
    downstream `build_repo_report` reports it as "no merges in window -
    ratio undefined", never as a silent 0.0.
    """
    fetch_merged_prs(repo, limit=1, timeout=timeout)

    since_str = since.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    try:
        proc = subprocess.run(
            [
                "gh",
                "pr",
                "list",
                "--repo",
                repo,
                "--state",
                "merged",
                "--search",
                f"merged:>={since_str}",
                "--limit",
                str(limit),
                "--json",
                "number,mergedAt,mergedBy",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise SystemExit(f"gh pr list --search timed out after {timeout}s for {repo}") from exc
    except FileNotFoundError as exc:
        raise SystemExit("gh CLI is not installed or not on PATH") from exc

    if proc.returncode != 0:
        raise SystemExit(
            f"gh pr list --search failed for {repo} ({proc.returncode}): {proc.stderr.strip()}"
        )

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"gh pr list --search returned unparseable JSON for {repo}: {exc}"
        ) from exc

    if not isinstance(data, list):
        raise SystemExit(
            f"gh pr list --search returned unexpected JSON shape for {repo}: {type(data).__name__}"
        )

    # An empty result here is NOT raised as a broken query, unlike
    # fetch_merged_prs's own empty-page guard above: the probe already
    # proved the repo/gh pathway works, so a genuinely empty window is a
    # real, reportable outcome -- see this function's docstring.
    return data


# --------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------


def _parse_since(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not a valid ISO8601 timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _print_human(reports: list[RepoReport]) -> None:
    for report in reports:
        print(f"=== {report.repo}  (since {report.since}) ===")
        if report.truncated:
            print(
                f"  WARNING: gh returned {report.raw_fetched} PRs, hit the page limit -- "
                "results may be truncated, widen --limit",
                file=sys.stderr,
            )
        stats = report.stats
        if stats.total == 0:
            print("  no merges in window - ratio undefined")
            print()
            continue
        print(f"  total merged PRs in window     : {stats.total}")
        print(f"  autonomous (app/aviator-app)   : {stats.autonomous}")
        print(f"  human-merged                   : {stats.human_total}")
        for login, count in sorted(stats.human_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"      {login:<20} {count}")
        print(f"  unknown (mergedBy null/absent) : {stats.unknown}")
        ratio = stats.ratio
        assert ratio is not None  # total > 0 here, so ratio is always defined
        print(f"  autonomy ratio: {stats.autonomous}/{stats.total} = {ratio:.3f}")
        print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--repo",
        action="append",
        dest="repos",
        metavar="OWNER/NAME",
        required=True,
        help="repo to report on, as owner/name (repeatable; at least one required)",
    )
    parser.add_argument(
        "--since",
        type=_parse_since,
        default=None,
        help="ISO8601 window start (default: 7 days before now)",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable JSON output")
    parser.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT, help=f"gh page size (default: {DEFAULT_LIMIT})"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"gh subprocess timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS})",
    )
    args = parser.parse_args(argv)

    repos = tuple(args.repos)
    since = (
        args.since
        if args.since is not None
        else datetime.now(timezone.utc) - timedelta(days=DEFAULT_WINDOW_DAYS)
    )

    reports: list[RepoReport] = []
    for repo in repos:
        raw = fetch_merged_prs_in_window(repo, since, args.limit, args.timeout)
        reports.append(build_repo_report(repo, since, raw, args.limit))

    if args.json:
        print(json.dumps([r.to_json_dict() for r in reports], indent=2))
    else:
        _print_human(reports)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
