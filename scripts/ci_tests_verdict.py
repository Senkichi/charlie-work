"""Verdict for charlie-work's aggregate ``Tests`` check (dedup spec component 2).

The aggregate ``needs`` the shard matrix, but ``needs.tests-shard.result`` is
one value for the whole matrix, so per-shard conclusions are read from the
jobs API (``gh api .../attempts/N/jobs``) and passed in as a JSON-lines file of
``[name, conclusion]`` pairs.

Verdicts:

* ``pass``   -- every shard succeeded, or the push was covered and the shards
  were skipped on purpose.
* ``fail``   -- a shard failed, a shard is missing, shards were skipped on an
  uncovered run, or the coverage job failed. A real failure wins over a
  cancelled sibling: it is a code fault and the rework path is right.
* ``cancel`` -- a shard, the coverage job or the collect-only gate was
  cancelled or timed out (an infrastructure fault). The workflow then cancels
  its own run so ``Tests`` ends ``cancelled``, which charlie-work's #841
  infra-rerun path classifies as infra instead of dispatching code rework.

A failed collect-only gate does not fail ``Tests``: the gate is its own
required check, and ``Tests`` needs only the head collection it uploads.
Unreadable input fails closed (``fail``).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

SHARD_PREFIX = "Tests shard "
_INFRA = frozenset({"cancelled", "timed_out", "startup_failure", "stale"})


@dataclass(frozen=True)
class Verdict:
    verdict: str
    reason: str


def decide(
    shard_jobs: Sequence[tuple[str, str | None]],
    *,
    shard_result: str,
    gate_result: str,
    coverage_result: str,
    covered: bool,
    splits: int,
) -> Verdict:
    if coverage_result in _INFRA:
        return Verdict("cancel", f"coverage job {coverage_result}")
    if coverage_result != "success":
        return Verdict("fail", f"coverage job {coverage_result}")
    if covered and shard_result == "skipped":
        return Verdict("pass", "covered push: the suite already ran on the merge-queue draft")
    if gate_result in _INFRA:
        return Verdict("cancel", f"collect-only-gate {gate_result}")
    shards = [(name, c) for name, c in shard_jobs if name.startswith(SHARD_PREFIX)]
    failed = sorted(name for name, c in shards if c == "failure")
    if failed:
        return Verdict("fail", "failed shard(s): " + ", ".join(failed))
    infra = sorted(name for name, c in shards if c is None or c in _INFRA)
    if infra:
        return Verdict("cancel", "infrastructure fault on: " + ", ".join(infra))
    if len(shards) != splits:
        return Verdict("fail", f"expected {splits} shard jobs, found {len(shards)}")
    not_ok = sorted(f"{name} ({c})" for name, c in shards if c != "success")
    if not_ok:
        return Verdict("fail", "shard(s) not successful: " + ", ".join(not_ok))
    return Verdict("pass", f"all {splits} shards passed")


def _read_jobs(path: Path) -> list[tuple[str, str | None]]:
    jobs: list[tuple[str, str | None]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        name, conclusion = json.loads(line)
        jobs.append((str(name), None if conclusion in (None, "") else str(conclusion)))
    return jobs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--shard-result", required=True)
    parser.add_argument("--gate-result", required=True)
    parser.add_argument("--coverage-result", required=True)
    parser.add_argument("--covered", required=True)
    parser.add_argument("--splits", required=True)
    args = parser.parse_args(argv)
    try:
        result = decide(
            _read_jobs(args.jobs),
            shard_result=args.shard_result,
            gate_result=args.gate_result,
            coverage_result=args.coverage_result,
            covered=args.covered == "true",
            splits=int(args.splits),
        )
    except (OSError, ValueError, TypeError) as exc:
        result = Verdict("fail", f"could not read shard conclusions: {exc}")
    print(f"verdict={result.verdict}")
    print(f"reason={result.reason}")
    print(f"Tests verdict: {result.verdict} ({result.reason})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
