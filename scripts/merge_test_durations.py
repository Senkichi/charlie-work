"""Merge pytest-split ``.test_durations`` files from every shard (DD-4).

Each shard restored the same baseline and rewrote it with its own fresh
timings, so a plain dict union would let the last shard's stale copy of a
test overwrite another shard's fresh value. The merge therefore starts from
the baseline and overlays only the entries a shard changed or added.
Exit 1 (and no write) when a shard file is unreadable or not a JSON object.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path


def merge(baseline: dict[str, float], shards: Sequence[dict[str, float]]) -> dict[str, float]:
    merged = dict(baseline)
    for shard in shards:
        for test, seconds in shard.items():
            if test not in baseline or baseline[test] != seconds:
                merged[test] = seconds
    return merged


def _load(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object")
    return {str(k): float(v) for k, v in data.items()}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("shards", nargs="+", type=Path)
    args = parser.parse_args(argv)
    try:
        baseline = _load(args.baseline) if args.baseline.is_file() else {}
        shards = [_load(path) for path in args.shards]
    except (OSError, ValueError) as exc:
        print(f"merge_test_durations: {exc}", file=sys.stderr)
        return 1
    merged = merge(baseline, shards)
    tmp = args.out.with_name(args.out.name + ".tmp")
    tmp.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, args.out)
    print(f"merge_test_durations: {len(merged)} tests from {len(shards)} shard file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
