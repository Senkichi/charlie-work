"""Record config-rejection matrix rows for config paths the fixture lacks.

``tests/test_config_rejection_set.py::test_every_live_config_path_has_a_recorded_row``
fails when a config knob is added (here, or by a ci-fleet upgrade that adds a
``runner_allocation`` field) until the knob has a recorded row. This computes
the missing rows with the test module's own probe code and inserts each one
after its siblings. Rows already in the fixture are never recomputed.

    uv run python scripts/record_config_rejection_rows.py --check   # list, exit 1 if any
    uv run python scripts/record_config_rejection_rows.py           # record them
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

import test_config_rejection_set as matrix  # noqa: E402


def _insert(mapping: dict[str, Any], key: str, value: Any) -> dict[str, Any]:
    """``mapping`` with ``key`` placed after the last key sharing its parent path."""
    parent = key.rsplit(".", 1)[0]
    keys = list(mapping)
    siblings = [i for i, k in enumerate(keys) if k == parent or k.startswith(parent + ".")]
    at = (siblings[-1] + 1) if siblings else len(keys)
    items = list(mapping.items())
    items.insert(at, (key, value))
    return dict(items)


def _row(target: dict[str, Any]) -> list[str]:
    return [
        matrix.code_of(matrix.place(target["path"], value, target["seed"]))
        for _, value in matrix.PROBES
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="list missing paths; change nothing")
    args = parser.parse_args(argv)
    fixture = json.loads(matrix.FIXTURE.read_text(encoding="utf-8"))
    missing = [t for t in matrix.walk_targets() if t["id"] not in fixture["targets"]]
    for target in missing:
        print(f"missing {target['id']}")
    if args.check or not missing:
        return 1 if missing else 0
    for target in missing:
        key = target["id"]
        fixture["targets"] = _insert(fixture["targets"], key, target["path"])
        fixture["seeds"] = _insert(fixture["seeds"], key, target["seed"])
        fixture["cells"] = _insert(fixture["cells"], key, _row(target))
        if len(target["path"]) == 1:
            fixture["sections"][key] = [
                matrix.code_of({key: matrix.PROBE_MAP[name]}) for name in matrix.SECTION_PROBES
            ]
        if target["sub"]:
            unknown = matrix.code_of(
                matrix.place([*target["path"], "zzz_unknown"], 1, target["seed"])
            )
            fixture["unknown"] = _insert(fixture["unknown"], key, unknown)
        print(f"recorded {key}")
    matrix.FIXTURE.write_text(json.dumps(fixture, indent=0) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
