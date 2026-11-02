"""Emit GitHub ``::error`` annotations for failed junit testcases (DD-4).

The sharded ``Tests`` aggregate runs no tests itself, so without this the
failing ``Tests`` check would carry no annotations and charlie-work's
CI-findings path (``ci_findings.py``) would have nothing to read. At most 50
annotations are emitted (GitHub keeps 50 per step); the full list goes to the
step summary. Always exits 0: the verdict step decides pass/fail.
"""

from __future__ import annotations

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

MAX_ANNOTATIONS = 50
_MAX_MESSAGE = 500


@dataclass(frozen=True)
class Failure:
    file: str | None
    nodeid: str
    message: str


def escape_data(text: str) -> str:
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def escape_property(text: str) -> str:
    return escape_data(text).replace(":", "%3A").replace(",", "%2C")


def _resolve(classname: str, root: Path) -> tuple[str | None, str]:
    parts = classname.split(".")
    for i in range(len(parts), 0, -1):
        candidate = "/".join(parts[:i]) + ".py"
        if (root / candidate).is_file():
            return candidate, "::".join(parts[i:])
    return None, classname


def collect_failures(xml_text: str, root: Path) -> list[Failure]:
    try:
        tree = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    found: list[Failure] = []
    for case in tree.iter("testcase"):
        bad = case.find("failure")
        if bad is None:
            bad = case.find("error")
        if bad is None:
            continue
        classname = case.get("classname", "")
        name = case.get("name", "")
        file, inner = _resolve(classname, root)
        if file is not None:
            nodeid = "::".join(part for part in (file, inner, name) if part)
        else:
            nodeid = f"{classname}::{name}"
        message = bad.get("message") or (bad.text or "").strip() or "failed"
        found.append(Failure(file=file, nodeid=nodeid, message=message[:_MAX_MESSAGE]))
    return found


def render(failures: Sequence[Failure], cap: int = MAX_ANNOTATIONS) -> list[str]:
    lines: list[str] = []
    for failure in failures[:cap]:
        props = f"title={escape_property(failure.nodeid)}"
        if failure.file is not None:
            props = f"file={escape_property(failure.file)},{props}"
        lines.append(f"::error {props}::{escape_data(failure.message)}")
    extra = len(failures) - cap
    if extra > 0:
        lines.append(
            f"::notice title=More failures::{extra} more failed tests are listed in the step summary"
        )
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("files", nargs="*", type=Path)
    args = parser.parse_args(argv)
    failures: list[Failure] = []
    for path in args.files:
        try:
            failures.extend(collect_failures(path.read_text(encoding="utf-8"), args.root))
        except OSError as exc:
            print(f"ci_junit_annotations: skipping {path}: {exc}", file=sys.stderr)
    for line in render(failures):
        print(line)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        try:
            with open(summary, "a", encoding="utf-8") as handle:
                handle.write(f"### {len(failures)} failed test(s)\n\n")
                for failure in failures:
                    handle.write(f"- `{failure.nodeid}`: {failure.message.splitlines()[0]}\n")
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
