"""Regenerate ``markdown_legacy_golden.json`` from a real origin/main checkout.

Not a test (no ``test_`` prefix). The golden table is the literal output of
origin/main's markdown guards over the regression corpora in
``markdown_guard_corpus.py``; ``test_markdown_guard_monotone.py`` compares the
branch's ``_legacy_*`` paths against it. Regenerate ONLY when the corpora
change or when origin/main's own guard behaviour legitimately changes -- never
to make a failing comparison pass.

    git archive --format=tar <ORIGIN_MAIN_SHA> src | tar -x -C <dir>
    PYTHONPATH=<dir>/src uv run --no-sync python tests/regen_markdown_legacy_golden.py \
        --main-src <dir>/src --main-sha <ORIGIN_MAIN_SHA>

The script refuses to run unless ``charlie_work`` resolves under ``--main-src``
and that tree predates ``markdown_guard`` (positive control that the branch was
not imported by mistake).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import markdown_guard_corpus as corpus  # noqa: E402

GOLDEN_PATH = HERE / "markdown_legacy_golden.json"
_LABEL_CODE = {"approved": "a", "request_changes": "r", "blocked": "b", "none": "n"}


def _code(verdict: object) -> str:
    if verdict is None:
        return _LABEL_CODE["none"]
    decision = verdict["decision"] if isinstance(verdict, dict) else verdict.decision  # type: ignore[attr-defined]
    return _LABEL_CODE[str(decision)]


def _changed_lines(original: str, masked: str) -> list[int]:
    return [i for i, (a, b) in enumerate(zip(original.split("\n"), masked.split("\n"))) if a != b]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-src", required=True, type=Path)
    parser.add_argument("--main-sha", required=True)
    parser.add_argument("--out", type=Path, default=GOLDEN_PATH)
    args = parser.parse_args()

    import charlie_work

    package_dir = Path(charlie_work.__file__).resolve().parent
    main_pkg = (args.main_src / "charlie_work").resolve()
    if package_dir != main_pkg:
        sys.exit(f"charlie_work resolved to {package_dir}, expected {main_pkg}")
    if (main_pkg / "markdown_guard.py").exists():
        sys.exit("--main-src already contains markdown_guard.py: not origin/main's tree")

    from charlie_work.outbound_body_guard import _mask_example_secret_fences, scan_outbound_text
    from charlie_work.rescue_review import _find_json_verdict
    from charlie_work.verdict_parsing import _extract_verdict_from_text

    verdict_texts = corpus.verdict_corpus()
    verdicts = {
        "primary": "".join(_code(_extract_verdict_from_text(t)) for t in verdict_texts),
        "cross_family": "".join(_code(_find_json_verdict(t)) for t in verdict_texts),
    }

    def outbound_table(texts: list[str]) -> dict[str, object]:
        unique: list[dict[str, object]] = []
        seen: dict[str, int] = {}
        index: list[int] = []
        for text in texts:
            masked = _mask_example_secret_fences(text)
            keys = sorted(
                [m.rule_id, m.line, m.match_sha256[:12]]
                for m in scan_outbound_text(text, part="body")
            )
            row = {
                "masked_sha": hashlib.sha256(masked.encode("utf-8", "surrogatepass")).hexdigest()[
                    :16
                ],
                "masked_lines": _changed_lines(text, masked),
                "keys": keys,
            }
            fingerprint = json.dumps(row, sort_keys=True)
            if fingerprint not in seen:
                seen[fingerprint] = len(unique)
                unique.append(row)
            index.append(seen[fingerprint])
        return {"unique": unique, "index": index}

    outbound_texts = corpus.outbound_corpus()
    secret_texts = corpus.secret_corpus()
    golden = {
        "origin_main_sha": args.main_sha,
        "fingerprints": {
            "verdict": corpus.corpus_fingerprint(verdict_texts),
            "outbound": corpus.corpus_fingerprint(outbound_texts),
            "secret": corpus.corpus_fingerprint(secret_texts),
        },
        "verdict_labels": verdicts,
        "outbound": outbound_table(outbound_texts),
        "secret": outbound_table(secret_texts),
        # Normalised-source hashes of the origin/main functions the branch keeps
        # verbatim under a `_legacy_` name.
        "verbatim_function_hashes": {
            "verdict_parsing._extract_verdict_from_text": corpus.source_hash(
                main_pkg / "verdict_parsing.py", "_extract_verdict_from_text"
            ),
            "rescue_review._find_json_verdict": corpus.source_hash(
                main_pkg / "rescue_review.py", "_find_json_verdict"
            ),
        },
    }
    import re

    import charlie_work.outbound_body_guard as obg
    import charlie_work.rescue_review as rr
    import charlie_work.verdict_parsing as vp

    golden["constants"] = {
        "verdict_parsing._VERDICT_FENCE_RE": vp._VERDICT_FENCE_RE.pattern,
        "verdict_parsing._VERDICT_FENCE_RE.flags": vp._VERDICT_FENCE_RE.flags & ~re.UNICODE,
        "rescue_review._VERDICT_FENCE_RE": rr._VERDICT_FENCE_RE.pattern,
        "rescue_review._VERDICT_FENCE_RE.flags": rr._VERDICT_FENCE_RE.flags & ~re.UNICODE,
        "outbound_body_guard._FENCE_OPEN_RE": obg._FENCE_OPEN_RE.pattern,
        "outbound_body_guard._EXAMPLE_FENCE_INFO": obg._EXAMPLE_FENCE_INFO,
    }
    # One line per top-level key: compact (the index arrays are large) yet diffable.
    body = ",\n".join(
        f"{json.dumps(key)}:{json.dumps(golden[key], sort_keys=True, separators=(',', ':'))}"
        for key in sorted(golden)
    )
    args.out.write_text("{\n" + body + "\n}\n", encoding="utf-8")
    print(
        f"wrote {args.out}: {len(verdict_texts)} verdict, {len(outbound_texts)} outbound, "
        f"{len(secret_texts)} secret texts; "
        f"{len(golden['outbound']['unique'])}+{len(golden['secret']['unique'])} unique outputs"  # type: ignore[index]
    )


if __name__ == "__main__":
    main()
