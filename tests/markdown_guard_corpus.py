"""Deterministic input corpora for the markdown-guard monotone/golden tests.

Pure data: no ``charlie_work`` import, so ``regen_markdown_legacy_golden.py``
can load it in an interpreter whose ``PYTHONPATH`` points at an origin/main
checkout. Corpus order is part of the contract -- the golden table indexes
into it -- and ``corpus_fingerprint`` lets the test fail loudly when a chunk
list is edited without regenerating the table.
"""

from __future__ import annotations

import ast
import hashlib
import itertools
import json
import re
from pathlib import Path
from typing import Any
from _src_ast import parsed, source_text

APPROVED = json.dumps({"decision": "approved", "summary": "looks fine"})
REQUEST = json.dumps({"decision": "request_changes", "summary": "fix it"})
BLOCKED = json.dumps({"decision": "blocked", "summary": "no"})
DRAFT = f"```json\n{APPROVED}\n```\nRevised:\n"

# Final-block shapes the old unanchored regex caught but scan() does not.
FINAL_SHAPES: dict[str, str] = {
    "mid-line-opener": f"My verdict: ```json\n{REQUEST}\n```",
    "glued-closer": f"```json\n{REQUEST}```",
    "list-nested-4col": f"- verdict:\n\n    ```json\n    {REQUEST}\n    ```",
    "tab-indented": f"\t```json\n{REQUEST}\n\t```",
}

# md-r3 review B1 inputs: a fence only `scan` recognises after a legacy request_changes.
SCAN_ONLY_AFTER_LEGACY: dict[str, str] = {
    "tilde-after": f"```json\n{REQUEST}\n```\n~~~json\n{APPROVED}\n~~~\n",
    "unclosed-after": f"```json\n{REQUEST}\n```\n\n```json\n{APPROVED}\n",
}


def verdict_chunks() -> list[str]:
    bodies = {"approved": APPROVED, "request_changes": REQUEST, "blocked": BLOCKED}
    chunks = ["no verdict here\n", DRAFT]
    for body in bodies.values():
        chunks += [
            f"```json\n{body}\n```\n",
            f"~~~json\n{body}\n~~~\n",
            f"```json\n{body}\n",
            f"My verdict: ```json\n{body}\n```\n",
            f"```json\n{body}```\n",
            f"\t```json\n{body}\n\t```\n",
            f"- v:\n\n    ```json\n    {body}\n    ```\n",
        ]
    chunks += list(FINAL_SHAPES.values()) + list(SCAN_ONLY_AFTER_LEGACY.values())
    return chunks


def verdict_corpus() -> list[str]:
    chunks = verdict_chunks()
    corpus: list[str] = []
    for size in (1, 2, 3):
        corpus.extend("\n".join(combo) for combo in itertools.product(chunks, repeat=size))
    return corpus


KEY = "KEYKEYKEY"
# A real detectable credential (gitleaks ``aws-access-token``), so the match-key
# half of the golden table is non-empty. Split so this file is not itself flagged.
AWS = "AKIA" + "BC2D3E4F5G6H7JKL"
# Two base64 halves that form one ``jwt-base64`` match once main joins them by
# deleting the example-secret block between them (md-r4 review B1).
SPLIT_TOKEN_HEAD = "ZXlKaGJHY2lPaU" + "Ab1" * 4
SPLIT_TOKEN_TAIL = "Qz9" * 14


def outbound_chunks(key: str) -> list[str]:
    return [
        "text\n",
        f"```example-secret\n{key}\n```\n",
        f"~~~example-secret\n{key}\n~~~\n",
        f"```example-secret\n{key}\n\t```\n",
        f"```example-secret\n{key}\n    ```\n",
        f"```example-secret\n{key}\n",
        f"```\n{key}\n```\n",
        f"\t```\n{key}\n```\n",
        f"\t```example-secret\n{key}\n```\n",
        f"```x`y\n```\n```example-secret\n{key}\n```\n",
        f"x\r```example-secret\r{key}\r```\r",
        f"prose\u2028```example-secret\n{key}\n```\n",
        f"~~~\n\t\t~~~example-secret\n{key}\n~~~\n",
        f"- ```example-secret\n  {key}\n  ```\n",
        f"````example-secret\n{key}\n```\n````\n",
    ]


def _product_corpus(chunks: list[str], sizes: tuple[int, ...]) -> list[str]:
    corpus: list[str] = []
    for size in sizes:
        corpus.extend("".join(combo) for combo in itertools.product(chunks, repeat=size))
    return corpus


def outbound_corpus() -> list[str]:
    """Structure-only corpus (the "key" matches no rule): mask/event properties."""
    return _product_corpus(outbound_chunks(KEY), (1, 2, 3))


def secret_corpus() -> list[str]:
    """Same fence shapes around a REAL credential, plus the split-token halves."""
    chunks = outbound_chunks(AWS) + [
        f"{SPLIT_TOKEN_HEAD}\n",
        f"{SPLIT_TOKEN_TAIL}\n",
        f"{SPLIT_TOKEN_HEAD}\n\t```example-secret\n\n```\n{SPLIT_TOKEN_TAIL}\n",
    ]
    return _product_corpus(chunks, (1, 2, 3))


def corpus_fingerprint(corpus: list[str]) -> str:
    digest = hashlib.sha256()
    for text in corpus:
        digest.update(text.encode("utf-8", "surrogatepass"))
        digest.update(b"\x00")
    return digest.hexdigest()[:16]


# --- source normalisation (interpreter-version independent) ----------------------------


def normalized_function_source(path: Path, name: str) -> str:
    """A function's source with docstring, comments, blank lines and its own name removed.

    Text-based on purpose: ``ast.dump`` / ``ast.unparse`` output shifts between
    Python minor versions (CI is unpinned above 3.11), which would turn a pure
    interpreter upgrade into a false "legacy path changed" failure.
    """
    source = source_text(path)
    lines = source.splitlines()
    node = next(
        n
        for n in ast.walk(parsed(path))
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
    )
    drop: set[int] = set()
    first = node.body[0]
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        drop.update(range(first.lineno, (first.end_lineno or first.lineno) + 1))
    kept: list[str] = []
    for number in range(node.lineno, (node.end_lineno or node.lineno) + 1):
        line = lines[number - 1].rstrip()
        if number in drop or not line.strip() or line.lstrip().startswith("#"):
            continue
        kept.append(line)
    kept[0] = re.sub(rf"\bdef\s+{re.escape(name)}\b", "def _", kept[0], count=1)
    return "\n".join(kept)


def source_hash(path: Path, name: str) -> str:
    return hashlib.sha256(normalized_function_source(path, name).encode("utf-8")).hexdigest()[:16]


GOLDEN_PATH = Path(__file__).resolve().with_name("markdown_legacy_golden.json")


def load_golden() -> dict[str, Any]:
    """The table of origin/main's literal guard outputs over the corpora above."""
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
