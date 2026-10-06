"""The branch's ``_legacy_*`` guard paths reproduce origin/main's literal outputs.

The monotone tests (``test_markdown_guard_monotone.py``) prove the composed
guards are never weaker than the ``_legacy_*`` functions -- but that is only as
good as those functions being origin/main. A later "simplification" of a legacy
body (say, dropping the backtick-in-info rejection) would keep every
"composed >= legacy" property green while the baseline quietly loosened.

This file anchors the baseline to the real thing, two independent ways:

* ``markdown_legacy_golden.json`` -- origin/main's masked text, masked line
  sets, match keys and verdict labels over the regression corpora, produced by
  running origin/main's code (``regen_markdown_legacy_golden.py``);
* pinned hashes of each ``_legacy_*`` function body (docstring, comments and
  its own name stripped): the two that are verbatim origin/main are compared to
  origin/main's own hash from the table, the three adapted ones are pinned.

Both name the origin/main commit they were generated from (``origin_main_sha``
in the table). Regenerate only when origin/main's guard behaviour legitimately
changes or a corpus changes -- never to silence a failure.
"""

from __future__ import annotations

import ast
import hashlib
import re
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import markdown_guard_corpus as corpus
import pytest
from _src_ast import parsed

from charlie_work import outbound_body_guard, rescue_review, verdict_parsing
from charlie_work.outbound_body_guard import _legacy_masked_text, _scan_masked
from charlie_work.rescue_review import _legacy_find_json_verdict
from charlie_work.verdict_parsing import _legacy_extract_verdict_from_text

_SRC = Path(rescue_review.__file__).resolve().parent
_GOLDEN = corpus.load_golden()
_CODE = {"approved": "a", "request_changes": "r", "blocked": "b", "none": "n"}

# Branch-adapted legacy helpers (NOT verbatim origin/main, so no origin/main hash
# exists for them): main's `_mask_example_secret_fences` state machine recording
# line indices instead of blanking in place, and the two views derived from it.
# Their behaviour is anchored by the golden table (masked text / line set); the
# hash makes an edit to the body a deliberate act. Each was verified against
# the golden table generated from origin/main@`origin_main_sha`.
_ADAPTED_PINS = {
    "outbound_body_guard._legacy_example_secret_line_indices": "065bc454c7998ced",
    "outbound_body_guard._legacy_mask_ranges": "f4ce68c9f7c653dd",
    "outbound_body_guard._legacy_masked_text": "28e9c0795a703994",
}
# branch `_legacy_*` name -> the origin/main function it must equal verbatim.
_VERBATIM = {
    "verdict_parsing._legacy_extract_verdict_from_text": (
        "verdict_parsing._extract_verdict_from_text"
    ),
    "rescue_review._legacy_find_json_verdict": "rescue_review._find_json_verdict",
}


def _label_code(verdict: object) -> str:
    if verdict is None:
        return _CODE["none"]
    decision = verdict["decision"] if isinstance(verdict, dict) else verdict.decision  # type: ignore[attr-defined]
    return _CODE[str(decision)]


def _verdict_mismatch(
    legacy: Callable[[str], object], golden_codes: str
) -> tuple[str, str, str] | None:
    for text, want in zip(corpus.verdict_corpus(), golden_codes, strict=True):
        got = _label_code(legacy(text))
        if got != want:
            return text, want, got
    return None


def _changed_lines(original: str, masked: str) -> list[int]:
    return [i for i, (a, b) in enumerate(zip(original.split("\n"), masked.split("\n"))) if a != b]


def _outbound_mismatch(texts: list[str], table: dict[str, Any]) -> str | None:
    """First text whose legacy masked text / line set / match keys differ from main's."""
    for text, index in zip(texts, table["index"], strict=True):
        row = table["unique"][index]
        masked = _legacy_masked_text(text)
        sha = hashlib.sha256(masked.encode("utf-8", "surrogatepass")).hexdigest()[:16]
        keys = sorted(
            [m.rule_id, m.line, m.match_sha256[:12]] for m in _scan_masked(masked, part="body")
        )
        if (sha, _changed_lines(text, masked), keys) != (
            row["masked_sha"],
            row["masked_lines"],
            row["keys"],
        ):
            return text
    return None


def test_golden_table_names_the_origin_main_commit_it_was_generated_from() -> None:
    assert re.fullmatch(r"[0-9a-f]{40}", _GOLDEN["origin_main_sha"])


def test_golden_corpora_are_the_ones_the_table_was_generated_over() -> None:
    """Editing a chunk list without regenerating the table must fail loudly."""
    assert _GOLDEN["fingerprints"] == {
        "verdict": corpus.corpus_fingerprint(corpus.verdict_corpus()),
        "outbound": corpus.corpus_fingerprint(corpus.outbound_corpus()),
        "secret": corpus.corpus_fingerprint(corpus.secret_corpus()),
    }


def test_golden_table_is_not_vacuous() -> None:
    """Positive controls: the table records non-trivial main behaviour of every kind."""
    labels = _GOLDEN["verdict_labels"]
    assert set(labels["primary"]) == set("arbn")
    assert set(labels["cross_family"]) >= set("arn")
    for name in ("outbound", "secret"):
        assert any(row["masked_lines"] for row in _GOLDEN[name]["unique"]), name
    secret_rows = _GOLDEN["secret"]["unique"]
    assert {key[0] for row in secret_rows for key in row["keys"]} >= {
        "aws-access-token",
        "jwt-base64",
    }


def test_legacy_primary_verdict_matches_origin_main_labels() -> None:
    codes = _GOLDEN["verdict_labels"]["primary"]
    assert _verdict_mismatch(_legacy_extract_verdict_from_text, codes) is None


def test_legacy_cross_family_verdict_matches_origin_main_labels() -> None:
    codes = _GOLDEN["verdict_labels"]["cross_family"]
    assert _verdict_mismatch(_legacy_find_json_verdict, codes) is None


@pytest.mark.parametrize("name", ["outbound", "secret"])
def test_legacy_outbound_mask_and_match_keys_match_origin_main(name: str) -> None:
    texts = corpus.outbound_corpus() if name == "outbound" else corpus.secret_corpus()
    assert _outbound_mismatch(texts, _GOLDEN[name]) is None


# --- the golden comparison can actually fail (controls) ----------------------------


@pytest.fixture
def _restore_legacy_constants() -> Iterator[None]:
    saved = (
        verdict_parsing._VERDICT_FENCE_RE,
        outbound_body_guard._FENCE_OPEN_RE,
        outbound_body_guard._EXAMPLE_FENCE_INFO,
    )
    yield
    (
        verdict_parsing._VERDICT_FENCE_RE,
        outbound_body_guard._FENCE_OPEN_RE,
        outbound_body_guard._EXAMPLE_FENCE_INFO,
    ) = saved


@pytest.mark.usefixtures("_restore_legacy_constants")
def test_control_loosened_verdict_fence_regex_is_detected() -> None:
    # Dropping DOTALL: a multi-line JSON body no longer matches.
    verdict_parsing._VERDICT_FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)\s*\n(.*?)```")
    codes = _GOLDEN["verdict_labels"]["primary"]
    assert _verdict_mismatch(_legacy_extract_verdict_from_text, codes) is not None


@pytest.mark.usefixtures("_restore_legacy_constants")
def test_control_wider_fence_indent_is_detected() -> None:
    outbound_body_guard._FENCE_OPEN_RE = re.compile(r"^[ \t]{0,7}(`{3,}|~{3,})[ \t]*(.*)$")
    assert _outbound_mismatch(corpus.secret_corpus(), _GOLDEN["secret"]) is not None


@pytest.mark.usefixtures("_restore_legacy_constants")
def test_control_different_exempt_info_string_is_detected() -> None:
    outbound_body_guard._EXAMPLE_FENCE_INFO = "example-secrets"
    assert _outbound_mismatch(corpus.outbound_corpus(), _GOLDEN["outbound"]) is not None


# --- source pins ---------------------------------------------------------------------


def _hash(module: str, function: str) -> str:
    return corpus.source_hash(_SRC / f"{module}.py", function)


def test_verbatim_legacy_functions_equal_origin_main_source() -> None:
    main_hashes = _GOLDEN["verbatim_function_hashes"]
    assert set(main_hashes) == set(_VERBATIM.values())
    for legacy, main in _VERBATIM.items():
        module, function = legacy.split(".")
        assert _hash(module, function) == main_hashes[main], legacy


def test_adapted_legacy_functions_are_pinned() -> None:
    for qualified, pinned in _ADAPTED_PINS.items():
        module, function = qualified.split(".")
        assert _hash(module, function) == pinned, (
            f"{qualified} changed. It is the origin/main@{_GOLDEN['origin_main_sha'][:8]} "
            "baseline: re-verify it against origin/main and the golden table before re-pinning."
        )


def test_every_legacy_function_in_src_is_pinned() -> None:
    """Derived from the source tree, so a NEW `_legacy_*` function cannot skip the pins."""
    found = set()
    for path in sorted(_SRC.glob("*.py")):
        tree = parsed(path)
        found |= {
            f"{path.stem}.{node.name}"
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name.startswith("_legacy_")
        }
    assert found == set(_VERBATIM) | set(_ADAPTED_PINS)


def test_legacy_constants_equal_origin_main() -> None:
    live = {
        "verdict_parsing._VERDICT_FENCE_RE": verdict_parsing._VERDICT_FENCE_RE,
        "rescue_review._VERDICT_FENCE_RE": rescue_review._VERDICT_FENCE_RE,
    }
    want = _GOLDEN["constants"]
    for name, pattern in live.items():
        assert pattern.pattern == want[name], name
        assert pattern.flags & ~re.UNICODE == want[f"{name}.flags"], name
    assert outbound_body_guard._FENCE_OPEN_RE.pattern == want["outbound_body_guard._FENCE_OPEN_RE"]
    assert (
        outbound_body_guard._EXAMPLE_FENCE_INFO == want["outbound_body_guard._EXAMPLE_FENCE_INFO"]
    )
