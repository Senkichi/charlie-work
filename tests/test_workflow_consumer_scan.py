"""The shared ``charlie_work.workflow`` consumer scan (issue #2914).

``_workflow_consumer_scan.consumer_referenced_names`` replaced six verbatim
copies of a per-candidate-name regex loop that parsed every file under
``tests/``, ``scripts/`` and ``src/`` (~21 s per scan, twelve scans per run,
the dominant cost in the shards holding the late-sorting split modules).

Two things must stay true of the replacement, and both are pinned here on a
small synthetic tree so neither costs a real-tree scan:

* it finds exactly what the loop it replaced found (``_legacy_scan`` below is
  that loop, kept as the oracle), and
* it does not parse a file whose text cannot hold a reference. That skip is
  where the cost went; a version that parses everything again still returns
  the right answers and is only visible as a slower shard.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import _workflow_consumer_scan as scan
import pytest
from _src_ast import parsed, source_files, source_text

_CANDIDATES = {"_alpha", "_beta", "_gamma", "_delta", "_epsilon", "_zeta", "_eta", "_alp"}

# One file per reference form, plus the near-misses each form must not match.
_CORPUS: dict[str, str] = {
    "a_absolute_import.py": "from charlie_work.workflow import _alpha\n",
    "b_relative_import.py": "from .workflow import _beta\n",
    "c_parenthesized_import.py": (
        "from charlie_work.workflow import (\n    _gamma,\n    unrelated,\n)\n"
    ),
    "d_continued_import.py": "from charlie_work.workflow \\\n    import _delta\n",
    "e_attribute.py": "workflow._epsilon(1)\n",
    "f_wf_alias.py": "wf._zeta(2)\n",
    "g_setattr_string.py": 'monkeypatch.setattr("charlie_work.workflow._eta", x)\n',
    # Near-misses: none of these is a reference to a candidate.
    "h_backtick_docstring.py": '"""See ``charlie_work.workflow._alpha`` for details."""\n',
    "i_longer_name.py": "workflow._alphabet(1)\nworkflow._betamax(2)\n",
    "j_fully_qualified_attribute.py": "charlie_work.workflow._beta(1)\n",
    "k_other_module.py": "from charlie_work.other import _gamma\nother._delta(1)\n",
    "l_no_token.py": "x = 1\n",
    # A later file that repeats an earlier file's reference: first one wins.
    "z_repeat.py": "from charlie_work.workflow import _alpha\nworkflow._epsilon(1)\n",
}


def _legacy_scan(
    candidates: set[str],
    search_roots: list[Path],
    display,
    receivers: tuple[str, ...],
) -> dict[str, str]:
    """The loop the six split modules each carried before #2914, verbatim in behavior."""
    alternation = "|".join(receivers)
    attr_patterns = {
        name: re.compile(rf"(?<![.\w])(?:{alternation})\.{re.escape(name)}\b")
        for name in candidates
    }
    string_patterns = {
        name: re.compile(rf"""(['"])charlie_work\.workflow\.{re.escape(name)}\b""")
        for name in candidates
    }
    found: dict[str, str] = {}
    for root in search_roots:
        if not root.is_dir():
            continue
        for path in source_files(root):
            source = source_text(path)
            try:
                tree = parsed(path)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    module_is_workflow = node.module == "charlie_work.workflow" or (
                        node.level == 1 and node.module == "workflow"
                    )
                    if not module_is_workflow:
                        continue
                    for alias in node.names:
                        if alias.name in candidates:
                            found.setdefault(alias.name, display(path))
            for name in candidates:
                if name in found:
                    continue
                if attr_patterns[name].search(source) or string_patterns[name].search(source):
                    found[name] = display(path)
    return found


def _write_corpus(root: Path, corpus: dict[str, str]) -> None:
    for name, source in corpus.items():
        (root / name).write_text(source, encoding="utf-8")


def _shown(root: Path):
    return lambda path: path.relative_to(root).as_posix()


@pytest.mark.parametrize(
    "receivers",
    [
        pytest.param(scan.DEFAULT_RECEIVERS, id="default-receivers"),
        pytest.param(("workflow", "workflow_module", "cw_workflow"), id="no-wf-receiver"),
    ],
)
def test_scan_agrees_with_the_loop_it_replaced(tmp_path: Path, receivers: tuple[str, ...]) -> None:
    _write_corpus(tmp_path, _CORPUS)

    new = scan.consumer_referenced_names(
        _CANDIDATES, [tmp_path], display=_shown(tmp_path), receivers=receivers
    )
    old = _legacy_scan(_CANDIDATES, [tmp_path], _shown(tmp_path), receivers)

    assert new == old
    # Positive control: the corpus really exercises the forms. Agreement on two
    # empty dicts would prove nothing.
    expected = {
        "_alpha": "a_absolute_import.py",
        "_beta": "b_relative_import.py",
        "_gamma": "c_parenthesized_import.py",
        "_delta": "d_continued_import.py",
        "_epsilon": "e_attribute.py",
        "_eta": "g_setattr_string.py",
    }
    if "wf" in receivers:
        expected["_zeta"] = "f_wf_alias.py"
    assert new == expected


def test_receivers_restrict_which_aliases_count(tmp_path: Path) -> None:
    (tmp_path / "alias.py").write_text("wf._zeta(1)\n", encoding="utf-8")

    with_wf = scan.consumer_referenced_names({"_zeta"}, [tmp_path], display=_shown(tmp_path))
    without_wf = scan.consumer_referenced_names(
        {"_zeta"}, [tmp_path], display=_shown(tmp_path), receivers=("workflow",)
    )

    assert with_wf == {"_zeta": "alias.py"}
    assert without_wf == {}


def test_a_file_without_a_workflow_token_is_never_parsed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cost fix: only a file that can hold an ``ImportFrom`` of the facade is parsed.

    Against the old loop this fails: it parsed all 22 files.
    """
    for index in range(20):
        (tmp_path / f"plain_{index:02d}.py").write_text(f"VALUE = {index}\n", encoding="utf-8")
    (tmp_path / "attribute_only.py").write_text("workflow._alpha(1)\n", encoding="utf-8")
    (tmp_path / "importer.py").write_text(
        "from charlie_work.workflow import _beta\n", encoding="utf-8"
    )
    parse_calls: list[str] = []

    def counting_parsed(path: Path) -> ast.Module:
        parse_calls.append(path.name)
        return parsed(path)

    monkeypatch.setattr(scan, "parsed", counting_parsed)

    found = scan.consumer_referenced_names(
        {"_alpha", "_beta"}, [tmp_path], display=_shown(tmp_path)
    )

    assert found == {"_alpha": "attribute_only.py", "_beta": "importer.py"}
    assert parse_calls == ["importer.py"]


def test_the_first_file_in_scan_order_wins_across_forms_and_roots(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "m.py").write_text("workflow._alpha(1)\n", encoding="utf-8")
    (second / "a.py").write_text(
        "from charlie_work.workflow import _alpha, _beta\n", encoding="utf-8"
    )

    found = scan.consumer_referenced_names(
        {"_alpha", "_beta"}, [first, second], display=_shown(tmp_path)
    )

    assert found == {"_alpha": "first/m.py", "_beta": "second/a.py"}


def test_a_missing_root_is_skipped(tmp_path: Path) -> None:
    (tmp_path / "m.py").write_text("workflow._alpha(1)\n", encoding="utf-8")

    found = scan.consumer_referenced_names(
        {"_alpha"}, [tmp_path / "absent", tmp_path], display=_shown(tmp_path)
    )

    assert found == {"_alpha": "m.py"}


def test_an_unparseable_file_still_yields_its_text_forms_but_no_import(tmp_path: Path) -> None:
    """Forms 2 and 3 read the text; only the ``ImportFrom`` form needs a tree."""
    (tmp_path / "broken.py").write_text(
        "def broken(:\n    workflow._alpha(1)\nfrom charlie_work.workflow import _beta\n",
        encoding="utf-8",
    )

    found = scan.consumer_referenced_names(
        {"_alpha", "_beta"}, [tmp_path], display=_shown(tmp_path)
    )

    assert found == {"_alpha": "broken.py"}
