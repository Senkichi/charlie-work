"""Tests for ``tests/_src_ast.py`` (HS-CW-3): the per-process parsed-source cache."""

from __future__ import annotations

import ast
import re
import subprocess
from collections.abc import Iterable
from pathlib import Path

import pytest

import _src_ast
from _src_ast import REUSE_ENV, SourceCache, source_files


def _module(tmp_path: Path, name: str = "mod.py", body: str = "x = 1\n") -> Path:
    path = tmp_path / "root" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def test_parsed_returns_one_shared_tree_per_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(REUSE_ENV, raising=False)
    path = _module(tmp_path)
    cache = SourceCache((path.parent,))
    first = cache.parsed(path)
    assert cache.parsed(path) is first
    assert ast.dump(first) == ast.dump(_src_ast.parsed_fresh(path))


def test_source_text_is_read_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(REUSE_ENV, raising=False)
    path = _module(tmp_path)
    cache = SourceCache((path.parent,))
    assert cache.source_text(path) == "x = 1\n"
    path.write_text("x = 2\n", encoding="utf-8")
    assert cache.source_text(path) == "x = 1\n"  # cached: no second read


def test_kill_switch_parses_on_every_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _module(tmp_path)
    cache = SourceCache((path.parent,))
    real_parse = ast.parse
    calls: list[object] = []

    def counting_parse(*args: object, **kwargs: object) -> ast.AST:
        calls.append(args[0])
        return real_parse(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(_src_ast.ast, "parse", counting_parse)
    monkeypatch.setenv(REUSE_ENV, "off")
    assert cache.parsed(path) is not cache.parsed(path)
    assert len(calls) == 2
    monkeypatch.setenv(REUSE_ENV, "on")  # read per call, not at import
    assert cache.parsed(path) is cache.parsed(path)
    assert len(calls) == 3


def test_paths_outside_the_roots_are_not_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(REUSE_ENV, raising=False)
    inside = _module(tmp_path)
    outside = tmp_path / "elsewhere.py"
    outside.write_text("y = 2\n", encoding="utf-8")
    cache = SourceCache((inside.parent,))
    assert cache.parsed(outside) is not cache.parsed(outside)
    assert cache.parsed(inside) is cache.parsed(inside)


def test_source_files_is_sorted_and_cached_per_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(REUSE_ENV, raising=False)
    root = _module(tmp_path, "b.py").parent
    _module(tmp_path, "pkg/a.py")
    cache = SourceCache((root,))
    files = cache.source_files(root)
    assert files == tuple(sorted(root.rglob("*.py")))
    _module(tmp_path, "c.py")
    assert cache.source_files(root) is files  # cached listing
    monkeypatch.setenv(REUSE_ENV, "off")
    assert len(cache.source_files(root)) == 3  # switched off: a fresh walk


def test_parsed_source_shares_the_path_tree_only_for_identical_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(REUSE_ENV, raising=False)
    path = _module(tmp_path)
    cache = SourceCache((path.parent,))
    shared = cache.parsed(path)
    assert cache.parsed_source("x = 1\n", str(path)) is shared
    assert cache.parsed_source("x = 2\n", str(path)) is not shared  # other text: fresh
    assert cache.parsed_source("x = 1\n") is not shared  # no file named: fresh
    monkeypatch.setenv(REUSE_ENV, "off")
    assert cache.parsed_source("x = 1\n", str(path)) is not shared


def test_mutating_a_shared_tree_fails_naming_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Positive control for the session guard: the failure names the file."""
    monkeypatch.delenv(REUSE_ENV, raising=False)
    path = _module(tmp_path)
    cache = SourceCache((path.parent,))
    cache.parsed(path)
    _src_ast.assert_no_mutations(cache)  # untouched: no failure
    cache.parsed(path).body.append(ast.Pass())
    with pytest.raises(pytest.fail.Exception, match=r"shared AST mutated by a test .*mod\.py"):
        _src_ast.assert_no_mutations(cache)


def test_parsed_fresh_is_a_private_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(REUSE_ENV, raising=False)
    path = _module(tmp_path)
    cache = SourceCache((path.parent,))
    shared = cache.parsed(path)
    private = _src_ast.parsed_fresh(path)
    assert private is not shared
    private.body.append(ast.Pass())
    _src_ast.assert_no_mutations(cache)  # the private copy is not watched


def test_mutation_guard_is_wired_into_every_session(request: pytest.FixtureRequest) -> None:
    assert "_src_ast_mutation_guard" in request.fixturenames


# -- guard: no new raw parse-of-read_text sites under tests/ ---------------------
#
# Issue #2541: a test that needs a ``src/`` tree must go through ``_src_ast``
# (``parsed`` / ``parsed_source`` / ``source_files``); a hand-rolled
# ``ast.parse`` of a ``path.read_text(...)`` call re-parses the file on every
# call and regresses the cache silently -- ``test_budget_governor_2442.py``
# grew exactly that scanner after the HS-CW-3 inventory. ``_src_ast.py`` itself
# is the one exempt file: it is the implementation. The banned spellings are
# never written literally here (this file is scanned too); the positive
# control builds them by concatenation.
#
# The scan runs in two stages: ``git grep -l`` narrows the tree to files
# carrying ``ast.parse`` (every banned spelling needs it), and the
# ``_RAW_SRC_PARSE`` regex then re-checks each candidate's full text -- so the
# regex is the sole definition of what is banned, and a spelling wrapped
# across lines cannot hide behind a line-oriented first stage. When git
# cannot scan the tree, every file is read.

_TESTS_DIR = Path(__file__).resolve().parent

# An ``ast.parse`` call whose argument list reaches a ``.read_text(`` call
# before its closing paren. ``[^()]`` spans newlines and ``\([^()]*\)`` covers
# one level of nested call -- e.g. the argument written as ``Path(x).`` +
# ``read_text()`` or ``find().`` + ``read_text()``; deeper nesting inside the
# argument list is not matched.
_RAW_SRC_PARSE = re.compile(r"\bast\.parse\s*\((?:[^()]|\([^()]*\))*?\.read_text\s*\(")
_SRC_AST_EXEMPT = frozenset({"_src_ast.py"})


def _raw_src_parse_lines(text: str) -> list[int]:
    """1-based line numbers of banned parse-of-read_text spellings."""
    return [text.count("\n", 0, m.start()) + 1 for m in _RAW_SRC_PARSE.finditer(text)]


def _ast_parse_candidates() -> list[Path]:
    """``tests/**/*.py`` files containing ``ast.parse`` -- or every file."""
    try:
        proc = subprocess.run(
            [
                "git",
                "grep",
                "-l",
                "-E",
                "--untracked",
                "-e",
                r"ast\.parse",
                "--",
                _TESTS_DIR.name + "/*.py",
            ],
            cwd=_TESTS_DIR.parent,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return list(source_files(_TESTS_DIR))
    if proc.returncode != 0:
        # ``_src_ast.py`` permanently contains ``ast.parse``, so "no matches"
        # already means the prefilter is not seeing the tree -- read all.
        return list(source_files(_TESTS_DIR))
    return [_TESTS_DIR.parent / rel for rel in proc.stdout.splitlines() if rel.strip()]


def _raw_src_parse_offenders(files: Iterable[Path]) -> list[str]:
    offenders: list[str] = []
    for file in files:
        if file.name in _SRC_AST_EXEMPT:
            continue
        for lineno in _raw_src_parse_lines(file.read_text(encoding="utf-8")):
            offenders.append(f"{file.relative_to(_src_ast.REPO_ROOT).as_posix()}:{lineno}")
    return offenders


def test_raw_src_parse_detector_catches_each_spelling() -> None:
    """Positive control: the scanner must catch every spelling it bans."""
    ast_parse = "ast." + "parse"
    read_text = "." + "read_text"
    spellings = [
        f"{ast_parse}(path{read_text}(encoding='utf-8'))",
        f"{ast_parse}(\n    path{read_text}()\n)",
        f"{ast_parse}(Path(x).resolve(){read_text}())",
        f"{ast_parse}(find_src(){read_text}(encoding='utf-8'), filename=str(p))",
    ]
    for spelling in spellings:
        assert _raw_src_parse_lines(spelling) == [1]
    # Negative controls: a parse of a variable, the sanctioned helpers, a
    # ``.read_text`` attribute that is not called, and a lone ``read_text``.
    assert _raw_src_parse_lines(f"{ast_parse}(source, filename=name)") == []
    assert _raw_src_parse_lines("parsed_source(source, filename)") == []
    assert _raw_src_parse_lines(f"{ast_parse}(obj{read_text}, filename=name)") == []
    assert _raw_src_parse_lines(f"text = path{read_text}(encoding='utf-8')") == []


def test_raw_src_parse_detector_fires_on_the_sanctioned_site() -> None:
    """``_src_ast.py`` carries the one sanctioned spelling -- proof the regex
    still sees it, so the filename exemption cannot neuter the detector."""
    sites = _raw_src_parse_lines(
        (_src_ast.REPO_ROOT / "tests" / "_src_ast.py").read_text(encoding="utf-8")
    )
    assert sites


def test_ast_parse_prefilter_sees_the_tree() -> None:
    """The git prefilter must report a real file or the scan may be empty."""
    candidates = _ast_parse_candidates()
    assert any(file.name == "_src_ast.py" for file in candidates)


def test_no_raw_ast_parse_of_read_text_under_tests() -> None:
    offenders = _raw_src_parse_offenders(_ast_parse_candidates())
    assert not offenders, (
        "raw ast.parse of a path.read_text() call bypasses the _src_ast "
        "parsed-source cache (HS-CW-3, issue #2541) -- use _src_ast.parsed / "
        "parsed_source / source_files instead:\n  " + "\n  ".join(offenders)
    )
