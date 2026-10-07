"""Tests for ``tests/_src_ast.py`` (HS-CW-3): the per-process parsed-source cache."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import _src_ast
from _src_ast import REUSE_ENV, SourceCache


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
    assert ast.dump(first) == ast.dump(ast.parse(path.read_text(encoding="utf-8")))


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
