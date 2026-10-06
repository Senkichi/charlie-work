"""Per-process cache of parsed repository sources for AST-scanning tests (HS-CW-3).

About sixty test modules read and ``ast.parse`` files under ``src/`` -- many
of them the whole package, several times per module. A full-tree parse costs
~0.65 s; this module parses each file once per pytest process (each xdist
worker is one) and hands every caller the same tree.

Only files under ``src/`` and ``scripts/`` are cached (~140 MiB per process
for all of ``src/``). ``tests/`` is twice that size and few modules walk it,
so it -- like any ``tmp_path`` file -- is read and parsed fresh on every call.

Shared trees must not be mutated. The session fixture
``conftest._src_ast_mutation_guard`` re-digests every handed-out tree at
session end and fails naming each file whose tree changed; a caller that
needs to edit a tree uses ``parsed_fresh``. ``CI_FLEET_TEST_REUSE=off`` turns
every cache here off; it is read on every call.
"""

from __future__ import annotations

import ast
import hashlib
import os
import pickle
from pathlib import Path

import pytest

REUSE_ENV = "CI_FLEET_TEST_REUSE"
REPO_ROOT = Path(__file__).resolve().parents[1]


def reuse_enabled() -> bool:
    return os.environ.get(REUSE_ENV, "").strip().lower() != "off"


def _digest(tree: ast.AST) -> str:
    # pickle, not ast.dump: ~3x faster over src/ and it also sees attributes a
    # consumer bolts on (``node.parent = ...``), which ast.dump ignores.
    return hashlib.sha256(pickle.dumps(tree, protocol=pickle.HIGHEST_PROTOCOL)).hexdigest()


def parsed_fresh(path: Path) -> ast.Module:
    """A private tree the caller may mutate; never cached."""
    return ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))


class SourceCache:
    """Path-keyed text/tree cache for files under ``roots``; ``_CACHE`` is the shared one."""

    def __init__(self, roots: tuple[Path, ...]) -> None:
        self._roots = tuple(Path(root).resolve() for root in roots)
        self._texts: dict[Path, str] = {}
        self._trees: dict[Path, tuple[ast.Module, str]] = {}
        self._files: dict[tuple[str, str], tuple[Path, ...]] = {}

    def _key(self, path: Path) -> Path | None:
        if not reuse_enabled():
            return None
        try:
            resolved = Path(path).resolve()
        except OSError:
            return None
        if any(resolved.is_relative_to(root) for root in self._roots):
            return resolved
        return None

    def source_text(self, path: Path) -> str:
        key = self._key(path)
        if key is None:
            return Path(path).read_text(encoding="utf-8")
        if key not in self._texts:
            self._texts[key] = key.read_text(encoding="utf-8")
        return self._texts[key]

    def parsed(self, path: Path) -> ast.Module:
        key = self._key(path)
        if key is None:
            return parsed_fresh(path)
        if key not in self._trees:
            tree = ast.parse(self.source_text(key), filename=str(key))
            self._trees[key] = (tree, _digest(tree))
        return self._trees[key][0]

    def parsed_source(self, source: str, filename: str = "<unknown>") -> ast.Module:
        """``ast.parse(source)``, shared when ``filename`` is a cached file with this text."""
        key = None if filename == "<unknown>" else self._key(Path(filename))
        if key is not None and key.is_file() and source == self.source_text(key):
            return self.parsed(key)
        return ast.parse(source, filename=filename)

    def source_files(self, root: Path, pattern: str = "*.py") -> tuple[Path, ...]:
        cacheable = self._key(root) is not None
        key = (str(root), pattern)
        if cacheable and key in self._files:
            return self._files[key]
        files = tuple(sorted(Path(root).rglob(pattern)))
        if cacheable:
            self._files[key] = files
        return files

    def mutated_paths(self) -> list[Path]:
        return [path for path, (tree, digest) in self._trees.items() if _digest(tree) != digest]


_CACHE = SourceCache((REPO_ROOT / "src", REPO_ROOT / "scripts"))


def source_text(path: Path) -> str:
    return _CACHE.source_text(path)


def parsed(path: Path) -> ast.Module:
    return _CACHE.parsed(path)


def parsed_source(source: str, filename: str = "<unknown>") -> ast.Module:
    return _CACHE.parsed_source(source, filename)


def source_files(root: Path, pattern: str = "*.py") -> tuple[Path, ...]:
    return _CACHE.source_files(root, pattern)


def assert_no_mutations(cache: SourceCache | None = None) -> None:
    mutated = (cache or _CACHE).mutated_paths()
    if mutated:
        pytest.fail(
            "shared AST mutated by a test (use _src_ast.parsed_fresh for a private copy): "
            + ", ".join(str(path) for path in mutated),
            pytrace=False,
        )
