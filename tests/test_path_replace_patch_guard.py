"""Guard: ``Path.replace`` is patched only through the scoped helper.

Issue #2290: ``monkeypatch.setattr`` targeting the ``Path`` class's
``replace`` method patches the *class* -- the patch is process-wide, so a
``Path.replace`` fired by any other thread during the window (a leftover
background writer from an earlier test on the same xdist worker) lands in
the fake too, where it is miscounted, raised at, or deadlocked on a
barrier it was never meant to join. ``patch_path_replace`` in
``tests/conftest.py`` routes only in-scope calls to the fake; this test
fails on any raw class-wide patch that bypasses it, so the flake cannot
be reintroduced site by site.

The banned spellings are never written literally in this file (the scan
covers this file too); ``test_guard_detects_each_raw_spelling`` builds
them by concatenation as a positive control.

The tree scan runs in two stages: ``git grep`` narrows the tree to lines
carrying both tokens a banned spelling needs (~50 ms for all of
``tests/``, versus ~1 s to open and read every file — issue #2389), and
the Python ``_RAW_PATCH`` regex then re-checks each reported line, so the
regex remains the sole definition of what is banned. When git cannot
scan the tree the test falls back to reading every file itself.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest
from _src_ast import source_files

TESTS_DIR = Path(__file__).resolve().parent

# Matches ``setattr`` calls whose first argument is ``Path`` or a
# module-qualified ``<module>.Path`` (``pathlib.Path``, ``workflow.Path``,
# ...) and whose second is the literal ``"replace"`` / ``'replace'``, plus
# monkeypatch's string-target form ``"pathlib.Path.replace"``.
# conftest.py is exempt: it holds the one sanctioned install site inside
# ``patch_path_replace``.
_RAW_PATCH = re.compile(
    r"setattr\s*\(\s*(?:\w+\s*\.\s*)?Path\s*,\s*['\"]replace['\"]"
    r"|setattr\s*\(\s*['\"][\w.]*\.Path\.replace['\"]"
)
_EXEMPT_FILENAMES = frozenset({"conftest.py"})

# Coarse prefilter, handed to ``git grep -E`` verbatim and compiled here for
# the positive control: every ``_RAW_PATCH`` match carries both ``setattr``
# and ``replace`` on one line, so this can only over-report. ``_RAW_PATCH``
# re-checks each line git reports, so a spelling added to ``_RAW_PATCH``
# without both tokens would be silently skipped — the positive control
# asserts every banned spelling survives this filter.
_PREFILTER_PATTERN = r"setattr.*replace|replace.*setattr"
_PREFILTER = re.compile(_PREFILTER_PATTERN)


def _raw_patch_lines(source: str) -> list[int]:
    """1-based line numbers in ``source`` carrying a raw class-wide patch."""
    return [n for n, line in enumerate(source.splitlines(), 1) if _RAW_PATCH.search(line)]


def _git_prefilter_lines() -> list[tuple[str, int, str]] | None:
    """(repo-relative path, lineno, text) rows git reports for the prefilter.

    Returns ``None`` when git cannot scan this tree (no binary, not a
    checkout), leaving the caller to take the read-every-file path. ``-z``
    gives ``path\\0lineno\\0content`` rows — a separator no filename or line
    can contain.
    """
    try:
        proc = subprocess.run(
            [
                "git",
                "grep",
                "-n",
                "-z",
                "-E",
                "--untracked",
                "-e",
                _PREFILTER_PATTERN,
                "--",
                TESTS_DIR.name + "/*.py",
            ],
            cwd=TESTS_DIR.parent,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode == 1:  # no matches
        return []
    if proc.returncode != 0:
        return None
    rows = []
    for record in proc.stdout.splitlines():
        path, sep_path, rest = record.partition("\0")
        lineno, sep_lineno, text = rest.partition("\0")
        if not (sep_path and sep_lineno and lineno.isdigit()):
            continue
        rows.append((path, int(lineno), text))
    return rows


def _disk_scan_lines() -> Iterator[tuple[str, int, str]]:
    """Every line of every ``*.py`` under ``tests/`` — the no-git path."""
    for file in source_files(TESTS_DIR):
        rel = file.relative_to(TESTS_DIR.parent).as_posix()
        for lineno, line in enumerate(file.read_text(encoding="utf-8").splitlines(), 1):
            yield rel, lineno, line


def test_guard_detects_each_raw_spelling() -> None:
    """Positive control: the scanner must catch every spelling it bans.

    The needle strings are concatenated so they never appear literally in
    this file -- keeping this file inside the guard's own scan.
    """
    attr = "Path, " + '"replace"'
    string_target = '"pathlib.Path.replace"'
    spellings = [
        f"monkeypatch.setattr({attr}, fake)",
        f"monkeypatch.setattr(pathlib.{attr}, fake)",
        f"patch.setattr( {attr} , fake)",
        f"monkeypatch.setattr({string_target}, fake)",
    ]
    for spelling in spellings:
        assert _raw_patch_lines(spelling) == [1]
        # A banned line must survive the git-grep prefilter or the fast scan
        # would skip it before ``_RAW_PATCH`` can check it.
        assert _PREFILTER.search(spelling)
    # Negative controls: a different attribute and the scoped helper itself.
    assert _raw_patch_lines('monkeypatch.setattr(Path, "unlink", fake)') == []
    assert _raw_patch_lines("patch_path_replace(fake, scope=tmp_path)") == []


def test_git_prefilter_is_not_vacuous() -> None:
    """The fast scan must report the exempt install site or it is not scanning.

    ``conftest.patch_path_replace`` carries the one sanctioned install line
    permanently; if ``git grep`` silently stops reporting rows (flag drift,
    output-format change) the guard above would pass vacuously while
    scanning nothing.
    """
    rows = _git_prefilter_lines()
    if rows is None:
        pytest.skip("git cannot scan this tree; the guard takes the read-all path")
    assert any(rel.endswith("/conftest.py") for rel, _, _ in rows)


def test_no_raw_class_wide_path_replace_patch_in_tests() -> None:
    rows = _git_prefilter_lines()
    if rows is None:
        rows = _disk_scan_lines()
    offenders = [
        f"{rel}:{lineno}"
        for rel, lineno, line in rows
        if rel.rsplit("/", 1)[-1] not in _EXEMPT_FILENAMES and _RAW_PATCH.search(line)
    ]
    assert not offenders, (
        "raw class-wide Path.replace patch bypasses the scoped helper "
        "patch_path_replace (issue #2290):\n  " + "\n  ".join(offenders)
    )
