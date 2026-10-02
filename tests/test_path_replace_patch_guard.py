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
"""

from __future__ import annotations

import re
from pathlib import Path

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


def _raw_patch_lines(source: str) -> list[int]:
    """1-based line numbers in ``source`` carrying a raw class-wide patch."""
    return [n for n, line in enumerate(source.splitlines(), 1) if _RAW_PATCH.search(line)]


def test_guard_detects_each_raw_spelling() -> None:
    """Positive control: the scanner must catch every spelling it bans.

    The needle strings are concatenated so they never appear literally in
    this file -- keeping this file inside the guard's own scan.
    """
    attr = "Path, " + '"replace"'
    assert _raw_patch_lines(f"monkeypatch.setattr({attr}, fake)") == [1]
    assert _raw_patch_lines(f"monkeypatch.setattr(pathlib.{attr}, fake)") == [1]
    assert _raw_patch_lines(f"patch.setattr( {attr} , fake)") == [1]
    string_target = '"pathlib.Path.replace"'
    assert _raw_patch_lines(f"monkeypatch.setattr({string_target}, fake)") == [1]
    # Negative controls: a different attribute and the scoped helper itself.
    assert _raw_patch_lines('monkeypatch.setattr(Path, "unlink", fake)') == []
    assert _raw_patch_lines("patch_path_replace(fake, scope=tmp_path)") == []


def test_no_raw_class_wide_path_replace_patch_in_tests() -> None:
    offenders: list[str] = []
    for file in sorted(TESTS_DIR.rglob("*.py")):
        if file.name in _EXEMPT_FILENAMES:
            continue
        rel = file.relative_to(TESTS_DIR.parent).as_posix()
        for lineno in _raw_patch_lines(file.read_text(encoding="utf-8")):
            offenders.append(f"{rel}:{lineno}")
    assert not offenders, (
        "raw class-wide Path.replace patch bypasses the scoped helper "
        "patch_path_replace (issue #2290):\n  " + "\n  ".join(offenders)
    )
