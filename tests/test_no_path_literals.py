"""Enforcement test: well-known path names must be spelled only in layout.py.

``layout.py`` exists precisely so that names like ``"supervisor.lock"`` or the
default state-dir literal ``".var/charlie-work"`` are never re-spelled at
other use sites (see that module's docstring for the live split-brain bug
this caused). A test that only checks layout.py's own behaviour cannot catch
a regression where some *other* module hand-spells one of these names again
-- that has to be a structural scan over the source tree.

The scan is built with the :mod:`ast` module rather than a text/regex search:

* Comments are not represented in the AST at all, so they are exempt from
  both rules automatically -- there is no special-casing required, and no
  risk of a comment ever being mistaken for a live literal.
* Docstrings *are* string constants and would otherwise be indistinguishable
  from a real hardcoded literal by a naive substring search; the AST lets us
  identify them precisely (the first statement of a module/class/function
  body) instead of guessing from indentation or triple-quote position.

Two rules:

Rule 1 -- no hardcoded path construction: a string constant that is a direct
operand of a ``/`` (path-join) binary operation, whose value matches a name
``layout.py`` already owns. Allowed file: ``layout.py`` only.

Rule 2 -- no hardcoded default state dir: a string constant containing
``.var/charlie-work`` (either slash spelling), unless it is a docstring.
Allowed files: ``layout.py`` and ``config.py`` (config.py legitimately holds
path *defaults* for its dataclasses; a follow-up PR replaces several of them
with sentinels that defer to layout.py).

The well-known name set for Rule 1 is derived from ``layout`` itself -- the
explicitly-enforced ``_ENFORCED_FILENAMES`` tuple, plus the explicitly-enforced
``_ENFORCED_DIRNAMES`` tuple -- rather than hand-copied here. A hand-maintained
duplicate list is exactly the brittleness this whole refactor exists to remove.

Note what Rule 1 deliberately does *not* cover: the generic per-repo
subdirectory names (``issues``, ``prs``, ``logs``, ``dispatches``,
``reviews``, ``notify``) and the generic fleet filename ``config.yaml``.
Those are already exposed as members on ``paths.RuntimePaths`` (or, for
``config.yaml``, are too generic to match safely by exact string), so the
correct fix at a site spelling ``root / "prs"`` is to use ``paths.prs`` -- not
to swap the literal for ``layout.PRS_DIRNAME``, which adds an indirection while
preventing nothing. Flagging them here would force an allowlist for every
legitimate composition, which is precisely the hand-maintained list this test
is supposed to eliminate. ``layout`` decides which names are enforced (via
``_ENFORCED_FILENAMES`` and ``_ENFORCED_DIRNAMES``); this test only reads that
decision.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

from charlie_work import layout

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src" / "charlie_work"

# Rule 2 substrings: the hardcoded default state-dir literal, in both slash
# spellings (a Windows-authored source file might spell it either way).
_RULE2_SUBSTRINGS = (".var/charlie-work", ".var\\charlie-work")

# Rule 1: only layout.py itself may compose a well-known name via `/` -- it is
# the single source of truth every other module should import from instead.
_RULE1_ALLOWED_BASENAMES = frozenset({"layout.py"})

# Rule 2: layout.py (defines DEFAULT_STATE_DIR) and config.py (dataclass
# defaults) are the only files allowed to hold the literal.
_RULE2_ALLOWED_BASENAMES = frozenset({"layout.py", "config.py"})


def _well_known_path_names() -> frozenset[str]:
    """Derive Rule 1's comparison set directly from ``layout``'s own constants.

    Two sources, both read from ``layout`` so nothing is hand-copied here:

    * ``layout._ENFORCED_FILENAMES`` -- the filenames ``layout`` itself
      declares dangerous to re-spell. Deliberately narrower than the full
      ``*_FILENAME`` set: ``config.yaml`` is too generic to enforce safely by
      exact string membership;
    * ``layout._ENFORCED_DIRNAMES`` -- the directory names ``layout`` itself
      declares dangerous to re-spell (``worktrees`` is the one that cost 74
      uncollected worktrees in production).

    Names *not* in those tuples are intentionally absent; see the module
    docstring for why enforcing them would require an allowlist.
    """
    names: set[str] = set(layout._ENFORCED_FILENAMES)
    names.update(layout._ENFORCED_DIRNAMES)
    return frozenset(names)


WELL_KNOWN_PATH_NAMES = _well_known_path_names()


@dataclass(frozen=True)
class Violation:
    """One occurrence of a re-spelled well-known path literal."""

    rule: str
    filename: str
    lineno: int
    literal: str

    def __str__(self) -> str:
        return f"{self.filename}:{self.lineno}: [{self.rule}] {self.literal!r}"


def _docstring_constant_ids(tree: ast.AST) -> set[int]:
    """Return ``id()`` of every ``Constant`` node that is a real docstring.

    A docstring is precisely the first statement of a module/class/function
    body, when that statement is a bare string-literal expression -- detected
    positionally via the AST (``body[0]``), not by guessing from indentation
    or triple-quote spelling. A second bare string literal later in the same
    body is *not* a docstring under this rule (Python itself only recognizes
    ``__doc__`` at position 0), and must still be checked.
    """
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                ids.add(id(body[0].value))
    return ids


def find_violations(
    source: str,
    filename: str,
    *,
    well_known_names: frozenset[str] = WELL_KNOWN_PATH_NAMES,
) -> list[Violation]:
    """Scan ``source`` (as if it were ``filename``) for re-spelled path literals.

    Plain function over a source string -- no filesystem access -- so it is
    directly unit-testable against small in-memory samples, independent of
    whatever the real source tree currently looks like.
    """
    tree = ast.parse(source, filename=filename)
    docstring_ids = _docstring_constant_ids(tree)
    basename = Path(filename).name

    violations: list[Violation] = []

    if basename not in _RULE1_ALLOWED_BASENAMES:
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                for operand in (node.left, node.right):
                    if (
                        isinstance(operand, ast.Constant)
                        and isinstance(operand.value, str)
                        and operand.value in well_known_names
                    ):
                        violations.append(
                            Violation("rule1", filename, operand.lineno, operand.value)
                        )

    if basename not in _RULE2_ALLOWED_BASENAMES:
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in docstring_ids
                and any(substring in node.value for substring in _RULE2_SUBSTRINGS)
            ):
                violations.append(Violation("rule2", filename, node.lineno, node.value))

    return violations


def _iter_src_files() -> list[Path]:
    return sorted(SRC_ROOT.rglob("*.py"))


def _scan_all_src_files() -> list[Violation]:
    violations: list[Violation] = []
    for path in _iter_src_files():
        source = path.read_text(encoding="utf-8")
        rel = path.relative_to(REPO_ROOT).as_posix()
        violations.extend(find_violations(source, rel))
    return violations


def _format_violations(violations: list[Violation]) -> str:
    lines = "\n".join(f"  {violation}" for violation in violations)
    return f"{len(violations)} violation(s):\n{lines}"


@pytest.fixture(scope="module")
def all_src_violations() -> list[Violation]:
    return _scan_all_src_files()


# ---------------------------------------------------------------------------
# checker teeth: prove find_violations can actually detect a violation
# ---------------------------------------------------------------------------
#
# A source-scan test that cannot fail is worthless. These exercise the
# checker directly against small dirty/clean samples, independent of the
# real source tree, so the enforcement below is proven to have teeth even if
# every current src/ file happened to be clean.


def test_find_violations_flags_rule1_hardcoded_path_join() -> None:
    dirty = (
        'from pathlib import Path\n\ndef f(root: Path) -> Path:\n    return root / "worktrees"\n'
    )
    violations = find_violations(dirty, "dirty_example.py")
    assert len(violations) == 1
    assert violations[0].rule == "rule1"
    assert violations[0].literal == "worktrees"
    assert violations[0].lineno == 4


def test_find_violations_flags_rule2_hardcoded_default_state_dir() -> None:
    dirty = 'STATE_DIR = ".var/charlie-work"\n'
    violations = find_violations(dirty, "dirty_example.py")
    assert len(violations) == 1
    assert violations[0].rule == "rule2"
    assert violations[0].literal == ".var/charlie-work"
    assert violations[0].lineno == 1


def test_find_violations_flags_rule2_backslash_spelling() -> None:
    dirty = 'STATE_DIR = ".var\\\\charlie-work"\n'
    violations = find_violations(dirty, "dirty_example.py")
    assert len(violations) == 1
    assert violations[0].rule == "rule2"


def test_find_violations_is_silent_on_clean_source() -> None:
    clean = (
        "from pathlib import Path\n"
        "\n"
        "from charlie_work import layout\n"
        "\n"
        "def f(root: Path) -> Path:\n"
        "    return layout.worktrees_dir(root)\n"
    )
    assert find_violations(clean, "clean_example.py") == []


def test_find_violations_exempts_rule2_hit_inside_docstring() -> None:
    documented = (
        'def f(state_dir):\n    """Args:\n        state_dir: e.g. .var/charlie-work\n    """\n'
    )
    assert find_violations(documented, "clean_example.py") == []


def test_find_violations_does_not_exempt_non_docstring_string_literal() -> None:
    """Only ``body[0]`` is exempt -- a later bare string literal is not a docstring.

    Guards against a checker that (incorrectly) treats every string literal
    inside a function as "documentation" just because a real docstring
    happens to precede it.
    """
    not_actually_a_docstring = 'def f():\n    """Real docstring."""\n    ".var/charlie-work"\n'
    violations = find_violations(not_actually_a_docstring, "clean_example.py")
    assert len(violations) == 1
    assert violations[0].literal == ".var/charlie-work"
    assert violations[0].lineno == 3


def test_find_violations_respects_rule1_layout_allowlist() -> None:
    dirty = 'root / "worktrees"\n'
    assert find_violations(dirty, "layout.py") == []


def test_find_violations_respects_rule2_allowlist() -> None:
    dirty = 'STATE_DIR = ".var/charlie-work"\n'
    assert find_violations(dirty, "config.py") == []
    assert find_violations(dirty, "layout.py") == []


def test_well_known_path_names_is_derived_and_nonempty() -> None:
    """Sanity: the derived set must actually contain layout.py's own names.

    The comparison set is read from ``layout._ENFORCED_FILENAMES`` and
    ``layout._ENFORCED_DIRNAMES`` (see ``_well_known_path_names``), which
    reference the underlying constants by Python identifier. A rename of one
    of those constants that is not applied inside the tuple raises a
    ``NameError`` at import time rather than silently emptying the set, so
    this test does not need to guard against that mode. What it does guard
    against is the tuples themselves being emptied or having these specific
    entries dropped -- which would silently narrow the comparison set and
    turn the enforcement tests below into a no-op.
    """
    assert layout.SUPERVISOR_LOCK_FILENAME in WELL_KNOWN_PATH_NAMES
    assert layout.WORKTREES_DIRNAME in WELL_KNOWN_PATH_NAMES
    assert ".var" in WELL_KNOWN_PATH_NAMES


def test_generic_fleet_config_filename_is_not_rule1_enforced() -> None:
    """The bare filename ``config.yaml`` is too generic for exact-match Rule 1.

    ``layout.GLOBAL_CONFIG_FILENAME`` (``config.yaml``) names the fleet-wide
    config layer under ``fleet_dir()``. Because Rule 1 matches by exact string
    membership, any unrelated ``some_dir / "config.yaml"`` elsewhere in the
    package would be flagged. The right fix at such a site is not to import
    ``layout.GLOBAL_CONFIG_FILENAME`` — that would falsely assert fleet-global
    identity for a different file — so Rule 1 must not enforce this name.
    """
    assert layout.GLOBAL_CONFIG_FILENAME == "config.yaml"
    assert layout.GLOBAL_CONFIG_FILENAME not in WELL_KNOWN_PATH_NAMES


def test_generic_runtimepaths_dirnames_are_not_enforced() -> None:
    """Pin the deliberate *exclusion* of names ``RuntimePaths`` already owns.

    Widening Rule 1 to these would flag every legitimate ``root / "prs"`` in
    the package and force an allowlist -- the hand-maintained list this test
    exists to avoid. The right fix at such a site is ``paths.prs``, which no
    scan can spot, so this must stay a review concern rather than become a
    lint. Asserted rather than merely commented so re-widening the set is a
    visible, deliberate act with a failing test attached.
    """
    for dirname in (
        layout.ISSUES_DIRNAME,
        layout.PRS_DIRNAME,
        layout.LOGS_DIRNAME,
        layout.DISPATCHES_DIRNAME,
        layout.REVIEWS_DIRNAME,
        layout.NOTIFY_DIRNAME,
    ):
        assert dirname not in WELL_KNOWN_PATH_NAMES, (
            f"{dirname!r} became enforced; either add it to RuntimePaths-aware "
            "review guidance or remove it from layout._ENFORCED_DIRNAMES"
        )


def test_var_dirname_is_the_sole_guard_for_generic_tailed_paths() -> None:
    """``.var`` must stay enforced: it is the only name catching some real shapes.

    ``.var`` looks like the most droppable entry in ``_ENFORCED_DIRNAMES`` -- it
    is a single short token, and in a composition like
    ``root / ".var" / "charlie-work" / "worktrees"`` the ``worktrees`` operand
    already trips Rule 1, making ``.var`` look redundant.

    It is not. When the composition ends in names that are *deliberately*
    unenforced (see :func:`test_generic_runtimepaths_dirnames_are_not_enforced`),
    ``.var`` is the only operand left to catch it. The sample below is the
    verbatim pre-refactor body of ``worktree._default_reviews_dir`` -- one of the
    two functions whose divergence caused the 74-uncollected-worktrees bug -- and
    with ``.var`` removed from the set it scans completely clean.

    Asserted as a differential (with vs. without) rather than as plain
    membership, because membership alone cannot distinguish an enforced name
    that does real work from one that is inert.
    """
    pre_refactor_reviews_dir = 'repo_root / ".var" / "charlie-work" / "dispatches" / "reviews"'
    narrowed = frozenset(WELL_KNOWN_PATH_NAMES - {layout._VAR_DIRNAME})

    assert find_violations(pre_refactor_reviews_dir, "probe.py"), (
        "the enforced set no longer catches the pre-refactor _default_reviews_dir "
        "spelling; Rule 1 has a hole"
    )
    assert not find_violations(pre_refactor_reviews_dir, "probe.py", well_known_names=narrowed), (
        "dropping '.var' no longer loses coverage -- if another enforced name now "
        "catches this shape, update this test rather than deleting it"
    )


def test_var_dirname_does_not_false_positive_on_similar_tokens() -> None:
    """``.var`` is matched exactly, as a ``/`` operand -- never as a substring.

    The counterpart to the test above: a short enforced token is only safe to
    enforce if it cannot fire on unrelated code. Rule 1 compares
    ``operand.value in well_known_names`` (exact set membership) on a string
    that is a direct operand of a path-join, so none of these are violations.
    A future rewrite of the scan as a text/substring search would flag all of
    them and drown the guard in noise -- which is how a guard test ends up
    deleted instead of fixed.
    """
    for clean_sample in (
        'root / ".variables"',  # longer dirname that merely starts with .var
        'p = "somewhere/.various/x"',  # substring hit inside an unrelated literal
        'dirname = ".var"',  # the exact token, but not a path-join operand
        'x = cfg.var / "thing"',  # attribute named var, not a string literal
    ):
        assert find_violations(clean_sample, "probe.py") == [], (
            f"false positive on {clean_sample!r}"
        )


# ---------------------------------------------------------------------------
# real source-tree enforcement
# ---------------------------------------------------------------------------


def test_no_rule1_hardcoded_well_known_path_segments(
    all_src_violations: list[Violation],
) -> None:
    """No `/`-operand elsewhere in src/charlie_work may re-spell a layout.py name."""
    violations = [v for v in all_src_violations if v.rule == "rule1"]
    assert not violations, _format_violations(violations)


def test_no_rule2_hardcoded_default_state_dir(all_src_violations: list[Violation]) -> None:
    """No non-docstring literal outside layout.py/config.py may hardcode the default state dir."""
    violations = [v for v in all_src_violations if v.rule == "rule2"]
    assert not violations, _format_violations(violations)
