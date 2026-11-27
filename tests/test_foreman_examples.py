"""The shipped foreman examples (issue #3124).

``examples/foremen/migration/`` is the bounded-migration charter an
operator scaffolds from; ``examples/foremen/_shared/issue-style.md`` is the
shared partial its ``instructions.md`` includes. The charter must load
through the real validator and the partial must resolve through the real
renderer — a broken example is what every ``gang new --from`` would copy.
"""

from __future__ import annotations

from _helpers import EXAMPLES_DIR

from charlie_work.foreman.charter import ARMING_MODES, load_charter
from charlie_work.foreman.instructions import load_instructions

FOREMEN = EXAMPLES_DIR / "foremen"
MIGRATION = FOREMEN / "migration"
PLACEHOLDER_REPO = "<owner>/<repo>"


def test_migration_charter_loads() -> None:
    """The example charter passes field and semantic validation."""
    result = load_charter(MIGRATION, registered_repos={PLACEHOLDER_REPO})
    assert result.ok, result.error
    charter = result.value
    assert charter.name == "migration"  # must equal the directory name
    assert charter.repos == (PLACEHOLDER_REPO,)
    # Bounded: no standing example ships, and done_when carries a real argv.
    assert not charter.standing
    assert charter.done_when is not None and charter.done_when.check
    assert charter.arming in ARMING_MODES


def test_shared_partial_resolves_from_migration_instructions() -> None:
    """``instructions.md`` includes ``_shared/issue-style.md`` and the
    rendered text carries the planning rules the operator asked it to."""
    result = load_instructions(EXAMPLES_DIR, MIGRATION)
    assert result.ok, result.error
    rendered = result.value
    for fragment in (
        "wasted worker run",
        "every caller, every test that imports or patches it",
        "A worker never makes a design decision",
        "Read the charter's outcome literally",
        "Never state a number you did not measure",
    ):
        assert fragment in rendered
