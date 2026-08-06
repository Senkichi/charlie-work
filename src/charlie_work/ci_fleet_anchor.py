"""The provider half of ``ci_fleet``'s provenance check.

``ci_fleet`` refuses to trust its own install artifacts to tell it which tree it
was imported from, and it cannot look the answer up itself: the boundary is
one-way (``ci_fleet`` may not import ``charlie_work``, enforced by
``tests/test_structural_invariants.py``), so the expectation has to be
*installed* by us. Same injection pattern as ``observability.set_event_sink``,
and installed in the same place for the same reason -- see
``instrumentation.py``'s seam block.

Why ``pyproject.toml`` and not something easier to read
-------------------------------------------------------

The check compares where ``ci_fleet`` *actually* loaded from against where we
*declare* it should load from. That is only a check if the two sides have
independent origins.

``direct_url.json`` and ``_editable_impl_ci_fleet.pth`` both fail that test.
They are written by the same ``uv pip install -e`` invocation, so they move
together: an install repointed at a different tree carries both, and reading
either one back would confirm whatever the install did. Two artifacts written
by one command cannot cross-check each other.

``pyproject.toml`` is independent. It lives in this repo, it is the input to
the install rather than its output, it is under version control where a change
to it is reviewable, and no ``.pth`` file can move it. That makes it the only
declaration on this box that can disagree with a wrong install.

What this does not prove
------------------------

A path comparison, which is the weaker of the two proofs available in general.
The stronger one -- an intrinsic, version-distinguishing feature of the loaded
module -- needs the supervisor to know which version *ought* to be loaded, and
it has no such expectation. So this catches "loaded from the wrong directory",
not "loaded the wrong code from the right directory". ``ci_fleet``'s
``provenance`` module says the same thing about its half; neither side should
be described as more than it is.

Why a non-existent declared root abstains instead of failing
------------------------------------------------------------

The declaration is a *relative* path -- ``../ci_runners`` -- so it only means
what it says when resolved from the checkout the install was actually run
from. This repo runs agents in git worktrees, and there are routinely twenty of
them under ``.claude/worktrees/`` and ``.var/charlie-work/worktrees/``. Resolved
from one of those, ``../ci_runners`` names a sibling of the *worktree*, which
does not exist.

That case must not be reported as a disagreement. ``mismatch`` sets
``blocks_actuation``, so an anchor that answered "expected
``.claude/worktrees/ci_runners``, got ``repos/ci_runners``" would refuse to
actuate the entire fleet from every worktree -- converting a check into an
outage, for a condition that is not evidence of anything being wrong.

So a declared root that does not exist on disk yields ``None`` -> ``no_anchor``
-> abstain-and-proceed, which is the honest answer: from here we cannot tell
where ci_fleet ought to load from. Worktrees that *are* siblings of the real
checkout (``repos/charlie-work-*``) still resolve correctly and are still
checked. This was found by running the anchor from a worktree, not by reasoning
about it.

Note the asymmetry this preserves: a genuinely repointed install still fails
loudly, because ``pyproject.toml`` read from the main checkout still names a
directory that exists and the comparison still happens.

Raising rather than returning ``None``
--------------------------------------

Structural failures are deliberately left to propagate. ``check_provenance``
wraps the anchor call and reports the exception type and message as the
``no_anchor`` detail, so a missing ``pyproject.toml`` or a renamed table
surfaces as ``provenance anchor raised KeyError: 'uv'`` rather than as an
undifferentiated "could not determine". ``None`` is reserved for the two cases
that are real answers rather than breakage: ci-fleet not being declared as a
local path dependency at all, and the resolved root not existing.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

__all__ = ["declared_ci_fleet_root", "repo_root"]

#: The dependency name as it appears in ``[tool.uv.sources]``. This has to be
#: spelled out because it *is* the identity of the thing being located; there
#: is nothing to derive it from that would not itself need naming.
_SOURCE_NAME = "ci-fleet"


def repo_root() -> Path:
    """This repository's root, derived from this module's own location.

    ``src/charlie_work/ci_fleet_anchor.py`` -> ``charlie_work`` -> ``src`` ->
    the repo root. ``resolve()`` first, because this file is reached through an
    editable install and worktree ``.venv`` entries on this box are Windows
    junctions -- an unresolved parent walk would climb the junction's path
    rather than the real one.
    """
    return Path(__file__).resolve().parent.parent.parent


def declared_ci_fleet_root() -> Path | None:
    """The directory ``ci_fleet`` is declared to be imported from.

    Returns the ``src`` directory, because that is what ``ci_fleet`` compares
    against: its own ``import_root()`` is the parent of the ``ci_fleet``
    package directory, i.e. ``<checkout>/src``.

    ``None`` in the two cases where there is genuinely nothing to compare
    against, both of which are abstentions rather than failures: ``ci-fleet``
    is not declared as a local path dependency, or the declared root does not
    exist because this module is running from a worktree that is not a sibling
    of the real checkout. See the module docstring -- reporting the latter as a
    disagreement would block fleet actuation from every worktree.
    """
    root = repo_root()
    with (root / "pyproject.toml").open("rb") as handle:
        pyproject = tomllib.load(handle)
    source = pyproject["tool"]["uv"]["sources"][_SOURCE_NAME]
    declared = source.get("path")
    if declared is None:
        return None
    expected = (root / declared / "src").resolve()
    if not expected.is_dir():
        return None
    return expected
