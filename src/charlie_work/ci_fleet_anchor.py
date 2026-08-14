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
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .subprocess_runner import RunResult, run_captured

__all__ = [
    "CiFleetProvenanceSnapshot",
    "ci_fleet_provenance_snapshot",
    "declared_ci_fleet_root",
    "repo_root",
]

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


#: Timeout for the sibling ``git`` probes. A local rev-parse / porcelain is
#: sub-second in practice; this is a backstop against a wedged index lock --
#: same value as ``dirty_tree._STATUS_TIMEOUT_SECONDS``.
_PROVENANCE_TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class CiFleetProvenanceSnapshot:
    """What ``ci_fleet`` the supervisor actually imported, recorded once at startup (#954).

    The live supervisor imports ``ci_fleet`` through an editable ``.pth`` that
    prepends the sibling repo's ``src`` to ``sys.path``, so it runs whatever is
    *saved* in that working tree -- committed or not. ``declared_ci_fleet_root``
    and ``venv_anchor`` check that the install has not been repointed, but
    neither records what the running process actually loaded. This snapshot
    closes that observability gap: it does not *prevent* anything, it makes the
    coupling *attributable* when something breaks. See issue #954's "Accept the
    coupling and instrument it" option -- the cheap, do-now half that makes the
    real fix's absence visible.

    Fields:
    - ``ci_fleet_file``: ``ci_fleet.__file__`` -- where the module actually
      loaded from. ``None`` if ``ci_fleet`` could not be imported.
    - ``sibling_root``: the sibling checkout root (``declared_ci_fleet_root()
      .parent``), or ``None`` if no local path dependency is declared /
      resolvable (e.g. running from a worktree that is not a sibling of the
      real checkout -- the same abstention as ``declared_ci_fleet_root``).
    - ``sibling_head``: ``git rev-parse HEAD`` in the sibling, or ``None`` if
      the probe could not run.
    - ``sibling_branch``: ``git rev-parse --abbrev-ref HEAD`` in the sibling,
      or ``None`` if the probe could not run.
    - ``sibling_dirty``: whether the sibling's tracked working tree differs
      from HEAD (``git status --porcelain --untracked-files=no`` non-empty).
      ``None`` if the probe could not run.
    - ``error``: a structural failure message (git probe failure or import
      error), or ``None``. A ``None`` sibling_root with ``error=None`` is an
      abstention, not a failure -- same vocabulary as ``declared_ci_fleet_root``.
    """

    ci_fleet_file: str | None
    sibling_root: str | None
    sibling_head: str | None
    sibling_branch: str | None
    sibling_dirty: bool | None
    error: str | None = None


def ci_fleet_provenance_snapshot(
    *,
    run_command: Callable[..., RunResult] | None = None,
    timeout: int = _PROVENANCE_TIMEOUT_SECONDS,
) -> CiFleetProvenanceSnapshot:
    """Capture the resolved ``ci_fleet`` import location and sibling repo git state.

    Returns a value, never raises -- a startup instrumentation probe must not
    break the supervisor's entry path. ``run_command`` is injectable for tests
    (defaults to :func:`run_captured`); production callers pass nothing.

    Abstention mirrors :func:`declared_ci_fleet_root`: when no local path
    dependency is declared or the resolved root does not exist (e.g. a
    worktree that is not a sibling of the real checkout), the sibling fields
    are ``None`` and ``error`` is ``None`` -- the honest answer is "from here
    we cannot tell", not "something is wrong".
    """
    if run_command is None:
        run_command = run_captured

    ci_fleet_file: str | None = None
    try:
        import ci_fleet  # noqa: PLC0415 -- deferred so a missing dep does not crash import

        ci_fleet_file = getattr(ci_fleet, "__file__", None)
    except Exception as exc:  # noqa: BLE001 -- probe must report, never crash
        return CiFleetProvenanceSnapshot(
            ci_fleet_file=None,
            sibling_root=None,
            sibling_head=None,
            sibling_branch=None,
            sibling_dirty=None,
            error=f"import ci_fleet raised {type(exc).__name__}: {exc}",
        )

    declared_src = declared_ci_fleet_root()
    if declared_src is None:
        # Abstention, not failure: same condition declared_ci_fleet_root
        # abstains on (no declared path source, or unresolvable from a
        # worktree). ci_fleet_file is still recorded -- it is the one fact
        # available without the sibling.
        return CiFleetProvenanceSnapshot(
            ci_fleet_file=ci_fleet_file,
            sibling_root=None,
            sibling_head=None,
            sibling_branch=None,
            sibling_dirty=None,
        )

    sibling = declared_src.parent
    try:
        head_res = run_command(["git", "rev-parse", "HEAD"], cwd=sibling, timeout_seconds=timeout)
        branch_res = run_command(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=sibling,
            timeout_seconds=timeout,
        )
        status_res = run_command(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=sibling,
            timeout_seconds=timeout,
        )
    except Exception as exc:  # noqa: BLE001 -- probe must report, never crash startup
        return CiFleetProvenanceSnapshot(
            ci_fleet_file=ci_fleet_file,
            sibling_root=str(sibling),
            sibling_head=None,
            sibling_branch=None,
            sibling_dirty=None,
            error=f"git probe raised {type(exc).__name__}: {exc}",
        )

    sibling_head = head_res.stdout.strip() if head_res.ok else None
    sibling_branch = branch_res.stdout.strip() if branch_res.ok else None
    sibling_dirty = bool(status_res.stdout.strip()) if status_res.ok else None

    error: str | None = None
    if not head_res.ok or not branch_res.ok or not status_res.ok:
        failed = [
            f"git {cmd[1]}: {res.error or res.stderr or 'exit %s' % res.returncode}"
            for cmd, res in (
                (["git", "rev-parse", "HEAD"], head_res),
                (["git", "rev-parse", "--abbrev-ref", "HEAD"], branch_res),
                (["git", "status", "--porcelain"], status_res),
            )
            if not res.ok
        ]
        error = "; ".join(failed)

    return CiFleetProvenanceSnapshot(
        ci_fleet_file=ci_fleet_file,
        sibling_root=str(sibling),
        sibling_head=sibling_head,
        sibling_branch=sibling_branch,
        sibling_dirty=sibling_dirty,
        error=error,
    )
