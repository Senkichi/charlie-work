"""Shared base for GitHub capability collaborators.

Part of the Track 2 god-object paydown (issue #1585; design doc
``docs/design/2026-09-03-github-class-mikado-graph-and-protocol-segmentation.md``,
Section 3.3). Every capability collaborator (``Comments``, ``Labels``,
``Checks``, ...) is constructed with a back-reference to the owning
``GitHub`` instance and forwards attribute lookups it does not itself define
back to that owner.

This is one half of the bounded, bidirectional resolution the delegation
seam relies on:

1. **owner -> collaborator**: an explicit ``_ROUTES`` table on ``GitHub``
   (no ``__getattr__`` on the owner), so that direction always terminates.
2. **collaborator -> owner**: ``__getattr__`` here, forwarding to
   ``self._owner``. A moved method body still says things like
   ``self.run(...)`` or ``self._list_cache``; on a collaborator instance
   that resolves through this ``__getattr__`` to the real owner attribute
   (or an owner-side delegate). This also terminates: the owner has no
   ``__getattr__`` of its own to recurse into, so lookup either finds a
   real attribute on the owner or raises ``AttributeError`` normally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from charlie_work.github import GitHub


# Moved from ``github.py`` (Track 2, issue #1588; design doc Section 5, L04).
# All six ``Checks`` members constructed in this leaf perform a real runtime
# ``isinstance(result, GitHubRunResult)`` check -- not just a type annotation
# -- so a ``TYPE_CHECKING``-only import (the pattern ``repo_meta.py``/
# ``pull_requests.py``/``merge_branch.py`` already use for their own
# not-yet-populated sub-protocols) cannot work here. Nor can it stay defined
# in ``github.py`` and be imported normally: ``github.py`` imports
# ``github_capabilities`` (and therefore ``checks.py``) before its own
# ``GitHubRunResult`` definition, so a plain top-level import from
# ``charlie_work.github`` into ``checks.py`` would hit a partially
# initialized module. ``_base.py`` has no dependency on ``github.py`` at
# runtime (only the ``TYPE_CHECKING``-only ``GitHub`` import above), so it is
# the one place every collaborator module can already reach unconditionally
# -- the same role it plays for ``CapabilityCollaborator`` itself.
#
# Re-exported through ``github_capabilities/__init__.py`` back into
# ``github.py``'s import block, mirroring the existing ``GitHubError``
# re-export (``github.py`` line ~38): identity must stay single because the
# class is ``isinstance``-checked and constructed pervasively both inside
# ``github.py`` (``commit``, ``pr_diff``, ``_pr_checks_fallback``, and more)
# and by 12+ other modules/tests that import it from ``charlie_work.github``.
# This is a disclosed design-gap resolution (design doc Section 3.3 covers
# only ``self.<attr>`` forwarding, not bare-global runtime symbols in moved
# bodies) that later leaves L05 (``RepoMeta.commit``), L06
# (``PullRequests.pr_ready``), and L08 (``MergeBranch.pr_close``/
# ``pr_reopen``/``push_empty_commit``) will hit identically -- they should
# import ``GitHubRunResult`` from here too rather than re-deriving a second
# answer to the same problem.
@dataclass(frozen=True)
class GitHubRunResult:
    """Result of a ``gh`` invocation when ``allow_failure=True``.

    Errors stay as values: callers check ``ok`` and ``error`` and only use
    ``value`` when ``ok`` is True. ``value`` is the parsed JSON (when
    ``json_output=True``) or the captured stdout (when ``json_output=False``).
    """

    ok: bool
    returncode: int
    stdout: str
    stderr: str
    value: Any | None = None
    error: str | None = None


class CapabilityCollaborator:
    """Base class for GitHub capability collaborators.

    Subclasses (``Comments``, ``Labels``, ``Checks``, ``RepoMeta``,
    ``PullRequests``, ``Issues``, ``MergeBranch``, ``Transport``) are
    otherwise empty in L01 -- no method bodies have moved yet. Later Mikado
    leaves add methods directly to a subclass's own body.

    ``__init__``/``__getattr__`` live here, not duplicated across the eight
    subclasses, so every collaborator gets identical construction and
    forwarding behavior. This also keeps ``vars(subclass)`` free of anything
    but the subclass's *own* declared members -- what ``github.py``'s
    ``_ROUTES`` construction inspects -- with no L01-specific special-casing
    needed to keep that table empty before any method has moved.
    """

    def __init__(self, owner: GitHub) -> None:
        self._owner = owner

    def __getattr__(self, name: str) -> Any:
        # __getattr__ only fires for names normal attribute lookup could not
        # resolve. `_owner` itself is set in __init__ via plain assignment,
        # so it lives in the instance __dict__ and normal lookup finds it
        # without ever reaching here -- except in the defensive case where an
        # instance was constructed without __init__ running (e.g.
        # object.__new__, copy/pickle edge cases). Guard both dunder probes
        # and a missing `_owner` explicitly so that case raises a clean
        # AttributeError instead of recursing back into this same method.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        if name == "_owner":
            raise AttributeError(name)
        owner = self.__dict__.get("_owner")
        if owner is None:
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r} "
                "(and no owner to forward to)"
            )
        return getattr(owner, name)
