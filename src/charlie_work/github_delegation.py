"""Capability delegation seam for the ``GitHub`` facade (Track 2, issue #1585).

``GitHub`` is being decomposed into capability collaborators living in
``charlie_work.github_capabilities`` (design doc
``docs/design/2026-09-03-github-class-mikado-graph-and-protocol-segmentation.md``,
Section 3.3). This module is pure infrastructure: it derives a routing table
from what the collaborator classes actually declare and builds one forwarding
delegate per routed name for ``github.py`` to install onto ``GitHub``. In L01
every collaborator class is still empty, so ``_ROUTES`` is empty and
installing it is a no-op -- ``GitHub``'s lexical member surface (and
therefore its member_count ratchet entry) is unchanged. Later leaves move one
method's *body* at a time out of ``GitHub`` into a collaborator class; the
moment a method leaves ``GitHub``, this same machinery (unchanged) picks it
up and re-installs it as a forwarding delegate automatically.

Lives in its own module rather than inline in ``github.py`` because
``github.py`` is over the file-size ratchet's high-water mark (issue #1442,
``tests/test_file_size_ratchet.py``): new code may not land in an over-cap
monolith. ``install_delegates`` takes the owner class as a parameter instead
of importing ``GitHub`` directly, so this module has no runtime dependency on
``github.py`` at all (only a ``TYPE_CHECKING``-only one, for the delegate's
``self`` annotation) -- there is nothing to order around a circular import.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

from .github_capabilities import (
    Checks,
    Comments,
    Issues,
    Labels,
    MergeBranch,
    PullRequests,
    RepoMeta,
    Transport,
)

if TYPE_CHECKING:
    from .github import GitHub

# `_COLLABORATORS` is the one declarative seed: which collaborator classes
# exist and what owner attribute each is stored under. It is not a list of
# *GitHubLike members* (that would be the hardcoded list CLAUDE.md rule 9
# forbids) -- `_ROUTES`/`_SIGNATURE_SOURCE` below are derived from it by
# introspecting the collaborator classes, never hand-typed per member.
_COLLABORATORS: tuple[tuple[str, type], ...] = (
    ("_comments", Comments),
    ("_labels", Labels),
    ("_checks", Checks),
    ("_repo_meta", RepoMeta),
    ("_pull_requests", PullRequests),
    ("_issues", Issues),
    ("_merge_branch", MergeBranch),
    ("_transport", Transport),
)


def _routable_members(collab_cls: type) -> Iterator[tuple[str, Callable[..., Any]]]:
    """Yield ``(name, function)`` for every callable ``collab_cls`` declares directly.

    Iterates ``vars(collab_cls)`` rather than ``dir()``/``inspect.getmembers``
    so only names the collaborator class itself declares are seen -- the
    ``__init__``/``__getattr__`` inherited from ``CapabilityCollaborator``
    never appear here. That is what keeps ``_ROUTES`` empty while every
    collaborator class is still an empty subclass (L01), with no
    L01-specific special-casing: the same introspection that yields nothing
    now yields the real routing table once a leaf adds a method to a
    collaborator class.

    Dunder names are excluded (never GitHubLike/internal members to route);
    single-underscore internals (``_run_bool``, ``_max_retries``, etc.) are
    included deliberately -- moved method bodies still call them by name via
    ``self.<name>``, and the owner has no ``__getattr__`` fallback to catch
    them (design doc Section 3.3).
    """
    for name, member in vars(collab_cls).items():
        if name.startswith("__") and name.endswith("__"):
            continue
        if not callable(member):
            continue
        yield name, member


def _build_routes() -> tuple[dict[str, str], dict[str, Callable[..., Any]]]:
    routes: dict[str, str] = {}
    signature_source: dict[str, Callable[..., Any]] = {}
    for collab_attr, collab_cls in _COLLABORATORS:
        for name, member in _routable_members(collab_cls):
            routes[name] = collab_attr
            signature_source[name] = member
    return routes, signature_source


_ROUTES, _SIGNATURE_SOURCE = _build_routes()


def _make_delegate(name: str, collab_attr: str) -> Callable[..., Any]:
    """Build a class-level forwarding delegate for a routed member.

    ``functools.wraps`` + an explicit ``__signature__`` make
    ``inspect.signature(GitHub.<name>)`` return the *source* method's exact
    signature, including its string return annotation (every
    ``github_capabilities`` module starts with
    ``from __future__ import annotations``) -- exactly what the GitHubLike
    conformance test compares
    (``tests/test_githublike_protocol.py::_compatible_signature``). A plain
    ``*args, **kwargs`` delegate would satisfy ``isinstance`` but fail that
    signature comparison (design doc Section 8.3).

    Both the collaborator (``getattr(self, collab_attr)``) and the target
    method (``getattr(collab, name)``) are resolved fresh on every call,
    never cached -- required so ``monkeypatch.setattr(GitHub, "run", ...)``
    and other per-call patches still intercept through the collaborator
    (design doc Section 3.3, the 134-patch-site invariant).
    """

    def _delegate(self: GitHub, *args: Any, **kwargs: Any) -> Any:
        collab = getattr(self, collab_attr)
        return getattr(collab, name)(*args, **kwargs)

    src_fn = _SIGNATURE_SOURCE[name]
    src_fn = getattr(src_fn, "__func__", src_fn)
    functools.wraps(src_fn)(_delegate)
    _delegate.__signature__ = inspect.signature(src_fn)
    return _delegate


def _install_delegates(owner_cls: type) -> None:
    """Install one forwarding delegate per ``_ROUTES`` entry onto ``owner_cls``.

    Skips any name already lexically defined on ``owner_cls``: a method not
    yet moved out of the class body always wins over a delegate for the same
    name, so a partially-completed leaf never shadows a real implementation.

    Takes the owner class as a parameter (rather than importing ``GitHub``
    directly) so this module never needs a runtime import of ``github.py`` --
    ``github.py`` calls ``_install_delegates(GitHub)`` itself once its class
    body is fully defined.
    """
    for name, collab_attr in _ROUTES.items():
        if name in owner_cls.__dict__:
            continue
        setattr(owner_cls, name, _make_delegate(name, collab_attr))
