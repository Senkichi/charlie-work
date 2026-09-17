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


# Moved from ``github.py`` (Track 2, issue #1590; design doc Section 5, L06).
# ``_LIST_LIMIT`` is referenced as a bare global by ``PullRequests.pr_list``/
# ``merged_pr_list`` (moved below) AND by ``GitHub.issue_list`` (not yet
# moved -- a future Issues-cluster leaf), the same cross-cutting shape that
# put ``GitHubRunResult`` here rather than in a single capability module: no
# capability module "owns" it, so ``_base.py`` -- the one place every
# collaborator module and ``github.py`` itself can already reach without a
# circular import -- is the right home, not ``pull_requests.py``. Re-exported
# through ``github_capabilities/__init__.py`` and re-imported into
# ``github.py`` (still used directly there in ``issue_list``, and read by
# ``reconcile.py`` via ``from .github import _LIST_LIMIT``), mirroring the
# ``GitHubRunResult`` re-export above. This is the same disclosed design-gap
# resolution (design doc Section 3.3 covers only ``self.<attr>`` forwarding,
# not bare-global runtime symbols in moved bodies) that recurs identically
# across leaves; unlike ``GitHubRunResult`` it was not named in this leaf's
# forward-reference comment, but the reasoning is identical.
_LIST_LIMIT = 500


# Moved from ``github.py`` (Track 2, issue #1590; design doc Section 5, L06),
# alongside ``_LIST_LIMIT`` above. ``_is_mutating`` (and its private helper
# chain ``_api_is_mutating``/``_is_graphql_query``/``_graphql_field_value``)
# is referenced as a bare global by ``PullRequests.pr_ready``,
# ``MergeBranch.pr_close``/``pr_reopen`` (moved in L08), ``Transport._run_bool``
# (moved in L09), and ``GitHub.run`` itself -- the one consumer that never
# relocates, since ``run`` is the interception seam and stays on the owner by
# design (design doc Section 3.2). This cross-cutting shape (one helper,
# consumers scattered across every leaf plus the owner) is why it lives here
# rather than in any single capability module -- re-relocating it leaf by leaf
# or importing it sideways from a PR-domain module would just move the same
# problem around. Only ``_is_mutating`` itself is referenced outside this
# chain (by name, from ``github.py``); the three helper functions have no
# consumer beyond ``_is_mutating``'s own body, so only ``_is_mutating`` is
# re-exported through ``github_capabilities/__init__.py`` and re-imported
# into ``github.py``. Bodies are unchanged from their former ``github.py``
# copies.
def _graphql_field_value(args: list[str], field: str) -> str | None:
    """Return the raw value of a `gh api graphql -f/--field name=value` pair.

    Handles detached (`-f query=...`), attached shorthand (`-fquery=...`),
    and `--field=query=...` spellings (#919). Returns `None` if the field is
    absent or its value is missing.
    """
    for i, arg in enumerate(args):
        if arg in ("-f", "--raw-field", "-F", "--field"):
            next_arg = args[i + 1] if i + 1 < len(args) else ""
            if "=" in next_arg and next_arg.split("=", 1)[0] == field:
                return next_arg.split("=", 1)[1]
        elif arg.startswith("-f") and len(arg) > 2:
            rest = arg[2:].lstrip("=")
            if "=" in rest and rest.split("=", 1)[0] == field:
                return rest.split("=", 1)[1]
        elif arg.startswith(("--field=", "--raw-field=")):
            rest = arg.split("=", 1)[1]
            if "=" in rest and rest.split("=", 1)[0] == field:
                return rest.split("=", 1)[1]
    return None


def _is_graphql_query(args: list[str]) -> bool:
    """A `gh api graphql -f query='query { ... }'` is a read-only query.

    `args` is the argv after the leading `gh` token, so a GraphQL call looks
    like `["api", "graphql", "-f", "query=..."]`.

    Fails closed: only an operation that *starts* with the GraphQL `query`
    keyword is treated as read-only. `mutation` or anything unparseable is
    classified as mutating so a stray write never runs under `--dry-run`.
    """
    if len(args) < 2 or args[0] != "api" or args[1] != "graphql":
        return False
    query = _graphql_field_value(args, "query")
    if not query:
        return False
    return query.lstrip()[:5].lower() == "query"


def _api_is_mutating(args: list[str]) -> bool:
    """Classify a `gh api` invocation, for the --dry-run gate.

    `gh api` defaults to GET, so a bare `gh api <path>` is read-only and MUST stay
    runnable under --dry-run — roughly a dozen call sites (rate_limit, commits/{sha},
    check-runs, compare, branches/*/protection, `fleet_registry.py`) depend on that.
    Blanket-denying `api` would turn --dry-run from "observes without mutating" into
    "cannot observe", so the classification keys off whether a method is *named*.

    Structured on flag PRESENCE rather than on enumerating accepted spellings, and
    fails CLOSED when a method flag is present but its value cannot be extracted.
    The previous version enumerated `--method`/`--method=` only and fell through to
    False for everything else, so `-X DELETE` — the form `delete_branch` builds —
    classified as read-only and a --dry-run really deleted PR head branches
    (#914, #917). Enumeration is the wrong shape here: it silently fails open on
    each spelling nobody thought of (`-X`, `-X=`, `-XDELETE`, a trailing `--method`
    with no value).

    Read-only `gh api graphql -f query='query { ... }'` is an exception: it is a
    GraphQL query and must be runnable under `--dry-run` so `fleet status` can
    batch issue dependency/state lookups in a single subprocess (#923).
    """
    if _is_graphql_query(args):
        return False

    for i, arg in enumerate(args):
        if arg in ("-X", "--method"):
            method = args[i + 1] if i + 1 < len(args) else ""
        elif arg.startswith("--method="):
            method = arg.split("=", 1)[1]
        elif arg.startswith("-X"):
            # pflag shorthand accepts an attached value: `-XDELETE` and `-X=DELETE`.
            method = arg[2:].lstrip("=")
        else:
            continue
        # A named-but-unparseable method is not evidence of a read; fail closed.
        return not method or method.upper() not in ("GET", "HEAD")
    # No explicit method. gh switches GET -> POST when request parameters are added
    # ("adding request parameters will automatically switch the request method to
    # POST" -- gh api --help). Prefix-matched, not membership-tested, for the same
    # reason as the method arm: pflag accepts both the detached (`-f title=x`,
    # `--field=labels[]=bug`) and the attached (`-ftitle=x`) spelling, and a
    # membership test sees only the detached one (#919). `--field`/`--raw-field`/
    # `--input` are prefixes rather than exact matches so the bare and `=` forms
    # collapse into one condition.
    param_prefixes = ("--raw-field", "--field", "--input")
    return any(arg.startswith(param_prefixes) or arg[:2] in ("-f", "-F") for arg in args)


# MERGED_PR_LIST_FIELDS moved on from here to
# github_capabilities/pull_requests.py (Track 2, issue #1613; design doc
# Section 5, L06b), alongside merged_prs_for_issue -- its one remaining
# ``GitHub``-side bare-global consumer once that method moved too.
# Transport.validate_field_lists (below) now imports it from pull_requests.py
# instead of from here. See pull_requests.py for the full field-contract
# rationale (unchanged).

# Moved from ``github.py`` (Track 2, issue #1593; design doc Section 5, L09).
# ``RUN_LIST_FIELDS`` is referenced as a bare global by the module-level
# ``cancel_superseded_runs`` (a ``GitHubLike``-typed helper function, not a
# ``GitHub`` member, so it stays in ``github.py`` untouched by this leaf) AND
# by ``Transport.validate_field_lists`` (moved below) -- the same
# staying-plus-moving-consumer shape as ``MERGED_PR_LIST_FIELDS`` (which moved
# on from here to ``github_capabilities/pull_requests.py`` in L06b; see the
# comment above).
# Re-exported through ``github_capabilities/__init__.py`` and re-imported
# into ``github.py`` (still used directly there in ``cancel_superseded_runs``).
RUN_LIST_FIELDS = "databaseId,status,createdAt,headBranch"


def _is_mutating(args: list[str]) -> bool:
    if not args:
        return False
    text = " ".join(args)
    # `gh api` defaults to GET and is read-only unless a mutating method is given.
    # run() passes args without the leading "gh" token.
    if text.startswith("api"):
        return _api_is_mutating(args)
    readonly_prefixes = (
        "issue list",
        "issue view",
        "pr list",
        "pr view",
        "pr diff",
        "pr checks",
        "label list",
        "auth status",
    )
    return not any(text.startswith(prefix) for prefix in readonly_prefixes)


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
