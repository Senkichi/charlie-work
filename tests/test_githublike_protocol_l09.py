"""Track 2 L09 (issue #1593): Transport capability move regression tests.

New file, not appended to ``test_githublike_protocol.py`` (deliberate test-
layout change, matching every earlier leaf) -- shares the AST/signature
helpers via ``tests/_githublike_protocol_helpers.py``.

L09 is the final Track 2 leaf: it moves the twelve non-``GitHubLike``
internals off ``GitHub`` and onto the ``Transport`` capability collaborator
(design doc Section 3.2/5). Unlike every prior leaf, ``Transport`` is not a
``GitHubLike`` sub-protocol cluster -- there is no ``TransportLike`` Protocol
and no ``isinstance`` conformance claim to make for it. What this file covers
instead: the twelve delegates exist and route correctly, ``run`` (the
134-patch-site interception seam) and ``__post_init__`` are provably
unaffected, retry/timeout constants still flow from ``RuntimeConfig`` through
``run`` via the now-relocated ``_max_retries``/``_retry_base_seconds``/
``_timeout_seconds``, the ``validate_field_lists`` import-depth fix reaches
the real ``charlie_work.config.ConfigError``, and a positive control
demonstrating the L07-established same-collaborator bypass hazard class
(present here as a documented hazard, not as an existing test's bug -- the
census in the L09 PR body found zero existing tests that it actually breaks).
"""

from __future__ import annotations

import ast
import inspect
import subprocess
import time
from pathlib import Path

import pytest

import charlie_work.github as _github_module
from charlie_work.config import ConfigError
from charlie_work.github import GitHub, GitHubLike, _ROUTES
from charlie_work.github_capabilities.transport import Transport
from charlie_work.github_delegation import _build_routes

from _githublike_protocol_helpers import _compatible_signature, _lexical_github_defs

TRANSPORT_MOVED_MEMBERS = (
    "_run_bool",
    "_list_json",
    "_repo_owner_name",
    "_graphql_query",
    "_graphql_issue_states",
    "_graphql_issue_dependencies",
    "_normalize_rest_pr",
    "_pr_checks_fallback",
    "_max_retries",
    "_retry_base_seconds",
    "_timeout_seconds",
    "validate_field_lists",
)

# The members that stay lexically on GitHub after this leaf. Originally three
# (this leaf was written as "the final Track 2 leaf"); Track 2, issue #1613
# (design doc Section 5, L06b) moved ``merged_prs_for_issue`` off GitHub too,
# once its ``linked_issue_number`` dependency had its own neutral home
# (``issue_linking.py``), leaving these two as the true final set. Updated
# here (not in L06b's own test file) because this tuple and the direct-def-
# count test below are this file's own assertions, broken by that leaf.
GITHUB_STAYING_MEMBERS = ("__post_init__", "run")


def test_github_isinstance_githublike(tmp_path: Path) -> None:
    """``GitHub`` must still satisfy ``GitHubLike`` after the L09 move.

    None of the twelve moved names are ``GitHubLike`` protocol members (design
    doc Section 3.2: Transport is not a sub-protocol cluster), so this is a
    pure regression check that removing them did not disturb the protocol's
    structural conformance -- covering the task's "isinstance sites in
    workflow.py/test_githublike_protocol.py" concern for this leaf.
    """
    assert isinstance(GitHub(tmp_path), GitHubLike)


def test_transport_members_are_not_lexical_github_defs() -> None:
    """``GitHub`` must no longer lexically define any of the twelve moved members.

    Track 2 L09 (issue #1593) moved them to ``Transport``; they are now served
    on ``GitHub`` by generated delegates (class-level assignments, not
    ``def``s) carrying ``__wrapped__``.
    """
    lexical = _lexical_github_defs()
    for name in TRANSPORT_MOVED_MEMBERS:
        assert name not in lexical, f"{name} is still a lexical GitHub def"
    for name in TRANSPORT_MOVED_MEMBERS:
        delegate = getattr(GitHub, name)
        assert hasattr(delegate, "__wrapped__"), (
            f"GitHub.{name} does not look like an installed delegate"
        )
        assert delegate.__wrapped__ is vars(Transport)[name], (
            f"GitHub.{name}.__wrapped__ does not point at Transport.{name}"
        )
    # And the non-dunder staying member ("run") is still a real lexical def,
    # not a delegate. ``_lexical_github_defs()`` deliberately excludes
    # dunder-shaped names (including hand-written ones like ``__post_init__``)
    # -- see its own docstring -- so dunders are checked separately below via
    # a direct, unfiltered AST walk in ``test_github_direct_def_count_is_two``.
    for name in GITHUB_STAYING_MEMBERS:
        if name.startswith("__"):
            continue
        assert name in lexical, f"{name} unexpectedly left GitHub's lexical body"
        assert not hasattr(getattr(GitHub, name), "__wrapped__"), (
            f"GitHub.{name} looks like a delegate, but it must stay a real method"
        )


def test_github_direct_def_count_is_two() -> None:
    """L09 dropped ``GitHub`` from 15 to exactly 3 direct defs; Track 2, issue
    #1613 (design doc Section 5, L06b) dropped it once more, to exactly 2
    (``__post_init__``, ``run``), by moving ``merged_prs_for_issue`` off
    ``GitHub`` too. Kept as this file's own test (not moved to L06b's test
    file) because it is this file's own assertion about ``GitHub``'s final
    shape after L09, now updated for the later leaf that revised it.

    Uses a direct, unfiltered AST walk (matching the attachment-contracts
    ``member_count`` ratchet's own counting rule: every direct
    ``FunctionDef``/``AsyncFunctionDef`` child of ``ClassDef.body``, dunders
    included) rather than the shared ``_lexical_github_defs()`` helper, which
    deliberately excludes dunders for its own (protocol-conformance) purpose.
    """
    source = Path(_github_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    (github_cls,) = [
        n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "GitHub"
    ]
    all_defs = {
        child.name
        for child in github_cls.body
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert all_defs == set(GITHUB_STAYING_MEMBERS), all_defs


def test_transport_routes_point_at_the_transport_collaborator() -> None:
    """``_ROUTES`` must route all twelve moved names to ``_transport``."""
    for name in TRANSPORT_MOVED_MEMBERS:
        assert _ROUTES[name] == "_transport"


def test_build_routes_matches_module_level_routes() -> None:
    """``_build_routes()`` must be pure and reproduce the frozen ``_ROUTES``/
    ``_SIGNATURE_SOURCE`` module-level tuple exactly -- the CLI-independent
    check task item 7 requires (never the CLI, per issue #1600).
    """
    routes, signature_source = _build_routes()
    assert routes == _ROUTES
    for name in TRANSPORT_MOVED_MEMBERS:
        assert signature_source[name] is vars(Transport)[name]


def test_transport_members_signature_compatible() -> None:
    """Each moved member's installed ``GitHub`` delegate signature must match
    the ``Transport`` method's exact signature (name/kind/return), including
    the string return annotation -- what ``functools.wraps`` +
    ``__signature__`` (``github_delegation.py``) are for.
    """
    for name in TRANSPORT_MOVED_MEMBERS:
        transport_sig = inspect.signature(getattr(Transport, name))
        delegate_sig = inspect.signature(getattr(GitHub, name))
        _compatible_signature(transport_sig, delegate_sig)


def test_max_retries_retry_base_seconds_timeout_seconds_have_no_decorator() -> None:
    """Answers the task's property-shaped-delegate question directly: none of
    the three retry-knob methods carries ``@property`` or any other
    decorator (design doc Section 3.1's decorator invariant), so no
    property-shaped delegate extension to ``github_delegation.py`` is needed.
    Checked by AST, not by reading source by eye.
    """
    src = inspect.getsource(Transport)
    tree = ast.parse(src)
    (transport_cls,) = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    for node in transport_cls.body:
        if isinstance(node, ast.FunctionDef) and node.name in (
            "_max_retries",
            "_retry_base_seconds",
            "_timeout_seconds",
        ):
            assert node.decorator_list == [], f"{node.name} unexpectedly has a decorator"


def test_retry_knobs_read_from_runtime_config_through_delegates(tmp_path: Path) -> None:
    """``_max_retries``/``_retry_base_seconds``/``_timeout_seconds`` must read
    ``self.runtime`` (forwarded from the ``Transport`` collaborator to the
    owning ``GitHub`` instance via ``CapabilityCollaborator.__getattr__``) and
    fall back to the relocated ``_DEFAULT_GH_*`` constants when ``runtime`` is
    ``None`` -- called directly through the installed delegates.
    """
    from charlie_work.config import RuntimeConfig

    gh_default = GitHub(tmp_path)
    assert gh_default._max_retries() == 3
    assert gh_default._retry_base_seconds() == 1.0
    assert gh_default._timeout_seconds() == 120.0

    gh_configured = GitHub(
        tmp_path,
        runtime=RuntimeConfig(
            gh_max_retries=7, gh_retry_base_seconds=0.5, gh_timeout_seconds=42.0
        ),
    )
    assert gh_configured._max_retries() == 7
    assert gh_configured._retry_base_seconds() == 0.5
    assert gh_configured._timeout_seconds() == 42.0


def test_retry_constants_read_through_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``run`` (unchanged, stays on the owner) must actually consult the
    relocated ``_max_retries()``/``_retry_base_seconds()`` through the
    installed delegate on every retry loop iteration -- not just that the
    delegate methods work in isolation (the test above), but that ``run``'s
    retry loop really calls through them. Patches
    ``charlie_work.github.subprocess.run`` (the dotted-tail form the L09
    advisory review confirmed reaches the shared object rather than replacing
    the whole ``github`` module attribute) and ``time.sleep`` to keep this
    fast and deterministic.
    """
    from charlie_work.config import RuntimeConfig

    call_count = 0

    def fake_subprocess_run(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return subprocess.CompletedProcess(
            args, returncode=1, stdout="", stderr="http 502 bad gateway"
        )

    monkeypatch.setattr(_github_module.subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(time, "sleep", lambda *_a, **_k: None)

    gh = GitHub(tmp_path, runtime=RuntimeConfig(gh_max_retries=2, gh_retry_base_seconds=0.01))
    with pytest.raises(_github_module.GitHubError):
        gh.run(["issue", "list"])

    # max_retries=2 -> 3 attempts total (initial + 2 retries). If `run` were
    # not reading _max_retries() through the delegate (e.g. a stale closure
    # over the default), this would be 4 (the _DEFAULT_GH_MAX_RETRIES=3 value).
    assert call_count == 3


def test_run_stays_lexical_not_a_delegate() -> None:
    """``run`` must remain a real lexical ``GitHub`` method, never replaced by
    a generated delegate -- it is the 134-patch-site interception seam and
    must never move (design doc Section 3.2).
    """
    assert "run" in GitHub.__dict__
    assert not hasattr(GitHub.__dict__["run"], "__wrapped__")


def test_run_monkeypatchable_at_class_level_with_moved_body_observing_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A class-level ``monkeypatch.setattr(GitHub, "run", ...)`` must still be
    observed by a MOVED body's internal ``self.run(...)`` call -- proving the
    134 existing `run` patch sites keep intercepting after ``_run_bool``/
    ``_list_json``/etc. move to ``Transport``. ``_list_json`` (moved) calls
    ``self.run(...)``, which resolves via ``CapabilityCollaborator.__getattr__``
    to the owner's ``run`` -- patched here at the class level, unaffected by
    which collaborator ``self`` is.
    """
    calls: list[list[str]] = []

    def fake_run(
        self: GitHub, args: list[str], *, json_output: bool = False, allow_failure: bool = False
    ):
        calls.append(args)
        return [{"number": 1}]

    monkeypatch.setattr(GitHub, "run", fake_run)

    gh = GitHub(tmp_path)
    result = gh._list_json(["issue", "list"], limit=10, kind="issues")

    assert result == [{"number": 1}]
    assert calls == [["issue", "list"]]


def test_run_monkeypatchable_at_instance_level_with_moved_body_observing_it(
    tmp_path: Path,
) -> None:
    """Instance-level ``run`` patching, with a moved body observing it.

    ``GitHub`` is a frozen dataclass, so ``monkeypatch.setattr(gh, "run",
    fake)`` raises ``FrozenInstanceError`` -- not a viable pattern for a real
    ``GitHub`` instance, and confirmed absent from the existing test suite's
    134 patch sites (all are class-level). The one sanctioned escape hatch is
    ``object.__setattr__``, the same one ``__post_init__`` itself uses to set
    ``_list_cache`` and the collaborator attributes on a frozen instance. This
    test proves that hatch also works for ``run`` and that a moved body
    (``_run_bool``) observes it -- covering "still monkeypatchable ... at
    instance level" for the one construction that is actually possible on a
    frozen dataclass.
    """
    from dataclasses import FrozenInstanceError

    gh = GitHub(tmp_path)
    with pytest.raises(FrozenInstanceError):
        gh.run = lambda *a, **k: None  # type: ignore[method-assign]

    calls: list[list[str]] = []

    def fake_run(args: list[str], *, allow_failure: bool = False):
        calls.append(args)
        return _github_module.GitHubRunResult(
            ok=True, returncode=0, stdout="", stderr="", value=None
        )

    object.__setattr__(gh, "run", fake_run)

    assert gh._run_bool(["label", "create", "x"]) is True
    assert calls == [["label", "create", "x"]]


def test_normalize_rest_pr_delegate_reachable(tmp_path: Path) -> None:
    """``gh._normalize_rest_pr(...)`` reaches the moved ``Transport`` body
    through the installed delegate and produces the documented mapping.
    """
    gh = GitHub(tmp_path)
    pr = {
        "number": 5,
        "title": "t",
        "body": "b",
        "head": {"ref": "feature", "sha": "abc123", "repo": {"full_name": "o/r"}},
        "base": {"repo": {"full_name": "o/r"}},
        "merge_commit_sha": "def456",
    }
    result = gh._normalize_rest_pr(pr)
    assert result == {
        "number": 5,
        "title": "t",
        "body": "b",
        "headRefName": "feature",
        "isCrossRepository": False,
        "state": "MERGED",
        "headRefOid": "abc123",
        "mergeCommitOid": "def456",
    }


def test_pr_checks_fallback_delegate_reachable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``gh._pr_checks_fallback(...)`` reaches the moved ``Transport`` body
    through the installed delegate, driven by a patched class-level ``run``.
    """

    def fake_run(
        self: GitHub, args: list[str], *, json_output: bool = False, allow_failure: bool = False
    ):
        return _github_module.GitHubRunResult(
            ok=True,
            returncode=0,
            stdout="",
            stderr="",
            value={"statusCheckRollup": []},
        )

    monkeypatch.setattr(GitHub, "run", fake_run)
    gh = GitHub(tmp_path)
    assert gh._pr_checks_fallback(1) == []


def test_repo_owner_name_delegate_uses_owner_shared_list_cache(tmp_path: Path) -> None:
    """``_repo_owner_name``'s cache read/write must hit the OWNER's
    ``_list_cache`` -- the same shared-by-reference decoupling every leaf's
    cache-backed member relies on (design doc Section 3.3).
    """
    gh = GitHub(tmp_path)
    assert gh._transport._list_cache is gh._list_cache

    gh._list_cache[("_repo_owner_name",)] = ("cached-owner", "cached-repo")
    assert gh._repo_owner_name() == ("cached-owner", "cached-repo")
    assert gh._transport.__dict__.get("_list_cache") is None


def test_validate_field_lists_import_depth_fix_reaches_real_config_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The one disclosed non-verbatim edit in this leaf: ``validate_field_lists``'s
    local ``from .config import ConfigError`` became ``from ..config import
    ConfigError`` because the function physically moved one package level
    deeper (``charlie_work`` -> ``charlie_work.github_capabilities``). Proves
    the fix reaches the identical, real ``charlie_work.config.ConfigError``
    type (not a shadow/duplicate at the wrong path) by forcing the
    ``FileNotFoundError`` branch and checking the raised exception's type
    identity directly against an import of ``charlie_work.config``.
    """
    from charlie_work.github_capabilities import transport as transport_mod

    def boom(*_args, **_kwargs):
        raise FileNotFoundError("gh not found")

    monkeypatch.setattr(transport_mod.subprocess, "run", boom)

    gh = GitHub(tmp_path)
    with pytest.raises(ConfigError) as excinfo:
        gh.validate_field_lists()
    assert type(excinfo.value) is ConfigError


class _SubclassOverridesRepoOwnerName(GitHub):
    """Positive control for the L07-established same-collaborator bypass
    hazard class, applied to L09.

    ``_repo_owner_name`` and ``_graphql_query`` both moved to ``Transport`` in
    this same leaf, and ``_graphql_query``'s body calls
    ``self._repo_owner_name()`` internally. A ``GitHub`` subclass override of
    ``_repo_owner_name`` is consulted for an EXTERNAL call
    (``gh._repo_owner_name()`` resolves via ordinary Python MRO on the ``gh``
    instance -- the subclass wins). It is NOT consulted for
    ``_graphql_query``'s INTERNAL ``self._repo_owner_name()`` call: that call's
    ``self`` is ``gh._transport`` (the ``Transport`` collaborator instance),
    whose own MRO (``Transport`` -> ``CapabilityCollaborator`` -> ``object``)
    has no path back to a ``GitHub`` subclass at all -- normal attribute
    lookup finds ``Transport._repo_owner_name`` directly and never reaches
    ``CapabilityCollaborator.__getattr__``, let alone a ``GitHub`` subclass's
    ``__dict__``.

    This is a documented hazard *class*, not a fix for an existing broken
    test: the L09 PR body's census (every ``GitHub.<name>`` attribute access,
    every ``monkeypatch.setattr``/``patch``/``patch.object`` at class,
    instance, and subclass granularity) found zero current tests that
    actually rely on a subclass override of any of the twelve moved names
    being observed by another moved member's internal call. This class exists
    solely to demonstrate what such a bypass would look like, per the task's
    instruction to document the hazard the way L07 did for
    ``are_issues_open``/``issue_view``.
    """

    def _repo_owner_name(self) -> tuple[str, str]:
        return ("subclass-owner", "subclass-repo")


def test_subclass_override_of_repo_owner_name_is_bypassed_by_graphql_query(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Positive control: prove the bypass empirically, not just by argument.

    The subclass override IS consulted for the external call. It is NOT
    consulted for ``_graphql_query``'s internal call: seeding the owner's
    shared ``_list_cache`` (bypassing ``_repo_owner_name``'s own subprocess
    call, keeping this hermetic) with a *different* owner/name pair proves
    ``_graphql_query`` used ``Transport._repo_owner_name`` -- which reads that
    cache -- rather than the subclass's hardcoded override.
    """
    gh = _SubclassOverridesRepoOwnerName(tmp_path)

    # External call: the subclass override wins, as ordinary Python attribute
    # resolution on the `gh` instance predicts.
    assert gh._repo_owner_name() == ("subclass-owner", "subclass-repo")

    # Internal call, from within Transport: seed a DIFFERENT owner/name pair
    # in the shared cache that only Transport's real _repo_owner_name reads.
    gh._list_cache[("_repo_owner_name",)] = ("real-owner", "real-repo")

    captured_args: list[list[str]] = []

    def fake_run(
        self: GitHub, args: list[str], *, json_output: bool = False, allow_failure: bool = False
    ):
        captured_args.append(args)
        return {"data": {}}

    monkeypatch.setattr(GitHub, "run", fake_run)

    gh._graphql_query("query { viewer { login } }")

    [args] = captured_args
    assert "owner=real-owner" in args
    assert "name=real-repo" in args
    assert "owner=subclass-owner" not in args
    assert "name=subclass-repo" not in args


def test_graphql_issue_states_and_dependencies_call_through_transport_internals(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``_graphql_issue_states``/``_graphql_issue_dependencies`` call
    ``self._repo_owner_name()`` and ``self._graphql_query()`` -- both moved to
    ``Transport`` in this same leaf, so these are same-collaborator internal
    calls too. Patching ``GitHub.run`` at class level (the one seam that never
    moves) must still be enough to drive both end to end, with no bypass:
    unlike the positive control above, these two call ANOTHER moved member
    (not a subclass override), and that call is a normal, unpatched
    same-class method lookup that always resolves correctly.
    """
    gh = GitHub(tmp_path)
    gh._list_cache[("_repo_owner_name",)] = ("o", "r")

    def fake_run(
        self: GitHub, args: list[str], *, json_output: bool = False, allow_failure: bool = False
    ):
        return {"data": {"repository": {"s_1": {"number": 1, "state": "OPEN"}}}}

    monkeypatch.setattr(GitHub, "run", fake_run)
    assert gh._graphql_issue_states([1]) == {1: True}

    def fake_run_deps(
        self: GitHub, args: list[str], *, json_output: bool = False, allow_failure: bool = False
    ):
        return {
            "data": {
                "repository": {
                    "i_1": {
                        "number": 1,
                        "blockedBy": {"nodes": [{"number": 2, "state": "OPEN"}], "pageInfo": {}},
                    }
                }
            }
        }

    monkeypatch.setattr(GitHub, "run", fake_run_deps)
    assert gh._graphql_issue_dependencies([1]) == {1: [2]}
    assert gh._list_cache[("issue_open", 2)] is True
