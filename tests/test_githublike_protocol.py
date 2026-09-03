"""Regression tests for the GitHubLike structural protocol surface."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

import charlie_work.github as _github_module
from charlie_work.github import GitHub, GitHubLike, _make_delegate, _ROUTES
from charlie_work.github_capabilities import (
    ChecksLike,
    CommentsLike,
    IssuesLike,
    LabelsLike,
    MergeBranchLike,
    PullRequestsLike,
    RepoMetaLike,
)


def test_githublike_protocol_dry_run_is_read_only_property() -> None:
    """GitHubLike.dry_run must be a read-only property, not a settable attribute.

    The concrete ``GitHub`` class is a frozen dataclass, so its ``dry_run``
    field is immutable. If the protocol declared ``dry_run`` as a settable
    attribute (``dry_run: bool``), Pyright would consider ``GitHub``
    incompatible with ``GitHubLike`` because a frozen field is read-only —
    every call site passing a ``GitHub`` where ``GitHubLike`` is annotated
    would be a ``reportArgumentType`` error (issue #733).

    A ``@property`` declaration makes the protocol require only a *readable*
    ``dry_run``, which the frozen dataclass field satisfies. Test doubles that
    set ``self.dry_run`` in ``__init__`` still satisfy a read-only property
    (settable is a superset of read-only).
    """
    raw = inspect.getattr_static(GitHubLike, "dry_run")
    assert isinstance(raw, property), (
        f"GitHubLike.dry_run must be a read-only property so the frozen "
        f"GitHub dataclass satisfies the protocol; got {type(raw).__name__}"
    )


def test_githublike_protocol_declares_branch_protection() -> None:
    """GitHubLike must declare ``branch_protection`` so ``GitHubLike``-typed
    ``self.gh`` can call it unguardedly without a pyright error.

    This is the review finding for PR #759 / issue #593.
    """
    assert "branch_protection" in GitHubLike.__dict__, "GitHubLike is missing branch_protection"
    sig = inspect.signature(GitHubLike.branch_protection)
    assert list(sig.parameters) == ["self", "base"]
    assert sig.return_annotation == "dict[str, Any] | None"


def test_githublike_protocol_declares_pr_ready() -> None:
    """GitHubLike must declare ``pr_ready`` so ``GitHubLike``-typed ``self.gh``
    can call it unguardedly without a pyright error.

    Same structural-omission class as ``branch_protection``.
    """
    assert "pr_ready" in GitHubLike.__dict__, "GitHubLike is missing pr_ready"
    sig = inspect.signature(GitHubLike.pr_ready)
    assert list(sig.parameters) == ["self", "number"]
    assert sig.return_annotation == "GitHubRunResult"


def test_githublike_protocol_declares_pr_close() -> None:
    """GitHubLike must declare ``pr_close`` (issue #1274, W17) so
    ``GitHubLike``-typed ``self.gh`` can call it unguardedly without a
    pyright error. Same structural-omission class as ``pr_ready``.
    """
    assert "pr_close" in GitHubLike.__dict__, "GitHubLike is missing pr_close"
    sig = inspect.signature(GitHubLike.pr_close)
    assert list(sig.parameters) == ["self", "number"]
    assert sig.return_annotation == "GitHubRunResult"


def test_githublike_protocol_declares_pr_reopen() -> None:
    """GitHubLike must declare ``pr_reopen`` (issue #1274, W17)."""
    assert "pr_reopen" in GitHubLike.__dict__, "GitHubLike is missing pr_reopen"
    sig = inspect.signature(GitHubLike.pr_reopen)
    assert list(sig.parameters) == ["self", "number"]
    assert sig.return_annotation == "GitHubRunResult"


def test_githublike_protocol_declares_push_empty_commit() -> None:
    """GitHubLike must declare ``push_empty_commit`` (issue #1274, W17) --
    the empty-commit-push fallback used when ``pr_close``/``pr_reopen``
    does not mechanically succeed.
    """
    assert "push_empty_commit" in GitHubLike.__dict__, "GitHubLike is missing push_empty_commit"
    sig = inspect.signature(GitHubLike.push_empty_commit)
    assert list(sig.parameters) == ["self", "branch"]
    assert sig.return_annotation == "GitHubRunResult"


def _compatible_signature(proto_sig: inspect.Signature, concrete_sig: inspect.Signature) -> None:
    proto_params = [p for n, p in proto_sig.parameters.items() if n != "self"]
    concrete_params = [p for n, p in concrete_sig.parameters.items() if n != "self"]
    assert len(proto_params) == len(concrete_params), (
        f"parameter count differs: {proto_sig} vs {concrete_sig}"
    )
    for proto_param, concrete_param in zip(proto_params, concrete_params):
        assert proto_param.name == concrete_param.name, (
            f"parameter name differs for {proto_sig}: "
            f"{proto_param.name!r} vs {concrete_param.name!r}"
        )
        assert proto_param.kind == concrete_param.kind, (
            f"parameter kind differs for {proto_param.name}: "
            f"{proto_param.kind} vs {concrete_param.kind}"
        )
    if proto_sig.return_annotation is not inspect.Signature.empty:
        assert concrete_sig.return_annotation is not inspect.Signature.empty, (
            f"concrete missing return annotation for {proto_sig}"
        )
        assert proto_sig.return_annotation == concrete_sig.return_annotation, (
            f"return annotation differs: {proto_sig.return_annotation} "
            f"vs {concrete_sig.return_annotation}"
        )


def test_github_satisfies_githublike_protocol(tmp_path: Path) -> None:
    """The concrete GitHub class must implement the full GitHubLike surface.

    Catches protocol drift before it breaks GitHubLike-typed callers.
    """
    gh = GitHub(tmp_path)
    assert isinstance(gh, GitHubLike), "GitHub does not satisfy GitHubLike at runtime"

    for name in sorted(GitHubLike.__protocol_attrs__):
        if name == "dry_run":
            assert hasattr(gh, name), f"GitHub is missing attribute {name}"
            continue
        concrete = getattr(gh, name, None)
        assert callable(concrete), f"GitHub is missing or non-callable method {name}"
        proto_sig = inspect.signature(getattr(GitHubLike, name))
        concrete_sig = inspect.signature(getattr(GitHub, name))
        _compatible_signature(proto_sig, concrete_sig)


def _assert_satisfies_subprotocol(gh: GitHub, proto: type) -> None:
    """Shared body for the seven per-sub-protocol conformance tests below.

    Mirrors ``test_github_satisfies_githublike_protocol`` but scoped to one
    capability's sub-protocol, so a later cluster move that drifts a
    signature fails its own sub-protocol test, not just the union test
    (design doc Section 4.2).
    """
    assert isinstance(gh, proto), f"GitHub does not satisfy {proto.__name__} at runtime"
    for name in sorted(proto.__protocol_attrs__):
        concrete = getattr(gh, name, None)
        assert callable(concrete), f"GitHub is missing or non-callable method {name}"
        proto_sig = inspect.signature(getattr(proto, name))
        concrete_sig = inspect.signature(getattr(GitHub, name))
        _compatible_signature(proto_sig, concrete_sig)


def test_github_satisfies_commentslike(tmp_path: Path) -> None:
    _assert_satisfies_subprotocol(GitHub(tmp_path), CommentsLike)


def test_github_satisfies_labelslike(tmp_path: Path) -> None:
    _assert_satisfies_subprotocol(GitHub(tmp_path), LabelsLike)


def test_github_satisfies_checkslike(tmp_path: Path) -> None:
    _assert_satisfies_subprotocol(GitHub(tmp_path), ChecksLike)


def test_github_satisfies_repometalike(tmp_path: Path) -> None:
    _assert_satisfies_subprotocol(GitHub(tmp_path), RepoMetaLike)


def test_github_satisfies_pullrequestslike(tmp_path: Path) -> None:
    _assert_satisfies_subprotocol(GitHub(tmp_path), PullRequestsLike)


def test_github_satisfies_issueslike(tmp_path: Path) -> None:
    _assert_satisfies_subprotocol(GitHub(tmp_path), IssuesLike)


def test_github_satisfies_mergebranchlike(tmp_path: Path) -> None:
    _assert_satisfies_subprotocol(GitHub(tmp_path), MergeBranchLike)


def _lexical_github_defs() -> set[str]:
    """Names GitHub defines directly via ``def`` in source (AST-derived).

    Mirrors the member_count ratchet's own counting rule (only direct
    ``FunctionDef``/``AsyncFunctionDef`` children of the ``ClassDef`` body),
    so this reflects "lexically defined" independent of whatever
    ``_install_delegates()`` has since added via ``setattr`` -- a class-level
    assignment is not an AST ``FunctionDef`` node and would not show up here.
    """
    source = Path(_github_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "GitHub":
            return {
                child.name
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                # Dunder-shaped names (including hand-written ones like
                # __post_init__) are excluded from this comparison the same
                # way they are excluded from `actual` below -- both sides
                # must apply the identical filter or a real dataclass hook
                # like __post_init__ reads as spurious "drift".
                and not (child.name.startswith("__") and child.name.endswith("__"))
            }
    raise AssertionError("GitHub class definition not found in charlie_work/github.py source")


def test_routes_names_never_collide_with_lexical_github_defs() -> None:
    """``_ROUTES`` must never claim a name GitHub already defines lexically.

    ``_install_delegates()`` itself skips such names at install time (a
    method not yet moved out of the class body always wins), but that skip
    would silently hide a genuine collision -- e.g. two collaborator classes
    both declaring the same public name. Assert directly that no collision
    exists, independent of the install-time skip behavior. Vacuous in L01
    (``_ROUTES`` is empty) and load-bearing from the first move leaf on.
    """
    colliding = set(_ROUTES) & _lexical_github_defs()
    assert not colliding, f"_ROUTES claims names GitHub already defines lexically: {colliding}"


def test_routes_names_resolve_to_callable_on_github() -> None:
    """Every routed name must resolve to a callable on ``GitHub``.

    Vacuous in L01 (``_ROUTES`` is empty, so this loop runs zero times), but
    asserted unconditionally so it starts enforcing the moment the first
    Mikado leaf populates ``_ROUTES``.
    """
    for name in _ROUTES:
        assert callable(getattr(GitHub, name, None)), (
            f"routed name {name!r} does not resolve to a callable on GitHub"
        )


def test_install_delegates_adds_only_routed_names() -> None:
    """GitHub's full callable surface must equal lexical defs plus ``_ROUTES``.

    The leaf-invariant check for ``_install_delegates()``: the set of
    non-dunder callables GitHub exposes at class level must be exactly the
    AST-lexical defs (real method bodies still on GitHub) unioned with the
    ``_ROUTES`` keys (forwarding delegates installed for moved bodies) --
    never more, never less. A stray class-level assignment, a delegate
    installed under the wrong name, or a member the install loop failed to
    add would all show up here as a set mismatch. Does not hardcode 53 (or
    any other count) so it stays valid as leaves move methods between the
    two sides of the union; vacuous-but-exact in L01, where ``_ROUTES`` is
    empty and this reduces to "GitHub's attribute surface equals its own
    lexical defs."
    """
    lexical = _lexical_github_defs()
    actual = {
        name
        for name, member in vars(GitHub).items()
        if not (name.startswith("__") and name.endswith("__")) and callable(member)
    }
    expected = lexical | set(_ROUTES)
    assert actual == expected, (
        f"GitHub's attribute surface drifted from lexical defs + _ROUTES: "
        f"extra={actual - expected} missing={expected - actual}"
    )


def test_make_delegate_preserves_signature_and_forwards_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_make_delegate`` must forward calls and preserve the source signature.

    ``_ROUTES`` is empty in L01, so the real delegate-install path
    (``_install_delegates()``) is exercised with zero live members -- every
    test above that references ``_ROUTES`` passes vacuously. This test
    converts that vacuous coverage into real evidence that ``_make_delegate``
    itself does what design doc Section 3.3 claims: (a) the returned
    function forwards the call to ``getattr(collab, name)(...)`` rather than
    running any body of its own, and (b) ``inspect.signature`` on the
    delegate matches the *source* function's signature -- including its
    string return annotation -- via the same ``_compatible_signature`` check
    the conformance tests use.
    """

    class _FakeSource:
        def greet(self, name: str, *, loud: bool = False) -> str:
            raise AssertionError("the delegate must never call its signature source directly")

    calls: list[tuple[str, bool]] = []

    class _FakeCollaborator:
        def greet(self, name: str, *, loud: bool = False) -> str:
            calls.append((name, loud))
            return "spied"

    class _FakeOwner:
        def __init__(self) -> None:
            self._collab = _FakeCollaborator()

    monkeypatch.setitem(_github_module._SIGNATURE_SOURCE, "greet", _FakeSource.greet)
    delegate = _make_delegate("greet", "_collab")

    result = delegate(_FakeOwner(), "world", loud=True)
    assert result == "spied", "delegate must forward to the collaborator's method"
    assert calls == [("world", True)]

    _compatible_signature(inspect.signature(_FakeSource.greet), inspect.signature(delegate))
