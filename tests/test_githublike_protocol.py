"""Regression tests for the GitHubLike structural protocol surface."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

import charlie_work.github as _github_module
import charlie_work.github_capabilities as _github_capabilities
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


def test_github_isinstance_commentslike(tmp_path: Path) -> None:
    """``GitHub`` must satisfy ``CommentsLike`` at runtime after the L02 move.

    Duplicates part of ``test_github_satisfies_commentslike`` deliberately: a
    standalone ``isinstance`` assertion so this specific claim (survives the
    L02 move of ``issue_comment``/``pr_comment`` off ``GitHub``) has its own
    named test independent of the broader signature-conformance loop.
    """
    assert isinstance(GitHub(tmp_path), CommentsLike)


def test_issue_comment_and_pr_comment_are_not_lexical_github_defs() -> None:
    """``GitHub`` must no longer lexically define ``issue_comment``/``pr_comment``.

    Track 2 L02 (issue #1586) moved both bodies to ``Comments``; the names
    are now served on ``GitHub`` by L01 generated delegates (class-level
    assignments, not ``def``s). Checking ``"issue_comment" not in
    GitHub.__dict__`` would be the wrong test here -- the installed delegate
    *is* in ``GitHub.__dict__`` (that's how ``getattr(GitHub, name)``
    resolves it). Instead walk the AST of ``GitHub``'s own class body and
    assert neither name appears as a lexical ``FunctionDef``/
    ``AsyncFunctionDef`` there.
    """
    lexical = _lexical_github_defs()
    assert "issue_comment" not in lexical
    assert "pr_comment" not in lexical
    # And confirm the names *are* still resolvable, as delegates.
    assert hasattr(GitHub.issue_comment, "__wrapped__")
    assert hasattr(GitHub.pr_comment, "__wrapped__")


def test_comments_routes_point_at_the_comments_collaborator() -> None:
    """``_ROUTES`` must route both moved names to the ``_comments`` collaborator.

    Forward-compatible: asserts only the two entries this leaf (L02) adds,
    not the full ``_ROUTES`` contents, so it keeps holding unmodified once
    later leaves (L03+) populate more of the table.
    """
    assert _ROUTES["issue_comment"] == "_comments"
    assert _ROUTES["pr_comment"] == "_comments"


def test_issue_comment_delegate_forwards_through_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Calling ``gh.issue_comment(...)`` through the delegate must reach the
    patched class-level ``GitHub.run`` with the same argv the moved body
    produces.

    The expected argv is transcribed by reading the moved
    ``Comments.issue_comment``/``pr_comment`` bodies directly (both are two
    -line wrappers around ``self.run([...])``), not derived by calling the
    same code under test -- calling it twice would make this a circular
    assertion that could not catch a body that silently changed.
    """
    calls: list[tuple[list[str], bool, bool]] = []

    def fake_run(
        self: GitHub, args: list[str], *, json_output: bool = False, allow_failure: bool = False
    ) -> None:
        calls.append((args, json_output, allow_failure))
        return None

    monkeypatch.setattr(GitHub, "run", fake_run)

    gh = GitHub(tmp_path)
    body_file = Path("comment-body.md")
    gh.issue_comment(7, body_file)
    gh.pr_comment(9, body_file)

    assert calls == [
        (["issue", "comment", "7", "--body-file", "comment-body.md"], False, False),
        (["pr", "comment", "9", "--body-file", "comment-body.md"], False, False),
    ]


def test_github_isinstance_labelslike(tmp_path: Path) -> None:
    """``GitHub`` must satisfy ``LabelsLike`` at runtime after the L03 move.

    Duplicates part of ``test_github_satisfies_labelslike`` deliberately: a
    standalone ``isinstance`` assertion so this specific claim (survives the
    L03 move of the six label members off ``GitHub``) has its own named test
    independent of the broader signature-conformance loop.
    """
    assert isinstance(GitHub(tmp_path), LabelsLike)


def test_labels_members_are_not_lexical_github_defs() -> None:
    """``GitHub`` must no longer lexically define any of the six Labels members.

    Track 2 L03 (issue #1587) moved ``add_issue_label``, ``remove_issue_label``,
    ``add_pr_label``, ``remove_pr_label``, ``label_list``, and ``label_create``
    to ``Labels``; the names are now served on ``GitHub`` by L01 generated
    delegates (class-level assignments, not ``def``s). Checking
    ``"add_issue_label" not in GitHub.__dict__`` would be the wrong test here
    -- the installed delegate *is* in ``GitHub.__dict__`` (that's how
    ``getattr(GitHub, name)`` resolves it). Instead walk the AST of
    ``GitHub``'s own class body and assert none of the six names appears as a
    lexical ``FunctionDef``/``AsyncFunctionDef`` there.
    """
    lexical = _lexical_github_defs()
    moved_names = [
        "add_issue_label",
        "remove_issue_label",
        "add_pr_label",
        "remove_pr_label",
        "label_list",
        "label_create",
    ]
    for name in moved_names:
        assert name not in lexical, f"{name} is still a lexical GitHub def"
    # And confirm the names *are* still resolvable, as delegates.
    for name in moved_names:
        assert hasattr(getattr(GitHub, name), "__wrapped__"), (
            f"GitHub.{name} does not look like an installed delegate"
        )


def test_labels_routes_point_at_the_labels_collaborator() -> None:
    """``_ROUTES`` must route all six moved names to the ``_labels`` collaborator.

    Forward-compatible: asserts only the six entries this leaf (L03) adds,
    not the full ``_ROUTES`` contents, so it keeps holding unmodified once
    later leaves (L04+) populate more of the table.
    """
    for name in (
        "add_issue_label",
        "remove_issue_label",
        "add_pr_label",
        "remove_pr_label",
        "label_list",
        "label_create",
    ):
        assert _ROUTES[name] == "_labels"


def test_add_issue_label_delegate_forwards_through_run_bool(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Calling ``gh.add_issue_label(...)`` through the delegate must reach the
    patched class-level ``GitHub.run`` with the same argv the moved body
    produces, via the ``self._run_bool`` -> ``self.run`` chain.

    This exercises a strictly deeper resolution chain than the comments/
    label_list tests: delegate -> ``Labels`` collaborator -> collaborator
    ``__getattr__`` (for ``_run_bool``, still a lexical ``GitHub`` method
    until L09) -> owner ``_run_bool`` -> owner ``run``. The expected argv is
    transcribed by reading the moved ``Labels.add_issue_label`` body directly
    (a one-line wrapper around ``self._run_bool([...])``, which itself is a
    one-line wrapper around ``self.run(args, allow_failure=True)``), not
    derived by calling the same code under test -- calling it twice would
    make this a circular assertion that could not catch a body that silently
    changed.
    """
    calls: list[tuple[list[str], bool, bool]] = []

    def fake_run(
        self: GitHub, args: list[str], *, json_output: bool = False, allow_failure: bool = False
    ) -> _github_module.GitHubRunResult:
        calls.append((args, json_output, allow_failure))
        return _github_module.GitHubRunResult(ok=True, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(GitHub, "run", fake_run)

    gh = GitHub(tmp_path)
    result = gh.add_issue_label(7, "bug")

    assert result is True
    assert calls == [
        (["issue", "edit", "7", "--add-label", "bug"], False, True),
    ]


def test_label_list_delegate_forwards_through_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Calling ``gh.label_list()`` through the delegate must reach the patched
    class-level ``GitHub.run`` with the same argv the moved body produces,
    including the ``LABEL_LIST_FIELDS`` constant resolved from the
    collaborator's own module globals (not re-derived from the constant here
    -- the literal ``"name"`` is asserted directly, since importing
    ``LABEL_LIST_FIELDS`` to build the expectation would re-derive it from
    the same code under test and could not catch the constant failing to
    resolve at all, e.g. a ``NameError`` from a missing module-level binding).

    The expected argv is transcribed by reading the moved
    ``Labels.label_list`` body directly, mirroring
    ``test_issue_comment_delegate_forwards_through_run``'s approach for L02.
    """
    calls: list[tuple[list[str], bool, bool]] = []

    def fake_run(
        self: GitHub, args: list[str], *, json_output: bool = False, allow_failure: bool = False
    ) -> list[dict[str, str]]:
        calls.append((args, json_output, allow_failure))
        return [{"name": "bug"}]

    monkeypatch.setattr(GitHub, "run", fake_run)

    gh = GitHub(tmp_path)
    result = gh.label_list()

    assert result == [{"name": "bug"}]
    assert calls == [
        (["label", "list", "--limit", "200", "--json", "name"], True, False),
    ]


def test_github_isinstance_checkslike(tmp_path: Path) -> None:
    """``GitHub`` must satisfy ``ChecksLike`` at runtime after the L04 move.

    Duplicates part of ``test_github_satisfies_checkslike`` deliberately: a
    standalone ``isinstance`` assertion so this specific claim (survives the
    L04 move of the six CI check/run members off ``GitHub``) has its own
    named test independent of the broader signature-conformance loop.
    """
    assert isinstance(GitHub(tmp_path), ChecksLike)


def test_checks_members_are_not_lexical_github_defs() -> None:
    """``GitHub`` must no longer lexically define any of the six Checks members.

    Track 2 L04 (issue #1588) moved ``pr_checks``, ``check_run_annotations``,
    ``commit_check_runs``, ``actions_job``, ``workflow_runs_for_head``, and
    ``check_graphql_rate_limit`` to ``Checks``; the names are now served on
    ``GitHub`` by L01 generated delegates (class-level assignments, not
    ``def``s). Mirrors ``test_labels_members_are_not_lexical_github_defs``'s
    approach for L03 -- walk the AST of ``GitHub``'s own class body and
    assert none of the six names appears as a lexical
    ``FunctionDef``/``AsyncFunctionDef`` there.
    """
    lexical = _lexical_github_defs()
    moved_names = [
        "pr_checks",
        "check_run_annotations",
        "commit_check_runs",
        "actions_job",
        "workflow_runs_for_head",
        "check_graphql_rate_limit",
    ]
    for name in moved_names:
        assert name not in lexical, f"{name} is still a lexical GitHub def"
    # And confirm the names *are* still resolvable, as delegates.
    for name in moved_names:
        assert hasattr(getattr(GitHub, name), "__wrapped__"), (
            f"GitHub.{name} does not look like an installed delegate"
        )


def test_checks_routes_point_at_the_checks_collaborator() -> None:
    """``_ROUTES`` must route all six moved names to the ``_checks`` collaborator.

    Forward-compatible: asserts only the six entries this leaf (L04) adds,
    not the full ``_ROUTES`` contents, so it keeps holding unmodified once
    later leaves (L05+) populate more of the table.
    """
    for name in (
        "pr_checks",
        "check_run_annotations",
        "commit_check_runs",
        "actions_job",
        "workflow_runs_for_head",
        "check_graphql_rate_limit",
    ):
        assert _ROUTES[name] == "_checks"


def test_check_graphql_rate_limit_delegate_forwards_through_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Calling ``gh.check_graphql_rate_limit()`` through the delegate must reach
    the patched class-level ``GitHub.run`` with the same argv the moved body
    produces, and must recognize a real ``GitHubRunResult`` returned across
    the collaborator boundary via ``isinstance`` -- the mechanism this leaf
    introduces (``GitHubRunResult`` is now defined in
    ``github_capabilities/_base.py`` and re-exported through ``github.py``,
    not defined in ``github.py`` itself; see ``_base.py`` for why).

    The expected argv and default-threshold arithmetic are transcribed by
    reading the moved ``Checks.check_graphql_rate_limit`` body directly, not
    derived by calling the same code under test.
    """
    calls: list[tuple[list[str], bool, bool]] = []

    def fake_run(
        self: GitHub, args: list[str], *, json_output: bool = False, allow_failure: bool = False
    ) -> _github_module.GitHubRunResult:
        calls.append((args, json_output, allow_failure))
        return _github_module.GitHubRunResult(
            ok=True,
            returncode=0,
            stdout="",
            stderr="",
            value={"resources": {"graphql": {"remaining": 5000, "reset": 1699999999}}},
        )

    monkeypatch.setattr(GitHub, "run", fake_run)

    gh = GitHub(tmp_path)
    result = gh.check_graphql_rate_limit()

    assert result == (True, 5000, 1699999999)
    assert calls == [
        (["api", "rate_limit"], True, True),
    ]


def test_pr_checks_delegate_forwards_through_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Calling ``gh.pr_checks(...)`` through the delegate must reach the
    patched class-level ``GitHub.run`` with the same argv the moved body
    produces, resolving ``PR_CHECKS_FIELDS`` from the collaborator's own
    module globals (not re-derived from the constant here, mirroring
    ``test_label_list_delegate_forwards_through_run``'s rationale for why the
    literal is asserted directly), and must derive ``databaseId``/``runId``
    via ``_job_id_from_link``/``_run_id_from_link`` -- the former relocated
    alongside ``PR_CHECKS_FIELDS`` into ``github_capabilities/checks.py``,
    the latter still imported from the top-level ``charlie_work.checks``
    module, exactly as the moved body does.

    This exercises the deepest new resolution chain this leaf introduces:
    delegate -> ``Checks`` collaborator -> a real ``GitHubRunResult``
    instance (built in ``_base.py``, re-exported through ``github.py``) ->
    two bare-global helper functions resolved from ``checks.py``'s own
    module namespace. The expected argv and output shape are transcribed by
    reading the moved ``Checks.pr_checks`` body directly.
    """
    calls: list[tuple[list[str], bool, bool]] = []
    link = "https://github.com/o/r/actions/runs/1/job/42"

    def fake_run(
        self: GitHub, args: list[str], *, json_output: bool = False, allow_failure: bool = False
    ) -> _github_module.GitHubRunResult:
        calls.append((args, json_output, allow_failure))
        return _github_module.GitHubRunResult(
            ok=True,
            returncode=0,
            stdout="",
            stderr="",
            value=[{"name": "build", "state": "SUCCESS", "bucket": "pass", "link": link}],
        )

    monkeypatch.setattr(GitHub, "run", fake_run)

    gh = GitHub(tmp_path)
    result = gh.pr_checks(5)

    assert result == [
        {
            "name": "build",
            "state": "SUCCESS",
            "bucket": "pass",
            "link": link,
            "databaseId": 42,
            "runId": 1,
        }
    ]
    assert calls == [
        (["pr", "checks", "5", "--json", "name,state,bucket,link"], True, True),
    ]


def test_every_capability_module_declares_future_annotations() -> None:
    """Every ``github_capabilities`` module must have ``from __future__ import
    annotations`` (design doc Section 3.1).

    Delegate signature fidelity depends on it: the future import keeps a moved
    method's annotations as *strings*, so ``inspect.signature(GitHub.<name>)``
    (copied verbatim by ``_make_delegate``) matches the sub-protocol's
    stringized annotation under ``_compatible_signature``. Drop the import from
    a future collaborator module and its annotations become live objects,
    silently breaking the L02+ conformance path (the failure mode is a
    ``'None'``-vs-``None`` return-annotation mismatch). Nothing else enforces
    this, so assert it structurally across the whole package.
    """
    pkg_dir = Path(_github_capabilities.__file__).parent
    modules = sorted(pkg_dir.glob("*.py"))
    assert modules, "no github_capabilities modules found to check"
    for path in modules:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        has_future_annotations = any(
            isinstance(node, ast.ImportFrom)
            and node.module == "__future__"
            and any(alias.name == "annotations" for alias in node.names)
            for node in tree.body
        )
        assert has_future_annotations, (
            f"{path.name} is missing `from __future__ import annotations` "
            f"(required by design doc Section 3.1 for delegate signature fidelity)"
        )
