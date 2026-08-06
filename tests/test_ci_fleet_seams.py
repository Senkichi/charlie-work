"""Seam integrity between charlie_work and ci_fleet.

STAGED FILE — this is not part of ci_fleet's own suite and will not run here.
It belongs to charlie-work, whose venv is the only place both packages are
importable. Phase 1 step 6 copies it to ``charlie-work/tests/`` in the same
atomic commit that repoints the consumers. See ``migration/step6/README.md``.

Why it exists
-------------
Three of the four seams in this extraction are enforced by nothing but a line
of prose in a docstring, and each fails silently:

* ``GitHubError`` was **moved**, not copied, because it is *caught*. If
  ``charlie_work/github.py`` keeps its own ``class GitHubError(RuntimeError)``
  instead of importing ci_fleet's, the two classes are unrelated types and
  every ``except GitHubError`` in charlie-work stops catching what ci_fleet
  raises. Nothing fails at import; nothing fails in the unit suite. The first
  symptom is an unhandled exception during a real GitHub API failure — which
  is to say, in production, during an incident, in the code that was supposed
  to handle the incident.

* The two config dataclasses were moved for the same reason one layer over:
  they are compared and isinstance-checked. Two structurally identical frozen
  dataclasses are not equal to each other.

* The event sink is **injected**. A missing ``set_event_sink`` call is the
  purest form of the verification ladder's L3 gap: the code exists, is
  imported, is tested, and is never called. Fleet events would simply stop
  being recorded, and the only evidence would be a warning in a log.

* The event **reader** is injected the same way and is easier to forget,
  because forgetting it breaks nothing visible. Edge-triggered capacity
  signaling (#799) asks the store "have I already signalled?"; with no reader
  it gets ``None``, correctly declines to guess, and stays silent forever. The
  fleet keeps running, every other event keeps being written, and
  ``runner_capacity_starved`` simply never appears — which is indistinguishable
  from a host that was never starved.

Each of these is a one-line omission at step 6 with no loud failure mode.
That is exactly the shape of bug that a test, and only a test, catches.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# The moved exception
# ---------------------------------------------------------------------------


def test_github_error_is_one_class_not_two() -> None:
    """``charlie_work.github.GitHubError`` must *be* ci_fleet's class."""
    import charlie_work.github

    import ci_fleet.github

    assert charlie_work.github.GitHubError is ci_fleet.github.GitHubError


def test_identity_check_would_fail_on_a_redefined_class() -> None:
    """Positive control for the assertion above.

    ``is`` on two names that happen to refer to the same object is a weak
    signal unless you show it discriminates. Two separately-declared classes
    with identical bodies are distinct objects and distinct ``except`` targets.
    """

    class A(RuntimeError):
        pass

    class B(RuntimeError):
        pass

    assert A is not B
    with pytest.raises(B):
        try:
            raise B("boom")
        except A:  # pragma: no cover - the point is that this does not fire
            pytest.fail("a structurally identical class caught a foreign one")


def test_subclasses_still_descend_from_the_moved_base() -> None:
    """The two subclasses stay in charlie-work but must inherit from ci_fleet.

    Only the base moved. If the subclasses were left inheriting from a stale
    local base, ``except GitHubError`` would catch the base and miss both
    refinements — the narrower failure, and the harder one to spot.
    """
    from charlie_work.github import GitHubNotFoundError, GraphQLBudgetError

    import ci_fleet.github

    assert issubclass(GitHubNotFoundError, ci_fleet.github.GitHubError)
    assert issubclass(GraphQLBudgetError, ci_fleet.github.GitHubError)


def test_a_ci_fleet_raise_is_caught_by_a_charlie_work_except() -> None:
    """The behavioural form of the identity check.

    Identity is the mechanism; this is the property anyone actually cares
    about, written so that a future refactor which satisfies the letter of the
    identity test but breaks catching would still fail here.
    """
    from charlie_work.github import GitHubError as WorkError

    from ci_fleet.github import GitHubError as FleetError

    caught = False
    try:
        raise FleetError("raised by the fleet")
    except WorkError:
        caught = True

    assert caught


# ---------------------------------------------------------------------------
# The moved config dataclasses
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["RunnerAllocationConfig", "RunnerScalingConfig"])
def test_config_dataclasses_are_one_class_not_two(name: str) -> None:
    """Equality and isinstance both depend on these being the same object."""
    import charlie_work.config

    import ci_fleet.config

    assert getattr(charlie_work.config, name) is getattr(ci_fleet.config, name)


def test_config_instances_compare_equal_across_the_seam() -> None:
    """The consequence the identity check is standing in for."""
    from charlie_work.config import RunnerAllocationConfig as WorkCfg

    from ci_fleet.config import RunnerAllocationConfig as FleetCfg

    assert FleetCfg(enabled=True) == WorkCfg(enabled=True)


# ---------------------------------------------------------------------------
# The injected event sink (verification ladder L3 -> L4)
# ---------------------------------------------------------------------------


def test_importing_charlie_work_installs_the_event_sink() -> None:
    """Weak form: a sink is registered, and it is charlie-work's.

    The provider self-registers on import, so merely importing
    ``charlie_work.instrumentation`` must be enough. If this needed an
    explicit call from the test to pass, the wiring would be missing in every
    process that did not make that call — including the supervisor.
    """
    import charlie_work.instrumentation

    from ci_fleet.observability import get_event_sink

    assert get_event_sink() is charlie_work.instrumentation.log_event


def test_an_event_logged_through_ci_fleet_lands_in_the_database(tmp_path: Path) -> None:
    """Strong form: the row actually arrives.

    ``get_event_sink() is not None`` proves a function was registered, not
    that calling it writes anything — the seam swallows sink exceptions by
    design, so a sink that raised on every call would still satisfy the weak
    check. This drives a real event through ci_fleet's seam and reads it back
    out of charlie-work's SQLite log.
    """
    import charlie_work.instrumentation  # noqa: F401  (registers the sink)

    from ci_fleet.observability import log_event

    state_path = tmp_path / "state.json"
    state_path.write_text("{}", encoding="utf-8")

    log_event(state_path, "ci_fleet_seam_probe", {"probe": True}, repo="Senkichi/probe")

    db = tmp_path / "events.db"
    assert db.exists(), "no events.db was created alongside state.json"

    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            "SELECT kind, repo FROM events WHERE kind = ?", ("ci_fleet_seam_probe",)
        ).fetchall()

    assert rows == [("ci_fleet_seam_probe", "Senkichi/probe")]


# ---------------------------------------------------------------------------
# The injected event reader — the seam whose omission is silent
# ---------------------------------------------------------------------------


def test_importing_charlie_work_installs_the_event_reader() -> None:
    """Weak form: a reader is registered, and it is charlie-work's."""
    import charlie_work.instrumentation

    from ci_fleet.observability import get_event_query

    assert get_event_query() is charlie_work.instrumentation.query_events


def test_an_event_written_through_the_seam_is_readable_back_through_it(tmp_path: Path) -> None:
    """Strong form, and the one the edge trigger actually depends on.

    Write and read are separate seams and can be wired independently, so a
    registered reader proves nothing about whether it can see what the sink
    wrote. This is the round trip: ci_fleet writes, charlie-work stores,
    ci_fleet reads it back and finds it.
    """
    import charlie_work.instrumentation  # noqa: F401  (registers both seams)

    from ci_fleet.observability import log_event, query_events

    state_path = tmp_path / "state.json"
    state_path.write_text("{}", encoding="utf-8")

    log_event(state_path, "ci_fleet_seam_probe", {"probe": True}, repo="Senkichi/probe")
    rows = query_events(state_path, kind="ci_fleet_seam_probe", repo="Senkichi/probe")

    assert rows is not None, "query_events returned None -- no reader is installed"
    assert len(rows) == 1


def test_the_reader_distinguishes_absent_from_unknown(tmp_path: Path) -> None:
    """``()`` and ``None`` must not collapse into each other.

    A wired reader asked about a kind nobody wrote returns an *empty* result:
    the store was consulted and holds nothing. That is a different answer from
    ``None``, which means the store was never consulted at all. The edge
    trigger emits on the first and declines on the second, so a reader that
    returned ``None`` for "nothing found" would make capacity signaling go
    permanently silent while looking correctly wired.
    """
    import charlie_work.instrumentation  # noqa: F401

    from ci_fleet.observability import query_events

    state_path = tmp_path / "state.json"
    state_path.write_text("{}", encoding="utf-8")

    rows = query_events(state_path, kind="a_kind_nobody_has_ever_written")

    assert rows is not None
    assert list(rows) == []


# ---------------------------------------------------------------------------
# The injected provenance anchor — the seam that says which tree is running
# ---------------------------------------------------------------------------


def test_importing_charlie_work_installs_the_provenance_anchor() -> None:
    """Weak form: an anchor is registered, and it is charlie-work's."""
    import charlie_work.instrumentation  # noqa: F401  (registers the seam)

    from charlie_work.ci_fleet_anchor import declared_ci_fleet_root
    from ci_fleet.provenance import get_provenance_anchor

    assert get_provenance_anchor() is declared_ci_fleet_root


def test_the_installed_anchor_agrees_with_the_live_import_root() -> None:
    """Strong form: in this venv the check actually returns ``ok``.

    A registered anchor proves a callable was handed over, not that it names
    the right directory. Under ``no_anchor`` the fleet keeps actuating and
    merely accumulates a streak, so an anchor that pointed somewhere wrong
    would look installed and report nothing useful until it started refusing
    passes for the wrong reason.
    """
    import charlie_work.instrumentation  # noqa: F401

    from charlie_work.ci_fleet_anchor import declared_ci_fleet_root
    from ci_fleet.provenance import check_provenance

    verdict = check_provenance()

    # Unconditional: whatever the layout, the installed anchor must never
    # disagree with the tree that is actually loaded. This is the assertion
    # that would catch a wrong declaration.
    assert verdict.status != "mismatch", verdict.detail
    assert not verdict.blocks_actuation, verdict.detail

    # Conditional on the declaration resolving at all, which it does from the
    # real checkout and from CI (which checks ci_runners out as a sibling), but
    # not from a worktree under .claude/worktrees/.
    if declared_ci_fleet_root() is not None:
        assert verdict.status == "ok", verdict.detail


def test_a_wrong_declaration_is_reported_as_mismatch(tmp_path: Path) -> None:
    """Control. Without this, the ``ok`` above could be vacuous.

    ``check_provenance`` reports ``no_anchor`` for anything it cannot
    evaluate, and ``no_anchor`` does not block. So a check that had quietly
    become incapable of ever saying ``mismatch`` would still let the test
    above pass. Drive a deliberately wrong declaration through the same
    function and require it to disagree.
    """
    from ci_fleet.provenance import check_provenance

    verdict = check_provenance(anchor=lambda: tmp_path)

    assert verdict.status == "mismatch", verdict.detail
    assert verdict.blocks_actuation


def test_the_anchor_reads_the_declaration_not_the_install_artifacts() -> None:
    """The anchor's whole value is that it has an independent origin.

    ``direct_url.json`` and ``_editable_impl_ci_fleet.pth`` are both written by
    the same ``uv pip install -e`` run, so an install repointed at another tree
    carries both and neither can contradict it. ``pyproject.toml`` is the
    install's *input*, lives in this repo, and no ``.pth`` can move it — which
    is the only reason the comparison is a check rather than a tautology.

    This recomputes the expected root straight from the TOML and requires the
    anchor to match, so rewiring the anchor to read an install artifact fails
    here rather than silently turning the guard into a no-op.
    """
    import tomllib

    from charlie_work.ci_fleet_anchor import declared_ci_fleet_root, repo_root

    root = repo_root()
    with (root / "pyproject.toml").open("rb") as handle:
        declared = tomllib.load(handle)["tool"]["uv"]["sources"]["ci-fleet"]["path"]

    expected = (root / declared / "src").resolve()
    # Not asserted unconditionally: this suite also runs from worktrees, where
    # the relative declaration resolves to a directory that does not exist and
    # the anchor correctly abstains. See the abstention test below.
    assert declared_ci_fleet_root() == (expected if expected.is_dir() else None)


def test_an_unresolvable_declaration_abstains_instead_of_blocking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one that keeps this guard from becoming an outage.

    ``../ci_runners`` is relative, so from a worktree under
    ``.claude/worktrees/`` it names a sibling of the *worktree* — a directory
    that does not exist. If the anchor returned that path anyway, the verdict
    would be ``mismatch``, ``blocks_actuation`` would be true, and the fleet
    would refuse to actuate from every worktree in the repo. This repo runs
    twenty of them.

    Abstaining is also the honest answer: from a worktree we genuinely cannot
    tell where ci_fleet ought to load from.
    """
    import charlie_work.ci_fleet_anchor as anchor_mod

    from ci_fleet.provenance import check_provenance

    fake_repo = tmp_path / "worktrees" / "agent-something"
    fake_repo.mkdir(parents=True)
    (fake_repo / "pyproject.toml").write_text(
        '[tool.uv.sources]\nci-fleet = { path = "../ci_runners", editable = true }\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(anchor_mod, "repo_root", lambda: fake_repo)

    assert not (fake_repo.parent / "ci_runners").exists(), "control: the sibling must be absent"
    assert anchor_mod.declared_ci_fleet_root() is None

    verdict = check_provenance(anchor=anchor_mod.declared_ci_fleet_root)
    assert verdict.status == "no_anchor", verdict.detail
    assert not verdict.blocks_actuation, "an unresolvable declaration must never block the fleet"


def test_a_resolvable_declaration_is_still_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Control for the abstention: existing-but-wrong must still be caught.

    The ``is_dir()`` guard above could have been written so broadly that it
    swallowed real disagreements too. A repointed install still resolves to a
    directory that exists, so it must still come back ``mismatch``.
    """
    import charlie_work.ci_fleet_anchor as anchor_mod

    from ci_fleet.provenance import check_provenance

    fake_repo = tmp_path / "charlie-work"
    fake_repo.mkdir()
    (tmp_path / "ci_runners" / "src").mkdir(parents=True)  # exists, but is not the live tree
    (fake_repo / "pyproject.toml").write_text(
        '[tool.uv.sources]\nci-fleet = { path = "../ci_runners", editable = true }\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(anchor_mod, "repo_root", lambda: fake_repo)

    assert anchor_mod.declared_ci_fleet_root() == (tmp_path / "ci_runners" / "src").resolve()

    verdict = check_provenance(anchor=anchor_mod.declared_ci_fleet_root)
    assert verdict.status == "mismatch", verdict.detail
    assert verdict.blocks_actuation


# ---------------------------------------------------------------------------
# The adapter surface
# ---------------------------------------------------------------------------


def test_adapter_surface_matches_what_consumers_import() -> None:
    """Every name the adapter promises resolves in charlie-work's venv.

    ci_fleet's own suite pins this list, but it cannot check it against a real
    installation — the point of the extraction is that ci_fleet's tests run
    without charlie-work present. This is the same assertion made where both
    halves exist.
    """
    from ci_fleet import charlie_work_adapter

    missing = [n for n in charlie_work_adapter.__all__ if not hasattr(charlie_work_adapter, n)]

    assert missing == []


def _names_imported_from_the_adapter() -> dict[str, str]:
    """Every name charlie_work imports from the adapter -> the file importing it."""
    import ast

    import charlie_work

    # __path__, not __file__: the latter is typed ``str | None`` and is None for
    # a namespace package, which would turn a packaging change into a TypeError
    # here rather than the import failure it actually is.
    root = Path(next(iter(charlie_work.__path__)))

    found: dict[str, str] = {}
    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - a broken tree is a different failure
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "ci_fleet.charlie_work_adapter":
                for alias in node.names:
                    found.setdefault(alias.name, path.name)
    return found


def test_adapter_exports_every_name_the_consumers_import() -> None:
    """The adapter is checked against its *consumers*, not against itself.

    The test above compares ``__all__`` to the module's own attributes, which
    is a self-consistency check: it passes for any internally coherent adapter,
    including one missing a name a consumer needs. That is not hypothetical.
    The list was written against a 2026-07-30 snapshot of 17 names; upstream
    `#683` later gave ``fleet_dispatch`` a reason to import
    ``save_allocation_skip``, and nothing in either suite objected.

    ``fleet_dispatch`` is the unattended path. A name missing here does not
    fail a test or an import of ``charlie_work`` — it raises ``ImportError`` on
    the next five-minute scheduled firing, allocation stops actuating with
    nothing alarming, and queued jobs sit until GitHub fails them at 24 hours
    (`R-09`). So the expectation is derived from the consumers by AST walk
    rather than restated by hand, because a hand-maintained copy of a list is
    the thing that just drifted.
    """
    from ci_fleet import charlie_work_adapter

    imported = _names_imported_from_the_adapter()

    # Positive control (R-20): an empty walk would pass this vacuously, and
    # "no consumer imports the adapter" is exactly what a half-finished step 6
    # looks like. The walk must find the consumers before its silence means
    # anything.
    assert imported, "no module imports from ci_fleet.charlie_work_adapter -- step 6 incomplete?"

    missing = {n: f for n, f in sorted(imported.items()) if not hasattr(charlie_work_adapter, n)}

    assert missing == {}, f"consumers import names the adapter does not export: {missing}"
