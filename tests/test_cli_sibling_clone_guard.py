"""Sibling-clone state-root misroute guard on state-mutating commands.

Split out of ``tests/test_cli.py`` (issue #1561, Track 1) -- bodies are
verbatim relocations; shared fakes live in ``tests/_cli_fixtures.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from _cli_fixtures import _FakeGitHub
from charlie_work import cli
from charlie_work.config import (
    ConfigError,
    OrchestratorConfig,
)
from charlie_work.paths import runtime_paths


# --------------------------------------------------------------------------
# sibling-clone state-root misroute guard (issue #1376)
# --------------------------------------------------------------------------


def _init_git_repo_with_origin(root: Path, remote_url: str) -> Path:
    """Create a real git repo with one commit and an ``origin`` remote."""
    import subprocess

    root.mkdir(parents=True, exist_ok=True)
    run = lambda args: subprocess.run(  # noqa: E731
        args, cwd=root, check=True, capture_output=True, text=True
    )
    run(["git", "init", "--initial-branch=main"])
    run(["git", "config", "user.email", "test@example.test"])
    run(["git", "config", "user.name", "Test User"])
    (root / "README.md").write_text("hello\n", encoding="utf-8")
    run(["git", "add", "README.md"])
    run(["git", "commit", "-m", "initial commit"])
    run(["git", "remote", "add", "origin", remote_url])
    return root


def _write_fleet_registry(
    fleet_dir: Path, name_with_owner: str, repo_root: Path, state_dir: Path
) -> None:
    """Write a minimal fleet.json with one registered repo."""
    fleet_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "repo_root": str(repo_root),
        "name_with_owner": name_with_owner,
        "config_path": str(repo_root / "orchestrator.config.yaml"),
        "state_dir": str(state_dir),
        "first_seen": "2026-01-01T00:00:00Z",
        "last_seen": "2026-01-01T00:00:00Z",
    }
    registry = {"version": 1, "repos": {name_with_owner: entry}}
    (fleet_dir / "fleet.json").write_text(json.dumps(registry), encoding="utf-8")


def _make_sibling_clone_ctx(sibling_root: Path, config: OrchestratorConfig) -> cli.CommandContext:
    """Build a CommandContext whose repo_root is the sibling clone."""
    from charlie_work.github import GitHub
    from charlie_work.paths import runtime_paths

    paths = runtime_paths(sibling_root, config.runtime.state_dir)
    gh = GitHub(repo_root=sibling_root, runtime=config.runtime, dry_run=True)
    return cli.CommandContext(repo_root=sibling_root, config=config, paths=paths, gh=gh)


def test_sibling_clone_verdict_refused(tmp_path: Path) -> None:
    """Issue #1376 acceptance criterion #5: a state-affecting command run
    from a sibling clone (separate git repo, same GitHub remote) must refuse
    with a three-path error, never silently write to the clone's phantom
    ``.var`` tree."""
    remote_url = "https://github.com/test/canonical.git"
    canonical = _init_git_repo_with_origin(tmp_path / "canonical", remote_url)
    sibling = _init_git_repo_with_origin(tmp_path / "sibling", remote_url)

    fleet_dir = tmp_path / "fleet"
    canonical_state = canonical / ".var" / "charlie-work"
    _write_fleet_registry(fleet_dir, "test/canonical", canonical, canonical_state)

    config = OrchestratorConfig()
    ctx = _make_sibling_clone_ctx(sibling, config)
    args = cli.build_parser().parse_args(
        ["--fleet-dir", str(fleet_dir), "verdict", "--pr", "1", "--decision", "approved"]
    )

    with pytest.raises(ConfigError) as excinfo:
        cli._assert_not_sibling_clone(ctx, args)

    message = str(excinfo.value)
    # The three required paths: cwd, would-be state root, canonical root.
    assert str(ctx.paths.root) in message
    assert str(canonical.resolve()) in message
    # The corrective invocation.
    assert "--repo" in message


def test_sibling_clone_merge_authorize_refused(tmp_path: Path) -> None:
    """Issue #1376 acceptance criterion #2: merge-authorize is also guarded."""
    remote_url = "https://github.com/test/canonical.git"
    canonical = _init_git_repo_with_origin(tmp_path / "canonical", remote_url)
    sibling = _init_git_repo_with_origin(tmp_path / "sibling", remote_url)

    fleet_dir = tmp_path / "fleet"
    _write_fleet_registry(
        fleet_dir, "test/canonical", canonical, canonical / ".var" / "charlie-work"
    )

    config = OrchestratorConfig()
    ctx = _make_sibling_clone_ctx(sibling, config)
    args = cli.build_parser().parse_args(
        ["--fleet-dir", str(fleet_dir), "merge-authorize", "1", "--reason", "ok"]
    )

    with pytest.raises(ConfigError):
        cli._assert_not_sibling_clone(ctx, args)


def test_sibling_clone_unescalate_refused(tmp_path: Path) -> None:
    """Issue #1376 acceptance criterion #2: unescalate is also guarded."""
    remote_url = "https://github.com/test/canonical.git"
    canonical = _init_git_repo_with_origin(tmp_path / "canonical", remote_url)
    sibling = _init_git_repo_with_origin(tmp_path / "sibling", remote_url)

    fleet_dir = tmp_path / "fleet"
    _write_fleet_registry(
        fleet_dir, "test/canonical", canonical, canonical / ".var" / "charlie-work"
    )

    config = OrchestratorConfig()
    ctx = _make_sibling_clone_ctx(sibling, config)
    args = cli.build_parser().parse_args(
        ["--fleet-dir", str(fleet_dir), "unescalate", "--pr", "1"]
    )

    with pytest.raises(ConfigError):
        cli._assert_not_sibling_clone(ctx, args)


def test_canonical_repo_verdict_allowed(tmp_path: Path) -> None:
    """Issue #1376: the guard must NOT fire when cwd is the canonical repo
    itself — the registered root matches the resolved root."""
    remote_url = "https://github.com/test/canonical.git"
    canonical = _init_git_repo_with_origin(tmp_path / "canonical", remote_url)

    fleet_dir = tmp_path / "fleet"
    _write_fleet_registry(
        fleet_dir, "test/canonical", canonical, canonical / ".var" / "charlie-work"
    )

    config = OrchestratorConfig()
    ctx = _make_sibling_clone_ctx(canonical, config)
    args = cli.build_parser().parse_args(
        ["--fleet-dir", str(fleet_dir), "verdict", "--pr", "1", "--decision", "approved"]
    )

    cli._assert_not_sibling_clone(ctx, args)  # must not raise


def test_linked_worktree_verdict_allowed_via_canonical_resolution(
    tmp_path: Path,
) -> None:
    """Issue #1376 acceptance criterion #1: a verdict run from a linked
    worktree of the canonical repo must NOT be refused — ``find_repo_root``
    resolves the linked worktree to the shared main checkout, so the
    resolved root matches the registered root."""
    import subprocess

    from charlie_work.paths import find_repo_root

    remote_url = "https://github.com/test/canonical.git"
    canonical = _init_git_repo_with_origin(tmp_path / "canonical", remote_url)

    # Create a linked worktree.
    linked_wt = tmp_path / "linked-wt"
    subprocess.run(
        ["git", "worktree", "add", "-b", "feature/test", str(linked_wt), "HEAD"],
        cwd=canonical,
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        fleet_dir = tmp_path / "fleet"
        _write_fleet_registry(
            fleet_dir, "test/canonical", canonical, canonical / ".var" / "charlie-work"
        )

        # find_repo_root from the linked worktree resolves to the main checkout.
        resolved = find_repo_root(linked_wt)
        assert resolved == canonical.resolve()

        config = OrchestratorConfig()
        ctx = _make_sibling_clone_ctx(resolved, config)
        args = cli.build_parser().parse_args(
            ["--fleet-dir", str(fleet_dir), "verdict", "--pr", "1", "--decision", "approved"]
        )

        cli._assert_not_sibling_clone(ctx, args)  # must not raise
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(linked_wt)],
            cwd=canonical,
            check=True,
            capture_output=True,
            text=True,
        )


def test_explicit_repo_skips_sibling_clone_guard(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Issue #1376 acceptance criterion #4: explicit ``--repo`` overrides
    the guard — the operator named the repo, so the resolution is
    intentional and the guard is not called."""
    remote_url = "https://github.com/test/canonical.git"
    canonical = _init_git_repo_with_origin(tmp_path / "canonical", remote_url)
    sibling = _init_git_repo_with_origin(tmp_path / "sibling", remote_url)

    fleet_dir = tmp_path / "fleet"
    _write_fleet_registry(
        fleet_dir, "test/canonical", canonical, canonical / ".var" / "charlie-work"
    )

    # build_app with --repo pointing at the sibling clone must NOT call the
    # guard.  We verify by monkeypatching the guard to raise if called.
    config = OrchestratorConfig()

    def _guard_should_not_fire(ctx, args):  # noqa: ANN001
        raise AssertionError("guard should not fire when --repo is explicit")

    monkeypatch.setattr(cli, "_assert_not_sibling_clone", _guard_should_not_fire)
    monkeypatch.setattr(cli, "find_repo_root", lambda repo, explicit=False, **kw: sibling)
    monkeypatch.setattr(cli, "load_layered_config", lambda *a, **k: config)
    monkeypatch.setattr(cli, "GitHub", _FakeGitHub)
    monkeypatch.setattr(cli, "touch_repo", lambda *a, **k: {})

    args = cli.build_parser().parse_args(
        [
            "--repo",
            str(sibling),
            "--fleet-dir",
            str(fleet_dir),
            "verdict",
            "--pr",
            "1",
            "--decision",
            "approved",
        ]
    )
    # Must not raise — the guard was skipped because args.repo is not None.
    cli.build_app(args)


def test_guard_fails_open_for_unregistered_repo(tmp_path: Path) -> None:
    """Issue #1376: when the repo is not in the fleet registry (fresh
    install), the guard must allow the command — there is no canonical root
    to compare against."""
    remote_url = "https://github.com/test/fresh.git"
    repo = _init_git_repo_with_origin(tmp_path / "fresh", remote_url)

    fleet_dir = tmp_path / "fleet"
    # No fleet.json written — registry is empty.

    config = OrchestratorConfig()
    ctx = _make_sibling_clone_ctx(repo, config)
    args = cli.build_parser().parse_args(
        ["--fleet-dir", str(fleet_dir), "verdict", "--pr", "1", "--decision", "approved"]
    )

    cli._assert_not_sibling_clone(ctx, args)  # must not raise


def test_guard_fails_open_for_no_origin_remote(tmp_path: Path) -> None:
    """Issue #1376: when the repo has no ``origin`` remote (or a non-GitHub
    URL), ``nameWithOwner`` cannot be resolved and the guard must allow the
    command — the guard is for GitHub-fleet repos, not arbitrary checkouts."""
    from _helpers import _init_git_repo

    repo = tmp_path / "no-remote"
    _init_git_repo(repo)

    fleet_dir = tmp_path / "fleet"
    _write_fleet_registry(
        fleet_dir, "test/canonical", tmp_path / "canonical", tmp_path / "canonical" / ".var"
    )

    config = OrchestratorConfig()
    ctx = _make_sibling_clone_ctx(repo, config)
    args = cli.build_parser().parse_args(
        ["--fleet-dir", str(fleet_dir), "verdict", "--pr", "1", "--decision", "approved"]
    )

    cli._assert_not_sibling_clone(ctx, args)  # must not raise


def test_guard_fails_open_for_stale_registered_root(tmp_path: Path) -> None:
    """Issue #1376: when the registered ``repo_root`` no longer exists, the
    guard must allow — a stale entry is not evidence of a sibling clone."""
    remote_url = "https://github.com/test/canonical.git"
    sibling = _init_git_repo_with_origin(tmp_path / "sibling", remote_url)

    fleet_dir = tmp_path / "fleet"
    # Register a root that doesn't exist.
    _write_fleet_registry(
        fleet_dir, "test/canonical", tmp_path / "deleted", tmp_path / "deleted" / ".var"
    )

    config = OrchestratorConfig()
    ctx = _make_sibling_clone_ctx(sibling, config)
    args = cli.build_parser().parse_args(
        ["--fleet-dir", str(fleet_dir), "verdict", "--pr", "1", "--decision", "approved"]
    )

    cli._assert_not_sibling_clone(ctx, args)  # must not raise


def test_read_only_command_skips_guard(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Issue #1376 acceptance criterion #2: read-only commands (e.g.
    ``roll-call``/status) must NOT trigger the sibling-clone guard."""
    remote_url = "https://github.com/test/canonical.git"
    canonical = _init_git_repo_with_origin(tmp_path / "canonical", remote_url)
    sibling = _init_git_repo_with_origin(tmp_path / "sibling", remote_url)

    fleet_dir = tmp_path / "fleet"
    _write_fleet_registry(
        fleet_dir, "test/canonical", canonical, canonical / ".var" / "charlie-work"
    )

    config = OrchestratorConfig()

    def _guard_should_not_fire(ctx, args):  # noqa: ANN001
        raise AssertionError("guard should not fire for read-only commands")

    monkeypatch.setattr(cli, "_assert_not_sibling_clone", _guard_should_not_fire)
    monkeypatch.setattr(cli, "find_repo_root", lambda repo, explicit=False, **kw: sibling)
    monkeypatch.setattr(cli, "load_layered_config", lambda *a, **k: config)
    monkeypatch.setattr(cli, "GitHub", _FakeGitHub)
    monkeypatch.setattr(cli, "touch_repo", lambda *a, **k: {})

    args = cli.build_parser().parse_args(["--fleet-dir", str(fleet_dir), "roll-call"])
    # roll-call is not in _STATE_AFFECTING_COMMANDS, so the guard is skipped.
    cli.build_app(args)


def test_build_app_guard_no_attribute_error_when_command_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression test for build_app's guard-condition ordering (issue #1376).

    ``build_app`` is called directly — outside the full argparse pipeline — by
    tests and other callers that hand-build a ``Namespace`` without a
    ``command`` attribute (e.g. ``test_cli_build_app_registers_repo`` in
    ``test_charlie_work.py``).  The guard condition must not crash with
    ``AttributeError`` on ``args.command`` for those callers.

    The fix has two parts, both exercised here:
    - ``args.repo is None`` is checked FIRST (cheap, always present), so when
      ``--repo`` is explicit the ``and`` short-circuits before ``command`` is
      touched.
    - ``command`` is read via ``getattr(args, "command", None)`` so even when
      ``repo`` is None the absent attribute yields ``None`` (not in
      ``_STATE_AFFECTING_COMMANDS``) instead of raising.
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_ctx = cli.CommandContext(repo_root=tmp_path, config=config, paths=paths, gh=_FakeGitHub())
    monkeypatch.setattr(cli, "bootstrap_command", lambda args: fake_ctx)
    monkeypatch.setattr(cli, "touch_repo", lambda *a, **k: {})
    monkeypatch.setattr(cli, "OrchestratorApp", lambda *a, **k: object())

    def _guard_must_not_fire(ctx, args):  # noqa: ANN001
        raise AssertionError("guard must not fire when command attribute is absent")

    monkeypatch.setattr(cli, "_assert_not_sibling_clone", _guard_must_not_fire)

    # Case 1: explicit --repo (repo is not None), no command attribute.
    # The `args.repo is None` check short-circuits before `command` is read.
    args_explicit = argparse.Namespace(repo=tmp_path, config=None, dry_run=False, fleet_dir=None)
    # Must not raise AttributeError.
    cli.build_app(args_explicit)

    # Case 2: no --repo (repo is None), no command attribute.
    # getattr(args, "command", None) returns None, not in _STATE_AFFECTING_COMMANDS.
    args_no_repo = argparse.Namespace(repo=None, config=None, dry_run=False, fleet_dir=None)
    # Must not raise AttributeError.
    cli.build_app(args_no_repo)


def test_verdict_round_trip_linked_worktree_to_canonical_merge_check(
    tmp_path: Path,
) -> None:
    """Issue #1376 acceptance criterion #3: a verdict recorded via the
    canonical-resolution (linked-worktree) path must land in the canonical
    state root and be visible to a subsequent merge-check run from the
    canonical root.

    This is the round-trip the issue explicitly mandates: the worktree-resolved
    app and the canonical-root app share one ``state_file`` / ``prs`` tree
    (because ``find_repo_root`` resolves the linked worktree to the shared
    main checkout, issue #648), so a verdict written through the former is read
    by the latter — never silently stranded in a phantom worktree ``.var``.
    """
    import subprocess

    from charlie_work.paths import find_repo_root
    from charlie_work.state import load_state
    from charlie_work.workflow import OrchestratorApp

    remote_url = "https://github.com/test/canonical.git"
    canonical = _init_git_repo_with_origin(tmp_path / "canonical", remote_url)

    # Linked worktree of the canonical repo (the AC#1 shape).
    linked_wt = tmp_path / "linked-wt"
    subprocess.run(
        ["git", "worktree", "add", "-b", "feature/round-trip", str(linked_wt), "HEAD"],
        cwd=canonical,
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        # Canonical-resolution path: find_repo_root from the linked worktree
        # resolves to the shared main checkout, identical to resolving from
        # the canonical root itself.
        resolved_from_wt = find_repo_root(linked_wt)
        resolved_from_canonical = find_repo_root(canonical)
        assert resolved_from_wt == resolved_from_canonical == canonical.resolve()

        config = OrchestratorConfig()
        paths_wt = runtime_paths(resolved_from_wt, config.runtime.state_dir)
        paths_canonical = runtime_paths(resolved_from_canonical, config.runtime.state_dir)
        # The core round-trip invariant: one canonical state root for both.
        assert paths_wt.root == paths_canonical.root
        assert paths_wt.state_file == paths_canonical.state_file
        assert paths_wt.prs == paths_canonical.prs

        # Record a verdict through the worktree-resolved (canonical-targeting)
        # app — the "canonical-resolution path" the issue names.
        app_wt = OrchestratorApp(resolved_from_wt, paths_wt, config, _FakeGitHub())
        result = app_wt.record_review(
            1,
            "approved",
            verdict_provenance="operator_manual",
            allow_stale_head=True,
        )
        assert result.ok, f"verdict recording failed: {result.message}"

        # The decision file landed in the canonical prs tree, not a phantom.
        decision_path = paths_canonical.prs / "pr-1" / "review-decision.json"
        assert decision_path.exists(), "verdict must land in the canonical prs tree"
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        assert decision["decision"] == "approved"
        assert decision["reviewed_head_sha"] == "sha-abc"

        # The canonical state.json carries the verdict's durable PR record.
        state = load_state(paths_canonical.state_file)
        assert state["prs"]["1"]["decision"] == "approved"

        # A subsequent merge-check run FROM THE CANONICAL ROOT sees it.
        app_canonical = OrchestratorApp(
            resolved_from_canonical, paths_canonical, config, _FakeGitHub()
        )
        mc = app_canonical.merge_check(1)
        assert mc.ok is True
        assert mc.data["authorized"] is True
        assert mc.data["reason"] == "approved_at_head"
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(linked_wt)],
            cwd=canonical,
            check=True,
            capture_output=True,
            text=True,
        )


def test_touch_repo_refuses_sibling_clone_repoint_keeps_canonical(
    tmp_path: Path,
) -> None:
    """Issue #1376 / #1372: an unguarded command (merge-check, status,
    tripwire ack, ship-it, why-charlie-hate, ...) run from a sibling clone
    reaches ``touch_repo`` with the clone's own ``repo_root``.  Without the
    repoint guard, ``touch_repo`` would overwrite the canonical registry
    entry, and a subsequent verdict / merge-authorize / unescalate run from
    the TRUE canonical repo would then be *refused* by
    ``_assert_not_sibling_clone`` -- the guard compares against a registry
    that now names the clone as canonical, so the refusal inverts and blocks
    the legitimate canonical repo instead of the sibling clone.

    ``touch_repo`` must keep the canonical ``repo_root`` / ``state_dir`` and
    bump ``last_seen`` only when the existing root is still a live git repo
    of a different canonical root.  The moved-repo case (old root gone or no
    longer a git repo) stays unaffected.
    """
    from charlie_work.fleet_registry import touch_repo
    from charlie_work.github import GitHub

    class _NWOGitHub(GitHub):
        def name_with_owner(self) -> str:
            return "test/canonical"

    remote_url = "https://github.com/test/canonical.git"
    canonical = _init_git_repo_with_origin(tmp_path / "canonical", remote_url)
    sibling = _init_git_repo_with_origin(tmp_path / "sibling", remote_url)

    fleet_dir = tmp_path / "fleet"
    canonical_paths = runtime_paths(canonical, ".var/charlie-work")

    # Register the canonical repo first (as a real touch_repo would).
    touch_repo(str(fleet_dir), canonical, canonical_paths, _NWOGitHub(repo_root=canonical))
    fleet_json = fleet_dir / "fleet.json"
    registry = json.loads(fleet_json.read_text(encoding="utf-8"))
    assert registry["repos"]["test/canonical"]["repo_root"] == str(canonical)

    # An unguarded command run from the sibling clone reaches touch_repo with
    # the sibling's repo_root (same nameWithOwner, different git repo).
    sibling_paths = runtime_paths(sibling, ".var/charlie-work")
    touch_repo(str(fleet_dir), sibling, sibling_paths, _NWOGitHub(repo_root=sibling))

    # The registry must STILL point at the canonical root -- the sibling clone
    # did not repoint it.  state_dir stays canonical too; only last_seen moved.
    registry = json.loads(fleet_json.read_text(encoding="utf-8"))
    entry = registry["repos"]["test/canonical"]
    assert entry["repo_root"] == str(canonical)
    assert entry["state_dir"] == str(canonical_paths.root)
    assert entry["config_path"] == str(canonical / "orchestrator.config.yaml")

    # Consequently, a subsequent verdict from the TRUE canonical repo is NOT
    # refused by _assert_not_sibling_clone: the registry still names the
    # canonical root, so the guard's comparison succeeds.  This is the exact
    # inversion the finding names -- without the touch_repo mitigation the
    # guard would fire here against the canonical repo.
    config = OrchestratorConfig()
    ctx = _make_sibling_clone_ctx(canonical, config)
    args = cli.build_parser().parse_args(
        ["--fleet-dir", str(fleet_dir), "verdict", "--pr", "1", "--decision", "approved"]
    )
    cli._assert_not_sibling_clone(ctx, args)  # must not raise


def test_touch_repo_allows_repoint_when_old_root_no_longer_a_git_repo(
    tmp_path: Path,
) -> None:
    """Issue #1376: the sibling-clone repoint guard must NOT block a
    legitimate moved-repo repoint.  When the existing registered root is no
    longer a git repo (the moved-repo shape: old checkout deleted or stripped
    of ``.git``), ``touch_repo`` must update ``repo_root`` to the new path.
    This is the discriminator from the sibling-clone case: a moved repo's old
    root is gone or not a git repo, a sibling clone's old root is still live."""
    from charlie_work.fleet_registry import touch_repo
    from charlie_work.github import GitHub

    class _NWOGitHub(GitHub):
        def name_with_owner(self) -> str:
            return "test/canonical"

    remote_url = "https://github.com/test/canonical.git"
    old_root = _init_git_repo_with_origin(tmp_path / "old", remote_url)
    new_root = _init_git_repo_with_origin(tmp_path / "new", remote_url)

    fleet_dir = tmp_path / "fleet"
    old_paths = runtime_paths(old_root, ".var/charlie-work")
    touch_repo(str(fleet_dir), old_root, old_paths, _NWOGitHub(repo_root=old_root))

    # The repo was moved: the old checkout's .git is taken out of the way (it
    # is no longer a git repo), and the operator re-registers from the new
    # location.  Renaming rather than rmtree-ing .git avoids Windows
    # read-only-packfile PermissionError; find_repo_root only looks for a
    # directory named exactly ``.git``, so the rename is enough to make the
    # old root resolve as "not a git repo".
    import os

    os.rename(old_root / ".git", old_root / ".git-disabled")

    new_paths = runtime_paths(new_root, ".var/charlie-work")
    touch_repo(str(fleet_dir), new_root, new_paths, _NWOGitHub(repo_root=new_root))

    fleet_json = fleet_dir / "fleet.json"
    registry = json.loads(fleet_json.read_text(encoding="utf-8"))
    entry = registry["repos"]["test/canonical"]
    # The repoint is allowed because the old root is no longer a git repo.
    assert entry["repo_root"] == str(new_root)
    assert entry["state_dir"] == str(new_paths.root)


def test_touch_repo_steady_state_skips_canonical_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issue #1376 round-2 review: the sibling-clone repoint guard must NOT
    spawn git subprocesses (via ``_resolved_canonical_root``) on the
    steady-state path -- the canonical repo touching its own
    already-registered entry with the same ``repo_root`` string.  That path
    runs on nearly every CLI invocation while the fleet-wide ``state_lock``
    is held, so unconditionally resolving the canonical root there is a
    real lock-hold-time regression.

    The cheap string-equality short-circuit (``existing_root_str !=
    str(repo_root)``) skips the resolution when the strings match.  This
    test monkeypatches ``_resolved_canonical_root`` to fail the test if
    called on a steady-state touch, then confirms a second touch from the
    same canonical root (a) does not call it and (b) still bumps
    ``last_seen``.
    """
    import charlie_work.fleet_registry as fr
    from charlie_work.fleet_registry import touch_repo
    from charlie_work.github import GitHub

    class _NWOGitHub(GitHub):
        def name_with_owner(self) -> str:
            return "test/canonical"

    remote_url = "https://github.com/test/canonical.git"
    canonical = _init_git_repo_with_origin(tmp_path / "canonical", remote_url)

    fleet_dir = tmp_path / "fleet"
    canonical_paths = runtime_paths(canonical, ".var/charlie-work")

    # First touch registers the entry (no existing entry -> short-circuit
    # branch is skipped because existing_root_str is None).
    touch_repo(str(fleet_dir), canonical, canonical_paths, _NWOGitHub(repo_root=canonical))
    fleet_json = fleet_dir / "fleet.json"
    registry = json.loads(fleet_json.read_text(encoding="utf-8"))
    first_seen = registry["repos"]["test/canonical"]["first_seen"]
    first_last_seen = registry["repos"]["test/canonical"]["last_seen"]

    # Sentinel: _resolved_canonical_root must NOT be called on a steady-state
    # touch (same repo_root string as the registered entry).
    def _no_resolution_call(path: Path) -> Path | None:  # noqa: ANN001
        raise AssertionError(
            "_resolved_canonical_root must not be called on a steady-state "
            "touch (same repo_root string); the short-circuit should skip it "
            f"(called with {path!r})"
        )

    monkeypatch.setattr(fr, "_resolved_canonical_root", _no_resolution_call)

    # Second touch from the SAME canonical root with the SAME string --
    # the steady-state path.  Must not raise (short-circuit took effect)
    # and must still bump last_seen.
    touch_repo(str(fleet_dir), canonical, canonical_paths, _NWOGitHub(repo_root=canonical))

    registry = json.loads(fleet_json.read_text(encoding="utf-8"))
    entry = registry["repos"]["test/canonical"]
    assert entry["repo_root"] == str(canonical)
    assert entry["first_seen"] == first_seen
    assert entry["last_seen"] >= first_last_seen
