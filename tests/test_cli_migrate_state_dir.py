"""``charlie migrate-state-dir`` plan/apply/dry-run gating tests.

Split out of ``tests/test_cli.py`` (issue #1561, Track 1) -- bodies are
verbatim relocations; shared fakes live in ``tests/_cli_fixtures.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _cli_fixtures import _make_repo
from charlie_work import cli
from charlie_work.dirty_tree import DirtyTreeReport
from charlie_work.quiesce import QuiesceReport
from charlie_work.state_migration import (
    MigrationChild,
    MigrationOutcome,
    MigrationPlan,
)


# --------------------------------------------------------------------------
# migrate-state-dir
# --------------------------------------------------------------------------


def _refuse_to_call(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("must not be called on this path")


def _fake_migration_plan(tmp_path: Path, *, blocked: bool = False) -> MigrationPlan:
    src_root = tmp_path / "src-state"
    dst_root = tmp_path / "dst-state"
    if blocked:
        child = MigrationChild(
            name="dispatches",
            src_path=src_root / "dispatches",
            dst_path=dst_root / "dispatches",
            disposition="blocked",
            reasons=("2 registered git worktrees nested inside this child: a, b",),
            remediation=(f'git -C "{tmp_path}" worktree move "a" "{dst_root / "a"}"',),
        )
        return MigrationPlan(src_root=src_root, dst_root=dst_root, children=(child,))
    child = MigrationChild(
        name="issues",
        src_path=src_root / "issues",
        dst_path=dst_root / "issues",
        disposition="move",
    )
    return MigrationPlan(src_root=src_root, dst_root=dst_root, children=(child,))


def _migrate_args(repo: Path, *extra: str) -> argparse.Namespace:
    return cli.build_parser().parse_args(["--repo", str(repo), "migrate-state-dir", *extra])


def _clean_tree(**_kwargs: Any) -> DirtyTreeReport:
    """A dirty-tree checker stub that always reports a clean working tree."""
    return DirtyTreeReport(ok=True, dirty_paths=())


def test_migrate_state_dir_parser_defaults_plan_only_and_apply_flips_it() -> None:
    """Requirement 1: acting requires the explicit ``--apply`` opt-in."""
    parser = cli.build_parser()

    plan_only = parser.parse_args(["migrate-state-dir"])
    assert plan_only.apply is False
    assert plan_only.src is None
    assert plan_only.dst is None
    assert plan_only.quiesce_patterns is None
    assert plan_only.allow_dirty is False

    applied = parser.parse_args(["migrate-state-dir", "--apply"])
    assert applied.apply is True

    allow_dirty = parser.parse_args(["migrate-state-dir", "--apply", "--allow-dirty"])
    assert allow_dirty.allow_dirty is True


def test_migrate_state_dir_src_equals_dst_reports_already_migrated_without_planning(
    tmp_path: Path,
) -> None:
    """A repo that never overrode ``runtime.state_dir`` has src == dst by
    default -- there is nothing to migrate, and the planner must never run.
    """
    repo = _make_repo(tmp_path)
    args = _migrate_args(repo)

    result = cli.run_migrate_state_dir_command(args, planner=_refuse_to_call)

    assert result.ok is True
    assert "already migrated" in result.message
    assert result.data["already_migrated"] is True

    # Both derived roots must be fully resolved, not just joined. If dst were a
    # bare join (as ``layout.default_state_root`` alone returns) while src goes
    # through ``runtime_paths`` (which calls ``.resolve()``), a repo whose
    # ``.var`` is a symlink/junction would make this comparison see two
    # different-looking paths for the same on-disk location -- the false
    # negative this short circuit exists to prevent.
    expected = (repo / ".var" / "charlie-work").resolve()
    assert Path(result.data["src_root"]) == expected
    assert Path(result.data["dst_root"]) == expected


def test_migrate_state_dir_overridden_runtime_state_dir_derives_distinct_roots(
    tmp_path: Path,
) -> None:
    """The actual scenario the command exists for: a repo whose
    ``orchestrator.config.yaml`` still points ``runtime.state_dir`` at a
    legacy location. With no ``--src``/``--dst`` given, derivation must
    produce two distinct, fully-resolved roots and hand them both to the
    planner -- exercising the real default derivation end-to-end rather than
    only the short-circuit-when-equal path covered above.
    """
    repo = tmp_path
    (repo / ".git").mkdir()
    legacy = repo / "legacy-orchestrator-state"
    legacy.mkdir()
    (legacy / "state.json").write_text(
        json.dumps({"version": 1, "issues": {}, "prs": {}, "events": []}),
        encoding="utf-8",
    )
    (repo / "orchestrator.config.yaml").write_text(
        "runtime:\n  state_dir: legacy-orchestrator-state\n", encoding="utf-8"
    )
    args = _migrate_args(repo)
    captured: dict[str, Path] = {}

    def _capture_and_plan(*, repo_root: Path, src_root: Path, dst_root: Path) -> MigrationPlan:
        captured["src_root"] = src_root
        captured["dst_root"] = dst_root
        return _fake_migration_plan(tmp_path)

    result = cli.run_migrate_state_dir_command(args, planner=_capture_and_plan)

    # Assert against what was actually handed to the planner -- not
    # ``result.data``, which reports the *plan's own* ``src_root``/``dst_root``
    # (here the fake plan's fixture paths, unrelated to the derivation this
    # test exercises).
    expected_src = legacy.resolve()
    expected_dst = (repo / ".var" / "charlie-work").resolve()
    assert captured["src_root"] == expected_src
    assert captured["dst_root"] == expected_dst
    assert captured["src_root"] != captured["dst_root"]
    assert "already_migrated" not in result.data


def test_migrate_state_dir_apply_refuses_without_any_quiesce_pattern(tmp_path: Path) -> None:
    """Fail-closed: no built-in pattern list exists (CLAUDE.md rule 9), so
    ``--apply`` with none supplied cannot prove quiescence and must refuse.
    """
    repo = _make_repo(tmp_path)
    src, dst = tmp_path / "src-state", tmp_path / "dst-state"
    args = _migrate_args(repo, "--src", str(src), "--dst", str(dst), "--apply")

    result = cli.run_migrate_state_dir_command(
        args,
        planner=lambda **kwargs: _fake_migration_plan(tmp_path),
        actuator=_refuse_to_call,
    )

    assert result.ok is False
    assert "no --quiesce-pattern given" in result.message


def test_migrate_state_dir_apply_refuses_when_quiesce_not_ok(tmp_path: Path) -> None:
    """Requirement 3: ``--apply`` calls ``check_quiescence`` and refuses to
    act when the fleet is not quiescent, saying exactly why.
    """
    repo = _make_repo(tmp_path)
    src, dst = tmp_path / "src-state", tmp_path / "dst-state"
    args = _migrate_args(
        repo,
        "--src",
        str(src),
        "--dst",
        str(dst),
        "--apply",
        "--quiesce-pattern",
        "fleet supervise",
    )
    not_quiescent = QuiesceReport(
        ok=False,
        matched=(),
        excluded_pids=frozenset(),
        summary="NOT quiescent: 1 matching process(es)",
    )

    result = cli.run_migrate_state_dir_command(
        args,
        planner=lambda **kwargs: _fake_migration_plan(tmp_path),
        quiescence_checker=lambda **kwargs: not_quiescent,
        actuator=_refuse_to_call,
    )

    assert result.ok is False
    assert "fleet is not quiescent" in result.message
    assert "NOT quiescent: 1 matching process(es)" in result.message


def test_migrate_state_dir_apply_refuses_when_plan_has_blocked_children(tmp_path: Path) -> None:
    """Requirement 5: on refusal, the blocked child's name, reasons, and
    remediation command lines must appear VERBATIM -- the operator runs them
    by hand.
    """
    repo = _make_repo(tmp_path)
    src, dst = tmp_path / "src-state", tmp_path / "dst-state"
    args = _migrate_args(
        repo,
        "--src",
        str(src),
        "--dst",
        str(dst),
        "--apply",
        "--quiesce-pattern",
        "fleet supervise",
    )
    plan = _fake_migration_plan(tmp_path, blocked=True)
    quiescent = QuiesceReport(ok=True, matched=(), excluded_pids=frozenset(), summary="quiescent")

    result = cli.run_migrate_state_dir_command(
        args,
        planner=lambda **kwargs: plan,
        quiescence_checker=lambda **kwargs: quiescent,
        dirty_tree_checker=_clean_tree,
    )

    assert result.ok is False
    (blocked_child,) = plan.blocked
    assert blocked_child.name in result.message
    for reason in blocked_child.reasons:
        assert reason in result.message
    for step in blocked_child.remediation:
        assert step in result.message


def test_migrate_state_dir_dry_run_succeeds_even_when_quiesce_not_ok(tmp_path: Path) -> None:
    """Requirement 3: a dry run must keep working on a live fleet -- quiesce
    is reported informationally only and never gates a non-``--apply`` run.
    """
    repo = _make_repo(tmp_path)
    src, dst = tmp_path / "src-state", tmp_path / "dst-state"
    args = _migrate_args(
        repo, "--src", str(src), "--dst", str(dst), "--quiesce-pattern", "fleet supervise"
    )
    assert args.apply is False
    not_quiescent = QuiesceReport(
        ok=False,
        matched=(),
        excluded_pids=frozenset(),
        summary="NOT quiescent: 1 matching process(es)",
    )

    result = cli.run_migrate_state_dir_command(
        args,
        planner=lambda **kwargs: _fake_migration_plan(tmp_path),
        quiescence_checker=lambda **kwargs: not_quiescent,
        actuator=_refuse_to_call,
    )

    assert result.ok is True
    assert "NOT quiescent" in result.message


def test_migrate_state_dir_dry_run_flag_overrides_apply_and_never_actuates(tmp_path: Path) -> None:
    """The global/subcommand ``--dry-run`` overrides ``--apply``, and this
    path never touches quiescence gating either -- it is still a non-acting
    path.
    """
    repo = _make_repo(tmp_path)
    src, dst = tmp_path / "src-state", tmp_path / "dst-state"
    args = _migrate_args(repo, "--src", str(src), "--dst", str(dst), "--apply", "--dry-run")

    result = cli.run_migrate_state_dir_command(
        args,
        planner=lambda **kwargs: _fake_migration_plan(tmp_path),
        actuator=_refuse_to_call,
    )

    assert result.ok is True
    assert "--apply ignored" in result.message


def test_migrate_state_dir_apply_happy_path_actuates_when_quiescent(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    src, dst = tmp_path / "src-state", tmp_path / "dst-state"
    args = _migrate_args(
        repo,
        "--src",
        str(src),
        "--dst",
        str(dst),
        "--apply",
        "--quiesce-pattern",
        "fleet supervise",
    )
    quiescent = QuiesceReport(ok=True, matched=(), excluded_pids=frozenset(), summary="quiescent")
    outcome = MigrationOutcome(ok=True, moved=("issues",), rewritten_paths=7)

    result = cli.run_migrate_state_dir_command(
        args,
        planner=lambda **kwargs: _fake_migration_plan(tmp_path),
        quiescence_checker=lambda **kwargs: quiescent,
        dirty_tree_checker=_clean_tree,
        actuator=lambda plan_arg: outcome,
    )

    assert result.ok is True
    assert "moved 1 children" in result.message
    # Issue #735: rewritten_paths is surfaced in both the data dict and the
    # human-readable message so the operator can see the rewrite happened.
    assert result.data["rewritten_paths"] == 7
    assert "rewrote 7 embedded paths" in result.message


def test_migrate_state_dir_apply_reports_zero_rewrites_in_message(tmp_path: Path) -> None:
    """When the rewrite found no embedded paths, the message still carries the
    ``rewrote 0 embedded paths`` suffix and ``rewritten_paths`` is 0 in data --
    a migration with no state.json or no embedded paths is a legitimate success.
    """
    repo = _make_repo(tmp_path)
    src, dst = tmp_path / "src-state", tmp_path / "dst-state"
    args = _migrate_args(
        repo,
        "--src",
        str(src),
        "--dst",
        str(dst),
        "--apply",
        "--quiesce-pattern",
        "fleet supervise",
    )
    quiescent = QuiesceReport(ok=True, matched=(), excluded_pids=frozenset(), summary="quiescent")
    outcome = MigrationOutcome(ok=True, moved=("issues",), rewritten_paths=0)

    result = cli.run_migrate_state_dir_command(
        args,
        planner=lambda **kwargs: _fake_migration_plan(tmp_path),
        quiescence_checker=lambda **kwargs: quiescent,
        dirty_tree_checker=_clean_tree,
        actuator=lambda plan_arg: outcome,
    )

    assert result.ok is True
    assert result.data["rewritten_paths"] == 0
    assert "rewrote 0 embedded paths" in result.message


def test_migrate_state_dir_apply_rewrite_failure_surfaces_in_data_and_message(
    tmp_path: Path,
) -> None:
    """Issue #735: when the state.json path rewrite fails, ``rewritten_paths``
    is 0 in the data dict and the failure message names the rewrite error --
    the children already moved, so this is an incomplete migration needing
    manual attention, not a rollback.
    """
    repo = _make_repo(tmp_path)
    src, dst = tmp_path / "src-state", tmp_path / "dst-state"
    args = _migrate_args(
        repo,
        "--src",
        str(src),
        "--dst",
        str(dst),
        "--apply",
        "--quiesce-pattern",
        "fleet supervise",
    )
    quiescent = QuiesceReport(ok=True, matched=(), excluded_pids=frozenset(), summary="quiescent")
    outcome = MigrationOutcome(
        ok=False,
        moved=("issues",),
        rewritten_paths=0,
        error="children moved but state.json path rewrite failed: missing target",
    )

    result = cli.run_migrate_state_dir_command(
        args,
        planner=lambda **kwargs: _fake_migration_plan(tmp_path),
        quiescence_checker=lambda **kwargs: quiescent,
        dirty_tree_checker=_clean_tree,
        actuator=lambda plan_arg: outcome,
    )

    assert result.ok is False
    assert result.data["rewritten_paths"] == 0
    assert result.data["applied"] is False
    assert "migration failed after 1 moved" in result.message
    assert "path rewrite failed" in result.message
    assert "missing target" in result.message


def test_migrate_state_dir_apply_refuses_when_working_tree_is_dirty(tmp_path: Path) -> None:
    """Issue #729: ``--apply`` executes the working tree, but CI only reviewed
    the committed tree. Refuse to actuate when the tracked working tree differs
    from HEAD, naming the divergent paths so the operator sees *what* changed.
    """
    repo = _make_repo(tmp_path)
    src, dst = tmp_path / "src-state", tmp_path / "dst-state"
    args = _migrate_args(
        repo,
        "--src",
        str(src),
        "--dst",
        str(dst),
        "--apply",
        "--quiesce-pattern",
        "fleet supervise",
    )
    quiescent = QuiesceReport(ok=True, matched=(), excluded_pids=frozenset(), summary="quiescent")
    dirty = DirtyTreeReport(ok=True, dirty_paths=("src/charlie_work/state_migration.py",))

    result = cli.run_migrate_state_dir_command(
        args,
        planner=lambda **kwargs: _fake_migration_plan(tmp_path),
        quiescence_checker=lambda **kwargs: quiescent,
        dirty_tree_checker=lambda **kwargs: dirty,
        actuator=_refuse_to_call,
    )

    assert result.ok is False
    assert "refusing to apply" in result.message
    assert "tracked working tree differs from HEAD" in result.message
    assert "src/charlie_work/state_migration.py" in result.message
    assert "--allow-dirty" in result.message
    assert result.data["applied"] is False


def test_migrate_state_dir_apply_refuses_when_dirty_tree_probe_fails(tmp_path: Path) -> None:
    """Issue #729: a probe that cannot determine cleanliness is not evidence of
    cleanliness -- fail closed, never silently proceed. The refusal message
    carries the probe's error so the operator can diagnose the git failure.
    """
    repo = _make_repo(tmp_path)
    src, dst = tmp_path / "src-state", tmp_path / "dst-state"
    args = _migrate_args(
        repo,
        "--src",
        str(src),
        "--dst",
        str(dst),
        "--apply",
        "--quiesce-pattern",
        "fleet supervise",
    )
    quiescent = QuiesceReport(ok=True, matched=(), excluded_pids=frozenset(), summary="quiescent")
    probe_failed = DirtyTreeReport(
        ok=False, error="could not check working tree cleanliness: git status failed"
    )

    result = cli.run_migrate_state_dir_command(
        args,
        planner=lambda **kwargs: _fake_migration_plan(tmp_path),
        quiescence_checker=lambda **kwargs: quiescent,
        dirty_tree_checker=lambda **kwargs: probe_failed,
        actuator=_refuse_to_call,
    )

    assert result.ok is False
    assert "refusing to apply" in result.message
    assert "could not check working tree cleanliness" in result.message
    assert result.data["applied"] is False


def test_migrate_state_dir_apply_allow_dirty_overrides_clean_tree_gate(tmp_path: Path) -> None:
    """Issue #729: ``--allow-dirty`` is the explicit override for deliberate
    local testing on a dirty tree. With it set, a dirty working tree does NOT
    block actuation -- the operator opted in.
    """
    repo = _make_repo(tmp_path)
    src, dst = tmp_path / "src-state", tmp_path / "dst-state"
    args = _migrate_args(
        repo,
        "--src",
        str(src),
        "--dst",
        str(dst),
        "--apply",
        "--allow-dirty",
        "--quiesce-pattern",
        "fleet supervise",
    )
    quiescent = QuiesceReport(ok=True, matched=(), excluded_pids=frozenset(), summary="quiescent")
    dirty = DirtyTreeReport(ok=True, dirty_paths=("src/charlie_work/state_migration.py",))
    outcome = MigrationOutcome(ok=True, moved=("issues",))

    result = cli.run_migrate_state_dir_command(
        args,
        planner=lambda **kwargs: _fake_migration_plan(tmp_path),
        quiescence_checker=lambda **kwargs: quiescent,
        dirty_tree_checker=lambda **kwargs: dirty,
        actuator=lambda plan_arg: outcome,
    )

    assert result.ok is True
    assert "moved 1 children" in result.message


def test_migrate_state_dir_plan_only_does_not_check_dirty_tree(tmp_path: Path) -> None:
    """Issue #729: plan-only paths stay usable on a dirty tree -- iterating on
    a plan is the normal development loop, so the clean-tree gate must not fire
    without ``--apply``.
    """
    repo = _make_repo(tmp_path)
    src, dst = tmp_path / "src-state", tmp_path / "dst-state"
    args = _migrate_args(repo, "--src", str(src), "--dst", str(dst))
    assert args.apply is False

    result = cli.run_migrate_state_dir_command(
        args,
        planner=lambda **kwargs: _fake_migration_plan(tmp_path),
        dirty_tree_checker=lambda **kwargs: DirtyTreeReport(
            ok=True, dirty_paths=("src/charlie_work/state_migration.py",)
        ),
        actuator=_refuse_to_call,
    )

    assert result.ok is True
    assert "plan only" in result.message
