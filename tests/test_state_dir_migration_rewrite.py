"""Issue #735 embedded-path rewrite tests for ``charlie_work.state_migration``.

Split out of ``tests/test_state_dir_migration.py`` (issue #1566, Track-1):
``apply_state_dir_migration`` moves the tree but must also rewrite every
embedded absolute path inside ``state.json`` so the file is internally
consistent with its new location. Covers ``_try_rewrite_path_string``,
``_walk_and_rewrite``, ``_rewrite_state_json_paths``, and the apply-side
integration (count reporting, injected ``state_rewriter`` seam, and the
refusal when a hit's rewritten target does not exist).
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from charlie_work.state import StateLockBusy
from charlie_work.state_migration import (
    StateRewriteResult,
    _rewrite_state_json_paths,
    _try_rewrite_path_string,
    _walk_and_rewrite,
    apply_state_dir_migration,
    plan_state_dir_migration,
)


# ---------------------------------------------------------------------------
# _try_rewrite_path_string -- the exact-prefix hit test
# ---------------------------------------------------------------------------


def test_try_rewrite_path_string_non_path_string_is_not_a_hit() -> None:
    """A string that is not under src_root is returned unchanged, count 0."""
    src = Path("C:/repos/x/.var/old")
    dst = Path("C:/repos/x/.var/new")
    result, count, error = _try_rewrite_path_string("not-a-path", src, dst)
    assert result == "not-a-path"
    assert count == 0
    assert error is None


def test_try_rewrite_path_string_issue_title_containing_prefix_is_not_a_hit(
    tmp_path: Path,
) -> None:
    """An issue title that merely *contains* the old-root string but does not
    start with it is not a hit -- the exact-prefix check rejects it, exactly
    the hazard a ``str.replace`` would get wrong.
    """
    src = tmp_path / "old-state"
    dst = tmp_path / "new-state"
    src.mkdir()
    dst.mkdir()
    title = f"Fix bug in {src} module"
    result, count, error = _try_rewrite_path_string(title, src, dst)
    assert result == title
    assert count == 0
    assert error is None


def test_try_rewrite_path_string_sibling_prefix_is_not_a_hit(tmp_path: Path) -> None:
    """A path like ``<src>-backup/...`` is not under ``src`` -- the exact-prefix
    check (separator after the root) rejects it, while a bare ``startswith``
    would wrongly match.
    """
    src = tmp_path / "old-state"
    dst = tmp_path / "new-state"
    src.mkdir()
    dst.mkdir()
    sibling = src.parent / "old-state-backup"
    sibling.mkdir()
    candidate = str(sibling / "file.txt")
    result, count, error = _try_rewrite_path_string(candidate, src, dst)
    assert result == candidate
    assert count == 0
    assert error is None


# ---------------------------------------------------------------------------
# _walk_and_rewrite -- the structural walk
# ---------------------------------------------------------------------------


def test_walk_and_rewrite_preserves_non_string_values() -> None:
    """Ints, floats, bools, None, and dict keys are passed through untouched."""
    src = Path("C:/repos/x/.var/old")
    dst = Path("C:/repos/x/.var/new")
    data: dict[str, Any] = {
        "count": 42,
        "ratio": 3.14,
        "flag": True,
        "nothing": None,
        "title": "some issue title",
    }
    new_data, count, error = _walk_and_rewrite(data, src, dst)
    assert error is None
    assert count == 0
    assert new_data == data


def test_walk_and_rewrite_does_not_touch_dict_keys(tmp_path: Path) -> None:
    """Dict keys are never rewritten -- only string *values* are candidates."""
    src = tmp_path / "old-state"
    dst = tmp_path / "new-state"
    src.mkdir()
    dst.mkdir()
    # The key "old-state" is not a path under src_root; the value is.
    target_file = src / "dispatches" / "prompt.md"
    target_file.parent.mkdir(parents=True)
    target_file.write_text("x")
    dst_target = dst / "dispatches" / "prompt.md"
    dst_target.parent.mkdir(parents=True)
    dst_target.write_text("x")
    data = {"old-state": str(target_file)}
    new_data, count, error = _walk_and_rewrite(data, src, dst)
    assert error is None
    assert count == 1
    assert "old-state" in new_data  # key unchanged
    assert new_data["old-state"] == str(dst_target)


def test_walk_and_rewrite_rewrites_path_strings_inside_non_empty_list(tmp_path: Path) -> None:
    """The list-handling branch of ``_walk_and_rewrite`` must descend into a
    non-empty list and rewrite path-string elements under ``src_root``.

    Real event payloads store paths inside the ``events`` array -- a list of
    event dicts whose ``data`` values include path strings like
    ``reviews_dir`` (see ``workflow._mark_review_checkout_removal_failed``).
    A walk that only recursed into dicts would miss every path embedded in
    that list; this test pins the list branch with the exact shape: a list of
    event dicts, each carrying a path-string value under the old root.
    """
    src = tmp_path / "old-state"
    dst = tmp_path / "new-state"
    src.mkdir()
    dst.mkdir()
    # Two event dicts, each with a ``reviews_dir`` path under src_root -- the
    # real ``review_checkout_removal_failed`` event payload shape.
    reviews_dir_1 = src / "dispatches" / "reviews" / "pr-1"
    reviews_dir_2 = src / "dispatches" / "reviews" / "pr-2"
    reviews_dir_1.mkdir(parents=True)
    reviews_dir_2.mkdir(parents=True)
    dst_reviews_dir_1 = dst / "dispatches" / "reviews" / "pr-1"
    dst_reviews_dir_2 = dst / "dispatches" / "reviews" / "pr-2"
    dst_reviews_dir_1.mkdir(parents=True)
    dst_reviews_dir_2.mkdir(parents=True)
    data = {
        "version": 1,
        "events": [
            {
                "kind": "review_checkout_removal_failed",
                "data": {"reviews_dir": str(reviews_dir_1)},
            },
            {
                "kind": "review_checkout_removal_failed",
                "data": {"reviews_dir": str(reviews_dir_2)},
            },
        ],
    }
    new_data, count, error = _walk_and_rewrite(data, src, dst)
    assert error is None
    assert count == 2
    events = new_data["events"]
    assert len(events) == 2
    assert events[0]["data"]["reviews_dir"] == str(dst_reviews_dir_1)
    assert events[1]["data"]["reviews_dir"] == str(dst_reviews_dir_2)
    # Non-path list elements (e.g. an event with no path) are passed through.
    assert events[0]["kind"] == "review_checkout_removal_failed"


def test_walk_and_rewrite_list_branch_propagates_rewrite_failure(tmp_path: Path) -> None:
    """A path string inside a list whose rewritten target does not exist must
    fail the whole walk -- the caller must not save a partially-rewritten tree.

    Exercises the list branch's error-return path (the ``if error is not
    None: return value, 0, error`` inside the list loop), which a walk that
    only tested dict-level failures would never reach.
    """
    src = tmp_path / "old-state"
    dst = tmp_path / "new-state"
    src.mkdir()
    dst.mkdir()
    # A path under src_root whose target under dst_root does NOT exist -- the
    # existence check in ``_try_rewrite_path_string`` must refuse it.
    bogus = src / "dispatches" / "missing.md"
    bogus.parent.mkdir(parents=True)
    data = {
        "version": 1,
        "events": [
            {"kind": "review_checkout_removal_failed", "data": {"reviews_dir": str(bogus)}},
        ],
    }
    new_data, count, error = _walk_and_rewrite(data, src, dst)
    assert error is not None
    assert "does not exist" in error
    assert count == 0
    # On error, the original value is returned unchanged -- no partial rewrite.
    assert new_data == data


# ---------------------------------------------------------------------------
# _rewrite_state_json_paths -- locked load/save cycle
# ---------------------------------------------------------------------------


def test_rewrite_state_json_paths_missing_file_is_ok_zero(tmp_path: Path) -> None:
    """A missing state.json is not an error -- nothing to rewrite."""
    state_path = tmp_path / "nonexistent-state.json"
    result = _rewrite_state_json_paths(state_path, tmp_path / "old", tmp_path / "new")
    assert result.ok is True
    assert result.rewritten == 0
    assert result.error is None


def test_rewrite_state_json_paths_returns_ok_false_on_state_lock_busy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """StateLockBusy from the state lock is caught and returned as ok=False
    with .error set -- never propagated to the caller.

    Pins the ``except StateLockBusy`` branch in ``_rewrite_state_json_paths``
    (CLAUDE.md invariant: errors from external processes come back as values,
    never raised). The branch had zero test coverage; this forces it by
    monkeypatching ``state_lock`` to raise ``StateLockBusy`` on entry, so the
    ``with state_lock(...)`` statement never acquires the lock.
    """
    src_root = tmp_path / "old-state"
    dst_root = tmp_path / "new-state"
    src_root.mkdir()
    dst_root.mkdir()
    # A state.json must exist so the function does not short-circuit on the
    # missing-file early return.
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({"version": 1}), encoding="utf-8")

    @contextmanager
    def raising_lock(path: Path) -> Any:
        raise StateLockBusy("lock held by another process")
        yield  # pragma: no cover -- makes this a generator like the real state_lock

    monkeypatch.setattr("charlie_work.state_migration.state_lock", raising_lock)

    result = _rewrite_state_json_paths(state_path, src_root, dst_root)

    assert result.ok is False
    assert result.rewritten == 0
    assert result.error is not None
    assert "could not acquire state lock" in result.error


def test_rewrite_state_json_paths_returns_ok_false_on_oserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """OSError during the load/save cycle is caught and returned as ok=False
    with .error set -- never propagated to the caller.

    Pins the ``except OSError`` branch in ``_rewrite_state_json_paths``
    (CLAUDE.md invariant: errors from external processes come back as values,
    never raised). The branch had zero test coverage; this forces it by
    monkeypatching ``load_state`` to raise ``OSError`` inside the lock, so the
    I/O error surfaces through the same try/except that guards the save path.
    """
    src_root = tmp_path / "old-state"
    dst_root = tmp_path / "new-state"
    src_root.mkdir()
    dst_root.mkdir()
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({"version": 1}), encoding="utf-8")

    def raising_load(path: Path) -> dict[str, Any]:
        raise OSError("disk I/O error")

    monkeypatch.setattr("charlie_work.state_migration.load_state", raising_load)

    result = _rewrite_state_json_paths(state_path, src_root, dst_root)

    assert result.ok is False
    assert result.rewritten == 0
    assert result.error is not None
    assert "state rewrite I/O error" in result.error


# ---------------------------------------------------------------------------
# apply_state_dir_migration -- state.json rewrite integration
# ---------------------------------------------------------------------------


def _make_state_with_embedded_paths(src_root: Path, *, pr_count: int = 1) -> dict[str, Any]:
    """Build a state.json-shaped dict with embedded absolute paths under src_root.

    Mirrors the real field names from the job-cannon incident: ``prompt_path``,
    ``decision_path``, ``cross_family_report``, and ``verdict_source``. Each
    path points at a file that exists under ``src_root`` so the post-move
    existence check passes.
    """
    prs: dict[str, Any] = {}
    for i in range(1, pr_count + 1):
        pr_dir = src_root / "dispatches" / "reviews" / f"pr-{i}"
        pr_dir.mkdir(parents=True, exist_ok=True)
        prompt = pr_dir / "prompt.md"
        prompt.write_text("prompt")
        decision = pr_dir / "decision.md"
        decision.write_text("decision")
        cross_family = pr_dir / "cross-family-report.md"
        cross_family.write_text("report")
        prs[str(i)] = {
            "number": i,
            "title": f"PR {i} -- has {src_root} in title for non-hit test",
            "prompt_path": str(prompt),
            "decision_path": str(decision),
            "cross_family_report": str(cross_family),
            "verdict_source": str(decision),
        }
    return {
        "version": 1,
        "issues": {"101": {"number": 101, "title": "some issue"}},
        "prs": prs,
        "events": [],
    }


def test_apply_rewrites_embedded_paths_in_state_json(tmp_path: Path) -> None:
    """The core #735 regression: after moving children, every embedded absolute
    path inside state.json is rewritten from the old root to the new one, the
    count is reported in ``rewritten_paths``, and the file on disk reflects
    the rewrite.
    """
    src_root = tmp_path / "src-state"
    dst_root = tmp_path / "dst-state"
    src_root.mkdir()

    state_data = _make_state_with_embedded_paths(src_root, pr_count=2)
    state_file = src_root / "state.json"
    state_file.write_text(json.dumps(state_data), encoding="utf-8")

    plan = plan_state_dir_migration(
        repo_root=tmp_path,
        src_root=src_root,
        dst_root=dst_root,
        src_children=list(src_root.iterdir()),
        dst_names=[],
        registered_worktrees=[],
    )
    assert plan.blocked == ()

    outcome = apply_state_dir_migration(plan)

    assert outcome.ok is True
    # 2 PRs × 4 path fields each = 8 rewrites
    assert outcome.rewritten_paths == 8
    assert outcome.error is None

    # Verify the file on disk: every path field now names dst_root.
    rewritten = json.loads((dst_root / "state.json").read_text(encoding="utf-8"))
    for pr_data in rewritten["prs"].values():
        for field in ("prompt_path", "decision_path", "cross_family_report", "verdict_source"):
            value = pr_data[field]
            assert str(dst_root) in value, f"{field} still names old root: {value}"
            assert str(src_root) not in value, f"{field} still names old root: {value}"
            assert Path(value).exists(), f"{field} rewritten path does not exist: {value}"

    # Non-path strings (issue/PR titles) are untouched even if they contain
    # the old-root string -- the exact-prefix check rejects them.
    for pr_data in rewritten["prs"].values():
        assert str(src_root) in pr_data["title"], "title should be unchanged"

    # Key sets are byte-identical before and after (the walk must not add or
    # drop keys -- mirrors the job-cannon remediation verification).
    assert set(rewritten["prs"]) == set(state_data["prs"])
    assert set(rewritten["issues"]) == set(state_data["issues"])


def test_apply_no_state_json_reports_zero_rewrites(tmp_path: Path) -> None:
    """When there is no state.json among the moved children, rewritten_paths
    is 0 and the migration still succeeds.
    """
    src_root = tmp_path / "src-state"
    dst_root = tmp_path / "dst-state"
    src_root.mkdir()
    (src_root / "issues").mkdir()
    (src_root / "logs").mkdir()

    plan = plan_state_dir_migration(
        repo_root=tmp_path,
        src_root=src_root,
        dst_root=dst_root,
        src_children=[src_root / "issues", src_root / "logs"],
        dst_names=[],
        registered_worktrees=[],
    )

    outcome = apply_state_dir_migration(plan)

    assert outcome.ok is True
    assert outcome.rewritten_paths == 0


def test_apply_state_json_with_no_embedded_paths_reports_zero(tmp_path: Path) -> None:
    """A state.json with no paths under the old root yields 0 rewrites but
    is still a successful migration.
    """
    src_root = tmp_path / "src-state"
    dst_root = tmp_path / "dst-state"
    src_root.mkdir()
    (src_root / "state.json").write_text(
        json.dumps({"version": 1, "issues": {}, "prs": {}, "events": []}),
        encoding="utf-8",
    )

    plan = plan_state_dir_migration(
        repo_root=tmp_path,
        src_root=src_root,
        dst_root=dst_root,
        src_children=[src_root / "state.json"],
        dst_names=[],
        registered_worktrees=[],
    )

    outcome = apply_state_dir_migration(plan)

    assert outcome.ok is True
    assert outcome.rewritten_paths == 0


def test_apply_rewrite_failure_returns_ok_false_with_moved(tmp_path: Path) -> None:
    """If the state rewrite fails (a hit whose rewritten target does not
    exist), the outcome is ``ok=False`` with ``moved`` listing what was
    moved -- the children are already on the new root, so this is an
    incomplete migration needing manual attention, not a rollback.
    """
    src_root = tmp_path / "src-state"
    dst_root = tmp_path / "dst-state"
    src_root.mkdir()

    # Create a state.json with a path under src_root, but do NOT create the
    # file it points at. After the move, the rewritten path will not exist.
    bogus_path = src_root / "dispatches" / "missing.md"
    state_data = {
        "version": 1,
        "issues": {},
        "prs": {"1": {"prompt_path": str(bogus_path)}},
        "events": [],
    }
    (src_root / "state.json").write_text(json.dumps(state_data), encoding="utf-8")

    plan = plan_state_dir_migration(
        repo_root=tmp_path,
        src_root=src_root,
        dst_root=dst_root,
        src_children=[src_root / "state.json"],
        dst_names=[],
        registered_worktrees=[],
    )

    outcome = apply_state_dir_migration(plan)

    assert outcome.ok is False
    assert outcome.moved == ("state.json",)
    assert outcome.rewritten_paths == 0
    assert outcome.error is not None
    assert "path rewrite failed" in outcome.error
    assert "does not exist" in outcome.error


def test_apply_uses_injected_state_rewriter_seam(tmp_path: Path) -> None:
    """The ``state_rewriter`` seam is injectable for testability, mirroring the
    ``mover`` seam. A fake that reports 5 rewrites is reflected in the outcome.
    """
    src_root = tmp_path / "src-state"
    dst_root = tmp_path / "dst-state"
    src_root.mkdir()
    (src_root / "issues").mkdir()

    plan = plan_state_dir_migration(
        repo_root=tmp_path,
        src_root=src_root,
        dst_root=dst_root,
        src_children=[src_root / "issues"],
        dst_names=[],
        registered_worktrees=[],
    )

    def fake_rewriter(state_path: Path, src: Path, dst: Path) -> StateRewriteResult:
        return StateRewriteResult(ok=True, rewritten=5)

    outcome = apply_state_dir_migration(plan, state_rewriter=fake_rewriter)

    assert outcome.ok is True
    assert outcome.rewritten_paths == 5


def test_apply_injected_state_rewriter_failure_propagates(tmp_path: Path) -> None:
    """A failing injected state_rewriter makes the whole outcome ``ok=False``."""
    src_root = tmp_path / "src-state"
    dst_root = tmp_path / "dst-state"
    src_root.mkdir()
    (src_root / "issues").mkdir()

    plan = plan_state_dir_migration(
        repo_root=tmp_path,
        src_root=src_root,
        dst_root=dst_root,
        src_children=[src_root / "issues"],
        dst_names=[],
        registered_worktrees=[],
    )

    def failing_rewriter(state_path: Path, src: Path, dst: Path) -> StateRewriteResult:
        return StateRewriteResult(ok=False, error="lock timeout")

    outcome = apply_state_dir_migration(plan, state_rewriter=failing_rewriter)

    assert outcome.ok is False
    assert outcome.moved == ("issues",)
    assert outcome.rewritten_paths == 0
    assert "lock timeout" in (outcome.error or "")
