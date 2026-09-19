"""Flag-like argv-injection guard tests for the janitor no-op-rework gate.

Split out of ``tests/test_janitor_no_op_rework.py`` (issue #1558, Track 1;
itself split out of ``tests/test_janitor.py``): flag-like
``reviewed_head_sha`` / ``headRefOid`` / ``headRefName`` / ``baseRefName``
values must surface as warnings before reaching a git argv, on all three
detection branches (merge-only, patch-id, sha-match).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _janitor_fixtures import _config, _green_checks, _green_pr

from charlie_work.janitor import JanitorVerdict, _calculate_patch_id, run_janitor


def test_no_op_rework_warns_on_flag_like_reviewed_head_sha() -> None:
    """A flag-like reviewed_head_sha is caught as a warning, not a crash."""
    pr = _green_pr(headRefOid="abc123")
    pr_state = {
        "decision": "request_changes",
        "reviewed_head_sha": "--exec=foo",
    }

    verdict = run_janitor(
        pr,
        _green_checks(),
        _config(),
        pr_state=pr_state,
        repo_root=Path.cwd(),
        review_decision=pr_state,
    )

    assert isinstance(verdict, JanitorVerdict)
    assert not verdict.is_no_op_rework
    assert any("_check_no_op_rework reviewed_head_sha" in w for w in verdict.warnings)


def test_no_op_rework_warns_on_non_hex_reviewed_head_sha() -> None:
    """A non-hex reviewed_head_sha is caught as a warning, not a crash."""
    pr = _green_pr(headRefOid="abc123")
    pr_state = {
        "decision": "request_changes",
        "reviewed_head_sha": "not-a-sha!",
    }

    verdict = run_janitor(
        pr,
        _green_checks(),
        _config(),
        pr_state=pr_state,
        repo_root=Path.cwd(),
        review_decision=pr_state,
    )

    assert isinstance(verdict, JanitorVerdict)
    assert not verdict.is_no_op_rework
    assert any("_check_no_op_rework reviewed_head_sha" in w for w in verdict.warnings)


def test_no_op_rework_warns_on_flag_like_current_head_sha() -> None:
    """A flag-like headRefOid is caught as a warning before reaching argv."""
    pr = _green_pr(headRefOid="--upload-pack=evil")
    pr_state = {
        "decision": "request_changes",
        "reviewed_head_sha": "abc123",
    }

    verdict = run_janitor(
        pr,
        _green_checks(),
        _config(),
        pr_state=pr_state,
        repo_root=Path.cwd(),
        review_decision=pr_state,
    )

    assert isinstance(verdict, JanitorVerdict)
    assert not verdict.is_no_op_rework
    assert any("_check_no_op_rework current_head_sha" in w for w in verdict.warnings)


def test_no_op_rework_warns_on_flag_like_head_ref() -> None:
    """A flag-like headRefName is caught as a warning before ``git fetch origin``."""
    pr = _green_pr(headRefOid="def456", headRefName="--exec=foo")
    pr_state = {
        "decision": "request_changes",
        "reviewed_head_sha": "abc123",
    }

    verdict = run_janitor(
        pr,
        _green_checks(),
        _config(),
        pr_state=pr_state,
        repo_root=Path.cwd(),
        review_decision=pr_state,
    )

    assert isinstance(verdict, JanitorVerdict)
    assert not verdict.is_no_op_rework
    assert any("_check_no_op_rework head_ref (merge-only)" in w for w in verdict.warnings)


def test_no_op_rework_warns_on_flag_like_base_ref() -> None:
    """A flag-like baseRefName is caught as a warning before ``git fetch origin``."""
    pr = _green_pr(headRefOid="def456", headRefName="agent/issue-1-fix", baseRefName="--exec=bar")
    pr_state = {
        "decision": "request_changes",
        "reviewed_head_sha": "abc123",
    }

    verdict = run_janitor(
        pr,
        _green_checks(),
        _config(),
        pr_state=pr_state,
        repo_root=Path.cwd(),
        review_decision=pr_state,
    )

    assert isinstance(verdict, JanitorVerdict)
    assert not verdict.is_no_op_rework
    assert any("_check_no_op_rework base_ref (merge-only)" in w for w in verdict.warnings)


def test_no_op_rework_accepts_valid_sha_and_ref_values(tmp_path: Path) -> None:
    """Valid SHA and ref-name values pass validation and reach the normal path."""
    pr = _green_pr(headRefOid="def456", headRefName="agent/issue-1-fix", baseRefName="main")
    pr_state = {
        "decision": "request_changes",
        "reviewed_head_sha": "abc123",
    }

    # Heads differ and no real git repo at tmp_path — the merge-only check will
    # fail gracefully (subprocess error → warning), but validation must pass.
    verdict = run_janitor(
        pr,
        _green_checks(),
        _config(),
        pr_state=pr_state,
        repo_root=tmp_path,
        review_decision=pr_state,
    )

    assert isinstance(verdict, JanitorVerdict)
    assert not verdict.is_no_op_rework
    assert any("git fetch/rev-list failed" in w for w in verdict.warnings)


def test_no_op_rework_warns_on_flag_like_head_ref_patch_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A flag-like headRefName on the patch-id branch must not reach
    ``_get_unpushed_commit_info`` (issue #659).

    Pins the conflict-resolution hunk that merges 694's validation with PR
    #761's ``base_ref`` addition: a naive marker-only merge-conflict
    resolution dedents the ``_get_unpushed_commit_info`` call out of the
    ``except ValueError`` block, so it runs unconditionally with the raw,
    unvalidated ref even when validation just rejected it. This test fails
    against that regression.
    """
    from charlie_work import janitor as janitor_module

    diff = """diff --git a/test.txt b/test.txt
index 1234567..abcdef0 100644
--- a/test.txt
+++ b/test.txt
@@ -1,2 +1,2 @@
 line 1
-line 2
+line 2 modified
"""
    current_patch_id = _calculate_patch_id(diff)
    pr = _green_pr(headRefName="--upload-pack=evil")
    pr_state = {
        "decision": "request_changes",
        "reviewed_patch_id": current_patch_id,
    }

    def _fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("_get_unpushed_commit_info should not be called with an invalid ref")

    monkeypatch.setattr(janitor_module, "_get_unpushed_commit_info", _fail_if_called)

    verdict = run_janitor(
        pr,
        _green_checks(),
        _config(),
        pr_state=pr_state,
        repo_root=Path.cwd(),
        pr_diff=diff,
        review_decision=pr_state,
    )

    assert verdict.ok is False
    assert any("_check_no_op_rework head_ref (patch-id)" in w for w in verdict.warnings)


def test_no_op_rework_warns_on_flag_like_base_ref_patch_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A flag-like baseRefName on the patch-id branch must not reach
    ``_get_unpushed_commit_info`` (issue #659, same hunk as the head_ref
    variant above).
    """
    from charlie_work import janitor as janitor_module

    diff = """diff --git a/test.txt b/test.txt
index 1234567..abcdef0 100644
--- a/test.txt
+++ b/test.txt
@@ -1,2 +1,2 @@
 line 1
-line 2
+line 2 modified
"""
    current_patch_id = _calculate_patch_id(diff)
    pr = _green_pr(headRefName="agent/issue-1-fix", baseRefName="--upload-pack=evil")
    pr_state = {
        "decision": "request_changes",
        "reviewed_patch_id": current_patch_id,
    }

    def _fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("_get_unpushed_commit_info should not be called with an invalid ref")

    monkeypatch.setattr(janitor_module, "_get_unpushed_commit_info", _fail_if_called)

    verdict = run_janitor(
        pr,
        _green_checks(),
        _config(),
        pr_state=pr_state,
        repo_root=Path.cwd(),
        pr_diff=diff,
        review_decision=pr_state,
    )

    assert verdict.ok is False
    assert any("_check_no_op_rework base_ref (patch-id)" in w for w in verdict.warnings)


def test_no_op_rework_warns_on_flag_like_head_ref_sha_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A flag-like headRefName on the sha-match branch must not reach
    ``_get_unpushed_commit_info`` (issue #659). Pins the second
    conflict-resolution hunk against the same marker-strip dedent hazard as
    the patch-id variant above.
    """
    from charlie_work import janitor as janitor_module

    pr = _green_pr(headRefOid="abc123", headRefName="--upload-pack=evil")
    pr_state = {
        "decision": "request_changes",
        "reviewed_head_sha": "abc123",
    }

    def _fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("_get_unpushed_commit_info should not be called with an invalid ref")

    monkeypatch.setattr(janitor_module, "_get_unpushed_commit_info", _fail_if_called)

    verdict = run_janitor(
        pr,
        _green_checks(),
        _config(),
        pr_state=pr_state,
        repo_root=Path.cwd(),
        review_decision=pr_state,
    )

    assert verdict.ok is False
    assert any("_check_no_op_rework head_ref (sha-match)" in w for w in verdict.warnings)


def test_no_op_rework_warns_on_flag_like_base_ref_sha_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A flag-like baseRefName on the sha-match branch must not reach
    ``_get_unpushed_commit_info`` (issue #659, same hunk as the head_ref
    variant above).
    """
    from charlie_work import janitor as janitor_module

    pr = _green_pr(
        headRefOid="abc123", headRefName="agent/issue-1-fix", baseRefName="--upload-pack=evil"
    )
    pr_state = {
        "decision": "request_changes",
        "reviewed_head_sha": "abc123",
    }

    def _fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("_get_unpushed_commit_info should not be called with an invalid ref")

    monkeypatch.setattr(janitor_module, "_get_unpushed_commit_info", _fail_if_called)

    verdict = run_janitor(
        pr,
        _green_checks(),
        _config(),
        pr_state=pr_state,
        repo_root=Path.cwd(),
        review_decision=pr_state,
    )

    assert verdict.ok is False
    assert any("_check_no_op_rework base_ref (sha-match)" in w for w in verdict.warnings)
