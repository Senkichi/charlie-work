"""main_sync_git against real temporary repositories."""

from __future__ import annotations

import sys
from pathlib import Path

from _main_sync_fixtures import git, make_sync_repos

from charlie_work import main_sync_git as msg
from charlie_work.main_sync_classify import CheckoutState


def test_reads_identity_of_a_fresh_clone(tmp_path: Path) -> None:
    repos = make_sync_repos(tmp_path)
    assert msg.has_origin(repos.clone)
    assert msg.is_primary_checkout(repos.clone)
    assert msg.resolve_default_branch(repos.clone, dry_run=True) == "main"
    common = msg.git_common_dir(repos.clone)
    assert common is not None and common.resolve() == (repos.clone / ".git").resolve()


def test_linked_worktree_is_not_primary_but_shares_common_dir(tmp_path: Path) -> None:
    repos = make_sync_repos(tmp_path)
    linked = tmp_path / "linked"
    git(repos.clone, "worktree", "add", "-q", "-b", "side", str(linked))
    assert not msg.is_primary_checkout(linked)
    common = msg.git_common_dir(linked)
    assert common is not None and common.resolve() == (repos.clone / ".git").resolve()


def test_no_origin(tmp_path: Path) -> None:
    repos = make_sync_repos(tmp_path)
    git(repos.clone, "remote", "remove", "origin")
    assert not msg.has_origin(repos.clone)


def test_state_when_behind_with_dirty_and_untracked(tmp_path: Path) -> None:
    repos = make_sync_repos(tmp_path)
    seed = repos.head()
    tip = repos.push_commit("uv.lock", "v2\n")
    (repos.clone / "README.md").write_text("local edit\n", encoding="utf-8")
    (repos.clone / "scratch.txt").write_text("x\n", encoding="utf-8")
    assert msg.fetch_default(repos.clone, "main").ok

    state = msg.read_checkout_state(repos.clone, "main")

    assert isinstance(state, CheckoutState)
    assert (state.local_sha, state.origin_sha, state.merge_base) == (seed, tip, seed)
    assert (state.ahead, state.behind) == (0, 1)
    assert state.current_branch == "main"
    assert state.dirty_tracked == ("README.md",)
    assert state.untracked == ("scratch.txt",)
    assert state.incoming_paths == ("uv.lock",)
    assert state.operation_in_progress is None and not state.index_locked


def test_state_on_feature_branch_skips_working_tree_reads(tmp_path: Path) -> None:
    repos = make_sync_repos(tmp_path)
    git(repos.clone, "checkout", "-q", "-b", "feature")
    repos.push_commit("a.txt", "a\n")
    repos.fetch()
    state = msg.read_checkout_state(repos.clone, "main")
    assert isinstance(state, CheckoutState)
    assert state.current_branch == "feature"
    assert not state.default_checked_out_elsewhere
    assert state.incoming_paths == ()


def test_state_sees_main_checked_out_in_linked_worktree(tmp_path: Path) -> None:
    repos = make_sync_repos(tmp_path)
    git(repos.clone, "checkout", "-q", "-b", "feature")
    git(repos.clone, "worktree", "add", "-q", str(tmp_path / "wt"), "main")
    repos.push_commit("a.txt", "a\n")
    repos.fetch()
    state = msg.read_checkout_state(repos.clone, "main")
    assert isinstance(state, CheckoutState) and state.default_checked_out_elsewhere


def test_state_flags_operation_and_lock(tmp_path: Path) -> None:
    repos = make_sync_repos(tmp_path)
    (repos.clone / ".git" / "MERGE_HEAD").write_text(repos.head() + "\n", encoding="utf-8")
    (repos.clone / ".git" / "index.lock").write_text("", encoding="utf-8")
    state = msg.read_checkout_state(repos.clone, "main")
    assert isinstance(state, CheckoutState)
    assert state.operation_in_progress == "merge_in_progress"
    assert state.index_locked


def test_missing_remote_branch_is_a_reason(tmp_path: Path) -> None:
    repos = make_sync_repos(tmp_path)
    assert msg.read_checkout_state(repos.clone, "nope") == "no_local_branch"
    git(repos.clone, "branch", "-q", "nope")
    assert msg.read_checkout_state(repos.clone, "nope") == "no_remote_branch"


def test_shallow_graft_is_not_unrelated_history(tmp_path: Path) -> None:
    repos = make_sync_repos(tmp_path)
    repos.graft_origin_tip()
    assert git(repos.clone, "rev-parse", "--is-shallow-repository") == "true"
    assert msg.read_checkout_state(repos.clone, "main") == "shallow_history"


def test_fetch_unshallows_so_the_merge_base_is_found(tmp_path: Path) -> None:
    repos = make_sync_repos(tmp_path)
    seed = repos.head()
    repos.graft_origin_tip()
    assert msg.fetch_default(repos.clone, "main").ok
    assert git(repos.clone, "rev-parse", "--is-shallow-repository") == "false"
    state = msg.read_checkout_state(repos.clone, "main")
    assert isinstance(state, CheckoutState)
    assert (state.merge_base, state.behind) == (seed, 2)


def test_incoming_commits_oldest_first_with_paths(tmp_path: Path) -> None:
    repos = make_sync_repos(tmp_path)
    seed = repos.head()
    c1 = repos.push_commit("a.txt", "a\n")
    c2 = repos.push_commit("uv.lock", "lock\n")
    repos.fetch()
    commits = msg.list_incoming_commits(repos.clone, seed, c2)
    assert commits is not None
    assert [(c.sha, c.paths) for c in commits] == [(c1, ("a.txt",)), (c2, ("uv.lock",))]
    assert msg.count_commits(repos.clone, seed, c2) == 2
    assert msg.changed_paths(repos.clone, seed, c2) == ("a.txt", "uv.lock")
    assert msg.is_ancestor(repos.clone, seed, c1)
    assert not msg.is_ancestor(repos.clone, c2, seed)


def test_fast_forward_update_ref_and_reset_keep(tmp_path: Path) -> None:
    repos = make_sync_repos(tmp_path)
    seed = repos.head()
    tip = repos.push_commit("a.txt", "a\n")
    repos.fetch()
    assert msg.fast_forward(repos.clone, tip).ok
    assert repos.head() == tip
    assert msg.reset_keep(repos.clone, seed).ok
    assert repos.head() == seed

    git(repos.clone, "checkout", "-q", "-b", "feature")
    assert not msg.update_ref(repos.clone, "main", tip, "0" * 40).ok  # stale old: CAS refuses
    assert msg.update_ref(repos.clone, "main", tip, seed).ok
    assert git(repos.clone, "rev-parse", "refs/heads/main") == tip
    assert repos.head() == seed  # working tree untouched


def test_reset_keep_refuses_to_discard_local_edits(tmp_path: Path) -> None:
    repos = make_sync_repos(tmp_path)
    seed = repos.head()
    tip = repos.push_commit("README.md", "upstream\n")
    repos.fetch()
    assert msg.fast_forward(repos.clone, tip).ok
    (repos.clone / "README.md").write_text("local after ff\n", encoding="utf-8")
    assert not msg.reset_keep(repos.clone, seed).ok
    assert repos.head() == tip


def test_run_post_sync_reports_exit_code(tmp_path: Path) -> None:
    repos = make_sync_repos(tmp_path)
    result = msg.run_post_sync(repos.clone, (sys.executable, "-c", "raise SystemExit(75)"), 30)
    assert result.returncode == 75 and not result.ok
