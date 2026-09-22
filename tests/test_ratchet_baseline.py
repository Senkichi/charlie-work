"""Tests for ``charlie_work.ratchet_baseline`` (issue #1802).

Covers the per-entry baseline primitives all three converted ratchets share:

* ``load_count_baseline`` / ``load_set_baseline`` -- fail-closed loading
  (missing directory, stray files, malformed entries all raise).
* ``write_count_baseline`` / ``write_set_baseline`` -- atomic sync writes:
  create, update (preserving ``#`` comment audit lines), delete stale, prune
  empty dirs, and reject keys that could escape the directory.
* ``count_delta_in_diff`` -- the diff-derived baseline-increase measure the
  private-slug gate uses instead of a stored total.
* Real ``git merge`` simulations proving the conflict property the whole
  change exists for: two branches touching DIFFERENT entries merge cleanly
  and the guard passes on the merged tree, while two branches raising the
  SAME entry to different values conflict.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from charlie_work.ratchet_baseline import (
    BaselineFormatError,
    count_delta_in_diff,
    load_count_baseline,
    load_set_baseline,
    parse_count_text,
    write_count_baseline,
    write_set_baseline,
)


# ---------------------------------------------------------------------------
# parse_count_text / load_count_baseline: parsing and fail-closed loading
# ---------------------------------------------------------------------------


def test_parse_count_text_accepts_integer_plus_comments() -> None:
    assert parse_count_text("42\n", name="x.count") == 42
    assert parse_count_text("42\n# rationale\n\n# more\n", name="x.count") == 42


def test_parse_count_text_rejects_non_integer_first_line() -> None:
    with pytest.raises(BaselineFormatError, match="first line"):
        parse_count_text("# comment first\n42\n", name="x.count")
    with pytest.raises(BaselineFormatError, match="first line"):
        parse_count_text("", name="x.count")
    with pytest.raises(BaselineFormatError, match="first line"):
        parse_count_text("4.2\n", name="x.count")
    with pytest.raises(BaselineFormatError, match="first line"):
        parse_count_text("-1\n", name="x.count")


def test_parse_count_text_rejects_non_comment_body_lines() -> None:
    with pytest.raises(BaselineFormatError, match="neither blank nor"):
        parse_count_text("42\nnot a comment\n", name="x.count")


def test_load_count_baseline_roundtrip(tmp_path: Path) -> None:
    write_count_baseline(tmp_path / "bl", {"a/b.py": 3, "c.py": 10})
    assert load_count_baseline(tmp_path / "bl") == {"a/b.py": 3, "c.py": 10}


def test_load_count_baseline_missing_dir_fails(tmp_path: Path) -> None:
    with pytest.raises(BaselineFormatError, match="not found"):
        load_count_baseline(tmp_path / "nope")


def test_load_count_baseline_stray_file_fails(tmp_path: Path) -> None:
    d = tmp_path / "bl"
    d.mkdir()
    (d / "a.py.count").write_text("3\n", encoding="utf-8")
    (d / "stray.txt").write_text("x\n", encoding="utf-8")
    with pytest.raises(BaselineFormatError, match="unexpected file"):
        load_count_baseline(d)


def test_load_count_baseline_malformed_entry_fails(tmp_path: Path) -> None:
    d = tmp_path / "bl"
    d.mkdir()
    (d / "a.py.count").write_text("bogus\n", encoding="utf-8")
    with pytest.raises(BaselineFormatError, match="malformed count entry"):
        load_count_baseline(d)


def test_load_count_baseline_nested_keys(tmp_path: Path) -> None:
    d = tmp_path / "bl"
    (d / "orchestration").mkdir(parents=True)
    (d / "orchestration" / "mod.py.count").write_text("7\n", encoding="utf-8")
    assert load_count_baseline(d) == {"orchestration/mod.py": 7}


# ---------------------------------------------------------------------------
# load_set_baseline
# ---------------------------------------------------------------------------


def test_load_set_baseline_member_names(tmp_path: Path) -> None:
    d = tmp_path / "set"
    (d / "sub").mkdir(parents=True)
    (d / "alpha").write_text("", encoding="utf-8")
    (d / "sub" / "beta").write_text("freeform rationale\n", encoding="utf-8")
    assert load_set_baseline(d) == frozenset({"alpha", "sub/beta"})


def test_load_set_baseline_missing_dir_fails(tmp_path: Path) -> None:
    with pytest.raises(BaselineFormatError, match="not found"):
        load_set_baseline(tmp_path / "nope")


# ---------------------------------------------------------------------------
# write_count_baseline / write_set_baseline: sync semantics
# ---------------------------------------------------------------------------


def test_write_count_baseline_syncs_create_update_delete(tmp_path: Path) -> None:
    d = tmp_path / "bl"
    write_count_baseline(d, {"keep.py": 1, "change.py": 5, "gone.py": 9})
    write_count_baseline(d, {"keep.py": 1, "change.py": 6, "new/nested.py": 2})
    assert load_count_baseline(d) == {
        "keep.py": 1,
        "change.py": 6,
        "new/nested.py": 2,
    }
    # The stale entry's file is deleted, not zeroed.
    assert not (d / "gone.py.count").exists()


def test_write_count_baseline_preserves_comment_lines_on_rewrite(tmp_path: Path) -> None:
    d = tmp_path / "bl"
    entry = d / "a.py.count"
    d.mkdir()
    entry.write_text("5\n# issue #1: accepted bump\n# second line\n", encoding="utf-8")
    write_count_baseline(d, {"a.py": 6})
    assert entry.read_text(encoding="utf-8") == "6\n# issue #1: accepted bump\n# second line\n"


def test_write_count_baseline_no_rewrite_when_value_unchanged(tmp_path: Path) -> None:
    d = tmp_path / "bl"
    d.mkdir()
    entry = d / "a.py.count"
    payload = "5\n# hand note\n"
    entry.write_text(payload, encoding="utf-8")
    write_count_baseline(d, {"a.py": 5})
    assert entry.read_text(encoding="utf-8") == payload


def test_write_count_baseline_prunes_empty_parent_dirs(tmp_path: Path) -> None:
    d = tmp_path / "bl"
    write_count_baseline(d, {"deep/nested/a.py": 1})
    write_count_baseline(d, {})
    assert not (d / "deep").exists()


def test_write_count_baseline_rejects_escaping_keys(tmp_path: Path) -> None:
    for key in ("../escape.py", "/abs.py", "a\\b.py", "a//b.py", "./a.py", ""):
        with pytest.raises(BaselineFormatError, match="invalid baseline entry key"):
            write_count_baseline(tmp_path / "bl", {key: 1})


def test_write_set_baseline_syncs_members(tmp_path: Path) -> None:
    d = tmp_path / "set"
    write_set_baseline(d, {"a", "b"})
    write_set_baseline(d, {"a", "c/d"})
    assert load_set_baseline(d) == frozenset({"a", "c/d"})
    assert not (d / "b").exists()


def test_write_set_baseline_preserves_existing_member_content(tmp_path: Path) -> None:
    d = tmp_path / "set"
    d.mkdir()
    member = d / "a"
    member.write_text("rationale text\n", encoding="utf-8")
    write_set_baseline(d, {"a", "b"})
    assert member.read_text(encoding="utf-8") == "rationale text\n"


# ---------------------------------------------------------------------------
# count_delta_in_diff: diff-derived baseline increase
# ---------------------------------------------------------------------------


_PREFIX = ".private-slug-baseline/files"


def _diff(path: str, removed: list[str] | None = None, added: list[str] | None = None) -> str:
    lines = [
        f"diff --git a/{path} b/{path}",
        f"--- a/{path}",
        f"+++ b/{path}",
        "@@ -1 +1 @@",
    ]
    lines += [f"-{r}" for r in (removed or [])]
    lines += [f"+{a}" for a in (added or [])]
    return "\n".join(lines) + "\n"


def test_count_delta_in_diff_bump() -> None:
    diff = _diff(f"{_PREFIX}/a.py.count", removed=["10"], added=["11"])
    assert count_delta_in_diff(diff, _PREFIX) == 1


def test_count_delta_in_diff_new_entry() -> None:
    diff = (
        f"diff --git a/{_PREFIX}/new.py.count b/{_PREFIX}/new.py.count\n"
        "--- /dev/null\n"
        f"+++ b/{_PREFIX}/new.py.count\n"
        "@@ -0,0 +1 @@\n"
        "+5\n"
    )
    assert count_delta_in_diff(diff, _PREFIX) == 5


def test_count_delta_in_diff_deleted_entry() -> None:
    diff = (
        f"diff --git a/{_PREFIX}/old.py.count b/{_PREFIX}/old.py.count\n"
        f"--- a/{_PREFIX}/old.py.count\n"
        "+++ /dev/null\n"
        "@@ -1 +0,0 @@\n"
        "-4\n"
    )
    assert count_delta_in_diff(diff, _PREFIX) == -4


def test_count_delta_in_diff_ignores_comments_and_other_files() -> None:
    diff = _diff(
        f"{_PREFIX}/a.py.count",
        removed=["# old note"],
        added=["# new note"],
    )
    diff += _diff("src/real.py", removed=["10"], added=["11"])
    diff += _diff(f"{_PREFIX}/README.md", added=["7"])
    assert count_delta_in_diff(diff, _PREFIX) == 0


def test_count_delta_in_diff_ignores_prefix_lookalikes() -> None:
    # A path sharing the prefix string but not beneath the directory, and a
    # non-.count name beneath the directory, must not count.
    diff = _diff(f"{_PREFIX}X/a.py.count", added=["9"])
    diff += _diff(f"{_PREFIX}/notes.txt", added=["9"])
    assert count_delta_in_diff(diff, _PREFIX) == 0


# ---------------------------------------------------------------------------
# Real-git merge simulations: the property the whole change exists for
# ---------------------------------------------------------------------------


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if check and result.returncode != 0:
        raise AssertionError(f"git {' '.join(args)} failed:\n{result.stderr}")
    return result


def _init_repo(repo: Path) -> None:
    repo.mkdir(parents=True)
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "test")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "commit.gpgsign", "false")
    # git-for-windows fails to write objects/index entries past MAX_PATH when
    # the pytest tmpdir sits under a deep worktree path; harmless elsewhere.
    _git(repo, "config", "core.longpaths", "true")


def _commit_entry(repo: Path, key: str, value: int, branch: str, message: str) -> None:
    _git(repo, "checkout", "-b", branch, check=True)
    entry = repo / "baseline" / (key + ".count")
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.write_text(f"{value}\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", message)
    _git(repo, "checkout", "main")


def test_distinct_entry_changes_merge_cleanly(tmp_path: Path) -> None:
    """Issue #1802's headline property: two branches that each add a DIFFERENT
    baseline entry merge with no conflict, and the guard passes on the merged
    result."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    # Seed commit: baseline dir with one entry.
    seed = repo / "baseline" / "a.py.count"
    seed.parent.mkdir(parents=True)
    seed.write_text("1\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "seed")

    _commit_entry(repo, "b.py", 2, "branch-b", "add b entry")
    _commit_entry(repo, "c.py", 3, "branch-c", "add c entry")

    _git(repo, "merge", "branch-b")
    merged = _git(repo, "merge", "branch-c", check=False)
    assert merged.returncode == 0, f"distinct-entry merge conflicted:\n{merged.stdout}"
    assert not (repo / "baseline" / "b.py.count").read_text(encoding="utf-8").startswith("<<<<<<<")

    # The guard (fail-closed load) passes on the merged tree.
    assert load_count_baseline(repo / "baseline") == {"a.py": 1, "b.py": 2, "c.py": 3}


def test_same_entry_divergent_values_conflict(tmp_path: Path) -> None:
    """The control direction: two branches raising the SAME entry to different
    values still conflict on merge -- the conflict-free layout must not
    silently merge two incompatible raises of one entry."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    seed = repo / "baseline" / "a.py.count"
    seed.parent.mkdir(parents=True)
    seed.write_text("1\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "seed")

    _commit_entry(repo, "a.py", 5, "branch-5", "raise a to 5")
    _commit_entry(repo, "a.py", 7, "branch-7", "raise a to 7")

    _git(repo, "merge", "branch-5")
    merged = _git(repo, "merge", "branch-7", check=False)
    assert merged.returncode != 0, "same-entry divergent raises merged cleanly"
    assert "CONFLICT" in merged.stdout or "conflict" in merged.stderr.lower()
    # And the guard itself would fail on the conflicted file if a resolver
    # blindly kept the markers.
    with pytest.raises(BaselineFormatError):
        load_count_baseline(repo / "baseline")
    _git(repo, "merge", "--abort")


def test_same_entry_same_value_merges_cleanly(tmp_path: Path) -> None:
    """Two branches writing the IDENTICAL value to the same entry merge
    cleanly -- deterministic convergence (the quantized-marks rule in
    refresh_file_size_ratchet.py relies on this)."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    seed = repo / "baseline" / "a.py.count"
    seed.parent.mkdir(parents=True)
    seed.write_text("1\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "seed")

    _commit_entry(repo, "a.py", 200, "branch-x", "raise a to 200")
    _commit_entry(repo, "a.py", 200, "branch-y", "raise a to 200")

    _git(repo, "merge", "branch-x")
    merged = _git(repo, "merge", "branch-y", check=False)
    assert merged.returncode == 0
    assert load_count_baseline(repo / "baseline") == {"a.py": 200}
