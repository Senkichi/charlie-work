"""Tests for ``tests/_git_templates.py`` (HS-CW-4): per-process git repo templates."""

from __future__ import annotations

import shutil
import subprocess
import warnings
from pathlib import Path

import pytest

import _git_templates
import _worktree_fixtures as wf
from _git_templates import REUSE_ENV, TemplateRegistry


def _forms(path: Path) -> tuple[str, str]:
    return str(path), str(path).replace("\\", "/")


def _observe(repo: Path, base: Path) -> dict[str, str]:
    """Everything a test can see of a repo, with ``base`` spelled ``<BASE>``."""

    def git(*args: str) -> str:
        out = subprocess.run(
            ["git", *args], cwd=repo, check=True, capture_output=True, text=True
        ).stdout
        for form in _forms(base):
            out = out.replace(form, "<BASE>")
        return out

    observed = {
        "config": git("config", "--list", "--local"),
        "head": git("symbolic-ref", "HEAD"),
        "refs": git("for-each-ref", "--format=%(refname) %(objecttype) %(contents:subject)"),
        "log": git("log", "--all", "--format=%T %s"),
        "remotes": git("remote", "-v"),
        "tree": git("ls-tree", "-r", "--name-only", "HEAD"),
    }
    if git("rev-parse", "--is-bare-repository").strip() == "false":
        observed["status"] = git("status", "--porcelain")
    return observed


def _fresh(shape: str, base: Path) -> list[Path]:
    if shape == "plain":
        wf._init_repo_fresh(base / "repo")
        return [base / "repo"]
    if shape == "bare":
        wf._init_repo_fresh(base / "remote.git", True)
        return [base / "remote.git"]
    if shape == "clone":
        wf._init_repo_fresh(base / "remote.git", True)
        wf._clone_repo_fresh(base / "remote.git", base / "repo")
        return [base / "repo"]
    remote, clone = wf._init_bare_remote_and_clone_fresh(base)
    return [remote, clone]


def _copied(shape: str, base: Path, registry: TemplateRegistry) -> list[Path]:
    if shape == "plain":
        assert _git_templates.init_repo(
            base / "repo", bare=False, build=wf._init_repo_fresh, registry=registry
        )
        return [base / "repo"]
    if shape in ("bare", "clone"):
        assert _git_templates.init_repo(
            base / "remote.git", bare=True, build=wf._init_repo_fresh, registry=registry
        )
        if shape == "bare":
            return [base / "remote.git"]
        assert _git_templates.clone_repo(
            base / "remote.git", base / "repo", build=wf._clone_repo_fresh, registry=registry
        )
        return [base / "repo"]
    pair = _git_templates.bare_remote_and_clone(
        base, build=wf._init_bare_remote_and_clone_fresh, registry=registry
    )
    assert pair is not None
    return list(pair)


@pytest.fixture(autouse=True)
def _reuse_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(REUSE_ENV, raising=False)


@pytest.mark.parametrize("shape", ["plain", "bare", "clone", "ebc"])
def test_template_copy_matches_a_fresh_build(tmp_path: Path, shape: str) -> None:
    registry = TemplateRegistry()
    fresh_base, copy_base = tmp_path / "fresh", tmp_path / "copy"
    fresh = _fresh(shape, fresh_base)
    _copied(shape, tmp_path / "warm", registry)  # first call builds the template
    copied = _copied(shape, copy_base, registry)  # second call is a pure copy
    expected_shape = {"clone": "clone-of-bare"}.get(shape, shape)
    assert registry.materialized[expected_shape] == 2
    for fresh_repo, copied_repo in zip(fresh, copied, strict=True):
        assert _observe(copied_repo, copy_base) == _observe(fresh_repo, fresh_base)


def test_materialized_clone_pushes_to_its_own_remote(tmp_path: Path) -> None:
    registry = TemplateRegistry()
    build = wf._init_bare_remote_and_clone_fresh
    first = _git_templates.bare_remote_and_clone(tmp_path / "one", build=build, registry=registry)
    second = _git_templates.bare_remote_and_clone(tmp_path / "two", build=build, registry=registry)
    assert first is not None and second is not None
    remote, clone = second
    (clone / "new.txt").write_text("new\n", encoding="utf-8")
    wf._git(clone, "add", "new.txt")
    wf._git(clone, "commit", "-m", "copy commit")
    wf._git(clone, "push", "origin", "main")
    pushed = wf._git(clone, "rev-parse", "HEAD").stdout.strip()
    assert wf._git(remote, "rev-parse", "main").stdout.strip() == pushed
    template = registry.lookup("ebc")
    assert template is not None and template.remote is not None
    assert wf._git(template.remote, "rev-parse", "main").stdout.strip() != pushed
    assert wf._git(first[0], "rev-parse", "main").stdout.strip() != pushed


def test_clone_of_a_materialized_bare_uses_the_template_and_pushes_home(tmp_path: Path) -> None:
    before = _git_templates._REGISTRY.materialized["clone-of-bare"]
    remote, repo = tmp_path / "remote.git", tmp_path / "repo"
    wf._init_repo(remote, bare=True)
    wf._clone_repo(remote, repo)
    assert _git_templates._REGISTRY.materialized["clone-of-bare"] == before + 1
    wf._git(repo, "checkout", "-b", "feature")
    wf._git(repo, "push", "origin", "feature")
    assert wf._git(remote, "branch", "--list", "feature").stdout.strip() == "feature"


def test_changed_remote_clones_fresh(tmp_path: Path) -> None:
    remote, repo = tmp_path / "remote.git", tmp_path / "repo"
    wf._init_repo(remote, bare=True)
    wf._git(remote, "update-ref", "refs/heads/extra", "main")
    before = _git_templates._REGISTRY.materialized["clone-of-bare"]
    wf._clone_repo(remote, repo)
    assert _git_templates._REGISTRY.materialized["clone-of-bare"] == before
    assert "origin/extra" in wf._git(repo, "branch", "-r").stdout


def test_non_empty_destination_builds_fresh(tmp_path: Path) -> None:
    dst = tmp_path / "repo"
    dst.mkdir()
    (dst / "keep.txt").write_text("keep\n", encoding="utf-8")
    before = _git_templates._REGISTRY.materialized["plain"]
    wf._init_repo(dst)
    assert _git_templates._REGISTRY.materialized["plain"] == before
    assert (dst / ".git").is_dir() and (dst / "keep.txt").is_file()


def test_kill_switch_builds_fresh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(REUSE_ENV, "off")
    before = _git_templates._REGISTRY.materialized["plain"]
    wf._init_repo(tmp_path / "repo")
    assert _git_templates._REGISTRY.materialized["plain"] == before
    assert wf._git(tmp_path / "repo", "rev-parse", "HEAD").stdout.strip()


def test_fresh_true_builds_fresh(tmp_path: Path) -> None:
    before = _git_templates._REGISTRY.materialized["plain"]
    wf._init_repo(tmp_path / "repo", fresh=True)
    assert _git_templates._REGISTRY.materialized["plain"] == before


def test_template_build_failure_fails_open_with_one_warning(tmp_path: Path) -> None:
    registry = TemplateRegistry()

    def broken(repo: Path, bare: bool = False) -> None:
        raise subprocess.CalledProcessError(128, ["git", "init"])

    with pytest.warns(UserWarning, match=r"^test template disabled: plain: "):
        assert not _git_templates.init_repo(
            tmp_path / "a", bare=False, build=broken, registry=registry
        )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert not _git_templates.init_repo(
            tmp_path / "b", bare=False, build=broken, registry=registry
        )


def test_bad_copy_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    registry = TemplateRegistry()

    def lossy_copy(src: Path, dst: Path) -> None:
        shutil.copytree(src, dst, dirs_exist_ok=True)
        for ref in (dst / ".git" / "refs" / "heads").glob("*"):
            ref.unlink()
        (dst / ".git" / "packed-refs").unlink(missing_ok=True)

    monkeypatch.setattr(_git_templates, "_copy_tree", lossy_copy)
    with pytest.raises(AssertionError, match="git template copy mismatch"):
        _git_templates.init_repo(
            tmp_path / "a", bare=False, build=wf._init_repo_fresh, registry=registry
        )


def test_template_with_a_linked_worktree_is_refused(tmp_path: Path) -> None:
    registry = TemplateRegistry()

    def with_worktree(repo: Path, bare: bool = False) -> None:
        wf._init_repo_fresh(repo, bare)
        wf._git(repo, "worktree", "add", "-b", "side", str(repo.parent / "wt"))

    with pytest.warns(UserWarning, match="linked worktrees"):
        assert not _git_templates.init_repo(
            tmp_path / "a", bare=False, build=with_worktree, registry=registry
        )
