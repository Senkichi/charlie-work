"""Tests for the test-run source-identity anchor (issue #1665).

A pytest invocation that runs without a per-project venv can silently import
``charlie_work`` from a different checkout -- on this host, a stray
``_editable_impl_charlie_work.pth`` in the system interpreter's user-site
points bare ``python`` at the read-only main checkout's ``src``. Collection
and rootdir still point at the worktree, so a run can report green while
exercising the wrong tree.

These tests pin the ``pytest_sessionstart`` assertion that conftest wires in:
``charlie_work.__file__`` must resolve under the repository root that owns
``tests/conftest.py``, unless the explicit opt-out marker is exported for the
sanctioned shared-venv / PYTHONPATH-shadow pattern.

Note on paths: this session's TMPDIR lives *inside* the worktree
(``.var/worker-tmp``), so a "foreign checkout" must be built as a sibling of
``REPO_ROOT`` and is deliberately never materialized -- containment is a
resolved-path question and does not require the path to exist (a stray
``.pth`` line points wherever it wants).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import charlie_work
from _source_anchor import (
    SOURCE_ANCHOR_OPT_OUT_VAR,
    enforce_source_anchor,
    source_anchor_violation,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _fake_package(root: Path) -> Path:
    """Materialize a minimal ``src/charlie_work`` tree and return its __init__."""
    pkg = root / "src" / "charlie_work"
    pkg.mkdir(parents=True)
    init = pkg / "__init__.py"
    init.write_text("", encoding="utf-8")
    return init


def _foreign_package() -> Path:
    """A package path outside ``REPO_ROOT`` -- a sibling checkout, never created."""
    return (
        REPO_ROOT.parent / "foreign-checkout-not-created" / "src" / "charlie_work" / "__init__.py"
    )


def _norm(path: Path) -> str:
    """The canonicalized form the violation message embeds (normcase + realpath)."""
    return os.path.normcase(os.path.realpath(path))


# ---------------------------------------------------------------------------
# source_anchor_violation: pure containment predicate.
# ---------------------------------------------------------------------------


def test_no_violation_when_package_resolves_inside_root(tmp_path: Path) -> None:
    package_file = _fake_package(tmp_path / "worktree")
    assert source_anchor_violation(tmp_path / "worktree", str(package_file)) is None


def test_violation_when_package_resolves_into_another_checkout(tmp_path: Path) -> None:
    """The reported bug shape: a stray editable .pth substitutes the main
    checkout's charlie_work for the worktree's."""
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    foreign = _fake_package(tmp_path / "main")

    violation = source_anchor_violation(worktree, str(foreign))

    assert violation is not None
    assert _norm(foreign) in violation
    assert _norm(worktree) in violation


def test_violation_when_package_file_is_none(tmp_path: Path) -> None:
    """A namespace-package resolution carries no __file__ -- nothing to
    anchor, so it must be a distinct failure, never a silent pass."""
    violation = source_anchor_violation(tmp_path / "repo", None)

    assert violation is not None
    assert "__file__" in violation


def test_sibling_prefix_is_not_containment(tmp_path: Path) -> None:
    """Exact-prefix vs glob-prefix trap: ``repo2`` shares the string prefix
    ``repo`` but is not inside it."""
    root = tmp_path / "repo"
    root.mkdir()
    foreign = _fake_package(tmp_path / "repo2")

    assert source_anchor_violation(root, str(foreign)) is not None


def test_lexical_escape_resolves_before_compare(tmp_path: Path) -> None:
    """A ``..`` segment must not be read as contained (is_relative_to is
    purely lexical)."""
    root = tmp_path / "repo"
    lexical = root / ".." / "outside" / "charlie_work" / "__init__.py"

    violation = source_anchor_violation(root, str(lexical))

    assert violation is not None
    assert _norm(tmp_path / "outside" / "charlie_work" / "__init__.py") in violation


# ---------------------------------------------------------------------------
# enforce_source_anchor: opt-out and exit behaviour.
# ---------------------------------------------------------------------------


def test_enforce_no_op_when_package_is_anchored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(SOURCE_ANCHOR_OPT_OUT_VAR, raising=False)
    monkeypatch.setattr(
        charlie_work, "__file__", str(REPO_ROOT / "src" / "charlie_work" / "__init__.py")
    )

    enforce_source_anchor(REPO_ROOT)  # must not raise


def test_enforce_aborts_session_when_package_escapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(SOURCE_ANCHOR_OPT_OUT_VAR, raising=False)
    foreign = _foreign_package()
    monkeypatch.setattr(charlie_work, "__file__", str(foreign))

    with pytest.raises(pytest.UsageError) as excinfo:
        enforce_source_anchor(REPO_ROOT)

    message = str(excinfo.value)
    assert _norm(foreign) in message
    assert _norm(REPO_ROOT) in message
    assert SOURCE_ANCHOR_OPT_OUT_VAR in message


def test_enforce_opt_out_env_var_skips_the_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sanctioned shared-venv / PYTHONPATH-shadow pattern exports the
    marker; a foreign __file__ then must not abort the session."""
    monkeypatch.setattr(charlie_work, "__file__", str(_foreign_package()))
    monkeypatch.setenv(SOURCE_ANCHOR_OPT_OUT_VAR, "1")

    enforce_source_anchor(REPO_ROOT)  # must not raise


def test_enforce_opt_out_empty_value_does_not_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exported-but-empty marker is not an opt-out -- it is almost
    certainly a scripting accident, so the assertion still fires."""
    monkeypatch.setattr(charlie_work, "__file__", str(_foreign_package()))
    monkeypatch.setenv(SOURCE_ANCHOR_OPT_OUT_VAR, "")

    with pytest.raises(pytest.UsageError):
        enforce_source_anchor(REPO_ROOT)


# ---------------------------------------------------------------------------
# Wiring: the real conftest must carry the hook, and this very session must
# be anchored (otherwise none of these results mean anything).
# ---------------------------------------------------------------------------


def test_conftest_defines_sessionstart_hook() -> None:
    import conftest

    assert callable(getattr(conftest, "pytest_sessionstart", None))


def test_conftest_sessionstart_enforces_the_anchor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drive the hook object pytest itself registered: a foreign
    ``charlie_work.__file__`` must abort the session."""
    import conftest

    monkeypatch.delenv(SOURCE_ANCHOR_OPT_OUT_VAR, raising=False)
    monkeypatch.setattr(charlie_work, "__file__", str(_foreign_package()))

    with pytest.raises(pytest.UsageError):
        conftest.pytest_sessionstart(None)


@pytest.mark.skipif(
    bool(os.environ.get(SOURCE_ANCHOR_OPT_OUT_VAR)),
    reason=(
        f"{SOURCE_ANCHOR_OPT_OUT_VAR} is set -- this session deliberately "
        "resolves charlie_work outside the checkout, so there is no anchor to assert"
    ),
)
def test_this_session_itself_is_anchored() -> None:
    """The hook ran for this very run; asserting the real import location is
    under this repo root is the live positive control. Sessions that export
    the opt-out marker legitimately resolve the package elsewhere, so this
    control stands down with them rather than going red."""
    assert source_anchor_violation(REPO_ROOT, charlie_work.__file__) is None
