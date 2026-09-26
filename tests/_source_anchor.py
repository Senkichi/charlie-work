"""Test-run source-identity anchor (issue #1665).

A pytest invocation that runs without a per-project venv on a host can
silently import ``charlie_work`` from a DIFFERENT checkout than the one its
``tests/`` tree belongs to. On this host the concrete mechanism was a stray
``_editable_impl_charlie_work.pth`` in the system interpreter's user-site
(left by an old ``pip install -e .``), which resolved bare ``python`` --
``uv run --no-project``, or a shell whose cwd had drifted -- at the
read-only main checkout's ``src``. Collection, rootdir, and cwd still point
at the worktree, so a run can report green while exercising the wrong tree.

The preflight/venv-anchor guards for this bug class are wired only into the
long-lived orchestrator path and are neutralized in tests by the autouse
healthy-preflight fixture, so nothing fires at the moment a pytest session
starts. ``pytest_sessionstart`` in ``tests/conftest.py`` delegates to
:func:`enforce_source_anchor`, which asserts -- before any collection or
fixture runs -- that the ``charlie_work`` actually loaded resolves under the
repository root that owns the conftest. That single check covers every
variant (stray ``.pth``, ancestor ``.venv`` discovery, bare ``python``)
because they all produce the same signature: ``charlie_work.__file__``
outside this tree.

The sanctioned shared-venv worker pattern sets ``PYTHONPATH`` at a
worktree's ``src`` deliberately -- ``charlie_work.__file__`` still lands
inside the conftest's root, so it passes unmarked. A run that intentionally
points ``PYTHONPATH`` somewhere else exports
``CHARLIE_WORK_TEST_SOURCE_ANCHOR_OPT_OUT`` (any non-empty value) as the
explicit marker; the opt-out is announced on stderr so an opted-out session
leaves a trace rather than silently diverging.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

import charlie_work

#: Export to any non-empty value to skip the source-identity assertion.
SOURCE_ANCHOR_OPT_OUT_VAR = "CHARLIE_WORK_TEST_SOURCE_ANCHOR_OPT_OUT"


def _canonical(path: Path) -> Path:
    """Resolved, case-normalized path for containment comparison.

    ``realpath`` follows reparse points (the worktree ``.venv`` junctions on
    this host) before comparing; ``normcase`` prevents a drive/case
    difference in how a ``sys.path`` entry was spelled from producing a
    false escape on Windows. Same recipe as ``worker_stop_gate._path_within``.
    """
    return Path(os.path.normcase(os.path.realpath(path)))


def source_anchor_violation(repo_root: Path, package_file: str | None) -> str | None:
    """Return the refusal message when *package_file* escapes *repo_root*.

    ``None`` means anchored. A ``package_file`` of ``None`` (namespace
    package resolution, which carries no ``__file__``) is a violation --
    there is no location to anchor.
    """
    root = _canonical(repo_root)
    if package_file is None:
        return (
            "charlie_work source-identity assertion failed (issue #1665): "
            "charlie_work has no __file__ (namespace-package resolution?) -- "
            f"cannot prove the package comes from {root}. "
            f"Export {SOURCE_ANCHOR_OPT_OUT_VAR}=1 to skip this check."
        )
    resolved = _canonical(Path(package_file))
    if resolved.is_relative_to(root):
        return None
    return (
        "charlie_work source-identity assertion failed (issue #1665): "
        f"charlie_work.__file__ resolved to {resolved}, which is outside the "
        f"repository root that owns this conftest.py ({root}). This run would "
        "collect tests from one checkout while executing another's code "
        "(stray editable .pth in user-site, ancestor .venv discovery, or a "
        "bare-python invocation with no project venv). Re-run via "
        "`uv run --extra dev pytest` from the checkout root. If this layout "
        f"is intentional (shared-venv PYTHONPATH shadow), export "
        f"{SOURCE_ANCHOR_OPT_OUT_VAR}=1 to skip this check."
    )


def enforce_source_anchor(repo_root: Path) -> None:
    """Assert ``charlie_work`` resolves under *repo_root*; abort the session otherwise.

    Called from ``pytest_sessionstart`` in ``tests/conftest.py`` with the
    repository root that owns the conftest
    (``Path(__file__).resolve().parents[1]``). Honours the explicit opt-out
    marker; raises ``pytest.UsageError`` (exit code 4, rendered as an ERROR)
    on violation so the session never reaches collection on the wrong tree.
    """
    if os.environ.get(SOURCE_ANCHOR_OPT_OUT_VAR):
        sys.stderr.write(
            f"note: {SOURCE_ANCHOR_OPT_OUT_VAR} is set; skipping the charlie_work "
            "source-identity check (issue #1665)\n"
        )
        return
    violation = source_anchor_violation(repo_root, getattr(charlie_work, "__file__", None))
    if violation is not None:
        raise pytest.UsageError(violation)


__all__ = [
    "SOURCE_ANCHOR_OPT_OUT_VAR",
    "enforce_source_anchor",
    "source_anchor_violation",
]
