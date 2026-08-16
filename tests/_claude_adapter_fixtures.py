"""Shared fixture for ``test_claude_code_adapter.py``'s externally-imported
surface.

Hoisted out of ``test_claude_code_adapter.py`` (issue #1284): a
``monkeypatch``-based ``create_worktree`` stand-in, imported by other test
modules. ``test_claude_code_adapter.py`` is one of the three monoliths
issue #1284 marks out of scope for a full split -- only this one exported
symbol moves; the rest of the file is untouched.

The ``_fake_worktree``/``_fake_worktree_with_venv`` builders this closes
over are NOT hoisted: every other test module that needs one defines its
own local copy rather than importing test_claude_code_adapter.py's (see
test_api_worker.py, test_devin_shell.py, test_fix_conflict_worktree.py,
test_fix_reviewer_argv.py) -- they are genuinely internal-only, not shared
fixtures. The import below is deferred to call time (inside the nested
``fake_create_worktree``) specifically to avoid a load-time circular
import: test_claude_code_adapter.py itself imports this module's
``_install_fake_create_worktree`` back (general back-reference rule), so a
module-level import here would try to load test_claude_code_adapter.py
while it is still mid-import.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _install_fake_create_worktree(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    calls: list[dict] | None = None,
    with_venv: bool = False,
) -> None:
    from charlie_work import claude_code

    def fake_create_worktree(
        repo_root,
        branch,
        *,
        base_ref="HEAD",
        worktrees_dir=None,
        venv_source=None,
        materialize_dirs=(),
        rework=False,
        recovery=None,
        issue_number=None,
        config=None,
        sessions_dir=None,
    ):
        # Deferred to call time -- see module docstring for why.
        from test_claude_code_adapter import _fake_worktree, _fake_worktree_with_venv

        if calls is not None:
            calls.append(
                {
                    "repo_root": repo_root,
                    "branch": branch,
                    "base_ref": base_ref,
                    "worktrees_dir": worktrees_dir,
                    "venv_source": venv_source,
                    "materialize_dirs": materialize_dirs,
                    "rework": rework,
                    "recovery": recovery,
                    "issue_number": issue_number,
                    "config": config,
                    "sessions_dir": sessions_dir,
                }
            )
        if with_venv:
            return _fake_worktree_with_venv(tmp_path, branch)
        return _fake_worktree(tmp_path, branch)

    monkeypatch.setattr(claude_code, "create_worktree", fake_create_worktree)
