"""Issue #2795: the local merge gate's suite launch env omits provider keys.

The gate env is built by ``env_sanitize.build_local_gate_env``, which copies
the supervisor's ambient environment (``{**os.environ, ...}``) and is handed
to ``local_suite_runner.launch_suite_gate``, which executes the PR branch's
worker-authored test suite. This test drives the real gate launch path for an
approved record and asserts the env it spawns with lacks the configured
``api_worker.providers[*].api_key_env`` variable, while a control variable the
supervisor sets does flow through -- so removing the
``strip_provider_api_keys`` call inside ``env_sanitize.build_local_gate_env``
makes it fail.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest

from charlie_work.local_suite_runner import SuiteGateLaunch

from _local_gate_async_fixtures import (  # noqa: E402
    UNALLOCATABLE_PID,
    _adopt_and_approve,
    _init_repo,
    _lane_app,
    _lane_config,
    _make_branch,
    _no_windows_child_enumeration,  # noqa: F401 -- registers the kill-stub autouse fixture
)

GATE_KEY_ENV = "EXAMPLE_GATE_PROVIDER_API_KEY"
GATE_CONTROL_ENV = "EXAMPLE_GATE_CONTROL_VAR"


def _config_with_provider(repo_root: Path, issues_dir: Path) -> Any:
    """Lane config carrying one enabled api provider keyed on GATE_KEY_ENV."""
    return _lane_config(
        repo_root,
        issues_dir,
        api_worker={
            "enabled": True,
            "provider": "example",
            "providers": {
                "example": {
                    "base_url": "https://api.example.test/anthropic",
                    "api_key_env": GATE_KEY_ENV,
                    "model": "example-model",
                    "input_usd_per_mtok": 3.0,
                    "output_usd_per_mtok": 15.0,
                }
            },
        },
    )


def test_merge_gate_launch_env_lacks_configured_provider_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = Path(tempfile.mkdtemp(prefix="cw-gate-env-"))
    try:
        _init_repo(repo_root)
        issues_dir = repo_root / "docs" / "issues"
        head = _make_branch(repo_root, "agent/issue-2795-x", "a.py", "a = 1\n")
        app = _lane_app(repo_root, issues_dir, config=_config_with_provider(repo_root, issues_dir))
        _adopt_and_approve(app, issues_dir, 2795, "agent/issue-2795-x", head)

        monkeypatch.setenv(GATE_KEY_ENV, "super-secret-value")
        monkeypatch.setenv(GATE_CONTROL_ENV, "control-value")

        captured: dict[str, Any] = {}

        def fake_launch_suite_gate(
            worktree_path: Path,
            suite_argv: list[str],
            *,
            paths: Any,
            head_sha: str,
            base_sha: str | None,
            env: dict[str, str] | None = None,
        ) -> SuiteGateLaunch:
            captured["env"] = dict(env or {})
            # The pid is unallocatable on every OS, so no real suite child and
            # no kill-tree teardown pressure follow from this fake launch.
            return SuiteGateLaunch(pid=UNALLOCATABLE_PID, process_start_time=None, paths=paths)

        monkeypatch.setattr(
            "charlie_work.local_suite_runner.launch_suite_gate", fake_launch_suite_gate
        )

        outcomes = app._local_merge_approved()
        assert outcomes and outcomes[0].get("outcome") == "suite_launched", outcomes
        assert captured["env"] is not None
        assert GATE_KEY_ENV not in captured["env"], (
            "the gate suite env inherited the configured provider key"
        )
        # Control: the ambient copy itself works, so the strip above is not
        # vacuously green because the var never entered the env at all.
        assert captured["env"].get(GATE_CONTROL_ENV) == "control-value"
    finally:
        # Unallocatable pid: the usual gate-kill teardown is a no-op, but keep
        # the shared helper in the path by contract.
        pass
