"""Shared fakes/helpers for the fleet-dispatch test modules.

Hoisted verbatim out of ``tests/test_fleet_dispatch.py`` (issue #1557,
Track 1) when that module was split into seam-named siblings -- the
``tests/_*.py`` hoisted-fixture convention is the sanctioned import target
for shared test helpers (see ``tests/test_zero_cross_test_import_guard.py``).
"""

from __future__ import annotations

import dataclasses
import json as _json
import os
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
import pytest
from charlie_work import fleet_dispatch as _fleet_dispatch_module
from charlie_work.config import (
    OrchestratorConfig,
    RunnerAllocationConfig,
    RunnerScalingConfig,
)
from charlie_work.fleet_dispatch import _CiFleetDirtyCheck
from charlie_work.notify import NotifyResult
from charlie_work.supervise import SelfDeployResult
from charlie_work.workflow import CommandResult


#: Where the notify-digest isolation in ``_patch_self_deploy_for_fleet_tests``
#: redirects file-sink writes, relative to the test's ``tmp_path``. Tests that
#: need to prove the redirect happened read the digest at this location.
ISOLATED_NOTIFY_DIGEST_REL = Path("notify") / "digest.jsonl"


def _iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


_supervisor_started = datetime.now(UTC).replace(microsecond=0)


SUPERVISOR_STARTED_AT = _iso(_supervisor_started)


SUPERVISOR_BEAT_AT = _iso(_supervisor_started + timedelta(seconds=2609))


class _FakeClock:
    """Monotonically advancing fake clock/sleep for supervisor tests."""

    def __init__(self, start: float = 0.0, auto_advance: float = 0.0) -> None:
        self._now = start
        self._auto_advance = auto_advance
        self.sleep_calls: list[float] = []

    def now(self) -> float:
        return self._now

    def sleep(self, seconds: float) -> None:
        self.sleep_calls.append(seconds)
        self._now += self._auto_advance if self._auto_advance else seconds


@pytest.fixture(autouse=True)
def _patch_self_deploy_for_fleet_tests(monkeypatch: Any, tmp_path: Path) -> dict[str, MagicMock]:
    """Self-deploy hits the real git/uv CLI; keep fleet supervisor unit tests hermetic.

    Also no-op the supervisor lifecycle instrumentation (issue #627) so existing
    supervisor tests do not write heartbeat/events to the real fleet dir. The
    lifecycle functions are replaced with MagicMocks keyed by name in the
    returned dict; dedicated wiring tests request this fixture to assert the
    calls. ``detect_prior_abnormal_exit`` defaults to ``None`` (no prior exit)
    and ``is_exit_alertable`` defaults to ``False`` so existing tests do not
    trip the prior-exit or alert branches.

    Issue #1859: notify emissions are redirected too. ``run_fleet_supervise``
    resolves ``notify.file_path`` from the *ambient* per-repo config at
    ``Path.cwd()``; a relative ``file_path`` (the production spelling
    ``.var/charlie-work/notify/digest.jsonl``) is then interpreted by
    ``_file_sink`` against the pytest cwd, so a suite run from a checkout
    whose own ``orchestrator.config.yaml`` enables notify appended synthetic
    ``"test no-op"`` entries to that checkout's REAL digest file -- the leak
    that made the dev checkout's digest look falsely alive for a month while
    the live daemon's writer was dead. The redirect mirrors what
    ``_patch_ci_fleet_dirty_for_hermetic_tests`` does for the real ci_fleet
    tree: file-sink emits run for real but land in this test's ``tmp_path``
    (under ``ISOLATED_NOTIFY_DIGEST_REL``), and the other sinks (desktop /
    shell / webhook -- also real side effects) return ``ok`` without firing.
    Dedicated tests that patch ``charlie_work.fleet_dispatch.emit_digest``
    themselves still see their own double.
    """
    monkeypatch.setattr(
        "charlie_work.fleet_dispatch.self_deploy",
        lambda _repo_root, **_kwargs: SelfDeployResult(
            ok=True,
            pulled=False,
            changed=False,
            synced=False,
            message="test no-op",
        ),
    )
    mocks: dict[str, MagicMock] = {}
    for name in (
        "detect_prior_abnormal_exit",
        "record_prior_abnormal_exit",
        "record_supervisor_started",
        "update_supervisor_heartbeat",
        "record_supervisor_exit",
        "is_exit_alertable",
    ):
        m = MagicMock(name=name)
        monkeypatch.setattr("charlie_work.fleet_dispatch." + name, m)
        mocks[name] = m
    mocks["detect_prior_abnormal_exit"].return_value = None
    mocks["is_exit_alertable"].return_value = False

    real_emit_digest = _fleet_dispatch_module.emit_digest
    isolated_digest = tmp_path / ISOLATED_NOTIFY_DIGEST_REL

    def _isolated_emit_digest(config: Any, digest: Any) -> NotifyResult:
        sink = str(getattr(config, "sink", "") or "").lower()
        if sink != "file":
            return NotifyResult(ok=True)
        file_path = getattr(config, "file_path", "")
        if not file_path:
            # Nothing to redirect -- preserve the real "file_path is empty"
            # error path, which cannot write anyway.
            return real_emit_digest(config, digest)
        try:
            redirected = dataclasses.replace(config, file_path=str(isolated_digest))
        except TypeError:
            # A non-dataclass config cannot be repointed safely; fail closed
            # rather than letting a real path through.
            return NotifyResult(
                ok=False,
                error="test isolation: notify config is not a dataclass; refusing real emit",
            )
        return real_emit_digest(redirected, digest)

    monkeypatch.setattr("charlie_work.fleet_dispatch.emit_digest", _isolated_emit_digest)
    return mocks


@pytest.fixture(autouse=True)
def _patch_ci_fleet_dirty_for_hermetic_tests(monkeypatch: Any) -> None:
    """Fleet dispatch tests must not fail because the real ci_fleet tree is dirty.

    The editable path dependency lives in a sibling checkout whose porcelain
    state is outside these tests' control. A dirty upstream tree would force
    every allocation-prologue test into dry-run mode and break assertions on
    the ``dry_run`` flag. This fixture makes the guard inert; tests that need
    to exercise the dirty path monkeypatch it explicitly.
    """
    monkeypatch.setattr(
        "charlie_work.fleet_dispatch._ci_fleet_worktree_dirty",
        lambda _module_file=None: _CiFleetDirtyCheck(is_dirty=False),
    )


def _drained_fleet_result() -> CommandResult:
    return CommandResult(
        True,
        "fleet pass complete",
        {
            "repos": {"owner/repo": {"ok": True}},
            "digest": {"count": 0, "events": []},
        },
    )


def _active_fleet_result(dispatched: int = 1) -> CommandResult:
    return CommandResult(
        True,
        "fleet pass complete",
        {
            "repos": {
                "owner/repo": {
                    "ok": True,
                    "dispatch": {"selected_count": dispatched},
                }
            },
            "digest": {"count": 0, "events": []},
        },
    )


def _failed_fleet_result(
    repo_messages: dict[str, str | None],
) -> CommandResult:
    """A fleet pass with one or more failing repos, each carrying its own message.

    Mirrors the shape ``fleet_loop`` actually returns (fleet_dispatch.py:1559-1565):
    every repo entry gets an ``ok`` bool and a ``message`` string alongside its
    ordinary data, regardless of pass outcome.
    """
    repos = {key: {"ok": False, "message": message} for key, message in repo_messages.items()}
    return CommandResult(
        False,
        "fleet pass complete",
        {"repos": repos, "digest": {"count": 0, "events": []}},
    )


def _mixed_fleet_result(
    conditions: dict[str, str | None] | None = None,
    errored: dict[str, str | None] | None = None,
    ok: dict[str, dict[str, Any]] | None = None,
) -> CommandResult:
    """A fleet pass with explicit errored / with-conditions / ok buckets.

    Mirrors the shape ``fleet_loop`` returns after issue #738: the exception
    path sets ``errored: True`` on the result data, while non-fatal
    ``ok=False`` conditions from ``app.loop()`` do not. This helper lets a
    test plant each bucket independently so the headline split is exercised
    in isolation.
    """
    repos: dict[str, dict[str, Any]] = {}
    for key, msg in (conditions or {}).items():
        repos[key] = {"ok": False, "message": msg}
    for key, msg in (errored or {}).items():
        repos[key] = {"ok": False, "message": msg, "errored": True}
    for key, data in (ok or {}).items():
        entry = {"ok": True}
        entry.update(data)
        repos[key] = entry
    return CommandResult(
        False,
        "fleet pass complete",
        {"repos": repos, "digest": {"count": 0, "events": []}},
    )


def _repair_payload(
    issue_numbers: list[int] | None = None,
    failures: list[int] | None = None,
    errored: list[int] | None = None,
    deferred: int = 0,
) -> dict[str, Any]:
    """Build a payload shaped like ``OrchestratorApp._repair_escalated_labels()``.

    Mirrors the real return shape (``workflow.py``'s ``_repair_escalated_labels``):
    ``issue_numbers`` is every subject whose ``transition()`` ran (successes and
    failures both), ``failures`` is the subset that did not fully apply, ``errored``
    is disjoint from both (nothing was written for those), and ``deferred`` is a
    plain count. Kept realistic rather than minimal so these tests exercise the
    same key combinations production actually emits.
    """
    return {
        "issue_numbers": issue_numbers or [],
        "failures": failures or [],
        "errored": errored or [],
        "deferred": deferred,
    }


_API_WORKER_YAML = """\
api_worker:
  enabled: {enabled}
  provider: kimi-k3
  providers:
    kimi-k3:
      base_url: https://api.moonshot.ai/anthropic
      api_key_env: MOONSHOT_API_KEY
      model: kimi-k3
      input_usd_per_mtok: 3.0
      output_usd_per_mtok: 15.0
      cached_input_usd_per_mtok: 0.30
  budget:
    max_usd_per_day: 5.0
    lifetime_usd: 15.0
"""


_BASE_YAML = """\
labels:
  ready: automated-ready
  queued: agent:queued
  in_progress: agent:in-progress
runtime:
  state_dir: .var/charlie-work
"""


def _make_repo(tmp_path: Path, name: str, *, api_worker: str | None) -> Path:
    """Create a repo dir with a config file. api_worker is the YAML snippet or None."""
    repo = tmp_path / name
    repo.mkdir(parents=True)
    config = repo / "orchestrator.config.yaml"
    content = _BASE_YAML
    if api_worker is not None:
        content += "\n" + api_worker
    config.write_text(content, encoding="utf-8")
    (repo / ".var" / "charlie-work").mkdir(parents=True)
    return repo


def _make_fleet_json(tmp_path: Path, fleet_dir: Path, repos: dict[str, dict[str, Any]]) -> None:
    fleet_json = fleet_dir / "fleet.json"
    fleet_json.parent.mkdir(parents=True, exist_ok=True)
    registry = {"version": 1, "repos": repos}
    fleet_json.write_text(_json.dumps(registry, indent=2), encoding="utf-8")


def _allocation_config(**overrides: Any) -> OrchestratorConfig:
    """A global config with the allocation section populated."""
    return OrchestratorConfig(
        runner_allocation=RunnerAllocationConfig(**overrides),
        runner_scaling=RunnerScalingConfig(managed_root="C:/fallback-root"),
    )


def _make_ci_fleet_git_repo(tmp_path: Path) -> Path:
    """Create a minimal editable-style ci_fleet repo with a clean ``src/`` tree."""
    repo = tmp_path / "ci_fleet_repo"
    repo.mkdir()
    pkg = repo / "src" / "ci_fleet"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("# ci_fleet", encoding="utf-8")

    # Use a per-test gitconfig so the commit does not depend on global config.
    gitconfig = tmp_path / "gitconfig"
    gitconfig.write_text("[user]\n\tname = Test\n\temail = test@test\n", encoding="utf-8")
    env = dict(os.environ, GIT_CONFIG_GLOBAL=str(gitconfig))

    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=repo,
        check=True,
        capture_output=True,
        env=env,
    )
    return repo
