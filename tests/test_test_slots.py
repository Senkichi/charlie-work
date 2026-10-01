"""Host test-slot pool (issue #2124): config, env arming, event drain, and the plugin.

The plugin tests drive a *real* inner ``pytest`` subprocess against a generated
project, with real OS file locks, because the contract is exactly about
processes: a killed holder frees its slot, the gate takes slot 0 while every
agent slot is held, a timeout exits (never hangs) with the distinct code.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from _script_loader import load_script_module
from charlie_work import test_slots
from charlie_work.config import ConfigError, build_config_from_data
from charlie_work.test_slots import SlotPoolConfig, arm_env, drain_wait_timeouts, slot_count

PLUGIN_PATH = test_slots.plugin_dir() / "test_slot_plugin.py"
plugin = load_script_module(PLUGIN_PATH, "test_slot_plugin_under_test")


# ---------------------------------------------------------------- config / env


def test_plugin_dir_holds_only_the_plugin() -> None:
    assert [p.name for p in test_slots.plugin_dir().glob("*.py")] == ["test_slot_plugin.py"]
    assert [p for p in test_slots.plugin_dir().iterdir() if p.is_file()] == [PLUGIN_PATH]


def test_plugin_and_orchestrator_agree_on_the_wire_names() -> None:
    names = ("ENV_DIR", "ENV_COUNT", "ENV_MIN_ITEMS", "ENV_TIMEOUT", "ENV_ROLE")
    for name in names:
        assert getattr(plugin, name) == getattr(test_slots, name), name
    assert plugin.ROLE_AGENT == test_slots.ROLE_AGENT
    assert plugin.ROLE_GATE == test_slots.ROLE_GATE
    assert plugin.TIMEOUT_RECORD_DIR == test_slots.TIMEOUT_RECORD_DIR


def test_defaults_are_on_width_4_min_300_timeout_8_minutes() -> None:
    cfg = build_config_from_data({}).test_slots
    assert cfg == SlotPoolConfig(enabled=True, width=4, min_items=300, wait_timeout_seconds=480)


@pytest.mark.parametrize(
    "bad",
    [{"width": 0}, {"width": "4"}, {"min_items": -1}, {"wait_timeout_seconds": 0}, {"enabled": 1}],
)
def test_config_rejects_invalid_values(bad: dict) -> None:
    with pytest.raises(ConfigError):
        build_config_from_data({"test_slots": bad})


def test_slot_count_floors_at_two_so_agents_always_have_a_slot() -> None:
    assert slot_count(4, cpu_count=16) == 4
    assert slot_count(4, cpu_count=8) == 2
    assert slot_count(16, cpu_count=4) == 2


def test_arm_env_disabled_is_empty() -> None:
    assert arm_env(SlotPoolConfig(enabled=False), role="agent", base_env={}) == {}


def test_arm_env_arms_plugin_and_prepends_ambient_values() -> None:
    env = arm_env(
        SlotPoolConfig(width=3),
        role="gate",
        base_env={"PYTHONPATH": "/x", "PYTEST_PLUGINS": "other"},
    )
    assert env["PYTHONPATH"].split(os.pathsep) == [str(test_slots.plugin_dir()), "/x"]
    assert env["PYTEST_PLUGINS"] == "test_slot_plugin,other"
    assert env["PYTEST_XDIST_AUTO_NUM_WORKERS"] == "3"
    assert env[test_slots.ENV_ROLE] == "gate"
    assert env[test_slots.ENV_DIR] == str(test_slots.slot_dir())


def test_drain_emits_each_record_once_and_claims_it(tmp_path: Path) -> None:
    records = tmp_path / "timeouts"
    records.mkdir()
    (records / "1.json").write_text(json.dumps({"role": "agent"}), encoding="utf-8")
    (records / "2.json").write_text("not json", encoding="utf-8")
    seen: list[dict] = []
    assert drain_wait_timeouts(tmp_path, seen.append) == 1
    assert seen == [{"role": "agent"}]
    assert list(records.iterdir()) == []
    assert drain_wait_timeouts(tmp_path, seen.append) == 0


# --------------------------------------------------------------- real processes


def _env(slot_dir: Path, *, role: str = "agent", count: int = 2, min_items: int = 3, **extra):
    env = {
        k: v for k, v in os.environ.items() if not k.startswith(("PYTEST_", "CHARLIE_TEST_SLOT"))
    }
    env.update(
        PYTHONPATH=str(test_slots.plugin_dir()),
        PYTEST_PLUGINS="test_slot_plugin",
        CHARLIE_TEST_SLOT_DIR=str(slot_dir),
        CHARLIE_TEST_SLOT_COUNT=str(count),
        CHARLIE_TEST_SLOT_MIN_ITEMS=str(min_items),
        CHARLIE_TEST_SLOT_WAIT_TIMEOUT_SECONDS="60",
        CHARLIE_TEST_SLOT_ROLE=role,
        CHARLIE_TEST_SLOT_POLL_SECONDS="0.05",
    )
    env.update({k: str(v) for k, v in extra.items()})
    return env


def _project(tmp_path: Path, n_tests: int, *, ready: Path | None = None, hold: float = 0) -> Path:
    proj = tmp_path / "proj"
    proj.mkdir(exist_ok=True)
    body = textwrap.dedent(
        f"""
        import pathlib, time
        import pytest

        @pytest.mark.parametrize("i", range({n_tests}))
        def test_n(i):
            if i == 0:
                ready = {str(ready)!r}
                if ready != "None":
                    pathlib.Path(ready).write_text("x")
                    time.sleep({hold})
        """
    )
    (proj / "test_wide.py").write_text(body, encoding="utf-8")
    return proj


def _pytest(proj: Path, env: dict, *args: str, **popen) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", *args],
        cwd=proj,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        **popen,
    )


def _locked(slot_dir: Path, k: int) -> bool:
    """True when slot ``k`` is currently held by someone else."""
    handle = plugin._try_lock(slot_dir / f"slot-{k}.lock")
    if handle is None:
        return True
    plugin._unlock(handle)
    return False


def _wait_for(predicate, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError("condition not reached in time")


_HOLDER = """
import sys, time, importlib.util
spec = importlib.util.spec_from_file_location("p", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
h = m._try_lock(__import__("pathlib").Path(sys.argv[2]))
print("held" if h else "busy", flush=True)
time.sleep(600)
"""


def _hold_slot(slot_dir: Path, k: int) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-c", _HOLDER, str(PLUGIN_PATH), str(slot_dir / f"slot-{k}.lock")],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout.readline().strip() == "held"
    return proc


def _force_kill(proc: subprocess.Popen) -> None:
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/PID", str(proc.pid)], capture_output=True, check=False)
    else:
        proc.kill()
    proc.wait(timeout=30)


def test_killed_holder_releases_its_slot(tmp_path: Path) -> None:
    slots = tmp_path / "slots"
    holder = _hold_slot(slots, 1)
    try:
        assert _locked(slots, 1)
    finally:
        _force_kill(holder)
    assert not _locked(slots, 1)


def test_run_under_threshold_never_takes_a_slot(tmp_path: Path) -> None:
    slots = tmp_path / "slots"
    proc = _pytest(_project(tmp_path, 2), _env(slots, min_items=3))
    out, _ = proc.communicate(timeout=120)
    assert proc.returncode == 0, out
    assert not (slots / "slot-1.lock").exists()


def test_run_at_threshold_takes_an_agent_slot_and_releases_it(tmp_path: Path) -> None:
    slots, ready = tmp_path / "slots", tmp_path / "ready"
    proc = _pytest(_project(tmp_path, 3, ready=ready, hold=3), _env(slots, min_items=3))
    _wait_for(ready.exists)
    assert _locked(slots, 1)
    assert not _locked(slots, 0), "agents must never take the gate slot"
    out, _ = proc.communicate(timeout=120)
    assert proc.returncode == 0, out
    assert not _locked(slots, 1)


def test_gate_acquires_slot_zero_while_every_agent_slot_is_held(tmp_path: Path) -> None:
    slots, ready = tmp_path / "slots", tmp_path / "ready"
    holder = _hold_slot(slots, 1)  # count=2 -> the only agent slot
    try:
        proc = _pytest(
            _project(tmp_path, 3, ready=ready, hold=3),
            _env(slots, role="gate", CHARLIE_TEST_SLOT_WAIT_TIMEOUT_SECONDS=10),
        )
        _wait_for(ready.exists)
        assert _locked(slots, 0)
        out, _ = proc.communicate(timeout=120)
        assert proc.returncode == 0, out
    finally:
        _force_kill(holder)


def test_timeout_exits_with_distinct_code_message_and_record(tmp_path: Path) -> None:
    slots = tmp_path / "slots"
    holder = _hold_slot(slots, 1)
    try:
        t0 = time.monotonic()
        proc = _pytest(
            _project(tmp_path, 3),
            _env(
                slots,
                CHARLIE_TEST_SLOT_WAIT_TIMEOUT_SECONDS=2,
                CHARLIE_TEST_SLOT_LOG_INTERVAL_SECONDS=0.2,
            ),
        )
        out, _ = proc.communicate(timeout=60)
    finally:
        _force_kill(holder)
    assert proc.returncode == plugin.HOST_BUSY_EXIT_CODE, out
    assert time.monotonic() - t0 < 30
    assert "test-slot: waiting (1 held, 1 queued)" in out
    assert "test-slot: host busy; retry or run targeted tests" in out
    (record,) = (slots / "timeouts").glob("*.json")
    assert json.loads(record.read_text(encoding="utf-8"))["role"] == "agent"


def test_xdist_only_the_controller_takes_a_slot(tmp_path: Path) -> None:
    pytest.importorskip("xdist")
    slots, ready = tmp_path / "slots", tmp_path / "ready"
    proc = _pytest(
        _project(tmp_path, 6, ready=ready, hold=3),
        _env(slots, CHARLIE_TEST_SLOT_WAIT_TIMEOUT_SECONDS=15),
        "-n",
        "2",
    )
    _wait_for(ready.exists)
    assert _locked(slots, 1)
    out, _ = proc.communicate(timeout=120)
    # A worker that also tried to take the (single) agent slot would block on
    # the controller's lock and the run would end in the host-busy exit code.
    assert proc.returncode == 0, out


def test_xdist_timeout_does_not_hang(tmp_path: Path) -> None:
    pytest.importorskip("xdist")
    slots = tmp_path / "slots"
    holder = _hold_slot(slots, 1)
    try:
        proc = _pytest(
            _project(tmp_path, 6),
            _env(slots, CHARLIE_TEST_SLOT_WAIT_TIMEOUT_SECONDS=2),
            "-n",
            "2",
        )
        out, _ = proc.communicate(timeout=90)
    finally:
        _force_kill(holder)
    assert proc.returncode == plugin.HOST_BUSY_EXIT_CODE, out
    assert "test-slot: host busy; retry or run targeted tests" in out


def test_plugin_is_inert_without_count(tmp_path: Path) -> None:
    slots = tmp_path / "slots"
    env = _env(slots, min_items=1)
    del env["CHARLIE_TEST_SLOT_COUNT"]
    proc = _pytest(_project(tmp_path, 3), env)
    out, _ = proc.communicate(timeout=120)
    assert proc.returncode == 0, out
    assert not slots.exists()


# ------------------------------------------------------------------- wiring


def _app(tmp_path: Path, config):
    from _fakes_github import FakeGitHub
    from charlie_work.paths import runtime_paths
    from charlie_work.workflow import OrchestratorApp

    return OrchestratorApp(
        tmp_path, runtime_paths(tmp_path, config.runtime.state_dir), config, FakeGitHub()
    )


def test_maintenance_pass_turns_timeout_records_into_events(tmp_path: Path, monkeypatch) -> None:
    from charlie_work.instrumentation import query_events
    from charlie_work.pass_deadline import PassDeadline, run_deadline_guarded_maintenance
    from charlie_work.workflow import CommandResult

    slots = tmp_path / "slots"
    (slots / "timeouts").mkdir(parents=True)
    (slots / "timeouts" / "1.json").write_text(
        json.dumps({"role": "agent", "waited_seconds": 480.0}), encoding="utf-8"
    )
    monkeypatch.setattr(test_slots, "slot_dir", lambda: slots)
    app = _app(tmp_path, build_config_from_data({}))
    run_deadline_guarded_maintenance(PassDeadline(None, CommandResult), app)
    (event,) = query_events(app.paths.state_file, kind="test_slot_wait_timeout")
    assert event["payload"]["waited_seconds"] == 480.0
    assert list((slots / "timeouts").iterdir()) == []


def test_maintenance_pass_leaves_records_alone_when_disabled(tmp_path: Path, monkeypatch) -> None:
    from charlie_work.pass_deadline import PassDeadline, run_deadline_guarded_maintenance
    from charlie_work.workflow import CommandResult

    slots = tmp_path / "slots"
    (slots / "timeouts").mkdir(parents=True)
    (slots / "timeouts" / "1.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(test_slots, "slot_dir", lambda: slots)
    app = _app(tmp_path, build_config_from_data({"test_slots": {"enabled": False}}))
    run_deadline_guarded_maintenance(PassDeadline(None, CommandResult), app)
    assert [p.name for p in (slots / "timeouts").iterdir()] == ["1.json"]


def test_agent_settings_arm_the_plugin_and_kill_switch_disarms(tmp_path: Path) -> None:
    on = _app(tmp_path, build_config_from_data({}))._adapter_settings(adapter="claude-code")
    assert on.worker_env[test_slots.ENV_ROLE] == "agent"
    assert on.worker_env["PYTEST_XDIST_AUTO_NUM_WORKERS"] == "4"
    off_cfg = build_config_from_data({"test_slots": {"enabled": False}})
    off = _app(tmp_path, off_cfg)._adapter_settings(adapter="claude-code")
    assert test_slots.ENV_COUNT not in off.worker_env


def test_launch_suite_gate_forwards_env_to_the_runner(tmp_path: Path, monkeypatch) -> None:
    from unittest import mock

    from charlie_work import local_suite_runner

    paths = local_suite_runner.suite_gate_paths(tmp_path, 7)
    with mock.patch.object(local_suite_runner, "popen_worker") as popen:
        popen.return_value.pid = 1
        local_suite_runner.launch_suite_gate(
            tmp_path, ["pytest"], paths=paths, head_sha="h", base_sha=None, env={"A": "1"}
        )
    assert popen.call_args.kwargs["env"] == {"A": "1"}
