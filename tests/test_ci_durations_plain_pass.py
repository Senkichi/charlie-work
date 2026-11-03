"""The shards' split durations are measured plain, never under map mode (issue #2924).

``--splitting-algorithm duration_based_chunks`` balances the six ``Tests shard``
jobs by the ``.test_durations`` the nightly saves. From #2349 the nightly ran in
map mode (coverage tracer, file-read audit hook, traced children) and stored its
durations from that pass. On the test ledger those per-test timings ran 2-3x a
pull request's for in-process tests but 0.63-0.68x for git-driven ones, so every
shard looked equally heavy to the splitter while shard 4/6 (the local-lane,
main-sync and orphan-sweep files, which drive git) cost ~1.4x the mean in a plain
pull-request run. The nightly now has one durations writer, the plain "Refresh
test durations" pass; these tests pin that shape.
Step bodies run under bash with a fake ``uv`` (``_ci_step_runner``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from _ci_step_runner import find_step, load_workflow, run_step

WF = load_workflow()
JOBS = WF["jobs"]
SHARD = find_step(WF, "tests-shard", "Run test shard")
REFRESH = find_step(WF, "tests-shard", "Refresh test durations")
STORE = "--store-durations"
# What decides which tests a shard runs; the two passes must agree on every one.
SPLIT_FLAGS = ("--splits", "--group", "--splitting-algorithm", "--durations-path")
# Everything that would switch a tracer or the file-read hook on in a child pytest.
MAP_ENV = ("CI_FLEET_MAP", "COVERAGE_CORE", "CI_FLEET_MAP_NODEID", "COVERAGE_PROCESS_START")

# One log line per call: the argv, then the env a tracer or recorder would key off.
SHIM = r"""#!/usr/bin/env bash
{
  printf 'uv'; printf ' %s' "$@"
  printf ' | ledger=%s map=%s core=%s nodeid=%s start=%s\n' \
    "${CI_FLEET_LEDGER:-}" "${CI_FLEET_MAP:-}" "${COVERAGE_CORE:-}" \
    "${CI_FLEET_MAP_NODEID:-}" "${COVERAGE_PROCESS_START:-}"
} >> "$FAKE_CMD_LOG"
case "$*" in
  *"map prepare"*) [ "${FAKE_PREPARE_RC:-0}" = 0 ] || exit "$FAKE_PREPARE_RC"
    printf -- '-p\r\nci_fleet.selection.map_mode\r\n'; exit 0;;
esac
exit "${FAKE_UV_RC:-0}"
"""


def _run(step: dict, tmp_path: Path, **env: str) -> tuple[int, list[str]]:
    """Execute a step body; return its exit code and the fake ``uv`` pytest calls."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    shim = bin_dir / "uv"
    shim.write_text(SHIM, encoding="utf-8", newline="\n")
    shim.chmod(0o755)
    log = tmp_path / "cmds.log"
    log.write_text("", encoding="utf-8")
    base = {
        "SPLITS": "3",
        "GROUP": "2",
        "REPO": "owner/repo-a",
        "BASE_SHA": "b" * 40,
        "FAKE_CMD_LOG": log.as_posix(),
        # Whatever map state the runner of THIS test inherited must not reach the step.
        **dict.fromkeys(MAP_ENV, ""),
    }
    result = run_step(step, tmp_path, {**base, **env})
    calls = [c for c in log.read_text(encoding="utf-8").splitlines() if " pytest " in c]
    return result.returncode, calls


def _flag_values(call: str) -> dict[str, str]:
    tokens = call.split(" | ")[0].split()
    return {flag: tokens[tokens.index(flag) + 1] for flag in SPLIT_FLAGS}


# --- the single writer ---------------------------------------------------------------------


def test_exactly_one_step_stores_durations_and_it_is_the_refresh() -> None:
    writers = [
        (job_id, step.get("name"))
        for job_id, job in JOBS.items()
        for step in job.get("steps", [])
        if STORE in str(step.get("run", ""))
    ]
    assert writers == [("tests-shard", "Refresh test durations")]


@pytest.mark.parametrize(
    ("event", "map_input", "prepare_ok"),
    [
        ("schedule", "", True),
        ("schedule", "", False),
        ("workflow_dispatch", "true", True),
        ("workflow_dispatch", "false", True),
        ("pull_request", "", True),
        ("push", "", True),
    ],
)
def test_the_shard_step_never_stores_durations(
    tmp_path: Path, event: str, map_input: str, prepare_ok: bool
) -> None:
    code, calls = _run(
        SHARD,
        tmp_path,
        EVENT_NAME=event,
        MAP_INPUT=map_input,
        FAKE_UV_RC="0",
        # A failed `map prepare` is the only path that runs the nightly unmapped.
        **({} if prepare_ok else {"FAKE_PREPARE_RC": "1"}),
    )
    assert code == 0
    assert calls, "the shard step ran no pytest"
    assert not [c for c in calls if STORE in c]


# --- the refresh pass ----------------------------------------------------------------------


def test_the_refresh_pass_runs_one_plain_recorded_free_pytest(tmp_path: Path) -> None:
    code, calls = _run(REFRESH, tmp_path)
    assert code == 0
    assert calls == [
        "uv run --no-sync pytest --tb=short -n auto --dist=load --splits 3 --group 2 "
        "--splitting-algorithm duration_based_chunks --durations-path .test_durations "
        "--store-durations | ledger=off map= core= nodeid= start="
    ]


def test_the_refresh_pass_runs_without_map_mode(tmp_path: Path) -> None:
    # Durations measured under the tracer are exactly what this pass avoids. It is
    # its own shell, so the map pass's `export`s do not reach it; this pins that
    # the step neither sets map env itself nor declares any (the fake `uv` logs the
    # env the pytest would have seen).
    code, calls = _run(REFRESH, tmp_path)
    assert code == 0
    [call] = calls
    assert "map= core= nodeid= start=" in call
    assert not set(MAP_ENV) & set(REFRESH.get("env", {}))
    assert "map prepare" not in REFRESH["run"] and "CI_FLEET_MAP" not in REFRESH["run"]


def test_the_refresh_pass_splits_exactly_like_the_map_pass(tmp_path: Path) -> None:
    # Same restored .test_durations + same split flags -> same tests per shard. A
    # drift in any one flag would store durations for a different slice of the suite.
    sub = {"map": tmp_path / "map", "refresh": tmp_path / "refresh"}
    for path in sub.values():
        path.mkdir()
    _, [mapped] = _run(SHARD, sub["map"], EVENT_NAME="schedule", SPLITS="6", GROUP="4")
    _, [refresh] = _run(REFRESH, sub["refresh"], SPLITS="6", GROUP="4")
    assert _flag_values(mapped) == _flag_values(refresh)
    assert _flag_values(refresh)["--splits"] == "6" and _flag_values(refresh)["--group"] == "4"
    assert "ci_fleet.selection.map_mode" in mapped  # the control: this really is the map pass
    assert "ci_fleet.selection.map_mode" not in refresh


def test_the_refresh_pass_exits_with_the_suite_s_code(tmp_path: Path) -> None:
    # Failing is the step's own business; continue-on-error (below) keeps it from
    # failing the nightly, so its exit code must be pytest's, not swallowed here.
    code, _ = _run(REFRESH, tmp_path, FAKE_UV_RC="3")
    assert code == 3


# --- the wiring ----------------------------------------------------------------------------


def test_the_refresh_step_wiring() -> None:
    assert REFRESH["if"] == "github.event_name == 'schedule'"
    assert REFRESH["continue-on-error"] is True
    assert REFRESH["shell"] == "bash"
    assert REFRESH["env"] == {
        "SPLITS": "${{ needs.coverage.outputs.splits }}",
        "GROUP": "${{ matrix.group }}",
        "CI_FLEET_LEDGER": "off",  # a string: a bare YAML off is the boolean False
    }
    assert "${{" not in REFRESH["run"]
    assert "--junit-xml" not in REFRESH["run"]  # the verdict reads the map pass's junit
    assert "uv run --no-sync" in REFRESH["run"]  # a plain `uv run` would resync the venv


def test_the_refresh_runs_exactly_when_the_cache_is_saved() -> None:
    save = find_step(WF, "Tests", "Save test durations")
    assert save["if"].startswith(REFRESH["if"])
    # Refreshing on any other event would be a second full run nobody consumes.
    assert "inputs.map" not in REFRESH["if"]


def test_the_refresh_runs_after_the_map_pass_and_before_the_durations_upload() -> None:
    names = [s.get("name") for s in JOBS["tests-shard"]["steps"]]
    assert names.index("Run test shard") < names.index("Refresh test durations")
    assert names.index("Refresh test durations") < names.index("Upload durations")
    # The upload reads the file the refresh wrote.
    upload = find_step(WF, "tests-shard", "Upload durations")
    assert upload["with"]["path"] == ".test_durations"
