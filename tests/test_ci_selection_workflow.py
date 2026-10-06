"""Test impact selection in charlie-work's CI (TIS-CW-1).

The nightly shards build coverage maps, the ``map`` job merges them into the
one ``ci-fleet-map`` artifact, shard 1 of every pull request (and merge-queue
draft) records a shadow selection, and ``ci-fleet-bisect.yml`` answers the
nightly triage's dispatches. The workflow files must agree with the ci-fleet
code that dispatches, polls and downloads them, so these tests import it.
Step bodies run under bash with fake ``gh``, ``uv`` and ``uvx``
(``_ci_step_runner``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from _ci_step_runner import CI_YML, REPO, find_step, load_workflow, run_step
from ci_fleet.selection import map_import, map_locate
from ci_fleet.selection.config import load_config
from ci_fleet.selection.nightly import bisect, triage

WF = load_workflow()
JOBS = WF["jobs"]
BISECT = load_workflow(CI_YML.parent / triage.BISECT_WORKFLOW)
SHARD = find_step(WF, "tests-shard", "Run test shard")
LOCATE = find_step(WF, "coverage", "Locate the map")
COMPLETE = find_step(WF, "map", "Check every shard mapped")
SHA = "b" * 40
PYTEST = (
    "uv run --no-sync pytest --tb=short -n auto --dist=load --splits 3 --group {g} "
    "--splitting-algorithm duration_based_chunks --durations-path .test_durations "
    "--junit-xml=pytest-junit-{g}.xml"
)
# Logs one line per call: the argv, then the recorder context it ran under.
# `map prepare` answers with CRLF line ends, as it does under Git Bash.
SHIM = r"""#!/usr/bin/env bash
{ printf '%s' NAME; printf ' %s' "$@"; printf ' ctx=%s\n' "${CI_FLEET_LEDGER_CONTEXT:-}"; } >> "$FAKE_CMD_LOG"
case "$*" in
  *"map prepare"*) [ "${FAKE_PREPARE_RC:-0}" = 0 ] || exit "$FAKE_PREPARE_RC"
    printf -- '-p\r\nci_fleet.selection.map_mode\r\n'; exit 0;;
esac
exit "${RC_VAR:-0}"
"""


def _triggers(doc: dict) -> dict:
    return doc.get("on", doc.get(True)) or {}


def _step(job: dict, name: str) -> dict:
    [step] = [s for s in job["steps"] if s.get("name") == name]
    return step


def _uses(job: dict, action: str) -> list[dict]:
    return [s for s in job["steps"] if str(s.get("uses", "")).startswith(action)]


def _shard(tmp_path: Path, **env: str) -> tuple:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, rc in (("uv", "FAKE_UV_RC"), ("uvx", "FAKE_UVX_RC")):
        shim = bin_dir / name
        shim.write_text(
            SHIM.replace("NAME", name).replace("RC_VAR", rc), encoding="utf-8", newline="\n"
        )
        shim.chmod(0o755)
    log = tmp_path / "cmds.log"
    log.write_text("", encoding="utf-8")
    base = {
        "SPLITS": "3",
        "GROUP": "1",
        "REPO": "owner/repo-a",
        "BASE_SHA": SHA,
        "CI_FLEET_LEDGER_CONTEXT": "",
        "FAKE_CMD_LOG": log.as_posix(),
    }
    result = run_step(SHARD, tmp_path, {**base, **env})
    return result, log.read_text(encoding="utf-8").splitlines()


# --- the shard step ------------------------------------------------------------------------


def test_a_pull_request_s_first_shard_records_a_shadow_selection(tmp_path: Path) -> None:
    result, cmds = _shard(tmp_path, EVENT_NAME="pull_request")
    assert result.returncode == 0, result.stderr
    assert cmds == [
        f"uvx --from ci-fleet~=0.7.0 ci-fleet test --repo owner/repo-a --base {SHA} "
        f"--context ci --shadow -- {PYTEST.format(g=1)} ctx="
    ]


@pytest.mark.parametrize(
    ("event", "group"),
    [
        ("pull_request", "2"),
        ("push", "1"),
        ("workflow_dispatch", "1"),
    ],
)
def test_other_shards_and_events_run_the_shard_unwrapped(
    tmp_path: Path, event: str, group: str
) -> None:
    result, cmds = _shard(tmp_path, EVENT_NAME=event, GROUP=group)
    assert result.returncode == 0, result.stderr
    assert cmds == [f"{PYTEST.format(g=group)} ctx="]


def test_the_nightly_shard_runs_in_map_mode(tmp_path: Path) -> None:
    result, cmds = _shard(tmp_path, EVENT_NAME="schedule", GROUP="2")
    assert result.returncode == 0, result.stderr
    assert cmds == [
        "uv run --no-sync ci-fleet map prepare ctx=nightly",
        f"{PYTEST.format(g=2)} --store-durations -p ci_fleet.selection.map_mode ctx=nightly",
    ]


def test_a_failed_map_prepare_still_runs_the_nightly_shard(tmp_path: Path) -> None:
    result, cmds = _shard(tmp_path, EVENT_NAME="schedule", FAKE_PREPARE_RC="1")
    assert result.returncode == 0, result.stderr
    assert cmds[-1] == f"{PYTEST.format(g=1)} --store-durations ctx=nightly"
    assert "::warning::" in result.stdout


@pytest.mark.parametrize(
    ("event", "var"),
    [
        ("pull_request", "FAKE_UVX_RC"),
        ("push", "FAKE_UV_RC"),
        ("schedule", "FAKE_UV_RC"),
    ],
)
def test_the_shard_exits_with_the_suite_s_code(tmp_path: Path, event: str, var: str) -> None:
    result, _ = _shard(tmp_path, EVENT_NAME=event, **{var: "3"})
    assert result.returncode == 3


def test_the_shard_step_wiring() -> None:
    env = SHARD["env"]
    assert env[map_locate.ARTIFACT_ID_ENV] == "${{ needs.coverage.outputs.map_artifact_id }}"
    assert env["CI_FLEET_SELECT_LEVEL"] == "${{ vars.CI_FLEET_SELECT_LEVEL }}"
    assert env["BASE_SHA"] == "${{ github.event.pull_request.base.sha }}"
    assert env["GH_TOKEN"] == "${{ github.token }}"
    assert "${{" not in SHARD["run"]
    shard = JOBS["tests-shard"]
    assert shard["permissions"] == {"contents": "read", "actions": "read"}
    [checkout] = _uses(shard, "actions/checkout@")
    depth = checkout["with"]["fetch-depth"]
    # Full history for the selector's diff on shard 1 of a PR. The string '0'
    # is truthy in an expression, so `&& '0' ||` yields it; a number 0 would not.
    assert "'0'" in depth and "matrix.group == 1" in depth and "pull_request" in depth


def test_every_uv_run_after_the_map_install_keeps_it() -> None:
    for step in JOBS["tests-shard"]["steps"]:
        run = str(step.get("run", ""))
        assert "uv run " not in run.replace("uv run --no-sync ", ""), step.get("name")


# --- locating the map ----------------------------------------------------------------------


def _locate(tmp_path: Path, **env: str):
    return run_step(LOCATE, tmp_path, {"REPO": "owner/repo-a", **env})


def test_the_map_is_located_once_per_run(tmp_path: Path) -> None:
    result = _locate(tmp_path, FAKE_ARTIFACTS_OUT="123456")
    assert result.returncode == 0, result.stderr
    assert result.outputs == {"id": "123456"}
    [call] = result.gh_calls
    assert call.startswith(
        f"api repos/owner/repo-a/actions/artifacts?name={map_import.ARTIFACT_NAME}&"
    )
    assert "select(.expired | not)" in call and "sort_by(.created_at)" in call


@pytest.mark.parametrize(
    "env",
    [
        {"FAKE_ARTIFACTS_RC": "1"},
        {"FAKE_ARTIFACTS_OUT": ""},
        {"FAKE_ARTIFACTS_OUT": "null"},
        {"FAKE_ARTIFACTS_OUT": "12 34"},
    ],
)
def test_any_locate_failure_leaves_the_id_empty(tmp_path: Path, env: dict) -> None:
    result = _locate(tmp_path, **env)
    assert result.returncode == 0, result.stderr
    assert result.outputs == {"id": ""}


def test_locate_wiring() -> None:
    coverage = JOBS["coverage"]
    assert LOCATE["id"] == "map"
    assert LOCATE["if"] == "github.event_name == 'pull_request'"
    assert "${{" not in LOCATE["run"]
    assert coverage["outputs"]["map_artifact_id"] == "${{ steps.map.outputs.id }}"
    assert coverage["permissions"]["actions"] == "read"


@pytest.mark.parametrize(
    ("event", "shards", "timeout"),
    [
        ("push", "6", "15"),
        ("pull_request", "2", "30"),
        ("schedule", "6", "45"),
        ("schedule", "2", "90"),
    ],
)
def test_the_nightly_shard_timeout_allows_map_mode(
    tmp_path: Path, event: str, shards: str, timeout: str
) -> None:
    step = find_step(WF, "coverage", "Plan shards")
    result = run_step(step, tmp_path, {"EVENT_NAME": event, "SHARDS_VAR": shards})
    assert result.returncode == 0, result.stderr
    assert result.outputs["shard_timeout"] == timeout


# --- building and merging the map ----------------------------------------------------------


def test_the_nightly_shards_build_their_maps() -> None:
    assert "schedule" in _triggers(WF)
    shard = JOBS["tests-shard"]
    for name in ("Map mode", "Build the shard map"):
        assert "github.event_name == 'schedule'" in _step(shard, name)["if"]
    install = _step(shard, "Map mode")["run"]
    assert 'uv pip install "ci-fleet[map]~=0.7.0"' in install
    # Map mode never leaves the shard step (gap 47): no `--github-env` anywhere.
    assert "map prepare" not in install
    run = _step(shard, "Run test shard")["run"]
    assert "export CI_FLEET_MAP=1 COVERAGE_CORE=ctrace" in run
    assert not [
        n
        for n, j in JOBS.items()
        for st in j.get("steps", [])
        if "--github-env" in str(st.get("run", ""))
    ]
    build = _step(shard, "Build the shard map")
    assert "!cancelled()" in build["if"]  # a failing nightly still maps what ran
    assert (
        '--map-sha "$SHA"' in build["run"] and '--out "map-shard-$GROUP.json.gz"' in build["run"]
    )
    assert "${{" not in build["run"]
    uploads = {s["with"]["name"]: s["with"] for s in _uses(shard, "actions/upload-artifact@")}
    assert (
        uploads["map-shard-${{ matrix.group }}"]["path"] == "map-shard-${{ matrix.group }}.json.gz"
    )
    data = uploads["map-coverage-data-${{ matrix.group }}"]
    assert data["include-hidden-files"] is True and data["retention-days"] == 7


def test_every_shard_uploads_the_ledger_spool() -> None:
    # Ledger S1's CI artifact: shard 1's shadow selection record travels in it.
    shard = JOBS["tests-shard"]
    [spool] = [
        s
        for s in _uses(shard, "actions/upload-artifact@")
        if s["with"]["name"].startswith("ci-fleet-ledger-")
    ]
    assert spool["with"]["name"] == (
        "ci-fleet-ledger-${{ github.job }}-${{ strategy.job-index }}-${{ github.run_attempt }}"
    )
    assert spool["with"]["path"] == "${{ runner.temp }}/ci-fleet-ledger/"
    assert spool["if"] == "always()" and spool["continue-on-error"] is True
    assert spool["with"]["retention-days"] == 3
    assert spool["with"]["if-no-files-found"] == "ignore"


def test_only_the_merged_map_carries_the_map_artifact_name() -> None:
    # The ledger routes artifacts by name prefix.
    names = [
        s["with"]["name"] for job in JOBS.values() for s in _uses(job, "actions/upload-artifact@")
    ]
    assert [n for n in names if n.startswith(map_import.ARTIFACT_NAME)] == [
        map_import.ARTIFACT_NAME
    ]


def _complete(tmp_path: Path, splits: str, present: list[int]):
    maps = tmp_path / "maps"
    maps.mkdir()
    for k in present:
        (maps / f"map-shard-{k}.json.gz").write_bytes(b"x")
    return run_step(COMPLETE, tmp_path, {"SPLITS": splits, "MAPS_DIR": maps.as_posix()})


@pytest.mark.parametrize(
    ("splits", "present", "complete"),
    [
        ("3", [1, 2, 3], "true"),
        ("3", [1, 3], "false"),
        ("0", [], "false"),
        ("", [], "false"),
        ("x", [1], "false"),
    ],
)
def test_the_map_merges_only_when_every_shard_mapped(
    tmp_path: Path, splits: str, present: list[int], complete: str
) -> None:
    result = _complete(tmp_path, splits, present)
    assert result.returncode == 0, result.stderr
    assert result.outputs == {"complete": complete}


def test_the_map_job_merges_and_uploads_the_map() -> None:
    job = JOBS["map"]
    assert set(job["needs"]) == {"coverage", "tests-shard"}
    assert "always()" in job["if"] and "github.event_name == 'schedule'" in job["if"]
    assert "map" not in JOBS["Tests"]["needs"]  # never part of the required check
    [download] = _uses(job, "actions/download-artifact@")
    assert download["with"]["pattern"] == "map-shard-*"
    assert download["with"]["path"] == COMPLETE["env"]["MAPS_DIR"]
    assert COMPLETE["env"]["SPLITS"] == "${{ needs.coverage.outputs.splits }}"
    merge = _step(job, "Merge")
    assert merge["if"] == "steps.complete.outputs.complete == 'true'"
    assert merge["run"].startswith('uvx --from "ci-fleet~=0.7.0" ci-fleet map build --merge ')
    assert f"--out {map_import.MAP_FILE_NAME}" in merge["run"]
    [upload] = _uses(job, "actions/upload-artifact@")
    assert upload["if"] == merge["if"]
    assert upload["with"]["name"] == map_import.ARTIFACT_NAME
    assert upload["with"]["path"] == map_import.MAP_FILE_NAME
    assert upload["with"]["retention-days"] == 30


# --- the bisect workflow -------------------------------------------------------------------


class _Gh:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def run(self, args, *, json_output=False, allow_failure=False):
        self.calls.append(list(args))
        return None


def test_bisect_inputs_are_exactly_what_triage_dispatches() -> None:
    gh = _Gh()
    brk = triage.Break(
        break_id=7,
        repo="owner/repo-a",
        nightly_run_id="n",
        nightly_sha="b" * 40,
        good_sha="a" * 40,
        classification=None,
        culprit_sha=None,
        culprit_pr=None,
        issue_ref=None,
        revert_pr=None,
        attempts=0,
        state="open",
        opened_at="2026-10-04T00:00:00.000000Z",
    )
    triage.GitHubBisector(gh).dispatch(brk, ["tests/test_x.py::test_y"], None)
    [call] = [c for c in gh.calls if c[:2] == ["workflow", "run"]]
    assert call[2] == triage.BISECT_WORKFLOW
    sent = {call[i + 1].split("=", 1)[0] for i, a in enumerate(call) if a == "-f"}
    inputs = _triggers(BISECT)["workflow_dispatch"]["inputs"]
    assert set(inputs) == sent
    assert {k for k, v in inputs.items() if v.get("required")} == sent - {"seed"}
    assert set(_triggers(BISECT)) == {"workflow_dispatch"}  # never a second nightly


def test_bisect_run_name_and_artifact_match_the_poller() -> None:
    assert BISECT["run-name"].replace("${{ inputs.break_id }}", "7") == triage.run_name(7)
    [job] = BISECT["jobs"].values()
    [upload] = _uses(job, "actions/upload-artifact@")
    assert upload["with"]["name"].replace("${{ inputs.break_id }}", "7") == "ci-fleet-bisect-7"
    assert upload["if"] == "always()"
    out = upload["with"]["path"].rstrip("/")
    run = _step(job, "Bisect")["run"]
    assert f"--out {out}" in run and 'uvx --from "ci-fleet~=0.7.0"' in run
    assert job["env"][bisect.LEDGER_ENV] == "off"
    assert job["env"]["UV_PROJECT_ENVIRONMENT"] == ".venv"


def test_bisect_inputs_never_reach_the_shell_directly() -> None:
    [job] = BISECT["jobs"].values()
    step = _step(job, "Bisect")
    assert "${{" not in step["run"]
    assert set(step["env"]) == {"GOOD", "BAD", "NODEIDS", "SEED"}


def test_bisect_runs_and_installs_where_the_suite_does() -> None:
    [job] = BISECT["jobs"].values()
    shard = JOBS["tests-shard"]
    assert job["runs-on"] == shard["runs-on"]
    assert job["timeout-minutes"] * 60 > bisect.BISECT_TIMEOUT_S
    [checkout] = _uses(job, "actions/checkout@")
    assert checkout["with"]["fetch-depth"] == 0
    assert [s["uses"] for s in _uses(job, "astral-sh/setup-uv@")] == [
        s["uses"] for s in _uses(shard, "astral-sh/setup-uv@")
    ]
    # Each bisect step syncs exactly as the shards do: pytest is in an extra.
    install = _step(shard, "Install dependencies")["run"]
    assert f'--sync "{install}"' in _step(job, "Bisect")["run"]


def test_ci_fleet_toml_is_valid_and_on() -> None:
    assert load_config((REPO / "ci-fleet.toml").read_text(encoding="utf-8")).enabled
