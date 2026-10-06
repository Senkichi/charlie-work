"""Structure of the sharded ``Tests`` check in ci.yml (DD-4).

Required check names are read from ``.aviator/config.yml``, never copied, so
a rename on either side fails here first.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from _ci_step_runner import CI_YML, REPO, find_step, load_workflow, run_step

WF = load_workflow()
JOBS = WF["jobs"]


def _triggers() -> dict:
    # PyYAML parses the bare key `on` as boolean True.
    return WF.get("on", WF.get(True))


def _required_checks() -> list[str]:
    aviator = yaml.safe_load((REPO / ".aviator" / "config.yml").read_text(encoding="utf-8"))
    return list(aviator["merge_rules"]["preconditions"]["required_checks"])


def _job_names() -> set[str]:
    return {job.get("name", job_id) for job_id, job in JOBS.items()}


def test_every_required_check_is_a_job_name() -> None:
    missing = set(_required_checks()) - _job_names()
    assert not missing, f"required checks with no job of that name: {sorted(missing)}"


def test_tests_aggregate_shape() -> None:
    tests = JOBS["Tests"]
    assert tests["name"] == "Tests"
    assert tests["needs"] == ["coverage", "collect-only-gate", "tests-shard"]
    assert tests["if"] == "always()"
    assert tests["runs-on"] == "ubuntu-latest"
    assert tests["permissions"] == {"contents": "read", "actions": "write"}
    assert tests.get("concurrency") is None


def test_shard_job_shape() -> None:
    shard = JOBS["tests-shard"]
    assert shard["name"].startswith("Tests shard ")
    assert shard["needs"] == "coverage"
    assert shard["runs-on"] == "windows-latest"
    assert shard["strategy"]["fail-fast"] is False
    assert shard["strategy"]["matrix"]["group"] == "${{ fromJSON(needs.coverage.outputs.shards) }}"
    assert shard["timeout-minutes"] == "${{ fromJSON(needs.coverage.outputs.shard_timeout) }}"
    run = find_step(WF, "tests-shard", "Run test shard")
    assert run["shell"] == "bash"
    assert "--splitting-algorithm duration_based_chunks" in run["run"]
    assert "${{" not in run["run"]


def test_no_shard_runs_its_own_collection() -> None:
    names = [s.get("name") for s in JOBS["tests-shard"]["steps"]]
    assert "Collect tests" not in names


def test_every_shard_uploads_the_ledger_spool() -> None:
    # LG-C1a's step, moved from the old Tests job; ledger S1 artifact name.
    names = [s.get("name") for s in JOBS["Tests"].get("steps", [])]
    assert "Upload test-ledger spool" not in names
    step = find_step(WF, "tests-shard", "Upload test-ledger spool")
    assert step["with"]["name"] == (
        "ci-fleet-ledger-${{ github.job }}-${{ strategy.job-index }}-${{ github.run_attempt }}"
    )
    assert step["with"]["path"] == "${{ runner.temp }}/ci-fleet-ledger/"
    assert step["if"] == "always()" and step["continue-on-error"] is True


def test_nightly_schedule_and_dispatch() -> None:
    triggers = _triggers()
    assert triggers["schedule"] == [{"cron": "0 9 * * *"}]
    assert "workflow_dispatch" in triggers


def test_schedule_runs_have_their_own_concurrency_group() -> None:
    group = WF["concurrency"]["group"]
    assert "${{ github.ref }}" in group
    assert "github.event_name == 'schedule'" in group


def test_collect_gate_uploads_head_collection_on_every_event() -> None:
    gate = JOBS["collect-only-gate"]
    assert "github.event_name == 'pull_request'" not in str(gate.get("if", ""))
    upload = find_step(WF, "collect-only-gate", "Upload head collection")
    assert upload["if"] == "always()"
    assert upload["with"]["name"] == "head-collect"
    for name in ("Collect tests at base", "Run collect-only gate"):
        assert (
            find_step(WF, "collect-only-gate", name)["if"] == "github.event_name == 'pull_request'"
        )


def test_durations_cache_is_cross_os() -> None:
    restore = find_step(WF, "tests-shard", "Restore test durations")
    assert restore["with"]["enableCrossOsArchive"] is True
    save = find_step(WF, "Tests", "Save test durations")
    assert save["with"]["enableCrossOsArchive"] is True
    assert save["if"].startswith("github.event_name == 'schedule'")


@pytest.mark.parametrize(
    ("var", "splits", "timeout"),
    [
        ("", "6", "15"),
        ("1", "1", "30"),
        ("2", "2", "30"),
        ("3", "3", "15"),
        ("abc", "6", "15"),
        ("0", "1", "30"),
        ("50", "20", "15"),
    ],
)
def test_plan_shards_step(tmp_path: Path, var: str, splits: str, timeout: str) -> None:
    step = find_step(WF, "coverage", "Plan shards")
    assert "${{" not in step["run"]
    result = run_step(step, tmp_path, {"SHARDS_VAR": var})
    assert result.returncode == 0, result.stderr
    assert result.outputs["splits"] == splits
    assert result.outputs["shard_timeout"] == timeout
    assert (
        result.outputs["shards"] == "[" + ",".join(str(i) for i in range(1, int(splits) + 1)) + "]"
    )


def test_ci_yml_path_is_the_repo_workflow() -> None:
    assert CI_YML == REPO / ".github" / "workflows" / "ci.yml"
