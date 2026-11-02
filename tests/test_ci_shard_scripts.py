"""CI helper scripts for the sharded ``Tests`` aggregate (DD-4)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from _script_loader import load_script_module

REPO = Path(__file__).resolve().parents[1]
verdict = load_script_module(REPO / "scripts" / "ci_tests_verdict.py", "ci_tests_verdict")
annotations = load_script_module(
    REPO / "scripts" / "ci_junit_annotations.py", "ci_junit_annotations"
)
durations = load_script_module(
    REPO / "scripts" / "merge_test_durations.py", "merge_test_durations"
)


def _shards(*conclusions: str | None) -> list[tuple[str, str | None]]:
    n = len(conclusions)
    return [(f"Tests shard {i}/{n}", c) for i, c in enumerate(conclusions, start=1)] + [
        ("Lint", "failure"),  # non-shard jobs are ignored
    ]


def _decide(jobs, **kw):
    base = {
        "shard_result": "success",
        "gate_result": "success",
        "coverage_result": "success",
        "covered": False,
        "splits": 3,
    }
    base.update(kw)
    return verdict.decide(jobs, **base)


# --------------------------------------------------------------------------- verdict


def test_all_shards_success_passes() -> None:
    assert _decide(_shards("success", "success", "success")).verdict == "pass"


def test_covered_push_with_skipped_shards_passes() -> None:
    result = _decide([], shard_result="skipped", gate_result="skipped", covered=True)
    assert result.verdict == "pass"
    assert "covered" in result.reason


def test_skipped_shards_without_coverage_fail() -> None:
    assert (
        _decide(_shards("skipped", "skipped", "skipped"), shard_result="skipped").verdict == "fail"
    )


def test_failed_shard_fails_even_with_a_cancelled_sibling() -> None:
    result = _decide(_shards("failure", "cancelled", "success"), shard_result="failure")
    assert result.verdict == "fail"
    assert "Tests shard 1/3" in result.reason


@pytest.mark.parametrize("conclusion", ["cancelled", "timed_out", None])
def test_infra_fault_cancels(conclusion: str | None) -> None:
    result = _decide(_shards("success", conclusion, "success"), shard_result="cancelled")
    assert result.verdict == "cancel"


def test_missing_shard_fails() -> None:
    assert _decide(_shards("success", "success")).verdict == "fail"


def test_cancelled_gate_cancels() -> None:
    assert (
        _decide(_shards("success", "success", "success"), gate_result="cancelled").verdict
        == "cancel"
    )


def test_failed_gate_does_not_fail_tests() -> None:
    # The gate is its own required check; Tests only needs its uploaded collection.
    assert (
        _decide(_shards("success", "success", "success"), gate_result="failure").verdict == "pass"
    )


@pytest.mark.parametrize(("result", "expected"), [("failure", "fail"), ("cancelled", "cancel")])
def test_coverage_job_problems(result: str, expected: str) -> None:
    assert _decide([], coverage_result=result, shard_result="skipped").verdict == expected


def test_verdict_cli_reads_jobs_file(tmp_path: Path, capsys) -> None:
    jobs = tmp_path / "jobs.jsonl"
    jobs.write_text(
        "\n".join(json.dumps([n, c]) for n, c in _shards("success", "success", "success")) + "\n",
        encoding="utf-8",
    )
    code = verdict.main(
        [
            "--jobs",
            str(jobs),
            "--shard-result",
            "success",
            "--gate-result",
            "success",
            "--coverage-result",
            "success",
            "--covered",
            "false",
            "--splits",
            "3",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "verdict=pass"
    assert out[1].startswith("reason=")


def test_verdict_cli_unreadable_jobs_fails_closed(tmp_path: Path, capsys) -> None:
    jobs = tmp_path / "jobs.jsonl"
    jobs.write_text("not json\n", encoding="utf-8")
    verdict.main(
        [
            "--jobs",
            str(jobs),
            "--shard-result",
            "success",
            "--gate-result",
            "success",
            "--coverage-result",
            "success",
            "--covered",
            "false",
            "--splits",
            "3",
        ]
    )
    assert capsys.readouterr().out.splitlines()[0] == "verdict=fail"


# --------------------------------------------------------------------------- annotations


def _failing_junit(n: int, classname: str = "tests.test_x.TestA") -> str:
    cases = "".join(
        f'<testcase classname="{classname}" name="test_{i}">'
        f'<failure message="assert {i} == 0&#10;detail">trace</failure></testcase>'
        for i in range(n)
    )
    ok = f'<testcase classname="{classname}" name="test_ok" />'
    return f"<testsuites><testsuite>{cases}{ok}</testsuite></testsuites>"


def test_failures_resolve_classname_to_file(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.py").write_text("", encoding="utf-8")
    found = annotations.collect_failures(_failing_junit(1), tmp_path)
    assert len(found) == 1
    assert found[0].file == "tests/test_x.py"
    assert found[0].nodeid == "tests/test_x.py::TestA::test_0"
    lines = annotations.render(found)
    assert lines == [
        "::error file=tests/test_x.py,title=tests/test_x.py%3A%3ATestA%3A%3Atest_0::assert 0 == 0%0Adetail"
    ]


def test_render_caps_at_fifty(tmp_path: Path) -> None:
    found = annotations.collect_failures(_failing_junit(60), tmp_path)
    lines = annotations.render(found)
    assert sum(1 for line in lines if line.startswith("::error ")) == 50
    assert (
        lines[-1]
        == "::notice title=More failures::10 more failed tests are listed in the step summary"
    )


def test_annotations_cli_writes_summary_and_exits_zero(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    junit = tmp_path / "pytest-junit-1.xml"
    junit.write_text(_failing_junit(2), encoding="utf-8")
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    assert (
        annotations.main(["--root", str(tmp_path), str(junit), str(tmp_path / "missing.xml")]) == 0
    )
    assert capsys.readouterr().out.count("::error ") == 2
    assert "2 failed test(s)" in summary.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- durations


def test_merge_keeps_fresh_values_from_every_shard() -> None:
    baseline = {"a": 1.0, "b": 2.0}
    shard1 = {"a": 1.5, "b": 2.0}  # re-timed a
    shard2 = {"a": 1.0, "b": 2.5, "c": 3.0}  # re-timed b, new c
    assert durations.merge(baseline, [shard1, shard2]) == {"a": 1.5, "b": 2.5, "c": 3.0}


def test_merge_without_baseline_is_a_union() -> None:
    assert durations.merge({}, [{"a": 1.0}, {"b": 2.0}]) == {"a": 1.0, "b": 2.0}


def test_merge_cli_writes_out_and_rejects_bad_shard(tmp_path: Path) -> None:
    base = tmp_path / ".test_durations"
    base.write_text(json.dumps({"a": 1.0}), encoding="utf-8")
    s1 = tmp_path / "s1"
    s1.write_text(json.dumps({"a": 1.0, "b": 2.0}), encoding="utf-8")
    assert durations.main(["--baseline", str(base), "--out", str(base), str(s1)]) == 0
    assert json.loads(base.read_text(encoding="utf-8")) == {"a": 1.0, "b": 2.0}
    bad = tmp_path / "bad"
    bad.write_text("[1, 2]", encoding="utf-8")
    assert durations.main(["--baseline", str(base), "--out", str(base), str(bad)]) == 1
    assert json.loads(base.read_text(encoding="utf-8")) == {"a": 1.0, "b": 2.0}


def test_merge_cli_missing_baseline_is_empty(tmp_path: Path) -> None:
    s1 = tmp_path / "s1"
    s1.write_text(json.dumps({"a": 1.0}), encoding="utf-8")
    out = tmp_path / "out.json"
    assert durations.main(["--baseline", str(tmp_path / "nope"), "--out", str(out), str(s1)]) == 0
    assert json.loads(out.read_text(encoding="utf-8")) == {"a": 1.0}
