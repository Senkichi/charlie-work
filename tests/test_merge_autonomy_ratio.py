"""Tests for scripts/merge_autonomy_ratio.py.

No test here touches the network: `compute_autonomy_stats`, `filter_since`,
and `build_repo_report` are pure functions over injected PR dicts, and the
gh-fetch guard tests below monkeypatch `subprocess.run` with a fake instead
of shelling out.

Loaded via ``_script_loader.load_script_module`` (scripts/ is not on
sys.path), which registers the module in ``sys.modules`` *before*
``exec_module`` runs -- required because the script pairs
``from __future__ import annotations`` with ``@dataclass(frozen=True)``
(``AutonomyStats``, ``RepoReport``), and dataclass creation resolves those
string annotations through ``sys.modules[cls.__module__]``. Skipping the
registration step reproduces #1023 (see tests/test_verify_events.py's
``test_loader_registers_module_before_exec_module`` for the regression this
mirrors) with an ``AttributeError`` at class-creation time, not a test-body
failure -- so this is load-bearing, not incidental.

This file originally hand-rolled that recipe, which is exactly the
duplication #1023's fix removed; the guard in
``tests/test_script_loader.py::test_no_hand_rolled_spec_from_file_location_in_tests``
caught it in CI. The shared helper is also strictly better than what was here:
it restores the prior ``sys.modules`` entry in a ``finally``, where the local
version leaked the test module into the interpreter for the rest of the run.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from _script_loader import load_script_module

_SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "merge_autonomy_ratio.py"


def _load_module() -> ModuleType:
    return load_script_module(_SCRIPT_PATH, "merge_autonomy_ratio_under_test")


@pytest.fixture(scope="module")
def mar() -> ModuleType:
    return _load_module()


def _pr(number: int, merged_at: str, login: str | None) -> dict[str, Any]:
    """Build one gh-shaped PR dict. login=None models an absent mergedBy.login;
    a wholly-absent mergedBy key is modeled separately in its own test.
    """
    return {
        "number": number,
        "mergedAt": merged_at,
        "mergedBy": {"login": login} if login is not None else None,
    }


NOW = datetime(2026, 8, 5, 12, 0, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------------------
# compute_autonomy_stats: the counting logic, in isolation.
# --------------------------------------------------------------------------


def test_all_aviator_window(mar: ModuleType) -> None:
    prs = [_pr(1, "2026-08-01T00:00:00Z", "app/aviator-app") for _ in range(3)]
    stats = mar.compute_autonomy_stats(prs)
    assert stats.total == 3
    assert stats.autonomous == 3
    assert stats.human_counts == {}
    assert stats.unknown == 0
    assert stats.ratio == pytest.approx(1.0)


def test_mixed_window(mar: ModuleType) -> None:
    prs = [
        _pr(1, "2026-08-01T00:00:00Z", "app/aviator-app"),
        _pr(2, "2026-08-01T00:00:00Z", "app/aviator-app"),
        _pr(3, "2026-08-01T00:00:00Z", "app/aviator-app"),
        _pr(4, "2026-08-01T00:00:00Z", "Senkichi"),
        _pr(5, "2026-08-01T00:00:00Z", None),
    ]
    stats = mar.compute_autonomy_stats(prs)
    assert stats.total == 5
    assert stats.autonomous == 3
    assert stats.human_counts == {"Senkichi": 1}
    assert stats.unknown == 1
    assert stats.ratio == pytest.approx(3 / 5)


def test_human_only_window(mar: ModuleType) -> None:
    prs = [
        _pr(1, "2026-08-01T00:00:00Z", "Senkichi"),
        _pr(2, "2026-08-01T00:00:00Z", "Senkichi"),
        _pr(3, "2026-08-01T00:00:00Z", "alice"),
    ]
    stats = mar.compute_autonomy_stats(prs)
    assert stats.total == 3
    assert stats.autonomous == 0
    assert stats.human_counts == {"Senkichi": 2, "alice": 1}
    assert stats.human_total == 3
    assert stats.unknown == 0
    # Zero autonomous merges in a NON-empty window is a real 0.0, not undefined.
    assert stats.ratio == pytest.approx(0.0)


def test_null_mergedby_bucket_never_folded_into_either_side(mar: ModuleType) -> None:
    """A null mergedBy AND a mergedBy with a null/absent login must both land
    in `unknown`, and must never move `autonomous` or `human_counts`.
    """
    prs = [
        _pr(1, "2026-08-01T00:00:00Z", "app/aviator-app"),
        {"number": 2, "mergedAt": "2026-08-01T00:00:00Z", "mergedBy": None},
        {"number": 3, "mergedAt": "2026-08-01T00:00:00Z"},  # mergedBy key wholly absent
        {"number": 4, "mergedAt": "2026-08-01T00:00:00Z", "mergedBy": {}},  # login absent
    ]
    stats = mar.compute_autonomy_stats(prs)
    assert stats.total == 4
    assert stats.autonomous == 1
    assert stats.human_counts == {}
    assert stats.unknown == 3


def test_empty_list_ratio_is_none_not_zero(mar: ModuleType) -> None:
    stats = mar.compute_autonomy_stats([])
    assert stats.total == 0
    assert stats.autonomous == 0
    assert stats.unknown == 0
    assert stats.ratio is None


# --------------------------------------------------------------------------
# filter_since: the window boundary, applied in Python.
# --------------------------------------------------------------------------


def test_filter_since_keeps_inclusive_boundary_and_drops_before(mar: ModuleType) -> None:
    since = NOW - timedelta(days=1)
    prs = [
        _pr(
            1, (since - timedelta(seconds=1)).isoformat().replace("+00:00", "Z"), "app/aviator-app"
        ),
        _pr(2, since.isoformat().replace("+00:00", "Z"), "app/aviator-app"),  # exactly at boundary
        _pr(3, (since + timedelta(hours=1)).isoformat().replace("+00:00", "Z"), "app/aviator-app"),
    ]
    kept = mar.filter_since(prs, since)
    assert [p["number"] for p in kept] == [2, 3]


def test_filter_since_drops_missing_or_unparseable_merged_at(mar: ModuleType) -> None:
    since = NOW - timedelta(days=7)
    prs = [
        {"number": 1, "mergedAt": None, "mergedBy": {"login": "app/aviator-app"}},
        {"number": 2, "mergedBy": {"login": "app/aviator-app"}},  # key absent
        {"number": 3, "mergedAt": "not-a-timestamp", "mergedBy": {"login": "app/aviator-app"}},
        _pr(4, NOW.isoformat().replace("+00:00", "Z"), "app/aviator-app"),
    ]
    kept = mar.filter_since(prs, since)
    assert [p["number"] for p in kept] == [4]


# --------------------------------------------------------------------------
# build_repo_report / RepoReport: the empty-window guard, end to end.
# --------------------------------------------------------------------------


def test_build_repo_report_empty_window_is_undefined_not_zero(mar: ModuleType) -> None:
    since = NOW  # window start after every PR below -> filtered set is empty
    raw = [_pr(1, "2020-01-01T00:00:00Z", "app/aviator-app")]  # non-empty raw fetch

    report = mar.build_repo_report("Senkichi/charlie-work", since, raw, limit=200)

    assert report.stats.total == 0
    assert report.stats.ratio is None
    payload = report.to_json_dict()
    assert payload["total"] == 0
    assert payload["ratio"] is None
    assert payload["note"] == "no merges in window - ratio undefined"


def test_build_repo_report_nonempty_window_has_no_note(mar: ModuleType) -> None:
    since = NOW - timedelta(days=1)
    raw = [
        _pr(1, NOW.isoformat().replace("+00:00", "Z"), "app/aviator-app"),
        _pr(2, NOW.isoformat().replace("+00:00", "Z"), "Senkichi"),
    ]

    report = mar.build_repo_report("Senkichi/charlie-work", since, raw, limit=200)

    payload = report.to_json_dict()
    assert payload["total"] == 2
    assert payload["autonomous"] == 1
    assert payload["ratio"] == pytest.approx(0.5)
    assert payload["note"] is None


def test_build_repo_report_truncation_flag(mar: ModuleType) -> None:
    since = NOW - timedelta(days=1)
    raw = [_pr(i, NOW.isoformat().replace("+00:00", "Z"), "app/aviator-app") for i in range(2)]

    not_truncated = mar.build_repo_report("Senkichi/charlie-work", since, raw, limit=200)
    truncated = mar.build_repo_report("Senkichi/charlie-work", since, raw, limit=2)

    assert not_truncated.truncated is False
    assert truncated.truncated is True


# --------------------------------------------------------------------------
# fetch_merged_prs: broken-query guards. subprocess.run is monkeypatched
# with a fake CompletedProcess -- no network, no real gh invocation.
# --------------------------------------------------------------------------


class _FakeCompletedProcess:
    def __init__(self, returncode: int, stdout: str, stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_fetch_merged_prs_raises_on_empty_result(
    mar: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        mar.subprocess, "run", lambda *_args, **_kwargs: _FakeCompletedProcess(0, "[]")
    )
    with pytest.raises(SystemExit, match="zero merged PRs"):
        mar.fetch_merged_prs("Senkichi/charlie-work", limit=200, timeout=120)


def test_fetch_merged_prs_raises_on_nonzero_exit(
    mar: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        mar.subprocess,
        "run",
        lambda *_args, **_kwargs: _FakeCompletedProcess(1, "", "authentication required"),
    )
    with pytest.raises(SystemExit, match="authentication required"):
        mar.fetch_merged_prs("Senkichi/charlie-work", limit=200, timeout=120)


def test_fetch_merged_prs_raises_on_unparseable_json(
    mar: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        mar.subprocess, "run", lambda *_args, **_kwargs: _FakeCompletedProcess(0, "not json")
    )
    with pytest.raises(SystemExit, match="unparseable JSON"):
        mar.fetch_merged_prs("Senkichi/charlie-work", limit=200, timeout=120)


def test_fetch_merged_prs_returns_data_on_success(
    mar: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = '[{"number": 1, "mergedAt": "2026-08-01T00:00:00Z", "mergedBy": {"login": "app/aviator-app"}}]'
    monkeypatch.setattr(
        mar.subprocess, "run", lambda *_args, **_kwargs: _FakeCompletedProcess(0, payload)
    )
    data = mar.fetch_merged_prs("Senkichi/charlie-work", limit=200, timeout=120)
    assert data == [
        {"number": 1, "mergedAt": "2026-08-01T00:00:00Z", "mergedBy": {"login": "app/aviator-app"}}
    ]


def test_fetch_merged_prs_raises_on_timeout(
    mar: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _raise_timeout(*_args: Any, **_kwargs: Any) -> Any:
        raise subprocess.TimeoutExpired(cmd=["gh"], timeout=5)

    monkeypatch.setattr(mar.subprocess, "run", _raise_timeout)
    with pytest.raises(SystemExit, match="timed out"):
        mar.fetch_merged_prs("Senkichi/charlie-work", limit=200, timeout=5)


# --------------------------------------------------------------------------
# fetch_merged_prs_in_window: the server-side windowed fetch, plus the
# repo-probe that closes the "gh pr list --search returns [] with exit 0
# for an unresolvable repo" gap. A `_RecordingFakeRun` scripts one response
# per subprocess.run call (in order: probe, then the windowed search) and
# records every invocation so tests can assert on call COUNT (proving the
# probe short-circuits a bad repo before the search ever runs) and on the
# constructed command (proving the `merged:>=` qualifier carries the exact
# `since` value).
# --------------------------------------------------------------------------


class _RecordingFakeRun:
    def __init__(self, responses: list[_FakeCompletedProcess]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> _FakeCompletedProcess:
        self.calls.append((args, kwargs))
        if not self._responses:
            raise AssertionError("subprocess.run called more times than scripted")
        return self._responses.pop(0)


def test_fetch_merged_prs_in_window_probe_failure_short_circuits_before_search(
    mar: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A repo `gh` cannot resolve must be caught by the probe (the first
    call, reusing `fetch_merged_prs`'s own loud nonzero-exit guard) -- the
    windowed search call must never run. If it did, and `gh pr list
    --search` behaved as it does empirically (see the next test's premise),
    a bad repo would silently report "no merges in window" instead of
    failing.
    """
    fake = _RecordingFakeRun([_FakeCompletedProcess(1, "", "Could not resolve to a Repository")])
    monkeypatch.setattr(mar.subprocess, "run", fake)

    with pytest.raises(SystemExit, match="Could not resolve to a Repository"):
        mar.fetch_merged_prs_in_window(
            "Senkichi/does-not-exist", NOW - timedelta(days=7), limit=200, timeout=120
        )

    assert len(fake.calls) == 1  # search call never reached


def test_fetch_merged_prs_in_window_trusts_empty_result_once_probe_passes(
    mar: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression coverage for the exact gap this fetch path exists to
    close: `gh pr list --search "merged:>=..."` returns `[]` with exit 0
    for a genuinely empty window, indistinguishable on its own from the
    exit-0-empty-list a bad repo also produces (verified against the real
    `gh` CLI, not assumed). Once the probe (first call) has proven the repo
    is real, that `[]` must be returned as a real empty window, not raised.
    """
    probe_ok = _FakeCompletedProcess(0, '[{"number": 1}]')
    windowed_empty = _FakeCompletedProcess(0, "[]")
    fake = _RecordingFakeRun([probe_ok, windowed_empty])
    monkeypatch.setattr(mar.subprocess, "run", fake)

    data = mar.fetch_merged_prs_in_window(
        "Senkichi/charlie-work", NOW - timedelta(days=7), limit=200, timeout=120
    )

    assert data == []
    assert len(fake.calls) == 2


def test_fetch_merged_prs_in_window_raises_when_search_call_fails(
    mar: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The windowed call keeps its own failure guard even after the probe
    passes -- a transient failure on the SECOND call (e.g. a rate limit hit
    between the probe and the real query) must still fail loudly.
    """
    probe_ok = _FakeCompletedProcess(0, '[{"number": 1}]')
    search_fail = _FakeCompletedProcess(1, "", "rate limit exceeded")
    fake = _RecordingFakeRun([probe_ok, search_fail])
    monkeypatch.setattr(mar.subprocess, "run", fake)

    with pytest.raises(SystemExit, match="rate limit exceeded"):
        mar.fetch_merged_prs_in_window(
            "Senkichi/charlie-work", NOW - timedelta(days=7), limit=200, timeout=120
        )


def test_fetch_merged_prs_in_window_search_query_carries_exact_since(
    mar: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe_ok = _FakeCompletedProcess(0, '[{"number": 1}]')
    windowed_ok = _FakeCompletedProcess(0, "[]")
    fake = _RecordingFakeRun([probe_ok, windowed_ok])
    monkeypatch.setattr(mar.subprocess, "run", fake)
    since = datetime(2026, 7, 30, 3, 30, 0, tzinfo=timezone.utc)

    mar.fetch_merged_prs_in_window("Senkichi/charlie-work", since, limit=200, timeout=120)

    windowed_command = fake.calls[1][0][0]
    assert "--search" in windowed_command
    search_value = windowed_command[windowed_command.index("--search") + 1]
    assert search_value == "merged:>=2026-07-30T03:30:00Z"


def test_fetch_merged_prs_in_window_cap_hit_still_flags_truncated(
    mar: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Server-side windowing narrows the result set; it does not prove the
    set is complete. A windowed result whose length equals --limit must
    still trip the existing truncation guard once piped through
    `build_repo_report`.
    """
    probe_ok = _FakeCompletedProcess(0, '[{"number": 1}]')
    merged_at = NOW.isoformat().replace("+00:00", "Z")  # inside the window below
    full_page_prs = ",".join(
        f'{{"number": {i}, "mergedAt": "{merged_at}", "mergedBy": {{"login": "app/aviator-app"}}}}'
        for i in range(3)
    )
    windowed_full = _FakeCompletedProcess(0, f"[{full_page_prs}]")
    fake = _RecordingFakeRun([probe_ok, windowed_full])
    monkeypatch.setattr(mar.subprocess, "run", fake)
    since = NOW - timedelta(days=1)

    data = mar.fetch_merged_prs_in_window("Senkichi/charlie-work", since, limit=3, timeout=120)
    report = mar.build_repo_report("Senkichi/charlie-work", since, data, limit=3)

    assert len(data) == 3
    assert report.truncated is True
    assert report.stats.total == 3


# --------------------------------------------------------------------------
# _print_human: the empty-window case must never render "0.000" anywhere.
# --------------------------------------------------------------------------


def test_print_human_empty_window_says_undefined_not_zero(
    mar: ModuleType, capsys: pytest.CaptureFixture
) -> None:
    since = NOW
    raw = [_pr(1, "2020-01-01T00:00:00Z", "app/aviator-app")]
    report = mar.build_repo_report("Senkichi/charlie-work", since, raw, limit=200)

    mar._print_human([report])

    out = capsys.readouterr().out
    assert "no merges in window - ratio undefined" in out
    assert "0.000" not in out
    assert "ratio: 0" not in out
