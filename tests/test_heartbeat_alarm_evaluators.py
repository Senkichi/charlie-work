"""Exact-value tests for the pure alarm evaluators (``charlie_work.heartbeat_alarms``
and ``heartbeat_alarms_fleet``): ``now``/``baseline`` injected, boundaries at the
thresholds, and a real-writer round trip (``log_event`` -> events.db rows ->
``eval_*``) so the row shape the readers pass is the shape the writer produces.

File name note: deliberately not test_heartbeat_alarms.py -- that basename would make
test_dormant_fleet_marking demand a rollback_path marker, but the leaf is not a
rollback island: scripts/heartbeat_*.py consume it (guarded import) and the
dashboard will import it from src/.
"""

from __future__ import annotations

import ast
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from charlie_work import heartbeat_alarms as ha
from charlie_work import heartbeat_alarms_fleet as haf
from charlie_work.instrumentation import log_event
from charlie_work.supervisor_lifecycle import write_supervisor_heartbeat

NOW = datetime(2026, 10, 1, 12, 0, 0, tzinfo=timezone.utc)
BASE = datetime(2026, 10, 1, 11, 0, 0, tzinfo=timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


class _Report:
    def __init__(self) -> None:
        self.lines: list[tuple[str, str, str]] = []

    def ok(self, check: str, facts: str) -> None:
        self.lines.append(("ok", check, facts))

    def warn(self, check: str, detail: str) -> None:
        self.lines.append(("warn", check, detail))

    def anom(self, check: str, detail: str) -> None:
        self.lines.append(("anomaly", check, detail))


# --- parse_iso / new_event_rows / Finding / emit -----------------------------


def test_parse_iso_handles_z_none_and_garbage() -> None:
    assert ha.parse_iso("2026-10-01T12:00:00Z") == NOW
    assert ha.parse_iso(None) is None
    assert ha.parse_iso("") is None
    assert ha.parse_iso("not a date") is None


def test_new_event_rows_is_strictly_after_baseline_and_keeps_unparseable() -> None:
    rows = [
        (_iso(BASE), "at"),
        (_iso(BASE + timedelta(seconds=1)), "after"),
        (_iso(BASE - timedelta(seconds=1)), "before"),
        ("garbage", "bad"),
    ]
    assert ha.new_event_rows(rows, BASE) == [rows[1], rows[3]]


def test_finding_is_frozen() -> None:
    f = ha.Finding("c", None, "ok", "", "x")
    with pytest.raises(AttributeError):
        f.detail = "y"  # type: ignore[misc]


def test_emit_maps_severity_to_report_lines() -> None:
    rep = _Report()
    ha.emit(rep, ha.Finding("a", "r", "ok", "", "facts"))
    ha.emit(rep, ha.Finding("b", "r", "warn", "w"))
    ha.emit(rep, ha.Finding("c", None, "anomaly", "z"))
    assert rep.lines == [("ok", "a", "facts"), ("warn", "b", "w"), ("anomaly", "c", "z")]


# --- event evaluators ---------------------------------------------------------


def test_error_events_ok_when_nothing_new() -> None:
    rows = [(_iso(BASE), "self_deploy_alarm")]  # exactly at baseline -> not new
    assert ha.eval_error_events("o/r", rows, BASE) == ha.Finding(
        "error-events o/r", "o/r", "ok", "", "error_rows=1 new_since_last_beat=0"
    )


def test_error_events_anomaly_lists_new_rows_and_unparseable() -> None:
    new_ts = _iso(BASE + timedelta(minutes=5))
    rows = [(_iso(BASE), "old"), (new_ts, "boom"), ("junk", "weird")]
    f = ha.eval_error_events("o/r", rows, BASE)
    facts = "error_rows=3 new_since_last_beat=2"
    assert f == ha.Finding(
        "error-events o/r",
        "o/r",
        "anomaly",
        f"new error-level event(s) since last beat: ['boom@{new_ts}', 'weird@junk'] ({facts})",
        facts,
    )


def test_warning_events_buckets_expected_ops_sorted_and_stays_warn() -> None:
    t = _iso(BASE + timedelta(minutes=1))
    rows = [(t, "zeta_routine"), (t, "alpha_routine"), (t, "zeta_routine"), (t, "rare")]
    out = ha.eval_warning_events("o/r", rows, BASE, frozenset({"alpha_routine", "zeta_routine"}))
    facts = "warning_rows=4 new_since_last_beat=4"
    assert [f.severity for f in out] == ["warn", "warn"]
    assert out[0].detail == f"new warning-level event(s) since last beat: ['rare@{t}'] ({facts})"
    assert out[1].detail == (
        f"3 routine operational warnings (alpha_routine=1, zeta_routine=2) ({facts})"
    )


def test_warning_events_ok_when_empty() -> None:
    assert ha.eval_warning_events("o/r", [], BASE, frozenset()) == [
        ha.Finding("warning-events o/r", "o/r", "ok", "", "warning_rows=0 new_since_last_beat=0")
    ]


def test_draft_pr_blocked_warns_only_when_new() -> None:
    new = (_iso(BASE + timedelta(seconds=1)),)
    old = (_iso(BASE),)
    assert ha.eval_draft_pr_blocked("o/r", [old], BASE).severity == "ok"
    f = ha.eval_draft_pr_blocked("o/r", [old, new], BASE)
    assert f.severity == "warn"
    assert f.detail == (
        "draft_pr_blocked since last beat: 1 event(s) (blocked_rows=2 new_since_last_beat=1)"
    )


def test_infra_blocked_escalated_outranks_blocked() -> None:
    t = _iso(BASE + timedelta(minutes=1))
    blocked = [(t, "check_infra_blocked")]
    assert ha.eval_infra_blocked("o/r", blocked, [], BASE).detail == (
        "check_infra_blocked since last beat: 1 event(s) (blocked_rows=1 escalated_rows=0)"
    )
    esc = ha.eval_infra_blocked("o/r", blocked, [(t,)], BASE)
    assert esc.severity == "anomaly"
    assert esc.detail == (
        f"infra_blocked_escalated since last beat: ['{t}'] (blocked_rows=1 escalated_rows=1)"
    )
    assert ha.eval_infra_blocked("o/r", [], [], BASE).severity == "ok"


def test_ci_headroom_buckets_reasons_and_tolerates_bad_payloads() -> None:
    t = _iso(BASE + timedelta(minutes=1))
    rows = [
        (t, json.dumps({"reason": "stale"})),
        (t, json.dumps({"reason": "stale"})),
        (t, json.dumps({"reason": "missing"})),
        (t, "not json"),
        (t, json.dumps([1])),
        (_iso(BASE), json.dumps({"reason": "old"})),
    ]
    f = ha.eval_ci_headroom_unavailable("o/r", rows, BASE)
    assert f.severity == "warn"
    assert f.detail == (
        "ci_headroom_unavailable since last beat: 5 event(s), "
        "reasons={'stale': 2, 'missing': 1, 'unknown': 2} "
        "(unavailable_rows=6 new_since_last_beat=5)"
    )


def test_local_lane_stalled_collects_switches_and_issues() -> None:
    t = _iso(BASE + timedelta(minutes=1))
    rows = [
        (t, json.dumps({"switch": "auto_merge.enabled", "issue_numbers": [7, 3, True, "x"]})),
        (t, json.dumps({"switch": "review_dispatch.enabled", "issue_numbers": [3]})),
        (t, "garbage"),
        (_iso(BASE), json.dumps({"switch": "old", "issue_numbers": [99]})),
    ]
    f = ha.eval_local_lane_stalled("o/r", rows, BASE)
    assert f.severity == "anomaly"
    assert f.detail == (
        "3 event(s) since last beat; disabled switch(es): "
        "['auto_merge.enabled', 'review_dispatch.enabled']; stranded issue(s): [3, 7] "
        "(stalled_rows=4 new_since_last_beat=3)"
    )
    assert ha.eval_local_lane_stalled("o/r", rows[3:], BASE).severity == "ok"


# --- freshness evaluators: boundary at the threshold ---------------------------


def test_loop_pass_freshness_boundary_is_strictly_greater() -> None:
    ha_min = haf.LOOP_PASS_STALE_MINUTES
    at = _iso(NOW - timedelta(minutes=ha_min))
    f = haf.eval_loop_pass_freshness("o/r", at, NOW, marker_hint="/m")
    assert f == ha.Finding(
        "loop-pass-freshness o/r",
        "o/r",
        "ok",
        "",
        f"newest_loop_started={at} age={ha_min}m",
    )
    over = _iso(NOW - timedelta(minutes=ha_min, seconds=30))
    g = haf.eval_loop_pass_freshness("o/r", over, NOW, marker_hint="/m")
    assert g.severity == "anomaly"
    assert g.detail.startswith(f"no loop pass in o/r for 90m (newest loop_started {over})")
    assert "/m is one thing worth checking" in g.detail


def test_loop_pass_freshness_edge_inputs() -> None:
    assert (
        haf.eval_loop_pass_freshness("o/r", None, NOW).facts == "no loop_started rows recorded yet"
    )
    bad = haf.eval_loop_pass_freshness("o/r", "zzz", NOW)
    assert (bad.severity, bad.detail) == ("anomaly", "newest loop_started ts unparseable: 'zzz'")


def test_log_freshness_boundary_and_missing() -> None:
    at = (NOW - timedelta(minutes=haf.LOG_FRESHNESS_STALE_MINUTES)).timestamp()
    ok = haf.eval_log_freshness("o/r", at, "state.json", NOW)
    assert (ok.severity, ok.facts) == ("ok", "freshest=state.json age=30m")
    over = haf.eval_log_freshness("o/r", at - 60, "state.json", NOW)
    assert over.severity == "anomaly"
    assert over.detail == "freshest file older than threshold=30m (freshest=state.json age=31m)"
    none = haf.eval_log_freshness("o/r", None, "", NOW)
    assert none.detail == "no log/state/checkpoint files found under state dir"


def _hb(**kw: object) -> dict[str, object]:
    return {
        "last_beat_at": _iso(NOW - timedelta(minutes=60)),
        "pid": 42,
        "exited_at": None,
        "max_pass_runtime_seconds": 1800,
        **kw,
    }


def test_supervisor_heartbeat_threshold_is_twice_pass_timeout() -> None:
    # threshold = 2 * 1800s = 60m: exactly 60m old is still OK, 61m is stale.
    ok = haf.eval_supervisor_heartbeat(_hb(), NOW)
    assert (ok.severity, ok.facts) == (
        "ok",
        "last_beat=60m ago pid=42 exited_at=None pass_timeout=1800s",
    )
    stale = haf.eval_supervisor_heartbeat(_hb(last_beat_at=_iso(NOW - timedelta(minutes=61))), NOW)
    assert stale.severity == "anomaly"
    assert stale.detail == (
        "supervisor heartbeat stale: last beat 61m ago with no clean exit "
        "(threshold=60m) — likely killed or hung "
        "(last_beat=61m ago pid=42 exited_at=None pass_timeout=1800s)"
    )


def test_supervisor_heartbeat_exited_cleanly_variant_and_fallbacks() -> None:
    old = _iso(NOW - timedelta(minutes=300))
    f = haf.eval_supervisor_heartbeat(_hb(last_beat_at=old, exited_at="E"), NOW)
    assert f.detail.startswith(
        "supervisor exited cleanly at E but has not restarted in 300m (threshold=60m)"
    )
    # no max_pass_runtime_seconds -> full_pass_interval_seconds (300s -> 10m threshold)
    data = _hb(full_pass_interval_seconds=300)
    del data["max_pass_runtime_seconds"]
    assert "pass_timeout=300s" in haf.eval_supervisor_heartbeat(data, NOW).detail
    # neither -> default 1800
    del data["full_pass_interval_seconds"]
    assert haf.eval_supervisor_heartbeat(data, NOW).facts.endswith("pass_timeout=1800s")


def test_supervisor_heartbeat_unusable_inputs() -> None:
    assert haf.eval_supervisor_heartbeat(None, NOW).severity == "anomaly"
    assert haf.eval_supervisor_heartbeat([1], NOW).detail == (
        "supervisor-heartbeat.json malformed (not a JSON object)"
    )
    assert haf.eval_supervisor_heartbeat({"last_beat_at": "x"}, NOW).detail == (
        "supervisor-heartbeat.json has no parseable last_beat_at"
    )


def test_wedge_kill_loop_lookback_boundary_is_inclusive() -> None:
    edge = _iso(NOW - timedelta(hours=haf.SUPERVISOR_WEDGE_LOOP_LOOKBACK_HOURS))
    older = _iso(NOW - timedelta(hours=24, seconds=1))
    f = haf.eval_wedge_kill_loop([edge, older], NOW)
    facts = "total_events=2 recent=1 lookback_hours=24"
    assert f == ha.Finding(
        "supervisor-wedge-kill-loop",
        None,
        "anomaly",
        f"supervisor_wedge_loop fired 1 time(s) in the last 24h (most recent {edge}) -- "
        f"the wedge-kill backstop is looping instead of recovering ({facts})",
        facts,
    )
    assert haf.eval_wedge_kill_loop([older], NOW).facts == (
        "total_events=1 recent=0 lookback_hours=24"
    )


# --- notify digest -------------------------------------------------------------


def _res(**payload: object) -> tuple[str, str]:
    return (_iso(NOW), json.dumps(payload))


def _probe(**kw: object):
    p = SimpleNamespace(exists=True, age_hours=1.0, age_source="mtime", error=None, **kw)
    return lambda _path: p


def test_notify_digest_cannot_tell_outcomes_are_warn() -> None:
    f = haf.eval_notify_digest(None, [], NOW, 72, _probe())
    assert f.severity == "warn"
    assert f.detail.endswith("(stale_events_72h=0)")
    off = haf.eval_notify_digest(_res(enabled=False), [], NOW, 72, _probe())
    assert off.severity == "warn" and "enabled=false" in off.detail
    notobj = haf.eval_notify_digest((_iso(NOW), "[1]"), [], NOW, 72, _probe())
    assert notobj.detail == "latest notify_resolution payload is not a JSON object: '[1]'"


def test_notify_digest_sink_and_path_verdicts() -> None:
    other = haf.eval_notify_digest(_res(enabled=True, sink="slack"), [], NOW, 72, _probe())
    assert (other.severity, other.facts) == (
        "ok",
        f"enabled with sink=slack (no digest file to tail; resolved at {NOW.isoformat()})",
    )
    unset = haf.eval_notify_digest(
        _res(enabled=True, sink="file", file_path_empty=True), [], NOW, 72, _probe()
    )
    assert unset.severity == "anomaly" and "file_path is unset" in unset.detail


def test_notify_digest_staleness_boundary_and_stale_event_window() -> None:
    res = _res(enabled=True, sink="file", resolved_file_path="D/digest.jsonl")
    stale_events = [_iso(NOW - timedelta(hours=72)), _iso(NOW - timedelta(hours=73))]

    def probe(age: float):
        p = SimpleNamespace(exists=True, age_hours=age, age_source="gen", error=None)
        return lambda _path: p

    ok = haf.eval_notify_digest(res, stale_events, NOW, 72, probe(72.0))
    assert ok.severity == "ok"
    assert ok.facts == (
        f"last entry 72.0h old (gen); threshold=72h path={Path('D/digest.jsonl')}; "
        "stale_events_72h=1"
    )
    dead = haf.eval_notify_digest(res, [], NOW, 72, probe(72.1))
    assert dead.severity == "anomaly"
    assert dead.detail.startswith("notify digest writer looks dead: last entry 72.1h old")


def test_notify_digest_probe_failures_are_anomalies() -> None:
    res = _res(enabled=True, sink="file", resolved_file_path="D/x.jsonl")
    err = SimpleNamespace(exists=False, age_hours=None, age_source="unreadable", error="EIO")
    f = haf.eval_notify_digest(res, [], NOW, 72, lambda _p: err)
    assert f.detail == f"{Path('D/x.jsonl')} unreadable: EIO (stale_events_72h=0)"
    gone = SimpleNamespace(exists=False, age_hours=None, age_source="missing", error=None)
    g = haf.eval_notify_digest(res, [], NOW, 72, lambda _p: gone)
    assert g.severity == "anomaly" and "does not exist" in g.detail


# --- real-writer round trips ---------------------------------------------------


def _rows(db: Path, sql: str, params: tuple = ()) -> list[tuple]:
    conn = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def test_log_event_rows_feed_the_evaluators(tmp_path: Path) -> None:
    state = tmp_path / "state.json"
    log_event(state, "draft_pr_blocked", {"pr": 1}, repo="o/r")
    log_event(state, "supervisor_wedge_loop", {"n": 3}, repo="o/r")
    log_event(state, "ci_headroom_unavailable", {"reason": "stale"}, repo="o/r")
    db = tmp_path / "events.db"
    # Written "now" by the real writer: new relative to an old baseline,
    # and stale relative to a future one.
    past = datetime.now(timezone.utc) - timedelta(hours=1)
    future = datetime.now(timezone.utc) + timedelta(hours=1)

    draft = _rows(db, "SELECT ts FROM events WHERE kind = ?", ("draft_pr_blocked",))
    assert ha.eval_draft_pr_blocked("o/r", draft, past).detail == (
        "draft_pr_blocked since last beat: 1 event(s) (blocked_rows=1 new_since_last_beat=1)"
    )
    assert ha.eval_draft_pr_blocked("o/r", draft, future).severity == "ok"

    hr = _rows(db, "SELECT ts, payload FROM events WHERE kind = ?", ("ci_headroom_unavailable",))
    assert "reasons={'stale': 1}" in ha.eval_ci_headroom_unavailable("o/r", hr, past).detail

    wedge = [
        ts
        for (ts,) in _rows(db, "SELECT ts FROM events WHERE kind = ?", ("supervisor_wedge_loop",))
    ]
    f = haf.eval_wedge_kill_loop(wedge, datetime.now(timezone.utc))
    assert f.severity == "anomaly" and f.facts == "total_events=1 recent=1 lookback_hours=24"

    errors = _rows(db, "SELECT ts, kind FROM events WHERE level = 'error'")
    # Only supervisor_wedge_loop is error-level among the three events written above.
    assert errors == [(errors[0][0], "supervisor_wedge_loop")]
    assert ha.eval_error_events("o/r", errors, past).facts == (
        "error_rows=1 new_since_last_beat=1"
    )


def test_supervisor_heartbeat_file_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "supervisor-heartbeat.json"
    write_supervisor_heartbeat(
        path,
        {
            "last_beat_at": _iso(NOW - timedelta(minutes=5)),
            "pid": 7,
            "exited_at": None,
            "max_pass_runtime_seconds": 1800,
        },
    )
    f = haf.eval_supervisor_heartbeat(json.loads(path.read_text(encoding="utf-8")), NOW)
    assert (f.severity, f.facts) == (
        "ok",
        "last_beat=5m ago pid=7 exited_at=None pass_timeout=1800s",
    )


# --- leaf contract ---------------------------------------------------------------


@pytest.mark.parametrize("name", ["heartbeat_alarms.py", "heartbeat_alarms_fleet.py"])
def test_modules_are_stdlib_only_leaves_without_clock_or_io(name: str) -> None:
    src = Path(__file__).parent.parent / "src" / "charlie_work" / name
    tree = ast.parse(src.read_text(encoding="utf-8"))
    imported = {
        (n.module if isinstance(n, ast.ImportFrom) else a.name)
        for n in ast.walk(tree)
        if isinstance(n, (ast.Import, ast.ImportFrom))
        for a in (n.names if isinstance(n, ast.Import) else [n])
    }
    allowed_pkg = {"charlie_work.heartbeat_alarms"}
    assert not {m for m in imported if m and m.startswith(("ci_fleet",))}
    assert {m for m in imported if m and m.startswith("charlie_work")} <= allowed_pkg
    assert not {"sqlite3", "subprocess", "os", "shutil"} & imported
    assert not _clock_or_io_calls(tree), "leaf must stay clock- and I/O-free"
    assert len(src.read_text(encoding="utf-8").splitlines()) < 400


_CLOCK_ATTR_CALLS = {
    ("datetime", "now"),
    ("datetime", "utcnow"),
    ("datetime", "today"),
    ("date", "today"),
    ("time", "time"),
    ("time", "monotonic"),
    ("time", "perf_counter"),
    ("sqlite3", "connect"),
}
_IO_METHODS = {"read_text", "read_bytes", "open", "write_text", "write_bytes"}


def _clock_or_io_calls(tree: ast.AST) -> list[str]:
    """Call nodes that read a clock or touch the filesystem/database."""
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if isinstance(fn, ast.Name) and fn.id == "open":
            found.append("open()")
        elif isinstance(fn, ast.Attribute):
            base = fn.value
            base_name = (
                base.id
                if isinstance(base, ast.Name)
                else base.attr
                if isinstance(base, ast.Attribute)
                else None
            )
            if (base_name, fn.attr) in _CLOCK_ATTR_CALLS:
                found.append(f"{base_name}.{fn.attr}()")
            elif fn.attr in _IO_METHODS:
                found.append(f".{fn.attr}()")
    return found


def test_clock_io_detector_flags_known_positives() -> None:
    """Positive control: the AST walk must catch each forbidden shape."""
    snippets = [
        "datetime.datetime.now()",
        "datetime.now()",
        "date.today()",
        "time.monotonic()",
        "open('x')",
        "Path('x').read_text()",
        "sqlite3.connect('x')",
    ]
    bad = ast.parse(chr(10).join(snippets))
    assert len(_clock_or_io_calls(bad)) == 7
