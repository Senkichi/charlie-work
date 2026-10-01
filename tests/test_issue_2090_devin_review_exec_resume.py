"""Issue #2090: resume a devin-shell review session that ended on a refused exec.

App-level: every test drives ``OrchestratorApp._reap_review_verdicts`` against a
real devin review sidecar + log on disk. Only the process boundary is faked:
``popen_worker`` (records the resume argv, writes what the "resumed Devin"
would print into the log handle it was given) and ``devin list`` (the session-id
source). State, events, sidecars, caps and the reaper itself are real.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from _review_fixtures import (
    _dispatch_reviews_app,
    _set_review_dispatched_state,
    _write_review_packet,
)
from charlie_work import devin_review_resume
from charlie_work.config import ConfigError, build_config_from_data
from charlie_work.devin_review_resume import (
    build_resume_command,
    find_devin_session_id,
    review_exec_nudge_text,
)
from charlie_work.devin_shell import _REVIEW_EXEC_ALLOWLIST
from charlie_work.dispatch_selection import _count_live_reviews
from charlie_work.state import load_state

PR = 2087
ISSUE = 2081
SESSION = "rough-tennis"
REJECTION = (
    "I'll review the PR.\nwarning: rejected a tool call that requires confirmation. "
    "Running in non-interactive mode.\n"
)
VERDICT = (
    "Final verdict:\n```json\n"
    '{"decision": "approved", "summary": "looks right", "required_changes": []}\n```\n'
)


class _Rig:
    """Fake process boundary + liveness for one app."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, app: Any) -> None:
        self.app = app
        self.reviews_dir: Path = app._layout.reviews_dir
        self.popens: list[dict[str, Any]] = []
        self.next_output: list[str] = []  # what each resumed session prints
        self.live: set[int] = set()
        self.launch_error: OSError | None = None
        self.list_rows: list[dict[str, Any]] | None = [
            {"id": SESSION, "last_activity_at": 4_000_000_000}
        ]
        self._pid = 50_000

        def fake_popen(argv: Any, **kwargs: Any) -> Any:
            if self.launch_error is not None:
                raise self.launch_error
            self._pid += 1
            self.popens.append({"argv": list(argv), "cwd": kwargs.get("cwd"), "pid": self._pid})
            out = self.next_output.pop(0) if self.next_output else REJECTION
            kwargs["stdout"].write(out)
            kwargs["stdout"].flush()
            self.live.add(self._pid)
            return SimpleNamespace(pid=self._pid)

        def fake_list(argv: Any, **kwargs: Any) -> Any:
            if self.list_rows is None:
                return SimpleNamespace(ok=False, stdout="", returncode=1)
            return SimpleNamespace(ok=True, stdout=json.dumps(self.list_rows), returncode=0)

        alive = lambda pid, *_: pid in self.live  # noqa: E731
        monkeypatch.setattr(devin_review_resume, "popen_worker", fake_popen)
        monkeypatch.setattr(devin_review_resume, "run_captured", fake_list)
        monkeypatch.setattr(devin_review_resume, "_get_process_start_time", lambda pid: 1.0)
        monkeypatch.setattr(
            devin_review_resume, "maybe_start_terminal_status_watcher", lambda *a, **k: None
        )
        monkeypatch.setattr(devin_review_resume, "write_worktree_marker", lambda *a, **k: None)
        monkeypatch.setattr("charlie_work.worker_fate.is_alive", alive)
        monkeypatch.setattr("charlie_work.process_utils.is_pid_alive", alive)
        monkeypatch.setattr("charlie_work.stalled_review_reap.is_pid_alive", alive)

    def seed_dead_review(self, log_text: str = REJECTION, *, pid: int = 40_001) -> None:
        checkout = self.reviews_dir / f"pr-{PR}"
        checkout.mkdir(parents=True, exist_ok=True)
        log = self.reviews_dir / f"issue-{PR}.log"
        log.write_text(log_text, encoding="utf-8")
        sidecar = {
            "issue_number": PR,
            "branch": "agent/issue-2081-x",
            "worktree_path": str(checkout),
            "prompt_path": str(self.reviews_dir / "p.md"),
            "command": [
                "devin",
                "--model",
                "swe-2-high",
                "--prompt-file",
                str(self.reviews_dir / "p.devin.md"),
                "--print",
                "--respect-workspace-trust",
                "false",
            ],
            "pid": pid,
            "started_at": "2026-09-30T18:18:26Z",
            "log_path": str(log),
            "process_start_time": 1_790_792_306.5,
            "session_id": "b146a91e-66c0-4ead-a822-e952cecbc9ce",
        }
        (self.reviews_dir / f"issue-{PR}.json").write_text(json.dumps(sidecar), encoding="utf-8")
        _set_review_dispatched_state(self.app, PR, ISSUE, "2026-09-30T18:18:27Z")

    def reap(self) -> dict[str, Any]:
        return self.app._reap_review_verdicts(self.reviews_dir)

    def finish_resumed(self) -> None:
        """The resumed process exits (its log keeps what it printed)."""
        self.live.clear()

    def events(self, kind: str) -> list[dict[str, Any]]:
        state = load_state(self.app.paths.state_file)
        return [e["payload"] for e in state["events"] if e["kind"] == kind]


def _rig(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **review_dispatch: Any) -> _Rig:
    pr = {
        "number": PR,
        "title": "Fix",
        "url": f"https://example.test/pull/{PR}",
        "headRefName": "agent/issue-2081-x",
        "baseRefName": "main",
        "headRefOid": "sha-2087",
        "mergeStateStatus": "CLEAN",
        "body": f"Closes #{ISSUE}",
        "labels": [],
        "isCrossRepository": False,
        "state": "OPEN",
    }
    app = _dispatch_reviews_app(tmp_path, prs=[pr])
    _write_review_packet(tmp_path, PR, "sha-2087")
    if review_dispatch:
        from dataclasses import replace

        rd = replace(app.config.review_dispatch, **review_dispatch)
        object.__setattr__(app.config, "review_dispatch", rd)
    return _Rig(monkeypatch, app)


# --- acceptance 1: reject once, resumed session emits a verdict ---------------


def test_rejected_review_is_resumed_and_the_verdict_is_recorded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rig = _rig(monkeypatch, tmp_path)
    rig.seed_dead_review()
    rig.next_output = [VERDICT]

    first = rig.reap()

    assert first == {"recorded": [], "missed": []}
    assert len(rig.popens) == 1
    argv = rig.popens[0]["argv"]
    assert argv[argv.index("--resume") + 1] == SESSION
    assert "--print" in argv and "--permission-mode" not in argv
    assert rig.popens[0]["cwd"] == str(rig.reviews_dir / f"pr-{PR}")
    resumed = rig.events("review_exec_rejection_resumed")
    assert len(resumed) == 1
    assert {k: resumed[0][k] for k in ("pr_number", "attempt", "session_id")} == {
        "pr_number": PR,
        "attempt": 1,
        "session_id": SESSION,
    }

    rig.finish_resumed()
    second = rig.reap()

    assert [r["decision"] for r in second["recorded"]] == ["approved"]
    assert second["missed"] == []
    assert rig.events("review_verdict_missed") == []
    assert len(rig.popens) == 1  # a verdict ends the resuming


def test_the_nudge_file_states_the_allow_list_rendered_from_the_constant(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rig = _rig(monkeypatch, tmp_path)
    rig.seed_dead_review()
    rig.reap()

    argv = rig.popens[0]["argv"]
    nudge = Path(argv[argv.index("--prompt-file") + 1])
    assert nudge != rig.reviews_dir / "p.devin.md"  # the original prompt is replaced
    text = nudge.read_text(encoding="utf-8")
    assert text == review_exec_nudge_text()
    for entry in _REVIEW_EXEC_ALLOWLIST:
        assert f"`{entry[len('Exec(') : -1]}`" in text, entry
    assert "REFUSED" in text and "Do not retry" in text and "verdict" in text


# --- acceptance 2: reject every time -> exactly N resumes, then a miss --------


@pytest.mark.parametrize("limit", [1, 2, 3])
def test_rejecting_every_attempt_yields_exactly_n_resumes_then_a_miss(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, limit: int
) -> None:
    rig = _rig(monkeypatch, tmp_path, review_exec_rejection_max_resumes=limit)
    rig.seed_dead_review()

    results = []
    for _ in range(limit + 1):
        results.append(rig.reap())
        rig.finish_resumed()

    assert len(rig.popens) == limit
    assert [e["attempt"] for e in rig.events("review_exec_rejection_resumed")] == list(
        range(1, limit + 1)
    )
    assert all(r["missed"] == [] for r in results[:limit])
    assert [m["cause"]["cause"] for m in results[limit]["missed"]] == ["reviewer_exec_rejected"]
    missed = rig.events("review_verdict_missed")
    assert len(missed) == 1 and missed[0]["cause"]["cause"] == "reviewer_exec_rejected"


def test_zero_disables_resuming(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    rig = _rig(monkeypatch, tmp_path, review_exec_rejection_max_resumes=0)
    rig.seed_dead_review()

    result = rig.reap()

    assert rig.popens == []
    assert len(result["missed"]) == 1
    assert rig.events("review_exec_rejection_resumed") == []


def test_a_new_dispatch_gets_a_fresh_resume_budget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rig = _rig(monkeypatch, tmp_path, review_exec_rejection_max_resumes=1)
    rig.seed_dead_review()
    rig.reap()
    rig.finish_resumed()
    assert len(rig.reap()["missed"]) == 1  # budget spent for this dispatch

    rig.seed_dead_review()  # re-dispatch: new review_dispatched_at
    _set_review_dispatched_state(rig.app, PR, ISSUE, "2026-09-30T19:00:00Z")
    rig.reap()

    assert len(rig.popens) == 2


# --- acceptance 3: a resume that cannot launch falls back to a miss -----------


def test_resume_launch_failure_falls_back_to_a_miss_as_a_value(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rig = _rig(monkeypatch, tmp_path)
    rig.seed_dead_review()
    rig.launch_error = FileNotFoundError("devin not found")

    result = rig.reap()  # must not raise

    assert len(result["missed"]) == 1
    failed = rig.events("review_exec_rejection_resume_failed")
    assert len(failed) == 1 and failed[0]["reason"].startswith("launch_failed")
    assert rig.events("review_exec_rejection_resumed") == []


def test_unknown_session_id_falls_back_to_a_miss(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rig = _rig(monkeypatch, tmp_path)
    rig.seed_dead_review()
    rig.list_rows = None  # `devin list` fails

    result = rig.reap()

    assert rig.popens == []
    assert len(result["missed"]) == 1
    assert rig.events("review_exec_rejection_resume_failed")[0]["reason"] == (
        "session_id_unavailable"
    )


def test_missing_checkout_falls_back_to_a_miss(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rig = _rig(monkeypatch, tmp_path)
    rig.seed_dead_review()
    (rig.reviews_dir / f"pr-{PR}").rmdir()

    result = rig.reap()

    assert rig.popens == [] and len(result["missed"]) == 1
    assert rig.events("review_exec_rejection_resume_failed")[0]["reason"] == (
        "review_checkout_missing"
    )


def test_only_exec_rejections_are_resumed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    rig = _rig(monkeypatch, tmp_path)
    rig.seed_dead_review("the model just stopped\n")

    result = rig.reap()

    assert rig.popens == [] and len(result["missed"]) == 1


def test_claude_code_reviews_are_not_resumed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rig = _rig(monkeypatch, tmp_path)
    rig.seed_dead_review()
    (rig.reviews_dir / f"issue-{PR}.json").rename(rig.reviews_dir / f"issue-{PR}.claude.json")

    rig.reap()

    assert rig.popens == []


# --- acceptance 4: the resumed session occupies the SAME slot -----------------


def test_resumed_session_is_not_double_counted_in_either_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rig = _rig(monkeypatch, tmp_path)
    rig.seed_dead_review()
    state_file = rig.app.paths.state_file
    assert _count_live_reviews(rig.reviews_dir, state_file) == 0  # dead original: no slot

    rig.reap()

    # Per-repo cap: one sidecar, one live pid -> exactly one slot, not two.
    assert len(list(rig.reviews_dir.glob("issue-*.json"))) == 1
    assert _count_live_reviews(rig.reviews_dir, state_file) == 1
    entry = load_state(state_file)["prs"][str(PR)]
    assert entry["reviewer_pid"] == rig.popens[0]["pid"]
    assert entry["review_dispatch_status"] == "review_dispatch_dispatched"

    # Fleet cap: the fleet registry sums the same per-repo count over the
    # registered repo; register this app's repo and count it.
    from charlie_work.fleet_registry import count_fleet_live_reviews

    fleet_dir = tmp_path / "fleet"
    fleet_dir.mkdir()
    (tmp_path / ".git").mkdir(exist_ok=True)
    (fleet_dir / "fleet.json").write_text(
        json.dumps(
            {
                "version": 1,
                "repos": {
                    "owner/r": {
                        "repo_root": str(tmp_path),
                        "name_with_owner": "owner/r",
                        "config_path": str(tmp_path / "orchestrator.config.yaml"),
                        "state_dir": str(rig.app.paths.root),
                        "first_seen": "2024-01-01T00:00:00Z",
                        "last_seen": "2024-01-01T00:00:00Z",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    count, skipped = count_fleet_live_reviews(str(fleet_dir))
    assert (count, list(skipped)) == (1, [])


# --- session-id source / command / config units -------------------------------


def _list_returning(monkeypatch: pytest.MonkeyPatch, payload: Any, *, ok: bool = True) -> None:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    monkeypatch.setattr(
        devin_review_resume,
        "run_captured",
        lambda *a, **k: SimpleNamespace(ok=ok, stdout=text),
    )


def test_find_session_id_picks_newest_active_since_launch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _list_returning(
        monkeypatch,
        [
            {"id": "old-round", "last_activity_at": 1000},
            {"id": "newest", "last_activity_at": 2100},
            {"id": "middle", "last_activity_at": 2050},
        ],
    )
    assert find_devin_session_id("devin", tmp_path, started_epoch=2000.4) == "newest"
    # Nothing active since the launch: an earlier round's session is never picked.
    assert find_devin_session_id("devin", tmp_path, started_epoch=9000.0) is None


@pytest.mark.parametrize("payload", ["not json", {"id": "x"}, [{"id": "x"}], []])
def test_find_session_id_tolerates_garbage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, payload: Any
) -> None:
    _list_returning(monkeypatch, payload)
    assert find_devin_session_id("devin", tmp_path, started_epoch=None) is None


def test_find_session_id_cli_failure_is_a_value(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _list_returning(monkeypatch, "", ok=False)
    assert find_devin_session_id("devin", tmp_path, started_epoch=None) is None


def test_resume_command_repins_the_read_only_posture(tmp_path: Path) -> None:
    cmd = build_resume_command(
        (
            "devin",
            "--permission-mode",
            "dangerous",
            "--prompt-file",
            "orig.md",
            "--print",
            "--resume",
            "stale",
        ),
        "rough-tennis",
        tmp_path / "nudge.md",
    )
    assert "dangerous" not in cmd and "--permission-mode" not in cmd
    assert "orig.md" not in cmd and "stale" not in cmd
    assert cmd[-4:] == ("--resume", "rough-tennis", "--prompt-file", str(tmp_path / "nudge.md"))


def test_the_allow_list_is_not_widened() -> None:
    assert "Exec(uv)" not in _REVIEW_EXEC_ALLOWLIST
    assert len(_REVIEW_EXEC_ALLOWLIST) == 11


def test_knob_defaults_to_two_and_validates() -> None:
    assert build_config_from_data({}).review_dispatch.review_exec_rejection_max_resumes == 2
    zero = build_config_from_data({"review_dispatch": {"review_exec_rejection_max_resumes": 0}})
    assert zero.review_dispatch.review_exec_rejection_max_resumes == 0
    for bad in (-1, True, "2"):
        with pytest.raises(ConfigError, match="review_exec_rejection_max_resumes"):
            build_config_from_data({"review_dispatch": {"review_exec_rejection_max_resumes": bad}})


def test_resume_never_blocks_on_the_worker() -> None:
    # CLAUDE.md invariant: adapters use Popen and return; no wait/communicate.
    src = Path(devin_review_resume.__file__).read_text(encoding="utf-8")
    assert ".wait(" not in src and ".communicate(" not in src
    assert subprocess  # (module imported for the Popen-stderr redirect constant)


# --- issue #2162: the stall sweep must not reap a resumed review ---------------


def test_resumed_review_survives_the_stall_sweep_and_its_verdict_is_recorded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from datetime import UTC, datetime

    from charlie_work.stalled_review_reap import _detect_and_handle_stalled_reviews

    rig = _rig(monkeypatch, tmp_path)
    rig.seed_dead_review()  # original dispatch is far older than the stale timeout
    rig.next_output = [VERDICT]
    rig.reap()  # resume launches inside the harvest
    rig.finish_resumed()  # ...and exits before the sweep runs

    app = rig.app
    stalled = _detect_and_handle_stalled_reviews(
        rig.reviews_dir,
        app.paths.state_file,
        app.config,
        app.repo_root,
        write_gate=app.write_gate,
        now=datetime.now(UTC),
    )

    assert stalled == []
    assert rig.events("review_dispatch_stalled") == []
    claim = load_state(app.paths.state_file)["prs"][str(PR)]
    assert claim["review_dispatch_status"] == "review_dispatch_dispatched"
    assert [r["decision"] for r in rig.reap()["recorded"]] == ["approved"]


def test_truly_dead_stale_claim_without_sidecar_is_still_reaped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from datetime import UTC, datetime

    rig = _rig(monkeypatch, tmp_path)
    rig.seed_dead_review()
    (rig.reviews_dir / f"issue-{PR}.json").unlink()

    rig.app._run_review_reap_sweeps(datetime.now(UTC))

    assert len(rig.events("review_dispatch_stalled")) == 1
    claim = load_state(rig.app.paths.state_file)["prs"][str(PR)]
    assert claim["review_dispatch_status"] == "review_dispatch_failed"


def test_resume_restamps_claim_age_and_keeps_the_budget_key_in_step(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rig = _rig(monkeypatch, tmp_path, review_exec_rejection_max_resumes=1)
    rig.seed_dead_review()
    rig.reap()
    claim = load_state(rig.app.paths.state_file)["prs"][str(PR)]
    assert claim["review_dispatched_at"] != "2026-09-30T18:18:27Z"
    assert claim["review_exec_resume_dispatched_at"] == claim["review_dispatched_at"]
    assert claim["review_exec_resume_count"] == 1
    rig.finish_resumed()
    assert len(rig.reap()["missed"]) == 1  # cap still enforced
    assert len(rig.popens) == 1


def test_fresh_sidecar_alone_protects_a_stale_claim_from_the_stall_sweep(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Isolates ``fresh_sidecar_pr_keys``: the claim is stale with a dead pid, and only the
    sidecar (written without a claim restamp) is fresh."""
    from datetime import UTC, datetime

    from charlie_work.stalled_review_reap import _detect_and_handle_stalled_reviews

    rig = _rig(monkeypatch, tmp_path)
    rig.seed_dead_review()
    sidecar_path = rig.reviews_dir / f"issue-{PR}.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    now = datetime.now(UTC)
    sidecar["started_at"] = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    before = load_state(rig.app.paths.state_file)["prs"][str(PR)]
    assert before["review_dispatched_at"] == "2026-09-30T18:18:27Z"  # stale claim

    app = rig.app
    stalled = _detect_and_handle_stalled_reviews(
        rig.reviews_dir,
        app.paths.state_file,
        app.config,
        app.repo_root,
        write_gate=app.write_gate,
        now=now,
    )

    assert stalled == []
    assert rig.events("review_dispatch_stalled") == []
    claim = load_state(app.paths.state_file)["prs"][str(PR)]
    assert claim["review_dispatch_status"] == "review_dispatch_dispatched"


# --- issue #2110: two concurrent reaps of one dead reviewer are single-owner --


def _reap_concurrently(rig: _Rig, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Run two reaps so both pass the status check before either proceeds.

    The barrier sits on the first log parse, i.e. right after the claim section,
    which is exactly the window the unguarded code raced in.
    """
    import threading

    from charlie_work.orchestration import misc_review_verdicts as mrv

    barrier = threading.Barrier(2, timeout=2)
    real = mrv._parse_review_verdict_from_log

    def gated(path: Path) -> Any:
        try:
            barrier.wait()
        except threading.BrokenBarrierError:
            pass  # the claim loser never reaches the parse; the winner proceeds alone
        return real(path)

    monkeypatch.setattr(mrv, "_parse_review_verdict_from_log", gated)
    results: list[dict[str, Any]] = []
    threads = [threading.Thread(target=lambda: results.append(rig.reap())) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    return results


def test_concurrent_reaps_of_an_exec_rejected_reviewer_emit_one_outcome(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rig = _rig(monkeypatch, tmp_path)
    rig.seed_dead_review()
    rig.next_output = [VERDICT]

    results = _reap_concurrently(rig, monkeypatch)

    assert len(results) == 2
    assert len(rig.events("review_exec_rejection_resumed")) == 1
    assert rig.events("review_exec_rejection_resume_failed") == []
    assert rig.events("review_verdict_missed") == []
    assert len(rig.popens) == 1
    assert all(r["missed"] == [] for r in results)


def test_concurrent_reaps_of_a_dead_reviewer_with_a_verdict_record_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rig = _rig(monkeypatch, tmp_path)
    rig.seed_dead_review(VERDICT)

    results = _reap_concurrently(rig, monkeypatch)

    assert sum(len(r["recorded"]) for r in results) == 1
    assert sum(len(r["missed"]) for r in results) == 0
    assert rig.popens == []


# --- issue #2110: the claim is released on every non-terminal exit ------------


def _claim(rig: _Rig) -> Any:
    state = load_state(rig.app.paths.state_file)
    return state["prs"][str(PR)].get("review_reaped_for")


def test_failed_record_review_releases_the_claim_and_the_next_pass_retries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rig = _rig(monkeypatch, tmp_path)
    rig.seed_dead_review(VERDICT)
    real = rig.app.record_review
    calls: list[int] = []

    def flaky(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        if len(calls) == 1:
            return SimpleNamespace(ok=False, message="boom")
        return real(*args, **kwargs)

    monkeypatch.setattr(rig.app, "record_review", flaky)

    first = rig.reap()
    assert [m["reason"] for m in first["missed"]] == ["boom"]
    assert _claim(rig) is None

    second = rig.reap()
    assert len(calls) == 2
    assert len(second["recorded"]) == 1


def test_record_review_exception_releases_the_claim_and_the_next_pass_retries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rig = _rig(monkeypatch, tmp_path)
    rig.seed_dead_review(VERDICT)
    real = rig.app.record_review
    calls: list[int] = []

    def raising(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("pr_view blew up")
        return real(*args, **kwargs)

    monkeypatch.setattr(rig.app, "record_review", raising)

    with pytest.raises(RuntimeError):
        rig.reap()
    assert _claim(rig) is None

    assert len(rig.reap()["recorded"]) == 1


def test_no_terminal_action_releases_the_claim_and_the_next_pass_retries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from charlie_work.orchestration import misc_review_verdicts as mrv

    rig = _rig(monkeypatch, tmp_path)
    rig.seed_dead_review("no verdict here\n")
    real = mrv._extract_review_session_summary
    outcomes: list[Any] = [None]

    def summary(*args: Any, **kwargs: Any) -> Any:
        return outcomes.pop(0) if outcomes else real(*args, **kwargs)

    monkeypatch.setattr(mrv, "_extract_review_session_summary", summary)

    assert rig.reap() == {"recorded": [], "missed": []}
    assert _claim(rig) is None

    second = rig.reap()
    assert len(second["missed"]) == 1
    assert _claim(rig) is not None


def test_stale_snapshot_of_a_resumed_session_is_not_resumed_or_torn_down(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from charlie_work.orchestration import misc_review_verdicts as mrv

    rig = _rig(monkeypatch, tmp_path)
    rig.seed_dead_review()
    stale = mrv.iter_workers(rig.reviews_dir)
    # A concurrent reaper resumed the session: new live pid + re-stamped sidecar.
    sidecar = rig.reviews_dir / f"issue-{PR}.json"
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    data.update(pid=40_002, started_at="2026-09-30T18:19:00Z")
    sidecar.write_text(json.dumps(data), encoding="utf-8")
    rig.live.add(40_002)
    removed: list[int] = []
    monkeypatch.setattr(
        "charlie_work.workflow.remove_review_checkout", lambda *a, **k: removed.append(1)
    )
    snapshots = [stale]
    real_iter = mrv.iter_workers
    monkeypatch.setattr(
        mrv, "iter_workers", lambda d: snapshots.pop(0) if snapshots else real_iter(d)
    )

    assert rig.reap() == {"recorded": [], "missed": []}
    assert rig.popens == [] and removed == []
    assert _claim(rig) is None
