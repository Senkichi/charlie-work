"""``propose_done`` running a charter ``done_when.check`` (design §8.3, issue #3119).

Split out of ``test_foreman_apply.py`` (per-module size cap): these cases
need a *real git* repo — the check runs in a fresh detached worktree at the
default branch tip — so the foreman and the repo both live under
``wt_scratch``. Covers the pass (foreman moves to ``done``, final report
item), the failure (refused with the command's tail, recorded for the next
briefing), the timeout, and dry-run.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from _worktree_fixtures import (
    _init_repo,
    _wt_scratch as _register_wt_scratch,  # noqa: F401 -- registers the wt_scratch fixture
)

from charlie_work.foreman import apply_handlers as handlers_mod
from charlie_work.foreman.apply import OUTCOME_APPLIED, OUTCOME_REFUSED, apply_turn
from charlie_work.foreman.charter import Charter, CharterBounds, DoneWhen
from charlie_work.foreman.items import load_items
from charlie_work.foreman.paths import applied_path
from charlie_work.foreman.state import ForemanState, read_state_snapshot, save_state
from charlie_work.local_issues import LocalFileGitHub

REPO = "acme/widgets"


def _write_actions(froot: Path, turn_id: str, *actions: dict, summary: str = "a turn") -> None:
    turn = froot / "turns" / turn_id
    turn.mkdir(parents=True, exist_ok=True)
    (turn / "actions.json").write_text(
        json.dumps({"turn_summary": summary, "actions": list(actions)}), encoding="utf-8"
    )


def _setup_done(
    wt_scratch: Path,
    *,
    check: tuple[str, ...],
    state: str = "active",
    standing: bool = False,
) -> tuple[Path, LocalFileGitHub, Charter]:
    """A foreman directory plus a *real git* repo — ``done_when.check`` worktrees from it."""
    froot = wt_scratch / "foremen" / "pepe"
    froot.mkdir(parents=True)
    save_state(froot, ForemanState(state=state))
    repo_root = wt_scratch / "repo"
    _init_repo(repo_root)
    issues_dir = repo_root / ".issues"
    issues_dir.mkdir(parents=True)
    gh = LocalFileGitHub(repo_root=repo_root, issues_dir=issues_dir)
    charter = Charter(
        name="pepe",
        outcome="ship the outcome",
        repos=(REPO,),
        standing=standing,
        done_when=DoneWhen(check=check),
        bounds=CharterBounds(),
    )
    return froot, gh, charter


def test_propose_done_runs_done_when_and_moves_foreman_to_done(wt_scratch: Path) -> None:
    ran_in = wt_scratch / "check-cwd.txt"
    froot, gh, charter = _setup_done(
        wt_scratch,
        check=(
            sys.executable,
            "-c",
            "import os, sys; open(sys.argv[1], 'w').write(os.getcwd())",
            str(ran_in),
        ),
    )
    _write_actions(froot, "t1", {"type": "propose_done", "evidence": "all landed"})
    result = apply_turn(froot, "t1", charter=charter, gh=gh)
    assert result.ok
    (outcome,) = result.value.outcomes
    assert outcome.outcome == OUTCOME_APPLIED
    assert "done" in (outcome.detail or "")
    # The check ran inside the detached check/ worktree, removed after.
    cwd = Path(ran_in.read_text(encoding="utf-8").strip())
    assert os.path.normcase(cwd) == os.path.normcase((froot / "check").resolve())
    assert not (froot / "check").exists()
    # A pass moves the foreman to done and raises a final report.
    assert read_state_snapshot(froot).state == "done"
    (item,) = load_items(froot)
    assert item.kind == "report"
    assert "all landed" in (item.text or "")


def test_propose_done_done_when_failure_is_refused_with_the_tail(wt_scratch: Path) -> None:
    froot, gh, charter = _setup_done(
        wt_scratch,
        check=(
            sys.executable,
            "-c",
            "import sys; sys.stderr.write('tail-marker boom\\n'); sys.exit(3)",
        ),
    )
    _write_actions(froot, "t1", {"type": "propose_done", "evidence": "x"})
    result = apply_turn(froot, "t1", charter=charter, gh=gh)
    assert result.ok
    (outcome,) = result.value.outcomes
    assert outcome.outcome == OUTCOME_REFUSED
    assert "exited 3" in (outcome.reason or "")
    # The command's tail rides the refusal into the next briefing.
    assert "tail-marker boom" in (outcome.reason or "")
    assert read_state_snapshot(froot).state == "active"
    assert not (froot / "check").exists()
    # The briefing's refused section reads applied.jsonl records by `outcome`.
    (record,) = [
        json.loads(line)
        for line in applied_path(froot, "t1").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert record["outcome"] == "refused"
    assert "tail-marker boom" in record["reason"]


def test_propose_done_done_when_timeout_is_refused(wt_scratch: Path, monkeypatch) -> None:
    monkeypatch.setattr(handlers_mod, "DONE_CHECK_TIMEOUT_SECONDS", 1)
    froot, gh, charter = _setup_done(
        wt_scratch,
        check=(sys.executable, "-c", "import time; time.sleep(60)"),
    )
    _write_actions(froot, "t1", {"type": "propose_done", "evidence": "x"})
    result = apply_turn(froot, "t1", charter=charter, gh=gh)
    assert result.ok
    (outcome,) = result.value.outcomes
    assert outcome.outcome == OUTCOME_REFUSED
    assert "timed out" in (outcome.reason or "")
    assert read_state_snapshot(froot).state == "active"
    assert not (froot / "check").exists()


def test_propose_done_done_when_dry_run_never_runs_the_check(wt_scratch: Path) -> None:
    ran_in = wt_scratch / "check-cwd.txt"
    froot, gh, charter = _setup_done(
        wt_scratch,
        check=(
            sys.executable,
            "-c",
            "import os, sys; open(sys.argv[1], 'w').write(os.getcwd())",
            str(ran_in),
        ),
    )
    _write_actions(froot, "t1", {"type": "propose_done", "evidence": "x"})
    result = apply_turn(froot, "t1", charter=charter, gh=gh, dry_run=True)
    assert result.ok
    (outcome,) = result.value.outcomes
    assert outcome.outcome == OUTCOME_APPLIED
    assert "would run done_when.check" in (outcome.detail or "")
    assert not ran_in.exists()
    assert not (froot / "check").exists()
    assert read_state_snapshot(froot).state == "active"
    assert load_items(froot) == ()
