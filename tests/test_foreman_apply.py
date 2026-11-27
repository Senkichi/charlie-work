"""The Apply engine against ``LocalFileGitHub``, a real backend (issue #3119).

Covers the design §8.3 contract: per-action check order (authority, bounds,
type-specific, effect, one event), blocker keys resolving to real numbers,
idempotent create via the hidden marker, arming rules (auto / propose /
degraded / signal-marked), crash-resume through ``applied.jsonl``, pause,
and dry-run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from charlie_work.config import LabelConfig
from charlie_work.foreman import apply as apply_mod
from charlie_work.foreman.apply import (
    EVENT_ACTION_APPLIED,
    EVENT_ACTION_REFUSED,
    OUTCOME_APPLIED,
    OUTCOME_REFUSED,
    OUTCOME_SKIPPED,
    apply_turn,
)
from charlie_work.foreman.charter import Charter, CharterBounds
from charlie_work.foreman.items import load_items
from charlie_work.foreman.paths import applied_path, state_path
from charlie_work.foreman.state import ForemanState, read_state_snapshot, save_state
from charlie_work.instrumentation import read_event_log
from charlie_work.issue_contract_lint import lint_issue_body
from charlie_work.local_issues import LocalFileGitHub

REPO = "acme/widgets"
OWNER = f"{LabelConfig().foreman_prefix}pepe"
READY = LabelConfig().ready

# A credential-shaped token the outbound guard's vendored rules match
# (same construction as tests/test_outbound_body_guard.py).
_TOKEN = "ghp_" + "x7K9pQ2mN4vB8wL1cR3tY6uI0oP5aS2dF4gH9jZq"


def _file_issue(key: str, **overrides) -> dict:
    base = {
        "type": "file_issue",
        "key": key,
        "repo": REPO,
        "title": f"Title {key}",
        "body": f"Body {key}.",
    }
    base.update(overrides)
    return base


def _setup(
    tmp_path: Path,
    *,
    state: str = "active",
    arming: str = "auto",
    bounds: CharterBounds | None = None,
    standing: bool = False,
) -> tuple[Path, LocalFileGitHub, Charter]:
    """A foreman directory under ``foremen/pepe`` plus its repo's local client."""
    froot = tmp_path / "foremen" / "pepe"
    froot.mkdir(parents=True)
    save_state(froot, ForemanState(state=state))
    repo_root = tmp_path / "repo"
    issues_dir = repo_root / ".issues"
    issues_dir.mkdir(parents=True)
    gh = LocalFileGitHub(repo_root=repo_root, issues_dir=issues_dir)
    charter = Charter(
        name="pepe",
        outcome="ship the outcome",
        repos=(REPO,),
        standing=standing,
        arming=arming,
        bounds=bounds or CharterBounds(),
    )
    return froot, gh, charter


def _write_actions(froot: Path, turn_id: str, *actions: dict, summary: str = "a turn") -> None:
    turn = froot / "turns" / turn_id
    turn.mkdir(parents=True, exist_ok=True)
    (turn / "actions.json").write_text(
        json.dumps({"turn_summary": summary, "actions": list(actions)}), encoding="utf-8"
    )


def _events(froot: Path) -> list[dict]:
    return read_event_log(state_path(froot))


def _items(froot: Path):
    return load_items(froot)


def _outcomes(result, kind: str) -> list:
    return [o for o in result.value.outcomes if o.outcome == kind]


# ---------------------------------------------------------------------------
# file_issue: filing, arming, blockers, bounds, markers
# ---------------------------------------------------------------------------


def test_file_issue_files_arms_and_records(tmp_path: Path) -> None:
    froot, gh, charter = _setup(tmp_path)
    _write_actions(froot, "t1", _file_issue("a"))
    result = apply_turn(froot, "t1", charter=charter, gh=gh)
    assert result.ok, result.error
    (outcome,) = result.value.outcomes
    assert outcome.outcome == OUTCOME_APPLIED
    assert outcome.issue_number == 1

    issue = gh.issue_view(1)
    names = {label["name"] for label in issue["labels"]}
    assert names == {OWNER, READY}
    assert "Blocked by" in issue["body"]
    assert "<!-- foreman:pepe turn:t1 key:a -->" in issue["body"]

    # The hash recorded in state.json is what amend later compares against.
    assert read_state_snapshot(froot).issues[1].body_hash

    kinds = {(e["kind"], e["level"]) for e in _events(froot)}
    # Arming goes through the label seam: the ready add emits lifecycle_transition.
    assert (EVENT_ACTION_APPLIED, "info") in kinds and ("lifecycle_transition", "info") in kinds


def test_blocked_by_key_resolves_to_a_real_number_and_lints_clean(
    tmp_path: Path,
) -> None:
    froot, gh, charter = _setup(tmp_path)
    _write_actions(
        froot,
        "t1",
        _file_issue("a"),
        _file_issue("b", blocked_by=["key:a"]),
    )
    result = apply_turn(froot, "t1", charter=charter, gh=gh)
    assert result.ok
    assert [o.issue_number for o in result.value.outcomes] == [1, 2]
    body = gh.issue_view(2)["body"]
    assert "Blocked by #1" in body
    # The rendered body passes the lint contract against exactly that set.
    assert lint_issue_body(body, frozenset({1})) == ()
    assert "<!-- foreman:pepe turn:t1 key:b -->" in body


def test_blocked_by_number_ref_renders_directly(tmp_path: Path) -> None:
    froot, gh, charter = _setup(tmp_path)
    _write_actions(froot, "t1", _file_issue("a", blocked_by=["#42"]))
    result = apply_turn(froot, "t1", charter=charter, gh=gh)
    assert result.ok
    body = gh.issue_view(1)["body"]
    assert "Blocked by #42" in body
    assert lint_issue_body(body, frozenset({42})) == ()


def test_file_issues_are_ordered_blockers_first(tmp_path: Path) -> None:
    froot, gh, charter = _setup(tmp_path)
    # b is declared first but depends on a: a must still get the lower number.
    _write_actions(
        froot,
        "t1",
        _file_issue("b", blocked_by=["key:a"]),
        _file_issue("a"),
    )
    result = apply_turn(froot, "t1", charter=charter, gh=gh)
    assert result.ok
    by_key = {o.key: o for o in result.value.outcomes}
    assert by_key["a"].issue_number == 1
    assert by_key["b"].issue_number == 2


def test_blocked_by_cycle_is_refused(tmp_path: Path) -> None:
    froot, gh, charter = _setup(tmp_path)
    _write_actions(
        froot,
        "t1",
        _file_issue("a", blocked_by=["key:b"]),
        _file_issue("b", blocked_by=["key:a"]),
        _file_issue("c"),
    )
    result = apply_turn(froot, "t1", charter=charter, gh=gh)
    assert result.ok
    by_key = {o.key: o for o in result.value.outcomes}
    assert by_key["a"].outcome == OUTCOME_REFUSED
    assert "cycle" in by_key["a"].reason
    assert by_key["b"].outcome == OUTCOME_REFUSED
    # One refused action never stops the others.
    assert by_key["c"].outcome == OUTCOME_APPLIED
    assert by_key["c"].issue_number == 1


def test_self_cycle_is_refused(tmp_path: Path) -> None:
    froot, gh, charter = _setup(tmp_path)
    _write_actions(froot, "t1", _file_issue("a", blocked_by=["key:a"]))
    result = apply_turn(froot, "t1", charter=charter, gh=gh)
    assert result.ok
    (outcome,) = result.value.outcomes
    assert outcome.outcome == OUTCOME_REFUSED
    assert "cycle" in outcome.reason


def test_unknown_blocked_by_key_is_refused(tmp_path: Path) -> None:
    froot, gh, charter = _setup(tmp_path)
    _write_actions(froot, "t1", _file_issue("a", blocked_by=["key:ghost"]))
    result = apply_turn(froot, "t1", charter=charter, gh=gh)
    assert result.ok
    (outcome,) = result.value.outcomes
    assert outcome.outcome == OUTCOME_REFUSED
    assert "key:ghost" in outcome.reason


def test_refused_dependency_refuses_dependents(tmp_path: Path) -> None:
    froot, gh, charter = _setup(tmp_path)
    _write_actions(
        froot,
        "t1",
        _file_issue("a", repo="other/repo"),
        _file_issue("b", blocked_by=["key:a"]),
    )
    result = apply_turn(froot, "t1", charter=charter, gh=gh)
    assert result.ok
    by_key = {o.key: o for o in result.value.outcomes}
    assert by_key["a"].outcome == OUTCOME_REFUSED
    assert "charter" in by_key["a"].reason
    assert by_key["b"].outcome == OUTCOME_REFUSED
    assert "refused" in by_key["b"].reason
    assert gh.issue_list() == []


def test_duplicate_file_issue_keys_are_refused(tmp_path: Path) -> None:
    """Two actions sharing a key would mint colliding markers and item ids."""
    froot, gh, charter = _setup(tmp_path)
    _write_actions(froot, "t1", _file_issue("a"), _file_issue("a", title="Other"))
    result = apply_turn(froot, "t1", charter=charter, gh=gh)
    assert result.ok
    assert all(o.outcome == OUTCOME_REFUSED for o in result.value.outcomes)
    assert all("duplicate" in o.reason for o in result.value.outcomes)
    assert gh.issue_list() == []


def test_max_open_issues_bound_names_the_bound(tmp_path: Path) -> None:
    froot, gh, charter = _setup(tmp_path, bounds=CharterBounds(max_open_issues=2))
    gh.create_issue("T1", "b", labels=[OWNER])
    gh.create_issue("T2", "b", labels=[OWNER])
    _write_actions(froot, "t1", _file_issue("a"))
    result = apply_turn(froot, "t1", charter=charter, gh=gh)
    assert result.ok
    (outcome,) = result.value.outcomes
    assert outcome.outcome == OUTCOME_REFUSED
    assert "max_open_issues" in outcome.reason


def test_escalated_owned_issue_still_counts_as_open(tmp_path: Path) -> None:
    """An owned issue in Human needed is open for the bound on purpose."""
    froot, gh, charter = _setup(tmp_path, bounds=CharterBounds(max_open_issues=2))
    gh.create_issue("T1", "b", labels=[OWNER])
    escalated = gh.create_issue("T2", "b", labels=[OWNER, LabelConfig().human_needed])
    assert escalated == 2
    _write_actions(froot, "t1", _file_issue("a"))
    result = apply_turn(froot, "t1", charter=charter, gh=gh)
    assert result.ok
    (outcome,) = result.value.outcomes
    assert outcome.outcome == OUTCOME_REFUSED
    assert "max_open_issues" in outcome.reason


def test_max_issues_per_turn_bound_names_the_bound(tmp_path: Path) -> None:
    froot, gh, charter = _setup(tmp_path, bounds=CharterBounds(max_issues_per_turn=1))
    _write_actions(froot, "t1", _file_issue("a"), _file_issue("b"))
    result = apply_turn(froot, "t1", charter=charter, gh=gh)
    assert result.ok
    by_key = {o.key: o for o in result.value.outcomes}
    assert by_key["a"].outcome == OUTCOME_APPLIED
    assert by_key["b"].outcome == OUTCOME_REFUSED
    assert "max_issues_per_turn" in by_key["b"].reason


def test_model_body_lint_refusal(tmp_path: Path) -> None:
    """A stray dependency declaration in the model's own body is refused."""
    froot, gh, charter = _setup(tmp_path)
    _write_actions(froot, "t1", _file_issue("a", body="Depends on #99 first."))
    result = apply_turn(froot, "t1", charter=charter, gh=gh)
    assert result.ok
    (outcome,) = result.value.outcomes
    assert outcome.outcome == OUTCOME_REFUSED
    assert "issue contract" in outcome.reason
    assert gh.issue_list() == []


def test_outbound_secret_refusal_is_a_refused_action(tmp_path: Path) -> None:
    """A guard match is a refused action, not an exception (acceptance)."""
    froot, gh, charter = _setup(tmp_path)
    _write_actions(froot, "t1", _file_issue("a", body=f"token: {_TOKEN}"))
    result = apply_turn(froot, "t1", charter=charter, gh=gh)
    assert result.ok, result.error
    (outcome,) = result.value.outcomes
    assert outcome.outcome == OUTCOME_REFUSED
    assert "credential" in outcome.reason
    assert gh.issue_list() == []


def test_private_slug_refusal(tmp_path: Path) -> None:
    froot, gh, charter = _setup(tmp_path)
    slugs_dir = gh.repo_root / ".private-slug-baseline" / "slugs"
    slugs_dir.mkdir(parents=True)
    (slugs_dir / "acme-secret").write_text("internal codename", encoding="utf-8")
    _write_actions(froot, "t1", _file_issue("a", title="Roll out acme-secret"))
    result = apply_turn(froot, "t1", charter=charter, gh=gh)
    assert result.ok
    (outcome,) = result.value.outcomes
    assert outcome.outcome == OUTCOME_REFUSED
    assert "private slug" in outcome.reason
    assert gh.issue_list() == []


# ---------------------------------------------------------------------------
# Arming
# ---------------------------------------------------------------------------


def test_arming_propose_leaves_unarmed_with_an_item(tmp_path: Path) -> None:
    froot, gh, charter = _setup(tmp_path, arming="propose")
    _write_actions(froot, "t1", _file_issue("a"))
    result = apply_turn(froot, "t1", charter=charter, gh=gh)
    assert result.ok
    (outcome,) = result.value.outcomes
    assert outcome.outcome == OUTCOME_APPLIED
    names = {label["name"] for label in gh.issue_view(1)["labels"]}
    assert names == {OWNER}
    items = _items(froot)
    assert len(items) == 1
    assert items[0].kind == "decision"
    assert READY in (items[0].command or "")


def test_arming_degraded_leaves_unarmed_with_an_item(tmp_path: Path) -> None:
    froot, gh, charter = _setup(tmp_path, state="degraded")
    _write_actions(froot, "t1", _file_issue("a"))
    result = apply_turn(froot, "t1", charter=charter, gh=gh)
    assert result.ok
    (outcome,) = result.value.outcomes
    assert outcome.outcome == OUTCOME_APPLIED
    names = {label["name"] for label in gh.issue_view(1)["labels"]}
    assert names == {OWNER}
    assert len(_items(froot)) == 1


def test_signal_marked_turn_files_unarmed(tmp_path: Path) -> None:
    froot, gh, charter = _setup(tmp_path)
    _write_actions(froot, "t1", _file_issue("a"))
    result = apply_turn(froot, "t1", charter=charter, gh=gh, signal_marked=True)
    assert result.ok
    (outcome,) = result.value.outcomes
    assert outcome.outcome == OUTCOME_APPLIED
    assert "signal-marked" in outcome.detail
    names = {label["name"] for label in gh.issue_view(1)["labels"]}
    assert names == {OWNER}
    assert len(_items(froot)) == 1


def test_arming_item_command_names_the_ready_label(tmp_path: Path) -> None:
    froot, gh, charter = _setup(tmp_path, arming="propose")
    _write_actions(froot, "t1", _file_issue("a"))
    apply_turn(froot, "t1", charter=charter, gh=gh)
    (item,) = _items(froot)
    assert item.command == f'gh issue edit 1 --add-label "{READY}"'


# ---------------------------------------------------------------------------
# amend / withdraw / comment authority
# ---------------------------------------------------------------------------


def _file_authored(froot: Path, gh: LocalFileGitHub, charter: Charter) -> int:
    _write_actions(froot, "t0", _file_issue("seed", title="Seed", body="Seed body."))
    result = apply_turn(froot, "t0", charter=charter, gh=gh)
    assert result.value.outcomes[0].outcome == OUTCOME_APPLIED
    return result.value.outcomes[0].issue_number


def test_amend_authored_issue_succeeds_and_preserves_marker(tmp_path: Path) -> None:
    froot, gh, charter = _setup(tmp_path)
    number = _file_authored(froot, gh, charter)
    _write_actions(
        froot, "t1", {"type": "amend_issue", "repo": REPO, "number": number, "body": "New body."}
    )
    result = apply_turn(froot, "t1", charter=charter, gh=gh)
    assert result.ok
    (outcome,) = result.value.outcomes
    assert outcome.outcome == OUTCOME_APPLIED
    body = gh.issue_view(number)["body"]
    assert "New body." in body
    assert "<!-- foreman:pepe turn:t0 key:seed -->" in body
    # The new live body is the hash any later amend must match.
    assert read_state_snapshot(froot).issues[number].body_hash


def test_amend_refused_on_tracked_issue(tmp_path: Path) -> None:
    """A tracked issue (label added by hand, no marker) can never be amended."""
    froot, gh, charter = _setup(tmp_path)
    tracked = gh.create_issue("Tracked", "hand-labelled", labels=[OWNER])
    _write_actions(
        froot, "t1", {"type": "amend_issue", "repo": REPO, "number": tracked, "body": "x"}
    )
    result = apply_turn(froot, "t1", charter=charter, gh=gh)
    assert result.ok
    (outcome,) = result.value.outcomes
    assert outcome.outcome == OUTCOME_REFUSED
    assert "marker" in outcome.reason


def test_amend_refused_after_operator_edit(tmp_path: Path) -> None:
    """The live body hashing differently than recorded means an operator edit."""
    froot, gh, charter = _setup(tmp_path)
    number = _file_authored(froot, gh, charter)
    # An operator edit keeps the hidden marker (it is invisible to them) but
    # changes the body — the recorded hash no longer matches.
    operator_body = gh.issue_view(number)["body"] + "\n\nOperator's own note."
    assert gh.edit_issue_body(number, operator_body)
    _write_actions(
        froot, "t1", {"type": "amend_issue", "repo": REPO, "number": number, "body": "x"}
    )
    result = apply_turn(froot, "t1", charter=charter, gh=gh)
    assert result.ok
    (outcome,) = result.value.outcomes
    assert outcome.outcome == OUTCOME_REFUSED
    assert "hash" in outcome.reason
    assert gh.issue_view(number)["body"] == operator_body


def test_amend_refused_in_active_or_terminal_state(tmp_path: Path) -> None:
    froot, gh, charter = _setup(tmp_path)
    number = _file_authored(froot, gh, charter)
    gh.add_issue_label(number, LabelConfig().in_progress)
    _write_actions(
        froot, "t1", {"type": "amend_issue", "repo": REPO, "number": number, "body": "x"}
    )
    result = apply_turn(froot, "t1", charter=charter, gh=gh)
    assert result.ok
    (outcome,) = result.value.outcomes
    assert outcome.outcome == OUTCOME_REFUSED
    assert "active/terminal" in outcome.reason


def test_withdraw_removes_ready_comments_and_closes(tmp_path: Path) -> None:
    froot, gh, charter = _setup(tmp_path)
    number = _file_authored(froot, gh, charter)
    _write_actions(
        froot,
        "t1",
        {"type": "withdraw_issue", "repo": REPO, "number": number, "reason": "obsolete"},
    )
    result = apply_turn(froot, "t1", charter=charter, gh=gh)
    assert result.ok
    (outcome,) = result.value.outcomes
    assert outcome.outcome == OUTCOME_APPLIED
    issue = gh.issue_view(number)
    assert issue["state"] == "CLOSED"
    names = {label["name"] for label in issue["labels"]}
    assert READY not in names
    assert any("obsolete" in c["body"] for c in issue["comments"])


def test_withdraw_refused_on_tracked_issue(tmp_path: Path) -> None:
    froot, gh, charter = _setup(tmp_path)
    tracked = gh.create_issue("Tracked", "hand-labelled", labels=[OWNER])
    _write_actions(
        froot,
        "t1",
        {"type": "withdraw_issue", "repo": REPO, "number": tracked, "reason": "x"},
    )
    result = apply_turn(froot, "t1", charter=charter, gh=gh)
    assert result.ok
    (outcome,) = result.value.outcomes
    assert outcome.outcome == OUTCOME_REFUSED
    assert gh.issue_view(tracked)["state"] == "OPEN"


def test_comment_on_owned_issue(tmp_path: Path) -> None:
    froot, gh, charter = _setup(tmp_path)
    tracked = gh.create_issue("Tracked", "hand-labelled", labels=[OWNER])
    _write_actions(
        froot, "t1", {"type": "comment_issue", "repo": REPO, "number": tracked, "body": "hi"}
    )
    result = apply_turn(froot, "t1", charter=charter, gh=gh)
    assert result.ok
    (outcome,) = result.value.outcomes
    assert outcome.outcome == OUTCOME_APPLIED
    comments = gh.issue_view(tracked)["comments"]
    assert any("hi" in c["body"] for c in comments)


def test_comment_refused_without_ownership_label(tmp_path: Path) -> None:
    froot, gh, charter = _setup(tmp_path)
    foreign = gh.create_issue("Foreign", "not ours", labels=[])
    _write_actions(
        froot, "t1", {"type": "comment_issue", "repo": REPO, "number": foreign, "body": "hi"}
    )
    result = apply_turn(froot, "t1", charter=charter, gh=gh)
    assert result.ok
    (outcome,) = result.value.outcomes
    assert outcome.outcome == OUTCOME_REFUSED
    assert "ownership label" in outcome.reason


def test_comment_deduped_on_replay(tmp_path: Path) -> None:
    froot, gh, charter = _setup(tmp_path)
    number = _file_authored(froot, gh, charter)
    _write_actions(
        froot, "t1", {"type": "comment_issue", "repo": REPO, "number": number, "body": "ping"}
    )
    apply_turn(froot, "t1", charter=charter, gh=gh)
    assert len(gh.issue_view(number)["comments"]) == 1
    apply_turn(froot, "t1", charter=charter, gh=gh)
    assert len(gh.issue_view(number)["comments"]) == 1


# ---------------------------------------------------------------------------
# raise_item, memory, propose_done
# ---------------------------------------------------------------------------


def test_raise_item_decision_and_report(tmp_path: Path) -> None:
    froot, gh, charter = _setup(tmp_path)
    _write_actions(
        froot,
        "t1",
        {
            "type": "raise_item",
            "key": "d1",
            "kind": "decision",
            "question": "Pick one",
            "options": ["x", "y"],
            "recommended": 0,
        },
        {"type": "raise_item", "key": "r1", "kind": "report", "question": "FYI body"},
        {
            "type": "raise_item",
            "key": "bad",
            "kind": "decision",
            "question": "no options",
        },
    )
    result = apply_turn(froot, "t1", charter=charter, gh=gh)
    assert result.ok
    outcomes = result.value.outcomes
    assert outcomes[0].outcome == OUTCOME_APPLIED
    assert outcomes[1].outcome == OUTCOME_APPLIED
    # A decision item without options is a type-specific refusal, and the
    # other actions still applied.
    assert outcomes[2].outcome == OUTCOME_REFUSED
    items = {i.id: i for i in _items(froot)}
    assert items["t1:d1"].kind == "decision"
    assert items["t1:d1"].options == ("x", "y")
    assert items["t1:r1"].kind == "report"
    assert items["t1:r1"].text == "FYI body"


def test_write_and_delete_memory(tmp_path: Path) -> None:
    froot, gh, charter = _setup(tmp_path)
    _write_actions(
        froot,
        "t1",
        {"type": "write_memory", "path": "notes", "content": "remember"},
        {"type": "delete_memory", "path": "ghost"},
    )
    result = apply_turn(froot, "t1", charter=charter, gh=gh)
    assert result.ok
    assert all(o.outcome == OUTCOME_APPLIED for o in result.value.outcomes)
    assert (froot / "memory" / "notes.md").read_text() == "remember"


def test_write_memory_escape_is_refused(tmp_path: Path) -> None:
    """Memory authority: the resolved path must stay inside memory/."""
    froot, gh, charter = _setup(tmp_path)
    _write_actions(
        froot,
        "t1",
        {"type": "write_memory", "path": "../charter.yaml", "content": "name: hacked"},
        {"type": "write_memory", "path": "sub/dir", "content": "x"},
    )
    result = apply_turn(froot, "t1", charter=charter, gh=gh)
    assert result.ok
    assert all(o.outcome == OUTCOME_REFUSED for o in result.value.outcomes)
    assert not (froot / "charter.yaml").exists()


def test_propose_done_raises_a_decision_item(tmp_path: Path) -> None:
    froot, gh, charter = _setup(tmp_path)
    _write_actions(froot, "t1", {"type": "propose_done", "evidence": "all landed"})
    result = apply_turn(froot, "t1", charter=charter, gh=gh)
    assert result.ok
    (outcome,) = result.value.outcomes
    assert outcome.outcome == OUTCOME_APPLIED
    (item,) = _items(froot)
    assert item.kind == "decision"
    assert "all landed" in (item.text or "")
    assert len(item.options) == 2


def test_propose_done_refused_when_standing(tmp_path: Path) -> None:
    froot, gh, charter = _setup(tmp_path, standing=True)
    _write_actions(froot, "t1", {"type": "propose_done", "evidence": "x"})
    result = apply_turn(froot, "t1", charter=charter, gh=gh)
    assert result.ok
    (outcome,) = result.value.outcomes
    assert outcome.outcome == OUTCOME_REFUSED
    assert "standing" in outcome.reason
    assert _items(froot) == ()


# ---------------------------------------------------------------------------
# Pause, dry-run, replay, crash resume
# ---------------------------------------------------------------------------


def test_paused_foreman_skips_every_action(tmp_path: Path) -> None:
    froot, gh, charter = _setup(tmp_path, state="paused")
    _write_actions(froot, "t1", _file_issue("a"))
    result = apply_turn(froot, "t1", charter=charter, gh=gh)
    assert result.ok
    assert result.value.stopped_by_pause
    (outcome,) = result.value.outcomes
    assert outcome.outcome == OUTCOME_SKIPPED
    assert outcome.reason == "foreman paused"
    assert gh.issue_list() == []
    # Nothing was processed, so nothing was recorded: a resume applies it.
    assert not applied_path(froot, "t1").exists()


def test_pause_mid_turn_stops_remaining_actions(tmp_path: Path, monkeypatch) -> None:
    froot, gh, charter = _setup(tmp_path)
    _write_actions(froot, "t1", _file_issue("a"), _file_issue("b"))
    original = apply_mod.read_state_snapshot
    calls = {"n": 0}

    def paused_after_first_check(root: Path) -> ForemanState:
        calls["n"] += 1
        state = original(root)
        # First call is the main loop's before-action-0 check; every later
        # call on this binding reports paused and stops the next loop check.
        # Inside-action helpers bind their own copy — never gated on it.
        return ForemanState(state="paused") if calls["n"] > 1 else state

    monkeypatch.setattr(apply_mod, "read_state_snapshot", paused_after_first_check)
    result = apply_turn(froot, "t1", charter=charter, gh=gh)
    assert result.ok
    assert result.value.stopped_by_pause
    outcomes = result.value.outcomes
    assert outcomes[0].outcome == OUTCOME_APPLIED
    assert outcomes[1].outcome == OUTCOME_SKIPPED
    assert len(gh.issue_list()) == 1


def test_dry_run_checks_everything_and_writes_nothing(tmp_path: Path) -> None:
    froot, gh, charter = _setup(tmp_path)
    _write_actions(
        froot,
        "t1",
        _file_issue("a"),
        {"type": "raise_item", "key": "q", "kind": "report", "question": "r"},
        {"type": "write_memory", "path": "m", "content": "x"},
    )
    result = apply_turn(froot, "t1", charter=charter, gh=gh, dry_run=True)
    assert result.ok
    assert result.value.dry_run
    details = [o.detail or "" for o in result.value.outcomes]
    assert all(o.outcome == OUTCOME_APPLIED for o in result.value.outcomes)
    assert any("would file" in d for d in details)
    assert any("would raise" in d for d in details)
    assert any("would write memory" in d for d in details)
    assert gh.issue_list() == []
    assert _items(froot) == ()
    assert not (froot / "memory").exists()
    assert not applied_path(froot, "t1").exists()
    assert _events(froot) == []


def test_replay_skips_recorded_actions(tmp_path: Path) -> None:
    froot, gh, charter = _setup(tmp_path)
    _write_actions(froot, "t1", _file_issue("a"), _file_issue("b"))
    apply_turn(froot, "t1", charter=charter, gh=gh)
    result = apply_turn(froot, "t1", charter=charter, gh=gh)
    assert result.ok
    assert all(o.outcome == OUTCOME_SKIPPED for o in result.value.outcomes)
    assert len(gh.issue_list()) == 2


def test_crash_between_create_and_record_files_exactly_once(tmp_path: Path, monkeypatch) -> None:
    """A crash after the issue lands but before the record must not re-file."""
    froot, gh, charter = _setup(tmp_path)
    _write_actions(
        froot,
        "t1",
        _file_issue("a"),
        _file_issue("b", blocked_by=["key:a"]),
    )

    def crash_on_record(path: Path, record: dict):
        raise RuntimeError("simulated crash between effect and record")

    monkeypatch.setattr(apply_mod, "_append", crash_on_record)
    with pytest.raises(RuntimeError):
        apply_turn(froot, "t1", charter=charter, gh=gh)
    assert len(gh.issue_list()) == 1

    monkeypatch.undo()
    result = apply_turn(froot, "t1", charter=charter, gh=gh)
    assert result.ok
    by_key = {o.key: o for o in result.value.outcomes}
    assert by_key["a"].outcome == OUTCOME_APPLIED
    assert "reused" in by_key["a"].detail
    # The dependent still resolves the dep's number — exactly one issue #1.
    assert by_key["b"].outcome == OUTCOME_APPLIED
    assert "Blocked by #1" in gh.issue_view(2)["body"]
    assert len(gh.issue_list()) == 2


def test_property_apply_twice_equals_apply_once(tmp_path: Path) -> None:
    """Applying a valid actions.json twice yields the same effects as once."""
    froot, gh, charter = _setup(tmp_path)
    _write_actions(
        froot,
        "t1",
        _file_issue("a"),
        _file_issue("b", blocked_by=["key:a"]),
        {"type": "comment_issue", "repo": REPO, "number": 1, "body": "note"},
        {"type": "raise_item", "key": "q", "kind": "report", "question": "r"},
        {"type": "write_memory", "path": "m", "content": "x"},
        {"type": "propose_done", "evidence": "done-ish"},
        _file_issue("bad", repo="foreign/repo"),
    )

    def snapshot() -> dict[str, bytes]:
        seen: dict[str, bytes] = {}
        for root in (gh.issues_dir, froot / "memory"):
            if root.is_dir():
                for path in sorted(root.rglob("*")):
                    if path.is_file():
                        seen[str(path.relative_to(tmp_path))] = path.read_bytes()
        for name in ("items.json", "state.json"):
            path = froot / name
            if path.exists():
                seen[name] = path.read_bytes()
        applied = applied_path(froot, "t1")
        if applied.exists():
            seen["applied.jsonl"] = applied.read_bytes()
        return seen

    first = apply_turn(froot, "t1", charter=charter, gh=gh)
    assert first.ok
    before = snapshot()
    second = apply_turn(froot, "t1", charter=charter, gh=gh)
    assert second.ok
    after = snapshot()
    assert before == after


def test_refused_action_emits_warning_event(tmp_path: Path) -> None:
    froot, gh, charter = _setup(tmp_path)
    _write_actions(froot, "t1", _file_issue("a", repo="foreign/repo"))
    result = apply_turn(froot, "t1", charter=charter, gh=gh)
    assert result.ok
    kinds = [(e["kind"], e["level"]) for e in _events(froot)]
    assert (EVENT_ACTION_REFUSED, "warning") in kinds


def test_schema_failure_fails_the_whole_turn(tmp_path: Path) -> None:
    froot, gh, charter = _setup(tmp_path)
    turn = froot / "turns" / "t1"
    turn.mkdir(parents=True)
    (turn / "actions.json").write_text(
        json.dumps({"turn_summary": "s", "actions": [{"type": "teleport"}]}),
        encoding="utf-8",
    )
    result = apply_turn(froot, "t1", charter=charter, gh=gh)
    assert not result.ok
    assert "failed schema" in result.error
    assert gh.issue_list() == []
    assert not applied_path(froot, "t1").exists()


def test_missing_actions_json_is_an_error_value(tmp_path: Path) -> None:
    froot, gh, charter = _setup(tmp_path)
    result = apply_turn(froot, "t1", charter=charter, gh=gh)
    assert not result.ok
    assert "actions.json" in result.error


def test_jabroni_display_label() -> None:
    from charlie_work.foreman import display_action_outcome

    assert display_action_outcome("refused") == "jabroni"
    assert display_action_outcome("applied") == "applied"
    assert display_action_outcome("skipped") == "skipped"
