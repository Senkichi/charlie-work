"""Characterisation of what ``record_review`` and ``record_local_review`` do today that
neither ``test_verdict_parity.py`` nor the per-path suites pin (issue #3102).

Written before the shared recording mechanics were extracted, and passing on the
unextracted source: refusal wording verbatim, the Local-only decision fields, comment
post failures, non-string ``required_changes``, ``pr_dir`` after each refusal class,
the PR terminal refusal taking no state lock, a failed ``pr_view`` with a packet, a
corrupt prior decision file, and the ``authorized_override`` carry-forward. The scenario
table and its declared differences are untouched; this file only adds observations.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import charlie_work.selection_wrapper as selection_wrapper
from _verdict_parity_driver import (
    ISSUE,
    Scenario,
    _call_args,
    _local_app,
    _make_repo,
    _pin_worker_selection,
    _remote_app,
    _seed,
    _write_packet,
)
import charlie_work.workflow as workflow
from charlie_work.github import GitHubError
from charlie_work.labels import TransitionOutcome, TransitionResult
from charlie_work.state import load_state, save_state, state_lock
from charlie_work.workflow import VERDICT_PROVENANCE_VALUES
from charlie_work.workflow import OrchestratorApp


@dataclass
class Side:
    app: OrchestratorApp
    gh: Any
    root: Path
    repo: Any

    @property
    def pr_dir(self) -> Path:
        return self.app.paths.prs / f"pr-{ISSUE}"

    def call(self, scenario: Scenario, **overrides: Any) -> Any:
        args = {**_call_args(self.app, scenario, self.repo), **overrides}
        if self.is_local:
            return self.app.record_local_review(**args)
        return self.app.record_review(**args)

    is_local: bool = False


def sides(scenario: Scenario, *, mutate_state=None):
    repo = _make_repo()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(selection_wrapper, "ci_fleet_executable", lambda: Path("ci-fleet"))
        local_app = _local_app(scenario, repo)
        _pin_worker_selection(repo.root)
        _seed(local_app, scenario, repo, local=True)
        _write_packet(local_app, scenario, repo)
        root = Path(tempfile.mkdtemp(prefix="cw-verdict-pr-"))
        remote_app, gh = _remote_app(scenario, repo, root)
        _pin_worker_selection(root)
        _seed(remote_app, scenario, repo, local=False)
        _write_packet(remote_app, scenario, repo)
    local = Side(local_app, local_app.gh, repo.root, repo)
    local.is_local = True
    remote = Side(remote_app, gh, root, repo)
    return remote, local


def _named(message: str, side: Side) -> str:
    for name, sha in (("HEAD", side.repo.head), ("OLDER", side.repo.older_head)):
        message = message.replace(sha, name)
    return message


def _refusal(scenario: Scenario, **overrides: Any) -> tuple[tuple[str, str], tuple[str, str]]:
    remote, local = sides(scenario)
    out = []
    for side in (remote, local):
        result = side.call(scenario, **overrides)
        assert result.ok is False
        out.append((_named(result.message, side), result.data["reason"]))
    return out[0], out[1]


_PROVENANCES = ", ".join(sorted(VERDICT_PROVENANCE_VALUES))
_MOVED = "verdict not recorded — head moved during build, will re-review next pass"


@pytest.mark.parametrize(
    ("scenario", "remote", "local"),
    [
        (
            Scenario("invalid_decision", "maybe"),
            "decision must be approved, request_changes, or blocked",
            "invalid decision: maybe",
        ),
        (
            Scenario("invalid_provenance", "approved", provenance="made_up"),
            f"verdict_provenance must be one of: {_PROVENANCES}",
            "invalid verdict provenance: made_up",
        ),
        (
            Scenario("missing_summary", "blocked", summary="  "),
            "--summary or --summary-file is required for decision 'blocked'",
            "--summary or --summary-file is required for decision 'blocked'",
        ),
        (
            Scenario("merged", "approved", pr_status="merged", packet="none"),
            "PR #7 is MERGED; verdict not recorded (terminal-state PR decision file is preserved)",
            "local record 7 is already merged; verdict not recorded",
        ),
        (
            Scenario("closed", "approved", pr_status="closed", packet="none"),
            "PR #7 is CLOSED; verdict not recorded (terminal-state PR decision file is preserved)",
            "local record 7 is already closed; verdict not recorded",
        ),
        (
            Scenario("escalated", "approved", pr_status="escalated", packet="none"),
            "PR #7 is escalated; verdict not recorded (run `charlie unescalate --pr 7` to "
            "re-arm it first)",
            "local record 7 is escalated; verdict not recorded (unescalate the issue first)",
        ),
        (
            Scenario("moved_with_head", "approved", packet="stale", reviewed_head="packet"),
            "reviewed head (OLDER) matches packet head (OLDER) but live PR head has moved to "
            f"(HEAD); {_MOVED}",
            "reviewed head (OLDER) matches packet head (OLDER) but live branch head has moved "
            f"to (HEAD); {_MOVED}",
        ),
        (
            Scenario("moved", "approved", packet="stale"),
            "review packet head (OLDER) differs from live PR head (HEAD); use --reviewed-head "
            "to choose the head the verdict applies to",
            "review packet head (OLDER) differs from live branch head (HEAD); pass "
            "reviewed_head to choose the head the verdict applies to",
        ),
        (
            Scenario("mismatch", "approved", reviewed_head="bogus"),
            f"--reviewed-head {'f' * 40} does not match packet head HEAD or live head HEAD",
            f"reviewed-head {'f' * 40} does not match packet head HEAD or live head HEAD",
        ),
        (
            Scenario("mismatch_no_packet", "approved", reviewed_head="bogus", packet="none"),
            f"--reviewed-head {'f' * 40} does not match live head HEAD",
            f"reviewed-head {'f' * 40} does not match live head HEAD",
        ),
    ],
    ids=lambda value: value.name if isinstance(value, Scenario) else None,
)
def test_refusal_wording_is_verbatim_on_both_entry_points(
    scenario: Scenario, remote: str, local: str
) -> None:
    (remote_message, remote_reason), (local_message, local_reason) = _refusal(scenario)
    assert remote_message == remote
    assert local_message == local
    assert remote_reason == local_reason


def test_no_head_available_wording() -> None:
    scenario = Scenario("no_head", "approved", packet="none")
    remote, local = sides(scenario)
    remote.gh.pr_view = lambda *args, **kwargs: {}  # a failed fetch: no live head
    with state_lock(local.app.paths.state_file):
        state = load_state(local.app.paths.state_file)
        state["prs"][str(ISSUE)].pop("branch")
        save_state(local.app.paths.state_file, state)
    results = [side.call(scenario) for side in (remote, local)]
    assert [(r.ok, r.message, r.data) for r in results] == [
        (False, "no packet or live PR head available", {"reason": "no_head_available"}),
        (False, "no packet or live branch head available", {"reason": "no_head_available"}),
    ]


def test_not_local_record_refusal_on_the_local_entry_point() -> None:
    scenario = Scenario("not_local", "approved", packet="none")
    _, local = sides(scenario)
    with state_lock(local.app.paths.state_file):
        state = load_state(local.app.paths.state_file)
        state["prs"][str(ISSUE)].pop("local")
        save_state(local.app.paths.state_file, state)
    result = local.call(scenario)
    assert (result.ok, result.message, result.data) == (
        False,
        "prs[7] is not a local-lane record",
        {"pr": 7, "reason": "not_local_record"},
    )
    assert local.pr_dir.is_dir()  # created before the guard


@pytest.mark.parametrize(
    ("scenario", "remote_dir", "local_dir"),
    [
        (Scenario("a", "maybe", packet="none"), False, False),
        (Scenario("b", "approved", provenance="made_up", packet="none"), False, False),
        (Scenario("c", "blocked", summary=" ", packet="none"), False, False),
        (Scenario("d", "approved", pr_status="merged", packet="none"), False, True),
        (Scenario("e", "approved", pr_status="escalated", packet="none"), False, True),
        (Scenario("f", "approved", packet="stale"), True, True),
    ],
    ids=[
        "invalid_decision",
        "invalid_provenance",
        "missing_summary",
        "terminal",
        "escalated",
        "head_moved",
    ],
)
def test_pr_dir_after_each_refusal_class(
    scenario: Scenario, remote_dir: bool, local_dir: bool
) -> None:
    remote, local = sides(scenario)
    for side in (remote, local):
        assert side.call(scenario).ok is False
    assert (remote.pr_dir.exists(), local.pr_dir.exists()) == (remote_dir, local_dir)


def test_pr_terminal_refusal_takes_no_state_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Path] = []
    real = workflow.state_lock

    def counting(path: Path, *args: Any, **kwargs: Any) -> Any:
        calls.append(path)
        return real(path, *args, **kwargs)

    monkeypatch.setattr(workflow, "state_lock", counting)
    merged = Scenario("merged", "approved", pr_status="merged", packet="none")
    remote, _ = sides(merged)
    assert remote.call(merged).data["reason"] == "pr_terminal_state"
    assert calls == []
    live = Scenario("live", "approved")
    remote, _ = sides(live)
    assert remote.call(live).ok is True
    assert calls  # the control: a recorded verdict does take the lock


def test_failed_pr_view_with_a_packet_records_an_empty_patch_id() -> None:
    scenario = Scenario("no_pr", "approved")
    remote, _ = sides(scenario)
    remote.gh.pr_view = lambda *args, **kwargs: {}
    result = remote.call(scenario)
    assert result.ok is True
    assert result.data["reviewed_head_source"] == "packet"
    decision = json.loads((remote.pr_dir / "review-decision.json").read_text("utf-8"))
    assert decision["reviewed_patch_id"] == ""
    assert decision["reviewed_changed_lines"] == []
    assert decision["reviewed_changed_files"] == []
    assert decision["reviewed_body_sha256"] == ""
    assert decision["issue_number"] is None


def _decision(side: Side) -> dict[str, Any]:
    return json.loads((side.pr_dir / "review-decision.json").read_text("utf-8"))


def test_local_only_decision_fields_and_their_absence_on_the_pr_path() -> None:
    finding = "This needs a human decision before it can ship."
    scenario = Scenario(
        "reclassified", "request_changes", summary="For the operator.", required_changes=(finding,)
    )
    remote, local = sides(scenario)
    for side in (remote, local):
        assert side.call(scenario).ok is True
    remote_decision, local_decision = _decision(remote), _decision(local)
    assert local_decision["local"] is True
    assert local_decision["human_call_reclassified_from"] == "request_changes"
    marker = local_decision["human_call_marker"]
    assert set(marker) == {"item", "marker"}
    assert marker["item"] == finding
    assert marker["marker"]
    for key in ("local", "human_call_reclassified_from", "human_call_marker"):
        assert key not in remote_decision
    plain = Scenario("plain", "approved")
    remote, local = sides(plain)
    for side in (remote, local):
        side.call(plain)
    assert _decision(local)["local"] is True
    assert "human_call_reclassified_from" not in _decision(local)
    assert "human_call_marker" not in _decision(local)


def _fail_labels(monkeypatch: pytest.MonkeyPatch, side: Side) -> None:
    failed = TransitionResult(TransitionOutcome.PARTIAL_FAILURE, [(ISSUE, "x")], [])
    if side.is_local:
        monkeypatch.setattr(type(side.app.write_gate), "transition", lambda *a, **k: failed)
    else:
        monkeypatch.setattr(workflow, "transition", lambda *a, **k: failed)


@pytest.mark.parametrize("fail_labels", [False, True])
def test_comment_post_failure_is_reported_on_both_entry_points(
    monkeypatch: pytest.MonkeyPatch, fail_labels: bool
) -> None:
    scenario = Scenario("approve", "approved")
    remote, local = sides(scenario)

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise GitHubError("comment boom")

    monkeypatch.setattr(remote.app, "_comment_pr", refuse)
    monkeypatch.setattr(type(local.gh), "issue_comment", refuse)
    for side in (remote, local):
        if fail_labels:
            _fail_labels(monkeypatch, side)
        result = side.call(scenario)
        assert result.ok is True
        label_error = result.data["label_error"]
        assert label_error["comment_error"] == "comment boom"
        if fail_labels:
            assert label_error["outcome"] == "partial_failure"
            assert result.message.endswith(" (label update failed: partial_failure)")
        else:
            assert label_error == {"comment_error": "comment boom"}
            assert result.message.endswith(
                " (label update failed: {'comment_error': 'comment boom'})"
            )


def test_local_comment_is_written_to_verdict_comment_md_and_catches_oserror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = Scenario(
        "rework", "request_changes", required_changes=("Add a test.",), summary="Needs work."
    )
    _, local = sides(scenario)
    assert local.call(scenario).ok is True
    body = (local.pr_dir / "verdict-comment.md").read_text("utf-8")
    assert body == (
        "## Fleet review - round 1 - request_changes\n\nNeeds work.\n\n"
        "### Required changes\n- Add a test."
    )
    _, local = sides(scenario)

    def disk_full(*args: Any, **kwargs: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(type(local.gh), "issue_comment", disk_full)
    result = local.call(scenario)
    assert result.ok is True
    assert result.data["label_error"] == {"comment_error": "disk full"}


def test_derived_findings_comment_shows_the_summary_once_on_the_local_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No explicit required changes: the findings are derived from the summary, so a
    # "Required changes" section would repeat it (issue #3104).
    scenario = Scenario("derived", "request_changes", summary="Fix the null check.")
    _, local = sides(scenario)
    posted: list[str] = []

    def capture(self: Any, issue_number: int, body_path: Path) -> None:
        posted.append(Path(body_path).read_text("utf-8"))

    monkeypatch.setattr(type(local.gh), "issue_comment", capture)
    assert local.call(scenario).ok is True
    assert _decision(local)["findings_channel"] == "derived"
    body = (local.pr_dir / "verdict-comment.md").read_text("utf-8")
    assert body == "## Fleet review - round 1 - request_changes\n\nFix the null check."
    assert posted == [body]


def test_explicit_required_changes_are_listed_on_both_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = Scenario(
        "explicit", "request_changes", summary="Needs work.", required_changes=("Add a test.",)
    )
    remote, local = sides(scenario)
    posted: list[str] = []
    monkeypatch.setattr(remote.app, "_comment_pr", lambda pr, body, *a, **k: posted.append(body))
    assert remote.call(scenario).ok is True
    assert local.call(scenario).ok is True
    for side in (remote, local):
        assert _decision(side).get("findings_channel") != "derived"
    local_body = (local.pr_dir / "verdict-comment.md").read_text("utf-8")
    assert local_body.endswith("### Required changes\n- Add a test.")
    assert len(posted) == 1
    assert posted[0].endswith("### Required changes\n- Add a test.")


def test_non_string_required_changes_items() -> None:
    scenario = Scenario("rework", "request_changes", summary="Needs work.")
    remote, local = sides(scenario)
    remote_result = remote.call(scenario, required_changes=[42, "Add a test."])
    local_result = local.call(scenario, required_changes=[42, "Add a test."])
    assert remote_result.ok and local_result.ok
    assert _decision(remote)["required_changes"] == [42, "Add a test."]
    assert _decision(local)["required_changes"] == ["42", "Add a test."]
    for side in (remote, local):
        events = load_state(side.app.paths.state_file)["events"]
        recorded = [e for e in events if e["kind"] == "record_review"][-1]
        assert recorded["payload"]["required_changes"] == "42\nAdd a test."


def test_corrupt_prior_decision_file_is_replaced_without_an_override() -> None:
    scenario = Scenario("approve", "approved")
    remote, local = sides(scenario)
    for side in (remote, local):
        side.pr_dir.mkdir(parents=True, exist_ok=True)
        (side.pr_dir / "review-decision.json").write_text("{not json", encoding="utf-8")
        assert side.call(scenario).ok is True
        decision = _decision(side)
        assert decision["decision"] == "approved"
        assert "authorized_override" not in decision


@pytest.mark.parametrize(
    ("override", "carried"),
    [({"by": "operator", "head": "abc"}, True), ("not-a-dict", False)],
)
def test_authorized_override_carry_forward(override: Any, carried: bool) -> None:
    scenario = Scenario("approve", "approved")
    remote, local = sides(scenario)
    for side in (remote, local):
        side.pr_dir.mkdir(parents=True, exist_ok=True)
        (side.pr_dir / "review-decision.json").write_text(
            json.dumps({"decision": "approved", "authorized_override": override}), encoding="utf-8"
        )
        assert side.call(scenario).ok is True
        decision = _decision(side)
        if carried:
            assert decision["authorized_override"] == override
        else:
            assert "authorized_override" not in decision
