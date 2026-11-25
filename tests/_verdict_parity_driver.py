"""Driver for ``test_verdict_parity.py``: runs one scenario through both Verdict
entry points and flattens each outcome to ``path -> value`` for comparison."""

from __future__ import annotations

import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from _fakes_github import FakeGitHub
from _local_lane_fixtures import (
    _app,
    _git,
    _init_repo,
    _local_config,
    _parked_issue,
    new_repo_root,
)
import charlie_work.selection_wrapper as selection_wrapper
from charlie_work.config import build_config_from_data
from charlie_work.instrumentation import query_events
from charlie_work.labels import transition
from charlie_work.local_issue_files import COMMENTS_MARKER
from charlie_work.local_lane import branch_diff
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state, save_state, state_lock
from charlie_work.workflow import ORCHESTRATOR_COMMENT_MARKER, OrchestratorApp

ISSUE = 7  # the PR number and the issue number: the Local path keys a record by its issue
BRANCH = "agent/issue-7-x"
PROVENANCE = "fresh_llm_review"


class _Missing:
    def __repr__(self) -> str:
        return "<missing>"


MISSING = _Missing()


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    name: str
    decision: str
    summary: str = "Add a regression test for the new branch."
    required_changes: tuple[str, ...] = ()
    provenance: str = PROVENANCE
    # The packet (pr.json + diff.patch) the reviewer read: pinned to the live
    # head ("live-head"), an older head ("stale"), or absent ("none").
    packet: str = "live-head"
    reviewed_head: str = "none"  # "none" | "live" | "packet"
    # prs[7] before the call. ``prior_head``: "none" (first verdict), "older"
    # (the head has advanced since the last verdict) or "same" (it has not).
    pr_status: str = "reviewing"
    prior_count: int = 0
    prior_head: str = "none"
    issue_status: str = "dispatched"
    issue_closed: bool = False
    rescue_enabled: bool = False
    external_comment: bool = False
    comment: bool = False
    post_verdict_comment: bool = True
    allow_stale_head: bool = False
    rescue_attempted: bool = False  # prs[7].rescue_attempted already set before the call
    verdict_source: str | None = None
    session_metrics: tuple[tuple[str, Any], ...] | None = None
    # (path, value) pairs asserted on BOTH outcomes, so a scenario cannot pass by
    # both entry points failing for a reason other than the one it names.
    expect: tuple[tuple[str, Any], ...] = ()
    # Per-side pins, for scenarios where the two paths differ by declaration.
    expect_remote: tuple[tuple[str, Any], ...] = ()  # record_review
    expect_local: tuple[tuple[str, Any], ...] = ()  # record_local_review


@dataclass(frozen=True)
class Diff:
    scenario: str
    path: str
    remote: Any  # the PR path (record_review)
    local: Any  # the Local path (record_local_review)

    def __str__(self) -> str:
        return f"[{self.scenario}] {self.path}: record_review={self.remote!r} record_local_review={self.local!r}"


# ---------------------------------------------------------------------------
# Driving the two entry points
# ---------------------------------------------------------------------------


@dataclass
class _Repo:
    root: Path
    older_head: str
    head: str
    diff_text: str  # the live head's diff against main
    older_diff_text: str  # the older head's diff against main


def _make_repo() -> _Repo:
    root = new_repo_root()
    _init_repo(root)
    _git(root, "checkout", "-b", BRANCH)
    (root / "a.py").write_text("a = 1\n", encoding="utf-8")
    _git(root, "add", "a.py")
    _git(root, "commit", "-m", "feat: a")
    older = _git(root, "rev-parse", "HEAD").stdout.strip()
    (root / "b.py").write_text("b = 2\n", encoding="utf-8")
    _git(root, "add", "b.py")
    _git(root, "commit", "-m", "feat: b")
    head = _git(root, "rev-parse", "HEAD").stdout.strip()
    _git(root, "checkout", "main")
    diff = branch_diff(root, "main", head)
    older_diff = branch_diff(root, "main", older)
    assert diff and older_diff and diff != older_diff
    return _Repo(root, older, head, diff, older_diff)


def _config_data(scenario: Scenario) -> dict[str, Any]:
    data: dict[str, Any] = {"review": {"post_verdict_comment": scenario.post_verdict_comment}}
    if scenario.rescue_enabled:
        data["rescue"] = {"enabled": True}
    return data


def _prior_head_sha(scenario: Scenario, repo: _Repo) -> str | None:
    return {"none": None, "older": repo.older_head, "same": repo.head}[scenario.prior_head]


def _seed(app: OrchestratorApp, scenario: Scenario, repo: _Repo, *, local: bool) -> None:
    """Seed prs[7]/issues[7]; the same fields on both paths, plus the Local marker."""
    record: dict[str, Any] = {
        "number": ISSUE,
        "issue_number": ISSUE,
        "status": scenario.pr_status,
        "request_changes_count": scenario.prior_count,
    }
    if scenario.rescue_attempted:
        record["rescue_attempted"] = True
    prior = _prior_head_sha(scenario, repo)
    if prior is not None:
        record["reviewed_head_sha"] = prior
    if local:
        record.update({"local": True, "branch": BRANCH, "baseRefName": "main"})
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"][str(ISSUE)] = record
        state["issues"][str(ISSUE)] = {
            "number": ISSUE,
            "title": "Test issue",
            "status": scenario.issue_status,
            "branch_name": BRANCH,
        }
        save_state(app.paths.state_file, state)


def _write_packet(app: OrchestratorApp, scenario: Scenario, repo: _Repo) -> None:
    if scenario.packet == "none":
        return
    packet_head = repo.head if scenario.packet == "live-head" else repo.older_head
    pr_dir = app.paths.prs / f"pr-{ISSUE}"
    pr_dir.mkdir(parents=True, exist_ok=True)
    (pr_dir / "pr.json").write_text(
        json.dumps({"number": ISSUE, "headRefOid": packet_head}), encoding="utf-8"
    )
    # The packet holds the diff of the head it was built for.
    packet_diff = repo.diff_text if scenario.packet == "live-head" else repo.older_diff_text
    (pr_dir / "diff.patch").write_text(packet_diff, encoding="utf-8")


def _reviewed_head_arg(scenario: Scenario, repo: _Repo) -> str | None:
    packet_head = repo.head if scenario.packet == "live-head" else repo.older_head
    return {
        "none": None,
        "live": repo.head,
        "packet": packet_head,
        "bogus": "f" * 40,
    }[scenario.reviewed_head]


def _call_args(app: OrchestratorApp, scenario: Scenario, repo: _Repo) -> dict[str, Any]:
    return {
        "pr_number": ISSUE,
        "decision": scenario.decision,
        "summary": scenario.summary,
        "comment": scenario.comment,
        "reviewed_head": _reviewed_head_arg(scenario, repo),
        "required_changes": list(scenario.required_changes) or None,
        "verdict_provenance": scenario.provenance,
        "allow_stale_head": scenario.allow_stale_head,
        "verdict_source": scenario.verdict_source,
        "session_metrics": dict(scenario.session_metrics) if scenario.session_metrics else None,
    }


def _remote_app(scenario: Scenario, repo: _Repo, root: Path) -> tuple[OrchestratorApp, FakeGitHub]:
    config = build_config_from_data(_config_data(scenario))
    paths = runtime_paths(root, config.runtime.state_dir)
    gh = FakeGitHub()
    gh.prs = [
        {
            "number": ISSUE,
            "title": "Fix #7: thing",
            "url": f"https://example.test/pull/{ISSUE}",
            "headRefName": BRANCH,
            "baseRefName": "main",
            "headRefOid": repo.head,
            "mergeStateStatus": "CLEAN",
            "body": f"Closes #{ISSUE}\n\nTests: added.",
            "labels": [],
            "isCrossRepository": False,
            "state": {"merged": "MERGED", "closed": "CLOSED"}.get(scenario.pr_status, "OPEN"),
        }
    ]
    gh.issues = [
        {
            "number": ISSUE,
            "title": "Test issue",
            "url": f"https://example.test/issues/{ISSUE}",
            "body": "Body text.",
            "labels": [],
            "state": "OPEN",
        }
    ]
    gh.diffs[ISSUE] = repo.diff_text
    # The commit the external-findings window is bounded by (``before``); the
    # external comment below is dated before it, so it falls inside the window.
    gh.commits[repo.head] = {
        "parents": [],
        "commit": {"committer": {"date": "2020-06-01T00:00:00Z"}},
    }
    if scenario.external_comment:
        gh.pr_external_issue_comments[ISSUE] = [
            {
                "body": "Please also handle the empty-input case.",
                "user": {"login": "reviewer-human", "type": "User"},
                "created_at": "2020-01-01T00:00:00Z",
                "author_association": "MEMBER",
            }
        ]
    if scenario.issue_closed:
        gh.close_issue(ISSUE)
        gh.closed_issues.clear()
    return OrchestratorApp(root, paths, config, gh), gh


def _local_app(scenario: Scenario, repo: _Repo) -> OrchestratorApp:
    issues_dir = repo.root / "docs" / "issues"
    config = _local_config(repo.root, issues_dir, **_config_data(scenario))
    app = _app(repo.root, issues_dir, config)
    _parked_issue(app, issues_dir, ISSUE, BRANCH)
    if scenario.issue_closed:
        app.gh.close_issue(ISSUE)
    return app


def _label_names(issue: dict[str, Any]) -> frozenset[str]:
    names = set()
    for label in issue.get("labels", ()):
        names.add(label["name"] if isinstance(label, dict) else str(label))
    return frozenset(names)


# ---------------------------------------------------------------------------
# Outcome extraction and normalisation
# ---------------------------------------------------------------------------

_TS_KEY = re.compile(r"(_at|_ts|_since)$")


def _normalise(value: Any, *, root: Path, extra_roots: tuple[Path, ...] = ()) -> Any:
    """Transport-only normalisation: timestamps become ``"<set>"``; absolute paths
    under a scratch root become ``"<path>"``. Nothing else is touched."""
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if _TS_KEY.search(str(key)):
                out[key] = "<set>" if item else item
            else:
                out[key] = _normalise(item, root=root, extra_roots=extra_roots)
        return out
    if isinstance(value, list):
        return [_normalise(item, root=root, extra_roots=extra_roots) for item in value]
    if isinstance(value, str):
        for prefix in (root, *extra_roots):
            text = str(prefix)
            if text:
                value = value.replace(text, "<path>")
        return value
    return value


def _flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        if not value:
            out[prefix] = {}
        for key, item in value.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), item, out)
    else:
        out[prefix] = value


def _events_since(state_path: Path, before: int) -> list[dict[str, Any]]:
    """Events emitted by the call, from the unbounded ``events.db`` (a superset of
    the capped ``state.json`` ring: ``lifecycle_transition`` lands only there)."""
    return query_events(state_path)[before:]


def _flatten_events(events: list[dict[str, Any]], root: Path, out: dict[str, Any]) -> None:
    seen: dict[str, int] = {}
    kinds: list[str] = []
    for event in events:
        kind = str(event["kind"])
        kinds.append(kind)
        index = seen.get(kind, 0)
        seen[kind] = index + 1
        base = f"events.{kind}#{index}"
        out[f"{base}.level"] = event.get("level")
        _flatten(f"{base}.payload", _normalise(event.get("payload") or {}, root=root), out)
    out["events.order"] = tuple(kinds)


def _verdict_comment_remote(gh: FakeGitHub) -> tuple[str | None, str | None]:
    comments = list(gh.pr_comments_posted)
    if not comments:
        return None, None
    body = comments[-1][1]
    marker = f"{ORCHESTRATOR_COMMENT_MARKER}\n"
    assert body.startswith(marker), "record_review stamps its PR comments with the marker"
    return body[len(marker) :], "pr"


_COMMENT_HEADER = re.compile(r"\*\*charlie-work\*\* \([^)]*\)\n\n")


def _comment_entries(text: str) -> list[str]:
    if COMMENTS_MARKER not in text:
        return []
    return _COMMENT_HEADER.split(text.split(COMMENTS_MARKER, 1)[1])[1:]


def _verdict_comment_local(
    app: OrchestratorApp, before_text: str
) -> tuple[str | None, str | None]:
    issue_file = next(app.gh.issues_dir.glob(f"{ISSUE:03d}_*.md"))
    new = _comment_entries(issue_file.read_text(encoding="utf-8"))[
        len(_comment_entries(before_text)) :
    ]
    if not new:
        return None, None
    return new[-1].rstrip("\r\n- ").strip(), "issue"


def _rounds_listing(pr_dir: Path) -> tuple[str, ...]:
    rounds = pr_dir / "rounds"
    if not rounds.is_dir():
        return ()
    return tuple(
        sorted(
            str(path.relative_to(rounds)).replace("\\", "/")
            for path in rounds.rglob("*")
            if path.is_file()
        )
    )


def _collect(
    *,
    app: OrchestratorApp,
    result: Any,
    events_before: int,
    labels_before: frozenset[str],
    labels_after: frozenset[str],
    comment: tuple[str | None, str | None],
    roots: tuple[Path, ...],
) -> dict[str, Any]:
    root = roots[0]
    extra = roots[1:]
    state = load_state(app.paths.state_file)
    out: dict[str, Any] = {}
    data = dict(result.data or {})
    for key in ("decision_path",):
        data.pop(key, None)  # absolute path: transport
    data["rework_path"] = bool(data.get("rework_path"))
    out["result.ok"] = result.ok
    out["result.message"] = _normalise(result.message, root=root, extra_roots=extra)
    _flatten("result.data", _normalise(data, root=root, extra_roots=extra), out)

    issue = _normalise(dict(state["issues"].get(str(ISSUE)) or {}), root=root, extra_roots=extra)
    _flatten("issue", issue, out)
    pr_record = dict(state["prs"].get(str(ISSUE)) or {})
    for seed_only in ("branch", "baseRefName"):
        pr_record.pop(seed_only, None)  # seeded on the Local path only; the call rewrites neither
    pr_record.pop("decision_path", None)
    _flatten("pr_record", _normalise(pr_record, root=root, extra_roots=extra), out)

    events = _events_since(app.paths.state_file, events_before)
    _flatten_events(events, root, out)
    edges = [
        (event["payload"].get("event") or event["payload"].get("edge") or event["payload"])
        for event in events
        if event["kind"] == "lifecycle_transition"
    ]
    out["label_edge"] = json.dumps(
        [
            {
                k: v
                for k, v in e.items()
                if k in {"event", "edge", "from_state", "to_state", "cause"}
            }
            if isinstance(e, dict)
            else e
            for e in edges
        ],
        sort_keys=True,
    )
    out["labels.added"] = tuple(sorted(labels_after - labels_before))
    out["labels.removed"] = tuple(sorted(labels_before - labels_after))
    out["labels.final"] = tuple(sorted(labels_after))

    pr_dir = app.paths.prs / f"pr-{ISSUE}"
    decision_path = pr_dir / "review-decision.json"
    if decision_path.is_file():
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        _flatten("decision_file", _normalise(decision, root=root, extra_roots=extra), out)
    else:
        out["decision_file"] = MISSING
    out["rework_prompt"] = (pr_dir / "rework-prompt.md").is_file()
    # The prompt itself is rendered from a per-path template and legitimately
    # differs; the dispatch note is not.
    note = pr_dir / "rework-dispatch-note.txt"
    out["rework_dispatch_note"] = (
        _normalise(note.read_text(encoding="utf-8"), root=root, extra_roots=extra)
        if (pr_dir / "rework-prompt.md").is_file() and note.is_file()
        else None
    )
    out["rounds"] = _rounds_listing(pr_dir)
    body, channel = comment
    out["verdict_comment"] = body
    out["verdict_comment_channel"] = channel
    return out


def _run_remote(scenario: Scenario, repo: _Repo, start_labels: frozenset[str]) -> dict[str, Any]:
    root = Path(tempfile.mkdtemp(prefix="cw-verdict-pr-"))
    app, gh = _remote_app(scenario, repo, root)
    _pin_worker_selection(root)
    _seed(app, scenario, repo, local=False)
    _write_packet(app, scenario, repo)
    labels_before = start_labels
    # Replay the parking edge ``_parked_issue`` applied on the Local path, so the
    # issue's event chain (which ``lifecycle_transition.from_state`` reads) matches.
    transition(gh, app.config.labels, ISSUE, "local_work_ready", state_path=app.paths.state_file)
    gh.labels_added.clear()
    gh.labels_removed.clear()
    gh.issues[0]["labels"] = [{"name": name} for name in sorted(labels_before)]
    events_before = len(query_events(app.paths.state_file))
    result = app.record_review(**_call_args(app, scenario, repo))
    labels_after = (
        labels_before | {label for number, label in gh.labels_added if number == ISSUE}
    ) - {label for number, label in gh.labels_removed if number == ISSUE}
    return _collect(
        app=app,
        result=result,
        events_before=events_before,
        labels_before=labels_before,
        labels_after=frozenset(labels_after),
        comment=_verdict_comment_remote(gh),
        roots=(root,),
    )


def _run_local(scenario: Scenario, repo: _Repo) -> tuple[dict[str, Any], frozenset[str]]:
    app = _local_app(scenario, repo)
    _pin_worker_selection(repo.root)
    _seed(app, scenario, repo, local=True)
    _write_packet(app, scenario, repo)
    issue_file = next(app.gh.issues_dir.glob(f"{ISSUE:03d}_*.md"))
    before_text = issue_file.read_text(encoding="utf-8")
    labels_before = _label_names(app.gh.issue_view(ISSUE))
    events_before = len(query_events(app.paths.state_file))
    result = app.record_local_review(**_call_args(app, scenario, repo))
    labels_after = _label_names(app.gh.issue_view(ISSUE))
    outcome = _collect(
        app=app,
        result=result,
        events_before=events_before,
        labels_before=labels_before,
        labels_after=labels_after,
        comment=_verdict_comment_local(app, before_text),
        roots=(app.repo_root, app.paths.state_file.parent),
    )
    return outcome, labels_before


def _diffs(scenario: str, remote: dict[str, Any], local: dict[str, Any]) -> list[Diff]:
    out: list[Diff] = []
    for path in sorted(set(remote) | set(local)):
        left = remote.get(path, MISSING)
        right = local.get(path, MISSING)
        if left != right:
            out.append(Diff(scenario, path, left, right))
    return out


_FAKE_CI_FLEET = Path("ci-fleet")


def _pin_worker_selection(*roots: Path) -> None:
    """Pin the worker test-selection answer so the Events carry no host detail.

    ``selection_wrapper.worker_selection`` reports why it cannot wrap the test
    command; with no executable the detail names ``sys.executable``, which differs
    per host. An executable that exists (patched, as ``test_selection_wrapper.py``
    does) plus the repo's own ``enabled = false`` kill switch gives the same
    deterministic ``worker_test_selection_disabled`` answer everywhere.
    """
    for root in roots:
        (root / "ci-fleet.toml").write_text(
            "[test_selection]\nenabled = false\n", encoding="utf-8"
        )


def run_both(scenario: Scenario) -> tuple[dict[str, Any], dict[str, Any]]:
    """The scenario through ``record_review`` and ``record_local_review``."""
    repo = _make_repo()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(selection_wrapper, "ci_fleet_executable", lambda: _FAKE_CI_FLEET)
        local, start_labels = _run_local(scenario, repo)
        remote = _run_remote(scenario, repo, start_labels)
    return remote, local
