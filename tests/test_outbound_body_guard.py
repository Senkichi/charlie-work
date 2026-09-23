"""Tests for the outbound PR/issue body-write secret guard (issue #1505).

The guard scans body/title text about to leave the process through
``pr_create`` / ``issue_comment`` / ``pr_comment`` and *refuses* the write
when a vendored gitleaks credential rule matches -- the #694 edit-history
incident showed that deleting a leaked token from a PR body does not remove
it from GitHub's edit history or secret-scanning surface, so the only safe
boundary is before the API call.

All token fixtures are synthetic strings constructed to match the vendored
regexes (including their entropy gates); none are or ever were real
credentials.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from charlie_work.github import GitHub, GitHubError
from charlie_work.instrumentation import query_events
from charlie_work.local_issues import LocalFileGitHub
from charlie_work.outbound_body_guard import (
    check_outbound_write,
    scan_outbound_text,
)

# A 40-character high-entropy alphanumeric run -- long enough to satisfy the
# vendored rules' Shannon-entropy gates (distinct characters, no repetition).
_MIX = "x7K9pQ2mN4vB8wL1cR3tY6uI0oP5aS2dF4gH9jZq"
_LOWER = "abcdefghijklmnopqrstuvwxyz0123456789"


def _fill(n: int, alphabet: str = _MIX) -> str:
    """Return n characters cycling the (distinct-char) alphabet."""
    return (alphabet * (n // len(alphabet) + 1))[:n]


_GHO = f"gho_{_fill(36)}"
_GHP = f"ghp_{_fill(36)}"
_GHS = f"ghs_{_fill(36)}"
_GHU = f"ghu_{_fill(36)}"
_GHR = f"ghr_{_fill(36)}"
_GH_FG_PAT = f"github_pat_{_fill(82)}"
_GLPAT = f"glpat-{_fill(20)}"
_SLACK_BOT = f"xoxb-{'1' * 11}-{'2' * 11}{_fill(24)}"
_AWS = f"AKIA{_fill(16, 'BC2D3E4F5G6H7JKLMNPQRSTUVWXYZ')}"
_PRIVATE_KEY = "-----BEGIN PRIVATE KEY-----\n" + _fill(80) + "\n-----END PRIVATE KEY-----"
_ANTHROPIC = f"sk-ant-api03-{_fill(93)}AA"
_NPM = f"npm_{_fill(36, _LOWER)}"
_PYPI = f"pypi-AgEIcHlwaS5vcmc{_fill(60)}"
_STRIPE = f"sk_live_{_fill(24)}"
_JWT = f"ey{_fill(18)}.ey{_fill(18)}.{_fill(12)}"
_OPENAI = f"sk-{_fill(20)}T3BlbkFJ{_fill(20)}"


# ---------------------------------------------------------------------------
# Scanner unit tests
# ---------------------------------------------------------------------------


def test_clean_body_produces_no_matches() -> None:
    body = (
        "## Summary\n\n- adds the guard\n\nCloses #1505\n\n"
        "See https://github.com/o/r/pull/1234 for context."
    )
    assert scan_outbound_text(body, part="body") == ()


def test_empty_and_none_body_produce_no_matches() -> None:
    assert scan_outbound_text("", part="body") == ()


@pytest.mark.parametrize(
    ("token", "expected_rule"),
    [
        (_GHO, "github-oauth"),
        (_GHP, "github-pat"),
        (_GHS, "github-app-token"),
        (_GHU, "github-app-token"),
        (_GHR, "github-refresh-token"),
        (_GH_FG_PAT, "github-fine-grained-pat"),
        (_GLPAT, "gitlab-pat"),
        (_SLACK_BOT, "slack-bot-token"),
        (_AWS, "aws-access-token"),
        (_PRIVATE_KEY, "private-key"),
        (_ANTHROPIC, "anthropic-api-key"),
        (_NPM, "npm-access-token"),
        (_PYPI, "pypi-upload-token"),
        (_STRIPE, "stripe-access-token"),
        (_JWT, "jwt"),
        (_OPENAI, "openai-api-key"),
    ],
)
def test_full_credential_shapes_match(token: str, expected_rule: str) -> None:
    matches = scan_outbound_text(f"deploy output:\n{token}\ndone", part="body")
    rule_ids = {m.rule_id for m in matches}
    assert expected_rule in rule_ids


def test_placeholder_prose_does_not_match() -> None:
    """The issue's own incident text writes the token prefix with an ellipsis
    -- it must not be refused, or this very issue could not be discussed."""
    body = "a `gho_...` OAuth token was briefly written into the PR body"
    assert scan_outbound_text(body, part="body") == ()


def test_truncated_token_does_not_match() -> None:
    """35 of the required 36 tail characters -- under the regex length."""
    assert scan_outbound_text(f"gho_{_fill(35)}", part="body") == ()


def test_low_entropy_lookalike_does_not_match() -> None:
    """``gho_`` followed by 36 identical chars matches the regex but fails
    the rule's entropy gate -- exactly the upstream false-positive control."""
    assert scan_outbound_text("gho_" + "a" * 36, part="body") == ()


def test_aws_example_key_is_allowlisted() -> None:
    """AWS's canonical documentation key ends in ``EXAMPLE``; the vendored
    rule's own allowlist excludes it -- credential-shaped prose stays free."""
    assert scan_outbound_text("key: AKIAIOSFODNN7EXAMPLE", part="body") == ()


def test_example_secret_fence_is_exempt() -> None:
    body = f"before\n\n```example-secret\n{_GHO}\n```\n\nafter\n"
    assert scan_outbound_text(body, part="body") == ()


def test_example_secret_fence_unclosed_runs_to_eof() -> None:
    body = f"```example-secret\n{_GHO}\n{_GHP}\n"
    assert scan_outbound_text(body, part="body") == ()


def test_other_fences_are_still_scanned() -> None:
    """Only the ``example-secret`` info string is exempt -- an ordinary code
    fence does not launder a credential past the guard."""
    for info in ("", "text", "sh"):
        body = f"```{info}\n{_GHO}\n```\n"
        matches = scan_outbound_text(body, part="body")
        assert any(m.rule_id == "github-oauth" for m in matches), info


def test_match_reports_part_and_line() -> None:
    matches = scan_outbound_text(f"line one\nline two {_GHO}\n", part="body")
    match = next(m for m in matches if m.rule_id == "github-oauth")
    assert match.part == "body"
    assert match.line == 2


# ---------------------------------------------------------------------------
# check_outbound_write: the shared chokepoint
# ---------------------------------------------------------------------------


def _state_file(repo_root: Path) -> Path:
    return repo_root / ".var" / "charlie-work" / "state.json"


def test_check_returns_empty_for_clean_parts(tmp_path: Path) -> None:
    matches = check_outbound_write(
        surface="pr_create",
        parts=(("title", "t"), ("body", "clean body")),
        repo_root=tmp_path,
    )
    assert matches == ()


def test_check_emits_event_and_returns_matches(tmp_path: Path) -> None:
    matches = check_outbound_write(
        surface="pr_create",
        parts=(("title", "t"), ("body", f"notes {_GHO}")),
        repo_root=tmp_path,
    )
    assert matches
    events = query_events(_state_file(tmp_path), kind="outbound_body_secret_refused")
    assert len(events) == 1
    event = events[0]
    assert event["level"] == "warning"
    payload = event["payload"]
    assert payload["surface"] == "pr_create"
    assert "github-oauth" in payload["rule_ids"]


def test_event_payload_never_contains_secret_material(tmp_path: Path) -> None:
    check_outbound_write(
        surface="issue_comment",
        parts=(("body", f"leaked {_GHP} here"),),
        repo_root=tmp_path,
        issue_number=7,
    )
    events = query_events(_state_file(tmp_path), kind="outbound_body_secret_refused")
    assert len(events) == 1
    assert events[0]["issue_number"] == 7
    raw = json.dumps(events[0]["payload"])
    assert _GHP not in raw
    assert "ghp_" not in raw  # not even the prefix line excerpt


def test_pr_number_is_indexed_on_pr_comment_refusal(tmp_path: Path) -> None:
    check_outbound_write(
        surface="pr_comment",
        parts=(("body", f"leaked {_GHP}"),),
        repo_root=tmp_path,
        pr_number=42,
    )
    events = query_events(_state_file(tmp_path), kind="outbound_body_secret_refused")
    assert events[0]["pr_number"] == 42


# ---------------------------------------------------------------------------
# GitHub client integration: refusal happens before any gh invocation
# ---------------------------------------------------------------------------


class _FakeCompleted:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _capture_subprocess(monkeypatch: pytest.MonkeyPatch, **result: Any) -> list[list[str]]:
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: Any) -> _FakeCompleted:
        calls.append(list(command))
        return _FakeCompleted(**result)

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


_PR_URL = "https://github.com/Senkichi/charlie-work/pull/1234"


def test_pr_create_refuses_secret_body_before_gh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _capture_subprocess(monkeypatch, returncode=0, stdout=_PR_URL)
    result = GitHub(repo_root=tmp_path).pr_create(
        head="h", base="main", title="t", body=f"notes {_GHO}"
    )
    assert result is None
    assert calls == []
    events = query_events(_state_file(tmp_path), kind="outbound_body_secret_refused")
    assert len(events) == 1
    assert events[0]["payload"]["surface"] == "pr_create"


def test_pr_create_refuses_secret_in_title(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The title is written by the same gh call and lands in the same edit
    history -- it is inside the guarded surface, not just the body."""
    calls = _capture_subprocess(monkeypatch, returncode=0, stdout=_PR_URL)
    result = GitHub(repo_root=tmp_path).pr_create(
        head="h", base="main", title=f"wip {_GHP}", body="clean"
    )
    assert result is None
    assert calls == []


def test_pr_create_clean_body_proceeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _capture_subprocess(monkeypatch, returncode=0, stdout=_PR_URL)
    result = GitHub(repo_root=tmp_path).pr_create(
        head="h", base="main", title="t", body="clean body"
    )
    assert result == 1234
    assert len(calls) == 1


def test_pr_create_dry_run_still_no_ops(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """dry_run never reaches gh -- scanning a body that will not be written
    would be a false refusal."""
    calls = _capture_subprocess(monkeypatch, returncode=0, stdout=_PR_URL)
    assert (
        GitHub(repo_root=tmp_path, dry_run=True).pr_create(
            head="h", base="main", title="t", body=f"notes {_GHO}"
        )
        == 0
    )
    assert calls == []


def test_issue_comment_refuses_before_gh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _capture_subprocess(monkeypatch, returncode=0)
    body_file = tmp_path / "comment.md"
    body_file.write_text(f"found it: {_GHO}\n", encoding="utf-8")
    with pytest.raises(GitHubError):
        GitHub(repo_root=tmp_path).issue_comment(9, body_file)
    assert calls == []
    events = query_events(_state_file(tmp_path), kind="outbound_body_secret_refused")
    assert len(events) == 1
    assert events[0]["payload"]["surface"] == "issue_comment"
    assert events[0]["issue_number"] == 9


def test_pr_comment_refuses_before_gh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _capture_subprocess(monkeypatch, returncode=0)
    body_file = tmp_path / "comment.md"
    body_file.write_text(f"token {_GHP}", encoding="utf-8")
    with pytest.raises(GitHubError):
        GitHub(repo_root=tmp_path).pr_comment(5, body_file)
    assert calls == []
    events = query_events(_state_file(tmp_path), kind="outbound_body_secret_refused")
    assert events[0]["payload"]["surface"] == "pr_comment"
    assert events[0]["pr_number"] == 5


def test_issue_comment_clean_body_calls_gh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _capture_subprocess(monkeypatch, returncode=0)
    body_file = tmp_path / "comment.md"
    body_file.write_text("all clean\n", encoding="utf-8")
    GitHub(repo_root=tmp_path).issue_comment(9, body_file)
    assert len(calls) == 1
    assert calls[0][:2] == ["gh", "issue"]


# ---------------------------------------------------------------------------
# Local-file backend: the comment lands in a git-visible issue file
# ---------------------------------------------------------------------------


def _local_issue(tmp_path: Path) -> LocalFileGitHub:
    issues_dir = tmp_path / "issues"
    issues_dir.mkdir()
    (issues_dir / "001_issue.md").write_text(
        '---\ntitle: "t"\nstate: open\nlabels: []\n---\nbody\n', encoding="utf-8"
    )
    return LocalFileGitHub(repo_root=tmp_path, issues_dir=issues_dir)


def test_local_issue_comment_refuses_secret(tmp_path: Path) -> None:
    gh = _local_issue(tmp_path)
    body_file = tmp_path / "comment.md"
    body_file.write_text(f"leak {_GHO}\n", encoding="utf-8")
    with pytest.raises(GitHubError):
        gh.issue_comment(1, body_file)
    events = query_events(_state_file(tmp_path), kind="outbound_body_secret_refused")
    assert len(events) == 1


def test_local_issue_comment_clean_appends(tmp_path: Path) -> None:
    gh = _local_issue(tmp_path)
    body_file = tmp_path / "comment.md"
    body_file.write_text("review-ready\n", encoding="utf-8")
    gh.issue_comment(1, body_file)
    issue = gh.issue_view(1)
    comments = [c["body"] for c in issue["comments"]]
    assert any("review-ready" in c for c in comments)


# ---------------------------------------------------------------------------
# Operator-facing consumer: refusal events surface through the digest
# ---------------------------------------------------------------------------

# Imported after ``charlie_work.workflow`` deliberately (same partial-module
# hazard as tests/test_issue_1314_operator_queue_followups.py documents:
# ``orchestration.state_maintenance`` does ``import charlie_work.workflow``
# at module level, so the workflow import must complete first).
import charlie_work.workflow as _wf  # noqa: E402
from _fakes_github import FakeGitHub  # noqa: E402
from charlie_work.config import (  # noqa: E402
    NotifyConfig,
    OrchestratorConfig,
    PostMortemConfig,
)
from charlie_work.instrumentation import _LEVEL_BY_KIND, log_event  # noqa: E402
from charlie_work.notify import _DESKTOP_SEVERITIES  # noqa: E402
from charlie_work.paths import runtime_paths  # noqa: E402
from charlie_work.state import load_state  # noqa: E402
from charlie_work.workflow import OrchestratorApp  # noqa: E402


def _app(
    tmp_path: Path,
    *,
    notify: NotifyConfig | None = None,
    dry_run: bool = False,
) -> OrchestratorApp:
    config = OrchestratorConfig(
        post_mortem=PostMortemConfig(db_path=str(tmp_path / "missing-sessions.db")),
        notify=notify or NotifyConfig(),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    return OrchestratorApp(tmp_path, paths, config, FakeGitHub(), dry_run=dry_run)


def _seed_refusal(state_file: Path, *, surface: str = "pr_create") -> None:
    log_event(
        state_file,
        "outbound_body_secret_refused",
        {
            "surface": surface,
            "rule_ids": ["github-oauth"],
            "parts": ["body"],
            "match_count": 1,
            "issue_number": 7,
        },
    )


def test_refusal_kind_registered_as_warning() -> None:
    assert _LEVEL_BY_KIND["outbound_body_secret_refused"] == "warning"


def test_refusal_kind_not_in_expected_operational_kinds() -> None:
    """A refusal is rare, not routine -- it keeps the flat detailed listing
    in heartbeat's warning report rather than the summarized count bucket."""
    from charlie_work.event_kinds import EXPECTED_OPERATIONAL_KINDS

    assert "outbound_body_secret_refused" not in EXPECTED_OPERATIONAL_KINDS


def test_desktop_severity_registered() -> None:
    assert "OUTBOUND_BODY_SECRET_REFUSED" in _DESKTOP_SEVERITIES


def test_unsurfaced_refusals_emit_digest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _app(tmp_path, notify=NotifyConfig(enabled=True))
    _seed_refusal(app.paths.state_file)

    emitted: list[Any] = []
    monkeypatch.setattr(_wf, "emit_digest", lambda cfg, digest: emitted.append(digest))
    app._maybe_report_outbound_secret_refusals()

    assert len(emitted) == 1
    entry = emitted[0].transitions[0]
    assert entry.health == "OUTBOUND_BODY_SECRET_REFUSED"
    assert "pr_create" in entry.last_log_line
    assert "github-oauth" in entry.last_log_line

    state = load_state(app.paths.state_file)
    assert state["outbound_secret_refusals_surfaced"] == 1


def test_consumer_is_idempotent_for_already_surfaced_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _app(tmp_path, notify=NotifyConfig(enabled=True))
    _seed_refusal(app.paths.state_file)
    emitted: list[Any] = []
    monkeypatch.setattr(_wf, "emit_digest", lambda cfg, digest: emitted.append(digest))

    app._maybe_report_outbound_secret_refusals()
    app._maybe_report_outbound_secret_refusals()
    assert len(emitted) == 1

    # A new refusal after the marker still fires exactly once.
    _seed_refusal(app.paths.state_file, surface="issue_comment")
    app._maybe_report_outbound_secret_refusals()
    assert len(emitted) == 2
    assert "issue_comment" in emitted[1].transitions[0].last_log_line


def test_consumer_silent_under_dry_run(tmp_path: Path) -> None:
    app = _app(tmp_path, notify=NotifyConfig(enabled=True), dry_run=True)
    _seed_refusal(app.paths.state_file)
    before = app.paths.state_file.read_bytes() if app.paths.state_file.exists() else b""
    app._maybe_report_outbound_secret_refusals()
    if app.paths.state_file.exists():
        assert app.paths.state_file.read_bytes() == before


def test_consumer_no_events_no_digest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _app(tmp_path, notify=NotifyConfig(enabled=True))
    emitted: list[Any] = []
    monkeypatch.setattr(_wf, "emit_digest", lambda cfg, digest: emitted.append(digest))
    app._maybe_report_outbound_secret_refusals()
    assert emitted == []


def test_example_secret_inside_ordinary_fence_is_not_exempt() -> None:
    """Nested-looking fences are literal content: a closing fence cannot
    carry an info string, so the ``example-secret`` line below is inside the
    ``text`` block -- the credential is scanned, not masked."""
    body = f"```text\n```example-secret\n{_GHO}\n```\n"
    matches = scan_outbound_text(body, part="body")
    assert any(m.rule_id == "github-oauth" for m in matches)


def test_ordinary_fence_then_example_secret_still_exempt() -> None:
    """A real ``example-secret`` fence after a closed ordinary fence is a
    top-level opener and exempts normally."""
    body = f"```text\nplain\n```\n```example-secret\n{_GHO}\n```\n"
    assert scan_outbound_text(body, part="body") == ()


def test_refusal_event_carries_repo_name(tmp_path: Path) -> None:
    check_outbound_write(
        surface="pr_create",
        parts=(("body", f"notes {_GHO}"),),
        repo_root=tmp_path,
    )
    events = query_events(_state_file(tmp_path), kind="outbound_body_secret_refused")
    assert events[0]["repo"] == tmp_path.name


# ---------------------------------------------------------------------------
# Fail-closed mechanics: guard machinery failure refuses in each surface's
# own error vocabulary (review: ruleset-load failure must not escape raw)
# ---------------------------------------------------------------------------


@pytest.fixture
def _broken_ruleset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Point the loader at a malformed vendored file."""
    import charlie_work.outbound_body_guard as obg

    bad = tmp_path / "broken-secrets.toml"
    bad.write_text("rules = [[[not toml", encoding="utf-8")
    monkeypatch.setattr(obg, "_RULES_PATH", bad)
    obg._rules.cache_clear()
    yield bad
    obg._rules.cache_clear()


def test_check_raises_guard_error_on_broken_ruleset(tmp_path: Path, _broken_ruleset: Path) -> None:
    from charlie_work.outbound_body_guard import OutboundBodyGuardError

    with pytest.raises(OutboundBodyGuardError):
        check_outbound_write(surface="pr_create", parts=(("body", "clean"),), repo_root=tmp_path)


def test_pr_create_guard_failure_returns_none_before_gh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _broken_ruleset: Path
) -> None:
    calls = _capture_subprocess(monkeypatch, returncode=0, stdout=_PR_URL)
    result = GitHub(repo_root=tmp_path).pr_create(
        head="h", base="main", title="t", body="clean body"
    )
    assert result is None
    assert calls == []


def test_issue_comment_guard_failure_raises_github_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _broken_ruleset: Path
) -> None:
    calls = _capture_subprocess(monkeypatch, returncode=0)
    body_file = tmp_path / "comment.md"
    body_file.write_text("clean\n", encoding="utf-8")
    with pytest.raises(GitHubError):
        GitHub(repo_root=tmp_path).issue_comment(9, body_file)
    assert calls == []


def test_local_issue_comment_guard_failure_raises_github_error(
    tmp_path: Path, _broken_ruleset: Path
) -> None:
    gh = _local_issue(tmp_path)
    body_file = tmp_path / "comment.md"
    body_file.write_text("clean\n", encoding="utf-8")
    with pytest.raises(GitHubError):
        gh.issue_comment(1, body_file)


def test_issue_comment_non_utf8_body_file_raises_github_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _capture_subprocess(monkeypatch, returncode=0)
    body_file = tmp_path / "comment.md"
    body_file.write_bytes(b"\xff\xfe\x00\x01 binary")
    with pytest.raises(GitHubError):
        GitHub(repo_root=tmp_path).issue_comment(9, body_file)
    assert calls == []


def test_issue_comment_dry_run_never_reads_or_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """dry_run skips the guard entirely -- a secret in a body that will not
    be written must not refuse, and the file need not even exist."""
    calls = _capture_subprocess(monkeypatch, returncode=0)
    GitHub(repo_root=tmp_path, dry_run=True).issue_comment(9, tmp_path / "nonexistent.md")
    GitHub(repo_root=tmp_path, dry_run=True).pr_comment(9, tmp_path / "nonexistent.md")
    assert calls == []


def test_local_issue_comment_dry_run_never_reads(tmp_path: Path) -> None:
    gh = _local_issue(tmp_path)
    gh = LocalFileGitHub(repo_root=tmp_path, issues_dir=gh.issues_dir, dry_run=True)
    gh.issue_comment(1, tmp_path / "nonexistent.md")


def test_digest_emit_failure_leaves_cursor_for_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If emit_digest raises, the batch must not be consumed -- the cursor
    is persisted only after a successful emit so the next pass retries."""
    app = _app(tmp_path, notify=NotifyConfig(enabled=True))
    _seed_refusal(app.paths.state_file)

    def boom(cfg: Any, digest: Any) -> None:
        raise RuntimeError("notify backend down")

    monkeypatch.setattr(_wf, "emit_digest", boom)
    with pytest.raises(RuntimeError):
        app._maybe_report_outbound_secret_refusals()
    state = load_state(app.paths.state_file)
    assert "outbound_secret_refusals_surfaced" not in state

    emitted: list[Any] = []
    monkeypatch.setattr(_wf, "emit_digest", lambda cfg, digest: emitted.append(digest))
    app._maybe_report_outbound_secret_refusals()
    assert len(emitted) == 1
