"""Session-failure classification: ``_classify_session_failure``
throttle/provider-auth/provider-suspended signatures and
``update_worker_record_with_failure_classification`` record writes.

Split out of ``tests/test_claude_code_adapter.py`` (issue #1560,
Track 1) -- bodies are verbatim relocations; shared helpers live in
``tests/_claude_adapter_fixtures.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

from charlie_work.config import (
    OrchestratorConfig,
    RuntimeConfig,
)
from charlie_work.claude_code import update_worker_record_with_failure_classification

# ---------------------------------------------------------------------------
# Throttle death classification tests (symmetric to devin_shell tests)
# ---------------------------------------------------------------------------


def test_classify_session_failure_rate_limit_with_reset_time(tmp_path: Path) -> None:
    """Test that rate-limit errors with 'resets in N minutes' are classified correctly."""
    from charlie_work.claude_code import _classify_session_failure

    log_path = tmp_path / "session.claude.log"
    log_path.write_text(
        "Some work done...\n"
        "Error: Reached overall message rate limit. Please try again later. "
        "Your limit will reset in 10 minutes.\n",
        encoding="utf-8",
    )

    failure_kind, throttled_until = _classify_session_failure(log_path)

    assert failure_kind == "rate_limited"
    assert throttled_until is not None
    # Verify it's a valid ISO timestamp
    assert "T" in throttled_until
    assert "Z" in throttled_until


def test_classify_session_failure_quota_exhausted(tmp_path: Path) -> None:
    """Test that quota-exhaustion errors are classified correctly."""
    from charlie_work.claude_code import _classify_session_failure

    log_path = tmp_path / "session.claude.log"
    log_path.write_text(
        "Some work done...\n"
        "Error: daily usage quota has been exhausted. Please try again tomorrow.\n",
        encoding="utf-8",
    )

    failure_kind, throttled_until = _classify_session_failure(log_path)

    assert failure_kind == "quota_exhausted"
    assert throttled_until is not None
    # Should use default 24 hour cooldown
    assert "T" in throttled_until
    assert "Z" in throttled_until


def test_classify_session_failure_includes_resume_margin(tmp_path: Path) -> None:
    """Issue #499: killed-worker rate-limit classification must include the resume margin."""
    from datetime import UTC, datetime, timedelta

    from charlie_work.claude_code import _classify_session_failure

    log_path = tmp_path / "session.claude.log"
    log_path.write_text(
        "Error: Reached overall message rate limit. Your limit will reset in 3 minutes.\n",
        encoding="utf-8",
    )

    now = datetime.now(UTC)
    failure_kind, throttled_until = _classify_session_failure(
        log_path, resume_margin_seconds=90, now=now
    )

    assert failure_kind == "rate_limited"
    assert throttled_until is not None
    parsed = datetime.fromisoformat(throttled_until.replace("Z", "+00:00"))
    expected = (now + timedelta(minutes=3, seconds=90)).replace(microsecond=0)
    assert parsed == expected


def test_update_worker_record_with_failure_classification(tmp_path: Path) -> None:
    """Test that worker records are updated with failure classification."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    # Create a worker sidecar
    sidecar_path = sessions_dir / "issue-42.claude.json"
    sidecar_path.write_text(
        json.dumps(
            {
                "issue_number": 42,
                "branch": "agent/issue-42",
                "worktree_path": "/tmp/wt/issue-42",
                "prompt_path": "p.md",
                "command": ["claude", "-p"],
                "pid": 1234,
                "started_at": "2026-01-01T00:00:00Z",
                "log_path": str(sessions_dir / "issue-42.claude.log"),
                "error": None,
            }
        ),
        encoding="utf-8",
    )

    # Create a log file with rate-limit error
    log_path = sessions_dir / "issue-42.claude.log"
    log_path.write_text(
        "Error: Reached overall message rate limit. Please try again later. "
        "Your limit will reset in 10 minutes.\n",
        encoding="utf-8",
    )

    failure_kind, throttled_until = update_worker_record_with_failure_classification(
        sessions_dir, 42
    )

    assert failure_kind == "rate_limited"
    assert throttled_until is not None

    # Verify the sidecar was updated
    updated_sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert updated_sidecar["failure_kind"] == "rate_limited"


def test_update_worker_record_with_failure_classification_includes_resume_margin(
    tmp_path: Path,
) -> None:
    """Issue #499: update wrapper applies config.runtime.throttle_resume_margin_s."""
    from datetime import UTC, datetime, timedelta

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    sidecar_path = sessions_dir / "issue-42.claude.json"
    sidecar_path.write_text(
        json.dumps(
            {
                "issue_number": 42,
                "branch": "agent/issue-42",
                "worktree_path": "/tmp/wt/issue-42",
                "prompt_path": "p.md",
                "command": ["claude", "-p"],
                "pid": 1234,
                "started_at": "2026-01-01T00:00:00Z",
                "log_path": str(sessions_dir / "issue-42.claude.log"),
                "error": None,
            }
        ),
        encoding="utf-8",
    )

    log_path = sessions_dir / "issue-42.claude.log"
    log_path.write_text(
        "Error: Reached overall message rate limit. Your limit will reset in 5 minutes.\n",
        encoding="utf-8",
    )

    config = OrchestratorConfig(runtime=RuntimeConfig(throttle_resume_margin_s=90))
    now = datetime.now(UTC)
    failure_kind, throttled_until = update_worker_record_with_failure_classification(
        sessions_dir, 42, config=config, now=now
    )

    assert failure_kind == "rate_limited"
    assert throttled_until is not None
    parsed = datetime.fromisoformat(throttled_until.replace("Z", "+00:00"))
    expected = (now + timedelta(minutes=5, seconds=90)).replace(microsecond=0)
    assert parsed == expected


def test_update_worker_record_with_failure_classification_session_completed_skips_log_tail(
    tmp_path: Path,
) -> None:
    """Issue #656: a completed session's own prose must not be reclassified quota_exhausted.

    Reproduces the live incident: a worker's completion summary quoted the
    throttle-marker text ("usage limit") while describing an unrelated fix,
    and log-tail classification stomped the caller's ``fallback_kind`` with
    ``quota_exhausted`` despite the worktree proving the work completed.
    ``session_completed=True`` must skip log-tail classification entirely.
    """
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    sidecar_path = sessions_dir / "issue-42.claude.json"
    sidecar_path.write_text(
        json.dumps(
            {
                "issue_number": 42,
                "branch": "agent/issue-42",
                "worktree_path": "/tmp/wt/issue-42",
                "prompt_path": "p.md",
                "command": ["claude", "-p"],
                "pid": 1234,
                "started_at": "2026-01-01T00:00:00Z",
                "log_path": str(sessions_dir / "issue-42.claude.log"),
                "error": None,
            }
        ),
        encoding="utf-8",
    )

    # Completion summary prose that happens to quote a throttle marker while
    # describing an unrelated fix -- not an actual provider death message.
    log_path = sessions_dir / "issue-42.claude.log"
    log_path.write_text(
        '## Summary\n\nFixed generic substrings ("rate limit", "usage limit") that '
        "legitimately appear in this codebase's rate-limit/quota domain commentary.\n"
        "- `ruff check` + `ruff format`: clean\n",
        encoding="utf-8",
    )

    failure_kind, throttled_until = update_worker_record_with_failure_classification(
        sessions_dir, 42, fallback_kind="unpublished_work", session_completed=True
    )

    assert failure_kind == "unpublished_work"
    assert throttled_until is None

    updated_sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert updated_sidecar["failure_kind"] == "unpublished_work"


def _make_worker_sidecar(sessions_dir: Path, issue_number: int, log_path: Path) -> Path:
    """Write a minimal worker sidecar for failure-classification tests."""
    sidecar_path = sessions_dir / f"issue-{issue_number}.claude.json"
    sidecar_path.write_text(
        json.dumps(
            {
                "issue_number": issue_number,
                "branch": f"agent/issue-{issue_number}",
                "worktree_path": "/tmp/wt",
                "prompt_path": "p.md",
                "command": ["claude", "-p"],
                "pid": 1234,
                "started_at": "2026-01-01T00:00:00Z",
                "log_path": str(log_path),
                "error": None,
            }
        ),
        encoding="utf-8",
    )
    return sidecar_path


def test_classify_session_failure_tool_rejected_is_not_throttle(tmp_path: Path) -> None:
    """Issue #260, corrected premise: 'A tool was rejected by the user' is the
    Devin CLI's own surfacing of a PreToolUse hook block, not a provider
    throttle condition — it must NOT classify as rate_limited (no retry
    semantics, no throttled_until). See test_devin_shell.py's mirror test
    and test_post_mortem_log_tail_fallback.py for the worker_blocked log-tail fallback that
    now owns this signature instead."""
    from charlie_work.claude_code import _classify_session_failure

    log_path = tmp_path / "session.claude.log"
    log_path.write_text(
        "Error: A tool was rejected by the user.\n",
        encoding="utf-8",
    )

    failure_kind, throttled_until = _classify_session_failure(log_path)

    assert failure_kind is None
    assert throttled_until is None


def test_update_worker_record_tool_rejected_is_not_rate_limited(tmp_path: Path) -> None:
    """Issue #260, corrected premise: a tool-rejected sidecar log must not be
    classified rate_limited by the adapter's own log-tail classifier."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    log_path = sessions_dir / "issue-42.claude.log"
    log_path.write_text(
        "Error: A tool was rejected by the user.\n",
        encoding="utf-8",
    )
    sidecar_path = _make_worker_sidecar(sessions_dir, 42, log_path)

    failure_kind, throttled_until = update_worker_record_with_failure_classification(
        sessions_dir, 42, fallback_kind="stalled"
    )

    assert failure_kind == "stalled"
    assert throttled_until is None
    updated_sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert updated_sidecar["failure_kind"] == "stalled"


def test_update_worker_record_unknown_tail_falls_back_to_stalled(tmp_path: Path) -> None:
    """Unknown log tail should fall back to the provided fallback_kind."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    log_path = sessions_dir / "issue-42.claude.log"
    log_path.write_text(
        "Error: something completely unrelated went wrong\n",
        encoding="utf-8",
    )
    sidecar_path = _make_worker_sidecar(sessions_dir, 42, log_path)

    failure_kind, throttled_until = update_worker_record_with_failure_classification(
        sessions_dir, 42, fallback_kind="stalled"
    )

    assert failure_kind == "stalled"
    assert throttled_until is None
    updated_sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert updated_sidecar["failure_kind"] == "stalled"


def test_update_worker_record_custom_throttle_markers(tmp_path: Path) -> None:
    """RuntimeConfig.throttle_error_markers is configurable without code changes."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    log_path = sessions_dir / "issue-42.claude.log"
    log_path.write_text(
        "Error: provider-specific frobnicate limit exceeded\n",
        encoding="utf-8",
    )
    sidecar_path = _make_worker_sidecar(sessions_dir, 42, log_path)

    config = OrchestratorConfig(
        runtime=RuntimeConfig(throttle_error_markers=("frobnicate limit exceeded",))
    )
    failure_kind, throttled_until = update_worker_record_with_failure_classification(
        sessions_dir, 42, config=config
    )

    assert failure_kind == "rate_limited"
    assert throttled_until is not None
    updated_sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert updated_sidecar["failure_kind"] == "rate_limited"


# ---------------------------------------------------------------------------
# Issue #484: provider-auth classification for api workers
# ---------------------------------------------------------------------------


def test_classify_session_failure_provider_auth_401(tmp_path: Path) -> None:
    """Issue #484: a 401 error in an api session log classifies as provider_auth."""
    from charlie_work.claude_code import _classify_session_failure

    log_path = tmp_path / "session.claude.log"
    log_path.write_text(
        "Working...\nError: 401 Unauthorized. Invalid API key.\n",
        encoding="utf-8",
    )

    failure_kind, throttled_until = _classify_session_failure(log_path, adapter_kind="api")

    assert failure_kind == "provider_auth"
    assert throttled_until is not None
    assert "T" in throttled_until
    assert "Z" in throttled_until


def test_classify_session_failure_provider_auth_403(tmp_path: Path) -> None:
    """Issue #484: a 403 error in an api session log classifies as provider_auth."""
    from charlie_work.claude_code import _classify_session_failure

    log_path = tmp_path / "session.claude.log"
    log_path.write_text(
        "Working...\nError: 403 Forbidden. Authentication failed.\n",
        encoding="utf-8",
    )

    failure_kind, throttled_until = _classify_session_failure(log_path, adapter_kind="api")

    assert failure_kind == "provider_auth"
    assert throttled_until is not None


def test_classify_session_failure_provider_auth_invalid_key(tmp_path: Path) -> None:
    """Issue #484: an invalid-api-key error classifies as provider_auth."""
    from charlie_work.claude_code import _classify_session_failure

    log_path = tmp_path / "session.claude.log"
    log_path.write_text(
        "Working...\nError: invalid api key provided. Please check your configuration.\n",
        encoding="utf-8",
    )

    failure_kind, throttled_until = _classify_session_failure(log_path, adapter_kind="api")

    assert failure_kind == "provider_auth"
    assert throttled_until is not None


def test_classify_session_failure_provider_auth_not_for_claude_code(
    tmp_path: Path,
) -> None:
    """Issue #484: auth patterns must NOT fire for claude-code sessions (only api)."""
    from charlie_work.claude_code import _classify_session_failure

    log_path = tmp_path / "session.claude.log"
    log_path.write_text(
        "Working...\nError: 401 Unauthorized. Invalid API key.\n",
        encoding="utf-8",
    )

    # Default adapter_kind="claude-code" — auth patterns are not checked.
    failure_kind, throttled_until = _classify_session_failure(log_path)

    assert failure_kind is None
    assert throttled_until is None


def test_classify_session_failure_throttle_not_misclassified_as_auth(
    tmp_path: Path,
) -> None:
    """Issue #484: a generic throttle log still classifies as rate_limited, not
    provider_auth, even for api sessions."""
    from charlie_work.claude_code import _classify_session_failure

    log_path = tmp_path / "session.claude.log"
    log_path.write_text(
        "Working...\n"
        "Error: Reached overall message rate limit. Please try again later. "
        "Your limit will reset in 10 minutes.\n",
        encoding="utf-8",
    )

    failure_kind, throttled_until = _classify_session_failure(log_path, adapter_kind="api")

    assert failure_kind == "rate_limited"
    assert throttled_until is not None


def test_classify_session_failure_provider_auth_cooldown_24h(tmp_path: Path) -> None:
    """Issue #484: provider_auth cooldown reuses the 24h quota-exhaustion constant."""
    from datetime import UTC, datetime, timedelta

    from charlie_work.claude_code import _classify_session_failure

    log_path = tmp_path / "session.claude.log"
    log_path.write_text("Error: 401 Unauthorized\n", encoding="utf-8")

    now = datetime.now(UTC)
    failure_kind, throttled_until = _classify_session_failure(
        log_path, adapter_kind="api", now=now
    )

    assert failure_kind == "provider_auth"
    parsed = datetime.fromisoformat(throttled_until.replace("Z", "+00:00"))
    expected = (now + timedelta(hours=24)).replace(microsecond=0)
    assert parsed == expected


def test_update_worker_record_api_provider_auth_classification(
    tmp_path: Path,
) -> None:
    """Issue #484: update_worker_record_with_failure_classification with
    adapter_kind='api' reads the api sidecar and classifies an auth failure."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    sidecar_path = sessions_dir / "issue-42.api.json"
    sidecar_path.write_text(
        json.dumps(
            {
                "issue_number": 42,
                "branch": "agent/issue-42",
                "worktree_path": "/tmp/wt/issue-42",
                "prompt_path": "p.md",
                "command": ["claude", "-p"],
                "pid": 1234,
                "started_at": "2026-01-01T00:00:00Z",
                "log_path": str(sessions_dir / "issue-42.claude.log"),
                "error": None,
                "adapter_kind": "api",
                "provider": "example",
            }
        ),
        encoding="utf-8",
    )

    log_path = sessions_dir / "issue-42.claude.log"
    log_path.write_text("Error: 403 Forbidden. Authentication failed.\n", encoding="utf-8")

    failure_kind, throttled_until = update_worker_record_with_failure_classification(
        sessions_dir, 42, adapter_kind="api"
    )

    assert failure_kind == "provider_auth"
    assert throttled_until is not None

    updated_sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert updated_sidecar["failure_kind"] == "provider_auth"


def test_update_worker_record_api_sidecar_path_correct(tmp_path: Path) -> None:
    """Issue #484: adapter_kind='api' reads issue-<n>.api.json, not .claude.json."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    # Write BOTH a claude-code and api sidecar for issue-42; the api call must
    # read the api one.
    claude_sidecar = sessions_dir / "issue-42.claude.json"
    claude_sidecar.write_text(
        json.dumps(
            {
                "issue_number": 42,
                "log_path": str(sessions_dir / "issue-42.claude.log"),
                "error": None,
            }
        ),
        encoding="utf-8",
    )
    api_sidecar = sessions_dir / "issue-42.api.json"
    api_sidecar.write_text(
        json.dumps(
            {
                "issue_number": 42,
                "log_path": str(sessions_dir / "issue-42.claude.log"),
                "error": None,
            }
        ),
        encoding="utf-8",
    )

    log_path = sessions_dir / "issue-42.claude.log"
    log_path.write_text("Error: 401 Unauthorized\n", encoding="utf-8")

    failure_kind, _ = update_worker_record_with_failure_classification(
        sessions_dir, 42, adapter_kind="api"
    )

    assert failure_kind == "provider_auth"
    # The api sidecar was updated; the claude-code sidecar was not.
    api_updated = json.loads(api_sidecar.read_text(encoding="utf-8"))
    assert api_updated["failure_kind"] == "provider_auth"
    claude_updated = json.loads(claude_sidecar.read_text(encoding="utf-8"))
    assert "failure_kind" not in claude_updated


def test_classify_session_failure_provider_auth_numeric_substring_no_false_positive(
    tmp_path: Path,
) -> None:
    """Issue #484 review finding: the bare 401/403 codes are anchored with word
    boundaries so a coincidental numeric substring in an unrelated log tail
    (e.g. "error code 14013", "4034 files processed", "issue #4019") cannot
    trip a false-positive 24h provider_auth cooldown. Every other pattern in
    the file matches natural-language phrases; without \\b the bare codes were
    the sole false-positive vector.
    """
    from charlie_work.claude_code import _classify_session_failure

    # Each tail contains "401" or "403" only as a substring of a larger number,
    # with no standalone HTTP status code and no auth-related phrasing.
    false_positive_tails = [
        "Processed 4034 files in 14013 ms.\n",
        "Error code 14013: connection reset by peer.\n",
        "Issue #4019 closed, PR #4032 merged.\n",
        "Remaining tokens: 4019, cache hits: 4031.\n",
    ]
    for tail in false_positive_tails:
        log_path = tmp_path / "session.claude.log"
        log_path.write_text(tail, encoding="utf-8")

        failure_kind, throttled_until = _classify_session_failure(log_path, adapter_kind="api")

        assert failure_kind is None, (
            f"numeric substring falsely matched provider_auth for tail: {tail!r}"
        )
        assert throttled_until is None


def test_classify_session_failure_provider_auth_word_boundary_still_matches(
    tmp_path: Path,
) -> None:
    """Issue #484 review finding: word boundaries must not regress the real
    matches — a standalone 401/403 (delimited by non-word characters: spaces,
    punctuation, start/end of string) still classifies as provider_auth.
    """
    from charlie_work.claude_code import _classify_session_failure

    real_auth_tails = [
        "Error: 401 Unauthorized\n",
        "HTTP 403 Forbidden\n",
        "status=401, message=invalid key\n",
        "401\n",
        "code:403\n",
    ]
    for tail in real_auth_tails:
        log_path = tmp_path / "session.claude.log"
        log_path.write_text(tail, encoding="utf-8")

        failure_kind, throttled_until = _classify_session_failure(log_path, adapter_kind="api")

        assert failure_kind == "provider_auth", (
            f"real auth tail no longer matched for tail: {tail!r}"
        )
        assert throttled_until is not None


def test_classify_session_failure_provider_suspended_insufficient_balance(
    tmp_path: Path,
) -> None:
    """Issue #1342: a provider account-suspension / insufficient-balance response
    classifies as ``provider_suspended`` (terminal) for api sessions, not as a
    transient rate-limit."""
    from charlie_work.claude_code import _classify_session_failure

    log_path = tmp_path / "session.claude.log"
    # Verbatim Moonshot suspension message observed 2026-08-18 (issue #1342).
    log_path.write_text(
        "Working...\n"
        "Error: suspended due to insufficient balance, please recharge your "
        "account or check your plan and billing details.\n",
        encoding="utf-8",
    )

    failure_kind, throttled_until = _classify_session_failure(log_path, adapter_kind="api")

    assert failure_kind == "provider_suspended"
    # Terminal failure: no cooldown window (it will not self-heal).
    assert throttled_until is None


def test_classify_session_failure_provider_suspended_account_suspended(
    tmp_path: Path,
) -> None:
    """Issue #1342: a generic ``account suspended`` billing message also
    classifies as ``provider_suspended`` (matched by semantics, not full-string)."""
    from charlie_work.claude_code import _classify_session_failure

    log_path = tmp_path / "session.claude.log"
    log_path.write_text(
        "Error: Your account is suspended. Please update your billing details.\n",
        encoding="utf-8",
    )

    failure_kind, throttled_until = _classify_session_failure(log_path, adapter_kind="api")

    assert failure_kind == "provider_suspended"
    assert throttled_until is None


def test_classify_session_failure_provider_suspended_not_for_claude_code(
    tmp_path: Path,
) -> None:
    """Issue #1342: suspension classification is api-only (mirrors provider_auth)
    — a claude-code session with the same message is not classified."""
    from charlie_work.claude_code import _classify_session_failure

    log_path = tmp_path / "session.claude.log"
    log_path.write_text(
        "Error: suspended due to insufficient balance, please recharge.\n",
        encoding="utf-8",
    )

    # Default adapter_kind="claude-code" — suspension patterns are not checked.
    failure_kind, throttled_until = _classify_session_failure(log_path)

    assert failure_kind is None
    assert throttled_until is None


def test_classify_session_failure_provider_suspended_takes_precedence_over_throttle(
    tmp_path: Path,
) -> None:
    """Issue #1342: a suspension signature must win over a coincidental
    rate-limit phrase in the same tail, so the session is not retried as a
    transient throttle."""
    from charlie_work.claude_code import _classify_session_failure

    log_path = tmp_path / "session.claude.log"
    log_path.write_text(
        "Error: rate limit exceeded.\n"
        "Error: suspended due to insufficient balance, please recharge your account.\n",
        encoding="utf-8",
    )

    failure_kind, throttled_until = _classify_session_failure(log_path, adapter_kind="api")

    assert failure_kind == "provider_suspended"
    assert throttled_until is None


def test_classify_session_failure_genuine_429_keeps_rate_limited(tmp_path: Path) -> None:
    """Issue #1342 acceptance criterion 4: a genuine transient 429 rate-limit
    keeps the existing ``rate_limited`` backoff behavior (no suspension match)."""
    from charlie_work.claude_code import _classify_session_failure

    log_path = tmp_path / "session.claude.log"
    log_path.write_text(
        "Working...\n"
        "Error: Reached overall message rate limit. Please try again later. "
        "Your limit will reset in 10 minutes.\n",
        encoding="utf-8",
    )

    failure_kind, throttled_until = _classify_session_failure(log_path, adapter_kind="api")

    assert failure_kind == "rate_limited"
    assert throttled_until is not None


def test_classify_session_failure_provider_suspended_quoted_prose_not_matched(
    tmp_path: Path,
) -> None:
    """PR #1426 round-2 review: the suspension phrase appearing only as
    quoted/reviewed prose (not the session's actual terminal error) must NOT
    classify as ``provider_suspended``. The structural anchor requires the
    billing phrase to co-occur on the same log line with an HTTP 402 status or
    a CLI ``Error:``/``API Error:`` line prefix; prose/code that merely quotes
    the trigger phrase does not start with ``Error:`` and so is not treated as
    a real API error. This is the self-inflicted-misclassification guard: a
    worker reviewing this very fix must not be classified as suspended."""
    from charlie_work.claude_code import _classify_session_failure

    log_path = tmp_path / "session.claude.log"
    # The session's actual terminal error is a normal test failure; the
    # suspension trigger appears only inside a code-string quote and a prose
    # sentence — neither line starts with ``Error:`` nor carries a 402.
    log_path.write_text(
        "Reviewing PR #1426...\n"
        '    log_path.write_text("Error: suspended due to insufficient '
        'balance, please recharge your account")\n'
        "The regex matches `suspended due to insufficient balance` and "
        "`recharge your account` phrases.\n"
        "Ran tests.\n"
        "Error: test failed: assertion error in test_worker.py\n",
        encoding="utf-8",
    )

    failure_kind, throttled_until = _classify_session_failure(log_path, adapter_kind="api")

    # Not provider_suspended — the quoted/prose lines have no structural
    # anchor on the same line, and the real terminal error is a test failure
    # (no billing phrase on that line either).
    assert failure_kind is None
    assert throttled_until is None


def test_classify_session_failure_provider_suspended_402_status_anchor(
    tmp_path: Path,
) -> None:
    """PR #1426 round-2 review: an HTTP 402 (Payment Required) status on the
    same line as a billing phrase is a valid structural anchor — the canonical
    billing-suspension status code, distinct from any prose phrase."""
    from charlie_work.claude_code import _classify_session_failure

    log_path = tmp_path / "session.claude.log"
    log_path.write_text(
        "Working...\n"
        "Request failed: 402 Payment Required - insufficient balance, please "
        "recharge your account.\n",
        encoding="utf-8",
    )

    failure_kind, throttled_until = _classify_session_failure(log_path, adapter_kind="api")

    assert failure_kind == "provider_suspended"
    assert throttled_until is None
