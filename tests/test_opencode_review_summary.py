"""``_extract_review_session_summary`` on opencode ``--format json`` logs.

Split out of ``tests/test_opencode_fate.py`` (file-size ratchet, 800-line
cap): the miss-summary classification of an opencode reviewer that died
without a verdict -- real turn/tool counts, decoded-text comments, and the
provider-throttle/quota boundary that keeps a zero-work quota death from
being posted as a substantial-work summary (which would defeat the #583
throttle-rollback guard in ``stalled_review_reap``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

FINAL_ERROR = (
    '{"type":"error","error":{"name":"APIError","data":{"message":"Usage limit reached",'
    '"statusCode":429,"isRetryable":true,"responseBody":"{\\"type\\":\\"error\\",'
    '\\"error\\":{\\"type\\":\\"GoUsageLimitError\\"}}"}}}'
)


def test_review_session_summary_decodes_opencode_log(tmp_path: Path) -> None:
    """A worked opencode reviewer that died without a verdict is classified
    on its real turn/tool counts (not launch_failed), and the PR comment
    carries decoded text, not raw JSONL."""
    from charlie_work.verdict_parsing import (
        REVIEW_MISS_DIED_MID_SESSION,
        _extract_review_session_summary,
    )

    log = tmp_path / "issue-7-review.claude.log"
    log.write_text(
        "\n".join(
            [
                '{"type":"step_start","part":{"id":"s1"}}',
                '{"type":"text","part":{"text":"reviewing the diff"}}',
                '{"type":"tool_use","part":{"tool":"bash"}}',
                '{"type":"step_finish","part":{}}',
                '{"type":"step_start","part":{"id":"s2"}}',
                '{"type":"text","part":{"text":"final analysis text"}}',
            ]
        ),
        encoding="utf-8",
    )
    outcome = _extract_review_session_summary(tmp_path / "missing.events.jsonl", log, 10)
    assert outcome is not None
    assert outcome.turn_count == 2
    assert outcome.tool_call_count == 1
    assert outcome.reason == REVIEW_MISS_DIED_MID_SESSION
    assert "final analysis text" in outcome.text
    assert '{"type"' not in outcome.text  # no raw JSONL dumped into the comment


def test_review_session_summary_opencode_quota_death_at_first_step_is_launch_failed(
    tmp_path: Path,
) -> None:
    """step_start + provider-quota error, zero tool calls: the reviewer died
    inside its first provider call. Must classify launch_failed --
    did_substantial_work=True here would post a 'substantial work' summary,
    set review_turn_limit_summary_posted, and burn the PR's review budget by
    defeating the #583 throttle-rollback guard in stalled_review_reap."""
    from charlie_work.verdict_parsing import (
        REVIEW_MISS_LAUNCH_FAILED,
        _extract_review_session_summary,
    )

    log = tmp_path / "issue-7-review.claude.log"
    log.write_text(
        "\n".join(['{"type":"step_start","part":{"id":"s1"}}', FINAL_ERROR]),
        encoding="utf-8",
    )
    outcome = _extract_review_session_summary(tmp_path / "missing.events.jsonl", log, 10)
    assert outcome is not None
    assert outcome.reason == REVIEW_MISS_LAUNCH_FAILED
    assert outcome.did_substantial_work is False


def test_review_session_summary_opencode_quota_death_after_text_is_launch_failed(
    tmp_path: Path,
) -> None:
    """A text event before the quota error keeps turn_count at 1, so the
    0/0 launch_failed branch no longer reaches it -- the provider_error_digest
    check must: the broad throttle/quota markers are safe there because the
    digest strips assistant text and tool output."""
    from charlie_work.verdict_parsing import (
        REVIEW_MISS_LAUNCH_FAILED,
        _extract_review_session_summary,
    )

    log = tmp_path / "issue-7-review.claude.log"
    log.write_text(
        "\n".join(
            [
                '{"type":"step_start","part":{"id":"s1"}}',
                '{"type":"text","part":{"text":"starting review"}}',
                FINAL_ERROR,
            ]
        ),
        encoding="utf-8",
    )
    outcome = _extract_review_session_summary(tmp_path / "missing.events.jsonl", log, 10)
    assert outcome is not None
    assert outcome.turn_count == 1
    assert outcome.tool_call_count == 0
    assert outcome.reason == REVIEW_MISS_LAUNCH_FAILED
    assert outcome.did_substantial_work is False


# opencode 2.x wire shapes (captured from the installed 2.3.0 binary,
# 2026-11-18): the flat structured error and the {"type":"retry"} event that
# carries it while opencode sleeps out retry-after. A quota death after text
# must classify launch_failed on the 2.x shapes too -- did_substantial_work
# would set review_turn_limit_summary_posted and defeat the #583 guard.
_FINAL_ERROR_V2 = (
    '{"type":"error","error":{"type":"provider.quota",'
    '"message":"Usage limit reached","status":429,'
    '"response":{"body":"{\\"error\\":{\\"message\\":\\"Usage limit reached\\",'
    '\\"type\\":\\"GoUsageLimitError\\"}}"}},"retryAfterMs":3600000}'
)
_RETRY_EVENT_V2 = (
    '{"type":"retry","messageID":"msg_1","attempt":1,"nextRetryAt":1763473200000,'
    '"error":{"type":"provider.quota","message":"Usage limit reached","status":429,'
    '"response":{"body":"{\\"error\\":{\\"message\\":\\"Usage limit reached\\",'
    '\\"type\\":\\"GoUsageLimitError\\"}}"}},"retryAfterMs":3600000}'
)
_PRINT_LOG_ERROR_V2 = (
    "timestamp=2026-11-18T11:45:00.000Z level=ERROR run=ses_x "
    'message="Failed to drain Session" cause="AI.Error: Usage limit reached"'
)


@pytest.mark.parametrize(
    "death_line",
    [
        pytest.param(_FINAL_ERROR_V2, id="v2-flat-error-event"),
        pytest.param(_RETRY_EVENT_V2, id="v2-retry-event"),
        pytest.param(_PRINT_LOG_ERROR_V2, id="v2-cause-print-log"),
    ],
)
def test_review_session_summary_v2_quota_death_after_text_is_launch_failed(
    tmp_path: Path, death_line: str
) -> None:
    from charlie_work.verdict_parsing import (
        REVIEW_MISS_LAUNCH_FAILED,
        _extract_review_session_summary,
    )

    log = tmp_path / "issue-7-review.claude.log"
    log.write_text(
        "\n".join(
            [
                '{"type":"step_start","part":{"id":"s1"}}',
                '{"type":"text","part":{"text":"starting review"}}',
                death_line,
            ]
        ),
        encoding="utf-8",
    )
    outcome = _extract_review_session_summary(tmp_path / "missing.events.jsonl", log, 10)
    assert outcome is not None
    assert outcome.reason == REVIEW_MISS_LAUNCH_FAILED
    assert outcome.did_substantial_work is False
    assert "Usage limit reached" in outcome.text


def test_review_session_summary_opencode_non_quota_error_with_no_tools_is_mid_session(
    tmp_path: Path,
) -> None:
    """Control: a 0-tool-call death on a NON-throttle error must stay
    died_mid_session -- only provider throttle/quota signatures reclassify."""
    from charlie_work.verdict_parsing import (
        REVIEW_MISS_DIED_MID_SESSION,
        _extract_review_session_summary,
    )

    log = tmp_path / "issue-7-review.claude.log"
    log.write_text(
        "\n".join(
            [
                '{"type":"step_start","part":{"id":"s1"}}',
                '{"type":"text","part":{"text":"inspecting"}}',
                '{"type":"error","error":{"name":"APIError","data":{"message":'
                '"connection reset by peer","statusCode":0}}}',
            ]
        ),
        encoding="utf-8",
    )
    outcome = _extract_review_session_summary(tmp_path / "missing.events.jsonl", log, 10)
    assert outcome is not None
    assert outcome.reason == REVIEW_MISS_DIED_MID_SESSION


def test_review_session_summary_opencode_tool_only_death_dumps_no_jsonl(
    tmp_path: Path,
) -> None:
    """step_start + tool_use, killed before any text/error event: recognized
    opencode JSONL with no recovered text must NOT run the raw-prose fallback
    (it would dump raw JSONL lines into the PR comment)."""
    from charlie_work.verdict_parsing import (
        REVIEW_MISS_DIED_MID_SESSION,
        _extract_review_session_summary,
    )

    log = tmp_path / "issue-7-review.claude.log"
    log.write_text(
        "\n".join(
            [
                '{"type":"step_start","part":{"id":"s1"}}',
                '{"type":"tool_use","part":{"tool":"read","state":{"output":"x"}}}',
            ]
        ),
        encoding="utf-8",
    )
    outcome = _extract_review_session_summary(tmp_path / "missing.events.jsonl", log, 10)
    assert outcome is not None
    assert outcome.reason == REVIEW_MISS_DIED_MID_SESSION
    assert '{"type"' not in outcome.text


PRINT_LOG_ERROR = (
    "timestamp=2026-10-06T12:00:00.000Z level=ERROR "
    'message="stream error" error.error="AI_APICallError: Usage limit reached"'
)


def _oc_review_outcome(tmp_path: Path, jsonl: str, max_turns: int):
    """Model a production opencode launch: ``launch_opencode_worker`` goes
    through ``launch_claude_worker`` with ``cli_pins`` and leaves
    ``tee_stream_json=False``, so opencode's ``--format json`` stdout is the
    worker log itself and no ``*.events.jsonl`` is ever written. The summary
    path must therefore recover everything from the log; the events path
    passed in does not exist, as in production."""
    from charlie_work.workflow import _extract_review_session_summary

    events_path = tmp_path / "issue-100-review.events.jsonl"  # never written
    log_path = tmp_path / "issue-100-review.claude.log"
    log_path.write_text(jsonl, encoding="utf-8")
    return _extract_review_session_summary(events_path, log_path, max_turns)


def test_extract_review_session_summary_bare_step_start_is_launch_failed(
    tmp_path: Path,
) -> None:
    """A watchdog kill after step_start before any token or tool call:
    the open step is discounted at EOF, so a reviewer that produced ZERO
    text and ZERO tool calls classifies launch_failed -- not died_mid_session
    (which would set review_turn_limit_summary_posted on the PR)."""
    outcome = _oc_review_outcome(
        tmp_path, '{"type":"step_start","part":{"id":"s1"}}\n', max_turns=10
    )
    assert outcome is not None
    assert outcome.turn_count == 0
    assert outcome.tool_call_count == 0
    assert outcome.reason == "launch_failed"
    assert outcome.did_substantial_work is False


def test_extract_review_session_summary_print_log_only_error_is_launch_failed(
    tmp_path: Path,
) -> None:
    """The primary opencode-Go 429 death mode: step_start + column-0
    --print-logs level=ERROR, killed during the retry-after sleep before any
    {"type":"error"} event. Two invariants: the output-less step counts zero
    turns, and the digest fallback surfaces the real stream error in the
    comment instead of the "no assistant text" placeholder."""
    outcome = _oc_review_outcome(
        tmp_path,
        '{"type":"step_start"}\n' + PRINT_LOG_ERROR + "\n",
        max_turns=10,
    )
    assert outcome is not None
    assert outcome.turn_count == 0
    assert outcome.tool_call_count == 0
    assert outcome.reason == "launch_failed"
    assert outcome.did_substantial_work is False
    assert "Usage limit reached" in outcome.text
    assert "no assistant text" not in outcome.text


def test_extract_review_session_summary_provider_death_beats_max_turns_1(
    tmp_path: Path,
) -> None:
    """step_start -> text -> provider error with max_turns == 1: turn_count
    (1) >= max_turns (1), but the 0-tool-call provider-death guard must run
    FIRST -- otherwise a quota death is misclassified REVIEW_MISS_TURN_LIMIT
    and burns the PR's attempt budget (#583 rollback guard)."""
    outcome = _oc_review_outcome(
        tmp_path,
        '{"type":"step_start","part":{"id":"s1"}}\n'
        '{"type":"text","part":{"text":"reading the diff"}}\n' + FINAL_ERROR + "\n",
        max_turns=1,
    )
    assert outcome is not None
    assert outcome.turn_count == 1
    assert outcome.tool_call_count == 0
    assert outcome.reason == "launch_failed"
    assert outcome.did_substantial_work is False


def test_claude_turn_limit_precedence_is_unchanged(tmp_path: Path) -> None:
    """The opencode digest branch moved AHEAD of the turn-limit branch; the
    pre-existing claude-code raw-tail guard must not follow it.

    A claude-code reviewer capped at max_turns, with zero tool calls and a
    narrow session-limit notice in its log tail, has always been classified
    ``turn_limit`` (the tail guard sits after the turn-limit branch for that
    harness). Reordering both guards together would silently reclassify that
    session as ``launch_failed`` -- a cross-harness behaviour change in an
    opencode fix -- so pin the claude ordering explicitly.
    """
    from charlie_work.verdict_parsing import (
        REVIEW_MISS_TURN_LIMIT,
        _extract_review_session_summary,
    )

    events = tmp_path / "issue-9-review.events.jsonl"
    # One assistant turn, no tool calls: a real claude-code reviewer tees its
    # stream to events.jsonl, which is where turn counts come from.
    events.write_text(
        '{"type":"assistant","message":{"content":[{"type":"text","text":"analysis"}]}}\n',
        encoding="utf-8",
    )
    log = tmp_path / "issue-9-review.claude.log"
    log.write_text(
        "You've hit your session limit · resets 4:40pm (America/Los_Angeles)\n",
        encoding="utf-8",
    )
    outcome = _extract_review_session_summary(events, log, max_turns=1)
    assert outcome is not None
    assert outcome.turn_count == 1
    assert outcome.tool_call_count == 0
    assert outcome.reason == REVIEW_MISS_TURN_LIMIT


def test_extract_review_session_summary_retry_steps_then_error_is_launch_failed(
    tmp_path: Path,
) -> None:
    """Internal provider retry before a fatal error: two step_starts, zero
    output -- every output-less attempt is discounted."""
    outcome = _oc_review_outcome(
        tmp_path,
        '{"type":"step_start"}\n'
        + PRINT_LOG_ERROR
        + '\n{"type":"step_start"}\n'
        + FINAL_ERROR
        + "\n",
        max_turns=10,
    )
    assert outcome is not None
    assert outcome.turn_count == 0
    assert outcome.tool_call_count == 0
    assert outcome.reason == "launch_failed"


def test_extract_review_session_summary_retry_killed_mid_attempt_keeps_digest(
    tmp_path: Path,
) -> None:
    """Watchdog kill during attempt 2's provider call: the trailing bare
    step_start must not erase attempt 1's print-log stream error from the
    digest -- it was not progress, just the retry marker."""
    outcome = _oc_review_outcome(
        tmp_path,
        '{"type":"step_start"}\n' + PRINT_LOG_ERROR + '\n{"type":"step_start"}\n',
        max_turns=10,
    )
    assert outcome is not None
    assert outcome.turn_count == 0
    assert outcome.reason == "launch_failed"
    assert "Usage limit reached" in outcome.text


def test_extract_review_session_summary_post_error_text_flush_keeps_digest(
    tmp_path: Path,
) -> None:
    """SessionProcessor.cleanup flushes {"type":"text","part":{"text":""}}
    after an aborted stream; the empty part is not progress and must not
    erase the stream error from the digest."""
    outcome = _oc_review_outcome(
        tmp_path,
        '{"type":"step_start"}\n' + PRINT_LOG_ERROR + '\n{"type":"text","part":{"text":""}}\n',
        max_turns=10,
    )
    assert outcome is not None
    assert outcome.turn_count == 0
    assert outcome.reason == "launch_failed"
    assert "Usage limit reached" in outcome.text


def test_extract_review_session_summary_blank_text_does_not_hide_digest(
    tmp_path: Path,
) -> None:
    """Whitespace-only part.text must not bypass the empty-text guard: the
    provider-error digest still lands in the summary."""
    outcome = _oc_review_outcome(
        tmp_path,
        '{"type":"step_start"}\n{"type":"text","part":{"text":"   "}}\n' + PRINT_LOG_ERROR + "\n",
        max_turns=10,
    )
    assert outcome is not None
    # The blank part marks no step output: turn_count stays 0, so the death
    # is launch_failed (not died_mid_session) and the provider digest is
    # surfaced instead of the "no assistant text" placeholder.
    assert outcome.turn_count == 0
    assert outcome.reason == "launch_failed"
    assert "Usage limit reached" in outcome.text
    assert "no assistant text" not in outcome.text


def test_provider_digest_implies_recoverable_progress() -> None:
    """The digest branch's ``oc_progress is not None`` gate is a type guard,
    not a second condition.

    ``_extract_review_session_summary`` only consults the broad-marker digest
    when ``recover_review_log_texts`` reported an opencode log. A non-empty
    digest requires an error/retry/print-log record, and every one of those
    makes ``session_progress`` return a value -- so the gate cannot hide a
    real quota death. Pinned here because the branch's comment claims it.
    """
    from charlie_work.opencode_log import provider_error_digest
    from charlie_work.verdict_log_extraction import recover_review_log_texts

    for log in (
        FINAL_ERROR,
        '{"type":"retry","error":{"type":"provider.quota",'
        '"message":"Usage limit"},"retryAfterMs":3600000}',
        PRINT_LOG_ERROR,
    ):
        assert provider_error_digest(log) != ""
        _texts, oc = recover_review_log_texts(log)
        assert oc is not None


def test_default_throttle_markers_are_resolved_per_call_not_at_import() -> None:
    """The broad marker fallbacks must not be module-level OrchestratorConfig
    instances.

    verdict_log_extraction sits on verdict_parsing's import path; building a
    config at import time makes every importer pay for it and freezes the
    defaults before a caller's runtime config could take effect. The accessors
    resolve on call -- and the single production call site passes its own
    config's markers anyway.
    """
    import charlie_work.verdict_log_extraction as vle
    from charlie_work.config import OrchestratorConfig

    assert not hasattr(vle, "_DEFAULT_REVIEW_THROTTLE_ERROR_MARKERS")
    assert not hasattr(vle, "_DEFAULT_REVIEW_QUOTA_ERROR_MARKERS")
    runtime = OrchestratorConfig().runtime
    assert vle._default_review_throttle_error_markers() == tuple(runtime.throttle_error_markers)
    assert vle._default_review_quota_error_markers() == tuple(runtime.quota_error_markers)
