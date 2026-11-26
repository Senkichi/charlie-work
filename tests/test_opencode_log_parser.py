"""``opencode_log`` parser unit tests: provider-error digest, last text
event, and session progress on ``--format json`` / ``--print-logs`` output.

Split out of test_opencode_fate.py (file-size ratchet): these exercise the
log-parser functions directly, not the fate/classification plumbing (which
stays in test_opencode_fate.py). The small FINAL_ERROR / PRINT_LOG_ERROR /
TOOL_OUTPUT_ONLY specimens are duplicated on purpose -- per-file fixtures,
same convention as test_opencode_review_summary.py's own PRINT_LOG_ERROR.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from charlie_work.opencode_log import (
    carries_work_output,
    last_text_event,
    opencode_data_dir,
    provider_error_digest,
    session_progress,
)
from charlie_work.opencode_worker import DEFAULT_COMMAND_TEMPLATE

FINAL_ERROR = (
    '{"type":"error","error":{"name":"APIError","data":{"message":"Usage limit reached",'
    '"statusCode":429,"isRetryable":true,"responseBody":"{\\"type\\":\\"error\\",'
    '\\"error\\":{\\"type\\":\\"GoUsageLimitError\\"}}"}}}'
)
PRINT_LOG_ERROR = (
    "timestamp=2026-10-06T12:00:00.000Z level=ERROR "
    'message="stream error" error.error="AI_APICallError: Usage limit reached"'
)
# opencode 2.x wire shapes, captured from the installed 2.3.0 binary
# (2026-11-18): the structured error is flat, the print-log line carries
# cause= instead of error.error=, and a failed attempt emits a retry event
# carrying the SAME error before the retry sleep.
FINAL_ERROR_V2 = (
    '{"type":"error","error":{"type":"provider.quota",'
    '"message":"Usage limit reached -- you have used your monthly budget",'
    '"status":429,"response":{"body":"{\\"error\\":{\\"message\\":'
    '\\"Usage limit reached\\",\\"type\\":\\"GoUsageLimitError\\"}}"}},"retryAfterMs":3600000}'
)
RETRY_EVENT_V2 = (
    '{"type":"retry","messageID":"msg_1","attempt":1,'
    '"nextRetryAt":1763473200000,"error":{"type":"provider.quota",'
    '"message":"Usage limit reached","status":429,'
    '"response":{"body":"{\\"error\\":{\\"message\\":\\"Usage limit reached\\",'
    '\\"type\\":\\"GoUsageLimitError\\"}}"}},"retryAfterMs":3600000}'
)
PRINT_LOG_ERROR_V2 = (
    'timestamp=2026-11-18T11:45:00.000Z level=ERROR run=ses_x message="Failed to drain Session" '
    'cause="AI.Error: Usage limit reached -- you have used your monthly budget"'
)
# "usage limit" appears only inside a tool_use event (the worker read throttle code).
TOOL_OUTPUT_ONLY = (
    '{"type":"step_start"}\n'
    '{"type":"tool_use","part":{"tool":"read","state":{"output":'
    '"if \\"usage limit\\" in tail: rate limit exceeded; quota exhausted"}}}\n'
)


def _log(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "issue-1.claude.log"
    path.write_text(text, encoding="utf-8")
    return path


# --- opencode_log -----------------------------------------------------------


def test_default_command_template_is_json_stream_with_print_logs() -> None:
    assert DEFAULT_COMMAND_TEMPLATE[:2] == ("opencode", "run")
    assert "--standalone" in DEFAULT_COMMAND_TEMPLATE  # never share the operator's service
    assert "--print-logs" in DEFAULT_COMMAND_TEMPLATE
    assert "--format" in DEFAULT_COMMAND_TEMPLATE
    # opencode 2.3.0 rejects the 1.x-era uppercase value (choices are
    # lowercase: all|trace|debug|info|warn|warning|error|fatal|none) -- the
    # wrong case killed every launch before any request.
    assert "--log-level" in DEFAULT_COMMAND_TEMPLATE
    assert DEFAULT_COMMAND_TEMPLATE[DEFAULT_COMMAND_TEMPLATE.index("--log-level") + 1] == "warn"


def test_opencode_data_dir_layout(tmp_path: Path) -> None:
    assert opencode_data_dir(tmp_path, 12) == tmp_path / "opencode-data" / "issue-12"


def test_digest_keeps_final_error_event_with_status_and_body() -> None:
    digest = provider_error_digest('{"type":"step_start"}\n' + FINAL_ERROR + "\n")
    assert "Usage limit reached" in digest
    assert "status=429" in digest
    assert "GoUsageLimitError" in digest


def test_digest_keeps_print_logs_error_lines() -> None:
    digest = provider_error_digest(PRINT_LOG_ERROR + "\n")
    assert "Usage limit reached" in digest


def test_digest_drops_tool_output_and_forged_level_error_in_tool_json() -> None:
    forged = (
        '{"type":"tool_use","part":{"state":{"output":"timestamp=x level=ERROR '
        'error.error=\\"usage limit\\""}}}\n'
    )
    assert provider_error_digest(TOOL_OUTPUT_ONLY) == ""
    assert provider_error_digest(forged) == ""


def test_carries_work_output_is_the_raw_tail_precondition() -> None:
    """carries_work_output marks the logs where a raw-tail marker could be a
    quoted review file rather than a provider message."""
    assert carries_work_output(TOOL_OUTPUT_ONLY) is True
    assert carries_work_output('{"type":"text","part":{"text":"rate limit"}}') is True
    # No output at all: only lifecycle/error records -- the raw tail is safe.
    assert carries_work_output('{"type":"step_start"}\n{"type":"error"}') is False
    # A blank text part is the cleanup flush, not output.
    assert carries_work_output('{"type":"text","part":{"text":"   "}}') is False
    assert carries_work_output("") is False


def test_digest_resets_on_progress_event() -> None:
    """A 429 opencode retried through must not classify a later unrelated
    death -- only real progress (non-blank assistant text or tool_use)
    resets the collected errors."""
    text = PRINT_LOG_ERROR + '\n{"type":"text","part":{"text":"recovered"}}\n'
    assert provider_error_digest(text) == ""
    # A bare step_start is a retry marker, not progress: errors on both
    # sides of it are kept so a kill inside the retried attempt still
    # surfaces attempt 1's failure.
    text = FINAL_ERROR + '\n{"type":"step_start"}\n' + PRINT_LOG_ERROR + "\n"
    assert provider_error_digest(text).count("Usage limit reached") == 2
    # ...but real output between the two errors does reset.
    text = (
        FINAL_ERROR
        + '\n{"type":"step_start"}\n'
        + '{"type":"text","part":{"text":"working"}}\n'
        + PRINT_LOG_ERROR
        + "\n"
    )
    assert provider_error_digest(text).count("Usage limit reached") == 1


# --- last_text_event / session_progress --------------------------------------


def test_last_text_event_returns_final_text_part() -> None:
    log = "\n".join(
        [
            '{"type":"step_start","part":{"id":"s1"}}',
            '{"type":"text","part":{"text":"first draft"}}',
            '{"type":"step_finish","part":{}}',
            '{"type":"step_start","part":{"id":"s2"}}',
            '{"type":"text","part":{"text":"final answer"}}',
            '{"type":"step_finish","part":{}}',
        ]
    )
    assert last_text_event(log) == "final answer"


def test_last_text_event_resets_on_later_step_tool_or_error() -> None:
    """A draft text superseded by a new step/tool call/error is mid-session
    output, not the final answer -- it must not be reaped as the verdict."""
    for tail in (
        '{"type":"step_start","part":{"id":"s2"}}',
        '{"type":"tool_use","part":{"tool":"bash"}}',
        '{"type":"error","error":{"name":"ProviderError","data":{}}}',
    ):
        log = "\n".join(
            [
                '{"type":"step_start","part":{"id":"s1"}}',
                '{"type":"text","part":{"text":"draft verdict here"}}',
                '{"type":"step_finish","part":{}}',
                tail,
            ]
        )
        assert last_text_event(log) is None, tail


def test_last_text_event_keeps_text_before_step_finish() -> None:
    # text -> step_finish is a completed turn: the normal verdict shape.
    log = "\n".join(
        [
            '{"type":"step_start"}',
            '{"type":"text","part":{"text":"the verdict"}}',
            '{"type":"step_finish"}',
        ]
    )
    assert last_text_event(log) == "the verdict"


def test_last_text_event_resets_on_print_log_error_line() -> None:
    """A provider 429 mid-session surfaces as a --print-logs level=ERROR line
    while opencode sleeps out retry-after -- BEFORE the next step_start /
    error event exists. A watchdog kill here must not reap the previous
    turn's draft text as the verdict."""
    log = "\n".join(
        [
            '{"type":"step_start","part":{"id":"s1"}}',
            '{"type":"text","part":{"text":"draft verdict"}}',
            '{"type":"step_finish","part":{}}',
            PRINT_LOG_ERROR,
        ]
    )
    assert last_text_event(log) is None


def test_last_text_event_resets_on_non_stop_step_finish() -> None:
    """step_finish with reason="tool-calls" means the turn CONTINUED into a
    tool call the process then died inside: the text before it was mid-turn
    draft, not the answer."""
    log = "\n".join(
        [
            '{"type":"step_start","part":{"id":"s1"}}',
            '{"type":"text","part":{"text":"mid-turn draft"}}',
            '{"type":"step_finish","part":{"reason":"tool-calls"}}',
        ]
    )
    assert last_text_event(log) is None


@pytest.mark.parametrize("reason", ["stop", "end-turn", "", None])
def test_last_text_event_keeps_text_before_terminal_step_finish(
    reason: str | None,
) -> None:
    part = {} if reason is None else {"reason": reason}
    log = "\n".join(
        [
            '{"type":"step_start","part":{"id":"s1"}}',
            '{"type":"text","part":{"text":"final answer"}}',
            json.dumps({"type": "step_finish", "part": part}),
        ]
    )
    assert last_text_event(log) == "final answer"


def test_last_text_event_skips_malformed_lines() -> None:
    log = "\n".join(
        [
            '{"type":"text","part":{"text":"good"}}',
            '{"type":"text","part":{"text":"TRUNCATED',
        ]
    )
    assert last_text_event(log) == "good"


def test_session_progress_counts_turns_tools_and_texts() -> None:
    log = "\n".join(
        [
            '{"type":"step_start","part":{"id":"s1"}}',
            '{"type":"text","part":{"text":"looking at diff"}}',
            '{"type":"tool_use","part":{"tool":"read"}}',
            '{"type":"tool_use","part":{"tool":"bash"}}',
            '{"type":"text","part":{"text":"final analysis"}}',
            '{"type":"step_finish","part":{}}',
        ]
    )
    progress = session_progress(log)
    assert progress is not None
    assert progress.turn_count == 1
    assert progress.tool_call_count == 2
    assert progress.texts == ("looking at diff", "final analysis")


def test_session_progress_includes_error_events_as_text() -> None:
    progress = session_progress(FINAL_ERROR)
    assert progress is not None
    assert progress.turn_count == 0
    assert progress.tool_call_count == 0
    assert progress.texts and "opencode error:" in progress.texts[0]


def test_session_progress_discounts_step_that_died_before_any_output() -> None:
    """step_start + error with no text/tool_use between them is a quota wall
    inside the step's first provider call, not a completed turn. Counting it
    would let _extract_review_session_summary fake a turn of review work and
    classify the death died_mid_session (did_substantial_work=True), which
    burns the PR's review budget by defeating the #583 rollback guard."""
    log = "\n".join(
        [
            '{"type":"step_start","part":{"id":"s1"}}',
            FINAL_ERROR,
        ]
    )
    progress = session_progress(log)
    assert progress is not None
    assert progress.turn_count == 0
    assert progress.tool_call_count == 0


def test_session_progress_second_error_does_not_double_discount() -> None:
    log = "\n".join(
        [
            '{"type":"step_start","part":{"id":"s1"}}',
            '{"type":"error","error":{"name":"APIError","data":{}}}',
            FINAL_ERROR,
        ]
    )
    progress = session_progress(log)
    assert progress is not None
    assert progress.turn_count == 0


def test_session_progress_step_with_output_counts_before_error() -> None:
    """A step that produced text or a tool call before dying IS a turn of
    attempted work -- only its classification (launch_failed vs
    died_mid_session) depends on the error digest, not the turn count."""
    for output in (
        '{"type":"text","part":{"text":"looking"}}',
        '{"type":"tool_use","part":{"tool":"read"}}',
    ):
        log = "\n".join(['{"type":"step_start","part":{"id":"s1"}}', output, FINAL_ERROR])
        progress = session_progress(log)
        assert progress is not None
        assert progress.turn_count == 1, output


def test_session_progress_bare_step_start_is_not_a_turn() -> None:
    """Watchdog kill / crash after step_start before any output or error
    line: the open step is discounted at EOF -- zero turns, zero tools."""
    progress = session_progress('{"type":"step_start","part":{"id":"s1"}}\n')
    assert progress is not None
    assert progress.turn_count == 0
    assert progress.tool_call_count == 0


def test_session_progress_retry_step_starts_all_discount() -> None:
    """Internal retry: each attempt emits step_start before the HTTP call
    completes; a PRINT_LOG_ERROR between attempts is not step output, so
    N output-less steps followed by a fatal error still count zero turns."""
    log = (
        '{"type":"step_start"}\n'
        + PRINT_LOG_ERROR
        + '\n{"type":"step_start"}\n'
        + FINAL_ERROR
        + "\n"
    )
    progress = session_progress(log)
    assert progress is not None
    assert progress.turn_count == 0
    assert progress.tool_call_count == 0


def test_session_progress_print_log_error_surfaces_digest_text() -> None:
    """The canonical opencode-Go 429 death: step_start + a column-0
    --print-logs level=ERROR line, then killed during retry-after sleep
    before any {"type":"error"} event is emitted. The step is discounted AND
    texts falls back to the provider digest -- the real stream error reaches
    the caller instead of the "no assistant text" placeholder."""
    progress = session_progress('{"type":"step_start"}\n' + PRINT_LOG_ERROR + "\n")
    assert progress is not None
    assert progress.turn_count == 0
    assert progress.tool_call_count == 0
    assert any("Usage limit reached" in t for t in progress.texts)


@pytest.mark.parametrize("blank", ["", "   ", " \t "])
def test_session_progress_ignores_blank_text_parts(blank: str) -> None:
    """Empty/whitespace-only part.text must not land in texts: a truthy
    blank would bypass the caller's empty-assistant_texts guard and render a
    blank section in the PR comment. Nor does it mark the step as having
    produced output -- SessionProcessor.cleanup flushes an empty text part
    after an aborted stream, so a blank-only step that dies still counts
    zero turns (launch_failed, not died_mid_session)."""
    log = (
        '{"type":"step_start"}\n'
        + json.dumps({"type": "text", "part": {"text": blank}}, separators=(",", ":"))
        + "\n"
    )
    progress = session_progress(log)
    assert progress is not None
    assert progress.turn_count == 0
    assert progress.texts == ()


def test_session_progress_truncated_text_line_is_not_step_output() -> None:
    """A truncated text event at EOF is unparseable -- it carries no
    recoverable text and must not mark the step as producing output."""
    log = '{"type":"step_start"}\n{"type":"text","part":{"text":"TRUNC'
    progress = session_progress(log)
    assert progress is not None
    assert progress.turn_count == 0
    assert progress.texts == ()


def test_last_text_event_blank_text_does_not_displace_final() -> None:
    """A whitespace-only text part flushed after the real text must not
    overwrite it -- it is a cleanup artifact, not a newer answer."""
    log = "\n".join(
        [
            '{"type":"step_start"}',
            '{"type":"text","part":{"text":"the verdict"}}',
            '{"type":"text","part":{"text":"   "}}',
        ]
    )
    assert last_text_event(log) == "the verdict"


def test_last_text_event_truncated_step_finish_resets() -> None:
    """A truncated step_finish cannot certify the text before it as a
    completed turn -- its reason might have said the turn continued."""
    log = "\n".join(
        [
            '{"type":"step_start"}',
            '{"type":"text","part":{"text":"mid-turn draft"}}',
            '{"type":"step_finish","part":{"reason":"tool-calls',
        ]
    )
    assert last_text_event(log) is None


def test_digest_keeps_stream_error_past_bare_step_start() -> None:
    """Internal retry: attempt 2 emits step_start before its HTTP response;
    a watchdog kill there must not erase attempt 1's stream error from the
    digest -- a bare step_start is not progress."""
    log = '{"type":"step_start"}\n' + PRINT_LOG_ERROR + '\n{"type":"step_start"}\n'
    digest = provider_error_digest(log)
    assert "Usage limit reached" in digest


def test_digest_keeps_stream_error_past_empty_text_flush() -> None:
    """SessionProcessor.cleanup flushes {"type":"text","part":{"text":""}}
    after an aborted stream -- the empty part is not progress and must not
    erase the stream error."""
    log = '{"type":"step_start"}\n' + PRINT_LOG_ERROR + '\n{"type":"text","part":{"text":""}}\n'
    digest = provider_error_digest(log)
    assert "Usage limit reached" in digest


def test_digest_keeps_error_past_step_finish() -> None:
    """step_finish terminates the step whose output already reset the
    digest; after an error it is the last event before the session dies.
    Treating it as progress would erase the only provider-error evidence
    from the digest AND empty the session_progress text fallback with it
    (adversarial review round 2: the GoUsageLimitError death below used to
    classify died_mid_session instead of launch_failed)."""
    log = FINAL_ERROR + '\n{"type":"step_finish","part":{"reason":"error"}}\n'
    assert provider_error_digest(log).count("Usage limit reached") == 1
    # Same for a print-logs record followed by the terminal step_finish:
    # the error text must still reach the caller.
    progress = session_progress(
        '{"type":"step_start"}\n'
        + PRINT_LOG_ERROR
        + '\n{"type":"step_finish","part":{"reason":"stop"}}\n'
    )
    assert progress is not None
    assert any("Usage limit reached" in t for t in progress.texts)


def test_digest_keeps_error_past_unknown_future_kind() -> None:
    """An unrecognized (future opencode) event kind must not silently erase
    collected error records: clearing there loses the classification signal
    outright, while keeping them can only over-report stale errors, which
    the callers' turn/tool-count guards absorb."""
    log = PRINT_LOG_ERROR + '\n{"type":"futurekind"}\n'
    assert provider_error_digest(log) != ""


def test_digest_still_resets_on_tool_use() -> None:
    """tool_use is real work product and must still reset the digest -- the
    non-erasure fix above must not keep stale errors across recovered
    sessions that only made tool calls."""
    log = PRINT_LOG_ERROR + '\n{"type":"tool_use"}\n'
    assert provider_error_digest(log) == ""


def test_session_progress_returns_none_for_claude_stream_json() -> None:
    """A claude log has no opencode-shaped events: callers must fall through
    to their existing plaintext path instead of misattributing the format."""
    claude_log = "\n".join(
        [
            '{"type":"system","subtype":"init"}',
            '{"type":"assistant","message":{"content":[{"type":"text","text":"hi"}]}}',
            '{"type":"result","result":"done"}',
        ]
    )
    assert session_progress(claude_log) is None


# --- verdict extraction on opencode JSONL -------------------------------------


def test_extract_verdict_from_log_text_decodes_opencode_jsonl() -> None:
    """The verdict fence is JSON-escaped inside part.text -- invisible to the
    plaintext scan, reachable only through the last_text_event decode."""
    from charlie_work.verdict_log_extraction import extract_verdict_from_log_text

    fence = (
        'Final verdict:\n```json\n{\n  "decision": "approved",\n'
        '  "summary": "lgtm",\n  "required_changes": []\n}\n```\n'
    )
    log = "\n".join(
        [
            '{"type":"step_start","part":{"id":"s1"}}',
            json.dumps(
                {"type": "text", "part": {"text": "analysis first\n\n" + fence}},
                separators=(",", ":"),
            ),
            '{"type":"step_finish","part":{}}',
        ]
    )
    verdict = extract_verdict_from_log_text(log)
    assert verdict is not None
    assert verdict["decision"] == "approved"
    assert verdict["summary"] == "lgtm"


def test_extract_verdict_ignores_draft_before_failed_final_step() -> None:
    """text(draft fence) -> step_finish -> step_start -> tool_use -> error:
    the draft is stale mid-session output, not the verdict."""
    from charlie_work.verdict_log_extraction import extract_verdict_from_log_text

    draft = '```json\n{"decision":"approved","summary":"draft","required_changes":[]}\n```'
    log = "\n".join(
        [
            json.dumps(
                {"type": "text", "part": {"text": "maybe:\n" + draft}},
                separators=(",", ":"),
            ),
            '{"type":"step_finish","part":{}}',
            '{"type":"step_start","part":{"id":"s2"}}',
            '{"type":"tool_use","part":{"tool":"bash"}}',
            '{"type":"error","error":{"name":"ProviderError","data":{}}}',
        ]
    )
    assert extract_verdict_from_log_text(log) is None


def test_extract_verdict_prefers_final_text_over_earlier_echo() -> None:
    """An earlier text echoing the prompt's example verdict loses to the
    final text event's real verdict."""
    from charlie_work.verdict_log_extraction import extract_verdict_from_log_text

    echo = '```json\n{"decision":"approved","summary":"echo","required_changes":[]}\n```'
    real = '```json\n{"decision":"request_changes","summary":"real","required_changes":["x"]}\n```'
    log = "\n".join(
        [
            json.dumps(
                {"type": "text", "part": {"text": "the format:\n" + echo}},
                separators=(",", ":"),
            ),
            '{"type":"step_finish","part":{}}',
            '{"type":"step_start","part":{"id":"s2"}}',
            '{"type":"tool_use","part":{"tool":"read"}}',
            json.dumps(
                {"type": "text", "part": {"text": "Verdict:\n" + real}},
                separators=(",", ":"),
            ),
            '{"type":"step_finish","part":{}}',
        ]
    )
    verdict = extract_verdict_from_log_text(log)
    assert verdict is not None
    assert verdict["decision"] == "request_changes"


# --- opencode 2.x wire shapes (installed 2.3.0 binary, 2026-11-18) ------------


def test_digest_decodes_v2_flat_error_event() -> None:
    """The 2.x structured error is flat (type/message/status/response.body)
    with the retry-after at the event top level -- the 1.x-only reader
    produced "name= status= retryable= message= body=" and every quota
    marker missed (the classification-blind bug this normalizer closes)."""
    digest = provider_error_digest('{"type":"step_start"}\n' + FINAL_ERROR_V2 + "\n")
    assert "provider.quota" in digest
    assert "status=429" in digest
    assert "Usage limit reached" in digest
    assert "GoUsageLimitError" in digest
    assert "retry-after-ms=3600000" in digest


def test_digest_collects_v2_retry_events() -> None:
    """A 2.x session killed inside the retry sleep may never emit the
    terminal error event -- the retry events are the only provider-error
    records and the digest must keep them."""
    log = "\n".join(
        ['{"type":"step_start","part":{"id":"s1"}}', RETRY_EVENT_V2, '{"type":"step_start"}']
    )
    digest = provider_error_digest(log)
    assert "provider.quota" in digest
    assert "retry-after-ms=3600000" in digest
    # Real progress still resets the collected retry errors.
    log = RETRY_EVENT_V2 + '\n{"type":"text","part":{"text":"recovered"}}\n'
    assert provider_error_digest(log) == ""


def test_digest_decodes_v2_print_log_cause_line() -> None:
    """2.x print-logs carry cause="AI.Error: ..." instead of
    error.error="..." -- both spellings feed the digest."""
    digest = provider_error_digest(PRINT_LOG_ERROR_V2 + "\n")
    assert "Usage limit reached" in digest


@pytest.mark.parametrize(
    "log_text",
    [
        pytest.param('{"type":"step_start"}\n' + FINAL_ERROR_V2 + "\n", id="flat-error-event"),
        pytest.param(RETRY_EVENT_V2 + "\n", id="retry-event"),
        pytest.param(PRINT_LOG_ERROR_V2 + "\n", id="cause-print-log"),
    ],
)
def test_v2_shapes_classify_quota_and_throttle(tmp_path: Path, log_text: str) -> None:
    """The broad throttle/quota marker sets (safe on digest-only matching)
    must fire on the 2.x error records -- the e2e reap/review-lane boundary
    depends on _log_died_on_provider_error seeing them."""
    from charlie_work.config import OrchestratorConfig
    from charlie_work.verdict_log_extraction import _log_died_on_provider_error

    runtime = OrchestratorConfig().runtime
    path = _log(tmp_path, log_text)
    assert _log_died_on_provider_error(
        path, runtime.throttle_error_markers, runtime.quota_error_markers
    )


def test_last_text_event_resets_on_v2_retry_event() -> None:
    """A pre-error draft verdict emitted before a 2.x retry must not be
    reaped as the session's verdict when the process dies inside the
    retry-after sleep (finding: retry events did not reset the candidate)."""
    log = "\n".join(
        [
            '{"type":"step_start","part":{"id":"s1"}}',
            '{"type":"text","part":{"text":"Verdict:\\ndraft"}}',
            RETRY_EVENT_V2,
        ]
    )
    assert last_text_event(log) is None


def test_last_text_event_resets_on_v2_error_event() -> None:
    log = "\n".join(
        [
            '{"type":"step_start","part":{"id":"s1"}}',
            '{"type":"text","part":{"text":"Verdict:\\ndraft"}}',
            FINAL_ERROR_V2,
        ]
    )
    assert last_text_event(log) is None


def test_session_progress_v2_retry_is_not_a_turn() -> None:
    """A retry discounts the still-open output-less step and is not itself
    progress: step_start + retry + step_start + error at launch stays 0/0
    launch_failed, and the digest fallback surfaces the error record."""
    log = "\n".join(
        [
            '{"type":"step_start","part":{"id":"s1"}}',
            RETRY_EVENT_V2,
            '{"type":"step_start","part":{"id":"s2"}}',
            FINAL_ERROR_V2,
        ]
    )
    progress = session_progress(log)
    assert progress is not None
    assert progress.turn_count == 0
    assert progress.tool_call_count == 0
    # The no-text fallback surfaces the provider error from the digest.
    assert progress.texts and "Usage limit reached" in progress.texts[-1]


def test_session_progress_v2_retry_after_text_discounts_open_step() -> None:
    log = "\n".join(
        [
            '{"type":"step_start","part":{"id":"s1"}}',
            '{"type":"text","part":{"text":"starting review"}}',
            RETRY_EVENT_V2,
            '{"type":"step_start","part":{"id":"s2"}}',
        ]
    )
    progress = session_progress(log)
    assert progress is not None
    # Attempt 1 produced text (counts 1); attempt 2's bare step_start at EOF
    # is discounted, so no phantom second turn.
    assert progress.turn_count == 1
