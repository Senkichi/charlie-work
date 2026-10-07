"""OpenCode Go usage-limit reset recovery (opencode_limits, AdapterFateProfile.quota_reset)."""

from __future__ import annotations

import io
import json
import urllib.error
from datetime import UTC, datetime, timedelta
from email.message import Message
from pathlib import Path

import pytest

from charlie_work import opencode_limits, worker_fate
from charlie_work.config import RuntimeConfig

FAKE_KEY = "sk-test-not-a-real-key"
GO_429_BODY = json.dumps(
    {
        "type": "error",
        "error": {"type": "GoUsageLimitError", "message": "Usage limit reached"},
        "metadata": {"workspace": "wrk_test", "limitName": "weekly"},
    }
)
PRINT_LOG_QUOTA = (
    "timestamp=2026-10-07T10:00:00Z level=ERROR service=session.processor "
    'error.error="AI_APICallError: Usage limit reached"'
)


def _headers(**values: str) -> Message:
    message = Message()
    for name, value in values.items():
        message[name.replace("_", "-")] = value
    return message


class _Response(io.BytesIO):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(body.encode("utf-8"))
        self.status = status
        self.headers = _headers()

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class _FakeGo:
    """Serves GET /models, then answers the chat probe with ``chat``."""

    def __init__(self, chat: _Response | urllib.error.HTTPError | OSError) -> None:
        self.chat = chat
        self.requests: list[object] = []

    def __call__(self, request: object, timeout: float) -> _Response:
        self.requests.append(request)
        if request.get_method() == "GET":  # type: ignore[attr-defined]
            return _Response(200, json.dumps({"data": [{"id": "glm-5.3-flash"}]}))
        if isinstance(self.chat, BaseException):
            raise self.chat
        return self.chat


def _http_429(body: str, **headers: str) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://opencode.ai/zen/go/v1/chat/completions",
        429,
        "Too Many Requests",
        _headers(**headers),
        io.BytesIO(body.encode("utf-8")),
    )


@pytest.fixture
def go(monkeypatch: pytest.MonkeyPatch):
    def install(chat: _Response | urllib.error.HTTPError | OSError) -> _FakeGo:
        fake = _FakeGo(chat)
        monkeypatch.setattr(opencode_limits, "_go_api_key", lambda: FAKE_KEY)
        monkeypatch.setattr(opencode_limits, "_open", fake)
        return fake

    return install


# --- pure parsing --------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "5-hour usage limit reached. It will reset in 2 hours 5 minutes.",
            timedelta(hours=2, minutes=5),
        ),
        ("monthly usage limit reached. It will reset in 12 days.", timedelta(days=12)),
        ("It will reset in 1 day, 3 hours and 4 minutes", timedelta(days=1, hours=3, minutes=4)),
        ("Usage limit reached", None),
    ],
)
def test_reset_from_text(text: str, expected: timedelta | None) -> None:
    assert opencode_limits.reset_from_text(text) == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("5-hour", timedelta(hours=5)),
        ("weekly", timedelta(days=7)),
        ("monthly", timedelta(days=30)),
        ("mystery", None),
    ],
)
def test_window_for_limit_name(name: str, expected: timedelta | None) -> None:
    assert opencode_limits.window_for_limit_name(name) == expected


def test_reset_from_429_prefers_retry_after_over_limit_name() -> None:
    reset = opencode_limits.reset_from_429(_headers(retry_after="7200"), GO_429_BODY)
    assert reset == timedelta(hours=2)


def test_reset_from_429_falls_back_to_limit_name_window() -> None:
    assert opencode_limits.reset_from_429(_headers(), GO_429_BODY) == timedelta(days=7)


def test_reset_from_429_ignores_a_plain_rate_limit() -> None:
    plain = json.dumps({"type": "error", "error": {"type": "RateLimitError"}})
    assert opencode_limits.reset_from_429(_headers(retry_after="30"), plain) is None


# --- the probe -----------------------------------------------------------------


def test_probe_limited_returns_retry_after_and_never_leaks_the_key(go) -> None:
    fake = go(_http_429(GO_429_BODY, retry_after="18000"))
    assert opencode_limits.probe_go_reset() == timedelta(hours=5)
    chat = fake.requests[-1]
    assert chat.get_header("Authorization") == f"Bearer {FAKE_KEY}"
    assert chat.get_header("X-opencode-session", "").startswith("ses_cw_probe_")
    body = json.loads(chat.data)
    assert body["max_tokens"] == 1 and body["model"] == "glm-5.3-flash"


def test_probe_cleared_limit_returns_short_cooldown(go) -> None:
    go(_Response(200, json.dumps({"choices": []})))
    assert opencode_limits.probe_go_reset() == opencode_limits.CLEARED_COOLDOWN


def test_probe_network_error_returns_none(go, caplog: pytest.LogCaptureFixture) -> None:
    go(OSError(f"connect failed for Bearer {FAKE_KEY}"))
    assert opencode_limits.probe_go_reset() is None
    assert FAKE_KEY not in caplog.text


def test_probe_without_credential_makes_no_request() -> None:
    # conftest's default: no credential, and _open refuses -- reaching it would raise.
    assert opencode_limits.probe_go_reset() is None


def test_go_quota_reset_caches_one_probe(go) -> None:
    fake = go(_http_429(GO_429_BODY, retry_after="600"))
    assert opencode_limits.go_quota_reset(PRINT_LOG_QUOTA) == timedelta(minutes=10)
    assert opencode_limits.go_quota_reset(PRINT_LOG_QUOTA) == timedelta(minutes=10)
    assert len(fake.requests) == 2  # one GET /models + one chat, not two of each


def test_go_quota_reset_skips_probe_for_free_tier_limit(go) -> None:
    fake = go(_http_429(GO_429_BODY, retry_after="600"))
    assert opencode_limits.go_quota_reset("FreeUsageLimitError: daily free usage") is None
    assert fake.requests == []


# --- wired through classify_for --------------------------------------------------


def _quota_log(tmp_path: Path) -> Path:
    log = tmp_path / "issue-1.log"
    log.write_text('{"type":"step_start"}\n' + PRINT_LOG_QUOTA + "\n", encoding="utf-8")
    return log


def _classify(log: Path, now: datetime) -> tuple[str | None, str | None]:
    return worker_fate.classify_for(
        "opencode", log, quota_error_markers=RuntimeConfig().quota_error_markers, now=now
    )


def test_classify_for_opencode_uses_the_probed_reset(go, tmp_path: Path) -> None:
    go(_http_429(GO_429_BODY, retry_after="18000"))
    now = datetime(2026, 10, 7, 12, 0, tzinfo=UTC)
    assert _classify(_quota_log(tmp_path), now) == ("quota_exhausted", "2026-10-07T17:00:00Z")


def test_classify_for_opencode_falls_back_to_24h_without_a_probe(tmp_path: Path) -> None:
    now = datetime(2026, 10, 7, 12, 0, tzinfo=UTC)
    assert _classify(_quota_log(tmp_path), now) == ("quota_exhausted", "2026-10-08T12:00:00Z")


def test_classify_for_other_harnesses_never_probe(go, tmp_path: Path) -> None:
    fake = go(_http_429(GO_429_BODY, retry_after="60"))
    log = tmp_path / "issue-1.log"
    log.write_text("Error: usage limit reached\n", encoding="utf-8")
    kind, _ = worker_fate.classify_for(
        "claude-code", log, quota_error_markers=RuntimeConfig().quota_error_markers
    )
    assert kind == "quota_exhausted"
    assert fake.requests == []
