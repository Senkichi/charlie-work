"""The bearer token never reaches an event, a log line, a detail or a repr (ADR-0006, F-D).

A token that is not a legal header value (an internal CR/LF) used to make
``http.client`` raise ``ValueError("Invalid header value b'Bearer <token>'")``;
the adapter turned that into an ADAPTER_DEFECT whose detail went, verbatim, into
the ``github_transport_fallback`` event (events.db) and the log. These tests
drive every failure kind through the real ``HttpAdapter`` and ``GuardedTransport``
and grep everything they emit for the token value.
"""

from __future__ import annotations

import http.client
import json
import logging
from http.client import HTTPSConnection
from pathlib import Path

import pytest
from _fake_transport import FakeAdapter, FakeConn, FakeRaw, Runtime, build_guard, ok

from charlie_work.github_transport import (
    Adapters,
    FailureKind,
    GuardedTransport,
    HttpAdapter,
    Response,
    RestRequest,
    TransportFailure,
)
from charlie_work.github_transport.http_adapter import _Prepared
from charlie_work.github_transport.token_hygiene import is_well_formed, redact
from charlie_work.instrumentation import query_events

SECRET = "ghp_SECRETTOKEN0123456789abcdef"
GET = RestRequest.of("GET", "repos/{owner}/{repo}/pulls/7")


class _RealHeaderConn(FakeConn):
    """Validates headers exactly like http.client (raises on a bad value)."""

    def request(self, method, path, body=None, headers=None) -> None:
        conn = HTTPSConnection("api.github.com")
        conn._HTTPConnection__state = "Idle"  # type: ignore[attr-defined]
        conn.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
        for key, value in (headers or {}).items():
            conn.putheader(key, value)
        super().request(method, path, body=body, headers=headers)


class _RaisingConn(FakeConn):
    def __init__(self, error: BaseException) -> None:
        super().__init__([])
        self.error = error

    def request(self, method, path, body=None, headers=None) -> None:
        raise self.error


def _everything(tmp_path: Path, caplog: pytest.LogCaptureFixture, *outcomes: object) -> str:
    events = json.dumps(query_events(tmp_path / "state.json"), default=str)
    return "\n".join([events, caplog.text, *map(repr, outcomes)])


def _assert_clean(text: str, token: str) -> None:
    assert token not in text
    assert "second-line" not in text
    assert repr(token)[1:-1] not in text
    assert "Bearer ghp_" not in text


def test_well_formed_accepts_real_tokens_and_rejects_anything_a_header_would_split() -> None:
    assert is_well_formed(SECRET)
    assert is_well_formed("github_pat_11ABC_xyz-123.456")
    for bad in ("", "a b", "a\nb", "a\rb", "a\tb", "a\x00b", "tök"):
        assert not is_well_formed(bad), bad


def test_redact_covers_the_literal_the_escaped_and_the_bearer_forms() -> None:
    token = "ghp_SECRET\nsecond-line"
    for text in (
        f"x {token} y",
        repr(f"Bearer {token}"),
        str(ValueError(f"Invalid header value {('Bearer ' + token).encode()!r}")),
        "Authorization: Bearer abc.def",
        "using ghp_" + "A" * 30,
    ):
        _assert_clean(redact(text, token), token)
    assert redact("Not Found", SECRET) == "Not Found"


def test_a_token_with_an_internal_newline_is_rejected_before_the_wire() -> None:
    token = f"{SECRET}\nsecond-line"
    conn = _RealHeaderConn([FakeRaw(200, body=b"{}")])
    adapter = HttpAdapter(connection_factory=lambda host, timeout: conn)

    out = adapter.send(GET.resolve("o", "r"), token=token, timeout=5.0)

    assert isinstance(out, TransportFailure) and out.kind is FailureKind.TOKEN_UNAVAILABLE
    _assert_clean(repr(out), token)
    assert conn.requests == []


def test_a_malformed_token_from_gh_is_not_used_and_never_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    token = f"{SECRET}\nsecond-line"
    http = FakeAdapter("http", [ok({"via": "http"})])
    gh = FakeAdapter("gh", [ok({"via": "gh"})], token=token)
    guard, _, _, _ = build_guard(
        http=http, gh=gh, state_path=tmp_path / "state.json", runtime=Runtime(gh_max_retries=0)
    )

    with caplog.at_level(logging.DEBUG):
        out = guard.send(GET)

    assert isinstance(out, Response) and out.json() == {"via": "gh"}
    assert http.calls == []  # the malformed token never reached the HTTP adapter
    _assert_clean(_everything(tmp_path, caplog, out), token)


def _failures() -> list[tuple[str, FakeConn]]:
    leak = f"Invalid header value {('Bearer ' + SECRET).encode()!r}"
    return [
        ("defect-value-error", _RaisingConn(ValueError(leak))),
        ("defect-type-error", _RaisingConn(TypeError(f"bad Bearer {SECRET}"))),
        (
            "sent-no-response-os-error",
            _RaisingConn(OSError(f"reset; Authorization: Bearer {SECRET}")),
        ),
        ("sent-no-response-http", _RaisingConn(http.client.HTTPException(f"Bearer {SECRET}"))),
        ("timeout", _RaisingConn(TimeoutError(f"timed out Bearer {SECRET}"))),
        ("connect", FakeConn([], connect_error=OSError(f"refused {SECRET}"))),
    ]


@pytest.mark.parametrize("label,conn", _failures(), ids=[label for label, _ in _failures()])
def test_no_failure_kind_leaks_the_token_into_events_logs_or_details(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, label: str, conn: FakeConn
) -> None:
    gh = FakeAdapter("gh", [ok({"via": "gh"})], token=SECRET)
    adapter = HttpAdapter(connection_factory=lambda host, timeout: conn)

    guard = GuardedTransport(
        Adapters(http=adapter, gh=gh),
        runtime=Runtime(gh_max_retries=0),
        state_path=tmp_path / "state.json",
        resolve_owner_repo=lambda: ("octo", "hello"),
        sleep=lambda _s: None,
        jitter=lambda lo, hi: 0.0,
        now=lambda: 1000.0,
    )

    with caplog.at_level(logging.DEBUG):
        out = guard.send(GET)

    assert isinstance(out, (Response, TransportFailure))
    direct = adapter.send(GET.resolve("octo", "hello"), token=SECRET, timeout=5.0)
    _assert_clean(_everything(tmp_path, caplog, out, direct), SECRET)


def test_the_fallback_event_redacts_a_detail_the_adapter_did_not_scrub(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Belt and braces: the guard scrubs on its own, whichever adapter built the detail."""
    leak = TransportFailure(FailureKind.ADAPTER_DEFECT, f"boom Bearer {SECRET}", "http")
    http = FakeAdapter("http", [leak])
    gh = FakeAdapter("gh", [ok({"via": "gh"})], token=SECRET)
    guard, _, _, _ = build_guard(
        http=http, gh=gh, state_path=tmp_path / "state.json", runtime=Runtime(gh_max_retries=0)
    )

    with caplog.at_level(logging.DEBUG):
        guard.send(GET)

    (event,) = query_events(tmp_path / "state.json", kind="github_transport_fallback")
    assert "<redacted>" in json.dumps(event, default=str)
    _assert_clean(_everything(tmp_path, caplog), SECRET)


def test_a_prepared_request_does_not_repr_its_authorization_header() -> None:
    adapter = HttpAdapter(connection_factory=lambda host, timeout: FakeConn([]))
    prepared = adapter._prepare(GET.resolve("o", "r"), SECRET, None)
    assert isinstance(prepared, _Prepared)
    assert prepared.headers["Authorization"] == f"Bearer {SECRET}"  # still sent
    assert SECRET not in repr(prepared)


def test_redact_scrubs_a_credential_that_is_not_the_cached_token() -> None:
    """r4: the generic patterns must match mid-text (a literal backspace never did)."""
    mine = "ghp_MINE" + "m" * 20
    foreign = "ghp_" + "F" * 30
    for text in (
        "Authorization: Bearer abcdef123456",
        f"proxy said: token {foreign}",
        f"leaked {foreign} here",
        "x github_pat_" + "Z" * 30 + " y",
    ):
        out = redact(text, mine)
        assert "abcdef123456" not in out and "F" * 30 not in out and "Z" * 30 not in out, out
    assert redact("Not Found", mine) == "Not Found"


@pytest.mark.parametrize("prefix", ["x", "_", "GH_TOKEN_", "9", "token="])
def test_a_foreign_token_glued_to_a_word_character_is_still_redacted(prefix: str) -> None:
    """r5 N5-1: no word boundary is required before a token shape."""
    for body in ("ghp_" + "F" * 30, "github_pat_" + "Z" * 30, "gho_" + "Q" * 30):
        out = redact(f"saw {prefix}{body} in text", "ghp_MINE" + "m" * 20)
        assert body not in out and body[-20:] not in out, out
