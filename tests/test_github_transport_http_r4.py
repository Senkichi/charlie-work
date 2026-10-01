"""Round-4 HttpAdapter hygiene: no leaked connection, no escaping InvalidURL."""

from __future__ import annotations

from http.client import InvalidURL

from _fake_transport import FakeConn, FakeRaw

from charlie_work.github_transport import FailureKind, Response, RestRequest, TransportFailure
from charlie_work.github_transport.http_adapter import HttpAdapter

GET = RestRequest.of("GET", "repos/o/r/pulls/7")


class _DefectConn(FakeConn):
    def request(self, method, path, body=None, headers=None) -> None:
        raise KeyError("boom")


def test_an_adapter_defect_mid_exchange_closes_the_checked_out_connection() -> None:
    conn = _DefectConn([])
    adapter = HttpAdapter(connection_factory=lambda host, timeout: conn)
    out = adapter.send(GET, token="tok", timeout=5.0)
    assert isinstance(out, TransportFailure) and out.kind is FailureKind.ADAPTER_DEFECT
    assert conn.closed


def test_a_malformed_redirect_target_comes_back_as_a_value() -> None:
    def factory(host: str, timeout: float) -> FakeConn:
        if host != "api.github.com":
            raise InvalidURL(f"nonnumeric port: '{host}'")
        return FakeConn([FakeRaw(302, headers={"Location": "https://logs.example:abc/x"})])

    adapter = HttpAdapter(connection_factory=factory)
    request = RestRequest.of("GET", "repos/o/r/actions/jobs/1/logs", follow_redirect=True)
    out = adapter.send(request, token="tok", timeout=5.0)
    assert isinstance(out, TransportFailure)
    assert out.kind is FailureKind.SENT_NO_RESPONSE
    assert not isinstance(out, Response)
