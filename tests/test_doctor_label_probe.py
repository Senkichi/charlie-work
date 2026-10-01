"""``doctor``'s LABEL_LIST_FIELDS probe on a real ``GitHub`` over fake adapters.

``label list`` has no ``gh ... --json`` argv row (the legacy-argv table only
shrinks), so the probe must read through ``label_list()``; routing it through
``GitHub.run`` raised "unsupported argv" and failed it for every healthy repo
([gt-fix-r1] B4).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from _fake_transport import FakeAdapter, failure, make_github, ok
from charlie_work import doctor
from charlie_work.github_transport import FailureKind, GraphQLRequest, RestRequest

_NAME = "gh field list: LABEL_LIST_FIELDS"


def _doctor_checks(gh: Any) -> dict[str, tuple[bool, str]]:
    checks: dict[str, tuple[bool, str]] = {}

    def add(name: str, passed: bool, detail: str = "", severity: str = "error") -> None:
        checks[name] = (passed, detail)

    doctor._validate_gh_field_lists(add, gh)
    return checks


def _http(labels_reply) -> FakeAdapter:
    def handler(request):
        if isinstance(request, RestRequest) and request.route.endswith("/labels"):
            return labels_reply
        if isinstance(request, GraphQLRequest):
            return ok({"data": {}})
        return ok([])

    return FakeAdapter("http", handler=handler)


def test_the_label_field_probe_passes_on_a_healthy_repo(tmp_path: Path) -> None:
    gh, http, _ = make_github(tmp_path, http=_http(ok([{"name": "agent:queued"}])))

    passed, detail = _doctor_checks(gh)[_NAME]

    assert passed, detail
    assert any(
        isinstance(r, RestRequest) and r.route.endswith("/labels") for r in http.api_requests
    )


def test_the_label_field_probe_still_fails_when_the_label_read_fails(tmp_path: Path) -> None:
    # Control: the probe is not vacuously green; an unreadable label list fails it.
    gh, _, _ = make_github(
        tmp_path, http=_http(failure(FailureKind.SENT_NO_RESPONSE, "reset by peer"))
    )

    passed, detail = _doctor_checks(gh)[_NAME]

    assert not passed
    assert "probe failed" in detail
