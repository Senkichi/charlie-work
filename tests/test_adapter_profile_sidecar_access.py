"""The per-harness profile owns Sidecar access (issue #3092).

A caller asks ``AdapterFateProfile.sidecar`` where a harness's Sidecar lives
and how to rewrite it, instead of branching on the harness name. These tests
pin that the profile answers exactly what the harness-name branches answered:
the same path per harness, the Adapter's own write, and no Sidecar for the
harnesses that never wrote one.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from charlie_work import adapter_fate_profile, role_quota_ledger
from charlie_work.claude_code import CLAUDE_RECORD_KINDS
from charlie_work.harnesses import HARNESS_REGISTRY

# harness, WorkerView.adapter_kind, Sidecar filename for number 7, the Adapter's own write.
SIDECAR_HARNESSES = [
    ("devin-shell", "devin", "issue-7.json", "charlie_work.devin_shell._write_json"),
    (
        "claude-code",
        "claude-code",
        "issue-7.claude.json",
        "charlie_work.claude_code._write_json_atomic",
    ),
    ("api", "api", "issue-7.api.json", "charlie_work.claude_code._write_json_atomic"),
    (
        "opencode",
        "opencode",
        "issue-7.opencode.json",
        "charlie_work.claude_code._write_json_atomic",
    ),
]


@pytest.mark.parametrize(("harness", "view_kind", "filename", "_writer"), SIDECAR_HARNESSES)
def test_profile_gives_each_harness_its_sidecar_path(
    tmp_path: Path, harness: str, view_kind: str, filename: str, _writer: str
) -> None:
    profile = adapter_fate_profile.profile_for(view_kind)

    assert profile is not None and profile.sidecar is not None
    assert profile.harness == harness
    assert profile.sidecar.path(tmp_path, 7) == tmp_path / filename
    # The quota ledger asks by harness name and gets the same file.
    assert role_quota_ledger.sidecar_path_for(tmp_path, harness, 7) == tmp_path / filename


@pytest.mark.parametrize(("_harness", "view_kind", "filename", "_writer"), SIDECAR_HARNESSES)
def test_profile_sidecar_write_replaces_the_payload(
    tmp_path: Path, _harness: str, view_kind: str, filename: str, _writer: str
) -> None:
    profile = adapter_fate_profile.profile_for(view_kind)
    assert profile is not None and profile.sidecar is not None
    path = profile.sidecar.path(tmp_path, 7)
    path.write_text(json.dumps({"issue_number": 7, "failure_kind": None}), encoding="utf-8")

    profile.sidecar.write(path, {"issue_number": 7, "failure_kind": "stalled"})

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "issue_number": 7,
        "failure_kind": "stalled",
    }
    assert [entry.name for entry in tmp_path.iterdir()] == [filename]  # no temp file left


@pytest.mark.parametrize("harness", ["command", "manual"])
def test_harness_without_a_sidecar_has_no_sidecar_access(tmp_path: Path, harness: str) -> None:
    profile = adapter_fate_profile.profile_for(HARNESS_REGISTRY[harness].adapter_kind)

    assert profile is not None and profile.sidecar is None
    assert role_quota_ledger.sidecar_path_for(tmp_path, harness, 7) is None


def test_unknown_harness_has_no_sidecar_path(tmp_path: Path) -> None:
    assert role_quota_ledger.sidecar_path_for(tmp_path, "no-such-harness", 7) is None


def test_sidecar_access_covers_exactly_the_kinds_the_branches_covered() -> None:
    """The replaced branches were ``== "devin"`` / ``in CLAUDE_RECORD_KINDS``."""
    with_sidecar = set()
    for capabilities in HARNESS_REGISTRY.values():
        profile = adapter_fate_profile.profile_for(capabilities.adapter_kind)
        assert profile is not None
        if profile.sidecar is not None:
            with_sidecar.add(capabilities.adapter_kind)
        # A failure classification is recorded on the Sidecar: both or neither.
        assert (profile.sidecar is None) == (profile.record_failure is None)

    assert with_sidecar == {"devin"} | CLAUDE_RECORD_KINDS


@pytest.mark.parametrize(("_harness", "view_kind", "_filename", "writer"), SIDECAR_HARNESSES)
@pytest.mark.parametrize("registry_prebuilt", [False, True])
def test_profile_sidecar_write_is_late_bound(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    _harness: str,
    view_kind: str,
    _filename: str,
    writer: str,
    registry_prebuilt: bool,
) -> None:
    """The registry is cached per process: a write bound at build time would
    make a patch of the Adapter's writer vacuous, or leak the mock."""
    monkeypatch.setattr(adapter_fate_profile, "_PROFILES", None)
    if registry_prebuilt:
        adapter_fate_profile.profile_for(view_kind)

    with patch(writer) as mock_writer:
        profile = adapter_fate_profile.profile_for(view_kind)
        assert profile is not None and profile.sidecar is not None
        path = profile.sidecar.path(tmp_path, 7)
        profile.sidecar.write(path, {"issue_number": 7})

    mock_writer.assert_called_once_with(path, {"issue_number": 7})
    assert not path.exists()
    survivor = adapter_fate_profile.profile_for(view_kind)
    assert survivor is not None and survivor.sidecar is not None
    assert not isinstance(survivor.sidecar.write, MagicMock)
