"""Regression tests for the conftest autospec helper.

Issue #949: monkeypatch doubles can declare a narrower signature than the real
method.  These tests prove the ``autospec`` helper turns that into an immediate,
local failure.
"""

from __future__ import annotations

import pytest

from charlie_work.github import GitHub


def test_autospec_rejects_narrow_double(
    tmp_path, monkeypatch: pytest.MonkeyPatch, autospec
) -> None:
    """A double omitting ``state`` fails when a caller uses the real interface."""

    def _narrow(self, label):
        return []

    autospec(monkeypatch, GitHub, "issue_list", side_effect=_narrow)
    gh = GitHub(repo_root=tmp_path, dry_run=True)

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        gh.issue_list(state="open")


def test_autospec_allows_conformant_double(
    tmp_path, monkeypatch: pytest.MonkeyPatch, autospec
) -> None:
    """A double matching the real signature works across all call shapes."""

    def _conformant(self, labels=None, state=None):
        return [{"number": 1, "title": "Test"}]

    autospec(monkeypatch, GitHub, "issue_list", side_effect=_conformant)
    gh = GitHub(repo_root=tmp_path, dry_run=True)

    assert gh.issue_list() == [{"number": 1, "title": "Test"}]
    assert gh.issue_list("automated-ready") == [{"number": 1, "title": "Test"}]
    assert gh.issue_list(state="open") == [{"number": 1, "title": "Test"}]


def test_autospec_return_value(tmp_path, monkeypatch: pytest.MonkeyPatch, autospec) -> None:
    """A static return value is returned for any conformant call."""
    autospec(monkeypatch, GitHub, "issue_list", return_value=[])
    gh = GitHub(repo_root=tmp_path, dry_run=True)

    assert gh.issue_list() == []
    assert gh.issue_list(state="open") == []
