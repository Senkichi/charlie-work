"""Label bootstrap/ensure: create every configured label, size limits, failure and event paths.

Split out of ``tests/test_charlie_work.py`` (issue #1554,
Track-1 wave 8/8).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from _fakes_github import FakeGitHub
from charlie_work.config import (
    LabelConfig,
    OrchestratorConfig,
)
from charlie_work.instrumentation import query_events
from charlie_work.local_issues import LocalFileGitHub
from charlie_work.paths import runtime_paths
from charlie_work.workflow import OrchestratorApp


def test_bootstrap_labels_creates_every_configured_label(tmp_path: Path) -> None:
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.bootstrap_labels()

    created = {name for name, _color, _desc in fake_gh.labels_created}
    assert created == set(config.labels.all)
    assert all(desc for _n, _c, desc in fake_gh.labels_created)
    # All labels verified present — must report honest success.
    assert result.ok is True
    assert result.data["missing"] == []


def test_bootstrap_labels_descriptions_fit_github_limit(tmp_path: Path) -> None:
    """GitHub rejects label descriptions over 100 chars with HTTP 422.

    FakeGitHub doesn't enforce this, so a too-long description passes the
    other bootstrap tests while silently 422ing against the real API (the
    failure is swallowed by label_create's allow_failure=True) — 'complexity:high'
    shipped with a 121-char description and was missing from every repo as a
    result. Assert on the real descriptions directly so this can't recur.
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.bootstrap_labels()

    too_long = {name: desc for name, _color, desc in fake_gh.labels_created if len(desc) > 100}
    assert too_long == {}


def test_bootstrap_labels_fails_when_creation_silently_missed(tmp_path: Path) -> None:
    """If label_create silently fails (e.g. no auth), bootstrap must report failure."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class FailingCreateGitHub(FakeGitHub):
        def label_create(self, label: str, color: str, description: str) -> None:
            # Silently drop all creates — simulates no-auth / wrong-repo scenario.
            pass

        def label_list(self) -> list[dict[str, object]]:
            return []  # nothing was created

    fake_gh = FailingCreateGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.bootstrap_labels()

    assert result.ok is False
    assert result.data["missing"] == config.labels.all


def test_bootstrap_labels_fails_when_label_list_raises(tmp_path: Path) -> None:
    """If label_list fails (e.g. network error), bootstrap must report failure."""
    from charlie_work.github import GitHubError

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class ErrorListGitHub(FakeGitHub):
        def label_list(self) -> list[dict[str, object]]:
            raise GitHubError("could not list labels: HTTP 401")

    fake_gh = ErrorListGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.bootstrap_labels()

    assert result.ok is False
    assert "verification failed" in result.message


def test_ensure_labels_creates_every_configured_label(tmp_path: Path) -> None:
    """ensure_labels is the automatic startup counterpart of bootstrap_labels.

    It must create every LabelConfig-derived label and record an ``ok`` event
    so convergence is observable in events.db without an operator running
    ``charlie bootstrap-labels``.
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.ensure_labels()

    created = {name for name, _color, _desc in fake_gh.labels_created}
    assert created == set(config.labels.all)
    assert result.ok is True
    assert result.data["missing"] == []
    ok_events = query_events(paths.state_file, kind="label_ensure_ok")
    assert len(ok_events) == 1, ok_events
    assert ok_events[0]["level"] == "info"


def test_ensure_labels_extra_labelconfig_field_is_ensured(tmp_path: Path) -> None:
    """AC #3: a LabelConfig with an extra field results in that label being ensured.

    The ensure set must derive from LabelConfig fields, so a new field ships its
    label with no extra wiring. Verified by subclassing LabelConfig with an
    extra field included in ``all``.
    """

    @dataclasses.dataclass(frozen=True)
    class _ExtraLabelConfig(LabelConfig):
        extra_label: str = "agent:extra-test-field"

        @property
        def all(self) -> list[str]:
            return [*LabelConfig.all.fget(self), self.extra_label]  # type: ignore[misc]

    config = OrchestratorConfig(labels=_ExtraLabelConfig())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.ensure_labels()

    created = {name for name, _color, _desc in fake_gh.labels_created}
    assert "agent:extra-test-field" in created
    assert created == set(config.labels.all)
    assert result.ok is True


def test_ensure_labels_removal_of_field_does_not_delete(tmp_path: Path) -> None:
    """AC #3: removing a field from LabelConfig must not delete the label.

    ensure_labels only creates/updates; it never deletes. A label that exists
    on the repo but is no longer in ``LabelConfig.all`` must survive the ensure.
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    # Pre-create a label that is NOT in the default LabelConfig.all, simulating
    # a label left behind after a field was removed from LabelConfig.
    orphan = "agent:removed-from-config"
    fake_gh.labels_created.append((orphan, "5319E7", "orphaned label"))
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.ensure_labels()

    assert result.ok is True
    live = {str(item.get("name") or "") for item in fake_gh.label_list()}
    # The orphan label survives — ensure never deletes.
    assert orphan in live


def test_ensure_labels_records_incomplete_event_on_missing(tmp_path: Path) -> None:
    """AC #2: ensure failures surface as events, not exceptions.

    When label_create silently fails (e.g. no auth), the ensure must record a
    ``label_ensure_incomplete`` warning event and return ok=False rather than
    raise.
    """

    class _FailingCreateGitHub(FakeGitHub):
        def label_create(self, label: str, color: str, description: str) -> None:
            pass  # silently drop all creates

        def label_list(self) -> list[dict[str, object]]:
            return []  # nothing was created

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, _FailingCreateGitHub())

    result = app.ensure_labels()

    assert result.ok is False
    assert result.data["missing"] == config.labels.all
    incomplete = query_events(paths.state_file, kind="label_ensure_incomplete")
    assert len(incomplete) == 1, incomplete
    assert incomplete[0]["level"] == "warning"


def test_ensure_labels_records_failed_event_on_list_error(tmp_path: Path) -> None:
    """AC #2: a verification (label_list) error surfaces as an error event."""
    from charlie_work.github import GitHubError

    class _ErrorListGitHub(FakeGitHub):
        def label_list(self) -> list[dict[str, object]]:
            raise GitHubError("could not list labels: HTTP 401")

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, _ErrorListGitHub())

    result = app.ensure_labels()

    assert result.ok is False
    assert "verification failed" in result.message
    failed = query_events(paths.state_file, kind="label_ensure_failed")
    assert len(failed) == 1, failed
    assert failed[0]["level"] == "error"


def test_ensure_labels_never_raises_on_unexpected_exception(tmp_path: Path) -> None:
    """AC #2: an unexpected exception from the gh impl is recorded, not raised."""

    class _RaisingCreateGitHub(FakeGitHub):
        def label_create(self, label: str, color: str, description: str) -> None:
            raise RuntimeError("boom")

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, _RaisingCreateGitHub())

    # Must not raise.
    result = app.ensure_labels()

    assert result.ok is False
    failed = query_events(paths.state_file, kind="label_ensure_failed")
    assert len(failed) == 1, failed
    assert failed[0]["level"] == "error"
    assert "RuntimeError" in failed[0]["payload"]["error"]


def test_ensure_labels_no_remote_backend_reports_ok(tmp_path: Path) -> None:
    """Issue #1862: a backend with no remote has no label registry to ensure.

    ``LocalFileGitHub.label_create`` is a documented no-op and
    ``label_list`` only reports labels already present in issue
    frontmatter, so every LabelConfig name that has not yet appeared on a
    local issue reads as permanently missing. The ensure must
    short-circuit to ``label_ensure_ok`` instead of emitting a
    ``label_ensure_incomplete`` warning on every supervisor startup.
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    issues_dir = tmp_path / "issues"
    issues_dir.mkdir()
    # An empty-but-real local issues dir: label_list() returns [], so
    # without the capability short-circuit every label reads missing.
    gh = LocalFileGitHub(repo_root=tmp_path, issues_dir=issues_dir)
    app = OrchestratorApp(tmp_path, paths, config, gh)

    result = app.ensure_labels()

    assert result.ok is True
    assert result.data["missing"] == []
    ok_events = query_events(paths.state_file, kind="label_ensure_ok")
    assert len(ok_events) == 1, ok_events
    assert ok_events[0]["level"] == "info"
    assert query_events(paths.state_file, kind="label_ensure_incomplete") == []


def test_bootstrap_labels_no_remote_backend_reports_ok(tmp_path: Path) -> None:
    """Issue #1862: bootstrap-labels shares ``_ensure_labels_core``, so the
    no-remote short-circuit applies to the explicit CLI door too — a local
    backend has no registry for bootstrap to populate."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    issues_dir = tmp_path / "issues"
    issues_dir.mkdir()
    gh = LocalFileGitHub(repo_root=tmp_path, issues_dir=issues_dir)
    app = OrchestratorApp(tmp_path, paths, config, gh)

    result = app.bootstrap_labels()

    assert result.ok is True
    assert result.data["missing"] == []
