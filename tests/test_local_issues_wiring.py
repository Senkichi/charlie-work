"""Tests for local_issues config wiring and the github_client_for factory boundary.

Includes an AST structural guard: every path that builds an
``OrchestratorApp``/``CommandContext`` must obtain its GitHub client from
``github_client_for`` -- the single point that chooses between the real
``gh``-backed client and ``LocalFileGitHub`` based on ``local_issues.enabled``.
A function that also constructs ``GitHub(`` directly, bare, in the same scope
silently bypasses that choice.
"""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

import charlie_work
from charlie_work.config import (
    ConfigError,
    LocalIssuesConfig,
    OrchestratorConfig,
    known_config_sections,
    load_config,
)
from charlie_work.github import GitHub
from charlie_work.local_issues import LocalFileGitHub, github_client_for

# -- 20. Config -----------------------------------------------------------------


def test_local_issues_config_defaults() -> None:
    cfg = LocalIssuesConfig()
    assert cfg.enabled is False
    assert cfg.issues_dir == "docs/issues"


def test_local_issues_config_is_frozen() -> None:
    cfg = LocalIssuesConfig()
    with pytest.raises(FrozenInstanceError):
        cfg.enabled = True  # type: ignore[misc]


def test_local_issues_in_known_config_sections() -> None:
    assert "local_issues" in known_config_sections()


def test_load_config_local_issues_section(tmp_path: Path) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        "local_issues:\n  enabled: true\n  issues_dir: tickets\n", encoding="utf-8"
    )

    config = load_config(config_file)

    assert config.local_issues.enabled is True
    assert config.local_issues.issues_dir == "tickets"


@pytest.mark.parametrize(
    ("yaml_body", "match"),
    [
        ('local_issues:\n  enabled: "yes"\n', "must be a bool"),
        ('local_issues:\n  issues_dir: ""\n', "must be a non-empty string"),
        ("local_issues:\n  issues_dir: 5\n", "must be a non-empty string"),
        ("local_issues:\n  issues_dir: ../x\n", "must not contain '..'"),
        ("local_issues:\n  issues_dir: a/../../x\n", "must not contain '..'"),
        ("local_issues:\n  issues_dir: /etc/x\n", "must be relative to the"),
        ("local_issues:\n  issues_dir: 'C:\\x'\n", "must be relative to the"),
        ("local_issues:\n  nope: 1\n", "unknown key"),
    ],
)
def test_load_config_local_issues_validation_errors(
    tmp_path: Path, yaml_body: str, match: str
) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(yaml_body, encoding="utf-8")

    with pytest.raises(ConfigError, match=match):
        load_config(config_file)


# -- 21. Factory ------------------------------------------------------------------


def test_github_client_for_enabled_returns_local_file_github(tmp_path: Path) -> None:
    config = replace(
        OrchestratorConfig(),
        local_issues=LocalIssuesConfig(enabled=True, issues_dir="tickets"),
    )

    client = github_client_for(tmp_path, config, github=GitHub, dry_run=True)

    assert isinstance(client, LocalFileGitHub)
    assert client.issues_dir == tmp_path / "tickets"
    assert client.dry_run is True


def test_github_client_for_disabled_returns_github(tmp_path: Path) -> None:
    # GitHub.__post_init__ only initializes _list_cache and constructs its
    # capability collaborators (see github_capabilities/_base.py) -- neither
    # does any I/O, so constructing it here needs no network monkeypatch.
    client = github_client_for(tmp_path, OrchestratorConfig(), github=GitHub, dry_run=False)

    assert isinstance(client, GitHub)


# -- 22. Structural guard: no direct GitHub( next to an app constructor ------------


_APP_BUILDER_NAMES = frozenset({"OrchestratorApp", "CommandContext"})
_BYPASS_NAME = "GitHub"


class _ScopedCallCollector(ast.NodeVisitor):
    """Collects ``Call`` nodes belonging to one function, not its nested functions."""

    def __init__(self) -> None:
        self.calls: list[ast.Call] = []

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        self.calls.append(node)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        return None

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        return None

    def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: N802
        return None


def _own_call_names(node: ast.AST) -> set[str]:
    collector = _ScopedCallCollector()
    collector.generic_visit(node)
    return {call.func.id for call in collector.calls if isinstance(call.func, ast.Name)}


def find_bypass_violations(source: str, filename: str = "<source>") -> list[str]:
    """Return ``filename:function`` for every function calling an app builder
    AND a bare ``GitHub(`` in its own (non-nested) body."""
    tree = ast.parse(source, filename=filename)
    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        names = _own_call_names(node)
        if names & _APP_BUILDER_NAMES and _BYPASS_NAME in names:
            violations.append(f"{filename}:{node.name}")
    return violations


def _app_builder_function_names(source: str, filename: str = "<source>") -> list[str]:
    tree = ast.parse(source, filename=filename)
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if _own_call_names(node) & _APP_BUILDER_NAMES:
            found.append(f"{filename}:{node.name}")
    return found


def test_bypass_detector_flags_the_forbidden_pair() -> None:
    source = (
        "def build():\n"
        "    gh = GitHub(repo_root=root)\n"
        "    return OrchestratorApp(root, paths, config, gh)\n"
    )
    assert find_bypass_violations(source, "synthetic.py") == ["synthetic.py:build"]


def test_bypass_detector_allows_the_factory() -> None:
    source = (
        "def build():\n"
        "    gh = github_client_for(root, config)\n"
        "    return OrchestratorApp(root, paths, config, gh)\n"
    )
    assert find_bypass_violations(source, "synthetic.py") == []


def test_github_client_for_calls_the_injected_constructor(tmp_path: Path) -> None:
    """The real-client constructor is the CALLER's name, never this module's.

    ``monkeypatch.setattr(cli, "GitHub", Fake)`` is the injection seam the CLI
    tests rely on; the factory must construct whatever it was handed.
    """
    calls: list[dict[str, object]] = []

    def fake_github(**kwargs: object) -> str:
        calls.append(kwargs)
        return "sentinel-client"

    config = OrchestratorConfig()
    client = github_client_for(tmp_path, config, github=fake_github, dry_run=True)

    assert client == "sentinel-client"
    assert calls == [{"repo_root": tmp_path, "runtime": config.runtime, "dry_run": True}]


def test_github_client_for_never_constructs_github_for_a_local_repo(tmp_path: Path) -> None:
    def exploding_github(**_kwargs: object) -> None:
        raise AssertionError("real client constructed for a local-file repo")

    config = OrchestratorConfig(local_issues=LocalIssuesConfig(enabled=True))
    (tmp_path / "docs" / "issues").mkdir(parents=True)

    client = github_client_for(tmp_path, config, github=exploding_github)

    assert isinstance(client, LocalFileGitHub)


def test_no_src_function_bypasses_github_client_for() -> None:
    src_root = Path(charlie_work.__file__).resolve().parent
    py_files = sorted(src_root.rglob("*.py"))
    assert len(py_files) > 50

    violations: list[str] = []
    app_builders: list[str] = []
    for path in py_files:
        source = path.read_text(encoding="utf-8")
        rel = path.relative_to(src_root).as_posix()
        violations.extend(find_bypass_violations(source, rel))
        app_builders.extend(_app_builder_function_names(source, rel))

    assert len(app_builders) >= 3
    assert violations == [], f"github_client_for bypass in: {violations}"
