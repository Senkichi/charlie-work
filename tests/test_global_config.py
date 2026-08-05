"""Tests for the layered-config merge itself (issue #704).

``load_layered_config`` used to reuse ``load_config``'s validation logic by
writing the merged dict to a real temp file and reading it straight back
(``tempfile.NamedTemporaryFile(delete=False)``). That cost a filesystem
write+read on every config load, left a temp file that had to be cleaned up,
and added failure modes (disk full, permissions, temp-file creation) that
have nothing to do with config merging.

The fix extracts the validation core into ``build_config_from_data`` (a
``dict -> OrchestratorConfig`` helper with no path involved) and has both
``load_config`` (path -> dict -> helper) and ``load_layered_config`` (merged
dict -> helper) call it directly, in memory.

The risk of this refactor is entirely in the merge semantics: a test that
only checks "the call succeeds" would pass against a version that silently
drops a layer -- the exact failure shape this repo has already been bitten by
(#590, #623). So the tests below pin actual values across precedence,
partial nested overrides, and missing layers, not just successful returns.
Sections below marked with a leading comment are new pins that did not exist
before this issue; the deep-merge (api_worker) and shallow-replace
(claude_code.worker_env) cases already have coverage in test_config.py and
are not duplicated here.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from charlie_work.config import OrchestratorConfig, load_config
from charlie_work.global_config import load_layered_config


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Precedence: repo wins on an overlapping key, in a *shallow* (non-api_worker)
# section. This also pins that the repo's section *replaces* the global
# section wholesale rather than merging key-by-key -- only api_worker gets
# the deep merge.
# ---------------------------------------------------------------------------


def test_repo_overrides_global_on_overlapping_key(tmp_path: Path) -> None:
    fleet = tmp_path / "fleet"
    global_path = _write(
        fleet / "config.yaml",
        "dispatch:\n  default_limit: 9\n  order: oldest\n",
    )
    repo_root = tmp_path / "repo"
    repo_path = _write(repo_root / "orchestrator.config.yaml", "dispatch:\n  default_limit: 3\n")

    config = load_layered_config(repo_root, None, fleet_dir_override=str(fleet))

    assert config.dispatch.default_limit == 3
    # Shallow replace: the repo's `dispatch` section fully replaces the
    # global one, so `order` reverts to the dataclass default instead of
    # inheriting "oldest" from the global layer -- that inheritance only
    # happens for api_worker's deep merge.
    assert config.dispatch.order == OrchestratorConfig().dispatch.order
    assert config.sources == (str(global_path), str(repo_path))


# ---------------------------------------------------------------------------
# Missing per-repo layer: global-only values must flow through to the
# resolved config, not just be reflected in `sources`.
# ---------------------------------------------------------------------------


def test_global_only_layer_value_flows_through(tmp_path: Path) -> None:
    fleet = tmp_path / "fleet"
    global_path = _write(fleet / "config.yaml", "dispatch:\n  default_limit: 42\n")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    config = load_layered_config(repo_root, None, fleet_dir_override=str(fleet))

    assert config.dispatch.default_limit == 42
    assert config.sources == (str(global_path),)


# ---------------------------------------------------------------------------
# Missing global layer: repo-only values must flow through unchanged, same
# as load_config on the repo file alone.
# ---------------------------------------------------------------------------


def test_repo_only_layer_value_flows_through(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_path = _write(repo_root / "orchestrator.config.yaml", "dispatch:\n  default_limit: 7\n")

    config = load_layered_config(
        repo_root, None, fleet_dir_override=str(tmp_path / "no-such-fleet-dir")
    )

    assert config.dispatch.default_limit == 7
    assert config == load_config(repo_path)
    assert config.sources == (str(repo_path),)


# ---------------------------------------------------------------------------
# No layers at all: pure dataclass defaults, matching load_config() with no
# path.
# ---------------------------------------------------------------------------


def test_no_layers_matches_bare_defaults(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    config = load_layered_config(
        repo_root, None, fleet_dir_override=str(tmp_path / "no-such-fleet-dir")
    )

    assert config == OrchestratorConfig()
    assert config.sources == ()


# ---------------------------------------------------------------------------
# The fix itself: no temp file is created, and the previous NamedTemporaryFile
# indirection is gone from the module. These are the two assertions that
# specifically distinguish the new implementation from the old one -- they
# fail against the pre-#704 tempfile-based implementation and pass against
# this one.
# ---------------------------------------------------------------------------


def test_layered_merge_leaves_no_temp_file(tmp_path: Path) -> None:
    fleet = tmp_path / "fleet"
    _write(fleet / "config.yaml", "dispatch:\n  default_limit: 9\n")
    repo_root = tmp_path / "repo"
    _write(repo_root / "orchestrator.config.yaml", "review:\n  require_issue_link: true\n")

    tmp_dir = Path(tempfile.gettempdir())
    before = {p.name for p in tmp_dir.glob("*.yaml")}

    load_layered_config(repo_root, None, fleet_dir_override=str(fleet))

    after = {p.name for p in tmp_dir.glob("*.yaml")}
    assert after == before, f"layered config load leaked temp file(s): {after - before}"


def test_global_config_module_does_not_use_tempfile() -> None:
    """Structural pin for the fix: the merge must not round-trip through disk
    at all, not merely clean up after itself. Guards against a regression
    that re-introduces ``tempfile.NamedTemporaryFile`` with cleanup that
    happens to work in tests but still costs the write+read on every call."""
    import charlie_work.global_config as global_config_module

    source = Path(global_config_module.__file__).read_text(encoding="utf-8")
    assert "tempfile" not in source
