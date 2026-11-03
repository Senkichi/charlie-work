"""Configured provider-key names reach ci_fleet's listener env strip (issue #2902).

The ``leaves_the_listener_env`` tests go through
``ci_fleet.runners._sanitize_env``, the function that builds every listener's
environment, because a name the list holds but the launch keeps would otherwise
pass. The rest are about which names the source finds.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest
from ci_fleet import runners

from charlie_work import layout
from charlie_work.ci_fleet_env_strip import fleet_provider_key_names

# An install older than ci-fleet 0.8.3 has no strip to test, and the import seam
# test in test_ci_fleet_seams.py covers what happens on one. This skips on the
# version, not the module, so a later release that drops the module fails here.
pytest.importorskip("ci_fleet", minversion="0.8.3")

from ci_fleet import env_strip  # noqa: E402


def _provider_yaml(key_name: str) -> str:
    return (
        "api_worker:\n"
        "  providers:\n"
        "    test-provider:\n"
        "      base_url: https://example.invalid/anthropic\n"
        f"      api_key_env: {key_name}\n"
        "      model: test-model\n"
        "      input_usd_per_mtok: 3.0\n"
        "      output_usd_per_mtok: 15.0\n"
    )


def _repo(tmp_path: Path, name: str, config_text: str) -> Path:
    root = tmp_path / name
    root.mkdir()
    (root / "orchestrator.config.yaml").write_text(config_text, encoding="utf-8")
    return root


def _fleet(tmp_path: Path, *roots: Path, global_config: str | None = None) -> str:
    fleet = tmp_path / "fleet"
    fleet.mkdir()
    registry = {"repos": {f"owner/{root.name}": {"repo_root": str(root)} for root in roots}}
    (fleet / layout.FLEET_REGISTRY_FILENAME).write_text(json.dumps(registry), encoding="utf-8")
    if global_config is not None:
        (fleet / layout.GLOBAL_CONFIG_FILENAME).write_text(global_config, encoding="utf-8")
    return str(fleet)


@pytest.fixture
def restore_source() -> Iterator[None]:
    """The installed source is process-wide, and the seam test reads it."""
    previous = env_strip.get_env_strip_source()
    yield
    assert previous is not None
    env_strip.set_env_strip_source(previous)


def test_every_repos_key_name_leaves_the_listener_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, restore_source: None
) -> None:
    """Names from different repos both go, and case does not protect a name.

    Windows upper-cases ``os.environ`` keys, so a config spelled in mixed case
    has to match the upper-cased variable.
    """
    fleet = _fleet(
        tmp_path,
        _repo(tmp_path, "alpha", _provider_yaml("ALPHA_PROVIDER_KEY")),
        _repo(tmp_path, "beta", _provider_yaml("Beta_Mixed_Key")),
    )
    monkeypatch.setenv("ALPHA_PROVIDER_KEY", "alpha-value")
    monkeypatch.setenv("BETA_MIXED_KEY", "beta-value")
    monkeypatch.setenv("CHARLIE_WORK_TEST_CONTROL", "kept")

    env_strip.set_env_strip_source(lambda: fleet_provider_key_names(fleet))
    env = {name.casefold(): value for name, value in runners._sanitize_env().items()}

    assert "alpha_provider_key" not in env
    assert "beta_mixed_key" not in env
    assert env["charlie_work_test_control"] == "kept"


def test_the_zero_argument_call_leaves_the_listener_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, restore_source: None
) -> None:
    """``ci_fleet`` passes no arguments, so the source has to find the fleet itself."""
    fleet = _fleet(tmp_path, _repo(tmp_path, "alpha", _provider_yaml("ALPHA_PROVIDER_KEY")))
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", fleet)
    monkeypatch.setenv("ALPHA_PROVIDER_KEY", "alpha-value")
    monkeypatch.setenv("CHARLIE_WORK_TEST_CONTROL", "kept")

    env_strip.set_env_strip_source(fleet_provider_key_names)
    env = {name.casefold(): value for name, value in runners._sanitize_env().items()}

    assert "alpha_provider_key" not in env
    assert env["charlie_work_test_control"] == "kept"


def test_a_config_that_fails_to_load_keeps_the_others(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """One bad repo costs its own names only, and says so in the log."""
    broken = _repo(tmp_path, "broken", "api_worker: [unterminated\n")
    fleet = _fleet(tmp_path, broken, _repo(tmp_path, "good", _provider_yaml("GOOD_PROVIDER_KEY")))

    with caplog.at_level(logging.WARNING, logger="charlie_work.ci_fleet_env_strip"):
        names = fleet_provider_key_names(fleet)

    assert names == {"GOOD_PROVIDER_KEY"}
    assert "owner/broken" in caplog.text


def test_the_global_layer_counts_when_no_repo_can_be_read(tmp_path: Path) -> None:
    """A fleet-wide key is still named with an empty or unreadable registry."""
    fleet = _fleet(tmp_path, global_config=_provider_yaml("FLEET_PROVIDER_KEY"))

    assert fleet_provider_key_names(fleet) == {"FLEET_PROVIDER_KEY"}


@pytest.mark.parametrize(
    "registry",
    [
        pytest.param(b'{"repos": []}', id="repos-is-a-list"),
        pytest.param(b'{"repos": 5}', id="repos-is-a-number"),
        pytest.param(b'{"repos": {"owner/x": {"repo_root": 5}}}', id="root-is-not-a-path"),
        pytest.param(b'{"repos": {"owner/x": "text"}}', id="entry-is-not-a-mapping"),
        pytest.param(b"\xff\xfe not text", id="not-utf8"),
        pytest.param(b'{"repos": {', id="truncated"),
        pytest.param(b'{"repos": {"owner/x": {"repo_root": "no/such/repo"}}}', id="root-is-gone"),
        pytest.param(
            b'{"repos": {"owner/x": {"repo_root": ".", "config_path": "no/such.yaml"}}}',
            id="config-path-is-gone",
        ),
    ],
)
def test_a_bad_registry_keeps_the_global_names_and_says_so(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, registry: bytes
) -> None:
    """A raising source strips nothing at all, so a bad registry must not raise."""
    fleet = _fleet(tmp_path, global_config=_provider_yaml("FLEET_PROVIDER_KEY"))
    (Path(fleet) / layout.FLEET_REGISTRY_FILENAME).write_bytes(registry)

    with caplog.at_level(logging.WARNING, logger="charlie_work.ci_fleet_env_strip"):
        names = fleet_provider_key_names(fleet)

    assert names == {"FLEET_PROVIDER_KEY"}
    assert "env strip:" in caplog.text


def test_an_explicit_config_path_is_the_one_read(tmp_path: Path) -> None:
    """The registry can point a repo at a config outside its root."""
    root = _repo(tmp_path, "alpha", _provider_yaml("DEFAULT_PATH_KEY"))
    explicit = tmp_path / "elsewhere.yaml"
    explicit.write_text(_provider_yaml("EXPLICIT_PATH_KEY"), encoding="utf-8")
    fleet = _fleet(tmp_path)
    registry = {"repos": {"owner/alpha": {"repo_root": str(root), "config_path": str(explicit)}}}
    (Path(fleet) / layout.FLEET_REGISTRY_FILENAME).write_text(
        json.dumps(registry), encoding="utf-8"
    )

    assert fleet_provider_key_names(fleet) == {"EXPLICIT_PATH_KEY"}


def test_a_missing_fleet_dir_names_nothing_and_does_not_raise(tmp_path: Path) -> None:
    """A plain single-repo checkout has no fleet dir at all."""
    assert fleet_provider_key_names(str(tmp_path / "no-such-fleet")) == frozenset()
