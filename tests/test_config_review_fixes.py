"""Regression tests for the adversarial review of the config-sections migration (ADR-0007).

B1/B2: ``key: null`` keeps the pre-migration meaning (``None``, never the default).
B3:    a construction error from a host-wide section is not swallowed by the #665
       discarded-global-layer rescue in ``load_layered_config``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from charlie_work.config import ConfigError, build_config_from_data
from charlie_work.config_validation import ConstructionError
from charlie_work.global_config import load_layered_config

# Bool fields whose default is True: ``key:`` (YAML null) used to store a falsy ``None``.
_TRUE_DEFAULT_NULL_KEYS = [
    ("auto_merge", "enabled"),
    ("auto_merge", "delete_branch"),
    ("watchdog", "enabled"),
    ("review", "require_issue_link"),
]


@pytest.mark.parametrize(("section", "key"), _TRUE_DEFAULT_NULL_KEYS)
def test_null_on_true_default_bool_stays_falsy_none(section: str, key: str) -> None:
    config = build_config_from_data({section: {key: None}})
    assert getattr(getattr(config, section), key) is None


def test_null_nested_on_true_default_bool_stays_none() -> None:
    config = build_config_from_data({"runtime": {"preflight": {"disk_floor_fatal": None}}})
    assert config.runtime.preflight.disk_floor_fatal is None


def test_null_require_current_base_does_not_trip_the_deadlock_hook() -> None:
    # Base accepted this (None is falsy, so no deadlock); the null -> True default made it fail.
    config = build_config_from_data(
        {"auto_merge": {"require_current_base": None, "update_branch_strategy": "off"}}
    )
    assert config.auto_merge.require_current_base is None


def test_null_normalized_to_default_where_base_did_so() -> None:
    config = build_config_from_data(
        {
            "dispatch": {"test_command": None},
            "deescalation": {
                "identical_reason_recurrence_window_minutes": None,
                "operator_queue_review_interval_minutes": None,
                "operator_queue_depth_threshold": None,
            },
        }
    )
    assert config.dispatch.test_command == ""
    assert config.deescalation.identical_reason_recurrence_window_minutes == 1440
    assert config.deescalation.operator_queue_review_interval_minutes == 0
    assert config.deescalation.operator_queue_depth_threshold == 5


def _layers(tmp_path: Path, global_yaml: str, repo_yaml: str) -> tuple[Path, Path, str]:
    fleet = tmp_path / "fleet"
    fleet.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    (fleet / "config.yaml").write_text(global_yaml, encoding="utf-8")
    repo_config = repo / "orchestrator.config.yaml"
    repo_config.write_text(repo_yaml, encoding="utf-8")
    return repo, repo_config, str(fleet)


def test_layered_null_keeps_both_layers(tmp_path: Path) -> None:
    repo, repo_config, fleet = _layers(
        tmp_path,
        "auto_merge:\n  update_branch_strategy: 'off'\n",
        "auto_merge:\n  require_current_base:\n",
    )
    config = load_layered_config(repo, repo_config, fleet_dir_override=fleet)
    assert [Path(s).name for s in config.sources] == ["config.yaml", "orchestrator.config.yaml"]
    assert config.auto_merge.require_current_base is None


@pytest.mark.parametrize(
    "global_yaml",
    [
        # ci_fleet ``__post_init__`` ValueError (mapping instead of [[repo, n]] pairs).
        "runner_allocation:\n  enabled: true\n  max_running_per_repo: {a/b: 1}\n",
        # R2-B1: a null on a field ci_fleet's ``__post_init__`` cannot compare used to be
        # caught by a use-site ``NotNull`` as a plain (rescuable) FieldError. Main raised the
        # construction TypeError, which the rescue re-raises -- keep it that way.
        "runner_allocation:\n  enabled: true\n  max_running_per_repo:\n",
        "runner_allocation:\n  enabled: true\n  threads_per_slot:\n",
        "runner_allocation:\n  enabled: true\n  reserved_threads:\n",
        "runner_allocation:\n  enabled: true\n  max_running_heavy:\n",
    ],
)
def test_broken_host_wide_section_in_global_layer_is_not_rescued(
    tmp_path: Path, global_yaml: str
) -> None:
    repo, repo_config, fleet = _layers(tmp_path, global_yaml, "dispatch:\n  default_limit: 2\n")
    with pytest.raises(ConstructionError) as excinfo:
        load_layered_config(repo, repo_config, fleet_dir_override=fleet)
    assert isinstance(excinfo.value, ConfigError)  # callers that catch ConfigError still do


def test_unknown_key_in_global_layer_is_still_rescued(tmp_path: Path) -> None:
    repo, repo_config, fleet = _layers(
        tmp_path, "dispatch:\n  nope: 1\n", "dispatch:\n  default_limit: 2\n"
    )
    config = load_layered_config(repo, repo_config, fleet_dir_override=fleet)
    assert [Path(s).name for s in config.sources] == ["orchestrator.config.yaml"]


# R2-B2 (design F3): main crashed with a raw TypeError/AttributeError on these shapes, which
# the #665 rescue never saw; every rejection is now a ``ConfigError`` so the rescue covers them
# like any other bad global layer. One layered pin per class keeps that flip documented.
@pytest.mark.parametrize(
    "global_yaml",
    [
        pytest.param("claude_code:\n  command: 5\n", id="command-template-non-string"),
        pytest.param("worker:\n  harness: [a]\n", id="harness-non-string"),
        pytest.param("review:\n  human_decision_markers:\n", id="not-null-tuple-key-null"),
        pytest.param("dispatch:\n  1: 2\n", id="non-string-section-key"),
        pytest.param("fleet_supervisor:\n  1: 3\n", id="non-string-host-wide-section-key"),
        pytest.param("rescue:\n  reviewer:\n    a: b\n", id="rescue-role-mapping-unknown-key"),
    ],
)
def test_former_raw_crash_shapes_are_now_rescued_like_any_bad_global_layer(
    tmp_path: Path, global_yaml: str
) -> None:
    repo, repo_config, fleet = _layers(tmp_path, global_yaml, "dispatch:\n  default_limit: 2\n")
    config = load_layered_config(repo, repo_config, fleet_dir_override=fleet)
    assert [Path(s).name for s in config.sources] == ["orchestrator.config.yaml"]
    assert config.dispatch.default_limit == 2
