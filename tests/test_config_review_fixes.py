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
from charlie_work.global_config import load_fleet_global_config, load_layered_config

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


# R3-B1: ``ConstructionError`` subclasses ``ConfigError``, and the fleet entry points catch
# ``ConfigError`` to degrade to per-repo/default config. On main a host-wide ``__post_init__``
# rejection was a raw ``ValueError`` those handlers did not catch, so the process refused to
# start; it must still refuse, not silently run with runner allocation and the fleet caps off.
_BAD_HOST_WIDE_GLOBAL = (
    "fleet:\n  global_max_concurrent_sessions: 7\n"
    "runner_allocation:\n  enabled: true\n  max_running_per_repo: [[o/r, 2], [o/r, 3]]\n"
)


def test_fleet_global_load_helper_refuses_host_wide_construction_error(tmp_path: Path) -> None:
    repo, _, fleet = _layers(tmp_path, _BAD_HOST_WIDE_GLOBAL, "dispatch:\n  default_limit: 2\n")
    reported: list[Exception] = []
    with pytest.raises(ConstructionError):
        load_fleet_global_config(
            load_layered_config,
            repo,
            fleet_dir_override=fleet,
            fallback=None,
            report=reported.append,
        )
    assert reported == [], "a refusal is not a degradation: nothing is reported-then-continued"


def test_fleet_global_load_helper_still_degrades_other_config_errors(tmp_path: Path) -> None:
    # Control (rescued half): an ordinary FieldError in the global layer is absorbed by the
    # #665 rescue *inside* ``load_layered_config``, so the helper's own ``except ConfigError``
    # never runs here; the next test drives that branch. The refusal above is caused by the
    # ConstructionError class alone.
    repo, _, fleet = _layers(tmp_path, "dispatch:\n  nope: 1\n", "dispatch:\n  default_limit: 2\n")
    reported: list[Exception] = []
    config = load_fleet_global_config(
        load_layered_config,
        repo,
        fleet_dir_override=fleet,
        fallback=None,
        report=reported.append,
    )
    assert config is not None
    assert config.dispatch.default_limit == 2
    assert reported == []  # the rescue inside load_layered_config handled it, no fallback needed


def test_fleet_global_load_helper_reports_a_bad_global_when_nothing_can_rescue_it(
    tmp_path: Path,
) -> None:
    # Control (helper half): with no repo layer to rescue onto, the same ordinary FieldError
    # escapes ``load_layered_config`` as a plain ConfigError, so the helper's own
    # ``except ConfigError`` runs: one report, then the caller's fallback (not a refusal).
    fleet = tmp_path / "fleet"
    fleet.mkdir()
    (fleet / "config.yaml").write_text("dispatch:\n  nope: 1\n", encoding="utf-8")
    repo = tmp_path / "repo"
    repo.mkdir()
    reported: list[Exception] = []
    sentinel = object()
    config = load_fleet_global_config(
        load_layered_config,
        repo,
        fleet_dir_override=str(fleet),
        fallback=sentinel,
        report=reported.append,
    )
    assert len(reported) == 1
    assert isinstance(reported[0], ConfigError)
    assert not isinstance(reported[0], ConstructionError)
    assert "nope" in str(reported[0])
    assert config is sentinel  # no repo layer either, so the reload fails too: fallback


def test_fleet_global_load_helper_falls_back_when_nothing_loads(tmp_path: Path) -> None:
    fleet = tmp_path / "fleet"
    fleet.mkdir()  # no global config.yaml: require_global raises a plain ConfigError
    reported: list[Exception] = []
    sentinel = object()
    config = load_fleet_global_config(
        load_layered_config,
        tmp_path,
        fleet_dir_override=str(fleet),
        fallback=sentinel,
        report=reported.append,
    )
    assert len(reported) == 1
    assert not isinstance(reported[0], ConstructionError)
    assert config is not sentinel  # per-repo reload (defaults here) beats the fallback


@pytest.mark.parametrize("command", ["work", "bash-rats"])
def test_fleet_cli_refuses_to_start_on_host_wide_construction_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, command: str
) -> None:
    from unittest.mock import MagicMock

    from charlie_work import cli

    repo, _, fleet = _layers(tmp_path, _BAD_HOST_WIDE_GLOBAL, "dispatch:\n  default_limit: 2\n")
    fleet_loop_mock = MagicMock()
    monkeypatch.setattr(cli, "fleet_loop", fleet_loop_mock)
    monkeypatch.chdir(repo)
    args = cli.build_parser().parse_args(["--fleet-dir", fleet, "fleet", command, "--limit", "1"])
    runner = cli.run_fleet_work if command == "work" else cli.run_fleet_bash_rats
    with pytest.raises(ConstructionError):
        runner(args)
    fleet_loop_mock.assert_not_called()


def test_fleet_supervise_refuses_to_start_on_host_wide_construction_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from unittest.mock import MagicMock

    from charlie_work import fleet_dispatch

    repo, _, fleet = _layers(tmp_path, _BAD_HOST_WIDE_GLOBAL, "dispatch:\n  default_limit: 2\n")
    fleet_loop_mock = MagicMock()
    monkeypatch.setattr(fleet_dispatch, "fleet_loop", fleet_loop_mock)
    monkeypatch.setattr(fleet_dispatch, "try_acquire_supervisor_lock", MagicMock())
    monkeypatch.chdir(repo)
    with pytest.raises(ConstructionError):
        fleet_dispatch.run_fleet_supervise(max_passes=1, fleet_dir_override=fleet)
    fleet_loop_mock.assert_not_called()


def test_null_construction_error_names_the_null_key() -> None:
    # R3-N2: stays a loud ConstructionError (not a FieldError), but the opaque
    # "'<' not supported ... NoneType" text now names which keys were null.
    with pytest.raises(
        ConstructionError, match=r"null keys: .*runner_allocation\.threads_per_slot"
    ):
        build_config_from_data({"runner_allocation": {"enabled": True, "threads_per_slot": None}})
