"""Regression tests for the adversarial review of the config-sections migration (ADR-0007).

B1/B2: (supersedes: issue #2116) ``key: null`` on a scalar field follows the
validator's class-level null rule -- a non-Optional scalar takes its declared
default (derived ``NullIsDefault``); an Optional scalar and every non-scalar
kind keep the pre-migration meaning (``None``, never the default).
B3:    a construction error from a host-wide section is not swallowed by the #665
discarded-global-layer rescue in ``load_layered_config``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from charlie_work.config import ConfigError, build_config_from_data
from charlie_work.config_validation import ConstructionError
from charlie_work.global_config import load_fleet_global_config, load_layered_config

# Bool fields whose default is True: since issue #2116 an explicit YAML null on
# these non-Optional scalars takes the default (derived ``NullIsDefault``) --
# the pre-fix falsy ``None`` is gone (each was a latent consumer TypeError, or a
# silent "feature off" for a consumer that did ``if x:``).
_TRUE_DEFAULT_NULL_KEYS = [
    ("auto_merge", "enabled"),
    ("auto_merge", "delete_branch"),
    ("watchdog", "enabled"),
    ("review", "require_issue_link"),
]


@pytest.mark.parametrize(("section", "key"), _TRUE_DEFAULT_NULL_KEYS)
def test_null_on_true_default_bool_stays_falsy_none(section: str, key: str) -> None:
    """Name pinned: the collect-only gate (issue #1538) fails a rename. The body
    asserts the ISSUE #2116 rule: a null on these non-Optional bool knobs takes
    the field's declared default (derived ``NullIsDefault``) -- the pre-#2116
    falsy ``None`` is what the #2116 fix retired."""
    config = build_config_from_data({section: {key: None}})
    expected = getattr(build_config_from_data({}), section)
    assert getattr(getattr(config, section), key) == getattr(expected, key)


def test_null_nested_on_true_default_bool_stays_none() -> None:
    """Name pinned: the collect-only gate (issue #1538) fails a rename. Body
    since #2116: the nested null takes the declared default, never ``None``."""
    config = build_config_from_data({"runtime": {"preflight": {"disk_floor_fatal": None}}})
    assert config.runtime.preflight.disk_floor_fatal is (
        build_config_from_data({}).runtime.preflight.disk_floor_fatal
    )


def test_null_require_current_base_does_not_trip_the_deadlock_hook() -> None:
    # Name pinned: the collect-only gate (issue #1538) fails a rename. Since
    # #2116 "does not trip" is the OLD meaning: the null takes the default True,
    # which -- with ``update_branch_strategy: off`` -- trips the deadlock hook
    # the same way an explicit ``true`` does: loud.
    # Base accepted this (None is falsy, so no deadlock); since #2116 the null
    # takes the default True, which -- with ``update_branch_strategy: off`` --
    # trips the deadlock hook the same way an explicit ``true`` does: loud.
    with pytest.raises(ConfigError, match="require_current_base"):
        build_config_from_data(
            {"auto_merge": {"require_current_base": None, "update_branch_strategy": "off"}}
        )
    # Without the deadlock the defaulted knob is indistinguishable from an
    # explicit default:
    plain = build_config_from_data({"auto_merge": {"require_current_base": None}})
    assert plain.auto_merge.require_current_base is True


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
    # Name pinned: the collect-only gate (issue #1538) fails a rename. Body
    # since #2116: the repo-layer null takes the default True, the merged config
    # (default True + global ``off`` strategy) trips the deadlock hook, the
    # rescuable FieldError sends the #665 rescue to the repo layer alone. The
    # layered null keeps BOTH LAYERS only pre-#2116; the rescue is what now
    # keeps the configuration loadable at all.
    repo, repo_config, fleet = _layers(
        tmp_path,
        "auto_merge:\n  update_branch_strategy: 'off'\n",
        "auto_merge:\n  require_current_base:\n",
    )
    config = load_layered_config(repo, repo_config, fleet_dir_override=fleet)
    assert [Path(s).name for s in config.sources] == ["orchestrator.config.yaml"]
    assert config.auto_merge.require_current_base is True  # default, not the null


@pytest.mark.parametrize(
    "global_yaml",
    [
        pytest.param(
            # ci_fleet ``__post_init__`` ValueError (mapping instead of [[repo, n]] pairs).
            "runner_allocation:\n  enabled: true\n  max_running_per_repo: {a/b: 1}\n",
        ),
        pytest.param(
            # A null on the non-scalar mapping field still reaches ci_fleet's
            # ``__post_init__`` as a construction TypeError -- the rescue re-raises
            # it (loud); #2116's null rule only retires this for SCALAR fields.
            "runner_allocation:\n  enabled: true\n  max_running_per_repo:\n",
        ),
        # The three scalar-null ids are kept VERBATIM for the collect-only gate
        # (issue #1538): since #2116 a scalar null takes the declared default, so
        # the layered load SUCCEEDS and no construction error is raised anymore
        # (that behavior is pinned by
        # test_scalar_null_in_global_host_wide_section_takes_the_default below).
        pytest.param(
            "runner_allocation:\n  enabled: true\n  threads_per_slot:\n",
            marks=pytest.mark.skip(
                reason="scalar null takes the default since #2116 (no refusal anymore); "
                "leaf id kept for the collect-only gate (#1538)"
            ),
        ),
        pytest.param(
            "runner_allocation:\n  enabled: true\n  reserved_threads:\n",
            marks=pytest.mark.skip(
                reason="scalar null takes the default since #2116 (no refusal anymore); "
                "leaf id kept for the collect-only gate (#1538)"
            ),
        ),
        pytest.param(
            "runner_allocation:\n  enabled: true\n  max_running_heavy:\n",
            marks=pytest.mark.skip(
                reason="scalar null takes the default since #2116 (no refusal anymore); "
                "leaf id kept for the collect-only gate (#1538)"
            ),
        ),
    ],
)
def test_broken_host_wide_section_in_global_layer_is_not_rescued(
    tmp_path: Path, global_yaml: str
) -> None:
    repo, repo_config, fleet = _layers(tmp_path, global_yaml, "dispatch:\n  default_limit: 2\n")
    with pytest.raises(ConstructionError) as excinfo:
        load_layered_config(repo, repo_config, fleet_dir_override=fleet)
    assert isinstance(excinfo.value, ConfigError)  # callers that catch ConfigError still do


def test_scalar_null_in_global_host_wide_section_takes_the_default(
    tmp_path: Path,
) -> None:
    repo, repo_config, fleet = _layers(
        tmp_path,
        "runner_allocation:\n  enabled: true\n  threads_per_slot:\n",
        "dispatch:\n  default_limit: 2\n",
    )
    config = load_layered_config(repo, repo_config, fleet_dir_override=fleet)
    assert [Path(s).name for s in config.sources] == ["config.yaml", "orchestrator.config.yaml"]
    assert config.runner_allocation.threads_per_slot == 0  # declared default, loud no longer


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
    # R3-N2: a null on a NON-SCALAR field stays a loud ConstructionError (not a
    # FieldError), and the opaque "'<' not supported ... NoneType" text still
    # names which keys were null. (Since #2116 a scalar null defaults earlier and
    # never reaches the constructor, so only non-scalar kinds land here.)
    with pytest.raises(
        ConstructionError, match=r"null keys: .*runner_allocation\.max_running_per_repo"
    ):
        build_config_from_data(
            {"runner_allocation": {"enabled": True, "max_running_per_repo": None}}
        )
