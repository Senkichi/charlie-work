"""Characterization of the config rejection set (ADR-0007 migration, step cfg-2).

Moving every section's hand-written checks into field metadata must not change
WHICH configs load, only the text of the error. This file is that invariant's
automatic signal: a probe matrix recorded against the pre-migration build code
(``tests/fixtures/config_rejection_matrix.json``) and replayed on every commit.

Each cell is one ``build_config_from_data`` call with one probe value placed at
one key path, recorded as:

* ``R``   -- rejected (``ValueError``/``TypeError``; ``ConfigError`` is a ``ValueError``)
* ``X``   -- crashed with some other exception (a clean rejection now also passes)
* ``A<h>`` -- accepted; ``h`` digests the fields that differ from the default
  config, so a change in coercion (``str(item)``, ``float(x)``, ``.strip()``)
  shows up, while a change to a DEFAULT does not.

Targets are derived from the dataclass fields at record time and stored in the
fixture; the replay iterates the recorded targets only, so it cannot silently
shrink when a field is renamed -- a missing path is a recorded failure.
Message text is deliberately NOT compared here (tests/test_config_validation.py
and the per-section tests own the grammar).
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import types
import typing
from collections.abc import Mapping
from pathlib import Path
from unittest import mock

import pytest

from charlie_work.config import OrchestratorConfig, build_config_from_data

FIXTURE = Path(__file__).parent / "fixtures" / "config_rejection_matrix.json"

PROBES: list[tuple[str, object]] = [
    ("none", None), ("true", True), ("false", False), ("0", 0), ("1", 1), ("-1", -1),
    ("2", 2), ("1.5", 1.5), ("0.5", 0.5), ("-0.5", -0.5), ("inf", math.inf), ("nan", math.nan),
    ("empty", ""), ("blank", "  "), ("x", "x"), ("ab", "a b"), ("../x", "../x"),
    ("/abs", "/abs"), ("oldest", "oldest"), ("warn", "warn"), ("http", "http"),
    ("off", "off"), ("front", "front_of_train"), ("codex", "codex"),
    ("claude", "claude-code"), ("manual", "manual"), ("--x", "--x"),
    ("ph_ok", "{prompt_path}"), ("ph_bad", "{bad}"), ("ph_empty", "{}"), ("re_bad", "["),
    ("l_empty", []), ("l_str", ["a"]), ("l_flag", ["--squash"]),
    ("l_dflag", ["--delete-branch"]), ("l_int", [1]), ("l_none", [None]),
    ("l_ph_ok", ["x", "{prompt_path}"]), ("l_ph_bad", ["x", "{bad}"]),
    ("d_empty", {}), ("d_ss", {"a": "b"}), ("d_si", {"a": 1}),
]  # fmt: skip
SECTION_PROBES = ("none", "true", "0", "x", "l_empty", "l_str", "d_empty")
PROBE_MAP = dict(PROBES)

_PROVIDER_BASE = {
    "base_url": "http://x",
    "api_key_env": "K",
    "model": "m",
    "input_usd_per_mtok": 1,
    "output_usd_per_mtok": 2,
}
_RULE_BASE = {"pattern": "a", "kind": "k"}
# Element seeds for ``tuple[Dataclass, ...]`` fields, by element class (default ``_RULE_BASE``).
_ENTRY_SEEDS = {"RoleEntry": {"harness": "devin-shell"}}
# A role chain is only valid beside a primary that records sessions (``manual`` -- the worker
# default -- is rejected in any chain with fallbacks), so probes under ``worker.fallbacks`` /
# ``reviewer.fallbacks`` are placed on top of one, else every cell would record ``R``.
_CONTEXT = {
    ("worker", "fallbacks"): {"worker": {"harness": "claude-code", "model": "m"}},
    ("reviewer", "fallbacks"): {"reviewer": {"harness": "claude-code", "model": "m"}},
}
_FB = {"harness": "devin-shell", "model": "x"}
_WORKER = {"harness": "claude-code", "model": "m"}

# Whole-config scenarios the per-key matrix cannot express (cross-field rules,
# legacy aliases, derived defaults, scoped sections). Name -> raw config.
CASES: dict[str, dict] = {
    "deadlock": {"auto_merge": {"require_current_base": True, "update_branch_strategy": "off"}},
    "deadlock_legacy": {"auto_merge": {"require_current_base": True, "update_open_prs": "off"}},
    "deadlock_bool": {"auto_merge": {"require_current_base": True, "update_open_prs": False}},
    "no_deadlock": {"auto_merge": {"require_current_base": False, "update_branch_strategy": "off"}},
    "strategy_bool": {"auto_merge": {"update_branch_strategy": True}},
    "strategy_upper": {"auto_merge": {"update_branch_strategy": "BROADCAST"}},
    "strategy_bad": {"auto_merge": {"update_branch_strategy": "nope"}},
    "strategy_int": {"auto_merge": {"update_branch_strategy": 3}},
    "strategy_nonnull_legacy": {
        "auto_merge": {"update_open_prs": "next", "update_branch_strategy": "off"}
    },
    "legacy_bad": {"auto_merge": {"update_open_prs": "sometimes"}},
    "legacy_int": {"auto_merge": {"update_open_prs": 4}},
    "legacy_true": {"auto_merge": {"update_open_prs": True}},
    "merge_flags_ok": {"auto_merge": {"merge_flags": ["--squash-x", "--foo=bar"]}},
    "merge_flags_managed": {"auto_merge": {"merge_flags": ["--delete-branch=1"]}},
    "merge_flags_nonflag": {"auto_merge": {"merge_flags": ["squash"]}},
    "merge_flags_empty": {"auto_merge": {"merge_flags": []}},
    "merge_flags_int": {"auto_merge": {"merge_flags": [1]}},
    "required_checks_list": {"auto_merge": {"required_checks": ["a", 1]}},
    "required_checks_str": {"auto_merge": {"required_checks": "a"}},
    "mq_label_strip": {"auto_merge": {"mergequeue_label": "  mq  "}},
    "mq_wedge_int": {"auto_merge": {"mergequeue_wedge_hours": 3}},
    "infra_blocked_ok": {"auto_merge": {"infra_blocked": {"enabled": False, "persistence_passes": 2}}},
    "reviewer_fraction_no_effort": {"reviewer": {"effort_experiment_fraction": 0.25}},
    "reviewer_fraction_effort": {
        "reviewer": {"effort_experiment_fraction": 0.25, "effort": "high"}
    },
    "reviewer_fraction_range": {"reviewer": {"effort_experiment_fraction": 1.5}},
    "reviewer_fraction_bool": {"reviewer": {"effort_experiment_fraction": True}},
    "reviewer_harness_bad": {"reviewer": {"harness": "nope"}},
    "reviewer_harness_devin": {"reviewer": {"harness": "devin"}},
    "reviewer_effort_int": {"reviewer": {"effort": 5}},
    "reviewer_salt_int": {"reviewer": {"effort_experiment_salt": 5}},
    "worker_harness_bad": {"worker": {"harness": "nope"}},
    "worker_harness_command": {"worker": {"harness": "command"}},
    "worker_harness_int": {"worker": {"harness": 5}},
    "worker_model_int": {"worker": {"model": 5}},
    "rescue_worker_nonmapping": {"rescue": {"worker": 5}},
    "rescue_worker_none": {"rescue": {"worker": None}},
    "rescue_worker_unknown": {"rescue": {"worker": {"bogus": 1}}},
    "rescue_worker_harness_any": {"rescue": {"worker": {"harness": "zzz"}}},
    "rescue_worker_absent": {"rescue": {"enabled": True}},
    "rescue_reviewer_list": {"rescue": {"reviewer": ["x"]}},
    "rescue_reviewer_ok": {"rescue": {"reviewer": {"harness": "devin", "model": "m"}}},
    # Issue #2086/#2088 role chain -- recorded against origin/main's own source.
    "worker_fallbacks_ok": {"worker": {**_WORKER, "fallbacks": [_FB]}},
    "worker_fallbacks_three_ok": {
        "worker": {**_WORKER, "fallbacks": [{**_FB, "model": str(i)} for i in range(3)]}
    },
    "worker_fallbacks_empty": {"worker": {**_WORKER, "fallbacks": []}},
    "worker_fallbacks_null": {"worker": {**_WORKER, "fallbacks": None}},
    "worker_fallbacks_null_manual": {"worker": {"fallbacks": None}},
    "worker_fallbacks_bad_harness": {"worker": {**_WORKER, "fallbacks": [{"harness": "bogus"}]}},
    "worker_fallbacks_harness_missing": {"worker": {**_WORKER, "fallbacks": [{"model": "x"}]}},
    "worker_fallbacks_harness_null": {"worker": {**_WORKER, "fallbacks": [{"harness": None}]}},
    "worker_fallbacks_harness_int": {"worker": {**_WORKER, "fallbacks": [{"harness": 5}]}},
    "worker_fallbacks_dup_primary": {
        "worker": {**_WORKER, "fallbacks": [{"harness": "claude-code", "model": "m"}]}
    },
    "worker_fallbacks_dup_fallbacks": {"worker": {**_WORKER, "fallbacks": [_FB, _FB]}},
    "worker_fallbacks_same_harness_other_model": {
        "worker": {**_WORKER, "fallbacks": [{"harness": "claude-code", "model": "n"}]}
    },
    "worker_fallbacks_four": {
        "worker": {**_WORKER, "fallbacks": [{**_FB, "model": str(i)} for i in range(4)]}
    },
    "worker_fallbacks_sync_primary": {"worker": {"harness": "manual", "fallbacks": [_FB]}},
    "worker_fallbacks_sync_default_primary": {"worker": {"fallbacks": [_FB]}},
    "worker_fallbacks_sync_entry": {
        "worker": {**_WORKER, "fallbacks": [{"harness": "command", "model": "x"}]}
    },
    "worker_fallbacks_effort": {"worker": {**_WORKER, "fallbacks": [{**_FB, "effort": "high"}]}},
    "worker_fallbacks_effort_empty": {"worker": {**_WORKER, "fallbacks": [{**_FB, "effort": ""}]}},
    "worker_fallbacks_unknown_key": {"worker": {**_WORKER, "fallbacks": [{**_FB, "bogus": 1}]}},
    "worker_fallbacks_model_null": {
        "worker": {**_WORKER, "fallbacks": [{"harness": "devin-shell", "model": None}]}
    },
    "worker_fallbacks_model_int": {"worker": {**_WORKER, "fallbacks": [{**_FB, "model": 5}]}},
    "worker_fallbacks_not_list": {"worker": {**_WORKER, "fallbacks": "devin-shell"}},
    "worker_fallbacks_mapping": {"worker": {**_WORKER, "fallbacks": {}}},
    "worker_fallbacks_entry_str": {"worker": {**_WORKER, "fallbacks": ["devin-shell"]}},
    "reviewer_fallbacks_ok": {"reviewer": {"fallbacks": [_FB]}},
    "reviewer_fallbacks_effort_ok": {"reviewer": {"fallbacks": [{**_FB, "effort": "high"}]}},
    "reviewer_fallbacks_effort_int": {"reviewer": {"fallbacks": [{**_FB, "effort": 5}]}},
    "reviewer_fallbacks_null": {"reviewer": {"fallbacks": None}},
    "reviewer_fallbacks_harness_set": {
        "reviewer": {"fallbacks": [{"harness": "manual", "model": "x"}]}
    },
    "reviewer_fallbacks_bad_harness": {"reviewer": {"fallbacks": [{"harness": "bogus"}]}},
    "reviewer_fallbacks_dup_effort_differs": {
        "reviewer": {"fallbacks": [{"harness": "claude-code", "model": "m", "effort": "high"}]}
    },
    "reviewer_fallbacks_dup_primary": {
        "reviewer": {"harness": "claude-code", "model": "m", "fallbacks": [_WORKER]}
    },
    "reviewer_fallbacks_four": {
        "reviewer": {"fallbacks": [{**_FB, "model": str(i)} for i in range(4)]}
    },
    "reviewer_fallbacks_devin_primary": {
        "reviewer": {"harness": "devin", "fallbacks": [_FB]}
    },
    # rescue.worker / rescue.reviewer reuse WorkerRoleConfig and store ``fallbacks`` unparsed
    # (main's known gap): anything is accepted, verbatim, and a null stays ``None``.
    "rescue_worker_fallbacks_raw": {"rescue": {"worker": {"fallbacks": [{"harness": "bogus"}]}}},
    "rescue_worker_fallbacks_str": {"rescue": {"worker": {"fallbacks": "zz"}}},
    "rescue_reviewer_fallbacks_null": {"rescue": {"reviewer": {"fallbacks": None}}},
    "rescue_reviewer_fallbacks_ok": {"rescue": {"reviewer": {"fallbacks": [_FB]}}},
    "api_enabled_empty": {"api_worker": {"enabled": True}},
    "api_enabled_missing_provider": {"api_worker": {"enabled": True, "provider": "zz"}},
    "api_enabled_ok": {
        "api_worker": {"enabled": True, "provider": "p", "providers": {"p": _PROVIDER_BASE}}
    },
    "api_enabled_zero_price": {
        "api_worker": {
            "enabled": True,
            "provider": "p",
            "providers": {"p": {**_PROVIDER_BASE, "input_usd_per_mtok": 0}},
        }
    },
    "api_disabled_zero_price": {
        "api_worker": {"providers": {"p": {**_PROVIDER_BASE, "input_usd_per_mtok": 0}}}
    },
    "api_enabled_neg_cached": {
        "api_worker": {
            "enabled": True,
            "provider": "p",
            "providers": {"p": {**_PROVIDER_BASE, "cached_input_usd_per_mtok": -1}},
        }
    },
    "api_enabled_blank_keyenv": {
        "api_worker": {
            "enabled": True,
            "provider": "p",
            "providers": {"p": {**_PROVIDER_BASE, "api_key_env": " "}},
        }
    },
    "api_provider_missing_key": {"api_worker": {"providers": {"p": {"base_url": "u"}}}},
    "api_provider_numeric_name": {"api_worker": {"providers": {1: _PROVIDER_BASE}}},
    "api_budget_ok": {"api_worker": {"budget": {"max_usd_per_session": 2}}},
    "api_budget_neg": {"api_worker": {"budget": {"lifetime_usd": -1}}},
    "runner_floors_disagree": {
        "runner_scaling": {"enabled": True, "min_runners": 1},
        "runner_allocation": {"enabled": True, "min_running_per_repo": 3},
    },
    "runner_floors_ok": {
        "runner_scaling": {"enabled": True, "min_runners": 3},
        "runner_allocation": {"enabled": True, "min_running_per_repo": 3},
    },
    "runner_floors_scaling_off": {
        "runner_scaling": {"enabled": False, "min_runners": 1},
        "runner_allocation": {"enabled": True, "min_running_per_repo": 3},
    },
    "alloc_max_running_mapping": {"runner_allocation": {"max_running_per_repo": {"a/b": 1}}},
    "alloc_max_running_pairs": {"runner_allocation": {"max_running_per_repo": [["a/b", 1]]}},
    "alloc_max_running_dup": {
        "runner_allocation": {"max_running_per_repo": [["a/b", 1], ["a/b", 2]]}
    },
    "alloc_min_above_ceiling": {
        "runner_allocation": {"max_running_per_repo": [["a/b", 1]], "min_running_per_repo": 2}
    },
    "alloc_reserved_neg": {"runner_allocation": {"reserved_threads": -1}},
    "alloc_bool_int": {"runner_allocation": {"max_running_runners": True}},
    "scaling_bool_int": {"runner_scaling": {"min_runners": True}},
    "scaling_min_gt_max": {"runner_scaling": {"min_runners": 5, "max_runners": 1}},
    "escalation_ok": {"runner_capacity_escalation": {"starvation_escalation_minutes": 5}},
    "escalation_zero": {"runner_capacity_escalation": {"starvation_escalation_minutes": 0}},
    "escalation_bool": {"runner_capacity_escalation": {"enabled": 1}},
    "escalation_unknown_and_bad": {
        "runner_capacity_escalation": {"bogus": 1, "starvation_escalation_minutes": "x"}
    },
    "escalation_nonmapping": {"runner_capacity_escalation": 5},
    "fleet_sup_ok": {"fleet_supervisor": {"poll_interval_seconds": 30}},
    "fleet_sup_bool_int": {"fleet_supervisor": {"poll_interval_seconds": True}},
    "fleet_sup_bad_bool": {"fleet_supervisor": {"zero_pass_alarm_enabled": 1}},
    "fleet_sup_legacy": {"supervisor": {"poll_interval_seconds": 30}},
    "fleet_sup_legacy_relocated": {"supervisor": {"zero_pass_alarm_passes": 4}},
    "fleet_sup_conflict": {
        "fleet_supervisor": {"zero_pass_alarm_passes": 3},
        "supervisor": {"zero_pass_alarm_passes": 4},
    },
    "fleet_sup_same": {
        "fleet_supervisor": {"zero_pass_alarm_passes": 3},
        "supervisor": {"zero_pass_alarm_passes": 3},
    },
    "fleet_sup_unknown": {"fleet_supervisor": {"bogus": 1}},
    "deesc_ok": {"deescalation": {"operator_queue_depth_threshold": 3}},
    "deesc_neg": {"deescalation": {"operator_queue_depth_threshold": -1}},
    "deesc_bool": {"deescalation": {"identical_reason_recurrence_window_minutes": True}},
    "deesc_unknown_ignored": {"deescalation": {"bogus": 1, "enabled": "zzz", "interval_minutes": "q"}},
    "deesc_nonmapping": {"deescalation": 5},
    "deesc_null_key": {"deescalation": {"operator_queue_depth_threshold": None}},
    "local_issues_enabled": {"local_issues": {"enabled": True}},
    "local_issues_enabled_explicit": {
        "local_issues": {"enabled": True},
        "dispatch": {"worker_template": "w.md"},
        "review_dispatch": {"enabled": False},
    },
    "local_issues_dir_abs": {"local_issues": {"issues_dir": "C:/x"}},
    "local_issues_dir_win_abs": {"local_issues": {"issues_dir": "\\\\srv\\share"}},
    "local_issues_dir_dotdot": {"local_issues": {"issues_dir": "a/../b"}},
    "local_issues_dir_dotdot_win": {"local_issues": {"issues_dir": "a\\..\\b"}},
    "local_issues_dir_ok": {"local_issues": {"issues_dir": "a/b"}},
    "local_issues_dir_blank": {"local_issues": {"issues_dir": "   "}},
    "dispatch_nondict": {"dispatch": 5},
    "dispatch_list": {"dispatch": ["a"]},
    "dispatch_test_command_null": {"dispatch": {"test_command": None}},
    "dispatch_injected_paths": {"dispatch": {"injected_paths": ["a", "./b", "a"]}},
    "dispatch_injected_paths_int": {"dispatch": {"injected_paths": [1]}},
    "dispatch_materialize_ints": {"dispatch": {"materialize_dirs": [1, 2]}},
    "dispatch_materialize_str": {"dispatch": {"materialize_dirs": "a"}},
    "dispatch_stagger_true": {"dispatch": {"launch_stagger_seconds": True}},
    "dispatch_stagger_float": {"dispatch": {"launch_stagger_seconds": 1.5}},
    "labels_blocked_dropped": {"labels": {"blocked": "agent:blocked"}},
    "labels_int": {"labels": {"queued": 5}},
    "notify_list": {"notify": {"shell_command": ["a", 1]}},
    "notify_bad_sink": {"notify": {"sink": "nope"}},
    "notify_str_cmd": {"notify": {"shell_command": "x"}},
    "heartbeat_str": {"heartbeat": {"stale_mention_parked_labels": "a"}},
    "heartbeat_list": {"heartbeat": {"stale_mention_parked_labels": ["a", "b"]}},
    "heartbeat_list_int": {"heartbeat": {"stale_mention_parked_labels": [1]}},
    "heartbeat_int": {"heartbeat": {"stale_mention_parked_labels": 3}},
    "post_mortem_rule_ok": {"post_mortem": {"signature_rules": [{"pattern": "a+", "kind": "k"}]}},
    "post_mortem_rule_regex": {"post_mortem": {"signature_rules": [{"pattern": "[", "kind": "k"}]}},
    "post_mortem_rule_missing": {"post_mortem": {"signature_rules": [{"pattern": "a"}]}},
    "post_mortem_rule_nonmap": {"post_mortem": {"signature_rules": ["a"]}},
    "post_mortem_rule_unknown": {
        "post_mortem": {"signature_rules": [{"pattern": "a", "kind": "k", "z": 1}]}
    },
    "post_mortem_rules_str": {"post_mortem": {"signature_rules": "a"}},
    "devin_cmd_ok": {"devin": {"dispatch_command": ["x", "{prompt_path}", "{branch}"]}},
    "devin_cmd_model_args": {"devin": {"dispatch_command": ["x", "{model_args}"]}},
    "devin_shell_model_args": {"devin": {"shell_command": ["x", "{model_args}"]}},
    "devin_cmd_str": {"devin": {"dispatch_command": "x {prompt_path}"}},
    "devin_cmd_str_bad": {"devin": {"dispatch_command": "x {nope}"}},
    "devin_cmd_int": {"devin": {"dispatch_command": 5}},
    "devin_cmd_items_int": {"devin": {"dispatch_command": [1, 2]}},
    "devin_cmd_malformed": {"devin": {"dispatch_command": ["x", "{prompt_path"]}},
    "devin_cmd_positional": {"devin": {"dispatch_command": ["x", "{0}"]}},
    "devin_cmd_empty_list": {"devin": {"dispatch_command": []}},
    "devin_worker_env_ok": {"devin": {"worker_env": {"A": "b", 1: 2}}},
    "devin_worker_env_list": {"devin": {"worker_env": ["a"]}},
    "claude_cmd_bad": {"claude_code": {"command": ["x", "{model}"]}},
    "claude_effort_int": {"claude_code": {"effort": 5}},
    "rescue_cmd_ok": {"rescue": {"reviewer_command": ["x", "{model}", "{prompt_path}"]}},
    "rescue_cmd_bad": {"rescue": {"reviewer_command": ["x", "{branch}"]}},
    "runtime_preflight_ok": {"runtime": {"preflight": {"disk_floor_gb": 5}}},
    "runtime_preflight_bool": {"runtime": {"preflight": {"disk_floor_gb": True}}},
    "runtime_preflight_unknown": {"runtime": {"preflight": {"bogus": 1}}},
    "runtime_preflight_nonmap": {"runtime": {"preflight": 5}},
    "runtime_breaker_zero": {"runtime": {"gh_circuit_breaker": {"failure_threshold": 0}}},
    "runtime_timeout_zero": {"runtime": {"gh_timeout_seconds": 0}},
    "runtime_timeout_ok": {"runtime": {"gh_timeout_seconds": 0.5}},
    "runtime_retries_true": {"runtime": {"gh_max_retries": True}},
    "runtime_retries_neg": {"runtime": {"gh_max_retries": -3}},
    "runtime_ring_zero": {"runtime": {"event_ring_size": 0}},
    "watchdog_action_bad": {"watchdog": {"cost_budget_action": "explode"}},
    "watchdog_budget_true": {"watchdog": {"token_budget": True}},
    "watchdog_budget_neg": {"watchdog": {"token_budget": -5}},
    "watchdog_cost_neg": {"watchdog": {"cost_budget_usd": -1.5}},
    "test_adequacy_marker_blank": {"test_adequacy": {"exempt_marker": "  "}},
    "test_adequacy_marker_empty": {"test_adequacy": {"exempt_marker": ""}},
    "test_adequacy_lists": {"test_adequacy": {"test_path_globs": ["a", "b"]}},
    "test_adequacy_cov_int": {"test_adequacy": {"min_diff_coverage": 3}},
    "coverage_probe_ratio_true": {"coverage_probe": {"branch_to_assert_ratio_threshold": True}},
    "fleet_launch_lock_inf": {"fleet": {"launch_lock_wait_seconds": math.inf}},
    "fleet_launch_lock_neg": {"fleet": {"launch_lock_wait_seconds": -1}},
    "fleet_launch_lock_true": {"fleet": {"launch_lock_wait_seconds": True}},
    "fleet_global_max_true": {"fleet": {"global_max_concurrent_sessions": True}},
    "quota_probe_model_blank": {"quota_probe": {"model": " "}},
    "quota_probe_zero_interval": {"quota_probe": {"interval_minutes": 0}},
    "review_confirm_zero": {"review": {"foreign_issue_ref_confirm_passes": 0}},
    "review_markers_int": {"review": {"human_decision_markers": [1]}},
    "unknown_section": {"zzz": {}},
    "unknown_sections_two": {"zzz": {}, "yyy": 1},
    "sources_forged": {"sources": ["x"]},
    "empty": {},
}  # fmt: skip


# ----------------------------------------------------------------------- machinery
def _unwrap(tp: typing.Any) -> typing.Any:
    if typing.get_origin(tp) in (typing.Union, types.UnionType):
        members = [a for a in typing.get_args(tp) if a is not type(None)]
        if len(members) == 1:
            return members[0]
    return tp


def _is_dc(tp: typing.Any) -> bool:
    return isinstance(tp, type) and dataclasses.is_dataclass(tp)


def walk_targets() -> list[dict]:
    """Every (key path, leaf seed, container?) reachable from OrchestratorConfig."""
    out: list[dict] = []

    def visit(cls: type, path: list, seed: dict) -> None:
        hints = typing.get_type_hints(cls)
        for f in dataclasses.fields(cls):
            if not f.init or f.metadata.get("provenance"):
                continue
            tp = _unwrap(hints[f.name])
            here = [*path, f.name]
            origin, args = typing.get_origin(tp), typing.get_args(tp)
            sub: tuple[list, type, dict] | None = None
            if _is_dc(tp):
                sub = (here, tp, {})
            elif origin is tuple and args and _is_dc(args[0]):
                sub = ([*here, 0], args[0], dict(_ENTRY_SEEDS.get(args[0].__name__, _RULE_BASE)))
            elif origin in (dict, Mapping) and len(args) == 2 and _is_dc(args[1]):
                sub = ([*here, "p"], args[1], dict(_PROVIDER_BASE))
            out.append(
                {"id": ".".join(map(str, here)), "path": here, "seed": seed, "sub": bool(sub)}
            )
            if sub is not None:
                if sub[0] is not here:
                    out.append(
                        {"id": ".".join(map(str, sub[0])), "path": sub[0], "seed": {}, "sub": True}
                    )
                visit(sub[1], sub[0], sub[2])

    for f in dataclasses.fields(OrchestratorConfig):
        if f.metadata.get("provenance"):
            continue
        tp = typing.get_type_hints(OrchestratorConfig)[f.name]
        out.append({"id": f.name, "path": [f.name], "seed": {}, "sub": True})
        if _is_dc(tp):
            visit(tp, [f.name], {})
    # Legacy ``supervisor.<key>`` spellings of the relocated fleet_supervisor keys.
    from charlie_work.fleet_supervisor_config import FleetSupervisorConfig

    for f in dataclasses.fields(FleetSupervisorConfig):
        out.append(
            {
                "id": f"supervisor.{f.name}",
                "path": ["supervisor", f.name],
                "seed": {},
                "sub": False,
            }
        )
    return out


def place(path: list, value: object, seed: dict) -> dict:
    """Raw config with ``value`` at ``path``; the leaf's parent mapping starts from ``seed``."""

    def make(i: int) -> typing.Any:
        key, last = path[i], i == len(path) - 1
        child = value if last else make(i + 1)
        if isinstance(key, int):
            return [child]
        node = copy.deepcopy(seed) if last else {}
        node[key] = child
        return node

    raw = make(0)
    for prefix, base in _CONTEXT.items():
        if tuple(path[: len(prefix)]) == prefix:
            ctx = copy.deepcopy(base)
            ctx[prefix[0]].update(raw[prefix[0]])
            raw = {**raw, **ctx}
    return raw


def _changed(cfg: typing.Any) -> list[tuple[str, str, str]]:
    default = build_config_from_data({})
    rows = []
    for f in dataclasses.fields(cfg):
        if f.metadata.get("provenance"):
            continue
        a, b = getattr(cfg, f.name), getattr(default, f.name)
        if not dataclasses.is_dataclass(a):
            if repr(a) != repr(b):
                rows.append((f.name, "", repr(a)))
            continue
        for g in dataclasses.fields(a):
            if repr(getattr(a, g.name)) != repr(getattr(b, g.name)):
                rows.append((f.name, g.name, repr(getattr(a, g.name))))
    return rows


# Defaults such as dispatch.host_load_max_pytest_trees derive from os.cpu_count(), and a
# cell's hash is its diff against the default config -- so an unpinned count makes the
# hash host-dependent (a 4-CPU runner's default of 2 trees erased the "<- 2" probe's diff).
# The fixture was recorded on a 16-CPU host; recording and replay both pin to it.
RECORDED_CPU_COUNT = 16


def code_of(raw: dict) -> str:
    with mock.patch("os.cpu_count", return_value=RECORDED_CPU_COUNT):
        return _code_of(raw)


def _code_of(raw: dict) -> str:
    try:
        cfg = build_config_from_data(copy.deepcopy(raw))
    except (ValueError, TypeError):
        return "R"
    except Exception:  # noqa: BLE001 - recorded as a crash class, not hidden
        return "X"
    return "A" + hashlib.md5(repr(_changed(cfg)).encode()).hexdigest()[:4]


def same(recorded: str, now: str) -> bool:
    if recorded == now:
        return True
    return {recorded, now} <= {"R", "X"}  # a clean rejection replacing a crash is fine


def compute() -> dict:
    targets = walk_targets()
    cells: dict[str, list[str]] = {}
    for t in targets:
        cells[t["id"]] = [code_of(place(t["path"], v, t["seed"])) for _, v in PROBES]
    sections = {
        t["id"]: [code_of({t["id"]: PROBE_MAP[n]}) for n in SECTION_PROBES]
        for t in targets
        if len(t["path"]) == 1
    }
    unknown = {
        t["id"]: code_of(place([*t["path"], "zzz_unknown"], 1, t["seed"]))
        for t in targets
        if t["sub"]
    }
    cases = {name: code_of(raw) for name, raw in CASES.items()}
    return {
        "probes": [n for n, _ in PROBES],
        "section_probes": list(SECTION_PROBES),
        "targets": {t["id"]: t["path"] for t in targets},
        "seeds": {t["id"]: t["seed"] for t in targets},
        "cells": cells,
        "sections": sections,
        "unknown": unknown,
        "cases": cases,
    }


# ------------------------------------------------------------------------- tests
@pytest.fixture(scope="module")
def recorded() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_probe_matrix_is_nontrivial(recorded):
    """Positive control: the fixture contains both rejections and acceptances, broadly."""
    flat = [c for row in recorded["cells"].values() for c in row]
    rejected = sum(1 for c in flat if c == "R")
    accepted = sum(1 for c in flat if c.startswith("A"))
    assert len(recorded["targets"]) > 300
    assert rejected > 2000
    assert accepted > 5000
    assert recorded["cases"]["deadlock"] == "R"
    assert recorded["cases"]["no_deadlock"].startswith("A")


def test_probe_replay_detects_a_loosened_validator(recorded, monkeypatch):
    """Neuter control: a build that never rejects must NOT replay equal to the fixture."""
    import charlie_work.config as config_module

    real = config_module.build_config_from_data

    def lax(data):
        try:
            return real(data)
        except (ValueError, TypeError):
            return real({})

    monkeypatch.setitem(globals(), "build_config_from_data", lax)
    mismatches = [
        tid
        for tid, row in recorded["cells"].items()
        if not all(
            same(rec, code_of(place(recorded["targets"][tid], v, recorded["seeds"][tid])))
            for rec, (_, v) in zip(row, PROBES, strict=True)
        )
    ]
    assert len(mismatches) > 100


def test_key_probe_matrix_unchanged(recorded):
    bad = []
    for tid, row in recorded["cells"].items():
        path, seed = recorded["targets"][tid], recorded["seeds"][tid]
        for rec, (name, value) in zip(row, PROBES, strict=True):
            now = code_of(place(path, value, seed))
            if not same(rec, now):
                bad.append(f"{tid} <- {name}: recorded {rec}, now {now}")
    assert not bad, "rejection set changed:\n" + "\n".join(bad[:60]) + f"\n({len(bad)} total)"


def test_section_level_probes_unchanged(recorded):
    bad = []
    for tid, row in recorded["sections"].items():
        for rec, name in zip(row, recorded["section_probes"], strict=True):
            now = code_of({tid: PROBE_MAP[name]})
            if not same(rec, now):
                bad.append(f"{tid} = {name}: recorded {rec}, now {now}")
    assert not bad, "\n".join(bad)


def test_unknown_key_probes_unchanged(recorded):
    bad = []
    for tid, rec in recorded["unknown"].items():
        path, seed = recorded["targets"][tid], recorded["seeds"][tid]
        now = code_of(place([*path, "zzz_unknown"], 1, seed))
        if not same(rec, now):
            bad.append(f"{tid}: recorded {rec}, now {now}")
    assert not bad, "\n".join(bad)
    assert all(v == "R" or v.startswith("A") for v in recorded["unknown"].values())


def test_whole_config_cases_unchanged(recorded):
    bad = [
        f"{name}: recorded {recorded['cases'].get(name)}, now {code_of(raw)}"
        for name, raw in CASES.items()
        if not same(recorded["cases"].get(name, "?"), code_of(raw))
    ]
    assert not bad, "\n".join(bad)


def test_every_live_config_path_has_a_recorded_row(recorded):
    """A knob added after the migration must be recorded against origin/main's source
    (see the ``review_exec_rejection_max_resumes`` and ``global_max_concurrent_reviews``
    rows), not silently left outside the matrix. Replay only iterates recorded targets,
    so without this guard a new key is invisible to the rejection-set signal."""
    live = {t["id"] for t in walk_targets()}
    missing = sorted(live - set(recorded["targets"]))
    assert not missing, f"config paths absent from the rejection matrix fixture: {missing}"
