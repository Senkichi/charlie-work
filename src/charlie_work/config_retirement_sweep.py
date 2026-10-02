"""Fleet-wide deprecated-config-key retirement sweep (issue #1976).

Once per fleet pass, ``run_config_retirement_sweep`` checks every entry in
``config_deprecations.DEPRECATED_CONFIG_KEYS`` against every config layer of
every registered repo -- the user-global fleet layer plus each repo's
``orchestrator.config.yaml`` (whether it is a tracked file or an untracked
local profile). Presence is derived from the raw layer files, never from
events: a repo that still sets the key but has not run a pass recently is
still a user. The sweep fails closed: a layer that exists but cannot be
parsed, and a registered repo whose ``repo_root`` is missing or unreachable,
both count as *unproven* rather than absent, and hold the quiet window
(forbid arming) until they read cleanly again.

Per key, a fleet-dir sidecar (``config_retirement_state.json``) records the
first pass the key was observed absent everywhere; reappearance resets that
timestamp. Once the ``runtime.config_retirement_quiet_days`` window elapses
with the key still absent, the sweep marks the registry's ``removal_issue``
Ready through the normal label edge (``config_retirement_ready``), posts one
comment naming the repos and layers checked plus the quiet-window start, and
records ``config_key_retirement_armed``. If the key reappears after arming,
the sweep records ``config_key_retirement_regressed`` and comments again but
leaves the label alone -- a worker may already be running.
"""

from __future__ import annotations

import datetime
import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from . import layout
from .config import OrchestratorConfig
from .config_deprecations import (
    DEPRECATED_CONFIG_KEYS,
    DeprecatedConfigKey,
    deprecated_keys_in,
)
from .fleet_paths import fleet_dir, warn_fleet_dir_virtualization_on_write
from .fleet_registry import _load_registry, _select_repos
from .global_config import config_layer_paths, describe_config_file
from .instrumentation import log_event
from .labels import transition
from .local_issue_files import write_text_atomic
from .state import utc_now
from .subprocess_runner import run_captured

logger = logging.getLogger(__name__)


def _iso(dt: datetime.datetime) -> str:
    """ISO-8601 UTC timestamp matching the fleet event-store convention."""
    return dt.astimezone(datetime.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_iso(text: Any) -> datetime.datetime | None:
    if not isinstance(text, str):
        return None
    try:
        dt = datetime.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.UTC)
    return dt


def _load_retirement_state(path: Path) -> dict[str, dict[str, Any]]:
    """Load the retirement sidecar's ``keys`` map. Missing/corrupt is empty.

    Starting fresh on corruption re-opens the quiet window rather than
    arming on stale data -- the safe direction, same as the capacity-
    starvation sidecar loader this mirrors (issue #763).
    """
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, LookupError, ValueError, OSError):
        logger.warning("Config retirement state %s unreadable; starting fresh", path)
        return {}
    keys = data.get("keys") if isinstance(data, dict) else None
    if not isinstance(keys, dict):
        return {}
    cleaned: dict[str, dict[str, Any]] = {}
    for dotted, entry in keys.items():
        if isinstance(dotted, str) and isinstance(entry, dict):
            cleaned[dotted] = dict(entry)
    return cleaned


def _save_retirement_state(path: Path, keys: dict[str, dict[str, Any]]) -> None:
    """Atomically persist the retirement sidecar (temp file + ``replace()``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    warn_fleet_dir_virtualization_on_write(
        path.parent, context="writing config_retirement_state.json"
    )
    payload = {"version": 1, "generated_at": utc_now(), "keys": keys}
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


@dataclass(frozen=True)
class _LayerSlot:
    """One config-layer file the sweep inspects for a registered key."""

    repo: str | None  # repo_key, or None for the shared fleet-global layer
    layer: str  # "user-global" | "repo-tracked" | "repo-untracked-local"
    path: Path


def _repo_layer_kind(repo_root: Path, path: Path, run_git: Callable[..., Any]) -> str:
    """Classify a repo config layer as ``repo-tracked`` or ``repo-untracked-local``.

    The distinction matters for retirement: a tracked file needs a committed
    change to drop the key, while an untracked local profile is a single
    machine's residue. ``git ls-files --error-unmatch`` is the verdict --
    it fails on untracked files and on non-git roots alike, and both mean
    "not tracked", which is the correct bucket either way.
    """
    try:
        rel = path.relative_to(repo_root)
    except ValueError:
        rel = path
    result = run_git(
        ["git", "ls-files", "--error-unmatch", "--", rel.as_posix()],
        cwd=repo_root,
        timeout_seconds=15,
    )
    return "repo-tracked" if getattr(result, "ok", False) else "repo-untracked-local"


def _enumerate_layer_slots(
    selected: list[tuple[str, dict[str, Any]]],
    *,
    fleet_dir_override: str | None,
    run_git: Callable[..., Any],
) -> tuple[list[_LayerSlot], list[dict[str, str]]]:
    """Every (repo, layer, path) slot the sweep inspects this pass.

    Returns ``(slots, blocked)``. The user-global fleet layer is host-wide --
    one slot for the whole pass. Repo layers come from ``config_layer_paths``,
    the pairing ``load_layered_config`` itself resolves through, so the sweep
    cannot drift away from the loader's real source set (issue #1976's "not a
    single file" requirement).

    *blocked* is one unreadable-layer record per registered repo whose layers
    cannot even be enumerated: no ``repo_root`` on the entry, or a
    ``repo_root`` that is missing/unreachable on disk. Absence of the key is
    unprovable for such a repo -- its config file could still set it -- so
    each record feeds the same fail-closed ``unreadable`` channel as an
    unparseable file and blocks arming until the registry entry is repaired
    or pruned (the #1372 stale-entry path owns that lifecycle).
    """
    slots: list[_LayerSlot] = [
        _LayerSlot(None, "user-global", layout.global_config_path(override=fleet_dir_override))
    ]
    blocked: list[dict[str, str]] = []
    for repo_key, entry in selected:
        repo_root_str = entry.get("repo_root")
        if not repo_root_str:
            blocked.append(
                {
                    "repo": repo_key,
                    "layer": "repo",
                    "path": str(entry.get("config_path") or "<unset>"),
                    "error": "registry entry has no repo_root",
                }
            )
            continue
        repo_root = Path(repo_root_str)
        try:
            root_is_dir = repo_root.is_dir()
        except (OSError, ValueError):
            root_is_dir = False
        if not root_is_dir:
            blocked.append(
                {
                    "repo": repo_key,
                    "layer": "repo",
                    "path": str(repo_root),
                    "error": (
                        "repo_root is not a readable directory "
                        f"({describe_config_file(repo_root)})"
                    ),
                }
            )
            continue
        explicit = entry.get("config_path")
        for layer, path in config_layer_paths(
            repo_root,
            Path(explicit) if explicit else None,
            fleet_dir_override=fleet_dir_override,
        ):
            if layer == "user-global":
                continue  # the host-wide slot was already enumerated once
            slots.append(_LayerSlot(repo_key, _repo_layer_kind(repo_root, path, run_git), path))
    return slots, blocked


def _scan_layer(slot: _LayerSlot) -> tuple[list[DeprecatedConfigKey], str | None]:
    """Read one layer file; return ``(found_keys, error)``.

    An absent file is a clean miss, not an error -- the slot was still
    "checked". A file that exists but cannot be read or parsed is an error:
    the sweep cannot prove the key absent from it, which must block arming
    (fail closed). ``exists()`` itself is inside the try because it raises
    (rather than returning False) on EACCES and unrepresentable paths --
    those are unproven-absence too, not a clean miss.
    """
    try:
        if not slot.path.exists():
            return [], None
        raw = yaml.safe_load(slot.path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError, yaml.YAMLError) as exc:
        return [], f"{type(exc).__name__}: {exc}"
    data = raw if isinstance(raw, dict) else {}
    return deprecated_keys_in(data), None


def _issue_labels(view: dict[str, Any]) -> set[str]:
    names = set()
    for label in view.get("labels") or []:
        if isinstance(label, dict) and isinstance(label.get("name"), str):
            names.add(label["name"])
    return names


def _post_comment(gh: Any, issue_number: int, body: str) -> bool:
    """Post one issue comment via a temp body file. Best-effort."""
    try:
        with tempfile.TemporaryDirectory(prefix="charlie-retire-") as tmp:
            body_path = Path(tmp) / "comment.md"
            write_text_atomic(body_path, body)
            gh.issue_comment(issue_number, body_path)
        return True
    except Exception:  # noqa: BLE001 — a comment failure must not fail the pass
        logger.exception("config-retirement comment post failed issue=%d", issue_number)
        return False


def _checked_layers_text(slots: list[_LayerSlot]) -> str:
    lines = []
    for slot in slots:
        scope = slot.repo if slot.repo is not None else "fleet"
        lines.append(f"- `{scope}` / {slot.layer}: `{slot.path}`")
    return "\n".join(lines)


def _arm_comment(
    entry: DeprecatedConfigKey,
    slots: list[_LayerSlot],
    quiet_start: str,
    quiet_days: float,
) -> str:
    replacement = (
        f" Use `{entry.replacement}` instead."
        if entry.replacement
        else " There is no replacement — the key was a retired no-op."
    )
    return (
        f"Config-retirement sweep (issue #1976): `{entry.dotted}` has been absent "
        f"from every config layer of every registered repo for the "
        f"{quiet_days}-day quiet window (first absent-everywhere observation "
        f"{quiet_start}).{replacement}\n\n"
        f"Layers checked:\n{_checked_layers_text(slots)}\n\n"
        f"Marking this issue Ready so removal can be scheduled. If the key "
        f"reappears anywhere a regression is recorded and commented here; the "
        f"Ready label stays on because a worker may already be running."
    )


def _regression_comment(entry: DeprecatedConfigKey, findings: list[dict[str, str]]) -> str:
    where = "\n".join(
        f"- `{f['repo'] or 'fleet'}` / {f['layer']}: `{f['path']}`" for f in findings
    )
    return (
        f"Config-retirement sweep (issue #1976): `{entry.dotted}` reappeared "
        f"after this issue was marked Ready. The key is still set in:\n{where}\n\n"
        f"The Ready label is retained -- a worker may already be running the "
        f"removal. Whoever re-added the key should remove it before this "
        f"issue's worker deletes the parser support."
    )


def _process_key(
    entry: DeprecatedConfigKey,
    *,
    slots: list[_LayerSlot],
    presence: dict[str, list[dict[str, str]]],
    unreadable: dict[str, list[dict[str, str]]],
    keys_state: dict[str, dict[str, Any]],
    quiet_days: float,
    labels: Any,
    gh: Any,
    fleet_state_path: Path,
    dry_run: bool,
    now: datetime.datetime,
) -> dict[str, Any]:
    """Advance one registry entry's retirement state; return its summary."""
    dotted = entry.dotted
    findings = presence.get(dotted, [])
    blocked_by = unreadable.get(dotted, [])
    rec = keys_state.setdefault(dotted, {})
    summary: dict[str, Any] = {
        "present_in": findings,
        "unreadable_layers": blocked_by,
        "armed": rec.get("armed_at") is not None,
        "regressed": False,
    }

    if findings:
        rec["last_present"] = findings
        if rec.get("armed_at"):
            # Reappeared after arming: record and comment once per episode,
            # but never strip Ready -- a worker may already be running.
            summary["regressed"] = True
            if not rec.get("regression_reported_at") and not dry_run:
                _post_comment(gh, entry.removal_issue, _regression_comment(entry, findings))
                # write-gate-exempt(issue=1976): fleet-level sweep; no write_gate param, same shape as capacity_starvation_escalation.
                log_event(
                    fleet_state_path,
                    "config_key_retirement_regressed",
                    {
                        "section": entry.section,
                        "key": entry.key,
                        "issue_number": entry.removal_issue,
                        "present_in": findings,
                    },
                )
                rec["regression_reported_at"] = _iso(now)
        else:
            rec["absent_since"] = None  # reappearance resets the quiet clock
        return summary

    # Absent from every readable layer this pass.
    if not rec.get("armed_at"):
        rec["regression_reported_at"] = None
        if blocked_by:
            # A layer exists but could not be read: absence is unproven, so
            # the quiet clock cannot advance (fail closed).
            rec["absent_since"] = None
            return summary
        absent_since = _parse_iso(rec.get("absent_since"))
        if absent_since is None:
            # First clean pass: record the window start and fall through so
            # quiet_days=0 -- the documented "retire on the first clean pass"
            # opt-out -- arms immediately (elapsed 0 >= 0).
            rec["absent_since"] = _iso(now)
            absent_since = now
        elapsed_days = (now - absent_since).total_seconds() / 86400.0
        summary["quiet_elapsed_days"] = elapsed_days
        if elapsed_days < quiet_days or dry_run:
            if dry_run and elapsed_days >= quiet_days:
                summary["would_arm"] = True
            return summary
        # Window elapsed: consult the removal issue before acting.
        try:
            view = gh.issue_view(entry.removal_issue)
        except Exception as exc:  # noqa: BLE001 — unreadable issue: try again next pass
            logger.warning(
                "config-retirement sweep: cannot view issue #%d: %s",
                entry.removal_issue,
                exc,
            )
            return summary
        state = str(view.get("state") or "").lower()
        label_names = _issue_labels(view)
        if state == "closed":
            # Closed removal issue: do nothing, and do not pretend it armed.
            summary["issue_closed"] = True
            return summary
        if labels.ready in label_names:
            # Already marked Ready by another path: adopt it in the sidecar
            # (regression tracking needs the flag) without re-posting.
            rec["armed_at"] = _iso(now)
            summary["armed"] = True
            summary["adopted_existing_label"] = True
            return summary
        # write-gate-exempt(issue=1976): fleet-level sweep; no write_gate param, same shape as capacity_starvation_escalation.
        result = transition(
            gh,
            labels,
            entry.removal_issue,
            "config_retirement_ready",
            state_path=fleet_state_path,
            repo="fleet",
        )
        if result.outcome.name == "PARTIAL_FAILURE":
            return summary  # retry next pass rather than recording a false arm
        _post_comment(
            gh,
            entry.removal_issue,
            _arm_comment(entry, slots, rec["absent_since"], quiet_days),
        )
        rec["armed_at"] = _iso(now)
        summary["armed"] = True
        summary["armed_now"] = True
        # write-gate-exempt(issue=1976): same out-of-wave fleet sweep shape as the regression log_event above.
        log_event(
            fleet_state_path,
            "config_key_retirement_armed",
            {
                "section": entry.section,
                "key": entry.key,
                "issue_number": entry.removal_issue,
                "quiet_since": rec["absent_since"],
                "quiet_days": quiet_days,
            },
        )
    elif not dry_run:
        # Armed and still absent: clear the per-episode regression marker so
        # a later reappearance reports as a fresh episode.
        rec["regression_reported_at"] = None
    return summary


def run_config_retirement_sweep(
    *,
    fleet_dir_override: str | None = None,
    config: OrchestratorConfig | None = None,
    github: Callable[..., Any] | None = None,
    gh: Any = None,
    registry: tuple[DeprecatedConfigKey, ...] = DEPRECATED_CONFIG_KEYS,
    quiet_days: float | None = None,
    labels: Any = None,
    dry_run: bool = False,
    now: datetime.datetime | None = None,
    run_git: Callable[..., Any] = run_captured,
) -> dict[str, Any]:
    """Run the deprecated-key retirement sweep once for this fleet pass.

    Reads every registered repo's config layers (user-global plus the repo's
    own file, tracked or untracked) for each key in ``registry``. ``gh`` /
    ``github`` identify the orchestrator's own repo, where ``removal_issue``
    lives: pass ``gh`` directly (tests), or ``github`` as the real-client
    constructor for ``github_client_for`` (the same injection seam the lane
    path uses). ``labels``/``quiet_days`` default from ``config`` -- the
    fleet pass's own layered config -- so a host can retune the window via
    ``runtime.config_retirement_quiet_days``.

    Returns a summary dict with per-key findings plus an ``attention`` list
    of digest entries for anything operator-visible (armed / regressed /
    errors). Never raises: a failing sweep must not break the fleet pass.
    """
    resolved_now = now if now is not None else datetime.datetime.now(datetime.UTC)
    effective = config if config is not None else OrchestratorConfig()
    if quiet_days is None:
        quiet_days = float(effective.runtime.config_retirement_quiet_days)
    if labels is None:
        labels = effective.labels
    fleet_root = fleet_dir(override=fleet_dir_override)
    fleet_state_path = layout.state_file_path(fleet_root)
    sidecar = layout.config_retirement_state_path(override=fleet_dir_override)

    summary: dict[str, Any] = {"keys": {}, "attention": []}
    try:
        if gh is None:
            from .github import GitHub
            from .local_issues import github_client_for
            from .supervise import orchestrator_root

            gh = github_client_for(
                orchestrator_root(),
                effective,
                github=github or GitHub,
                dry_run=dry_run,
            )
        registry_json = _load_registry(layout.fleet_registry_path(override=fleet_dir_override))
        # All registered repos, not the pass's --repos subset: a filtered pass
        # must still see a repo where the key remains set.
        selected = _select_repos(registry_json, None)
        slots, blocked = _enumerate_layer_slots(
            selected, fleet_dir_override=fleet_dir_override, run_git=run_git
        )

        presence: dict[str, list[dict[str, str]]] = {}
        unreadable: dict[str, list[dict[str, str]]] = {}
        checked: list[dict[str, str]] = []
        # Repos whose layers could not even be enumerated (missing
        # ``repo_root`` / unreachable root) block every registered key from
        # arming -- absence is unproven there (fail closed).
        for record in blocked:
            for entry in registry:
                unreadable.setdefault(entry.dotted, []).append(dict(record))
        for slot in slots:
            found, error = _scan_layer(slot)
            checked.append(
                {"repo": slot.repo or "fleet", "layer": slot.layer, "path": str(slot.path)}
            )
            if error is not None:
                rec = {
                    "repo": slot.repo or "fleet",
                    "layer": slot.layer,
                    "path": str(slot.path),
                    "error": error,
                }
                for entry in registry:
                    unreadable.setdefault(entry.dotted, []).append(rec)
                continue
            for entry in found:
                presence.setdefault(entry.dotted, []).append(
                    {"repo": slot.repo or "fleet", "layer": slot.layer, "path": str(slot.path)}
                )

        keys_state = _load_retirement_state(sidecar)
        for entry in registry:
            summary["keys"][entry.dotted] = _process_key(
                entry,
                slots=slots,
                presence=presence,
                unreadable=unreadable,
                keys_state=keys_state,
                quiet_days=quiet_days,
                labels=labels,
                gh=gh,
                fleet_state_path=fleet_state_path,
                dry_run=dry_run,
                now=resolved_now,
            )
            per_key = summary["keys"][entry.dotted]
            if per_key.get("armed_now"):
                summary["attention"].append(
                    {
                        "repo_key": "fleet",
                        "type": "config_retirement_armed",
                        "key": entry.dotted,
                        "issue_number": entry.removal_issue,
                        "reason": f"{entry.dotted} absent everywhere for {quiet_days}d; removal issue marked Ready",
                    }
                )
            elif per_key.get("regressed"):
                summary["attention"].append(
                    {
                        "repo_key": "fleet",
                        "type": "config_retirement_regressed",
                        "key": entry.dotted,
                        "issue_number": entry.removal_issue,
                        "reason": f"{entry.dotted} reappeared after its removal issue was marked Ready",
                    }
                )
        summary["checked"] = checked
        if not dry_run:
            # Per-pass presence record: where each registered key is still
            # set (or that it is absent everywhere), which layers were
            # checked, and the armed flag. This is the sweep's "report where
            # the key is still set" channel -- ``config_key_deprecated_read``
            # only fires on config loads, while this reflects the sweep's own
            # file-level view of every registered repo.
            # write-gate-exempt(issue=1976): same out-of-wave fleet sweep shape as the armed/regressed log_event calls above.
            log_event(
                fleet_state_path,
                "config_retirement_sweep",
                {
                    "keys": {
                        dotted: {
                            "present_in": per_key.get("present_in") or [],
                            "unreadable_layers": per_key.get("unreadable_layers") or [],
                            "armed": bool(per_key.get("armed")),
                            "absent_since": keys_state.get(dotted, {}).get("absent_since"),
                        }
                        for dotted, per_key in summary["keys"].items()
                    },
                    "checked": checked,
                    "quiet_days": quiet_days,
                },
            )
            _save_retirement_state(sidecar, keys_state)
    except Exception:  # noqa: BLE001 — the sweep is observability+scheduling; the pass must survive it
        logger.exception("config-retirement sweep failed")
        summary["error"] = "sweep raised; see log"
        summary["attention"].append(
            {
                "repo_key": "fleet",
                "type": "config_retirement_error",
                "reason": "config-retirement sweep failed; see fleet log",
            }
        )
    return summary
