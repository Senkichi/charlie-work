"""Doctor checks for config pairs that drift silently (issue #1849).

Two repo-side files each restate a value the orchestrator config owns, and
neither restatement is verified anywhere today:

* ``docs/agents/triage-labels.md`` maps the mattpocock-skills plugin's
  canonical ``ready-for-agent`` triage role onto the repo's tracker label.
  If it names anything other than the label intake dispatches on
  (``LabelConfig.ready``), skill-filed "ready" tickets never dispatch --
  and nothing reports it.
* ``.aviator/config.yml`` lists the merge queue's required checks under
  ``merge_rules.preconditions.required_checks`` -- the same list
  ``auto_merge.required_checks`` owns for the merge gate. The existing
  required-check verification compares the config list against workflow job
  names only, never against Aviator's copy; a drift between the two lists
  across two sibling repos (2026-09-23) surfaced nowhere.

Both checks are opt-in by adoption: a repo without the file passes with a
"not adopted"/"absent" detail, so no consumer repo has to add anything for
its doctor run to stay green.

Lives outside ``doctor`` on purpose: ``doctor.py`` is over the 800-line
module cap and pinned by the file-size high-water-mark ratchet
(``tests/test_file_size_ratchet.py``), which never allows an over-cap file
to grow past its recorded mark -- new code lands in a domain module instead
(the same seam ``doctor_local_backend`` / ``doctor_cross_repo`` use).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .config import OrchestratorConfig

TRIAGE_LABEL_MAP_PATH = Path("docs") / "agents" / "triage-labels.md"
AVIATOR_CONFIG_PATH = Path(".aviator") / "config.yml"
TRIAGE_READY_ROLE = "ready-for-agent"


def _triage_map_ready_label(text: str) -> str | None:
    """Return the tracker label the map assigns to ``ready-for-agent``.

    Scans Markdown table rows (lines whose first non-space character is
    ``|``) for the row whose first cell is ``ready-for-agent``, backticks
    optional, and returns the second cell with backticks stripped. ``None``
    when no such row exists -- including a one-cell ``ready-for-agent`` row,
    which has no second cell to read a label from.
    """
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = stripped.split("|")
        # The outer pipes produce empty edge cells; drop exactly those so a
        # deliberately empty middle cell cannot shift columns.
        if cells and not cells[0].strip():
            cells = cells[1:]
        if cells and not cells[-1].strip():
            cells = cells[:-1]
        cells = [cell.strip().strip("`").strip() for cell in cells]
        if len(cells) >= 2 and cells[0] == TRIAGE_READY_ROLE:
            return cells[1]
    return None


def _check_triage_label_map(add: Any, repo_root: Path, config: OrchestratorConfig) -> None:
    """Verify the skills-plugin triage map names the configured ready label.

    Fleet repos adopting the mattpocock-skills plugin ship
    ``docs/agents/triage-labels.md`` so skill-filed issues know which label
    means "ready for an agent". The fleet dispatcher reads labels only, so
    that file's ``ready-for-agent`` row must name ``config.labels.ready``
    exactly; any other value -- or a missing row -- means a skill's "ready"
    decision lands on a label nothing dispatches on, with no signal.

    File absent: pass ("not adopted") -- the check is opt-in and no repo has
    to ship the map. Row missing or label different: error naming both
    values. Read-only; never raises on an unreadable file.
    """
    if not (repo_root / TRIAGE_LABEL_MAP_PATH).is_file():
        add(
            "triage-label map",
            True,
            f"{TRIAGE_LABEL_MAP_PATH} not adopted — nothing to verify",
            severity="warning",
        )
        return

    path = repo_root / TRIAGE_LABEL_MAP_PATH
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        add("triage-label map", False, f"could not read {path}: {exc}")
        return

    mapped = _triage_map_ready_label(text)
    expected = config.labels.ready
    if mapped is None:
        add(
            "triage-label map",
            False,
            f"{TRIAGE_LABEL_MAP_PATH} has no `{TRIAGE_READY_ROLE}` row — the "
            f"skills plugin's ready role maps to nothing; the row must name "
            f"the configured ready label `{expected}`",
        )
        return
    if mapped != expected:
        add(
            "triage-label map",
            False,
            f"`{TRIAGE_READY_ROLE}` maps to `{mapped}` in "
            f"{TRIAGE_LABEL_MAP_PATH} but the configured ready label is "
            f"`{expected}` — issues a skill marks ready would sit on a label "
            "intake never dispatches on",
        )
        return
    add(
        "triage-label map",
        True,
        f"`{TRIAGE_READY_ROLE}` maps to `{mapped}`, matching labels.ready",
    )


def _check_aviator_required_checks(add: Any, repo_root: Path, config: OrchestratorConfig) -> None:
    """Compare Aviator's required checks with ``auto_merge.required_checks``.

    The merge queue's precondition list lives in
    ``.aviator/config.yml`` under ``merge_rules.preconditions.required_checks``
    while the merge gate's list lives in ``auto_merge.required_checks``; they
    are the same fact stated twice. Doctor already validates the config list
    against workflow job names, but nothing compared the two lists to each
    other -- the 2026-09-23 sibling-repo drift was found by hand.

    File absent, or present without a ``required_checks`` list: pass. Any
    set difference: warning (not error -- Aviator and the merge gate fail
    independently and neither blocks the other on drift) naming the checks
    found on only one side. Read-only; an unparseable file is a warning, not
    a silent pass.
    """
    if not (repo_root / AVIATOR_CONFIG_PATH).is_file():
        add(
            "aviator required checks",
            True,
            f"{AVIATOR_CONFIG_PATH} absent — nothing to compare",
            severity="warning",
        )
        return

    path = repo_root / AVIATOR_CONFIG_PATH
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        add(
            "aviator required checks",
            False,
            f"could not parse {AVIATOR_CONFIG_PATH}: {exc}",
            severity="warning",
        )
        return

    rules = raw.get("merge_rules") if isinstance(raw, dict) else None
    preconditions = rules.get("preconditions") if isinstance(rules, dict) else None
    aviator_checks = (
        preconditions.get("required_checks") if isinstance(preconditions, dict) else None
    )
    if not isinstance(aviator_checks, list):
        add(
            "aviator required checks",
            True,
            f"{AVIATOR_CONFIG_PATH} declares no merge_rules.preconditions.required_checks list",
            severity="warning",
        )
        return

    aviator_set = {str(item) for item in aviator_checks}
    config_set = {str(item) for item in config.auto_merge.required_checks}
    if aviator_set == config_set:
        add(
            "aviator required checks",
            True,
            f"{len(aviator_set)} required check(s) match auto_merge.required_checks",
        )
        return

    parts: list[str] = []
    only_config = sorted(config_set - aviator_set)
    only_aviator = sorted(aviator_set - config_set)
    if only_config:
        parts.append(f"only in auto_merge.required_checks: {only_config}")
    if only_aviator:
        parts.append(f"only in {AVIATOR_CONFIG_PATH}: {only_aviator}")
    add(
        "aviator required checks",
        False,
        "required_checks differ — " + "; ".join(parts),
        severity="warning",
    )
