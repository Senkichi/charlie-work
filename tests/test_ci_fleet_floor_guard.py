"""CI must run against the ci-fleet floor, and every use-site config rule must name a real field.

``RunnerAllocationConfig`` is ci-fleet's class; ``config.py`` attaches ``FieldRules`` to
its fields at the use site. ``validate_section`` only checks those rule keys lazily, when
a config actually contains the section, and then raises ``TypeError`` -- so a rule for a
field the installed ci-fleet lacks surfaces at runtime, on the host, not at import.

Two guards close that gap together:

* ``test_ci_fleet_floor_equals_locked_version``: the ``ci-fleet>=X`` floor in
  pyproject.toml equals the version in uv.lock, which CI installs (``uv sync --frozen``).
  So the suite always runs against the oldest ci-fleet the pin admits; a floor left
  behind a lock bump fails here instead of admitting a version the code was never
  tested on.
* ``test_every_use_site_rule_names_a_field_of_its_section``: walks the whole config
  tree eagerly and fails on any ``FieldRules``/``Entries`` key that is not a field of
  the class it targets -- whether or not any test config contains that section.
"""

from __future__ import annotations

import importlib.metadata
import re
import tomllib
from pathlib import Path

from charlie_work.config import OrchestratorConfig
from charlie_work.config_validation import Entries, FieldRules, field_specs

ROOT = Path(__file__).resolve().parent.parent


def _floor() -> str:
    deps = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "dependencies"
    ]
    pins = [m for d in deps if (m := re.fullmatch(r"ci-fleet\s*>=\s*([0-9][^,\s]*)", d.strip()))]
    assert len(pins) == 1, f"expected exactly one 'ci-fleet>=X' dependency, got {deps}"
    return pins[0].group(1)


def _locked() -> str:
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    versions = [p["version"] for p in lock["package"] if p["name"] == "ci-fleet"]
    assert len(versions) == 1, f"expected one locked ci-fleet, got {versions}"
    return versions[0]


def test_ci_fleet_floor_equals_locked_version() -> None:
    floor, locked = _floor(), _locked()
    assert floor == locked, (
        f"pyproject pins ci-fleet>={floor} but uv.lock resolves {locked}: CI tests only "
        f"{locked}, so raise the floor to {locked} (or lock the floor) to keep them equal"
    )


def test_suite_runs_on_the_locked_ci_fleet() -> None:
    # Without this, a venv that drifted from the lock would make the floor check above
    # a statement about a file rather than about what the suite just exercised.
    assert importlib.metadata.version("ci-fleet") == _locked()


def _rule_sites(cls: type, path: str, seen: set[type]) -> list[tuple[str, type, set[str]]]:
    """Every (dotted path, target class, rule keys) for use-site rules under ``cls``."""
    if cls in seen:
        return []
    seen = seen | {cls}
    sites: list[tuple[str, type, set[str]]] = []
    for spec in field_specs(cls):
        here = f"{path}.{spec.name}" if path else spec.name
        if spec.item is None:
            continue
        keys = {
            k for m in spec.markers if isinstance(m, (FieldRules, Entries)) for k, _ in m.items
        }
        if keys:
            sites.append((here, spec.item, keys))
        sites.extend(_rule_sites(spec.item, here, seen))
    return sites


def test_every_use_site_rule_names_a_field_of_its_section() -> None:
    sites = _rule_sites(OrchestratorConfig, "", set())
    # Positive control: the walk must reach the ci-fleet-owned section whose rules
    # motivated this test, or an empty "no bad rules" result proves nothing.
    assert any(path == "runner_allocation" for path, _, _ in sites), sites
    bad = {
        path: sorted(keys - {s.name for s in field_specs(target)})
        for path, target, keys in sites
        if keys - {s.name for s in field_specs(target)}
    }
    assert not bad, f"use-site rules for fields the section class lacks: {bad}"
