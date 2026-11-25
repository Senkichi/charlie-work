"""The foreman charter: frozen dataclasses validated by ADR-0007 field metadata.

``charter.yaml`` is the contract a human writes for one foreman (design §5).
It is *not* part of the layered ``config.yaml`` tree — it is a standalone
document loaded with the same ``config_validation`` machinery the config tree
uses, so the check vocabulary (``Typed``/``NotNull``/``OneOf``/``Entries``/
``Positive``) and error shape (``FieldError`` naming the key path) stay
identical.

Validation happens in two layers:

* **Field checks** — ``validate_section`` against the dataclass annotations:
  type, nullability, known keys, ``arming`` membership, positive bounds.
* **Semantic checks** — the cross-field rules only ``load_charter`` can see:
  the name must match the directory, ``repos`` must name exactly one fleet
  registry entry, ``done_when.check`` must be a non-empty argv, and
  ``cadence.report_at`` must be a local ``HH:MM``.

Every failure returns a ``Result.error`` naming the field — never an
exception that would stop other foremen from loading.
"""

from __future__ import annotations

import re
from collections.abc import Collection
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import yaml

from ..config_validation import (
    ConfigError,
    NonEmpty,
    NotNull,
    NullIsDefault,
    OneOf,
    Positive,
    Typed,
    validate_section,
)
from . import Result
from .paths import FOREMAN_NAME_RE, charter_path, is_foreman_name

ARMING_AUTO = "auto"
ARMING_PROPOSE = "propose"
ARMING_MODES = (ARMING_AUTO, ARMING_PROPOSE)

_REPORT_AT_RE = re.compile(r"([01]\d|2[0-3]):[0-5]\d")


@dataclass(frozen=True)
class DoneWhen:
    """The completion check: an argv the foreman runs to decide it is done."""

    check: Annotated[tuple[str, ...], Typed, NotNull] = ()


@dataclass(frozen=True)
class Cadence:
    """The report cadence: a local ``HH:MM`` at which the foreman reports."""

    report_at: Annotated[str | None, Typed] = None


@dataclass(frozen=True)
class CharterBounds:
    """Per-turn and per-day caps the charter grants its foreman (design §10)."""

    max_open_issues: Annotated[int, Typed, Positive] = 5
    max_issues_per_turn: Annotated[int, Typed, Positive] = 3
    min_turn_interval_minutes: Annotated[int, Typed, Positive] = 30
    max_quiet_hours: Annotated[int, Typed, Positive] = 24
    max_turns_per_day: Annotated[int, Typed, Positive] = 12
    max_unanswered_days: Annotated[int, Typed, Positive] = 7


@dataclass(frozen=True)
class Charter:
    """One foreman's contract, exactly as written in ``charter.yaml``.

    ``repos`` carries exactly one fleet registry key — enforced by the
    semantic check in ``load_charter`` (``Entries`` cannot annotate a
    ``seq_str`` field, so both the empty and over-long cases land there).
    ``done_when`` and ``cadence`` stay
    ``None`` when absent so a ``null`` in the file is never conflated with an
    omitted section (P2 null-preservation).
    """

    name: Annotated[str, Typed, NonEmpty]
    outcome: Annotated[str, Typed, NonEmpty]
    repos: Annotated[tuple[str, ...], Typed, NotNull]
    standing: Annotated[bool, Typed] = False
    done_when: Annotated[DoneWhen | None, NullIsDefault] = None
    arming: Annotated[str, Typed, NotNull, OneOf(*ARMING_MODES)] = ARMING_AUTO
    bounds: Annotated[CharterBounds, NullIsDefault] = field(default_factory=CharterBounds)
    cadence: Annotated[Cadence, NullIsDefault] = field(default_factory=Cadence)


def _semantic_checks(
    charter: Charter, foreman_root: Path, registered_repos: Collection[str]
) -> str | None:
    """Cross-field rules field metadata cannot express. Returns the error or None."""
    if not is_foreman_name(charter.name):
        return f"charter.name {charter.name!r}: must match {FOREMAN_NAME_RE.pattern!r}"
    if charter.name != foreman_root.name:
        return (
            f"charter.name {charter.name!r}: must equal the foreman directory "
            f"name {foreman_root.name!r}"
        )
    if len(charter.repos) != 1:
        return f"charter.repos: exactly one fleet registry key, got {len(charter.repos)}"
    if charter.repos[0] not in registered_repos:
        return (
            f"charter.repos[0] {charter.repos[0]!r}: not a registered repo in the fleet registry"
        )
    if charter.done_when is not None and not charter.done_when.check:
        return "charter.done_when.check: a non-empty argv when done_when is present"
    if charter.cadence is not None and charter.cadence.report_at is not None:
        if _REPORT_AT_RE.fullmatch(charter.cadence.report_at) is None:
            return (
                f"charter.cadence.report_at {charter.cadence.report_at!r}: "
                "local HH:MM (00:00-23:59)"
            )
    return None


def load_charter(foreman_root: Path, *, registered_repos: Collection[str]) -> Result[Charter]:
    """Load and validate ``charter.yaml`` under ``foreman_root``.

    ``registered_repos`` is the fleet registry's repo keys — the charter's
    single ``repos`` entry must be one of them. Every failure mode (missing
    file, unreadable YAML, field error, semantic error, escaped path) is a
    ``Result.error`` naming the field or file.
    """
    try:
        path = charter_path(foreman_root)
    except ValueError as exc:
        return Result(error=str(exc))
    if not path.is_file():
        return Result(error=f"charter.yaml not found at {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        return Result(error=f"charter.yaml: invalid YAML: {exc}")
    except OSError as exc:
        return Result(error=f"charter.yaml: unreadable: {exc}")
    if raw is None:
        return Result(error="charter.yaml: empty file")
    try:
        charter = validate_section(Charter, raw, path="charter")
    except ConfigError as exc:
        return Result(error=str(exc))
    problem = _semantic_checks(charter, foreman_root, registered_repos)
    if problem is not None:
        return Result(error=problem)
    return Result(value=charter)
