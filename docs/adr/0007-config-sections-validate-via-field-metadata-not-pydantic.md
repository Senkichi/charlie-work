# Config sections validate themselves through field metadata, not pydantic

Accepted 2026-09-29; implementation pending (architecture review candidate 6).

Each config section declares its own type and range rules as metadata on the fields of its existing frozen dataclass, for example `Annotated[int, NonNeg]`. One generic validator reads that metadata. Rules that span fields or sections, such as the `runner_scaling` floor check against `runner_allocation`, live in a per-section `validate()` hook. The loader only routes raw section dicts to their sections. It no longer holds hand-written checks: before this, `build_config_from_data` was about 1,800 lines with 160 `isinstance` checks and 214 `raise ConfigError` sites.

## Considered Options

- **pydantic models.** Rejected. Config and value objects must stay `@dataclass(frozen=True)` (`CLAUDE.md` invariant). ADR-0002 also records that `RunnerAllocationConfig` and `RunnerScalingConfig` are re-exported from `ci_fleet.config` and compared, and `isinstance`-checked, across that seam. Swapping the class machinery on either side breaks equality at runtime without breaking any import.
- **msgspec or attrs validators.** Rejected. They would add a dependency to replace a generic validator of a few dozen lines, and attrs would also replace the dataclass machinery.

## Consequences

- Adding a knob touches one section: its field, its metadata, and that section's tests.
- Host-wide-only sections (`runner_allocation`, `runner_capacity_escalation`) declare that scope themselves, instead of being hardcoded in `load_layered_config`.
- The dataclasses stay plain frozen dataclasses, so the `tests/test_ci_fleet_seams.py` guarantees continue to hold.
- Every rejection is now a `ConfigError`. Shapes that main rejected with a raw `TypeError`/`AttributeError` (non-string command templates or harness, a null on a non-optional tuple key, a non-string key inside a section, an unknown key inside a `rescue` role mapping) are therefore covered by the #665 layered-load rescue, which discards a bad global layer. Main crashed there by accident, not by design; `tests/test_config_review_fixes.py` pins one layered case per class. Host-wide construction errors (`ci_fleet`'s `__post_init__`, including `key: null` on the four fields it cannot compare) stay loud as `ConstructionError`. The one host-wide exception is a non-string key in a host-wide section, which is now a rescuable unknown-key error.
