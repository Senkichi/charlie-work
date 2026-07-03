from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from string import Template

TEMPLATE_DIR = Path(__file__).with_name("prompts")


def resolve_template(template_name: str, search_dirs: Sequence[Path] = ()) -> Path:
    """Repo-local template dirs win over the package defaults, per filename."""
    for directory in search_dirs:
        candidate = Path(directory) / template_name
        if candidate.is_file():
            return candidate
    return TEMPLATE_DIR / template_name


def render_prompt(
    template_name: str,
    values: Mapping[str, object],
    *,
    search_dirs: Sequence[Path] = (),
) -> str:
    from .prompt_sections import section_variables

    template_path = resolve_template(template_name, search_dirs)
    template = Template(template_path.read_text(encoding="utf-8"))
    # Explicit values win over section text on any future key collision.
    merged = {**section_variables(search_dirs=tuple(search_dirs)), **values}
    safe_values = {key: str(value) for key, value in merged.items()}
    # Render section partials first: each partial's internal $placeholders are
    # resolved against safe_values, producing fully-resolved section strings.
    # This prevents attacker-controlled values (issue_body, etc.) from being
    # re-scanned in a second pass, which would allow prompt injection.
    resolved_sections = {
        key: Template(section_text).safe_substitute(safe_values)
        for key, section_text in section_variables(search_dirs=tuple(search_dirs)).items()
    }
    # Merge resolved sections with explicit values (explicit values still win).
    final_values = {**resolved_sections, **values}
    # Convert all values to strings for the final substitution
    final_safe_values = {key: str(value) for key, value in final_values.items()}
    # Single substitution over the template: attacker-supplied values are leaf
    # replacements that are never re-scanned.
    return template.safe_substitute(final_safe_values)
