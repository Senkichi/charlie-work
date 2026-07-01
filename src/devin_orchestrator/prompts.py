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
    template_path = resolve_template(template_name, search_dirs)
    template = Template(template_path.read_text(encoding="utf-8"))
    safe_values = {key: str(value) for key, value in values.items()}
    return template.safe_substitute(safe_values)
