from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from devin_orchestrator.prompts import TEMPLATE_DIR

WORKER_SECTIONS_DIRNAME = "worker_sections"

PACKAGE_WORKER_SECTIONS_DIR = TEMPLATE_DIR / WORKER_SECTIONS_DIRNAME


def _worker_sections_dirs(search_dirs: Sequence[Path]) -> tuple[Path, ...]:
    """Repo-local `worker_sections/` dirs first, then the package default."""
    return tuple(Path(directory) / WORKER_SECTIONS_DIRNAME for directory in search_dirs) + (
        PACKAGE_WORKER_SECTIONS_DIR,
    )


def section_variables(search_dirs: tuple[Path, ...] = ()) -> dict[str, str]:
    """Discover shared worker prompt partials as `$section_<stem>` template values.

    Every `*.md` file under a `worker_sections/` directory becomes a
    `section_<stem>` key holding that file's text. Repo-local `<search_dir>/
    worker_sections/` directories win over the package's own `prompts/
    worker_sections/`, first-hit-wins by filename — mirroring
    `prompts.resolve_template`. No section names are hardcoded: the available
    set is whatever `*.md` files exist on disk.
    """
    sections: dict[str, str] = {}
    seen_stems: set[str] = set()
    for directory in _worker_sections_dirs(search_dirs):
        if not directory.is_dir():
            continue
        for candidate in sorted(directory.glob("*.md")):
            stem = candidate.stem
            if stem in seen_stems:
                continue
            seen_stems.add(stem)
            sections[f"section_{stem}"] = candidate.read_text(encoding="utf-8")
    return sections
