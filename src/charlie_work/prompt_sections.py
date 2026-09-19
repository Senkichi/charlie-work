from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from charlie_work.prompts import TEMPLATE_DIR

WORKER_SECTIONS_DIRNAME = "worker_sections"

PACKAGE_WORKER_SECTIONS_DIR = TEMPLATE_DIR / WORKER_SECTIONS_DIRNAME


def _section_roots(search_dirs: Sequence[Path]) -> tuple[Path, ...]:
    """Repo-local `worker_sections/` dirs first, then the package default."""
    return tuple(Path(directory) / WORKER_SECTIONS_DIRNAME for directory in search_dirs) + (
        PACKAGE_WORKER_SECTIONS_DIR,
    )


def section_variant_names(search_dirs: Sequence[Path] = ()) -> tuple[str, ...]:
    """Variant names on disk: every subdirectory of a `worker_sections/` dir.

    A variant is an overlay: `worker_sections/<variant>/<stem>.md` replaces
    `worker_sections/<stem>.md` when that variant is active. Nothing declares the
    set -- it is whatever subdirectories exist, so a new variant is a new directory.
    """
    return tuple(
        sorted(
            {
                child.name
                for root in _section_roots(search_dirs)
                if root.is_dir()
                for child in root.iterdir()
                if child.is_dir()
            }
        )
    )


def section_variables(
    search_dirs: tuple[Path, ...] = (), variants: Sequence[str] = ()
) -> dict[str, str]:
    """Discover shared worker prompt partials as `$section_<stem>` template values.

    Every `*.md` file under a `worker_sections/` directory becomes a
    `section_<stem>` key holding that file's text. Repo-local `<search_dir>/
    worker_sections/` directories win over the package's own `prompts/
    worker_sections/`, first-hit-wins by filename — mirroring
    `prompts.resolve_template`. No section names are hardcoded: the available
    set is whatever `*.md` files exist on disk.

    `variants` names the active overlays. Within each root, `<root>/<variant>/`
    is searched before `<root>` itself, so an active variant's partial replaces the
    base one of the same stem, while a repo-local base partial still outranks the
    package's variant (the more specific override wins). A stem that exists only
    inside some variant directory renders as an empty string while that variant is
    inactive: an overlay-only partial is by definition text that should appear only
    when the variant applies.
    """
    sections: dict[str, str] = {}
    seen_stems: set[str] = set()
    for root in _section_roots(search_dirs):
        for directory in (*(root / variant for variant in variants), root):
            if not directory.is_dir():
                continue
            for candidate in sorted(directory.glob("*.md")):
                stem = candidate.stem
                if stem in seen_stems:
                    continue
                seen_stems.add(stem)
                sections[f"section_{stem}"] = candidate.read_text(encoding="utf-8")
    for variant in section_variant_names(search_dirs):
        for root in _section_roots(search_dirs):
            for candidate in sorted((root / variant).glob("*.md")):
                sections.setdefault(f"section_{candidate.stem}", "")
    return sections
