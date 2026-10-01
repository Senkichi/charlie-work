"""Static assets (base.css, fonts, htmx) and the combined stylesheet.

Everything resolves through ``importlib.resources`` so it works from a wheel as well as a
checkout. The server mounts :func:`static_dir_traversable` contents at ``/static/``.
"""

from __future__ import annotations

import functools
from importlib import resources
from importlib.resources.abc import Traversable
from types import MappingProxyType
from typing import Mapping

from charlie_work.dashboard.theme.theme import generate_css

_STATIC = "static"


def static_dir_traversable() -> Traversable:
    """Root of the packaged static directory."""
    return resources.files(__package__).joinpath(_STATIC)


def _walk(node: Traversable, prefix: str) -> dict[str, Traversable]:
    found: dict[str, Traversable] = {}
    for child in node.iterdir():
        rel = f"{prefix}{child.name}"
        if child.is_dir():
            found.update(_walk(child, rel + "/"))
        elif child.is_file():
            found[rel] = child
    return found


@functools.cache
def _asset_index() -> Mapping[str, Traversable]:
    """Every packaged static file keyed by its relative POSIX path (built once)."""
    return MappingProxyType(_walk(static_dir_traversable(), ""))


def static_asset(relpath: str) -> Traversable:
    """Resolve a packaged asset by its ``/static/``-relative path.

    Closed by construction: the caller's string is never joined onto a filesystem path. It
    is looked up verbatim in the set of packaged files, so drive-qualified (``C:..``), UNC,
    backslash, percent-encoded, NUL, absolute and ``..`` inputs are simply non-members.

    Raises ``ValueError`` on a string that could not be a packaged path (empty, backslash,
    colon, ``%``, NUL, leading slash, or a ``.``/``..``/empty segment) and
    ``FileNotFoundError`` for a well-formed path that is not a packaged file, so a request
    handler can map both to 404.
    """
    if (
        not relpath
        or any(c in relpath for c in ("\\", ":", "%", "\x00"))
        or relpath.startswith("/")
        or any(p in ("", ".", "..") for p in relpath.split("/"))
    ):
        raise ValueError(f"invalid static path: {relpath!r}")
    try:
        return _asset_index()[relpath]
    except KeyError:
        raise FileNotFoundError(relpath) from None


def base_css() -> str:
    """The hand-written base stylesheet."""
    return static_asset("base.css").read_text(encoding="utf-8")


def stylesheet() -> str:
    """Generated token custom properties followed by base.css (the cascade order matters)."""
    return generate_css() + "\n" + base_css()
