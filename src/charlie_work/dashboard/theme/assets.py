"""Static assets (base.css, fonts, htmx) and the combined stylesheet.

Everything resolves through ``importlib.resources`` so it works from a wheel as well as a
checkout. The server mounts :func:`static_dir_traversable` contents at ``/static/``.
"""

from __future__ import annotations

from importlib import resources
from importlib.resources.abc import Traversable

from charlie_work.dashboard.theme.theme import generate_css

_STATIC = "static"


def static_dir_traversable() -> Traversable:
    """Root of the packaged static directory."""
    return resources.files(__package__).joinpath(_STATIC)


def static_asset(relpath: str) -> Traversable:
    """Resolve a packaged asset by its ``/static/``-relative path.

    Raises ``ValueError`` on an absolute or ``..`` path and ``FileNotFoundError`` when the
    asset does not exist, so a request handler can map both to 404 without path traversal.
    """
    parts = relpath.replace("\\", "/").split("/")
    if relpath.startswith("/") or any(p in ("", ".", "..") for p in parts):
        raise ValueError(f"invalid static path: {relpath!r}")
    node = static_dir_traversable().joinpath(*parts)
    if not node.is_file():
        raise FileNotFoundError(relpath)
    return node


def base_css() -> str:
    """The hand-written base stylesheet."""
    return static_asset("base.css").read_text(encoding="utf-8")


def stylesheet() -> str:
    """Generated token custom properties followed by base.css (the cascade order matters)."""
    return generate_css() + "\n" + base_css()
