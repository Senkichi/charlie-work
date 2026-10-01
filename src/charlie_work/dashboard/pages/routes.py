"""The one registry of page routes the dashboard server actually serves.

Every renderer that would emit an ``<a href>`` asks :func:`routed` first: a number whose
drill-down route is not registered renders as plain text, never as a dotted-underlined
link to a 404. The drill-downs (``/repo``, ``/issue``, ``/pr``, ``/pass``, ``/flow``) are
registered here and handled by ``server_drill.HANDLERS``; ``/prs``, ``/backlog`` and
``/capacity`` are not built yet, so their numbers on Now stay plain text.
``tests/test_dashboard_server.py`` and ``tests/test_dashboard_drill_server.py`` hold the
registry to the server in both directions (every registered prefix has a handler; a
known-but-unregistered view is a 404).

URL *construction* stays in ``now_fmt`` (slug validation); this module only decides
whether a built URL is live.
"""

from __future__ import annotations

from dataclasses import dataclass

# Path prefixes with a real handler. A prefix matches itself and anything below it
# (``/now`` covers ``/now/fragment``); query strings and fragments are ignored.
ROUTES: frozenset[str] = frozenset(
    {"/now", "/history", "/repo", "/issue", "/pr", "/pass", "/flow"}
)


@dataclass(frozen=True)
class View:
    """A top-level view in the header nav, reachable by ``g <key>``."""

    key: str
    name: str
    href: str


VIEWS: tuple[View, ...] = (View("n", "Now", "/now"), View("h", "History", "/history"))


def is_routed(href: str) -> bool:
    """True for an in-page fragment or a path under a registered route."""
    if href.startswith("#"):
        return True
    path = href.split("#", 1)[0].split("?", 1)[0]
    return any(path == r or path.startswith(r + "/") for r in ROUTES)


def routed(href: str | None) -> str | None:
    """``href`` when its route exists, else ``None`` (the caller renders plain text)."""
    return href if href is not None and is_routed(href) else None


def live_views() -> tuple[View, ...]:
    return tuple(v for v in VIEWS if is_routed(v.href))
