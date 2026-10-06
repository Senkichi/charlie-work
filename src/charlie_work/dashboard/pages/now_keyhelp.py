"""The keyboard-shortcut help panel (toggled by ``?`` in dashboard.js).

Only shortcuts that lead somewhere are listed: the ``g <key>`` views come from the route
registry (dashboard.js navigates through the header nav's ``data-go`` links, so an
unbuilt view has neither a link nor a key), and ``o`` is listed only once the issue
drill-down route exists.
"""

from __future__ import annotations

from .routes import is_routed, live_views


def _keys(page: str) -> tuple[tuple[str, str], ...]:
    views = live_views()
    go = (" / ".join(f"g {v.key}" for v in views), "go to " + " / ".join(v.name for v in views))
    if page != "now":  # j/k, Enter, c and o act on the Needs-you list, which only Now has
        return (
            go,
            ("Tab", "move through tabs, range and links"),
            ("?", "show or hide this panel"),
        )
    keys = [
        go,
        ("j / k", "focus next / previous row in the open list"),
        ("Enter", "expand or collapse the focused row"),
        ("c", "copy the focused row's command"),
    ]
    if is_routed("/issue/o/r/1"):
        keys.append(("o", "open the focused row's drill-down"))
    keys += [
        ("← / →", "pick a bar while the chart is focused (Esc clears)"),
        ("?", "show or hide this panel"),
    ]
    return tuple(keys)


def key_help(page: str = "now") -> str:
    return (
        '<aside id="keyhelp" class="keyhelp" role="dialog" aria-label="Keyboard shortcuts" '
        "hidden><h2>Keys</h2><dl>"
        + "".join(f"<dt><kbd>{k}</kbd></dt><dd>{d}</dd>" for k, d in _keys(page))
        + "</dl></aside>"
    )
