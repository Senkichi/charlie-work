"""The keyboard-shortcut help panel (static markup; toggled by ``?`` in dashboard.js)."""

from __future__ import annotations

_KEYS = (
    ("g n / g h", "go to Now / History"),
    ("/", "filter Needs-me rows (Esc clears)"),
    ("j / k", "select next / previous row"),
    ("Enter", "open the selected row's drill-down"),
    ("c", "copy the selected row's command"),
    ("?", "show or hide this panel"),
)

KEY_HELP = (
    '<aside id="keyhelp" class="keyhelp" role="dialog" aria-label="Keyboard shortcuts" hidden>'
    "<h2>Keys</h2><dl>"
    + "".join(f"<dt><kbd>{k}</kbd></dt><dd>{d}</dd>" for k, d in _KEYS)
    + "</dl></aside>"
)
