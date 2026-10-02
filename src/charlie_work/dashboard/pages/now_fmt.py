"""Shared formatting helpers for the Now page renderers.

Every helper that returns markup escapes its dynamic inputs; callers pass raw values.
Drill-down URLs are built here (single point) so the routes can land later without the
renderers changing: ``/repo/<key>``, ``/issue/<repo>/<n>``, ``/pr/<repo>/<n>``,
``/pass/<repo>/<correlation id>``, ``/flow/<stage>``. Whether a
built URL becomes an ``<a>`` is decided by ``routes.routed`` (the route registry), never here.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from html import escape
from urllib.parse import quote, urlencode

from .routes import routed


def esc(value: object) -> str:
    """HTML-escape text or an attribute value (quotes included)."""
    return escape(str(value), quote=True)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "x"


def short_repo(key: str) -> str:
    """``owner/name`` -> ``name`` for display; the full key stays in the link and title."""
    return key.rsplit("/", 1)[-1] or key


# ``owner/name`` (GitHub) or ``local/<name>`` (no-remote lane): exactly two segments of
# slug characters, neither of them a dot segment. Anything else (``..``, extra ``/``,
# ``fleet``, hostile text) is not a repo and gets no link.
_SLUG = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")


def valid_slug(key: str) -> bool:
    return _SLUG.fullmatch(key) is not None and all(p.strip(".") for p in key.split("/"))


def repo_url(key: str, **query: str) -> str | None:
    """Drill-down href for a validated repo slug, else ``None`` (render without a link)."""
    if not valid_slug(key):
        return None
    base = "/repo/" + quote(key, safe="/")
    return f"{base}?{urlencode(query)}" if query else base


def issue_url(repo: str, number: int) -> str | None:
    if not valid_slug(repo):
        return None
    return f"/issue/{quote(repo, safe='/')}/{int(number)}"


def pr_url(repo: str, number: int) -> str | None:
    if not valid_slug(repo):
        return None
    return f"/pr/{quote(repo, safe='/')}/{int(number)}"


# ``correlation_context`` mints ``uuid4().hex[:12]``; callers may pass their own id, so accept
# any short token of id-safe characters (never a path, quote or whitespace).
_CORRELATION_ID = re.compile(r"[A-Za-z0-9_.:-]{1,64}")


def valid_correlation_id(value: str) -> bool:
    return isinstance(value, str) and _CORRELATION_ID.fullmatch(value) is not None


def pass_url(repo: str, correlation_id: str) -> str | None:
    if not valid_slug(repo) or not valid_correlation_id(correlation_id):
        return None
    return f"/pass/{quote(repo, safe='/')}/{quote(correlation_id, safe='')}"


def flow_url(stage: str) -> str:
    return f"/flow/{slug(stage)}"


def link(href: str | None, text: object, cls: str = "n", title: str | None = None) -> str:
    """An anchor when ``href`` names a registered route, else the same text in a span."""
    tip = f' title="{esc(title)}"' if title else ""
    href = routed(href)
    if href is None:
        return f'<span class="{esc(cls)}"{tip}>{esc(text)}</span>'
    return f'<a class="{esc(cls)}" href="{esc(href)}"{tip}>{esc(text)}</a>'


def cap_text(cap: int | None, *, compact: bool = False) -> str:
    """A concurrency cap: 0 means no cap (the config convention), None means unknown."""
    if cap is None:
        return "?" if compact else "cap ?"
    if cap == 0:
        return "∞" if compact else "no cap"
    return esc(cap)


def local_time(moment: datetime) -> str:
    """``HH:MM:SS`` in the server host's local zone inside a re-localisable ``<time>``."""
    return (
        f'<time class="js-local" datetime="{esc(moment.isoformat())}">'
        f"{esc(moment.astimezone().strftime('%H:%M:%S'))}</time>"
    )


def age(seconds: float | None) -> str:
    """Compact age: ``24s``, ``1m53s``, ``2h05m``, ``3d4h``; ``-`` when unknown."""
    if seconds is None:
        return "—"
    s = max(0, int(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    if s < 86400:
        return f"{s // 3600}h{(s % 3600) // 60:02d}m"
    return f"{s // 86400}d{(s % 86400) // 3600}h"


def stable_id(prefix: str, *parts: object) -> str:
    """A deterministic DOM id so htmx can restore focus to the same element after a swap."""
    digest = hashlib.sha1("\x1f".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return f"{prefix}-{digest[:10]}"


def fmt_float(value: float) -> str:
    return f"{value:.1f}".rstrip("0").rstrip(".")
