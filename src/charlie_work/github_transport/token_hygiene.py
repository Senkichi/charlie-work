"""Keep the GitHub bearer token out of every value the transport emits (ADR-0006).

Two single points of enforcement, used at the two seams a token crosses:

* ``is_well_formed`` -- token *resolution* (``GuardedTransport``) and the HTTP
  adapter's own entry reject a token that cannot be a legal header value (any
  whitespace or control character, e.g. an internal CR/LF). Such a token makes
  ``http.client`` raise ``ValueError("Invalid header value b'Bearer <token>'")``,
  and that message used to travel into ``TransportFailure.detail`` and from
  there into ``events.db`` and the log.
* ``redact`` -- every failure detail is scrubbed at the adapter boundary (the
  adapter knows the token) and again where the guard writes an event (the
  guard knows the cached token), so a detail built from any exception text
  cannot carry the token or an ``Authorization`` value.
"""

from __future__ import annotations

import re

REDACTED = "<redacted>"

_WELL_FORMED = re.compile(r"[\x21-\x7e]+")
# An ``Authorization`` credential (``Bearer <x>``, or ``token <long x>``) and the
# shapes GitHub issues (classic, OAuth, app, fine-grained).
_AUTH_VALUE = re.compile(r"(?i)\b(?:bearer\s+\S+|token\s+[A-Za-z0-9_.\-=]{20,})")
# No boundary before a shape: a foreign token glued to a letter or underscore
# (``xghp_...``, ``GH_TOKEN_ghp_...``) is still a token.
_TOKEN_SHAPES = re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{16,}|github_pat_[A-Za-z0-9_]{16,})")


def is_well_formed(token: str) -> bool:
    """True when *token* is printable ASCII with no whitespace or control char."""
    return _WELL_FORMED.fullmatch(token) is not None


def _spellings(secret: str) -> list[str]:
    """The literal token plus the escaped forms an exception message would show."""
    forms = {secret, repr(secret)[1:-1], ascii(secret)[1:-1]}
    forms.add(repr(secret.encode("utf-8", errors="replace"))[2:-1])
    return sorted((f for f in forms if f), key=len, reverse=True)


def redact(text: str, *secrets: str | None) -> str:
    """*text* with every known secret and any bearer/Authorization value replaced."""
    for secret in secrets:
        if not secret:
            continue
        for form in _spellings(secret):
            text = text.replace(form, REDACTED)
    text = _AUTH_VALUE.sub(f"Bearer {REDACTED}", text)
    return _TOKEN_SHAPES.sub(REDACTED, text)
