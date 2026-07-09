from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

REDACTED = "<redacted>"

_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "headers.authorization",
    "prompt",
    "messages",
    "request_body",
    "body",
    "content",
    "password",
    "token",
    "secret",
}

_SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{16,}\b", re.IGNORECASE),
]


def scrub(value: Any, *, _path: tuple[str, ...] = ()) -> Any:
    """Return a JSON-safe copy with secret-bearing fields and strings redacted."""

    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            key_s = str(key)
            path = (*_path, key_s.lower())
            if _is_sensitive_key(path):
                out[key_s] = REDACTED
            else:
                out[key_s] = scrub(item, _path=path)
        return out
    if isinstance(value, str):
        return _scrub_string(value)
    if isinstance(value, bytes):
        return REDACTED
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [scrub(item, _path=_path) for item in value]
    if value is None or isinstance(value, bool | int | float):
        return value
    return str(value)


def _is_sensitive_key(path: tuple[str, ...]) -> bool:
    leaf = path[-1]
    dotted = ".".join(path[-2:]) if len(path) >= 2 else leaf
    return leaf in _SENSITIVE_KEYS or dotted in _SENSITIVE_KEYS


def _scrub_string(value: str) -> str:
    out = value
    for pattern in _SECRET_PATTERNS:
        out = pattern.sub(REDACTED, out)
    return out
