"""PowerShell 5.1 quoting for the copy-paste commands the Now page hands the operator.

The operator pastes into PowerShell, not a POSIX shell, so ``shlex`` quoting is wrong
(``'it'"'"'s'`` splits into three arguments there). This is the single quoting seam:
``now_needs_me._cli`` builds every command through :func:`join_command`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

# Tokens PowerShell passes through verbatim. Deliberately excludes ``,`` (array
# operator), ``@`` (splatting), backslash (kept quoted so the path is visibly literal)
# and everything else with meaning to the parser.
_SAFE = re.compile(r"[A-Za-z0-9_./:+=-]+")

# PowerShell treats the typographic single quotes as quote characters too, so each one
# must be doubled exactly like an ASCII apostrophe.
_SINGLE_QUOTES = "'\u2018\u2019\u201a\u201b"


def ps_quote(arg: str) -> str:
    """One argument as a PowerShell token: bare when safe, else single-quoted."""
    if _SAFE.fullmatch(arg):
        return arg
    return "'" + "".join(c * 2 if c in _SINGLE_QUOTES else c for c in arg) + "'"


def join_command(args: Iterable[str]) -> str:
    return " ".join(ps_quote(a) for a in args)
