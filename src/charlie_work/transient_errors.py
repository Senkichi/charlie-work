"""Single definition of "transient network failure" (issue #1773).

``GitHub.run()`` (``github.py``) has always classified failed ``gh`` CLI
invocations against an allowlist of transient-network substrings before
deciding whether to retry. Raw ``git`` subprocess calls (``git fetch``,
``git pull --ff-only``, ``git ls-remote``) talk to the exact same GitHub
edge over the exact same network path and fail with the exact same class of
TLS/connection blip, but historically had zero retry -- see
:mod:`charlie_work.git_retry`, which wraps them.

Both codepaths reuse the classifier in this module rather than each keeping
its own allowlist: two independently-maintained lists of "what counts as
transient" are exactly the kind of duplicated invariant that drifts apart
one edit at a time (this repo's own ``ci_fleet/github.py`` module docstring
warns about the cost of having duplicated ``gh``'s retry/backoff/jitter
behaviour once already). ``github.py``'s ``_is_transient_gh_error`` is a
thin alias onto :func:`is_transient_network_error`; it does not carry its
own copy of the allowlist any more.

The allowlist is intentionally tool-agnostic, but "tool-agnostic" cannot mean
"``gh``'s wording with a couple of git-flavored patterns bolted on" -- ``gh``
(Go's ``net/http`` idiom -- "TLS handshake timeout", "connection reset",
"HTTP 502") and git's own libcurl/OpenSSL/schannel/Winsock backends describe
the same physical phenomena in genuinely different words, word for word, not
just different casing. A classifier ported verbatim from the ``gh`` list and
merely extended with one or two git-shaped strings is inert for git: on this
host (libcurl + schannel on Windows) a real unreachable-remote ``git fetch``
fails with

    fatal: unable to access 'https://...': Failed to connect to
    <host> port 443 after N ms: Couldn't connect to server

-- "Couldn't connect", not "could not connect"; "Failed to connect", not
"connection refused". Every pattern below that is not already shared with
``gh`` was pinned against either this host's own real ``git`` stderr or the
libcurl/schannel/OpenSSL documentation's own verbatim error text (see
``tests/test_transient_errors.py`` for the parametrized cases), specifically
so this module does not repeat the mistake of looking tool-agnostic while
only actually matching one tool. A pattern that never appears in one tool's
real output is simply inert there -- adding it costs nothing and does not
change that tool's classification of any error it can actually produce.
"""

from __future__ import annotations

import re

__all__ = ["is_transient_network_error"]


def is_transient_network_error(error: str) -> bool:
    """Classify a subprocess stderr/stdout string as transient (retryable).

    Transient signals are an allowlist; anything not explicitly listed is
    treated as terminal so genuine logic/auth/validation/merge-conflict
    errors still fail fast and are never retried by mistake. Terminal
    signals are checked first and win outright, so a message that happens to
    mention both (e.g. a 403 that is *not* a rate limit) is never misread as
    transient by accident.
    """
    text = error.lower()

    # Terminal signals that must never be retried.
    if "bad credentials" in text:
        return False
    if "could not resolve to a" in text or "not_found" in text:
        return False
    if re.search(r"\bhttp 401\b", text):
        return False
    # 403 is terminal unless it is a rate-limit/secondary-rate-limit response.
    if re.search(r"\bhttp 403\b", text) and not (
        "rate limit" in text
        or "secondary rate limit" in text
        or "was submitted too quickly" in text
    ):
        return False
    if re.search(r"\bhttp 422\b", text):
        return False

    # Transient allowlist.
    if "tls handshake timeout" in text:
        return True
    if "net/http:" in text:
        return True
    if "connection reset" in text or "connection was reset" in text:
        return True
    if "connection refused" in text:
        return True
    if "i/o timeout" in text:
        return True
    if re.search(r"\beof\b", text):
        return True
    if "timeout awaiting response headers" in text:
        return True
    if re.search(r"\bhttp (?:502|503|504|429)\b", text):
        return True
    if re.search(r"requested url returned error: (?:5\d\d|429)\b", text):
        return True
    if "was submitted too quickly" in text:
        return True
    if "you have exceeded a secondary rate limit" in text:
        return True
    # Primary and other GitHub rate-limit responses (often HTTP 403 or 429).
    if "rate limit" in text:
        return True
    if "error connecting to" in text:
        return True
    if "could not connect" in text or "couldn't connect" in text:
        return True
    # git-specific transport shapes (libcurl / OpenSSL / schannel / Winsock),
    # absent from gh's own Go-idiom error text and therefore inert for gh
    # classification. Each is pinned against either a real string this host's
    # git produced or the libcurl/schannel/OpenSSL wording those tools
    # document verbatim -- see module docstring and
    # tests/test_transient_errors.py.
    if "could not resolve host" in text:
        return True
    if "recv failure" in text or "send failure" in text:
        return True
    if "ssl_read" in text or "ssl_connect" in text:
        return True
    if "schannel:" in text:
        return True
    if "remote end hung up" in text:
        return True
    if "operation timed out after" in text:
        return True
    if "empty reply from server" in text:
        return True
    if "gnutls_handshake" in text:
        return True
    # Windows Winsock's transport error text, surfaced through Go's own
    # syscall-level wrapping (gh's net/http backend on Windows, e.g. "dial
    # tcp ...: connectex: ..."). git itself never emits this token -- inert
    # there, load-bearing for gh on Windows.
    if "connectex" in text:
        return True

    # Unknown errors are terminal by default.
    return False
