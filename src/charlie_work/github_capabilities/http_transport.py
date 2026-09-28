"""Pooled stdlib HTTP transport for `gh api` REST-GET / graphql calls
(issue #1834; builds on the #1833 circuit breaker in `circuit_breaker.py` /
`circuit_breaker_transport.py`).

`GitHub.run()` is the single interception seam every `gh` CLI invocation
passes through (see that method's own comments and the Track 2 Mikado-graph
design doc -- it never moves off the owner). This module supplies
`run_gh_command`, the one function `run()` calls in place of its previous
single `subprocess.run(command, ...)` line. `run_gh_command` always returns
a `subprocess.CompletedProcess[str]` (or raises `FileNotFoundError` /
`subprocess.TimeoutExpired`, exactly like the real `subprocess.run` call it
replaces) so every downstream line of `run()` -- retry loop, circuit
breaker recording, JSON parsing, `GitHubRunResult` construction, the
`_should_retry`/`_is_not_found_gh_error`/`transient_errors` classifiers --
runs completely unchanged regardless of which transport actually produced
the result. This is the mechanism that makes "error translation at the
seam" tractable without a second, separately-maintained copy of `run()`'s
retry/classification logic: HTTP failures are translated into the exact
stderr *text* `gh` itself would have produced (`gh: {message} (HTTP
{status})` for REST, `GraphQL: {message} ({path})` for graphql -- both
verified against this repo's own live `gh` output, see
`tests/test_http_transport.py`), so every existing string-matching consumer
keeps working without modification.

Transport selection and per-call fallback (owner directives, issue #1834):

* `runtime.gh_transport` (`config.RuntimeConfig`, default `"http"`) picks the
  transport; `"gh"` is a kill-switch, not an opt-in. `GitHub` instances
  constructed with `runtime=None` (tests, legacy direct callers) get `"gh"`
  from `_DEFAULT_GH_TRANSPORT` below -- the same "tests and legacy callers"
  carve-out `transport.py`'s `_max_retries`/`_timeout_seconds`/etc. already
  use, not a new pattern.
* Only `http_translate.is_http_candidate(args)` shapes are ever attempted
  over HTTP; everything else takes the `gh` subprocess path with no HTTP
  involvement at all (not a "fallback" -- ordinary routing).
* For a candidate call, if the HTTP path cannot itself produce a result --
  `gh auth token` resolution fails, or an internal exception occurs while
  building the request or parsing a structurally-unexpected response body
  (never a normal GitHub error response, which IS a result: see
  `_HttpTransportUnavailable`'s docstring) -- the call falls through to the
  real `gh` subprocess and a `github_transport_fallback` event is recorded.
* Network-level failures during an actually-attempted HTTP request
  (connection refused/reset, TLS handshake timeout, read timeout) are NOT
  fallback triggers: they are translated into the same stderr vocabulary a
  real `gh` network failure would produce and returned as an ordinary
  failed `CompletedProcess` (or a raised `TimeoutExpired`), so `run()`'s
  *existing* retry loop and circuit breaker handle them exactly as they
  already handle a `gh`-subprocess network failure -- one shared breaker
  state and one shared retry policy for both transports, not a duplicate.

Stdlib only (issue #1834 hard requirement): `http.client.HTTPSConnection`,
reused across calls on one `GitHub` instance via `HttpTransportState`
(pooled keep-alive) rather than reconnecting per call.
"""

from __future__ import annotations

import json
import logging
import socket
import ssl
import subprocess
from dataclasses import dataclass
from http.client import HTTPSConnection
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from .. import layout
from ..instrumentation import log_event
from ..subprocess_runner import no_console_window_kwargs
from . import http_cache
from .circuit_breaker_transport import circuit_breaker_state_path
from .http_translate import HttpRequestPlan, build_request_plan, is_http_candidate

if TYPE_CHECKING:
    from ..config import RuntimeConfig

logger = logging.getLogger(__name__)

GITHUB_API_HOST = "api.github.com"

# Fallback used only when `GitHub` is constructed with `runtime=None` (tests
# and legacy direct callers) -- mirrors `transport.py`'s
# `_DEFAULT_GH_MAX_RETRIES`/`_DEFAULT_GH_TIMEOUT_SECONDS` "tests and legacy
# callers" carve-out. Deliberately NOT "http": a large share of this repo's
# existing test suite constructs `GitHub(tmp_path)` (no runtime) and
# monkeypatches `subprocess.run` directly to simulate `gh` CLI responses for
# `gh api`-shaped calls -- exactly the shapes this module would otherwise
# intercept before they ever reach that mock. Real orchestrator code always
# passes `runtime=config.runtime` (never `None`), so this fallback is never
# reached in production; it only governs the construction shape this
# repo's own tests already rely on.
_DEFAULT_GH_TRANSPORT = "gh"

# Safety cap on `--paginate` follow-the-Link-header loops. `gh --paginate`
# itself has no hardcoded cap and will follow a Link chain indefinitely; this
# bound exists only to protect the orchestrator from a malformed or
# adversarial Link chain, at a page count far beyond any real call site in
# this codebase actually needs (`merged_pr_list`'s own `_LIST_LIMIT` cap is
# 500 items / 100 per page = 5 pages).
_MAX_PAGINATE_PAGES = 50

# Response status that means "unchanged since the cached ETag".
_NOT_MODIFIED = 304


class _HttpTransportUnavailable(Exception):
    """Raised only when the HTTP path itself cannot produce a result at all:
    token resolution failed, or an internal exception occurred while
    building the request / parsing a structurally-unexpected response body.

    Never raised for a normal GitHub error response (404, 401, 403, 429,
    5xx, a graphql `errors` array, ...) -- those ARE results, translated by
    `_translate_error_response`/`_translate_graphql_errors` into an ordinary
    failed `CompletedProcess`, exactly like a real `gh` failure. Catching
    this exception is what `run_gh_command` uses to decide to fall back to
    the `gh` subprocess and emit `github_transport_fallback`.
    """


@dataclass
class HttpTransportState:
    """Mutable per-`GitHub`-instance transport state: pooled connection plus
    cached bearer token.

    Constructed once per `GitHub` instance (mirrors `_circuit_breaker_state`/
    `_list_cache` in `GitHub.__post_init__`) -- NOT a frozen config/value
    object, so CLAUDE.md's frozen-dataclass invariant does not apply here;
    it is runtime state, the same carve-out those two existing attributes
    already use.
    """

    connection: HTTPSConnection | None = None
    token: str | None = None


def build_http_transport_state() -> HttpTransportState:
    """Construct a fresh, empty `HttpTransportState` (issue #1834).

    Called exactly once, from `GitHub.__post_init__`, alongside
    `build_circuit_breaker_state`.
    """
    return HttpTransportState()


def gh_transport_mode(runtime: "RuntimeConfig | None") -> str:
    """Resolve the effective transport ("http" or "gh") for a `GitHub` instance.

    See `_DEFAULT_GH_TRANSPORT`'s docstring for why `runtime=None` resolves
    differently from an explicit `RuntimeConfig` (whose `gh_transport` field
    defaults to `"http"`).
    """
    if runtime is not None:
        return runtime.gh_transport
    return _DEFAULT_GH_TRANSPORT


def _resolve_token(repo_root: Path) -> str | None:
    """Run `gh auth token` once. Returns None on any failure -- the caller
    treats that as `_HttpTransportUnavailable` (falls back to `gh`, which
    will surface the real auth problem itself, e.g. "not logged in").
    """
    try:
        result = subprocess.run(
            ["gh", "auth", "token"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=10,
            **no_console_window_kwargs(),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    token = result.stdout.strip()
    return token or None


def _ensure_token(state: HttpTransportState, repo_root: Path) -> str | None:
    if state.token is None:
        state.token = _resolve_token(repo_root)
    return state.token


def _get_connection(state: HttpTransportState, timeout_seconds: float) -> HTTPSConnection:
    if state.connection is None:
        state.connection = HTTPSConnection(GITHUB_API_HOST, timeout=timeout_seconds)
    return state.connection


def _drop_connection(state: HttpTransportState) -> None:
    """Discard a possibly-broken pooled connection so the next call reconnects."""
    if state.connection is not None:
        try:
            state.connection.close()
        except OSError:
            pass
        state.connection = None


def _request_headers(
    plan: HttpRequestPlan, token: str, cached: http_cache.CachedResponse | None
) -> dict[str, str]:
    headers = dict(plan.headers)
    headers.setdefault("Authorization", f"Bearer {token}")
    headers.setdefault("Accept", "application/vnd.github+json")
    headers.setdefault("User-Agent", "charlie-work-http-transport")
    headers.setdefault("X-GitHub-Api-Version", "2022-11-28")
    headers["Host"] = GITHUB_API_HOST
    if cached is not None:
        headers["If-None-Match"] = cached.etag
    return headers


@dataclass
class _RawResponse:
    status: int
    headers: dict[str, str]
    body: str


def _send_once(
    state: HttpTransportState,
    repo_root: Path,
    method: str,
    path: str,
    headers: dict[str, str],
    body: bytes | None,
    timeout_seconds: float,
) -> _RawResponse:
    """Send exactly one HTTP request over the pooled connection.

    Raises the underlying `OSError`/`socket.timeout`/`ssl.SSLError` on any
    connection-level failure -- the caller translates those, it does not
    catch them here, so a fresh connection is created by `_get_connection`
    on the next attempt via `_drop_connection`.
    """
    connection = _get_connection(state, timeout_seconds)
    try:
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        raw_body = response.read()
        resp_headers = {name: value for name, value in response.getheaders()}
        status = response.status
    except (OSError, socket.timeout, ssl.SSLError):
        _drop_connection(state)
        raise
    text = raw_body.decode("utf-8", errors="replace")
    return _RawResponse(status=status, headers=resp_headers, body=text)


def _next_link(headers: dict[str, str]) -> str | None:
    """Extract the `rel="next"` URL from a `Link` header, if present."""
    link = headers.get("Link") or headers.get("link")
    if not link:
        return None
    for part in link.split(","):
        segments = part.split(";")
        if len(segments) < 2:
            continue
        url = segments[0].strip().lstrip("<").rstrip(">")
        rel_part = ";".join(segments[1:])
        if 'rel="next"' in rel_part:
            return url
    return None


def _path_from_url(url: str) -> str:
    """Reduce an absolute `https://api.github.com/...` Link URL to a bare
    request path -- pagination Link headers always stay on the same host.
    """
    marker = f"https://{GITHUB_API_HOST}"
    if url.startswith(marker):
        return url[len(marker) :] or "/"
    return url


def _extract_message(status: int, body: str) -> str:
    """Best-effort extraction of GitHub's own REST error message field,
    passed through verbatim (never hand-translated) so it carries whatever
    substrings GitHub itself encodes -- "Bad credentials", "API rate limit
    exceeded", etc. -- that this repo's classifiers already match on.
    """
    try:
        parsed = json.loads(body) if body else None
    except (json.JSONDecodeError, ValueError):
        parsed = None
    if isinstance(parsed, dict) and isinstance(parsed.get("message"), str) and parsed["message"]:
        return parsed["message"]
    return body.strip()[:200] or f"HTTP {status}"


def _translate_error_response(status: int, body: str) -> str:
    """Build the stderr text for a non-2xx REST response, matching gh's own
    `gh: {message} (HTTP {status})` format -- verified against this
    repository's real `gh api` error output (see
    tests/test_http_transport.py). Every transient/terminal classifier in
    this repo matches on the literal `HTTP {status}` substring (case
    -insensitive, word-boundary), which this format always contains.
    """
    return f"gh: {_extract_message(status, body)} (HTTP {status})"


def _translate_graphql_errors(errors: list[Any]) -> str:
    """Build the stderr text for a 200-status graphql response whose body
    carries a non-empty `errors` array -- gh's own CLI treats this as a
    command failure (non-zero exit) even though the HTTP status is 200, and
    formats it as `GraphQL: {message} ({dotted.path})` (verified against
    this repo's own live `gh` output, e.g. `Could not resolve to a
    PullRequest with the number of N. (repository.pullRequest)`).
    """
    first = errors[0] if errors else {}
    message = str(first.get("message", first)) if isinstance(first, dict) else str(first)
    path = first.get("path") if isinstance(first, dict) else None
    if isinstance(path, list) and path:
        dotted = ".".join(str(p) for p in path)
        return f"GraphQL: {message} ({dotted})"
    return f"GraphQL: {message}"


def _emit_fallback_event(state_path: Path, *, command: list[str], reason: str) -> None:
    """Record a `github_transport_fallback` event for a call that could not
    be served over HTTP and fell through to the `gh` subprocess (issue
    #1834). Direct literal-kind `log_event` call (not a wrapper that
    forwards a `kind` parameter) for the same reason
    `circuit_breaker_transport.note_circuit_breaker_result` inlines its
    calls: this repo's AST-based event-kind scanners
    (`tests/test_instrumentation_event_kind_registry.py`,
    `tests/test_event_kind_consumers.py`) resolve `kind` by literal
    matching.
    """
    payload = {"command": " ".join(command), "reason": reason}
    # write-gate-exempt(issue=1834): GitHub client layer has no WriteGate; transport fallback bookkeeping is lock-free
    log_event(state_path, "github_transport_fallback", payload)


def _cache_path(runtime: "RuntimeConfig | None", repo_root: Path) -> Path:
    """Resolves the ETag cache path the same way
    `circuit_breaker_transport.circuit_breaker_state_path` resolves
    `state.json`'s -- without importing `.config` (would cycle).
    """
    state_dir = runtime.state_dir if runtime is not None else layout.DEFAULT_STATE_DIR
    root = Path(state_dir)
    if not root.is_absolute():
        root = repo_root / root
    return layout.http_etag_cache_path(root.resolve())


def _run_via_gh_subprocess(
    command: list[str], cwd: Path, timeout_seconds: float
) -> subprocess.CompletedProcess[str]:
    """The original `run()` subprocess call, unchanged -- both the
    default-`gh`-transport path and every HTTP fallback path go through this
    single function so there is exactly one place that spawns `gh`.
    """
    return subprocess.run(
        command,
        cwd=cwd,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
        timeout=timeout_seconds,
        **no_console_window_kwargs(),
    )


def _execute_rest_get(
    state: HttpTransportState,
    repo_root: Path,
    plan: HttpRequestPlan,
    token: str,
    cache_path: Path,
    timeout_seconds: float,
) -> subprocess.CompletedProcess[str]:
    cached = http_cache.get_cached(cache_path, plan.path)
    headers = _request_headers(plan, token, cached)

    response = _send_once(
        state, repo_root, plan.method, plan.path, headers, plan.body, timeout_seconds
    )
    if response.status == 401:
        # Re-resolve once on 401 (issue #1834 hard requirement) and retry
        # this single request exactly once more before treating it as a
        # genuine terminal auth failure.
        state.token = _resolve_token(repo_root)
        if state.token is None:
            raise _HttpTransportUnavailable("gh auth token re-resolution failed after HTTP 401")
        headers = _request_headers(plan, state.token, cached)
        response = _send_once(
            state, repo_root, plan.method, plan.path, headers, plan.body, timeout_seconds
        )

    if response.status == _NOT_MODIFIED and cached is not None:
        body_text = cached.body
        status = cached.status
    else:
        body_text = response.body
        status = response.status
        etag = response.headers.get("ETag") or response.headers.get("etag")
        if status == 200 and etag:
            http_cache.record_response(
                cache_path, plan.path, etag=etag, status=status, body=body_text
            )

    if plan.paginate and 200 <= status < 300:
        body_text = _follow_pagination(
            state, repo_root, plan, headers, timeout_seconds, body_text, response.headers
        )

    if not (200 <= status < 300):
        return subprocess.CompletedProcess(
            args=["gh", "api", plan.path],
            returncode=1,
            stdout="",
            stderr=_translate_error_response(status, body_text),
        )
    return subprocess.CompletedProcess(
        args=["gh", "api", plan.path], returncode=0, stdout=body_text, stderr=""
    )


def _follow_pagination(
    state: HttpTransportState,
    repo_root: Path,
    plan: HttpRequestPlan,
    headers: dict[str, str],
    timeout_seconds: float,
    first_body: str,
    first_headers: dict[str, str],
) -> str:
    """Follow `Link: rel="next"` pages, concatenating JSON arrays -- the
    response shape every `--paginate` call site in this codebase actually
    relies on (`_gh_api_list`, `merged_pr_list`).

    Every failure mode below raises `_HttpTransportUnavailable` instead of
    returning whatever was collected so far, so `run_gh_command` falls back
    to the real `gh` subprocess (which itself follows the full Link chain)
    rather than silently handing a caller a truncated page set it cannot
    tell apart from a complete one:

      * the first page is not a JSON array -- an object response with
        `--paginate` is a shape `gh` handles by merging keys, which no call
        site in this codebase uses and this module does not replicate, so
        it is unsupported rather than assumed complete;
      * a later page responds non-2xx, or its body is not parseable JSON,
        or is not a JSON array;
      * `_MAX_PAGINATE_PAGES` is reached while a `rel="next"` link is still
        present -- the safety cap firing at all means more data existed
        than was fetched.

    An `OSError`/`socket.timeout`/`ssl.SSLError` raised by `_send_once`
    while fetching a later page is deliberately NOT caught here: it
    propagates to `run_gh_command`'s own translation, exactly like a
    first-page connection failure (module docstring: "NOT a fallback
    trigger").
    """
    try:
        items = json.loads(first_body)
    except (json.JSONDecodeError, ValueError):
        raise _HttpTransportUnavailable("--paginate first page was not valid JSON") from None
    if not isinstance(items, list):
        raise _HttpTransportUnavailable("--paginate first page was a JSON object, not an array")

    all_items = list(items)
    page_headers = first_headers
    pages = 1
    while True:
        next_url = _next_link(page_headers)
        if not next_url:
            break
        if pages >= _MAX_PAGINATE_PAGES:
            raise _HttpTransportUnavailable(
                f"--paginate exceeded {_MAX_PAGINATE_PAGES} pages with a next link still present"
            )
        next_path = _path_from_url(next_url)
        response = _send_once(state, repo_root, "GET", next_path, headers, None, timeout_seconds)
        if not (200 <= response.status < 300):
            raise _HttpTransportUnavailable(
                f"--paginate page {pages + 1} returned HTTP {response.status}"
            )
        try:
            next_items = json.loads(response.body)
        except (json.JSONDecodeError, ValueError):
            raise _HttpTransportUnavailable(
                f"--paginate page {pages + 1} was not valid JSON"
            ) from None
        if not isinstance(next_items, list):
            raise _HttpTransportUnavailable(
                f"--paginate page {pages + 1} was a JSON object, not an array"
            )
        all_items.extend(next_items)
        page_headers = response.headers
        pages += 1
    return json.dumps(all_items)


def _execute_graphql(
    state: HttpTransportState,
    repo_root: Path,
    plan: HttpRequestPlan,
    token: str,
    timeout_seconds: float,
) -> subprocess.CompletedProcess[str]:
    headers = _request_headers(plan, token, cached=None)
    response = _send_once(
        state, repo_root, plan.method, plan.path, headers, plan.body, timeout_seconds
    )
    if response.status == 401:
        state.token = _resolve_token(repo_root)
        if state.token is None:
            raise _HttpTransportUnavailable("gh auth token re-resolution failed after HTTP 401")
        headers = _request_headers(plan, state.token, cached=None)
        response = _send_once(
            state, repo_root, plan.method, plan.path, headers, plan.body, timeout_seconds
        )

    if not (200 <= response.status < 300):
        return subprocess.CompletedProcess(
            args=["gh", "api", "graphql"],
            returncode=1,
            stdout="",
            stderr=_translate_error_response(response.status, response.body),
        )

    try:
        parsed = json.loads(response.body) if response.body else None
    except (json.JSONDecodeError, ValueError) as exc:
        raise _HttpTransportUnavailable(f"graphql response was not valid JSON: {exc}") from None
    if not isinstance(parsed, dict):
        raise _HttpTransportUnavailable("graphql response was not a JSON object")

    errors = parsed.get("errors")
    if isinstance(errors, list) and errors:
        # Keep the response body on stdout even though the call "failed":
        # that is what real `gh api graphql` does -- `processResponse` in
        # cli/cli's `pkg/cmd/api/api.go` tees the body and io.Copy's it to
        # the output writer *before* emitting the error to stderr. The body
        # carries partial `data` (resolved aliases alongside nulls for the
        # nodes `errors` names), which `GitHub.run(allow_failure=True)`
        # parses into `GitHubRunResult.value` -- `_graphql_issue_states`
        # consumes it to scope its per-issue fallback to just the
        # unresolvable numbers (issue #1933).
        return subprocess.CompletedProcess(
            args=["gh", "api", "graphql"],
            returncode=1,
            stdout=response.body,
            stderr=_translate_graphql_errors(errors),
        )
    return subprocess.CompletedProcess(
        args=["gh", "api", "graphql"], returncode=0, stdout=response.body, stderr=""
    )


def run_gh_command(
    *,
    args: list[str],
    command: list[str],
    cwd: Path,
    timeout_seconds: float,
    runtime: "RuntimeConfig | None",
    transport_state: HttpTransportState,
    resolve_owner_repo: Callable[[], tuple[str, str]],
) -> subprocess.CompletedProcess[str]:
    """Execute one `gh` invocation, choosing HTTP or the `gh` subprocess.

    This is the single function `GitHub.run()` calls in place of its former
    direct `subprocess.run(command, ...)` line -- see the module docstring
    for the full contract (return shape, fallback triggers, what is and is
    not translated here vs. left to `run()`'s existing retry loop).
    """
    transport = gh_transport_mode(runtime)
    if transport != "http" or not is_http_candidate(args):
        return _run_via_gh_subprocess(command, cwd, timeout_seconds)

    state_path = circuit_breaker_state_path(runtime, cwd)
    try:
        token = _ensure_token(transport_state, cwd)
        if token is None:
            raise _HttpTransportUnavailable("gh auth token resolution failed")
        owner, repo = resolve_owner_repo()
        plan = build_request_plan(args, owner, repo)
        if plan.method == "GET":
            cache_path = _cache_path(runtime, cwd)
            return _execute_rest_get(
                transport_state, cwd, plan, token, cache_path, timeout_seconds
            )
        return _execute_graphql(transport_state, cwd, plan, token, timeout_seconds)
    except _HttpTransportUnavailable as exc:
        _emit_fallback_event(state_path, command=command, reason=str(exc))
        return _run_via_gh_subprocess(command, cwd, timeout_seconds)
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        # Genuinely unexpected internal failure while building the request
        # or parsing a response we did not anticipate the shape of -- the
        # same "unsupported gh subcommand" / "unexpected response shape"
        # class the owner directive names, just discovered at call time
        # rather than by `is_http_candidate` up front. Never lets a defect
        # in this module take down an orchestrator pass; the real `gh`
        # subprocess still answers the call.
        _emit_fallback_event(state_path, command=command, reason=f"unexpected error: {exc}")
        return _run_via_gh_subprocess(command, cwd, timeout_seconds)
    except (OSError, socket.timeout, ssl.SSLError) as exc:
        # Connection-level failure during an actually-attempted HTTP
        # request. NOT a fallback trigger (see module docstring): translated
        # into the same failure shape a real `gh` network error would
        # produce, so `run()`'s existing retry loop and circuit breaker
        # handle it identically for both transports.
        if isinstance(exc, socket.timeout):
            raise subprocess.TimeoutExpired(cmd=command, timeout=timeout_seconds) from exc
        return subprocess.CompletedProcess(
            args=command,
            returncode=1,
            stdout="",
            stderr=f"error connecting to https://{GITHUB_API_HOST}: {exc}",
        )


__all__ = [
    "HttpTransportState",
    "build_http_transport_state",
    "gh_transport_mode",
    "run_gh_command",
]
