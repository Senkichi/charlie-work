"""The internal Adapter seam (design doc section 7): one place that knows how
each harness reports liveness and failure, and ``classify_for`` -- the single
classifier entry point that reads the harness's detection flags off its profile.

Split out of ``worker_fate.py`` (file-size ratchet); ``worker_fate``
re-exports ``AdapterFateProfile``, ``profile_for`` and ``classify_for``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .config import OrchestratorConfig
from .failure_classifier import classify_failure
from .worker import WorkerView


@dataclass(frozen=True)
class AdapterFateProfile:
    """The internal Adapter seam (design doc §7): one place that knows how
    each harness reports liveness and failure, replacing the 14
    ``w.adapter_kind ==`` branches in ``dead_worker_reap.py``.
    """

    harness: str  # key in harnesses.HARNESS_REGISTRY / WORKER_HARNESSES
    view_kinds: frozenset[str]  # WorkerView.adapter_kind spellings ("devin" for devin-shell)
    account_error_detection: bool  # api only: provider_suspended / provider_auth
    headless_permission_detection: bool  # issue #2010: permission_denied (claude-code, api)
    record_failure: Callable[..., tuple[str | None, str | None]] | None
    # (sessions_dir, issue_number, *, fallback_kind, config, now) -> the
    # existing update_worker_record_with_failure_classification /
    # update_session_record_with_failure_classification sidecar writer.
    over_budget: Callable[[WorkerView, OrchestratorConfig], bool] | None  # api only
    # Issue #2052: whether this harness's launcher runs the terminal-status
    # watcher (process_utils.start_terminal_status_watcher) that leaves a
    # durable ``issue-<n>.<suffix>.terminal.json`` when the spawned process
    # exits. True for every Popen-backed harness (claude-code, api,
    # devin-shell); command and manual never spawn a worker process, so
    # there is nothing to watch.
    writes_terminal_record: bool
    # Narrows a log to the harness's own provider-error records before any
    # tail classifier reads it -- for a harness whose log also carries tool
    # output (opencode: opencode_log.provider_error_digest). None = raw log.
    log_digest: Callable[[str], str] | None = None
    # (provider-error tail) -> how long until the tripped quota resets, or None
    # for the classifier's fixed 24h -- for a provider whose quota windows are
    # not 24h (opencode Go: opencode_limits.go_quota_reset). None = 24h.
    quota_reset: Callable[[str], timedelta | None] | None = None


_PROFILES: dict[str, AdapterFateProfile] | None = None


def _opencode_quota_reset(tail: str) -> timedelta | None:
    """Late-bound ``opencode_limits.go_quota_reset`` (it imports the worker module)."""
    from . import opencode_limits

    return opencode_limits.go_quota_reset(tail)


def _devin_record_failure(*args: Any, **kwargs: Any) -> tuple[str | None, str | None]:
    """Late-bound ``devin_shell.update_session_record_with_failure_classification``."""
    from . import devin_shell

    return devin_shell.update_session_record_with_failure_classification(*args, **kwargs)


def _claude_code_record_failure(*args: Any, **kwargs: Any) -> tuple[str | None, str | None]:
    """Late-bound ``claude_code.update_worker_record_with_failure_classification``."""
    from . import claude_code

    return claude_code.update_worker_record_with_failure_classification(
        *args, adapter_kind="claude-code", **kwargs
    )


def _api_record_failure(*args: Any, **kwargs: Any) -> tuple[str | None, str | None]:
    """Late-bound ``claude_code.update_worker_record_with_failure_classification`` (api)."""
    from . import claude_code

    return claude_code.update_worker_record_with_failure_classification(
        *args, adapter_kind="api", **kwargs
    )


def _opencode_record_failure(*args: Any, **kwargs: Any) -> tuple[str | None, str | None]:
    """Late-bound ``claude_code.update_worker_record_with_failure_classification`` (opencode)."""
    from . import claude_code

    return claude_code.update_worker_record_with_failure_classification(
        *args, adapter_kind="opencode", **kwargs
    )


def _build_profiles() -> dict[str, AdapterFateProfile]:
    """Build the by-harness profile table, then index it by ``view_kinds``.

    Imports the adapter modules lazily (function body, not module top
    level): ``claude_code``/``devin_shell`` reach back into this module's
    ``classify_failure`` from inside their own functions, so a top-level
    import here would cycle. This function only ever runs at call time
    (from ``profile_for``), by which point every module involved has
    already finished loading, so the cycle risk does not apply to a lazy
    import -- only to a module-level one.

    ``update_worker_record_with_failure_classification`` (claude-code, api)
    takes an ``adapter_kind`` kwarg that selects both the sidecar filename
    suffix and account-error detection -- unlike ``update_session_record_
    with_failure_classification`` (devin), which has no such parameter. The
    ``_*_record_failure`` shims bind each profile's value so every call site
    can call ``record_failure`` with the exact same positional/keyword shape
    regardless of which adapter it resolved to. They resolve the writer at
    CALL time (see :func:`_api_over_budget`): the registry is cached for the
    process, so binding the function objects here would freeze whichever
    implementation -- including a test's ``patch`` mock -- was live at the
    first ``profile_for`` call.
    """
    from .harnesses import WORKER_HARNESSES
    from .opencode_log import provider_error_digest

    by_harness = {
        "devin-shell": AdapterFateProfile(
            harness="devin-shell",
            view_kinds=frozenset({"devin"}),
            account_error_detection=False,
            headless_permission_detection=False,
            record_failure=_devin_record_failure,
            over_budget=None,
            writes_terminal_record=True,
        ),
        "claude-code": AdapterFateProfile(
            harness="claude-code",
            view_kinds=frozenset({"claude-code"}),
            account_error_detection=False,
            headless_permission_detection=True,
            record_failure=_claude_code_record_failure,
            over_budget=None,
            writes_terminal_record=True,
        ),
        "api": AdapterFateProfile(
            harness="api",
            view_kinds=frozenset({"api"}),
            account_error_detection=True,
            headless_permission_detection=True,
            record_failure=_api_record_failure,
            over_budget=_api_over_budget,
            writes_terminal_record=True,
        ),
        # opencode sessions are ClaudeWorkerRecord sidecars written by
        # launch_claude_worker, so they share its classifier/terminal watcher.
        # The api-only provider-account and Claude-CLI permission-denial
        # detectors match claude-specific text and stay off.
        "opencode": AdapterFateProfile(
            harness="opencode",
            view_kinds=frozenset({"opencode"}),
            account_error_detection=False,
            headless_permission_detection=False,
            record_failure=_opencode_record_failure,
            over_budget=None,
            writes_terminal_record=True,
            log_digest=provider_error_digest,
            quota_reset=_opencode_quota_reset,
        ),
        # "command" and "manual" have no failure-classification or budget
        # consumer today: dead_worker_reap.py's 14 sites never branch on
        # either adapter_kind, and neither harness has an issue driving
        # account-error detection or an over-budget check. Declared here
        # (all capabilities off) purely so the WORKER_HARNESSES completeness
        # assert below covers all 5 harnesses, matching the adapters.py
        # #1513 pattern this seam follows -- not because either capability
        # is known to be correct for them. A future consumer that needs
        # real values for these two should fill them in then, not infer
        # them from this placeholder.
        "command": AdapterFateProfile(
            harness="command",
            view_kinds=frozenset({"command"}),
            account_error_detection=False,
            headless_permission_detection=False,
            record_failure=None,
            over_budget=None,
            writes_terminal_record=False,
        ),
        "manual": AdapterFateProfile(
            harness="manual",
            view_kinds=frozenset({"manual"}),
            account_error_detection=False,
            headless_permission_detection=False,
            record_failure=None,
            over_budget=None,
            writes_terminal_record=False,
        ),
    }
    # N6 (wf-review-opus.md): an explicit raise, not a bare `assert` --
    # this runs lazily on first `profile_for` call (mid-reap-pass, not at
    # import), and a bare `assert` here is silently stripped under
    # `python -O`, disabling the completeness guard exactly when a
    # harness/profile drift would otherwise be caught.
    if {p.harness for p in by_harness.values()} != WORKER_HARNESSES:
        raise AssertionError(
            "adapter_fate_profile._build_profiles must declare exactly the harnesses "
            "harnesses.WORKER_HARNESSES declares valid -- keep both in sync "
            "(adapters.py #1513 completeness-assert pattern)"
        )
    return by_harness


def _api_over_budget(view: WorkerView, config: OrchestratorConfig) -> bool:
    """Resolve ``worker._api_session_over_budget`` at call time.

    The profile registry is built once per process, so binding the function
    object at build time would freeze whichever implementation happened to be
    live at the first ``profile_for`` call -- making a later patch of
    ``charlie_work.worker._api_session_over_budget`` depend on test order.
    """
    from . import worker

    return worker._api_session_over_budget(view, config)


def profile_for(adapter_kind: str) -> AdapterFateProfile | None:
    """Look up the seam profile by ``WorkerView.adapter_kind`` spelling
    (e.g. ``"devin"``, not the harness name ``"devin-shell"``).

    Returns ``None`` for an unrecognized value rather than raising. The
    design doc's §7 says an unknown value "raises KeyError at the seam" --
    deviated from deliberately here: the 4 triplet call sites in
    ``dead_worker_reap.py`` disagree today on what an unrecognized
    ``adapter_kind`` should do (one keeps a pre-set ``fallback_kind``,
    three default to ``(None, None)``), so a raising ``profile_for`` would
    turn each site's existing graceful degradation into a crash. Returning
    ``None`` lets every call site keep its own pre-existing fallback.
    """
    global _PROFILES
    if _PROFILES is None:
        by_harness = _build_profiles()
        _PROFILES = {
            view_kind: profile
            for profile in by_harness.values()
            for view_kind in profile.view_kinds
        }
    return _PROFILES.get(adapter_kind)


def classify_for(
    adapter_kind: str,
    log_path: Path,
    throttle_error_markers: Sequence[str] | None = None,
    *,
    quota_error_markers: Sequence[str] | None = None,
    resume_margin_seconds: int = 0,
    now: datetime | None = None,
) -> tuple[str | None, str | None]:
    """Classify a session failure for ``adapter_kind`` (a ``WorkerView``
    spelling: ``"devin"``, ``"claude-code"``, ``"api"``, ...).

    The single classifier entry point: the profile supplies the three
    per-harness flags (``account_error_detection``,
    ``headless_permission_detection``) so the adapters no
    longer carry their own ``_classify_session_failure`` wrappers that
    hard-code them. An unknown ``adapter_kind`` returns ``(None, None)`` --
    the same graceful "nothing classified" ``profile_for`` gives its callers.
    """
    profile = profile_for(adapter_kind)
    if profile is None:
        return None, None
    return classify_failure(
        log_path,
        throttle_error_markers,
        quota_error_markers=quota_error_markers,
        resume_margin_seconds=resume_margin_seconds,
        account_error_detection=profile.account_error_detection,
        headless_permission_detection=profile.headless_permission_detection,
        now=now,
        log_digest=profile.log_digest,
        quota_reset=profile.quota_reset,
    )
