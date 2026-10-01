"""Infra-vs-code classification for the local merge gate's suite outcomes (#2127).

The gate used to treat every non-ok suite outcome as a code defect and route
it to worker rework, consuming the merge-rework cap. Two outcomes are not code
defects and a worker cannot act on them:

* a **timeout** -- under host CPU contention the serial suite runs 2-3x
  slower, so wall clock measures the host, not the code;
* a **summary-less death** -- the runner (or its process tree) was killed
  externally mid-run: no pytest terminal summary, no traceback.

``classify_suite_outcome`` separates them by a pure test on the log tail: a
real pytest run that *finished* (pass or fail) always prints a terminal
summary / short-summary / INTERNALERROR / interruption / collection-error
marker. When unsure the classifier answers ``CODE_FAILURE`` -- a false
"infra" verdict wastes a suite relaunch, a false "code" verdict is the
pre-#2127 behavior.

``derived_suite_timeout`` replaces the flat 3600s limit with
``clamp(4 x median(recent ok durations))`` so the limit follows the suite's
real baseline instead of a constant.
"""

from __future__ import annotations

import re
import statistics
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from charlie_work import local_suite_runner

DEFAULT_SUITE_TIMEOUT_SECONDS = 3600
MAX_SUITE_TIMEOUT_SECONDS = 10800
TIMEOUT_MEDIAN_MULTIPLIER = 4
MIN_TIMEOUT_SAMPLES = 3
# Issue #2127: bound on infra relaunches per gate episode (see local_gate_infra_relaunch).
LOCAL_SUITE_GATE_MAX_INFRA_RELAUNCHES = 2
MAX_TIMEOUT_SAMPLES = 10


class SuiteOutcome(Enum):
    CODE_FAILURE = "code_failure"
    SUITE_KILLED = "suite_killed"
    SUITE_TIMED_OUT = "suite_timed_out"


# Each pattern is matched per line (after CR/CRLF normalization).
_SUMMARY_MARKERS = (
    # ``=== 2 failed, 10 passed in 3.2s ===`` and the banner-less ``-q`` form.
    re.compile(
        r"\b(?:passed|failed|errors?|skipped|deselected|xfailed|xpassed|warnings?"
        r"|no tests ran)\b.*\bin [\d.]+s\b"
    ),
    re.compile(r"short test summary info"),
    re.compile(r"INTERNALERROR"),
    re.compile(r"\bInterrupted:"),
    re.compile(r"!{3,}.+!{3,}"),
    re.compile(r"ERROR collecting"),
)


def has_pytest_summary(tail: str) -> bool:
    """True when ``tail`` carries any marker of a pytest run that finished."""
    text = tail.replace("\r\n", "\n").replace("\r", "\n")
    return any(marker.search(text) for marker in _SUMMARY_MARKERS) if text else False


def classify_suite_outcome(*, timed_out: bool, tail: str) -> SuiteOutcome:
    """Classify a NON-ok suite outcome (callers never pass a green result)."""
    if timed_out:
        return SuiteOutcome.SUITE_TIMED_OUT
    if has_pytest_summary(tail):
        return SuiteOutcome.CODE_FAILURE
    return SuiteOutcome.SUITE_KILLED


def derived_suite_timeout(results: list[dict[str, Any]]) -> int:
    """``clamp(4 x median(last <=10 ok durations), 3600, 10800)``; 3600 when <3 samples.

    Malformed entries (``ok`` not true, missing/non-positive/non-numeric
    duration) are ignored. "Last" is by ``ended_at`` (ISO strings sort
    chronologically); entries without one sort oldest.
    """
    samples: list[tuple[str, float]] = []
    for result in results:
        duration = result.get("duration_seconds") if isinstance(result, dict) else None
        if (
            not isinstance(result, dict)
            or result.get("ok") is not True
            or isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or duration <= 0
        ):
            continue
        ended = result.get("ended_at")
        samples.append((ended if isinstance(ended, str) else "", float(duration)))
    recent = sorted(samples, key=lambda s: s[0], reverse=True)[:MAX_TIMEOUT_SAMPLES]
    if len(recent) < MIN_TIMEOUT_SAMPLES:
        return DEFAULT_SUITE_TIMEOUT_SECONDS
    derived = TIMEOUT_MEDIAN_MULTIPLIER * statistics.median(d for _, d in recent)
    return int(min(MAX_SUITE_TIMEOUT_SECONDS, max(DEFAULT_SUITE_TIMEOUT_SECONDS, round(derived))))


def read_gate_results(dispatches_dir: Path) -> list[dict[str, Any]]:
    """Every parseable per-PR ``suite-result.json`` under the gate dir.

    A result without ``ended_at`` gets its file mtime as an ISO fallback so
    recency ordering never depends on a field the wrapper may omit.
    """
    gate_root = local_suite_runner.suite_gate_paths(dispatches_dir, 0).gate_dir.parent
    results: list[dict[str, Any]] = []
    try:
        pr_dirs = [d for d in gate_root.glob("pr-*") if d.is_dir()]
    except OSError:
        return results
    for pr_dir in pr_dirs:
        try:
            paths = local_suite_runner.suite_gate_paths(dispatches_dir, int(pr_dir.name[3:]))
            result = local_suite_runner.read_gate_result(paths)
            if result is None:
                continue
            if not isinstance(result.get("ended_at"), str):
                mtime = paths.result.stat().st_mtime
                result = {**result, "ended_at": datetime.fromtimestamp(mtime, UTC).isoformat()}
        except (ValueError, OSError):
            continue
        results.append(result)
    return results


def effective_suite_timeout(dispatches_dir: Path) -> int:
    """The in-flight suite time limit for the gate under ``dispatches_dir``."""
    return derived_suite_timeout(read_gate_results(dispatches_dir))
