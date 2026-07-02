"""Cross-family adversarial review — run a NON-Claude model (e.g. codex via the
Devin CLI) against a spec or PR and capture its findings.

Rationale: the orchestrator's own adversarial review is Claude-family end to end,
so it shares Claude's blind spots. A different model family surfaces what
same-family redundancy cannot. See the standing practice note in memory
(feedback_cross_family_devin_review): findings are LEADS, not verdicts — the
cross-family model over-escalates severity, so every finding must be verified
against live code before it is accepted, and it never gates a merge on its own.

Hard contract: ``run_cross_family_review`` NEVER raises into its caller. A codex
outage, timeout, or non-zero exit must not break review-packet generation — the
failure is captured as a stub report and a not-ok result instead.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

# Signature-compatible with subprocess.run for the happy path; tests inject a fake.
Runner = Callable[..., "subprocess.CompletedProcess[str]"]

_CAVEAT = (
    "> Findings below are **leads, not verdicts** — this cross-family model "
    "over-escalates severity. Verify each against the live code before folding "
    "it in, reject over-escalations with a reason, and never let this report "
    "gate a merge on its own."
)


@dataclass(frozen=True)
class CrossFamilyResult:
    """Outcome of one cross-family review invocation."""

    ok: bool
    report_path: str
    model: str
    returncode: int | None = None
    error: str | None = None
    reused: bool = False


def render_command(command: Sequence[str] | str, values: dict[str, str]) -> list[str] | str:
    """Render the configured command template with ``{name}`` placeholders.

    A sequence renders per-part (no shell); a string renders whole (shell=True),
    mirroring ``adapters._render_command``.
    """
    if isinstance(command, str):
        return command.format(**values)
    return [str(part).format(**values) for part in command]


def run_cross_family_review(
    *,
    model: str,
    command: Sequence[str] | str,
    repo_root: Path,
    prompt_text: str,
    prompt_path: Path,
    report_path: Path,
    timeout_seconds: int,
    runner: Runner = subprocess.run,
) -> CrossFamilyResult:
    """Write ``prompt_text`` to ``prompt_path``, run the cross-family model from
    ``repo_root`` (so it can read the real code), and capture stdout to
    ``report_path``. Returns a result; never raises."""
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(prompt_text, encoding="utf-8")

    rendered = render_command(command, {"model": model, "prompt_path": str(prompt_path)})
    try:
        completed = runner(
            rendered,
            cwd=str(repo_root),
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout_seconds,
            shell=isinstance(rendered, str),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        partial = (
            exc.stdout.decode("utf-8", "replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        return _fail(
            report_path,
            model,
            f"cross-family review timed out after {timeout_seconds}s",
            partial=str(partial),
        )
    except OSError as exc:
        return _fail(report_path, model, f"cross-family runner failed to start: {exc}")
    except subprocess.SubprocessError as exc:  # any other subprocess failure
        return _fail(report_path, model, f"cross-family review errored: {exc}")

    stdout = completed.stdout or ""
    if completed.returncode != 0:
        detail = (completed.stderr or "").strip()
        return _fail(
            report_path,
            model,
            f"cross-family runner exited {completed.returncode}"
            + (f": {detail}" if detail else ""),
            partial=stdout,
            returncode=completed.returncode,
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_report(model, stdout), encoding="utf-8")
    return CrossFamilyResult(ok=True, report_path=str(report_path), model=model, returncode=0)


def _report(model: str, body: str) -> str:
    return f"# Cross-family adversarial review — `{model}`\n\n{_CAVEAT}\n\n---\n\n{body.strip()}\n"


def _fail(
    report_path: Path,
    model: str,
    reason: str,
    *,
    partial: str = "",
    returncode: int | None = None,
) -> CrossFamilyResult:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    stub = f"# Cross-family adversarial review — `{model}` (UNAVAILABLE)\n\n> {reason}\n"
    if partial.strip():
        stub += f"\n---\n\nPartial output before failure:\n\n{partial.strip()}\n"
    report_path.write_text(stub, encoding="utf-8")
    return CrossFamilyResult(
        ok=False, report_path=str(report_path), model=model, returncode=returncode, error=reason
    )
