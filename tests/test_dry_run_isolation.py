"""``--dry-run`` must not mutate local state.

``dry_run`` was introduced to suppress mutating ``gh`` calls (``github._is_mutating``)
and nothing else, so several paths kept writing local state underneath it. The worst
was ``self_deploy``, which fast-forward-pulls the *live deployed checkout* and may
``uv sync`` its venv — and because a HEAD move terminates a running ``fleet supervise``
by design (drift exit), an ungated preview could end the fleet rather than describe it
(issues #609, #613).

The behavioural tests pin each fixed site in both directions: previewing writes
nothing, and a real run still writes. The AST guards at the bottom are the part that
closes the *class* rather than the instances — they derive every call site from the
source tree, so a newly added caller that forgets to thread ``dry_run`` fails the
suite instead of shipping. That is the same shape as the parser-walk guard in
``test_cli.py``, which is why the flag-parsing half of this bug class stayed closed.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any, Callable

import pytest

from charlie_work.cross_family import run_cross_family_review
from charlie_work.subprocess_runner import RunResult
from charlie_work.supervise import SelfDeployResult, self_deploy

SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "charlie_work"

# Functions that mutate something outside the process and are reachable from a
# CLI path carrying --dry-run. Every production call site must thread the flag.
#
# Only add a callee here when the *function itself* takes ``dry_run``. The guard
# looks for the keyword at the call site, so it cannot see a call that is instead
# protected by an enclosing ``if not dry_run:`` -- which is a correct pattern, just
# a different one. ``save_idle_streaks`` is the live example: it is properly gated
# by its caller in ``runner_allocation_pass``, and adding it here would report a
# false offender and invite someone to "fix" working code.
DRY_RUN_REQUIRED_CALLEES = ("self_deploy", "observe_runner_pool")


def _make_fake_runner(
    responses: list[RunResult],
) -> tuple[Callable[..., RunResult], list[list[str]]]:
    """Return a run_command stub that consumes ``responses`` and records commands.

    Deliberately local rather than imported from ``test_supervise``: sharing a
    mutable recorder across modules is exactly the cross-test coupling the suite
    avoids, and the helper is six lines.
    """
    commands: list[list[str]] = []

    def runner(command: list[str], *, cwd: Path, timeout_seconds: int) -> RunResult:
        commands.append(command)
        return responses.pop(0)

    return runner, commands


# ---------------------------------------------------------------------------
# self_deploy (issue #613)
# ---------------------------------------------------------------------------


def test_self_deploy_dry_run_runs_no_mutating_command(tmp_path: Path) -> None:
    """A previewed self-deploy issues only read-only git commands."""
    runner, commands = _make_fake_runner(
        [
            RunResult(0, "abc123\n", ""),  # rev-parse HEAD
            RunResult(0, "abc123\n", ""),  # rev-parse origin/main
        ]
    )

    result = self_deploy(tmp_path, run_command=runner, dry_run=True)

    assert result.ok is True
    assert result.pulled is False
    assert result.changed is False
    assert result.synced is False
    assert commands == [
        ["git", "rev-parse", "HEAD"],
        ["git", "rev-parse", "origin/main"],
    ]
    # Stated explicitly rather than relying on the equality above, because these
    # are the two commands whose absence is the whole point of the fix.
    assert not any("pull" in command for command in commands)
    assert not any(command[0] == "uv" for command in commands)


def test_self_deploy_dry_run_reports_the_pending_fast_forward(tmp_path: Path) -> None:
    """Previewing is useful, not merely safe: it names the FF and the dependency sync."""
    runner, _commands = _make_fake_runner(
        [
            RunResult(0, "aaaaaaaaaaaa1\n", ""),  # HEAD
            RunResult(0, "bbbbbbbbbbbb2\n", ""),  # origin/main (ahead)
            RunResult(0, "pyproject.toml\nsrc/foo.py\n", ""),  # diff
        ]
    )

    result = self_deploy(tmp_path, run_command=runner, dry_run=True)

    assert result.ok is True
    assert result.pulled is False
    assert result.from_sha == "aaaaaaaaaaaa1"
    assert result.to_sha == "bbbbbbbbbbbb2"
    assert "would fast-forward" in result.message
    assert "uv sync" in result.message


def test_self_deploy_dry_run_reports_when_nothing_is_pending(tmp_path: Path) -> None:
    """HEAD already at the last-known origin/main reports no pending fast-forward."""
    runner, _commands = _make_fake_runner(
        [
            RunResult(0, "same111\n", ""),
            RunResult(0, "same111\n", ""),
        ]
    )

    result = self_deploy(tmp_path, run_command=runner, dry_run=True)

    assert result.ok is True
    assert "no fast-forward pending" in result.message
    assert "uv sync" not in result.message


def test_self_deploy_dry_run_does_not_touch_the_venv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate must sit above ``_check_venv``, which repairs the ``.pth`` in place.

    ``_check_venv`` rewrites the editable ``.pth`` as a side effect of *checking* it,
    so a gate placed after it would still mutate the venv on a preview. Pinning the
    ordering here because it is invisible from the call site.
    """

    def boom(_repo_root: Path) -> Any:
        raise AssertionError("_check_venv must not run under dry_run")

    monkeypatch.setattr("charlie_work.supervise._check_venv", boom)
    runner, _commands = _make_fake_runner(
        [
            RunResult(0, "abc\n", ""),
            RunResult(0, "abc\n", ""),
        ]
    )

    result = self_deploy(tmp_path, run_command=runner, dry_run=True)

    # self_deploy funnels exceptions into ok=False, so a reached _check_venv would
    # surface here as a crash result rather than a test error.
    assert result.ok is True, result.error


def test_self_deploy_without_dry_run_still_pulls(tmp_path: Path) -> None:
    """The other direction: a real run must still fast-forward.

    Without this, gating the preview could silently disable self-deploy entirely and
    the suite would still be green.
    """
    runner, commands = _make_fake_runner(
        [
            RunResult(0, "abc123\n", ""),  # before HEAD
            RunResult(0, "", ""),  # pull
            RunResult(0, "abc123\n", ""),  # after HEAD (unchanged)
        ]
    )

    result = self_deploy(tmp_path, run_command=runner, dry_run=False)

    assert result.pulled is True
    assert ["git", "pull", "--ff-only", "origin", "main"] in commands


def test_self_deploy_dry_run_marks_the_result_as_previewed(tmp_path: Path) -> None:
    """The preview must be *reported*, not merely safe.

    Both callers print ``message`` only on a notable outcome, and the pre-existing
    conditions for notable were ``synced`` and ``venv_repaired`` -- both False for a
    preview. Without a flag of its own the preview ran completely silently, so an
    operator would see no deploy line and conclude the step did nothing.
    """
    runner, _commands = _make_fake_runner(
        [
            RunResult(0, "abc\n", ""),
            RunResult(0, "abc\n", ""),
        ]
    )

    result = self_deploy(tmp_path, run_command=runner, dry_run=True)

    assert result.previewed is True
    assert result.message, "a preview with no message prints an empty line"


def test_self_deploy_real_run_is_not_marked_previewed(tmp_path: Path) -> None:
    """A real deploy must never claim to have been a preview."""
    runner, _commands = _make_fake_runner(
        [
            RunResult(0, "abc123\n", ""),
            RunResult(0, "", ""),
            RunResult(0, "abc123\n", ""),
        ]
    )

    result = self_deploy(tmp_path, run_command=runner, dry_run=False)

    assert result.previewed is False


@pytest.mark.parametrize(
    ("responses", "label"),
    [
        ([RunResult(1, "", "fatal: not a git repository")], "HEAD"),
        (
            [RunResult(0, "abc123\n", ""), RunResult(1, "", "fatal: bad revision")],
            "origin/main",
        ),
    ],
)
def test_self_deploy_preview_read_failure_is_not_alertable(
    tmp_path: Path, responses: list[RunResult], label: str
) -> None:
    """A preview that cannot read a ref stays a preview, and raises no alert.

    Both failure returns are genuinely reachable under ``--dry-run``: the preview
    deliberately does not fetch, so an absent or stale ``origin/main`` tracking ref
    fails ``rev-parse`` on a checkout where the real deploy would have succeeded.

    Without ``previewed=True`` on these returns, the callers' ``if not deploy.ok:``
    arm fires and appends an ERROR entry to the notify sink -- a real, persistent
    write performed by a dry run, and one an operator cannot tell apart from a
    genuine self-deploy failure.
    """
    runner, _commands = _make_fake_runner(responses)

    result = self_deploy(tmp_path, run_command=runner, dry_run=True)

    assert result.ok is False, f"the {label} read was supposed to fail"
    assert result.previewed is True, "a preview failure is still a preview"
    assert result.alertable is False, "a preview must never raise a durable alert"


def test_real_self_deploy_failure_is_alertable() -> None:
    """The positive half: a genuine failure must still reach the operator.

    Without this, ``alertable`` could be hardcoded ``False`` -- silencing every real
    self-deploy alarm -- and the negative tests above would all still pass.
    """
    genuine = SelfDeployResult(
        ok=False,
        pulled=False,
        changed=False,
        synced=False,
        error="fatal: Not possible to fast-forward, aborting.",
        previewed=False,
    )

    assert genuine.alertable is True


def test_successful_preview_is_not_alertable() -> None:
    """A preview that worked is not an alert either -- ``ok`` alone is not the test."""
    assert (
        SelfDeployResult(
            ok=True, pulled=False, changed=False, synced=False, previewed=True
        ).alertable
        is False
    )


# ---------------------------------------------------------------------------
# run_cross_family_review (issue #613)
#
# This one is the sharpest member of the class: the bug was *inside* the dry-run
# branch, not at the call sites. The branch bailed out via ``_fail``, which mkdirs
# and write_text()s an "(UNAVAILABLE)" stub -- so previewing destroyed a real report.
# A call site that correctly threaded ``dry_run`` was therefore still destructive,
# which is why the AST guard alone could never have caught this.
# ---------------------------------------------------------------------------


def _cross_family_kwargs(tmp_path: Path) -> dict[str, Any]:
    """Minimal valid kwargs for run_cross_family_review."""
    return {
        "model": "some-model",
        "command": ["some-cli", "--prompt", "{prompt_path}"],
        "repo_root": tmp_path,
        "prompt_text": "review this",
        "prompt_path": tmp_path / "prompt.md",
        "report_path": tmp_path / "report.md",
        "timeout_seconds": 30,
    }


def _exploding_runner(*_args: Any, **_kwargs: Any) -> Any:
    raise AssertionError("dry_run must not spawn the cross-family subprocess")


def test_cross_family_dry_run_does_not_clobber_an_existing_report(tmp_path: Path) -> None:
    """Previewing must not overwrite a real prior report with a DRY-RUN stub.

    The reports are keyed by PR, so the destroyed file is *the* cross-family review
    for that PR -- there is no second copy. This is the regression that matters most:
    the pre-fix behaviour lost real review output on a command whose entire promise
    is that it changes nothing.
    """
    kwargs = _cross_family_kwargs(tmp_path)
    report_path = kwargs["report_path"]
    real_report = "# Cross-family adversarial review\n\nVERDICT: request_changes\n"
    report_path.write_text(real_report, encoding="utf-8")

    result = run_cross_family_review(**kwargs, dry_run=True, runner=_exploding_runner)

    assert report_path.read_text(encoding="utf-8") == real_report
    assert "DRY-RUN" in (result.error or "")


def test_cross_family_dry_run_creates_no_files(tmp_path: Path) -> None:
    """A preview writes neither the report nor the prompt, and spawns nothing."""
    kwargs = _cross_family_kwargs(tmp_path)

    result = run_cross_family_review(**kwargs, dry_run=True, runner=_exploding_runner)

    assert not kwargs["report_path"].exists()
    # The prompt file is written before the subprocess runs, so it is a separate
    # write the dry-run branch has to short-circuit past.
    assert not kwargs["prompt_path"].exists()
    assert result.ok is False, "the ok=False dry-run contract is relied on by callers"


def _valid_stale_report(head_sha: str) -> str:
    """A report that BOTH ``extract_head_ref_oid`` and ``report_body_is_valid`` accept.

    Three conditions are load-bearing here and none of them is guessable:
    the text must open with the exact orchestrator header (cross_family.py:88), the
    SHA must sit in an HTML comment ``<!-- PR head SHA: ... -->`` rather than a bare
    ``HEAD_REF_OID:`` line (cross_family.py:91), and the body after the ``---``
    separator must carry a severity marker or a Verdict line or the staleness branch
    treats it as "not a real review" and skips.

    Getting any one of them wrong leaves ``old_head_sha`` as None, the guarded block
    unentered, and the tests below passing against unfixed code. That is exactly how
    the first version of this fixture was wrong.
    """
    return (
        "# Cross-family adversarial review — `some-model`\n\n"
        f"<!-- PR head SHA: {head_sha} -->\n\n"
        "---\n\n"
        "**MAJOR** the retry loop drops the last error.\n"
    )


def _osfail_runner(*_args: Any, **_kwargs: Any) -> Any:
    raise OSError("no such binary")


def test_cross_family_staleness_check_tolerates_a_missing_head(tmp_path: Path) -> None:
    """``head_ref_oid`` is optional, and the staleness warning subscripts it.

    ``spec_review`` calls without a PR head at all, so a pre-existing report carrying
    a head SHA drove ``None[:12]`` -> TypeError. That block sits OUTSIDE every ``try``
    in the function — the excepts begin below it and catch only OSError,
    SubprocessError and TimeoutExpired — so pre-fix the TypeError escaped a function
    whose docstring promises it never raises.

    Reaching the runner at all is the proof the staleness block was survived.
    """
    kwargs = _cross_family_kwargs(tmp_path)
    kwargs["report_path"].write_text(_valid_stale_report("abcdef1234567890"), encoding="utf-8")

    result = run_cross_family_review(**kwargs, head_ref_oid=None, runner=_osfail_runner)

    assert result.ok is False
    assert "failed to start" in (result.error or ""), (
        "expected to reach the runner; a different error means the staleness block "
        "diverted us and this test is no longer pinning the None-head path"
    )


def test_cross_family_staleness_still_warns_when_both_heads_are_known(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The positive half of the same guard: a genuinely stale report must still warn.

    The fix added an ``and head_ref_oid`` term to the staleness condition. Without this
    test, tightening that condition to something always-False would keep the negative
    test above green while staleness detection silently stopped working — the failure
    mode where a report reviewed against old code is reused as if current.
    """
    kwargs = _cross_family_kwargs(tmp_path)
    kwargs["report_path"].write_text(_valid_stale_report("aaaaaaaaaaaa1111"), encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        run_cross_family_review(**kwargs, head_ref_oid="bbbbbbbbbbbb2222", runner=_osfail_runner)

    assert "staleness detected" in caplog.text, caplog.text


# ---------------------------------------------------------------------------
# Structural guards: the class, not the instances
# ---------------------------------------------------------------------------


def _call_sites(callee: str) -> list[tuple[str, int, bool]]:
    """Return ``(relpath, lineno, threads_dry_run)`` for every call to ``callee``.

    Derived from the source tree rather than a hand-maintained list, so a new caller
    is covered by construction. Matches both bare (``f(...)``) and attribute
    (``mod.f(...)``) call forms.
    """
    sites: list[tuple[str, int, bool]] = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute):
                name = func.attr
            elif isinstance(func, ast.Name):
                name = func.id
            else:
                continue
            if name != callee:
                continue
            threads = any(keyword.arg == "dry_run" for keyword in node.keywords)
            sites.append((path.name, node.lineno, threads))
    return sites


@pytest.mark.parametrize("callee", DRY_RUN_REQUIRED_CALLEES)
def test_every_production_call_site_threads_dry_run(callee: str) -> None:
    """Every production caller of a state-mutating helper must pass ``dry_run``.

    ``observe_runner_pool`` is the reason this guard exists in this shape: it took no
    ``dry_run`` parameter at all, so no caller *could* gate its pool-sample writes.
    Adding the parameter fixed the instances; this test is what keeps the next caller
    from reintroducing them.
    """
    sites = _call_sites(callee)

    # A guard that finds nothing must fail rather than pass vacuously — if the
    # function is renamed, this test should break loudly, not go quiet.
    assert sites, f"no call sites found for {callee!r}; the guard would pass vacuously"

    offenders = [f"{name}:{lineno}" for name, lineno, threads in sites if not threads]
    assert not offenders, (
        f"{callee} called without dry_run at: {', '.join(offenders)}. "
        "These paths mutate state outside the process, so a preview that reaches them "
        "is not a preview. Thread the flag through."
    )
