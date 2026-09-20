"""Cross-repo gate: shared-prefix shorthand citation lists (issue #1761).

Ground truth (from the issue, confirmed live against the escalated issues):
job-cannon issues were escalated to ``agent:operator-queue`` with
``escalation_reason: cross_repo_target`` solely because their Evidence
bullets used the shared-prefix shorthand convention — a comma-separated run
of backtick spans that spells the directory out once and then repeats only
the remainder, prefixed with ``/`` or ``...``:

    `logs/nightly_monitor/.../.credentials.json`, `/.claude.json`, `/sessions/30392.json`
    `logs/nightly_monitor/.../checkpoint_A.json:1`, `.../checkpoint_B.json:1`

The shorthand items extracted as their own literal candidates — ``/.claude.json``,
``/sessions/30392.json``, ``.../checkpoint_B.json`` — which are absent by
construction (no file is literally named with a leading ``/`` or ``...``) and
survived every neutralization arm (not evidence markers, not write
destinations, not citation-section headings, not gitignored as literal
strings).

The fix classifies them neutral the way the issue prefers: an all-dots first
segment (``..``/``...``) is neutral by construction, and a leading-separator
item that continues a comma/whitespace-separated backtick-span run is
resolved against the nearest preceding non-shorthand span's directory
prefix and classified as the resolved path — so shorthand behaves exactly
as if the full path had been spelled out, while a standalone ``/abs/path``
or a continuation whose resolved form is genuinely missing still escalates.

Uses a real ``git init``-ed ``tmp_path`` repo whose ``.gitignore`` ignores
``logs/`` — the same rule job-cannon carries for its nightly-monitor runtime
artifacts — so the resolved shorthand paths classify through the real
``git check-ignore`` path rather than a stub.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from charlie_work.cross_repo_gate import cross_repo_gate, extract_referenced_paths


def _git(repo: Path, *args: str) -> None:
    """Run a git command in *repo*, raising on failure."""
    subprocess.run(
        ["git", *args],
        cwd=str(repo),
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _init_git_repo_with_logs_gitignore(repo: Path) -> Path:
    """A real ``git init``-ed repo whose ``.gitignore`` ignores ``logs/`` —
    the job-cannon runtime-artifact layout from the issue's evidence."""
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init")
    (repo / ".gitignore").write_text("logs/\n", encoding="utf-8")
    return repo


# job-cannon#1953's original Evidence bullet, verbatim per the issue body.
# ``/sessions/30392.<hash>.key`` drops at extraction as a placeholder
# segment; the bug was ``/.claude.json`` and ``/sessions/30392.json``
# extracting as their own missing candidates.
_1953_HEAD = "logs/nightly_monitor/2026-08-26/claude_config_3aroezxr/.credentials.json"
_1953_BULLET = (
    f"- `{_1953_HEAD}`, `/.claude.json`, `/sessions/30392.json`, `/sessions/30392.<hash>.key`\n"
)

# job-cannon#2129's original Evidence bullet, verbatim per the issue body.
_2129_HEAD = (
    "logs/nightly_monitor/2026-08-29/checkpoint_ATS-source-URL-promotion_53992_1788003900.json"
)
_2129_BULLET = (
    f"- `{_2129_HEAD}:1`, `.../checkpoint_Company-linkage_53992_1788005215.json:1`, "
    "`.../checkpoint_health_53992_1787987768.json:1`\n"
)


def test_jc_1953_shorthand_evidence_bullet_abstains(tmp_path: Path) -> None:
    """The #1953 fixture: ``/``-prefixed continuation items resolve against
    the run's spelled-out directory prefix, land under the gitignored
    ``logs/`` runtime-artifact dir, and classify neutral — the gate abstains
    instead of escalating ``cross_repo_target``."""
    repo = _init_git_repo_with_logs_gitignore(tmp_path / "repo")

    result = cross_repo_gate(_1953_BULLET, repo)

    assert result.passed is True
    assert result.referenced_paths == ()
    assert result.missing_paths == ()
    assert set(result.neutral_paths) == {
        _1953_HEAD,
        "/.claude.json",
        "/sessions/30392.json",
    }
    assert "abstaining" in result.reason


def test_jc_2129_ellipsis_shorthand_evidence_bullet_abstains(tmp_path: Path) -> None:
    """The #2129 fixture: ``...``-prefixed continuation items are neutral by
    construction (an all-dots first segment never names a real file), so the
    gate abstains instead of escalating."""
    repo = _init_git_repo_with_logs_gitignore(tmp_path / "repo")

    result = cross_repo_gate(_2129_BULLET, repo)

    assert result.passed is True
    assert result.referenced_paths == ()
    assert result.missing_paths == ()
    assert set(result.neutral_paths) == {
        _2129_HEAD,
        ".../checkpoint_Company-linkage_53992_1788005215.json",
        ".../checkpoint_health_53992_1787987768.json",
    }
    assert "abstaining" in result.reason


def test_shorthand_items_still_extract_then_classify() -> None:
    """Shorthand items remain extraction candidates — they are binned at
    classification, not hidden from ``extract_referenced_paths`` — so
    ``neutral_paths`` reporting can name what was cited."""
    paths = extract_referenced_paths(_1953_BULLET)
    assert "/.claude.json" in paths
    assert "/sessions/30392.json" in paths


def test_standalone_leading_slash_path_still_escalates(tmp_path: Path) -> None:
    """A leading-``/`` candidate NOT continuing a backtick-span run is a
    genuine absolute path, not shorthand — a missing one still escalates
    exactly as before (no weakening of cross-repo detection)."""
    repo = _init_git_repo_with_logs_gitignore(tmp_path / "repo")
    body = "The file is at `/sessions/30392.json` on the runner."

    result = cross_repo_gate(body, repo)

    assert result.passed is False
    assert result.referenced_paths == ("/sessions/30392.json",)
    assert result.missing_paths == ("/sessions/30392.json",)
    assert result.neutral_paths == ()
    assert "cross_repo_target" in result.reason


def test_absolute_path_continuation_with_missing_resolved_still_escalates(
    tmp_path: Path,
) -> None:
    """A leading-``/`` continuation whose resolved form is genuinely missing
    still escalates: `` `src/a.py`, `/home/operator/other-repo/x.py` ``
    resolves to ``src/home/operator/other-repo/x.py`` (absent), so the raw
    candidate survives and reports missing. This is the guard that the run
    detection does not blanket-neutralize every ``/``-starting path that
    happens to follow a backtick span."""
    repo = _init_git_repo_with_logs_gitignore(tmp_path / "repo")
    (repo / "src").mkdir()
    body = "Fix `src/missing_a.py` and `/home/operator/other-repo/missing_b.py`."

    result = cross_repo_gate(body, repo)

    assert result.passed is False
    assert set(result.missing_paths) == {
        "src/missing_a.py",
        "/home/operator/other-repo/missing_b.py",
    }
    assert result.neutral_paths == ()
    assert "cross_repo_target" in result.reason


def test_shorthand_resolving_to_existing_file_counts_as_evidence(
    tmp_path: Path,
) -> None:
    """`` `src/a.py`, `/b.py` `` with ``src/b.py`` present: the shorthand
    resolves to a real in-repo file, so it counts as pass evidence exactly
    as if ``src/b.py`` had been spelled out — reported under its raw cited
    form in ``referenced_paths``."""
    repo = _init_git_repo_with_logs_gitignore(tmp_path / "repo")
    (repo / "src").mkdir()
    (repo / "src" / "a.py").write_text("# a\n", encoding="utf-8")
    (repo / "src" / "b.py").write_text("# b\n", encoding="utf-8")
    body = "The change touches `src/a.py`, `/b.py`."

    result = cross_repo_gate(body, repo)

    assert result.passed is True
    assert set(result.referenced_paths) == {"src/a.py", "/b.py"}
    assert result.missing_paths == ()


def test_shorthand_resolving_to_missing_repo_shaped_path_still_escalates(
    tmp_path: Path,
) -> None:
    """`` `src/a.py`, `/b.py` `` with both resolved files missing: the
    shorthand item behaves as the spelled-out ``src/b.py`` would — missing
    and repo-shaped — so the gate escalates rather than hiding it."""
    repo = _init_git_repo_with_logs_gitignore(tmp_path / "repo")
    (repo / "src").mkdir()
    body = "The change touches `src/missing_a.py`, `/missing_b.py`."

    result = cross_repo_gate(body, repo)

    assert result.passed is False
    assert set(result.missing_paths) == {"src/missing_a.py", "/missing_b.py"}
    assert result.neutral_paths == ()


def test_shorthand_run_with_real_missing_path_still_escalates(
    tmp_path: Path,
) -> None:
    """A genuine missing dispatch target in the same body still escalates:
    the shorthand item bins neutral (resolved under gitignored ``logs/``)
    and the fully-qualified ``src/missing.py`` survives — the fix narrows
    ``missing_paths`` to exactly the real target."""
    repo = _init_git_repo_with_logs_gitignore(tmp_path / "repo")
    (repo / "src").mkdir()
    body = (
        "Artifacts: `logs/nightly_monitor/run/a.json`, `/b.json`. The bug is in `src/missing.py`."
    )

    result = cross_repo_gate(body, repo)

    assert result.passed is False
    assert result.missing_paths == ("src/missing.py",)
    assert "/b.json" in result.neutral_paths
    assert "cross_repo_target" in result.reason


def test_newline_separated_run_also_resolves(tmp_path: Path) -> None:
    """The run separator set includes newlines, not just commas — the same
    convention written one-item-per-line resolves the same way."""
    repo = _init_git_repo_with_logs_gitignore(tmp_path / "repo")
    body = "- `logs/nightly_monitor/run/a.json`\n  `/b.json`\n"

    result = cross_repo_gate(body, repo)

    assert result.passed is True
    assert "/b.json" in result.neutral_paths


def test_ellipsis_shorthand_standalone_is_neutral(tmp_path: Path) -> None:
    """A ``...``-first-segment candidate is neutral by construction even
    with no preceding span — ``.../x`` is never a real path, standalone or
    in a list, so no run context is required."""
    repo = _init_git_repo_with_logs_gitignore(tmp_path / "repo")
    body = "The output landed at `.../checkpoint_health.json:1`."

    result = cross_repo_gate(body, repo)

    assert result.passed is True
    assert result.neutral_paths == (".../checkpoint_health.json",)
    assert result.referenced_paths == ()


def test_prose_separated_slash_path_is_not_a_continuation(tmp_path: Path) -> None:
    """Words between backtick spans end the run: `` `a/b.json`. Later,
    `/etc/x.py` `` cites ``/etc/x.py`` on its own terms — a genuine absolute
    path that still escalates when missing."""
    repo = _init_git_repo_with_logs_gitignore(tmp_path / "repo")
    body = "`logs/run/a.json`. Later, `/etc/x.py` was implicated."

    result = cross_repo_gate(body, repo)

    assert result.passed is False
    assert result.missing_paths == ("/etc/x.py",)
    assert result.neutral_paths == ("logs/run/a.json",)
