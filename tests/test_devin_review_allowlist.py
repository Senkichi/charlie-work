"""Issue #2011: read-only exec allow-list for headless devin-shell reviewers,
and distinct classification of an exec-rejected session."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from charlie_work import devin_shell
from charlie_work.devin_shell import _REVIEW_EXEC_ALLOWLIST, launch_devin_session
from charlie_work.verdict_parsing import (
    REVIEW_MISS_LAUNCH_FAILED,
    REVIEW_SESSION_FAILED_HEADING,
    REVIEW_SESSION_SUMMARY_HEADING,
    _extract_review_session_summary,
    _extract_terminating_cause,
    body_has_crash_signature,
)
from charlie_work.worktree import WorktreeInfo

REJECTION_LINE = "warning: rejected a tool call that requires confirmation (exec)"


def _patch_launch(
    monkeypatch: pytest.MonkeyPatch, checkouts: dict[str, Path]
) -> list[dict[str, Any]]:
    popen_calls: list[dict[str, Any]] = []

    def fake_create_review_checkout(
        repo_root: Path, pr: int, sha: str, *, reviews_dir: Path
    ) -> WorktreeInfo:
        path = reviews_dir / f"pr-{pr}"
        path.mkdir(parents=True, exist_ok=True)
        checkouts["review"] = path
        return WorktreeInfo(path=path, branch=sha, venv_junction=None)

    def fake_create_worktree(repo_root: Path, branch: str, **kwargs: Any) -> WorktreeInfo:
        path = repo_root.parent / "worker-wt"
        path.mkdir(parents=True, exist_ok=True)
        checkouts["worker"] = path
        return WorktreeInfo(path=path, branch=branch, venv_junction=None)

    def fake_popen_worker(args: Any, **kwargs: Any) -> Any:
        popen_calls.append({"argv": list(args), "cwd": kwargs.get("cwd")})
        return SimpleNamespace(pid=4242)

    monkeypatch.setattr(devin_shell, "create_review_checkout", fake_create_review_checkout)
    monkeypatch.setattr(devin_shell, "create_worktree", fake_create_worktree)
    monkeypatch.setattr(devin_shell, "popen_worker", fake_popen_worker)
    return popen_calls


def _launch(tmp_path: Path, *, review: bool) -> Any:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(exist_ok=True)
    prompt = tmp_path / "prompt.md"
    prompt.write_text("go\n", encoding="utf-8")
    kwargs: dict[str, Any] = {
        "repo_root": repo_root,
        "sessions_dir": tmp_path / "sessions",
    }
    if review:
        kwargs.update(review=True, head_sha="a" * 40)
    else:
        kwargs.update(worktrees_dir=tmp_path / "wts")
    return launch_devin_session(7, "agent/issue-7", prompt, **kwargs)


def test_review_launch_writes_allowlist_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkouts: dict[str, Path] = {}
    _patch_launch(monkeypatch, checkouts)
    record = _launch(tmp_path, review=True)

    assert record.error is None
    cfg = checkouts["review"] / ".devin" / "config.local.json"
    assert json.loads(cfg.read_text(encoding="utf-8")) == {
        "permissions": {"allow": list(_REVIEW_EXEC_ALLOWLIST)}
    }
    # Atomic temp-file + replace: no leftover temp file.
    assert not cfg.with_suffix(cfg.suffix + ".tmp").exists()


def test_review_allowlist_written_before_popen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkouts: dict[str, Path] = {}
    seen: list[bool] = []

    def fake_create(repo_root: Path, pr: int, sha: str, *, reviews_dir: Path) -> WorktreeInfo:
        path = reviews_dir / f"pr-{pr}"
        path.mkdir(parents=True, exist_ok=True)
        checkouts["review"] = path
        return WorktreeInfo(path=path, branch=sha, venv_junction=None)

    def fake_popen(args: Any, **kwargs: Any) -> Any:
        seen.append((Path(kwargs["cwd"]) / ".devin" / "config.local.json").is_file())
        return SimpleNamespace(pid=1)

    monkeypatch.setattr(devin_shell, "create_review_checkout", fake_create)
    monkeypatch.setattr(devin_shell, "popen_worker", fake_popen)
    _launch(tmp_path, review=True)
    assert seen == [True]


def test_worker_launch_writes_no_allowlist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkouts: dict[str, Path] = {}
    _patch_launch(monkeypatch, checkouts)
    record = _launch(tmp_path, review=False)

    assert record.error is None
    assert not (checkouts["worker"] / ".devin").exists()


def test_allowlist_write_failure_does_not_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    checkouts: dict[str, Path] = {}
    popen_calls = _patch_launch(monkeypatch, checkouts)

    def boom(path: Path, value: Any) -> None:
        if path.name == "config.local.json":
            raise OSError("disk full")
        real_write_json(path, value)

    real_write_json = devin_shell._write_json
    monkeypatch.setattr(devin_shell, "_write_json", boom)

    with caplog.at_level("WARNING"):
        record = _launch(tmp_path, review=True)

    assert record.error is None
    assert record.pid == 4242
    assert len(popen_calls) == 1
    assert "exec allow-list" in caplog.text


_FORBIDDEN = re.compile(
    r"\b(rm|push|commit|checkout|reset|python\d?|uv|node|bash|sh|pwsh|find|sed|awk"
    r"|rg|sort|uniq|api|grep -O|git grep|git diff|git log|git show)\b"
)

# Exact read-only gh subcommands; anything else under ``gh`` is forbidden
# (``gh api`` can write, ``gh pr merge``/``edit``/``comment`` mutate).
_ALLOWED_GH = {
    "Exec(gh issue view)",
    "Exec(gh pr view)",
    "Exec(gh pr diff)",
    "Exec(gh pr checks)",
}


def test_allowlist_entries_are_readonly_exec_rules() -> None:
    assert _REVIEW_EXEC_ALLOWLIST
    for entry in _REVIEW_EXEC_ALLOWLIST:
        assert entry.startswith("Exec(") and entry.endswith(")"), entry
        if entry.startswith("Exec(gh"):
            assert entry in _ALLOWED_GH, entry
            continue
        assert not _FORBIDDEN.search(entry), entry
    for bare in ("Exec(git)", "Exec(gh)", "Exec(git branch)", "Exec(gh pr)", "Exec(gh issue)"):
        assert bare not in _REVIEW_EXEC_ALLOWLIST
    assert len(set(_REVIEW_EXEC_ALLOWLIST)) == len(_REVIEW_EXEC_ALLOWLIST)


def test_allowlist_covers_the_commands_that_ended_2011_sessions() -> None:
    # Regression anchor: these were the prompted-then-rejected commands that
    # ended the missed #2011 reviews (Devin sessions.db, 2026-09-29).
    for rule in ("Exec(gh issue view)", "Exec(gh pr view)"):
        assert rule in _REVIEW_EXEC_ALLOWLIST


def test_sanitizer_still_strips_dangerous() -> None:
    out = devin_shell._sanitize_review_command_template(
        ("devin", "--permission-mode", "dangerous", "--print")
    )
    assert "dangerous" not in out and "--permission-mode" not in out


# --- classification --------------------------------------------------------


def test_cause_exec_rejected(tmp_path: Path) -> None:
    log = tmp_path / "review.log"
    log.write_text(f"some output\n{REJECTION_LINE}\n", encoding="utf-8")
    cause = _extract_terminating_cause(log, log)
    assert cause["cause"] == "reviewer_exec_rejected"


def test_cause_exec_rejected_wins_over_exit_code(tmp_path: Path) -> None:
    log = tmp_path / "review.log"
    log.write_text(REJECTION_LINE + "\n", encoding="utf-8")
    cause = _extract_terminating_cause(log, log, exit_code=1)
    assert cause["cause"] == "reviewer_exec_rejected"


def test_cause_stream_cut_without_signature(tmp_path: Path) -> None:
    log = tmp_path / "review.log"
    log.write_text(json.dumps({"type": "system", "subtype": "x"}) + "\n", encoding="utf-8")
    cause = _extract_terminating_cause(log, log)
    assert cause["cause"] == "stream_cut_no_result_event"


def test_exec_rejected_summary_is_truthful(tmp_path: Path) -> None:
    log = tmp_path / "review.log"
    log.write_text(REJECTION_LINE + "\n", encoding="utf-8")
    outcome = _extract_review_session_summary(tmp_path / "missing.jsonl", log, max_turns=0)
    assert outcome is not None
    # reason unchanged (feeds the rollback guard); comment + cause are truthful.
    assert outcome.reason == REVIEW_MISS_LAUNCH_FAILED
    assert outcome.terminating_cause["cause"] == "reviewer_exec_rejected"
    assert REVIEW_SESSION_FAILED_HEADING not in outcome.text
    assert outcome.text.startswith(REVIEW_SESSION_SUMMARY_HEADING)
    assert "failed to start" not in outcome.text
    assert body_has_crash_signature(outcome.text)


def test_plain_launch_failure_keeps_failed_heading(tmp_path: Path) -> None:
    log = tmp_path / "review.log"
    log.write_text("error: bad argv\n", encoding="utf-8")
    outcome = _extract_review_session_summary(tmp_path / "missing.jsonl", log, max_turns=0)
    assert outcome is not None
    assert outcome.text.startswith(REVIEW_SESSION_FAILED_HEADING)


# --- issue #2024: the prompt states the allow-list ---------------------------


def _argv_prompt(popen_calls: list[dict[str, Any]]) -> Path:
    argv = popen_calls[0]["argv"]
    return Path(argv[argv.index("--prompt-file") + 1])


def test_review_prompt_states_every_allowlisted_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    popen_calls = _patch_launch(monkeypatch, {})
    record = _launch(tmp_path, review=True)

    assert record.error is None
    launched = _argv_prompt(popen_calls)
    assert launched != tmp_path / "prompt.md"
    text = launched.read_text(encoding="utf-8")
    assert text.startswith("go\n")
    assert devin_shell.REVIEW_EXEC_SECTION_HEADING in text
    for entry in _REVIEW_EXEC_ALLOWLIST:
        assert f"`{entry[len('Exec(') : -1]}`" in text, entry
    assert "pytest" in text and "Do NOT run tests" in text
    # The shared packet prompt other harnesses read is untouched.
    assert (tmp_path / "prompt.md").read_text(encoding="utf-8") == "go\n"


def test_worker_prompt_has_no_review_exec_section(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    popen_calls = _patch_launch(monkeypatch, {})
    _launch(tmp_path, review=False)

    launched = _argv_prompt(popen_calls)
    assert launched == tmp_path / "prompt.md"
    assert devin_shell.REVIEW_EXEC_SECTION_HEADING not in launched.read_text(encoding="utf-8")


def test_review_prompt_write_failure_falls_back_to_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    popen_calls = _patch_launch(monkeypatch, {})
    real_write_text = Path.write_text

    def boom(self: Path, *args: Any, **kwargs: Any) -> int:
        if ".devin.md" in self.name:
            raise OSError("disk full")
        return real_write_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", boom)
    with caplog.at_level("WARNING"):
        record = _launch(tmp_path, review=True)

    assert record.error is None
    assert _argv_prompt(popen_calls) == tmp_path / "prompt.md"
    assert "exec section" in caplog.text
