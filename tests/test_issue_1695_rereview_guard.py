"""Issue #1695: ``charlie why-charlie-hate`` must not silently discard a
still-valid recorded verdict.

``run_command`` dispatched the subcommand straight into ``app.review()``,
which unconditionally rebuilds the review packet: a terminal verdict
pinned to a superseded head is voided back to a ``pending`` stub
(destroying the carry-forward baseline ``reviewed_patch_id`` /
``reviewed_changed_lines`` / ``reviewed_changed_files``), the PR's state
entry flips to ``reviewing``, and the ``review_started`` label transition
fires. When the verdict is still valid -- pinned to the live head, or
carrying forward to it via ``_check_carry_forward`` -- that burns a
reviewer session on unchanged content and delays a merge the queue would
otherwise carry.

These tests pin the CLI-boundary guard: a bare ``why-charlie-hate`` call
refuses (non-zero exit, the reason on the first AND last output line, the
decision file byte-identical, state untouched, no label transition), an
explicit ``--force-rereview`` opts into the old behavior while archiving
the voided verdict first, and ``app.review()`` itself stays ungated for
the loop's internal callers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from _cli_fixtures import _make_repo
from _fakes_github import FakeGitHub
from _review_fixtures import _round_archive_app, _write_review_packet
from charlie_work import cli
from charlie_work.janitor import _calculate_patch_id, _diff_content_signature


_PR_NUMBER = 456
_LIVE_HEAD = "sha-abc123"  # FakeGitHub's default headRefOid for PR #456
_STATE_REL = Path(".var") / "charlie-work" / "state.json"


def _cli_repo_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, FakeGitHub]:
    """Wire ``cli.main`` to the full FakeGitHub for one test."""
    fake_gh = FakeGitHub()
    monkeypatch.setattr(cli, "GitHub", lambda *a, **k: fake_gh)
    return _make_repo(tmp_path), fake_gh


def _run_cli(repo: Path, *extra: str) -> int:
    return cli.main(["--repo", str(repo), "why-charlie-hate", "--pr", str(_PR_NUMBER), *extra])


def _decision_path(repo: Path) -> Path:
    return repo / ".var" / "charlie-work" / "prs" / f"pr-{_PR_NUMBER}" / "review-decision.json"


def _seed_decision(repo: Path, decision: dict[str, Any]) -> bytes:
    """Write ``decision`` as the flat review-decision.json; return its bytes."""
    pr_dir = repo / ".var" / "charlie-work" / "prs" / f"pr-{_PR_NUMBER}"
    pr_dir.mkdir(parents=True, exist_ok=True)
    path = pr_dir / "review-decision.json"
    path.write_text(json.dumps(decision), encoding="utf-8")
    return path.read_bytes()


def _assert_nothing_mutated(
    repo: Path,
    fake_gh: FakeGitHub,
    decision_before: bytes,
    state_before: bytes,
) -> None:
    """The refusal must be mutation-free: no decision rewrite, no state
    write, no label transition attempt."""
    assert _decision_path(repo).read_bytes() == decision_before
    assert (repo / _STATE_REL).read_bytes() == state_before
    assert fake_gh.labels_added == []
    assert fake_gh.labels_removed == []
    assert fake_gh.pr_labels_added == []


def test_cli_rereview_refuses_when_approved_verdict_pinned_to_live_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Approved at the live head: a bare why-charlie-hate must refuse --
    re-reviewing unchanged content would flip status to ``reviewing`` and
    fire ``review_started`` for a verdict that is still valid."""
    repo, fake_gh = _cli_repo_app(tmp_path, monkeypatch)
    decision_before = _seed_decision(
        repo,
        {
            "decision": "approved",
            "reviewed_head_sha": _LIVE_HEAD,
            "verdict_provenance": "fresh_llm_review",
        },
    )
    state_before = (repo / _STATE_REL).read_bytes()

    rc = _run_cli(repo)

    assert rc == 1
    lines = capsys.readouterr().out.strip().splitlines()
    # The refusal reason must survive `head -1` AND `tail -1`.
    assert "refus" in lines[0]
    assert "still valid" in lines[0]
    assert "refus" in lines[-1]
    _assert_nothing_mutated(repo, fake_gh, decision_before, state_before)


def test_cli_rereview_refuses_when_carry_forward_succeeds_patch_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Tier 1 (issue #412): head moved but the live diff's stable patch-id
    equals the recorded ``reviewed_patch_id`` -- the approval still covers
    the content, so regenerating must be refused, not voided."""
    diff_text = (
        "diff --git a/file b/file\n"
        "index 123..456 100644\n"
        "--- a/file\n"
        "+++ b/file\n"
        "@@ -1,3 +1,4 @@\n"
        " line1\n"
        " line2\n"
        "+line3\n"
        " line4\n"
    )
    patch_id = _calculate_patch_id(diff_text)
    old_head = "sha-superseded"
    new_head = "sha-rebased"
    repo, fake_gh = _cli_repo_app(tmp_path, monkeypatch)
    fake_gh.diffs[_PR_NUMBER] = diff_text
    fake_gh.pr_head_shas[_PR_NUMBER] = new_head
    decision_before = _seed_decision(
        repo,
        {
            "decision": "approved",
            "reviewed_head_sha": old_head,
            "reviewed_patch_id": patch_id,
            "carried_forward_from": [],
            "verdict_provenance": "fresh_llm_review",
        },
    )
    state_before = (repo / _STATE_REL).read_bytes()

    rc = _run_cli(repo)

    assert rc == 1
    lines = capsys.readouterr().out.strip().splitlines()
    assert "refus" in lines[0]
    assert "patch-id" in lines[0]
    assert "refus" in lines[-1]
    _assert_nothing_mutated(repo, fake_gh, decision_before, state_before)


def test_cli_rereview_refuses_when_carry_forward_succeeds_line_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Tier 2 (issue #414): patch-ids differ because the merge-base moved
    with main, but the ordered +/- line stream and changed-file set are
    identical -- the approval still covers the content."""
    # Same fixture shape as test_charlie_work's tier-2 carry-forward test:
    # main advanced and rewrote a CONTEXT line, so patch-id drifts while
    # the PR's own changed lines stay byte-identical.
    reviewed_diff = (
        "diff --git a/file b/file\n"
        "--- a/file\n"
        "+++ b/file\n"
        "@@ -1,3 +1,4 @@\n"
        " alpha\n"
        " beta\n"
        "+gamma\n"
        " delta\n"
    )
    live_diff = (
        "diff --git a/file b/file\n"
        "--- a/file\n"
        "+++ b/file\n"
        "@@ -1,3 +1,4 @@\n"
        " alpha\n"
        " beta-updated\n"
        "+gamma\n"
        " delta\n"
    )
    reviewed_patch_id = _calculate_patch_id(reviewed_diff)
    live_patch_id = _calculate_patch_id(live_diff)
    assert reviewed_patch_id != live_patch_id, "fixture must reproduce patch-id drift"
    reviewed_signature = _diff_content_signature(reviewed_diff)
    assert reviewed_signature == _diff_content_signature(live_diff)

    repo, fake_gh = _cli_repo_app(tmp_path, monkeypatch)
    fake_gh.diffs[_PR_NUMBER] = live_diff
    fake_gh.pr_head_shas[_PR_NUMBER] = "sha-after-main-advance"
    decision_before = _seed_decision(
        repo,
        {
            "decision": "approved",
            "reviewed_head_sha": "sha-old-head",
            "reviewed_patch_id": reviewed_patch_id,
            "reviewed_changed_lines": list(reviewed_signature.changed_lines),
            "reviewed_changed_files": sorted(reviewed_signature.changed_files),
            "carried_forward_from": [],
            "verdict_provenance": "fresh_llm_review",
        },
    )
    state_before = (repo / _STATE_REL).read_bytes()

    rc = _run_cli(repo)

    assert rc == 1
    lines = capsys.readouterr().out.strip().splitlines()
    assert "refus" in lines[0]
    assert "line-content" in lines[0]
    assert "refus" in lines[-1]
    _assert_nothing_mutated(repo, fake_gh, decision_before, state_before)


def test_cli_rereview_force_regenerates_packet_and_archives_prior_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--force-rereview opts into discarding the verdict: the packet is
    regenerated and the flat decision is voided to pending as before, but
    the voided verdict must land in the rounds archive FIRST -- a
    carried-forward re-pin lives only in the flat file
    (``_update_approval_head`` deliberately writes flat-only), so
    overwriting it without archiving loses the only copy."""
    repo, fake_gh = _cli_repo_app(tmp_path, monkeypatch)
    fake_gh.pr_head_shas[_PR_NUMBER] = "sha-moved-again"
    pr_dir = repo / ".var" / "charlie-work" / "prs" / f"pr-{_PR_NUMBER}"
    # Round 1: the original reviewer verdict, archived by record_review.
    round_one_dir = pr_dir / "rounds" / "round-1"
    round_one_dir.mkdir(parents=True)
    (round_one_dir / "review-decision.json").write_text(
        json.dumps(
            {
                "decision": "approved",
                "summary": "looks good",
                "reviewed_head_sha": "sha-original",
                "verdict_provenance": "fresh_llm_review",
            }
        ),
        encoding="utf-8",
    )
    # Flat file: the carried-forward re-pin (flat-only write, never
    # archived -- the exact gap the issue describes).
    _seed_decision(
        repo,
        {
            "decision": "approved",
            "summary": "looks good",
            "reviewed_head_sha": "sha-carried",
            "carried_forward_from": ["sha-original"],
            "verdict_provenance": "carried_forward",
        },
    )

    rc = _run_cli(repo, "--force-rereview")

    assert rc == 0
    # The stale-head void still fires: the flat decision is the pending stub.
    flat = json.loads(_decision_path(repo).read_text(encoding="utf-8"))
    assert flat["decision"] == "pending"
    # The voided verdict is preserved in the rounds archive -- a NEW round,
    # because its reviewed_head_sha differs from round-1's on the compare
    # keys, not a silent overwrite of round-1.
    round_two = json.loads(
        (pr_dir / "rounds" / "round-2" / "review-decision.json").read_text(encoding="utf-8")
    )
    assert round_two["decision"] == "approved"
    assert round_two["reviewed_head_sha"] == "sha-carried"


def test_cli_rereview_proceeds_when_no_verdict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No recorded decision at all: existing behavior is preserved --
    the packet is generated and the pending stub written."""
    repo, _fake_gh = _cli_repo_app(tmp_path, monkeypatch)

    rc = _run_cli(repo)

    assert rc == 0
    flat = json.loads(_decision_path(repo).read_text(encoding="utf-8"))
    assert flat["decision"] == "pending"


def test_cli_rereview_proceeds_when_decision_pending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``pending`` placeholder is not a recorded verdict: the guard must
    not fire and the packet regenerates as before."""
    repo, _fake_gh = _cli_repo_app(tmp_path, monkeypatch)
    _seed_decision(repo, {"decision": "pending", "reviewed_head_sha": _LIVE_HEAD})

    rc = _run_cli(repo)

    assert rc == 0


def test_cli_rereview_proceeds_when_verdict_genuinely_stale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Head moved AND carry-forward fails: the verdict is genuinely stale
    and the pre-existing void path runs unchanged (decision -> pending)."""
    repo, fake_gh = _cli_repo_app(tmp_path, monkeypatch)
    fake_gh.pr_head_shas[_PR_NUMBER] = "sha-genuinely-new-content"
    fake_gh.diffs[_PR_NUMBER] = (
        "diff --git a/file b/file\n"
        "--- a/file\n"
        "+++ b/file\n"
        "@@ -1,2 +1,3 @@\n"
        " alpha\n"
        "+entirely-new-line\n"
        " delta\n"
    )
    _seed_decision(
        repo,
        {
            "decision": "approved",
            "reviewed_head_sha": "sha-superseded",
            "reviewed_patch_id": "patch-id-that-does-not-match-live",
            "verdict_provenance": "fresh_llm_review",
        },
    )

    rc = _run_cli(repo)

    assert rc == 0
    flat = json.loads(_decision_path(repo).read_text(encoding="utf-8"))
    assert flat["decision"] == "pending"


def test_review_method_direct_call_is_not_gated(tmp_path: Path) -> None:
    """The loop's internal callers reach ``app.review()`` directly and must
    stay ungated: a carry-forwardable verdict at a superseded head is still
    voided by a direct call exactly as before -- the refusal exists only at
    the CLI boundary."""
    app, _paths = _round_archive_app(tmp_path)
    diff_text = (
        "diff --git a/file b/file\n"
        "index 123..456 100644\n"
        "--- a/file\n"
        "+++ b/file\n"
        "@@ -1,3 +1,4 @@\n"
        " line1\n"
        " line2\n"
        "+line3\n"
        " line4\n"
    )
    app.gh.diffs[_PR_NUMBER] = diff_text
    app.gh.pr_head_shas[_PR_NUMBER] = "sha-rebased"
    pr_dir = _write_review_packet(
        tmp_path,
        _PR_NUMBER,
        "sha-rebased",
        {
            "decision": "approved",
            "reviewed_head_sha": "sha-superseded",
            "reviewed_patch_id": _calculate_patch_id(diff_text),
            "carried_forward_from": [],
            "verdict_provenance": "fresh_llm_review",
        },
    )

    result = app.review(_PR_NUMBER)

    assert result.ok is True
    flat = json.loads((pr_dir / "review-decision.json").read_text(encoding="utf-8"))
    assert flat["decision"] == "pending"


def test_run_command_guard_fires_for_namespace_without_force_attr(
    tmp_path: Path,
) -> None:
    """``run_command`` is also invoked with hand-built Namespaces (tests and
    other callers that never went through argparse). A Namespace lacking
    ``force_rereview`` must default to the guarded path, not crash or
    silently bypass the guard."""
    app, _paths = _round_archive_app(tmp_path)
    _write_review_packet(
        tmp_path,
        _PR_NUMBER,
        _LIVE_HEAD,
        {
            "decision": "approved",
            "reviewed_head_sha": _LIVE_HEAD,
            "verdict_provenance": "fresh_llm_review",
        },
    )

    result = cli.run_command(app, argparse.Namespace(command="why-charlie-hate", pr=_PR_NUMBER))

    assert result.ok is False
    assert "refus" in result.message


def test_rereview_help_documents_the_mutation_and_force_flag(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``charlie why-charlie-hate --help`` must say the command writes
    review state (regenerates the packet, resets review state) -- it is not
    a read-only diagnostic despite the name -- and must document the
    --force-rereview opt-out."""
    parser = cli.build_parser()

    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["why-charlie-hate", "--help"])

    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--force-rereview" in out
    assert "reset" in out
    assert "review" in out

    args = parser.parse_args(["why-charlie-hate", "--pr", str(_PR_NUMBER)])
    assert args.force_rereview is False
    args = parser.parse_args(["why-charlie-hate", "--pr", str(_PR_NUMBER), "--force-rereview"])
    assert args.force_rereview is True
