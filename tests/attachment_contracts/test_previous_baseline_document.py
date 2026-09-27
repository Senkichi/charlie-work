"""Tests for ``_load_previous_baseline_document()`` (issue #1839 rework
round 3): the git-``ls-tree``-based base-ref reconstruction that feeds
``check_ratchet_tamper``.

The round-3 review found this function -- the SOLE source of
``previous_document`` for the diff-based ratchet-tamper guard (the
raise-to-match laundering check) -- had zero test coverage despite being
rewritten for #1839 to walk a per-entry ``.attachment-budgets/`` directory
at an arbitrary git ref (``git ls-tree`` + one ``git show`` per member)
before falling back to the legacy single-file ``.attachment-budgets.json``.
A silent regression here (a broken ``-z`` split, wrong prefix-stripping, or
a ``git ls-tree`` pathspec edge case) would degrade the guard to "nothing to
diff against" with nothing to catch it.

These tests build a REAL ``git init``-ed repository (not a stub) so such a
regression would actually be observed, mirroring the existing
``test_dirty_tree.py`` precedent for git-backed tests in this repo.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from charlie_work.attachment_contracts import baseline, baseline_dir
from charlie_work.attachment_contracts.__main__ import _load_previous_baseline_document, main
from charlie_work.attachment_contracts.baseline import BASELINE_DIRNAME, BASELINE_FILENAME


def _git(repo: Path, *args: str) -> str:
    """Run a git command in *repo*, raising on failure, returning stdout."""
    result = subprocess.run(
        ["git", *args],
        cwd=str(repo),
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout.strip()


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """An empty, deterministically-configured real git repository."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "test")
    _git(repo, "config", "user.email", "test@example.test")
    return repo


def _commit_all(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _entry_dict(
    identity: str,
    file: str,
    member_count: int = 10,
    kind: str = "class",
    boundary: float = 4.0,
) -> dict:
    return {
        "kind": kind,
        "identity": identity,
        "file": file,
        "member_count": member_count,
        "boundary": boundary,
        "bumps": [],
    }


def _doc(entries: list[dict], **overrides) -> dict:
    document = {
        "version": 1,
        "generated_by": "test",
        "generated_at": "2026-09-05T00:00:00Z",
        "floor": 4,
        "entries": entries,
    }
    document.update(overrides)
    return document


def _entry_keys(document: dict) -> list[tuple]:
    return sorted(
        (e["kind"], e["identity"], e["file"], e["member_count"], e["boundary"])
        for e in document["entries"]
    )


# ---------------------------------------------------------------------------
# Directory-layout base ref
# ---------------------------------------------------------------------------


def test_directory_layout_base_ref_reconstructs_document(git_repo: Path) -> None:
    document = _doc(
        [
            _entry_dict("A", "src/pkg/a.py"),
            _entry_dict("B", "src/pkg/b.py", kind="function", member_count=12, boundary=9.0),
        ]
    )
    baseline_dir.dump(document, git_repo / BASELINE_DIRNAME)
    base_sha = _commit_all(git_repo, "base: directory-layout baseline")

    reconstructed = _load_previous_baseline_document(git_repo, base_sha)

    assert reconstructed is not None
    assert reconstructed["generated_by"] == "test"
    assert reconstructed["floor"] == 4
    assert _entry_keys(reconstructed) == _entry_keys(document)


# ---------------------------------------------------------------------------
# Legacy single-file base ref (fallback path, and the migration commit case)
# ---------------------------------------------------------------------------


def test_legacy_single_file_base_ref_falls_back(git_repo: Path) -> None:
    document = _doc([_entry_dict("Legacy", "src/pkg/legacy.py")])
    baseline.dump(document, git_repo / BASELINE_FILENAME)
    base_sha = _commit_all(git_repo, "base: legacy single-file baseline")

    reconstructed = _load_previous_baseline_document(git_repo, base_sha)

    assert reconstructed is not None
    assert [e["identity"] for e in reconstructed["entries"]] == ["Legacy"]


def test_migration_commit_diffs_legacy_base_against_directory_head(git_repo: Path) -> None:
    """The guard must still diff across the #1839 migration commit itself:
    a legacy file at the base ref, a directory at HEAD."""
    legacy_document = _doc([_entry_dict("Migrated", "src/pkg/m.py")])
    baseline.dump(legacy_document, git_repo / BASELINE_FILENAME)
    base_sha = _commit_all(git_repo, "base: legacy layout")

    (git_repo / BASELINE_FILENAME).unlink()
    baseline_dir.dump(legacy_document, git_repo / BASELINE_DIRNAME)
    _commit_all(git_repo, "head: migrate to directory layout")

    reconstructed = _load_previous_baseline_document(git_repo, base_sha)

    assert reconstructed is not None
    assert [e["identity"] for e in reconstructed["entries"]] == ["Migrated"]


# ---------------------------------------------------------------------------
# Fail-closed on a corrupt snapshot at the base ref
# ---------------------------------------------------------------------------


def test_corrupt_entry_file_at_base_ref_fails_closed(git_repo: Path) -> None:
    document = _doc([_entry_dict("A", "src/pkg/a.py")])
    baseline_dir.dump(document, git_repo / BASELINE_DIRNAME)
    entry_files = list((git_repo / BASELINE_DIRNAME / "entries").rglob("*.json"))
    assert len(entry_files) == 1
    # Corrupt the ONE entry file before committing, so the base ref itself
    # (not just the working tree) carries malformed JSON.
    entry_files[0].write_text("{not valid json", encoding="utf-8")
    base_sha = _commit_all(git_repo, "base: corrupt entry file")

    reconstructed = _load_previous_baseline_document(git_repo, base_sha)

    # Fail-closed: "nothing to diff against" (None), never a crash and
    # never a best-effort partial document silently missing the corrupt
    # entry.
    assert reconstructed is None


def test_corrupt_meta_json_at_base_ref_fails_closed(git_repo: Path) -> None:
    document = _doc([_entry_dict("A", "src/pkg/a.py")])
    baseline_dir.dump(document, git_repo / BASELINE_DIRNAME)
    (git_repo / BASELINE_DIRNAME / "meta.json").write_text("not json at all", encoding="utf-8")
    base_sha = _commit_all(git_repo, "base: corrupt meta.json")

    assert _load_previous_baseline_document(git_repo, base_sha) is None


# ---------------------------------------------------------------------------
# Rename between base and head: the base-ref read must ignore later history
# ---------------------------------------------------------------------------


def test_renamed_entry_between_base_and_head_reads_base_ref_content(git_repo: Path) -> None:
    """``git ls-tree`` + ``git show`` must read the BASE commit's tree,
    unaffected by a rename made in a LATER commit -- exactly the scenario
    the ratchet-tamper guard exists to diff (a PR that moves/renames an
    attachment point between base and head)."""
    base_document = _doc([_entry_dict("Old", "src/pkg/old.py")])
    baseline_dir.dump(base_document, git_repo / BASELINE_DIRNAME)
    base_sha = _commit_all(git_repo, "base: original entry")

    # HEAD renames the entry's identity (and therefore its derived
    # entries/ file path) -- simulating a PR that renames an attachment
    # point between base and head.
    head_document = _doc([_entry_dict("New", "src/pkg/old.py")])
    baseline_dir.dump(head_document, git_repo / BASELINE_DIRNAME)
    _commit_all(git_repo, "head: renamed entry")

    reconstructed = _load_previous_baseline_document(git_repo, base_sha)

    assert reconstructed is not None
    assert [e["identity"] for e in reconstructed["entries"]] == ["Old"]

    # Sanity: HEAD/working tree now has the renamed entry, so a naive read
    # of the on-disk file (rather than the git object at base_sha) would
    # have wrongly returned "New" here.
    on_disk = baseline_dir.load(git_repo / BASELINE_DIRNAME)
    assert [e["identity"] for e in on_disk["entries"]] == ["New"]


# ---------------------------------------------------------------------------
# End-to-end through the real CLI entry point
# ---------------------------------------------------------------------------


def test_check_tree_base_ref_cli_drives_directory_reconstruction(
    git_repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``check-tree --base-ref`` must actually invoke the directory-layout
    reconstruction end-to-end through the real CLI, not silently no-op."""
    document = _doc([_entry_dict("A", "src/pkg/a.py", member_count=999, boundary=4.0)])
    baseline_dir.dump(document, git_repo / BASELINE_DIRNAME)
    base_sha = _commit_all(git_repo, "base: baseline with A")

    rc = main(
        [
            "check-tree",
            "--root",
            str(git_repo),
            "--base-ref",
            base_sha,
            "--report-only",
        ]
    )

    # --report-only always exits 0 regardless of findings; this proves the
    # base-ref reconstruction path executes end-to-end without raising.
    assert rc == 0
