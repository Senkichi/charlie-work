"""Tests for scripts/backfill_stale_rework_briefs.py (F6).

Loads the script as a module without adding scripts/ to sys.path, mirroring
tests/test_heartbeat_check.py's pattern for the other standalone script.

The core requirement (from docs/plans/rework-findings-channel.md section 6):
a test that imports the REAL ``_is_verdict_newer_than_brief`` from
workflow.py (not a reimplementation) and proves the script's mtime bump
flips that gate from False to True. See test_bump_flips_real_gate below --
it is deliberately built on the same import workflow.py's own callers use,
so it can never silently drift from what dispatch_rework actually checks.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

import _git_templates
from _script_loader import load_script_module
from _worktree_fixtures import _init_repo
from charlie_work.workflow import _is_verdict_newer_than_brief


def _load_backfill_script() -> ModuleType:
    path = Path(__file__).parent.parent / "scripts" / "backfill_stale_rework_briefs.py"
    return load_script_module(path, "backfill_stale_rework_briefs")


@pytest.fixture(scope="module")
def bf() -> ModuleType:
    return _load_backfill_script()


@pytest.fixture(scope="module", autouse=True)
def _warm_plain_git_template(
    tmp_path_factory: pytest.TempPathFactory,
) -> _git_templates.EnvKey | None:
    """HS-CW-4: charge the ``plain`` git-template cold build to module setup.

    ``test_repo_builders_go_through_the_git_template`` is this module's pin
    for the repo builders materializing from the per-process ``plain`` git
    template, and the deployment-gate tests below it call the same builders
    (``tiny_git_repo`` / ``_make_repo_with_fix`` / ``_make_repo_without_fix``).
    Whichever test lands first on its xdist worker pays the template BUILD --
    five ``git`` spawns whose cost swells from ~0.15 s idle to 1.3-5 s on
    loaded self-hosted Windows runners -- and once the worker's first
    ``plain`` user is the pin test itself, the ledger sees the whole build
    inside the pin test's call. The ledger's samples are therefore bimodal
    (~0.25 s warm, 1.3-5.1 s cold) with the cold share stochastic, and the
    alert over a 10-sample window fires on the share rather than on any
    repo change (issue #2577).

    The order dependence is the root cause and the build is shared per
    process, so the fix charges it to module setup: the idiom
    ``test_charlie_work_review_claims.py`` already established for the
    same ledger mechanism on its flagged test. The build runs with the
    ``GIT_CONFIG_COUNT/KEY_0/VALUE_0`` trio that ``conftest._isolate_git_env``
    injects into every test (applied through its own ``MonkeyPatch``
    context so the builder's spawned ``git`` sees it), because
    ``env_fingerprint()`` keys the registry by ``GIT_*`` env and the
    per-test fixtures that set the trio are NOT active at module setup:
    a warm-up without the trio builds a template under a fingerprint the
    call phase never looks up. The trio's values are constants shared
    with the conftest injection, so the local env match is provably safe.
    Returns the fingerprint it warmed under (``None`` without the kill
    switch) for the phase pin below.
    """
    if not _git_templates.reuse_enabled():
        return None
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("GIT_CONFIG_COUNT", "1")
        mp.setenv("GIT_CONFIG_KEY_0", "core.longpaths")
        mp.setenv("GIT_CONFIG_VALUE_0", "true")
        _init_repo(tmp_path_factory.mktemp("plain-template-warmup") / "repo")
        return _git_templates.env_fingerprint()


def _write_pr(
    prs_root: Path,
    pr_number: int,
    *,
    decision: str = "request_changes",
    brief: str | None = "brief content",
    verdict_mtime_ns: int,
    brief_mtime_ns: int | None = None,
) -> tuple[Path, Path | None]:
    pr_dir = prs_root / f"pr-{pr_number}"
    pr_dir.mkdir(parents=True)
    decision_path = pr_dir / "review-decision.json"
    decision_path.write_text(
        f'{{"decision": "{decision}", "pr_number": {pr_number}}}', encoding="utf-8"
    )
    import os

    os.utime(decision_path, ns=(verdict_mtime_ns, verdict_mtime_ns))

    brief_path: Path | None = None
    if brief is not None:
        brief_path = pr_dir / "rework-prompt.md"
        brief_path.write_text(brief, encoding="utf-8")
        assert brief_mtime_ns is not None
        os.utime(brief_path, ns=(brief_mtime_ns, brief_mtime_ns))
    return decision_path, brief_path


class FakeGitHub:
    """Stub for the GitHub surface derive_entries() needs: .pr_view(number)."""

    def __init__(self, open_prs: set[int]) -> None:
        self.open_prs = open_prs
        self.calls: list[int] = []

    def pr_view(self, number: int) -> dict:
        self.calls.append(number)
        return {"state": "OPEN" if number in self.open_prs else "CLOSED"}


class FailingGitHub:
    def pr_view(self, number: int):
        from charlie_work.github import GitHubError

        raise GitHubError("gh: rate limited")


# ---------------------------------------------------------------------------
# The required core test: real gate flips False -> True via the script's own
# bump function, using workflow.py's real _is_verdict_newer_than_brief.
# ---------------------------------------------------------------------------


def test_bump_flips_real_gate(tmp_path: Path, bf: ModuleType) -> None:
    base_ns = 1_700_000_000_000_000_000
    decision_path, brief_path = _write_pr(
        tmp_path,
        1,
        verdict_mtime_ns=base_ns,
        brief_mtime_ns=base_ns + 5_000_000,  # brief newer -> will-not-regenerate
    )
    assert brief_path is not None

    # Precondition: brief mtime >= verdict mtime, so dispatch_rework would
    # reuse the brief verbatim -- this is the real function, not a copy.
    assert _is_verdict_newer_than_brief(decision_path, brief_path) is False

    candidate = bf.Candidate(
        pr_number=1,
        decision_path=decision_path,
        brief_path=brief_path,
        verdict_mtime_ns=decision_path.stat().st_mtime_ns,
        brief_mtime_ns=brief_path.stat().st_mtime_ns,
        decision="request_changes",
    )
    bf.bump_verdict_mtime(candidate)

    # Postcondition, checked with the exact same real function.
    assert _is_verdict_newer_than_brief(decision_path, brief_path) is True


def test_bump_does_not_touch_brief_content_or_mtime(tmp_path: Path, bf: ModuleType) -> None:
    base_ns = 1_700_000_000_000_000_000
    decision_path, brief_path = _write_pr(
        tmp_path, 1, verdict_mtime_ns=base_ns, brief_mtime_ns=base_ns
    )
    assert brief_path is not None
    before_brief_mtime = brief_path.stat().st_mtime_ns
    before_brief_text = brief_path.read_text(encoding="utf-8")
    before_decision_text = decision_path.read_text(encoding="utf-8")

    candidate = bf.Candidate(
        pr_number=1,
        decision_path=decision_path,
        brief_path=brief_path,
        verdict_mtime_ns=decision_path.stat().st_mtime_ns,
        brief_mtime_ns=brief_path.stat().st_mtime_ns,
        decision="request_changes",
    )
    bf.bump_verdict_mtime(candidate)

    assert brief_path.stat().st_mtime_ns == before_brief_mtime
    assert brief_path.read_text(encoding="utf-8") == before_brief_text
    # Content of the verdict (the evidentiary record) must never be rewritten
    # -- only its mtime changes.
    assert decision_path.read_text(encoding="utf-8") == before_decision_text


# ---------------------------------------------------------------------------
# derive_entries: buckets + open-state cross section
# ---------------------------------------------------------------------------


def test_derive_entries_buckets_and_funnel(tmp_path: Path, bf: ModuleType) -> None:
    prs_root = tmp_path / "prs"
    base = 1_700_000_000_000_000_000

    # PR 1: request_changes, brief absent -> brief_absent bucket, open.
    # NTFS mtime resolution is 100ns; use a multi-millisecond margin so the
    # bucket split can't collapse into "equal" on round-trip through the
    # filesystem (mirrors _BUMP_MARGIN_NS's own rationale in the script).
    margin = 5_000_000
    _write_pr(prs_root, 1, decision="request_changes", brief=None, verdict_mtime_ns=base)
    # PR 2: request_changes, verdict newer -> will_self_heal, open.
    _write_pr(
        prs_root,
        2,
        decision="request_changes",
        verdict_mtime_ns=base + margin,
        brief_mtime_ns=base,
    )
    # PR 3: request_changes, brief mtime >= verdict -> will_not_regenerate, open.
    _write_pr(
        prs_root,
        3,
        decision="request_changes",
        verdict_mtime_ns=base,
        brief_mtime_ns=base + margin,
    )
    # PR 4: same bucket as 3 but CLOSED on GitHub -> must not become a candidate.
    _write_pr(
        prs_root,
        4,
        decision="request_changes",
        verdict_mtime_ns=base,
        brief_mtime_ns=base + margin,
    )
    # PR 5: approved decision, will_not_regenerate bucket -> must not appear in
    # any request_changes cross-section count.
    _write_pr(
        prs_root, 5, decision="approved", verdict_mtime_ns=base, brief_mtime_ns=base + margin
    )

    gh = FakeGitHub(open_prs={1, 2, 3})  # 4 is deliberately absent -> CLOSED

    entries, counts = bf.derive_entries(prs_root, gh)

    assert counts.verdict_dirs_total == 5
    assert counts.unreadable_decision == 0
    assert counts.brief_absent == 1
    assert counts.will_self_heal == 1
    assert counts.will_not_regenerate == 3  # PRs 3, 4, 5

    assert counts.request_changes_total == 4  # PRs 1-4 (5 is "approved")
    assert counts.request_changes_brief_absent_total == 1
    assert counts.request_changes_brief_absent_open == 1
    assert counts.request_changes_will_self_heal_total == 1
    assert counts.request_changes_will_self_heal_open == 1
    assert counts.request_changes_will_not_regenerate_total == 2  # PRs 3, 4
    assert counts.request_changes_will_not_regenerate_open == 1  # PR 3 only

    # gh.pr_view is only ever called for request_changes verdicts (bounded
    # cost) -- PR 5 (approved) must never trigger a network call.
    assert 5 not in gh.calls
    assert set(gh.calls) == {1, 2, 3, 4}

    candidates, excluded_present = bf.select_candidates(entries, exclude=set())
    assert [c.pr_number for c in candidates] == [3]
    assert excluded_present == []


def test_derive_entries_gh_failure_is_not_treated_as_closed(
    tmp_path: Path, bf: ModuleType
) -> None:
    prs_root = tmp_path / "prs"
    base = 1_700_000_000_000_000_000
    _write_pr(
        prs_root, 9, decision="request_changes", verdict_mtime_ns=base, brief_mtime_ns=base + 10
    )

    entries, counts = bf.derive_entries(prs_root, FailingGitHub())

    assert len(entries) == 1
    assert entries[0].is_open is None
    # A gh lookup failure must never silently count as "open" (would wrongly
    # add it as a candidate) or as counted in the "_open" tallies at all.
    assert counts.request_changes_will_not_regenerate_open == 0
    candidates, _ = bf.select_candidates(entries, exclude=set())
    assert candidates == []


def test_select_candidates_respects_exclude(tmp_path: Path, bf: ModuleType) -> None:
    prs_root = tmp_path / "prs"
    base = 1_700_000_000_000_000_000
    _write_pr(
        prs_root, 10, decision="request_changes", verdict_mtime_ns=base, brief_mtime_ns=base + 10
    )
    _write_pr(
        prs_root, 11, decision="request_changes", verdict_mtime_ns=base, brief_mtime_ns=base + 10
    )
    gh = FakeGitHub(open_prs={10, 11})
    entries, _ = bf.derive_entries(prs_root, gh)

    candidates, excluded_present = bf.select_candidates(entries, exclude={11, 999})
    assert [c.pr_number for c in candidates] == [10]
    # Only the exclude entries that were actually present get reported --
    # 999 wasn't in the pool at all, so it's not claimed as "excluded".
    assert excluded_present == [11]


def test_parse_exclude(bf: ModuleType) -> None:
    assert bf._parse_exclude(None) == set()
    assert bf._parse_exclude("") == set()
    assert bf._parse_exclude("696") == {696}
    assert bf._parse_exclude("696,700, 683") == {696, 700, 683}
    assert bf._parse_exclude("696,696") == {696}


# ---------------------------------------------------------------------------
# Issue #1269 (W12): _decision_has_crash_signature + --crash-signature-only.
# ---------------------------------------------------------------------------

_CRASH_BODY = "## Reviewer session summary (no verdict produced)\n\nNo verdict was produced."


def test_decision_has_crash_signature_new_shape_external_findings(bf: ModuleType) -> None:
    decision = {"decision": "request_changes", "external_findings": [_CRASH_BODY]}
    assert bf._decision_has_crash_signature(decision) is True


def test_decision_has_crash_signature_old_shape_external_channel(bf: ModuleType) -> None:
    decision = {
        "decision": "request_changes",
        "findings_channel": "external",
        "required_changes": [_CRASH_BODY],
    }
    assert bf._decision_has_crash_signature(decision) is True


def test_decision_has_crash_signature_old_shape_non_external_channel_ignored(
    bf: ModuleType,
) -> None:
    # required_changes is only ever known to carry merged-in external comment
    # bodies when findings_channel == "external" -- mirrors the render
    # guard's own condition. A non-external channel (or missing channel)
    # must not be scanned, matching _render_required_changes_section.
    decision = {
        "decision": "request_changes",
        "required_changes": [_CRASH_BODY],
    }
    assert bf._decision_has_crash_signature(decision) is False


def test_decision_has_crash_signature_false_for_genuine_findings(bf: ModuleType) -> None:
    decision = {
        "decision": "request_changes",
        "findings_channel": "external",
        "required_changes": ["the migration script drops the index without a guard"],
        "external_findings": ["the migration script drops the index without a guard"],
    }
    assert bf._decision_has_crash_signature(decision) is False


def _write_pr_json(
    prs_root: Path,
    pr_number: int,
    decision_json: dict,
    *,
    verdict_mtime_ns: int,
    brief_mtime_ns: int,
    brief: str = "brief content",
) -> tuple[Path, Path]:
    """Like _write_pr, but with full control over the decision JSON body so
    findings_channel/required_changes/external_findings can be set (_write_pr
    only ever writes {"decision": ..., "pr_number": ...})."""
    import os

    pr_dir = prs_root / f"pr-{pr_number}"
    pr_dir.mkdir(parents=True)
    decision_path = pr_dir / "review-decision.json"
    decision_path.write_text(json.dumps(decision_json), encoding="utf-8")
    os.utime(decision_path, ns=(verdict_mtime_ns, verdict_mtime_ns))

    brief_path = pr_dir / "rework-prompt.md"
    brief_path.write_text(brief, encoding="utf-8")
    os.utime(brief_path, ns=(brief_mtime_ns, brief_mtime_ns))
    return decision_path, brief_path


def test_select_candidates_crash_signature_only_narrows_pool(
    tmp_path: Path, bf: ModuleType
) -> None:
    prs_root = tmp_path / "prs"
    base = 1_700_000_000_000_000_000
    margin = 5_000_000  # will-not-regenerate: brief mtime >= verdict mtime

    # PR 30: crash-signature poisoned, will-not-regenerate, open.
    _write_pr_json(
        prs_root,
        30,
        {
            "decision": "request_changes",
            "pr_number": 30,
            "external_findings": [_CRASH_BODY],
        },
        verdict_mtime_ns=base,
        brief_mtime_ns=base + margin,
    )
    # PR 31: genuine findings only, same bucket/open state.
    _write_pr_json(
        prs_root,
        31,
        {
            "decision": "request_changes",
            "pr_number": 31,
            "external_findings": ["a genuine, non-crash finding"],
        },
        verdict_mtime_ns=base,
        brief_mtime_ns=base + margin,
    )
    gh = FakeGitHub(open_prs={30, 31})
    entries, _ = bf.derive_entries(prs_root, gh)

    # Default: crash_signature_only=False is a strict no-op -- both PRs
    # selected, identical to the pre-W12 behavior.
    default_candidates, _ = bf.select_candidates(entries, exclude=set())
    assert {c.pr_number for c in default_candidates} == {30, 31}

    # Opt-in narrowing: only the crash-poisoned PR survives.
    narrowed_candidates, _ = bf.select_candidates(
        entries, exclude=set(), crash_signature_only=True
    )
    assert [c.pr_number for c in narrowed_candidates] == [30]


# ---------------------------------------------------------------------------
# Deployment gate
# ---------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


@pytest.fixture
def tiny_git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    _init_repo(repo)
    return repo


def test_deployment_gate_fails_when_fix_not_merged(tiny_git_repo: Path, bf: ModuleType) -> None:
    # Fix commit lives on a branch that is never merged into the checked-out
    # branch's HEAD.
    _git(tiny_git_repo, "checkout", "-q", "-b", "fix-branch")
    (tiny_git_repo / "b.txt").write_text("b", encoding="utf-8")
    _git(tiny_git_repo, "add", "b.txt")
    _git(tiny_git_repo, "commit", "-q", "-m", "the fix")
    fix_sha = _git(tiny_git_repo, "rev-parse", "HEAD").stdout.strip()
    _git(tiny_git_repo, "checkout", "-q", "-")  # back to original branch, fix not merged

    ok, failures = bf.check_deployment_gate(tiny_git_repo, [fix_sha])
    assert ok is False
    assert len(failures) == 1
    assert fix_sha in failures[0]


def test_deployment_gate_passes_once_fix_is_ancestor(tiny_git_repo: Path, bf: ModuleType) -> None:
    _git(tiny_git_repo, "checkout", "-q", "-b", "fix-branch")
    (tiny_git_repo / "b.txt").write_text("b", encoding="utf-8")
    _git(tiny_git_repo, "add", "b.txt")
    _git(tiny_git_repo, "commit", "-q", "-m", "the fix")
    fix_sha = _git(tiny_git_repo, "rev-parse", "HEAD").stdout.strip()

    ok, failures = bf.check_deployment_gate(tiny_git_repo, [fix_sha])
    assert ok is True
    assert failures == []


def test_deployment_gate_requires_all_of_multiple_commits(
    tiny_git_repo: Path, bf: ModuleType
) -> None:
    _git(tiny_git_repo, "checkout", "-q", "-b", "fix-branch")
    (tiny_git_repo / "b.txt").write_text("b", encoding="utf-8")
    _git(tiny_git_repo, "add", "b.txt")
    _git(tiny_git_repo, "commit", "-q", "-m", "fix one")
    fix_one = _git(tiny_git_repo, "rev-parse", "HEAD").stdout.strip()

    unrelated_sha = "0" * 40  # well-formed but unreachable -> git reports "not valid"/failure
    ok, failures = bf.check_deployment_gate(tiny_git_repo, [fix_one, unrelated_sha])
    assert ok is False
    assert len(failures) == 1  # only the bad one fails; fix_one is HEAD itself


# ---------------------------------------------------------------------------
# Issue #1332: the deployment gate must anchor to the RENDERER checkout, not
# the state root selected by --repo. In the live fleet topology these are
# different checkouts (the renderer is the daemon deployment; --repo may be a
# different repo's state root, e.g. job-cannon). Two defects this tests:
#   1. cw false-PASS: gate checked --repo's HEAD while the daemon (renderer)
#      sat at an older commit without the fix.
#   2. jc unevaluable: a charlie-work fix SHA cannot resolve in job-cannon's
#      object store (git exits 128).
# ---------------------------------------------------------------------------


def _make_repo_with_fix(tmp_path: Path, name: str) -> tuple[Path, str]:
    """Create a git repo whose HEAD contains a 'fix' commit; return (repo, fix_sha).

    The base repo materializes from the per-process ``plain`` git template
    (``_worktree_fixtures._init_repo`` -> ``tests/_git_templates.py``,
    HS-CW-4: ``shutil.copytree``, no subprocesses); only the per-test fix
    commit itself still spawns git. The fix SHA stays unique to this repo
    because the commit is created after the copy -- no other template copy's
    object store contains it, which is what the renderer-vs-state-repo tests
    rely on."""
    repo = tmp_path / name
    _init_repo(repo)
    (repo / "fix.txt").write_text("fix", encoding="utf-8")
    _git(repo, "add", "fix.txt")
    _git(repo, "commit", "-q", "-m", "the renderer fix")
    fix_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()
    return repo, fix_sha


def _make_repo_without_fix(tmp_path: Path, name: str) -> Path:
    """Create a git repo whose HEAD does NOT contain the fix (simulates a
    state root / different repo that cannot resolve the fix SHA).

    A ``plain`` template copy is sufficient here: the gate only requires a
    real git work tree whose object store lacks the fix SHA, and the fix
    commit is only ever created inside ``_make_repo_with_fix``'s copy."""
    repo = tmp_path / name
    _init_repo(repo)
    return repo


def test_repo_builders_go_through_the_git_template(tmp_path: Path) -> None:
    """Issue #2463 regression pin: the repo builders must materialize through
    the per-process ``plain`` git template (``_git_templates``, HS-CW-4)
    instead of re-running the ``init``/``config``/``commit`` spawn sequence
    per call -- the ledger measured
    ``test_main_gate_fails_when_renderer_lacks_fix_even_if_state_repo_has_it``
    at ~2.5x baseline because each of this module's gate tests paid the full
    spawn cost on the Windows CI host. Mirrors ``test_git_templates``'s #2387
    pin for ``_helpers._init_git_repo`` and the #2425 pin in
    ``test_local_issues_loop_gates``."""
    import _git_templates

    before = _git_templates._REGISTRY.materialized["plain"]
    repo, fix_sha = _make_repo_with_fix(tmp_path, "with-fix")
    assert _git_templates._REGISTRY.materialized["plain"] == before + 1
    # The fix commit is a real object in this copy's store.
    assert _git(repo, "rev-parse", "--verify", fix_sha).stdout.strip()
    without = _make_repo_without_fix(tmp_path, "without-fix")
    assert _git_templates._REGISTRY.materialized["plain"] == before + 2
    assert _git(without, "rev-parse", "--verify", "main").stdout.strip()


@pytest.mark.skipif(
    not _git_templates.reuse_enabled(),
    reason="CI_FLEET_TEST_REUSE=off forces the fresh path; there is no template to pin",
)
def test_plain_git_template_warmed_by_module_setup(
    _warm_plain_git_template: _git_templates.EnvKey | None, tmp_path: Path
) -> None:
    """Pin the module warm-up to the environment the call phase looks up.

    The ``test_charlie_work_review_claims.py`` pin's lesson: a warm-up that
    builds under a different ``env_fingerprint`` than the call phase
    silently buys nothing -- the lookup misses and the cold ``git`` boot
    lands in the call phase anyway. The fixture returns the fingerprint
    it warmed under; comparing it against this test's own call-phase
    fingerprint fails on any drift regardless of what other modules
    warmed in this worker. The probe then drives the real consumer path
    (_init_repo) and proves it copies the warmed template:
    ``materialized["plain"]`` increments while the registry's key set
    stays fixed -- a cold build would add a ``("plain", env)`` key
    instead (issue #2577).
    """
    warmed_env = _warm_plain_git_template
    call_env = _git_templates.env_fingerprint()
    assert warmed_env == call_env, (
        "module warm-up ran under a different env fingerprint than the call "
        "phase; its template can never be looked up"
    )
    registry = _git_templates._REGISTRY
    assert registry.lookup("plain") is not None, (
        "no plain template under the call-phase fingerprint; the warm-up took the fresh path"
    )
    keys_before = set(registry._templates)
    copies_before = registry.materialized["plain"]
    _init_repo(tmp_path / "probe")
    assert registry.materialized["plain"] == copies_before + 1
    assert set(registry._templates) == keys_before, (
        "the call phase built a new plain template instead of copying the warmed one"
    )


def test_deployment_gate_anchors_to_renderer_not_state_repo(
    tmp_path: Path, bf: ModuleType
) -> None:
    """The fix SHA lives in the renderer repo's history but NOT in the state
    repo's object store. Gate against the renderer -> PASS; gate against the
    state repo -> FAIL with git 128 (the cross-repo defect #2)."""
    renderer_repo, fix_sha = _make_repo_with_fix(tmp_path, "renderer")
    state_repo = _make_repo_without_fix(tmp_path, "state")

    # Gate against the renderer checkout: fix is an ancestor of HEAD -> PASS.
    ok, failures = bf.check_deployment_gate(renderer_repo, [fix_sha])
    assert ok is True
    assert failures == []

    # Gate against the state repo: the fix SHA does not exist in its object
    # store -> git exits 128 ("Not a valid commit name"). This is defect #2:
    # the OLD code evaluated against --repo (the state root) and could never
    # satisfy the gate for the jc lane.
    ok, failures = bf.check_deployment_gate(state_repo, [fix_sha])
    assert ok is False
    assert len(failures) == 1
    assert "128" in failures[0]


def test_main_renderer_repo_separates_gate_from_state_repo(
    tmp_path: Path, bf: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """--renderer-repo anchors the gate to the renderer checkout, independent
    of --repo. The state repo lacks the fix SHA entirely (defect #2: git 128
    against --repo), but the gate PASSES because it evaluates against the
    renderer. The OLD code evaluated against --repo and would have FAILED."""
    # Isolate from the real host fleet config layer.
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(tmp_path / "fleet-dir"))
    renderer_repo, fix_sha = _make_repo_with_fix(tmp_path, "renderer")
    state_repo = _make_repo_without_fix(tmp_path, "state")

    rc = bf.main(
        [
            "--repo",
            str(state_repo),
            "--renderer-repo",
            str(renderer_repo),
            "--require-commit",
            fix_sha,
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Deployment gate: PASS" in out
    # The gate message names the renderer checkout, not the state repo.
    assert str(renderer_repo) in out
    assert "Renderer repo (deployment gate anchor)" in out


def test_main_gate_fails_when_renderer_lacks_fix_even_if_state_repo_has_it(
    tmp_path: Path, bf: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Defect #1 (cw false-PASS): --repo's HEAD contains the fix, but the
    renderer checkout does NOT. The gate must FAIL (the renderer is not
    deployed), even though the OLD code would have PASSed against --repo."""
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(tmp_path / "fleet-dir"))
    state_repo, fix_sha = _make_repo_with_fix(tmp_path, "state")
    renderer_repo = _make_repo_without_fix(tmp_path, "renderer")

    rc = bf.main(
        [
            "--repo",
            str(state_repo),
            "--renderer-repo",
            str(renderer_repo),
            "--require-commit",
            fix_sha,
        ]
    )
    # Dry-run returns 0 even on a failing gate (only --apply is refused), but
    # the gate report must say FAIL.
    assert rc == 0
    out = capsys.readouterr().out
    assert "Deployment gate: FAIL" in out


def test_main_apply_refused_when_renderer_lacks_fix(
    tmp_path: Path, bf: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """--apply is refused when the renderer checkout lacks the fix, even
    though --repo has it (defect #1). The OLD code would have allowed the
    apply, regenerating briefs through the pre-fix renderer."""
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(tmp_path / "fleet-dir"))
    state_repo, fix_sha = _make_repo_with_fix(tmp_path, "state")
    renderer_repo = _make_repo_without_fix(tmp_path, "renderer")

    rc = bf.main(
        [
            "--repo",
            str(state_repo),
            "--renderer-repo",
            str(renderer_repo),
            "--require-commit",
            fix_sha,
            "--apply",
        ]
    )
    assert rc == 1
    captured = capsys.readouterr()
    assert "Deployment gate: FAIL" in captured.out
    # ABORT is printed to stderr (file=sys.stderr in main()).
    assert "ABORT" in captured.err


def test_main_renderer_repo_defaults_to_repo(
    tmp_path: Path, bf: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Without --renderer-repo, the gate evaluates against --repo (backward
    compatibility for the single-checkout layout where the state repo IS the
    renderer)."""
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(tmp_path / "fleet-dir"))
    repo, fix_sha = _make_repo_with_fix(tmp_path, "repo")

    rc = bf.main(["--repo", str(repo), "--require-commit", fix_sha])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Deployment gate: PASS" in out
    # No separate renderer-repo line when it equals --repo.
    assert "Renderer repo (deployment gate anchor)" not in out


# ---------------------------------------------------------------------------
# Hash-based post-apply verification (manifest + --check-regenerated)
# ---------------------------------------------------------------------------


def test_manifest_roundtrip_detects_regeneration(tmp_path: Path, bf: ModuleType) -> None:
    prs_root = tmp_path / "prs"
    base = 1_700_000_000_000_000_000
    decision_path, brief_path = _write_pr(
        prs_root,
        20,
        decision="request_changes",
        brief="old renderer output",
        verdict_mtime_ns=base,
        brief_mtime_ns=base + 10,
    )
    assert brief_path is not None
    candidate = bf.Candidate(
        pr_number=20,
        decision_path=decision_path,
        brief_path=brief_path,
        verdict_mtime_ns=base,
        brief_mtime_ns=base + 10,
        decision="request_changes",
    )
    manifest_path = tmp_path / "manifest.json"
    bf.write_verification_manifest(manifest_path, [candidate])
    assert manifest_path.exists()

    # Nothing changed yet -> check_regenerated must report FAILURE (exit 1),
    # never silently "success" -- this is the exact silent-no-op the plan
    # calls out.
    assert bf.check_regenerated(manifest_path) == 1

    # Simulate the fixed renderer actually regenerating the brief.
    brief_path.write_text("new renderer output with real findings", encoding="utf-8")
    assert bf.check_regenerated(manifest_path) == 0


def test_check_regenerated_missing_manifest(tmp_path: Path, bf: ModuleType) -> None:
    assert bf.check_regenerated(tmp_path / "does-not-exist.json") == 1


def test_hash_file_is_sha256(tmp_path: Path, bf: ModuleType) -> None:
    import hashlib

    p = tmp_path / "f.txt"
    p.write_text("hello", encoding="utf-8")
    assert bf._hash_file(p) == hashlib.sha256(b"hello").hexdigest()
