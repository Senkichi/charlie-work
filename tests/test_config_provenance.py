"""Provenance on the resolved config (issue #943).

The defect these cover is not a wrong value -- it is that a *correct* value
carries no record of whether any file produced it. ``load_config()`` with no
path returns pristine dataclass defaults, which is byte-identical to a
fully-configured fleet whose features happen to be switched off. Twice in one
day that ambiguity was read as "the feature is disabled" when the truth was "no
config was loaded at all", and #590 spent hours on the same fork.

So the assertions here are deliberately paired: same values, different
``sources``. A test that only checked ``sources == (path,)`` on a happy path
would pass against an implementation that recorded provenance nobody could use
to break the tie.
"""

from __future__ import annotations

import pathlib
from dataclasses import replace

import pytest

from charlie_work.config import ConfigError, OrchestratorConfig, load_config
from charlie_work.global_config import load_layered_config


def _repo_with_config(root: pathlib.Path, text: str) -> pathlib.Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "orchestrator.config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# load_config
# --------------------------------------------------------------------------


def test_no_path_records_no_sources() -> None:
    assert load_config().sources == ()


def test_absent_file_records_no_sources(tmp_path: pathlib.Path) -> None:
    """A path that does not exist is not a source. ``load_config`` silently
    falls back to defaults for it, so the empty tuple is the only signal that
    the fallback happened."""
    assert load_config(tmp_path / "nope.yaml").sources == ()


def test_read_file_is_recorded(tmp_path: pathlib.Path) -> None:
    path = _repo_with_config(tmp_path, "dispatch:\n  default_limit: 7\n")
    config = load_config(path)
    assert config.dispatch.default_limit == 7
    assert config.sources == (str(path),)


def test_configured_off_is_distinguishable_from_never_loaded(
    tmp_path: pathlib.Path,
) -> None:
    """The actual bug. A file that sets a key to exactly its dataclass default
    yields a config equal to the no-file config -- only ``sources`` separates
    them.

    The written value is derived from the default rather than hard-coded, so
    this keeps testing the ambiguity even if the default flips.
    """
    defaults = OrchestratorConfig()
    literal = "true" if defaults.review_dispatch.enabled else "false"
    path = _repo_with_config(tmp_path, f"review_dispatch:\n  enabled: {literal}\n")

    loaded = load_config(path)

    # Indistinguishable by value...
    assert loaded == defaults
    assert loaded.review_dispatch.enabled == defaults.review_dispatch.enabled
    # ...and that is precisely why provenance has to be carried separately.
    assert loaded.sources == (str(path),)
    assert defaults.sources == ()


def test_empty_file_still_records_a_source(tmp_path: pathlib.Path) -> None:
    """A 0-byte config parses to no sections, so every value is a default --
    but the file *was* read. Collapsing this into ``()`` would recreate the
    #590 fork (truncated file vs. no file) that provenance exists to settle."""
    path = _repo_with_config(tmp_path, "")
    config = load_config(path)
    assert config == OrchestratorConfig()
    assert config.sources == (str(path),)


def test_sources_cannot_be_declared_by_a_config_file(tmp_path: pathlib.Path) -> None:
    """``known_sections`` is derived from the dataclass fields, so a provenance
    field would become a writable YAML key by default -- letting an untrusted
    input forge the one field whose entire value is being trustworthy."""
    path = _repo_with_config(tmp_path, "sources:\n  - /somewhere/else.yaml\n")
    with pytest.raises(ConfigError) as excinfo:
        load_config(path)
    assert "sources" in str(excinfo.value)


def test_provenance_is_excluded_from_the_valid_section_list(
    tmp_path: pathlib.Path,
) -> None:
    """Control for the test above: the error must reject ``sources`` *and* not
    advertise it as valid. Without this, an implementation that listed it as a
    valid section while rejecting it for another reason would still pass."""
    path = _repo_with_config(tmp_path, "bogus_section: {}\n")
    with pytest.raises(ConfigError) as excinfo:
        load_config(path)
    message = str(excinfo.value)
    valid_list = message.split("(valid:", 1)[1]
    assert "sources" not in valid_list
    # positive control: the list is populated, so the absence above is a real
    # exclusion rather than an empty string trivially not containing anything.
    assert "dispatch" in valid_list


def test_provenance_does_not_affect_equality(tmp_path: pathlib.Path) -> None:
    """``compare=False``: provenance describes how a value was obtained, not
    what it is. Config equality checks elsewhere must not start failing because
    two identical configs came from different paths."""
    a = _repo_with_config(tmp_path / "a", "dispatch:\n  default_limit: 3\n")
    b = _repo_with_config(tmp_path / "b", "dispatch:\n  default_limit: 3\n")
    left, right = load_config(a), load_config(b)
    assert left.sources != right.sources
    assert left == right


# --------------------------------------------------------------------------
# load_layered_config
# --------------------------------------------------------------------------


def test_layered_records_both_layers_in_merge_order(tmp_path: pathlib.Path) -> None:
    fleet = tmp_path / "fleet"
    fleet.mkdir()
    global_path = fleet / "config.yaml"
    global_path.write_text("dispatch:\n  default_limit: 9\n", encoding="utf-8")
    repo_root = tmp_path / "repo"
    repo_path = _repo_with_config(repo_root, "review:\n  require_issue_link: true\n")

    config = load_layered_config(repo_root, None, fleet_dir_override=str(fleet))

    # Global first, per-repo last -- the same order as override precedence, so
    # the last entry is always the file that won a conflict.
    assert config.sources == (str(global_path), str(repo_path))
    assert config.dispatch.default_limit == 9


def test_layered_never_reports_the_internal_temp_file(tmp_path: pathlib.Path) -> None:
    """Historically (pre-#704) the merge round-tripped through a
    NamedTemporaryFile to reuse validation, and that path was deleted before
    the call returned -- so leaking it as provenance would have named a file
    the operator cannot open, worse than reporting none. #704 removed the
    temp file entirely (the merge is now built in memory), which makes this
    pin trivially satisfied by construction; it stays as a regression guard
    against a future reintroduction of a file-backed merge step."""
    fleet = tmp_path / "fleet"
    fleet.mkdir()
    (fleet / "config.yaml").write_text("dispatch:\n  default_limit: 9\n", encoding="utf-8")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    config = load_layered_config(repo_root, None, fleet_dir_override=str(fleet))

    assert config.sources == (str(fleet / "config.yaml"),)
    assert all(pathlib.Path(s).exists() for s in config.sources)


def test_layered_with_no_files_records_nothing(tmp_path: pathlib.Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    config = load_layered_config(
        repo_root, None, fleet_dir_override=str(tmp_path / "absent-fleet")
    )
    assert config == OrchestratorConfig()
    assert config.sources == ()


def test_layered_records_a_present_but_empty_global_layer(
    tmp_path: pathlib.Path,
) -> None:
    """Both layers present, neither contributing a section, takes the
    ``not merged_data`` shortcut. It must still report what was read: "the
    global layer is empty" and "there is no global layer" demand opposite
    fixes."""
    fleet = tmp_path / "fleet"
    fleet.mkdir()
    global_path = fleet / "config.yaml"
    global_path.write_text("", encoding="utf-8")
    repo_root = tmp_path / "repo"
    repo_path = _repo_with_config(repo_root, "")

    config = load_layered_config(repo_root, None, fleet_dir_override=str(fleet))

    assert config == OrchestratorConfig()
    assert config.sources == (str(global_path), str(repo_path))


def test_discarded_global_layer_is_not_claimed_as_a_source(
    tmp_path: pathlib.Path,
) -> None:
    """An invalid global layer is dropped and the per-repo config is used
    alone (#665). Provenance must follow that rollback -- listing the global
    file would claim a contribution that was thrown away, and the log line
    saying so scrolls out of the buffer long before the config does."""
    fleet = tmp_path / "fleet"
    fleet.mkdir()
    (fleet / "config.yaml").write_text("not_a_real_section:\n  key: 1\n", encoding="utf-8")
    repo_root = tmp_path / "repo"
    repo_path = _repo_with_config(repo_root, "dispatch:\n  default_limit: 4\n")

    config = load_layered_config(repo_root, None, fleet_dir_override=str(fleet))

    assert config.dispatch.default_limit == 4
    assert config.sources == (str(repo_path),)


def test_discarded_global_layer_rescue_reaches_an_empty_bodied_bogus_section(
    tmp_path: pathlib.Path,
) -> None:
    """Same rescue as above (#665), but the global layer's bogus section has
    an *empty* body -- `not_a_real_section: {}` rather than `{key: 1}`. Before
    issue #962's fix this never reached the rescue at all: the merge dropped
    the falsy section before validation, so the bad global layer was silently
    accepted instead of being detected-then-discarded. Guards against a fix
    for #962 that makes empty-bodied unknown sections raise but bypasses the
    #665 rescue path in the process (e.g. by validating before the merge
    instead of feeding the merged dict to build_config_from_data, which is
    what actually raises "unknown config section(s)" post-#704)."""
    fleet = tmp_path / "fleet"
    fleet.mkdir()
    (fleet / "config.yaml").write_text("not_a_real_section: {}\n", encoding="utf-8")
    repo_root = tmp_path / "repo"
    repo_path = _repo_with_config(repo_root, "dispatch:\n  default_limit: 4\n")

    config = load_layered_config(repo_root, None, fleet_dir_override=str(fleet))

    assert config.dispatch.default_limit == 4
    assert config.sources == (str(repo_path),)


# --------------------------------------------------------------------------
# Issue #962: load_layered_config must reject an unknown section regardless
# of whether its body is empty, exactly like load_config does for the same
# raw YAML. The matrix below is the one from the issue report, reproduced as
# assertions so each cell has a durable pin.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("body", ["{}", "null", "[]", "{key: 1}"])
def test_layered_rejects_unknown_section_regardless_of_body(
    tmp_path: pathlib.Path, body: str
) -> None:
    """The core #962 bug: `bogus_section: {}` (and `null`, `[]`) merged to a
    falsy value and vanished from the merged dict *before* load_config's
    unknown-section check ever saw the name, so it was silently accepted --
    while the identical file loaded directly through load_config, or the same
    section with a non-empty body, was correctly rejected. No per-repo config
    is required to trigger this: the merge dropped the section before the
    #665 rescue path (which needs a per-repo config to fall back to) is even
    reachable, so this is a distinct failure from the rescue-path tests above.
    """
    fleet = tmp_path / "fleet"
    fleet.mkdir()
    (fleet / "config.yaml").write_text(f"bogus_section: {body}\n", encoding="utf-8")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    with pytest.raises(ConfigError, match="unknown config section"):
        load_layered_config(repo_root, None, fleet_dir_override=str(fleet))


def test_layered_rejects_unknown_per_repo_section_with_empty_body(
    tmp_path: pathlib.Path,
) -> None:
    """Same bug, per-repo side: an unknown section with an empty body in the
    per-repo file must raise too, not just when it originates in the global
    layer."""
    repo_root = tmp_path / "repo"
    _repo_with_config(repo_root, "bogus_section: {}\n")

    with pytest.raises(ConfigError, match="unknown config section"):
        load_layered_config(repo_root, None, fleet_dir_override=str(tmp_path / "no-fleet"))


def test_layered_control_matches_direct_for_a_valid_section(tmp_path: pathlib.Path) -> None:
    """Positive control from the issue report: a *valid* section must be
    accepted by both loaders, so the reject/accept split measured above is a
    real behavioural difference and not an artifact of how the two loaders
    were invoked."""
    fleet = tmp_path / "fleet"
    fleet.mkdir()
    (fleet / "config.yaml").write_text("dispatch:\n  default_limit: 3\n", encoding="utf-8")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    direct = load_config(_repo_with_config(tmp_path / "direct", "dispatch:\n  default_limit: 3\n"))
    layered = load_layered_config(repo_root, None, fleet_dir_override=str(fleet))

    assert direct.dispatch.default_limit == 3
    assert layered.dispatch.default_limit == 3


def test_layered_still_drops_a_known_section_with_an_empty_body(
    tmp_path: pathlib.Path,
) -> None:
    """Pin for a cell #962 must *not* change: a known section (`dispatch`)
    with an empty body in the global layer is still dropped from the merged
    dict rather than passed through to build_config_from_data. This is safe
    only because build_config_from_data's own `_section()` defaults an
    absent-but-known section identically to a present-but-empty one -- so
    dropping it changes nothing observable, unlike the unknown-section case
    above."""
    fleet = tmp_path / "fleet"
    fleet.mkdir()
    (fleet / "config.yaml").write_text("dispatch: {}\n", encoding="utf-8")
    repo_root = tmp_path / "repo"
    repo_path = _repo_with_config(repo_root, "review:\n  require_issue_link: true\n")

    config = load_layered_config(repo_root, None, fleet_dir_override=str(fleet))

    assert config == replace(
        load_config(repo_path), sources=(str(fleet / "config.yaml"), str(repo_path))
    )


def test_layered_control_known_section_unknown_key_already_consistent(
    tmp_path: pathlib.Path,
) -> None:
    """Pin for the matrix's third cell (not #962's bug): a *known* section
    with an *unknown key* inside it was already rejected consistently by both
    loaders before this fix, because `_build_section`'s key check runs on the
    section's own dict regardless of whether the section name is known at the
    top level. Guards against a #962 fix that accidentally touches this path."""
    direct_path = _repo_with_config(tmp_path / "direct", "dispatch:\n  not_a_real_key: 1\n")
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(direct_path)

    fleet = tmp_path / "fleet"
    fleet.mkdir()
    (fleet / "config.yaml").write_text("dispatch:\n  not_a_real_key: 1\n", encoding="utf-8")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    with pytest.raises(ConfigError, match="unknown key"):
        load_layered_config(repo_root, None, fleet_dir_override=str(fleet))


# --------------------------------------------------------------------------
# Role-config Phase 1 (issue TBD): layered-merge coverage for the new
# worker/reviewer sections and their dual-accept legacy counterparts. Note
# the global/fleet layer file is always ``config.yaml`` (``GLOBAL_CONFIG_
# FILENAME`` in layout.py) -- distinct from the per-repo
# ``orchestrator.config.yaml`` that ``_repo_with_config`` writes -- matching
# every other layered test above.
# --------------------------------------------------------------------------


def test_layered_merges_worker_section_repo_overrides_global(tmp_path: pathlib.Path) -> None:
    fleet = tmp_path / "fleet"
    fleet.mkdir()
    (fleet / "config.yaml").write_text(
        "worker:\n  harness: devin-shell\n  model: glm-5-2\n", encoding="utf-8"
    )
    repo_root = tmp_path / "repo"
    _repo_with_config(repo_root, "worker:\n  model: sonnet-5\n")

    config = load_layered_config(repo_root, None, fleet_dir_override=str(fleet))

    # Shallow per-section merge: repo's worker.model wins, global's
    # worker.harness (absent from the repo layer) survives.
    assert config.worker.harness == "devin-shell"
    assert config.worker.model == "sonnet-5"


def test_layered_cross_layer_old_new_conflict_falls_back_to_repo_alone(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    # The #665/#623 silent-global-discard shape, reachable during a
    # staggered global/repo role-config cutover: global sets the OLD key,
    # repo sets the disagreeing NEW key. The merged dict fails
    # build_config_from_data's conflict check, so the loader falls back to
    # the per-repo config alone (with a warning) rather than raising.
    fleet = tmp_path / "fleet"
    fleet.mkdir()
    (fleet / "config.yaml").write_text("devin:\n  adapter: devin-shell\n", encoding="utf-8")
    repo_root = tmp_path / "repo"
    _repo_with_config(repo_root, "worker:\n  harness: claude-code\n")

    with caplog.at_level("WARNING"):
        config = load_layered_config(repo_root, None, fleet_dir_override=str(fleet))

    # Per-repo-alone fallback: only the repo's worker.harness took effect,
    # the global devin.adapter was discarded.
    assert config.worker.harness == "claude-code"
    assert config.devin.adapter == "claude-code"
    assert any("global layer was discarded" in record.message for record in caplog.records)


def test_layered_mixed_old_worker_new_reviewer_across_layers_merges_cleanly(
    tmp_path: pathlib.Path,
) -> None:
    # The claim-asymmetry rule from Task 4 also makes this the OK case, not
    # a conflict: global keeps the legacy claude_code.model driving a
    # claude-code worker, repo adds a new reviewer.model to decouple the
    # reviewer. Confirms the rule holds across the layered-merge boundary,
    # not just within one file.
    fleet = tmp_path / "fleet"
    fleet.mkdir()
    (fleet / "config.yaml").write_text(
        "devin:\n  adapter: claude-code\nclaude_code:\n  model: claude-opus-4-1\n",
        encoding="utf-8",
    )
    repo_root = tmp_path / "repo"
    _repo_with_config(repo_root, "reviewer:\n  model: claude-sonnet-5\n")

    config = load_layered_config(repo_root, None, fleet_dir_override=str(fleet))

    assert config.worker.model == "claude-opus-4-1"
    assert config.reviewer.model == "claude-sonnet-5"
