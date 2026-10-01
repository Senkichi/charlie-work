"""Dashboard theme: CSS generation determinism, base-token fidelity, WCAG, swole drift."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from charlie_work.dashboard.theme import (
    base_groups,
    contrast_ratio,
    find_swole_root,
    generate_css,
    load_tokens,
    swole_drift,
)

# Live-parity tests read a real swole checkout only via explicit opt-in (tests isolate the
# fleet registry dir, so it cannot be discovered here). The status group needs a checkout
# that carries swole PR #467 (branch feat/living-journal-status-extension).
_SWOLE_ENV = os.environ.get("CHARLIE_WORK_SWOLE_ROOT")
SWOLE = Path(_SWOLE_ENV) if _SWOLE_ENV else None
SWOLE_TOKENS = SWOLE / "docs/design/living-journal/tokens.json" if SWOLE else None
_SKIP_REASON = "set CHARLIE_WORK_SWOLE_ROOT to a swole checkout to run live parity"
TEXT_COLOR_TOKENS = ("ink", "inkSecondaryText", "semanticText")
TEXT_STATUS_TOKENS = ("danger", "warn")
SERIES_TOKENS = ("series1", "series2", "series3")
MODES = ("light", "dark")


def _write_swole(root: Path, tokens: dict) -> None:
    path = root / "docs/design/living-journal/tokens.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(tokens), encoding="utf-8")


def test_css_is_deterministic_and_structured() -> None:
    css = generate_css()
    assert css == generate_css(load_tokens())
    assert css.count(":root {") == 1
    assert '@media (prefers-color-scheme: dark) {\n  :root:not([data-theme="light"]) {' in css
    assert ':root[data-theme="dark"] {' in css
    assert "body {\n  background: var(--lj-page);" in css


def test_css_exact_values_light_and_dark() -> None:
    css = generate_css()
    light, rest = css.split("@media", 1)
    media, explicit = rest.split(':root[data-theme="dark"]', 1)
    assert "--lj-page: #FAF6EF;" in light and "--lj-danger: #B3261E;" in light
    assert "--lj-warn: #9A5B00;" in light and "--lj-series-3: #8C7A62;" in light
    assert "--lj-space-6: 28px;" in light and "--lj-text-7: 31px;" in light
    assert "--lj-serif: 'Fraunces', 'Georgia', serif;" in light
    for dark in (media, explicit):
        assert "--lj-page: #1E1611;" in dark and "--lj-card: #2A2018;" in dark
        assert "--lj-danger: #E5604F;" in dark and "--lj-warn: #E8924A;" in dark
        assert "--lj-raised: #33271C;" in dark and "--lj-rule: #3A2D20;" in dark
        assert "--lj-series-3: #C2B49C;" in dark


def test_base_groups_match_swole_unchanged() -> None:
    vendored = load_tokens()
    assert {"status", "dashboardLocal"} <= set(vendored)
    assert not {"status", "dashboardLocal"} & set(base_groups(vendored))
    base = base_groups(vendored)
    assert set(base) == {
        "$schema", "$description", "color", "typography",
        "stroke", "radius", "motion", "iconography",
    }  # fmt: skip
    # Spot-pin the load-bearing values independent of the swole checkout.
    assert base["color"]["light"]["page"]["$value"] == "#FAF6EF"
    assert base["color"]["dark"]["semanticText"]["$value"] == "#3DBD55"
    assert base["radius"]["card"]["$value"] == "16px"
    assert "charlie-work-only" in vendored["dashboardLocal"]["$description"]


@pytest.mark.skipif(SWOLE_TOKENS is None or not SWOLE_TOKENS.is_file(), reason=_SKIP_REASON)
def test_vendored_base_equals_live_swole_checkout() -> None:
    upstream = json.loads(SWOLE_TOKENS.read_text(encoding="utf-8"))
    assert base_groups(load_tokens()) == base_groups(upstream)
    assert swole_drift(SWOLE).status in {"in_sync", "drifted"}


@pytest.mark.skipif(SWOLE_TOKENS is None or not SWOLE_TOKENS.is_file(), reason=_SKIP_REASON)
def test_vendored_status_group_equals_live_swole_pr467() -> None:
    upstream = json.loads(SWOLE_TOKENS.read_text(encoding="utf-8"))
    if "status" not in upstream:
        pytest.skip("swole checkout predates PR #467 (no status group)")
    assert load_tokens()["status"] == upstream["status"]
    assert swole_drift(SWOLE).status == "in_sync"


def test_vendored_status_group_shape_is_exact() -> None:
    """A re-vendor that renames/drops a key must fail here, not silently unmap a CSS var."""
    status = load_tokens()["status"]
    assert set(status) == {"$description", "light", "dark", "chart"}
    for mode in MODES:
        assert set(status[mode]) == {"$description", "danger", "warn"}
        assert set(status["chart"][mode]) == set(SERIES_TOKENS)
        for tok in (status[mode]["danger"], status[mode]["warn"], *status["chart"][mode].values()):
            assert tok["$type"] == "color" and tok["$value"].startswith("#")
            assert tok["$description"]
    assert set(status["chart"]) == {"$description", "light", "dark"}
    assert (status["light"]["danger"]["$value"], status["dark"]["warn"]["$value"]) == (
        "#B3261E",
        "#E8924A",
    )
    assert status["chart"]["dark"]["series2"]["$value"] == "#A89A88"
    local = load_tokens()["dashboardLocal"]
    assert set(local) == {"$description", "spacing", "typeSize"}


def test_status_values_reach_css_vars() -> None:
    css = generate_css()
    for mode in MODES:
        for key, var in (("series1", "--lj-series-1"), ("series2", "--lj-series-2")):
            assert f"{var}: {load_tokens()['status']['chart'][mode][key]['$value']};" in css


def test_drift_compares_status_group(tmp_path: Path) -> None:
    tokens = load_tokens()
    same, edited = tmp_path / "a", tmp_path / "b"
    _write_swole(same, {**base_groups(tokens), "status": tokens["status"]})
    assert swole_drift(same).status == "in_sync"
    changed = json.loads(json.dumps(tokens["status"]))
    changed["chart"]["light"]["series3"]["$value"] = "#000000"
    _write_swole(edited, {**base_groups(tokens), "status": changed})
    result = swole_drift(edited)
    assert (result.status, result.differing) == ("drifted", ("status",))


def test_find_swole_root_from_registry(tmp_path: Path) -> None:
    swole, other = tmp_path / "swole", tmp_path / "other"
    swole.mkdir()
    other.mkdir()
    registry = {
        "repos": {
            "o/charlie-work": {"repo_root": str(other)},
            "Senkichi/swole": {"repo_root": str(swole)},
        }
    }
    assert find_swole_root(registry) == swole
    assert find_swole_root({"repos": {"o/swole": {"repo_root": str(tmp_path / "gone")}}}) is None
    assert find_swole_root({"repos": {"o/other": {"repo_root": str(other)}}}) is None
    for junk in ({}, {"repos": None}, {"repos": {"o/swole": "x"}}, {"repos": {"o/swole": {}}}):
        assert find_swole_root(junk) is None


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_text_tokens_meet_4_5_on_page_and_card(mode: str) -> None:
    tokens = load_tokens()
    color, status = tokens["color"][mode], tokens["status"][mode]
    fg = [color[n]["$value"] for n in TEXT_COLOR_TOKENS]
    fg += [status[n]["$value"] for n in TEXT_STATUS_TOKENS]
    for surface in ("page", "card"):
        bg = color[surface]["$value"]
        for value in fg:
            assert contrast_ratio(value, bg) >= 4.5, (mode, surface, value)


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_chart_series_meet_3_on_card(mode: str) -> None:
    tokens = load_tokens()
    card = tokens["color"][mode]["card"]["$value"]
    for name in SERIES_TOKENS:
        value = tokens["status"]["chart"][mode][name]["$value"]
        assert contrast_ratio(value, card) >= 3.0, (mode, name, value)


def test_contrast_ratio_known_values() -> None:
    assert contrast_ratio("#000000", "#FFFFFF") == pytest.approx(21.0)
    assert contrast_ratio("#70707A", "#FAF6EF") == pytest.approx(4.546, abs=0.01)


def test_drift_unavailable_never_raises(tmp_path: Path) -> None:
    assert swole_drift(None).status == "unavailable"
    missing = swole_drift(tmp_path / "nope")
    assert missing.status == "unavailable" and missing.differing == ()
    (tmp_path / "docs/design/living-journal").mkdir(parents=True)
    (tmp_path / "docs/design/living-journal/tokens.json").write_text("{bad", encoding="utf-8")
    assert swole_drift(tmp_path).status == "unavailable"


def test_drift_detects_changed_and_added_groups(tmp_path: Path) -> None:
    same, changed, added = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    base = base_groups(load_tokens())
    _write_swole(same, base)
    assert swole_drift(same).status == "in_sync"
    edited = json.loads(json.dumps(base))
    edited["color"]["light"]["page"]["$value"] = "#FFFFFF"
    _write_swole(changed, edited)
    result = swole_drift(changed)
    assert (result.status, result.differing) == ("drifted", ("color",))
    _write_swole(added, {**base, "elevation": {}})
    assert swole_drift(added).differing == ("elevation",)


PR467_TOKENS = Path(
    os.environ.get(
        "CHARLIE_WORK_SWOLE_PR467_TOKENS",
        "C:/Users/senki/repos/swole-lj-status/docs/design/living-journal/tokens.json",
    )
)


@pytest.mark.skipif(
    not PR467_TOKENS.is_file(), reason=f"swole PR #467 worktree absent: {PR467_TOKENS}"
)
def test_vendored_status_group_equals_swole_pr467_branch() -> None:
    upstream = json.loads(PR467_TOKENS.read_text(encoding="utf-8"))
    assert load_tokens()["status"] == upstream["status"]
    assert base_groups(load_tokens()) == base_groups(upstream)
