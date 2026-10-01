"""Dashboard theme: CSS generation determinism, base-token fidelity, WCAG, swole drift."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from charlie_work.dashboard.theme import (
    base_groups,
    contrast_ratio,
    generate_css,
    load_tokens,
    swole_drift,
)

SWOLE = Path("C:/Users/senki/repos/swole")
SWOLE_TOKENS = SWOLE / "docs/design/living-journal/tokens.json"
TEXT_COLOR_TOKENS = ("ink", "inkSecondaryText", "semanticText")
TEXT_STATUS_TOKENS = ("danger", "warn")
SERIES_TOKENS = ("seriesInk", "seriesGray", "seriesTan")


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
    assert "--lj-warn: #9A5B00;" in light and "--lj-series-tan: #8C7A62;" in light
    assert "--lj-space-6: 28px;" in light and "--lj-text-7: 31px;" in light
    assert "--lj-serif: 'Fraunces', 'Georgia', serif;" in light
    for dark in (media, explicit):
        assert "--lj-page: #1E1611;" in dark and "--lj-card: #2A2018;" in dark
        assert "--lj-danger: #E5604F;" in dark and "--lj-warn: #E8924A;" in dark
        assert "--lj-raised: #33271C;" in dark and "--lj-rule: #3A2D20;" in dark
        assert "--lj-series-tan: #C2B49C;" in dark


def test_base_groups_match_swole_unchanged() -> None:
    vendored = load_tokens()
    assert "status" in vendored and "status" not in base_groups(vendored)
    base = base_groups(vendored)
    assert set(base) == {
        "$schema", "$description", "color", "typography",
        "stroke", "radius", "motion", "iconography",
    }  # fmt: skip
    # Spot-pin the load-bearing values independent of the swole checkout.
    assert base["color"]["light"]["page"]["$value"] == "#FAF6EF"
    assert base["color"]["dark"]["semanticText"]["$value"] == "#3DBD55"
    assert base["radius"]["card"]["$value"] == "16px"
    assert vendored["status"]["$description"].startswith("EXTENSION: operational surfaces only")
    assert "pending upstream to swole" in vendored["status"]["$description"]


@pytest.mark.skipif(not SWOLE_TOKENS.is_file(), reason="swole checkout absent")
def test_vendored_base_equals_live_swole_checkout() -> None:
    upstream = json.loads(SWOLE_TOKENS.read_text(encoding="utf-8"))
    assert base_groups(load_tokens()) == upstream
    assert swole_drift(SWOLE).status == "in_sync"


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
        value = tokens["status"][mode][name]["$value"]
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
