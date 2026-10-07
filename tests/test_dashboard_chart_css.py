"""Chart/drill stylesheet contracts: legible label contrast, no sub-11px text token."""

from __future__ import annotations

import re

import pytest

from charlie_work.dashboard.theme import generate_css, static_asset


def _themes() -> dict[str, dict[str, str]]:
    css = generate_css()
    blocks = dict(re.findall(r'(:root|:root\[data-theme="dark"\])\s*\{([^}]*)\}', css))
    return {
        name: dict(re.findall(r"(--lj-[\w-]+):\s*(#[0-9A-Fa-f]{6})", blocks[sel]))
        for name, sel in (("light", ":root"), ("dark", ':root[data-theme="dark"]'))
    }


def _lum(hex_: str) -> float:
    def ch(c: int) -> float:
        v = c / 255
        return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4

    r, g, b = (int(hex_[i : i + 2], 16) for i in (1, 3, 5))
    return 0.2126 * ch(r) + 0.7152 * ch(g) + 0.0722 * ch(b)


def _contrast(a: str, b: str) -> float:
    hi, lo = sorted((_lum(a), _lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_direct_label_text_meets_aa_on_the_card(theme: str) -> None:
    tokens = _themes()[theme]
    css = static_asset("charts.css").read_text(encoding="utf-8")
    fills = re.findall(r"svg text\.direct\.s\d \{ fill: var\((--lj-[\w-]+)\); \}", css)
    assert len(fills) == 3  # positive control: one rule per series style
    for tok in fills:
        assert _contrast(tokens[tok], tokens["--lj-card"]) >= 4.5, (theme, tok)


def test_charts_css_uses_only_tokens_and_px_text_at_least_11() -> None:
    css = static_asset("charts.css").read_text(encoding="utf-8")
    defined = set(re.findall(r"(--[\w-]+)\s*:", generate_css()))
    used = set(re.findall(r"var\((--[\w-]+)", css))
    assert used and used <= defined, sorted(used - defined)
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b|rgba?\(", css)
    sizes = re.findall(r"font-size:\s*([\d.]+)px", css)
    assert sizes and all(float(s) >= 11 for s in sizes)


@pytest.mark.parametrize("sheet", ["charts.css", "history.css", "drill.css"])
def test_new_sheets_never_use_the_10px_text_token(sheet: str) -> None:
    css = static_asset(sheet).read_text(encoding="utf-8")
    assert "var(--lj-" in css  # positive control: the sheet does use tokens
    assert "--lj-text-1)" not in css
