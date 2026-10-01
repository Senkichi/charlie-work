"""Static assets and the combined stylesheet for the dashboard theme."""

from __future__ import annotations

import hashlib
import base64
import re

import pytest

from charlie_work.dashboard.theme import (
    base_css,
    generate_css,
    static_asset,
    static_dir_traversable,
    stylesheet,
)

HTMX_SHA384 = "sha384-2OatzQy1H+Zd/IIrjr1TcuDGqLXeHhbooAyJY1KdQMKnr4LZ22k31GBLdYKHmVjg"
ASSETS = (
    "base.css",
    "htmx.min.js",
    "VENDORED.md",
    "fonts/fraunces-latin-wght-normal.woff2",
    "fonts/fraunces-latin-wght-italic.woff2",
    "fonts/inter-latin-wght-normal.woff2",
    "fonts/OFL-Fraunces.txt",
    "fonts/OFL-Inter.txt",
)


@pytest.mark.parametrize("rel", ASSETS)
def test_asset_resolves_and_is_nonempty(rel: str) -> None:
    assert len(static_asset(rel).read_bytes()) > 1000


def test_woff2_magic_bytes() -> None:
    for rel in ASSETS:
        if rel.endswith(".woff2"):
            assert static_asset(rel).read_bytes()[:4] == b"wOF2"


def test_htmx_matches_published_sri_hash() -> None:
    digest = hashlib.sha384(static_asset("htmx.min.js").read_bytes()).digest()
    assert "sha384-" + base64.b64encode(digest).decode() == HTMX_SHA384
    assert HTMX_SHA384 in static_asset("VENDORED.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("bad", ["../theme.py", "/etc/passwd", "fonts/../base.css", "", "a//b"])
def test_static_asset_rejects_traversal(bad: str) -> None:
    with pytest.raises(ValueError):
        static_asset(bad)


def test_static_asset_missing_is_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        static_asset("nope.css")


def test_static_dir_is_a_directory() -> None:
    assert static_dir_traversable().is_dir()


def _defined(css: str) -> set[str]:
    return set(re.findall(r"(--[\w-]+)\s*:", css))


def test_base_css_uses_only_generated_custom_properties() -> None:
    generated = _defined(generate_css())
    used = set(re.findall(r"var\((--[\w-]+)", base_css()))
    assert used, "base.css should reference tokens"
    assert used <= generated, sorted(used - generated)
    # base.css must not mint its own properties; tokens.json is the single source.
    assert not _defined(base_css())


def test_reduced_motion_block_disables_animation_and_transition() -> None:
    css = base_css()
    m = re.search(r"@media \(prefers-reduced-motion: reduce\) \{(.*?)\n\}", css, re.S)
    assert m is not None
    body = m.group(1)
    assert "animation-iteration-count: 1 !important" in body
    assert "transition-duration: 0.001ms !important" in body
    assert "animation-duration: 0.001ms !important" in body


def test_focus_ring_and_phone_breakpoint_present() -> None:
    css = base_css()
    assert ":focus-visible { outline: 2px solid var(--lj-ink); outline-offset: 2px; }" in css
    assert "@media (max-width: 640px)" in css
    assert ".page > :not(header):not(.needs-me) { display: none; }" in css


def test_fonts_referenced_by_base_css_exist() -> None:
    for rel in re.findall(r'url\("/static/([^"]+)"\)', base_css()):
        assert static_asset(rel).is_file()


def test_stylesheet_is_tokens_then_base() -> None:
    css = stylesheet()
    assert css == generate_css() + "\n" + base_css()
    assert css.index(":root {") < css.index("@font-face")
