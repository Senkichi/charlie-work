"""Now page review fixes: staleness, layout, numerals, charts, contrast, landmarks, alerts.

Structure and CSS-rule tests (no browser); the staleness rule itself runs under node when
node is on PATH. Pixels are checked by the screenshot pass, not here.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections import Counter
from dataclasses import replace

import pytest
from _dashboard_page_fixtures import NOW, _model, _page, _parse

from charlie_work.dashboard.pages.now import render_fragment, render_now
from charlie_work.dashboard.read_model import ModelState
from charlie_work.dashboard.theme import contrast_ratio, load_tokens, static_asset
from charlie_work.dashboard.theme.assets import static_dir_traversable

LANDMARKS = {"main", "nav", "aside", "section", "header", "form"}


def _css(name: str) -> str:
    return re.sub(r"/\*.*?\*/", "", static_asset(name).read_text(encoding="utf-8"), flags=re.S)


def _rules(css: str, media: str | None = None) -> list[tuple[list[str], str]]:
    """(selectors, declarations) of top-level rules, or of the rules in one @media block."""
    if media is not None:
        start = css.index(f"@media {media}")
        depth, i = 0, css.index("{", start)
        for j in range(i, len(css)):
            depth += {"{": 1, "}": -1}.get(css[j], 0)
            if depth == 0:
                css = css[i + 1 : j]
                break
    else:
        css = re.sub(r"@media[^{]*\{(?:[^{}]*\{[^{}]*\})*[^{}]*\}", "", css)
    return [
        ([s.strip() for s in sel.split(",")], body)
        for sel, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css)
    ]


def _decl(css: str, selector: str, prop: str, media: str | None = None) -> str | None:
    """The last value ``prop`` gets from a rule naming ``selector`` exactly."""
    value = None
    for sels, body in _rules(css, media):
        if selector in sels:
            m = re.search(rf"(?:^|;)\s*{re.escape(prop)}\s*:\s*([^;]+)", body)
            if m:
                value = m.group(1).strip()
    return value


# ---- (1) client-side staleness -------------------------------------------------------


def test_page_carries_the_stable_status_region_and_poll_interval() -> None:
    page = _page()
    tags = _parse(page).tags
    region = [a for t, a in tags if a.get("id") == "client-status"]
    assert region == [
        {"id": "client-status", "class": "client-status", "role": "status", "aria-live": "polite"}
    ]
    assert page.index('id="client-status"') < page.index('id="now"')  # outside the swap
    assert 'data-poll="15"' in render_fragment(ModelState(model=_model()), 15)
    scripts = [a["src"] for t, a in tags if t == "script"]
    assert scripts.index("/static/staleness.js") < scripts.index("/static/dashboard.js")


def test_dashboard_js_wires_every_htmx_failure_event_and_success() -> None:
    js = static_asset("dashboard.js").read_text(encoding="utf-8")
    for event in ("htmx:responseError", "htmx:sendError", "htmx:timeout"):
        assert re.search(rf'addEventListener\("{event}", pollFailed\)', js), event
    assert re.search(r'addEventListener\("htmx:afterSwap", function \(\) \{ pollOk\(\)', js)
    assert "setInterval(checkStale, 1000)" in js
    css = _css("now.css")
    assert "html.is-stale #now .n" in css  # the class the JS toggles mutes the numbers


_NODE = shutil.which("node")


@pytest.mark.skipif(_NODE is None, reason="node not on PATH")
def test_staleness_rule_under_node() -> None:
    path = static_dir_traversable().joinpath("staleness.js")
    script = (
        f"const s = require({json.dumps(str(path))});"
        "const out = {"
        " fresh: s.staleState(40000, 0, 20, null),"
        " stale: s.staleState(40001, 0, 20, null),"
        " down: s.staleState(90000, 1000, 20, 50000),"
        " oldFail: s.staleState(90000, 60000, 10, 50000),"
        " badPoll: s.staleState(40001, 0, 0, null),"
        " msg: s.message({since: 0, unreachable: true}, '14:26:12')};"
        "process.stdout.write(JSON.stringify(out));"
    )
    run = subprocess.run([_NODE, "-e", script], capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stderr
    out = json.loads(run.stdout)
    assert out["fresh"] is None  # exactly 2x poll: still fresh
    assert out["stale"] == {"since": 0, "unreachable": False}
    assert out["down"] == {"since": 1000, "unreachable": True}
    assert out["oldFail"] == {"since": 60000, "unreachable": False}  # fail before last ok
    assert out["badPoll"] == {"since": 0, "unreachable": False}  # falls back to 20s
    assert out["msg"].startswith("Not updating — last update 14:26:12")


# ---- (2) mid-width layout -----------------------------------------------------------


def test_mid_breakpoint_stacks_the_rail_as_two_columns_and_nothing_clips() -> None:
    now = _css("now.css")
    mid = "(max-width: 1280px)"
    assert _decl(now, ".shell", "grid-template-columns", mid) == "minmax(0, 1fr)"
    assert _decl(now, ".rail", "grid-template-columns", mid) == "repeat(2, minmax(0, 1fr))"
    assert '"repos repos"' in (_decl(now, ".rail", "grid-template-areas", mid) or "")
    assert _decl(now, ".row", "grid-template-areas", "(max-width: 960px)")
    assert _decl(now, ".table-scroll", "overflow-x") == "auto"
    base = _css("base.css")
    assert "overflow-x: hidden" not in base and "overflow-x:hidden" not in base
    page = _page()
    assert '<div class="table-scroll" role="region"' in page
    assert page.index('class="table-scroll"') < page.index('<table class="repos">')


# ---- (3) numerals ------------------------------------------------------------------


@pytest.mark.parametrize(
    "selector", [".row .age", ".repos td .n", ".cap-row .val .n", ".fresh .n", ".nd-row .hn"]
)
def test_aligned_numerals_use_inter_with_tabular_figures(selector: str) -> None:
    css = _css("now.css")
    assert _decl(css, selector, "font-family") == "var(--lj-sans)"
    assert "tabular-nums" in (_decl(css, selector, "font-variant-numeric") or "")


def test_no_rule_puts_aligned_numerals_back_in_fraunces() -> None:
    css = _css("now.css")
    for sels, body in _rules(css):
        if "--lj-serif" in body:
            bad = [s for s in sels if re.search(r"\.(age|repos td|cap-row \.val)", s)]
            assert not bad, bad


# ---- (5) charts: no shrinking SVG text, no links inside role=img ---------------------


def test_charts_have_no_svg_text_and_no_role_img() -> None:
    page = _page()
    assert "<text" not in page and 'role="img"' not in page
    for svg in re.findall(r"<svg\b[^>]*>.*?</svg>", page, re.S):
        assert 'aria-hidden="true"' in svg.split(">", 1)[0], svg[:80]
        assert "<a " not in svg
    css = _css("now.css")
    for selector in (".stage-l", ".stage-s", ".rework", ".nd-row .hl"):
        size = _decl(css, selector, "font-size")
        assert size == "var(--lj-text-2)", (selector, size)  # 11.5px, CSS px, never scaled


# ---- (6) over-cap is not colour-only ---------------------------------------------------


def test_over_cap_is_stated_in_words_with_a_glyph() -> None:
    cap = replace(_model().capacity, reviewers_live=8, reviewers_cap=6)
    page = _page(_model(capacity=cap))
    reviewers = page[page.index(">Reviewers<") : page.index(">CI runners<")]
    assert '<span class="over text-warn">over cap</span>' in reviewers
    assert "over cap" not in page.replace(reviewers, "")


# ---- (7) contrast of danger text on the highlighted-row surface ----------------------


@pytest.mark.parametrize("mode", ["light", "dark"])
@pytest.mark.parametrize(
    "selector",
    [".row.tone-danger:hover", ".row.tone-danger:focus-within", ".row.tone-danger.is-sel"],
)
def test_danger_row_highlight_keeps_danger_text_at_4_5(mode: str, selector: str) -> None:
    bg_var = _decl(_css("now.css"), selector, "background")
    m = re.fullmatch(r"var\(--lj-(\w+)\)", bg_var or "")
    assert m, bg_var
    tokens = load_tokens()
    surface = tokens["color"][mode].get(m.group(1)) or tokens["color"][mode]["card"]
    danger = tokens["status"][mode]["danger"]["$value"]
    assert contrast_ratio(danger, surface["$value"]) >= 4.5, (mode, m.group(1))


def test_raised_is_the_surface_that_fails_so_the_override_matters() -> None:
    tokens = load_tokens()  # positive control: without the override, dark would fail
    dark = tokens["color"]["dark"]
    assert (
        contrast_ratio(tokens["status"]["dark"]["danger"]["$value"], dark["raised"]["$value"])
        < 4.5
    )


# ---- (8) landmarks -----------------------------------------------------------------


@pytest.mark.parametrize(
    "state",
    [
        ModelState(model=_model()),
        ModelState(model=_model(items=())),
        ModelState(),
        ModelState(model=_model(), collector_error="E", collector_failing_since=NOW),
    ],
    ids=["items", "calm", "collecting", "failing"],
)
def test_one_main_one_h1_skip_link_and_unique_landmark_names(state: ModelState) -> None:
    page = render_now(state, poll_seconds=15)
    tags = _parse(page).tags
    names = [t for t, _ in tags]
    assert names.count("main") == 1 and names.count("h1") == 1
    ids = {a.get("id") for _, a in tags}
    skips = [a for t, a in tags if t == "a" and "skip" in (a.get("class") or "")]
    assert len(skips) == 1 and skips[0]["href"].lstrip("#") in ids
    labelled = [
        a.get("aria-label") or a.get("aria-labelledby")
        for t, a in tags
        if (t in LANDMARKS or a.get("role") == "region")
        and (a.get("aria-label") or a.get("aria-labelledby"))
    ]
    dupes = [n for n, c in Counter(labelled).items() if c > 1]
    assert not dupes, dupes


def test_copy_buttons_are_named_by_their_command() -> None:
    labels = [
        a["aria-label"] for t, a in _parse(_page()).tags if t == "button" and "data-copy" in a
    ]
    assert labels and "copy command" not in labels
    assert all(label.startswith(("copy: ", "copy alternative command: ")) for label in labels)


def test_row_age_is_text_so_each_row_has_at_most_one_link() -> None:
    page = _page()
    assert not re.search(r'<a [^>]*class="age', page)


# ---- (9) banners are announced once ---------------------------------------------------


def test_banners_carry_a_stable_alert_key_and_js_dedupes_on_it() -> None:
    frag = render_fragment(
        ModelState(model=_model(), collector_error="E", collector_failing_since=NOW),
        15,
        stalled="no pass",
    )
    keys = re.findall(r'data-alert="(\w+)"', frag)
    assert sorted(keys) == ["collector", "stall"] and 'role="alert"' not in frag
    js = static_asset("dashboard.js").read_text(encoding="utf-8")
    assert 'querySelectorAll("#now [data-alert]")' in js
    assert "alertKeys.split" in js  # only keys not seen on the previous swap are spoken
