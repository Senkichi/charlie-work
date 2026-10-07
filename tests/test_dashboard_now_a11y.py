"""Now page review fixes: staleness, layout, numerals, controls, landmarks, alerts.

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
from _dashboard_page_fixtures import NOW, _model, _page, _parse, _progress

from charlie_work.dashboard.pages.now import render_fragment, render_now
from charlie_work.dashboard.read_model import ModelState
from charlie_work.dashboard.theme import static_asset
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


# ---- (2) layout: fills the window, one column below 1151px ----------------------------------


def test_wide_layout_is_exactly_the_window_high_with_the_needs_list_scrolling() -> None:
    css = _css("now-page.css")
    assert _decl(css, ".grid", "grid-template-areas") == '"needs progress" "needs flow"'
    assert _decl(css, ".shell", "height", "(min-width: 1151px)") == "100vh"
    assert _decl(css, ".list", "overflow") == "auto" and _decl(css, ".list", "min-height") == "0"
    assert _decl(css, ".shell", "max-width") is None  # no fixed content width on Now
    assert "1600px" not in css


def test_narrow_layout_is_one_column_in_reading_order() -> None:
    css = _css("now-page.css")
    narrow = "(max-width: 1150px)"
    assert _decl(css, ".grid", "grid-template-columns", narrow) == "1fr"
    assert _decl(css, ".grid", "grid-template-areas", narrow) == '"needs" "progress" "flow"'
    assert _decl(css, ".flowrow", "grid-template-columns", narrow) == "1fr"
    base = _css("base.css")
    assert "overflow-x: hidden" not in base and "overflow-x:hidden" not in base


def test_type_scale_is_fluid_and_hidden_means_hidden() -> None:
    css = _css("now-page.css")
    for var in ("--n-hero", "--n-h", "--n-num", "--n-body", "--n-small", "--n-gap"):
        assert re.search(rf"{var}:\s*clamp\(", css), var
    assert _decl(css, "body.now [hidden]", "display") == "none !important"
    assert "outline: none" not in css and "outline:none" not in css  # focus rings stay


# ---- (3) numerals ------------------------------------------------------------------


@pytest.mark.parametrize("selector", [".fresh .n", ".hn"])
def test_aligned_numerals_use_inter_with_tabular_figures(selector: str) -> None:
    css = _css("now.css")
    assert _decl(css, selector, "font-family") == "var(--lj-sans)"
    assert "tabular-nums" in (_decl(css, selector, "font-variant-numeric") or "")


def test_ages_are_inter_and_big_numbers_are_tabular() -> None:
    css = _css("now-page.css")
    assert _decl(css, ".age", "font-family") == "var(--lj-sans)"
    assert "tabular-nums" in (_decl(css, ".age", "font-variant-numeric") or "")
    assert "tabular-nums" in (_decl(css, "body.now .num", "font-variant-numeric") or "")


# ---- (5) charts: decorative svg is hidden from assistive tech, the chart is operable ------


def test_decorative_svgs_are_aria_hidden_and_the_chart_is_a_labelled_group() -> None:
    page = _page(progress=_progress())
    for svg in re.findall(r"<svg[^>]*>.*?</svg>", page, re.S):
        assert 'aria-hidden="true"' in svg.split(">", 1)[0], svg[:80]
        assert "<a " not in svg and 'role="img"' not in svg
    chart = [a for t, a in _parse(page).tags if a.get("id") == "chart"]
    assert chart and chart[0]["tabindex"] == "0" and chart[0]["role"] == "group"
    assert "arrows" in chart[0]["aria-label"]


def test_toggles_expose_their_state() -> None:
    page = _page(progress=_progress())
    seg = [a for t, a in _parse(page).tags if t == "button" and ("data-m" in a or "data-r" in a)]
    assert len(seg) == 2 + 4 and all(a["aria-pressed"] in ("true", "false") for a in seg)
    assert sum(a["aria-pressed"] == "true" for a in seg) == 2


# ---- (6) over-cap is not colour-only ---------------------------------------------------


def test_over_cap_is_stated_in_words_with_a_glyph() -> None:
    cap = replace(_model().capacity, reviewers_live=8, reviewers_cap=6)
    page = _page(_model(capacity=cap))
    reviewers = page[page.index('id="meter-reviewers"') : page.index('id="meter-runners"')]
    assert '<span class="note">over cap</span>' in reviewers
    assert "over cap" not in page.replace(reviewers, "")


def test_no_rule_puts_aligned_numerals_back_in_fraunces() -> None:
    for name in ("now.css", "now-page.css"):
        for sels, body in _rules(_css(name)):
            if "--lj-serif" in body:
                bad = [s for s in sels if re.search(r"\.(age|hn|val)", s)]
                assert not bad, (name, bad)


def test_focus_is_ink_weight_not_a_second_colour() -> None:
    """Colour has one job: --lj-warn marks "needs you" (the count and the selected tile)."""
    css = _css("now-page.css")
    users = [
        sels[0] for sels, body in _rules(css) if "--lj-warn" in body or "--n-accent-bg" in body
    ]
    assert users == [
        "body.now",  # the --n-accent-bg definition
        ".hero .num",
        'body.now .tile[aria-pressed="true"]',
        '.tile[aria-pressed="true"] .num',
    ]
    assert all("--lj-danger" not in body for sels, body in _rules(css) if "dot" not in sels[-1])


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
