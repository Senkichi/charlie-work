"""Render tests for the Now page (pages/now*.py) on literal models: structure, not pixels."""

from __future__ import annotations

import re
from dataclasses import replace

import pytest
from _dashboard_page_fixtures import (
    NOW,
    QUEUE_CMD,
    REQUEUE_CMD,
    VERDICT_CMD,
    _item,
    _model,
    _page,
    _parse,
)

from charlie_work.dashboard.now_types import RepoFreshness
from charlie_work.dashboard.pages import now_fmt, routes
from charlie_work.dashboard.pages.now import render_fragment, render_now
from charlie_work.dashboard.read_model import ModelState
from charlie_work.dashboard.theme import generate_css, static_asset

DRILL_DOWNS = frozenset({"/now", "/repo", "/issue", "/flow", "/prs", "/backlog", "/capacity"})


@pytest.fixture
def drilldowns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register the drill-down routes, as the drill-down PR will."""
    monkeypatch.setattr(routes, "ROUTES", DRILL_DOWNS)


def test_groups_render_in_decision_order() -> None:
    html_text = _page()
    ids = re.findall(r'<li class="grp" id="grp-([a-z-]+)"', html_text)
    assert ids == ["exceptions", "awaiting-your-verdict", "human-needed", "operator-queue"]
    # danger row before warn row inside Exceptions
    assert html_text.index("tone-danger") < html_text.index("tone-warn")
    assert '>5</a><h2 id="needs-h">need you</h2>' in html_text


def test_calm_state_is_one_journal_line_not_a_table() -> None:
    html_text = _page(_model(items=()))
    assert "Nothing needs you. Last pass 24s ago." in html_text
    assert 'id="needs-list"' not in html_text and "<table" not in html_text.split("rail")[0]


def test_unknown_cap_is_dashed_and_labelled_never_faked() -> None:
    html_text = _page()
    workers = html_text[html_text.index(">Workers<") : html_text.index(">Reviewers<")]
    assert 'class="trk-open"' in workers and "cap not reported" in workers
    assert 'class="captick"' not in workers and "cap ?" in workers
    reviewers = html_text[html_text.index(">Reviewers<") : html_text.index(">CI runners<")]
    assert 'class="captick"' in reviewers and "cap not reported" not in reviewers


def test_hostile_reason_is_escaped() -> None:
    html_text = _page()
    assert "<script>alert(1)" not in html_text
    assert "&lt;script&gt;alert(1)&lt;/script&gt; &amp; more" in html_text


def test_copy_buttons_carry_the_exact_command() -> None:
    copies = [
        a["data-copy"] for t, a in _parse(_page()).tags if t == "button" and "data-copy" in a
    ]
    assert copies == [VERDICT_CMD, REQUEUE_CMD, QUEUE_CMD, QUEUE_CMD]
    html_text = _page()
    assert html_text.count('class="cmd cmd2"') == 1  # secondary command as a second line


def test_no_inline_style_or_script_anywhere() -> None:
    for html_text in (
        _page(),
        _page(_model(items=())),
        _page(None, collector_error="Boom <b>", collector_failing_since=NOW),
        render_now(ModelState(), poll_seconds=15),
    ):
        assert not re.search(r"\sstyle\s*=", html_text, re.I)
        assert not re.search(r"<style", html_text, re.I)
        assert not re.search(r"\son[a-z]+\s*=", html_text, re.I)
        for tag, attrs in _parse(html_text).tags:
            if tag == "script":
                assert attrs.get("src", "").startswith("/static/")
        scripts = re.findall(r"<script\b[^>]*>(.*?)</script>", html_text, re.S | re.I)
        assert all(body == "" for body in scripts)


def test_collector_failure_banner() -> None:
    html_text = _page(collector_error="OSError: <nope>", collector_failing_since=NOW)
    assert 'data-alert="collector"' in html_text and "OSError: &lt;nope&gt;" in html_text
    assert "<nope>" not in html_text
    # Inside the swapped #now a role=alert would be re-announced on every poll; the
    # page's one polite region is the stable #client-status (plus dashboard.js's own).
    assert 'role="alert"' not in html_text


def test_header_as_of_local_time_nav_and_freshness() -> None:
    html_text = _page()
    local = NOW.astimezone().strftime("%H:%M:%S")
    assert f'<time class="js-local" datetime="{NOW.isoformat()}">{local}</time>' in html_text
    assert "(local)" in html_text
    assert '<span class="soon">History <small>(soon)</small></span>' in html_text
    assert 'href="/history"' not in html_text  # unbuilt view: text, not a link to a 404
    assert 'class="chip is-warn text-warn"' in html_text and "stale</span>" in html_text


def test_unrouted_numbers_render_as_text_not_dead_links() -> None:
    html_text = _page()
    hrefs = [a["href"] for t, a in _parse(html_text).tags if t == "a" and a.get("href")]
    assert hrefs and all(routes.is_routed(h) for h in hrefs), hrefs
    assert not any(h.startswith(("/repo", "/issue", "/flow", "/backlog")) for h in hrefs)
    assert '<span class="n">84</span>' in html_text  # the held count is still shown


def test_numbers_link_to_drill_downs_and_done_is_unrecorded(drilldowns: None) -> None:
    html_text = _page()
    assert 'href="/issue/Senkichi/charlie-work/6"' in html_text
    assert 'href="/flow/in-progress"' in html_text and 'href="/flow/needs-rework"' in html_text
    assert 'href="/repo/Senkichi/swole?view=runners"' in html_text
    assert "not yet" in html_text and 'class="bar unk"' in html_text
    done = _page(_model(flow=replace(_model().flow, done_24h=9)))
    assert 'class="bar done"' in done and 'class="stage-n done"' in done


def test_fragment_is_the_htmx_target_with_stable_ids() -> None:
    frag = render_fragment(ModelState(model=_model()), 15)
    assert frag.startswith('<div id="now" class="shell" hx-get="/now/fragment"')
    assert 'hx-trigger="every 15s" hx-swap="outerHTML"' in frag
    assert frag == render_fragment(ModelState(model=_model()), 15)  # ids deterministic
    ids = [a["id"] for _, a in _parse(frag).tags if a.get("id")]
    assert len(ids) == len(set(ids))


def test_static_assets_served_and_token_only() -> None:
    css = static_asset("now.css").read_text(encoding="utf-8")
    assert static_asset("dashboard.js").read_bytes()
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b", css)
    assert not re.search(r"^\s*--[\w-]+\s*:", css, re.M)  # no new custom properties
    assert "@media (max-width: 699px)" in css
    tokens = generate_css()
    assert "@media (prefers-color-scheme: dark)" in tokens
    assert ':root[data-theme="dark"]' in tokens and "#1E1611" in tokens  # Lamplit Paper page


def test_dashboard_scripts_avoid_dynamic_code_and_html_injection() -> None:
    for name in ("dashboard.js", "theme-init.js"):
        src = static_asset(name).read_text(encoding="utf-8")
        for banned in ("eval(", "new Function", "innerHTML", "outerHTML", "document.write"):
            assert banned not in src, f"{name} uses {banned}"
    assert "cw-dash-theme" in static_asset("theme-init.js").read_text(encoding="utf-8")


def test_row_budget_hides_overflow_behind_a_toggle_but_never_exceptions() -> None:
    from charlie_work.dashboard.pages.now_needs import ROW_BUDGET, ROW_FLOOR, visible_counts

    alarms = tuple(
        _item("Exceptions", kind="alarm", severity="anomaly", reason=f"alarm {n}", number=None)
        for n in range(ROW_BUDGET + 2)
    )
    queue = tuple(
        _item("Operator queue", reason=f"Operator queue: #{n} t", number=n) for n in range(9)
    )
    html_text = _page(_model(items=alarms + queue))
    assert html_text.count('class="row tone-danger"') == ROW_BUDGET + 2  # alarms: never capped
    assert html_text.count(' over"') == 9 - ROW_FLOOR
    assert f'data-more="{9 - ROW_FLOOR}" aria-expanded="false">+{9 - ROW_FLOOR} more' in html_text
    assert visible_counts(
        {"Awaiting your verdict": 3, "Human needed": 4, "Operator queue": 11}
    ) == {
        "Awaiting your verdict": 3,
        "Human needed": 4,
        "Operator queue": ROW_BUDGET - 7,
    }
    assert "more-btn" not in _page()  # within budget: no toggle


def test_reason_drops_the_group_prefix_and_bolds_the_lead_ref(drilldowns: None) -> None:
    html_text = _page()
    assert '<b class="ref">PR #7</b> (issue #6) awaits an operator verdict</a>' in html_text
    assert '<b class="ref">#9</b> x</a>' in html_text
    assert 'title="Human needed: #9 x · as of snapshot"' in html_text  # full reason kept


def test_done_24h_is_off_the_stock_scale() -> None:
    html_text = _page(_model(flow=replace(_model().flow, done_24h=500)))
    done = re.search(r'<rect class="bar done"[^>]*height="([\d.]+)"', html_text)
    assert done and float(done.group(1)) < 10  # a token bar, not 500 against stocks of 5
    tall = re.findall(r'<rect class="bar bar"[^>]*height="([\d.]+)"', html_text)
    assert max(float(h) for h in tall) == 75.0  # the biggest stock still fills the chart
    assert 'class="stage done unit-sep"' in html_text


def test_theme_query_override_is_not_persisted() -> None:
    js = static_asset("theme-init.js").read_text(encoding="utf-8")
    assert 'get("theme")' in js and "setItem" not in js


@pytest.mark.parametrize(
    "key", ["..", "../..", "a/..", "../x", "a/b/c", "fleet", "x/y z", 'a/"><b', "", "a//b", "./x"]
)
def test_hrefs_are_built_only_from_valid_slugs(key: str) -> None:
    assert now_fmt.repo_url(key) is None
    assert now_fmt.issue_url(key, 7) is None
    assert now_fmt.link(now_fmt.repo_url(key), "t") == '<span class="n">t</span>'


@pytest.mark.parametrize("key", ["owner/name", "local/mdls", "a.b/c_d-e"])
def test_valid_slugs_keep_their_links(key: str) -> None:
    assert now_fmt.repo_url(key) == f"/repo/{key}"
    assert now_fmt.issue_url(key, 7) == f"/issue/{key}/7"


def test_page_has_no_href_for_a_dot_dot_repo_key() -> None:
    base = _model()
    model = replace(
        base,
        freshness=(RepoFreshness("../..", NOW, 24.0, False, None), *base.freshness),
        needs_me=(_item("Needs a look", repo="..", number=7), *base.needs_me),
    )
    page = _page(model)
    assert "/repo/.." not in page and "/issue/../" not in page
    assert 'href="/repo/../' not in page
