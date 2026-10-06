"""Render tests for the Now page (pages/now*.py) on literal models: structure, not pixels."""

from __future__ import annotations

import html as html_mod
import json
import re
from dataclasses import replace

import pytest
from _dashboard_page_fixtures import (
    CW,
    NOW,
    QUEUE_CMD,
    REQUEUE_CMD,
    SW,
    VERDICT_CMD,
    _item,
    _model,
    _page,
    _parse,
)

from charlie_work.dashboard.now_progress_data import ProgressData, ProgressSeries
from charlie_work.dashboard.now_types import RepoFreshness
from charlie_work.dashboard.pages import now_fmt, routes
from charlie_work.dashboard.pages.now import render_fragment, render_now
from charlie_work.dashboard.pages.now_flow import ARC_COLUMNS
from charlie_work.dashboard.pages.now_health import pill_text
from charlie_work.dashboard.pages.now_needs import ACTIVE_ROWS, THEN_ROWS
from charlie_work.dashboard.read_model import ModelState
from charlie_work.dashboard.theme import generate_css, static_asset

DRILL_DOWNS = frozenset({"/now", "/repo", "/issue", "/flow", "/prs", "/backlog", "/capacity"})
NOW_SCRIPTS = ("dashboard.js", "theme-init.js", "now.js", "now-chart.js")


@pytest.fixture
def drilldowns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register the drill-down routes, as the drill-down PR will."""
    monkeypatch.setattr(routes, "ROUTES", DRILL_DOWNS)


def _between(text: str, start: str, end: str) -> str:
    i = text.index(start)
    return text[i : text.index(end, i)]


# ---- Needs you: the count, three tiles, one list ----------------------------------------


def test_count_is_decisions_only_and_exceptions_leave_the_panel() -> None:
    page = _page()
    assert '<span class="num" id="needs-n">3</span><h2 id="needs-h">need you</h2>' in page
    needs = _between(page, 'class="panel n-needs"', 'class="panel n-progress"')
    assert "loop_errors" not in needs and "snapshot is 700s old" not in needs
    assert "loop_errors" in _between(page, 'id="health-pop"', "</div>")  # they went to the pill


def test_groups_render_in_decision_order() -> None:
    page = _page()
    tiles = re.findall(r'class="tile" id="tile-(\w+)" data-g="\w+" aria-pressed="(\w+)"', page)
    assert tiles == [("verdicts", "true"), ("decisions", "false"), ("requeue", "false")]
    assert re.findall(r'<section class="grp" data-g="(\w+)"', page) == [
        "verdicts",
        "decisions",
        "requeue",
    ]
    assert re.findall(r'<span class="num">(\d+)</span><span class="lbl">', page) == ["1", "1", "1"]
    assert 'data-g="verdicts" aria-label="Verdicts" data-on="1"' in page  # active at first paint
    assert 'id="needs-list" data-active="verdicts" data-default="verdicts"' in page


def test_first_group_with_rows_is_the_default() -> None:
    items = tuple(i for i in _model().needs_me if i.group == "Operator queue")
    assert 'data-default="requeue"' in _page(_model(items=items))


def test_calm_state_is_one_line_not_a_table() -> None:
    page = _page(_model(items=()))
    assert "Nothing needs you. Last pass 24s ago." in page
    assert 'id="needs-list"' not in page and "<table" not in page
    assert '<span class="num" id="needs-n">0</span>' in page


def test_rows_are_two_lines_and_commands_exist_only_in_the_detail() -> None:
    page = _page()
    rows = re.findall(r'<li class="row[^"]*" id="[^"]+">(.*?)</li>', page, re.S)
    assert len(rows) == 3
    for row in rows:
        head, _, detail = row.partition('<div class="detail"')
        assert "<code" not in head and "data-copy" not in head and "<a " not in head
        assert re.fullmatch(
            r'<button[^>]*class="rowbtn"[^>]*><span class="what">[^<]*</span>'
            r'<span class="age">[^<]*</span><span class="why">[^<]*</span></button>',
            head,
        ), head
        assert " hidden>" in detail.split(">", 1)[0] + ">"


def test_every_row_button_names_its_detail_and_starts_collapsed() -> None:
    page = _page()
    ids = {a["id"] for _, a in _parse(page).tags if a.get("id")}
    buttons = [a for t, a in _parse(page).tags if t == "button" and a.get("class") == "rowbtn"]
    assert len(buttons) == 3
    for b in buttons:
        assert b["aria-expanded"] == "false" and b["aria-controls"] in ids
        assert b["id"] == b["aria-controls"].removesuffix("-d") + "-b"


def test_copy_buttons_carry_the_exact_command() -> None:
    copies = [
        a["data-copy"] for t, a in _parse(_page()).tags if t == "button" and "data-copy" in a
    ]
    assert copies == [VERDICT_CMD, REQUEUE_CMD, QUEUE_CMD, QUEUE_CMD]
    assert _page().count('class="cmd"') == 4  # the secondary command is a second .cmd


def test_command_display_drops_the_repo_prefix_and_choice_list() -> None:
    page = _page()
    assert "verdict --pr 7 --decision &lt;…&gt;" in page
    assert "charlie --repo" not in re.sub(r'(title|data-copy|aria-label)="[^"]*"', "", page)


def test_reason_drops_the_group_prefix_and_bolds_the_lead_ref(drilldowns: None) -> None:
    page = _page()
    assert (
        '<span class="what">Senkichi/charlie-work · PR #7</span>'.replace("Senkichi/", "") in page
    )
    assert '<span class="why">awaits your verdict</span>' in page
    assert 'title="Human needed: PR #7 (issue #6) awaits an operator verdict' in page
    assert '<span class="why">x</span>' in page  # "Human needed: #9 x" -> "x"
    assert "Human needed:" not in re.sub(r'title="[^"]*"', "", page)  # group prefix dropped


def test_row_budget_shows_six_then_a_toggle_and_four_behind_other_groups() -> None:
    queue = tuple(
        _item("Operator queue", reason=f"Operator queue: #{n} t", number=n) for n in range(9)
    )
    page = _page(_model(items=queue))
    assert sum(page.count(f'class="row{c}"') for c in ("", " gt4", " gt4 gt6")) == 9
    assert page.count(" gt6") == 9 - ACTIVE_ROWS
    assert page.count(" gt4") == 9 - THEN_ROWS
    toggle = re.search(r'<button[^>]*class="more"[^>]*data-more="requeue" data-n="(\d+)"', page)
    assert toggle and int(toggle.group(1)) == 9 - ACTIVE_ROWS
    assert 'class="more"' not in _page()  # within budget: no toggle


def test_hostile_reason_is_escaped(drilldowns: None) -> None:
    hostile = '<img src=x onerror=alert(1)>"&'
    esc = "&lt;img src=x onerror=alert(1)&gt;&quot;&amp;"
    base = _model()
    model = replace(
        base,
        needs_me=(
            _item("Awaiting your verdict", reason=f"Human needed: PR #7 {hostile}", number=7),
            _item("Operator queue", reason=f"Operator queue: #3 {hostile}", repo=f"a/{hostile}"),
            *base.needs_me,
        ),
        freshness=(RepoFreshness(f"x/{hostile}", NOW, 24.0, True, hostile), *base.freshness),
    )
    series = ProgressSeries(
        "merged", "7d", (("2026-10-01T00:00:00Z", 1.0),), {hostile: ((0, 1.0),)}, None, True, False
    )
    page = _page(model, progress=ProgressData((series,)))
    assert "<img src=x" not in page and "<script>alert(1)" not in page
    assert esc in page  # rows, drawers and the health pill all go through esc()
    pill = _between(page, 'id="health-pop"', "</div>")
    assert esc in pill or "&lt;script&gt;" in pill
    attr = re.search(r'data-progress="([^"]*)"', page)
    assert attr and "<" not in attr.group(1) and ">" not in attr.group(1)
    assert (
        hostile
        in json.loads(html_mod.unescape(attr.group(1)))["metrics"]["merged"]["ranges"]["7d"][
            "repo"
        ]
    )
    assert "&lt;script&gt;alert(1)&lt;/script&gt; &amp; more" in _page()


# ---- exceptions are the header health pill ----------------------------------------------


def test_stale_source_and_alarm_become_a_pill_not_needs_rows() -> None:
    page = _page()
    pill = re.search(r'<button[^>]*id="health-btn" class="([^"]*)"[^>]*>(.*?)</button>', page)
    assert pill and pill.group(1) == "health"  # not "health ok": something is wrong
    assert "swole stale · 1 alarm" in pill.group(2)
    pop = _between(page, 'id="health-pop"', "</div>")
    assert "snapshot is 700s old" in pop and "last pass 24s ago" in pop
    assert 'aria-controls="health-pop"' in page and 'id="health-pop"' in page


def test_unreadable_source_is_named_and_error_beats_stale() -> None:
    fresh = (
        RepoFreshness(CW, NOW, 24.0, False, None),
        RepoFreshness("Senkichi/empericus", None, None, True, "missing: status.json"),
    )
    model = _model(freshness=fresh, needs_me=())
    assert pill_text(model) == "empericus unreadable"
    page = _page(model)
    assert "empericus unreadable" in page


def test_all_fresh_is_a_quiet_grey_pill() -> None:
    fresh = (RepoFreshness(CW, NOW, 24.0, False, None), RepoFreshness(SW, NOW, 30.0, False, None))
    items = tuple(i for i in _model().needs_me if i.group != "Exceptions")
    page = _page(_model(freshness=fresh, needs_me=items))
    assert re.search(r'id="health-btn" class="health ok"', page)
    assert "2 sources fresh" in page
    assert '<ul class="hlist" aria-label="Problems">' not in page


def test_paused_and_supervisor_exceptions_are_named_in_the_pill() -> None:
    items = (
        _item("Exceptions", kind="paused", severity="warn", repo="fleet", reason="fleet paused"),
        _item("Exceptions", kind="supervisor", severity="warn", repo="fleet", reason="silent"),
    )
    fresh = (RepoFreshness(CW, NOW, 24.0, False, None),)
    text = pill_text(_model(freshness=fresh, needs_me=items))
    assert "fleet paused" in text and "supervisor not beating" in text


def test_exceptions_never_change_the_needs_count() -> None:
    decisions = tuple(i for i in _model().needs_me if i.group != "Exceptions")
    with_exc = _page()
    without = _page(_model(items=decisions))
    for page in (with_exc, without):
        assert 'id="needs-n">3<' in page


# ---- Pipeline and Capacity ----------------------------------------------------------------


def test_pipeline_headline_names_the_longest_queue_and_nodes_are_buttons() -> None:
    page = _page()
    assert '<h2 id="flow-h">4 issues piled up at <b>In progress</b></h2>' in page
    nodes = re.findall(
        r'<button[^>]*class="(node[^"]*)" id="node-([\w-]+)" data-pipe="([\w-]+)"', page
    )
    assert [n[1] for n in nodes] == [
        "dispatchable",
        "queued",
        "in-progress",
        "pr-open",
        "reviewing",
        "merged",
    ]
    assert all(n[1] == n[2] for n in nodes)
    assert [n[1] for n in nodes if "focus" in n[0].split()] == ["in-progress", "merged"]
    for _, key, _ in nodes:
        assert f'id="pipe-{key}" data-pipe="{key}" hidden' in page  # a drawer per node
    assert 'id="pipe-why" data-pipe="why" hidden' in page and 'id="why-btn"' in page


def test_pipeline_stage_drawer_splits_by_repo_with_links(drilldowns: None) -> None:
    drawer = _between(_page(), 'id="pipe-in-progress"', "</div></div>")
    assert "In progress by repo" in drawer
    assert 'href="/repo/Senkichi/charlie-work"' in drawer


def test_rework_arc_is_pinned_to_the_reviewing_to_in_progress_columns() -> None:
    assert ARC_COLUMNS == (3, 6)  # a reordered track must update the CSS, loudly
    css = static_asset("now-page.css").read_text(encoding="utf-8")
    assert f"grid-column: {ARC_COLUMNS[0]} / {ARC_COLUMNS[1]}" in css
    assert "1 back for rework" in _page(
        _model(
            flow=replace(
                _model().flow,
                stages=tuple(
                    replace(s, count=1) if s.name == "Needs rework" else s
                    for s in _model().flow.stages
                ),
            )
        )
    )
    no_rework = _model(
        flow=replace(
            _model().flow,
            stages=tuple(
                replace(s, count=0) if s.name == "Needs rework" else s
                for s in _model().flow.stages
            ),
        )
    )
    assert "back for rework" not in _page(no_rework)


def test_done_24h_unknown_is_a_dash_and_known_is_a_number() -> None:
    page = _page()
    merged = _between(page, 'id="node-merged"', "</button>")
    assert '<span class="cnt num">—</span>' in merged and 'class="unk"' in merged
    assert "rollup is not current" in page
    done = _page(_model(flow=replace(_model().flow, done_24h=9)))
    assert '<span class="cnt num">9</span>' in _between(done, 'id="node-merged"', "</button>")
    assert "rollup is not current" not in done


def test_nothing_in_flight_headline() -> None:
    flow = replace(_model().flow, stages=tuple(replace(s, count=0) for s in _model().flow.stages))
    assert "Nothing in flight" in _page(_model(flow=flow))


def test_unknown_cap_is_dashed_and_labelled_never_faked() -> None:
    page = _page()
    workers = _between(page, 'id="meter-workers"', "</button>")
    assert 'class="trk open"' in workers and "cap not reported" in workers
    assert 'width="' not in workers  # no fill rect against an invented cap
    reviewers = _between(page, 'id="meter-reviewers"', "</button>")
    assert 'class="fill"' in reviewers and "cap not reported" not in reviewers


def test_capacity_headline_and_over_cap_are_words() -> None:
    cap = replace(
        _model().capacity, workers_live=3, workers_cap=3, reviewers_live=8, reviewers_cap=6
    )
    page = _page(_model(capacity=cap))
    assert '<h2 id="cap-h">Workers and Reviewers at capacity</h2>' in page
    assert '<span class="note">over cap</span>' in _between(
        page, 'id="meter-reviewers"', "</button>"
    )
    assert "Capacity has headroom" in _page(
        _model(capacity=replace(_model().capacity, reviewers_cap=6, reviewers_live=0))
    )


def test_every_capacity_meter_has_a_drawer() -> None:
    page = _page()
    for key in ("workers", "reviewers", "runners"):
        assert f'id="meter-{key}" data-cap="{key}" aria-pressed="false"' in page
        assert f'id="cap-{key}" data-cap="{key}" hidden' in page


# ---- the frame ----------------------------------------------------------------------------


def test_header_as_of_local_time_nav_and_freshness() -> None:
    page = _page()
    local = NOW.astimezone().strftime("%H:%M:%S")
    assert f'<time class="js-local" datetime="{NOW.isoformat()}">{local}</time>' in page
    assert "(local)" in page and 'class="brand">Fleet</h1>' in page
    assert '<a href="/now" aria-current="page" data-go="n">Now</a>' in page
    assert '<a href="/history" data-go="h">History</a>' in page
    assert 'id="theme-toggle"' in page
    assert 'class="fresh"' not in page  # the ledger strip left the front page
    assert "swole stale" in page  # freshness: the health pill names the stale source


def test_page_is_a_four_panel_grid_in_reading_order() -> None:
    page = _page()
    order = [page.index(f'id="{i}"') for i in ("needs", "progress", "flow", "capacity")]
    assert order == sorted(order)
    assert '<body class="now">' in page and "/static/now-page.css" in page
    assert "<table" not in page  # the per-repo ledger is gone


def test_registered_drill_downs_are_links_and_unbuilt_ones_stay_text() -> None:
    page = _page()
    hrefs = [a["href"] for t, a in _parse(page).tags if t == "a" and a.get("href")]
    assert hrefs and all(routes.is_routed(h) for h in hrefs), hrefs
    assert any(h.startswith("/repo/") for h in hrefs)
    assert not any(h.startswith(("/backlog", "/prs", "/capacity")) for h in hrefs)


def test_unrouted_numbers_render_as_text_not_dead_links(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(routes, "ROUTES", frozenset({"/now", "/history"}))
    page = _page()
    hrefs = [a["href"] for t, a in _parse(page).tags if t == "a" and a.get("href")]
    assert hrefs and not any(h.startswith(("/repo", "/issue", "/flow")) for h in hrefs)
    assert 'class="drill' not in page  # a drill-down link with nowhere to go is not offered


def test_numbers_link_to_drill_downs_and_done_is_unrecorded(drilldowns: None) -> None:
    page = _page()
    assert 'href="/issue/Senkichi/charlie-work/6"' in page  # expanded row -> the issue
    assert 'href="/repo/Senkichi/charlie-work"' in _between(
        page, 'id="pipe-in-progress"', "</div></div>"
    )
    merged = _between(page, 'id="node-merged"', "</button>")
    assert '<span class="cnt num">—</span>' in merged and "rollup is not current" in page
    done = _page(_model(flow=replace(_model().flow, done_24h=9)))
    assert '<span class="cnt num">9</span>' in _between(done, 'id="node-merged"', "</button>")


def test_no_inline_style_or_script_anywhere() -> None:
    for page in (
        _page(),
        _page(_model(items=())),
        _page(None, collector_error="Boom <b>", collector_failing_since=NOW),
        render_now(ModelState(), poll_seconds=15),
    ):
        assert not re.search(r"\sstyle\s*=", page, re.I)
        assert not re.search(r"<style", page, re.I)
        assert not re.search(r"\son[a-z]+\s*=", page, re.I)
        for tag, attrs in _parse(page).tags:
            if tag == "script":
                assert attrs.get("src", "").startswith("/static/")
        assert all(body == "" for body in _parse(page).script_bodies)


def test_inline_script_check_sees_through_spaced_end_tags() -> None:
    # Positive control for the CSP check above: the parser must find these bodies.
    assert _parse("<script>a()</script >").script_bodies == ["a()"]
    assert _parse('<script src="/static/x.js"></script>').script_bodies == [""]


def test_scripts_load_in_dependency_order() -> None:
    scripts = [a["src"] for t, a in _parse(_page()).tags if t == "script"]
    assert [s.removeprefix("/static/") for s in scripts] == [
        "theme-init.js",
        "htmx.min.js",
        "staleness.js",
        "dashboard.js",
        "now-chart.js",
        "now.js",
    ]


def test_collector_failure_banner() -> None:
    page = _page(collector_error="OSError: <nope>", collector_failing_since=NOW)
    assert 'data-alert="collector"' in page and "OSError: &lt;nope&gt;" in page
    assert "<nope>" not in page
    # Inside the swapped #now a role=alert would be re-announced on every poll; the
    # page's one polite region is the stable #client-status (plus dashboard.js's own).
    assert 'role="alert"' not in page


def test_collecting_state_renders_the_frame_and_a_skip_target() -> None:
    page = render_now(ModelState(), poll_seconds=15)
    assert "Collecting the first read of the fleet" in page
    assert 'id="needs"' in page and 'class="brand">Fleet' in page


# ---- the htmx region and the state it must not lose ----------------------------------------


def test_fragment_is_the_htmx_target_with_stable_ids() -> None:
    frag = render_fragment(ModelState(model=_model()), 15)
    assert frag.startswith('<div id="now" class="shell" hx-get="/now/fragment"')
    assert 'hx-trigger="every 15s" hx-swap="outerHTML"' in frag
    assert frag == render_fragment(ModelState(model=_model()), 15)  # ids deterministic
    ids = [a["id"] for _, a in _parse(frag).tags if a.get("id")]
    assert len(ids) == len(set(ids))


def test_every_control_has_a_stable_id_so_htmx_restores_focus() -> None:
    frag = render_fragment(ModelState(model=_model()), 15)
    for tag, attrs in _parse(frag).tags:
        if tag == "button":
            assert attrs.get("id"), attrs
    assert frag.count("<button") >= 20


def test_ui_state_is_carried_by_attributes_not_classes() -> None:
    """htmx settles ``class`` back to the server's value, so state hooks are aria/data/hidden."""
    css = static_asset("now-page.css").read_text(encoding="utf-8")
    for hook in (
        '.tile[aria-pressed="true"]',
        ".grp[data-on]",
        ".grp[data-all=",
        '.seg button[aria-pressed="true"]',
        "body.now [hidden]",
    ):
        assert hook in css, hook
    js = static_asset("now.js").read_text(encoding="utf-8")
    assert "classList" not in js and "className" not in js
    assert "afterSettle" in js and "afterSwap" in js and "beforeSwap" in js
    assert "replaceState" in js and "ResizeObserver" in js


def test_htmx_config_disables_style_settle_eval_and_script_tags() -> None:
    meta = re.search(r'<meta name="htmx-config" content="([^"]*)">', _page())
    assert meta is not None
    config = json.loads(html_mod.unescape(meta.group(1)))
    assert config["includeIndicatorStyles"] is False
    assert "style" not in config["attributesToSettle"]
    assert config["allowEval"] is False and config["allowScriptTags"] is False


def test_keyhelp_lists_the_now_keys_and_no_filter() -> None:
    page = _page()
    keys = _between(page, 'id="keyhelp"', "</aside>")
    for key in ("j / k", "Enter", "<kbd>c</kbd>", "<kbd>?</kbd>"):
        assert key in keys
    assert "filter" not in keys.lower()


# ---- assets --------------------------------------------------------------------------------


def test_static_assets_served_and_token_only() -> None:
    shared = static_asset("now.css").read_text(encoding="utf-8")
    page_css = static_asset("now-page.css").read_text(encoding="utf-8")
    for css in (shared, page_css):
        assert not re.search(r"#[0-9a-fA-F]{3,8}\b", css)
        assert "@import" not in css and "url(http" not in css
    assert not re.search(r"^\s*--[\w-]+\s*:", shared, re.M)  # the shared frame adds no properties
    declared = re.findall(r"^\s*(--[\w-]+)\s*:", page_css, re.M)
    assert declared and all(d.startswith("--n-") for d in declared), declared  # type scale only
    assert not re.search(r"--lj-[\w-]+\s*:", page_css)  # tokens are read here, never redefined
    for name in ("dashboard.js", "now.js", "now-chart.js"):
        assert static_asset(name).read_bytes()
    tokens = generate_css()
    assert "@media (prefers-color-scheme: dark)" in tokens
    assert ':root[data-theme="dark"]' in tokens and "#1E1611" in tokens  # Lamplit Paper page


def test_dashboard_scripts_avoid_dynamic_code_and_html_injection() -> None:
    for name in NOW_SCRIPTS:
        src = static_asset(name).read_text(encoding="utf-8")
        for banned in ("eval(", "new Function", "innerHTML", "outerHTML", "document.write"):
            assert banned not in src, f"{name} uses {banned}"
    assert "cw-dash-theme" in static_asset("theme-init.js").read_text(encoding="utf-8")


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


def test_page_has_no_href_for_a_dot_dot_repo_key(drilldowns: None) -> None:
    base = _model()
    model = replace(
        base,
        freshness=(RepoFreshness("../..", NOW, 24.0, False, None), *base.freshness),
        needs_me=(_item("Awaiting your verdict", repo="..", number=7), *base.needs_me),
    )
    page = _page(model)
    assert "/repo/.." not in page and "/issue/../" not in page
    assert 'href="/repo/../' not in page


def test_every_page_head_declares_an_icon_so_no_favicon_404() -> None:
    from charlie_work.dashboard.pages.nav import HEAD_BASE

    assert '<link rel="icon" href="data:,">' in HEAD_BASE
    assert HEAD_BASE in _page()


def test_age_compact_is_one_unit() -> None:
    assert [now_fmt.age_compact(s) for s in (None, 120, 3 * 3600 + 1, 9 * 86400)] == [
        "—",
        "2m",
        "3h",
        "9d",
    ]
