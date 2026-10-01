"""Bullet bars, strip plots and the chart stylesheet: literal structural assertions."""

from __future__ import annotations

import re

from charlie_work.dashboard.charts import Bullet, Distribution, bullet_bar, bullet_bars, strip_plot
from charlie_work.dashboard.charts.strip import nearest_rank
from charlie_work.dashboard.theme import generate_css, static_asset


def test_bullet_known_cap_draws_fill_and_cap_tick() -> None:
    html = bullet_bar(Bullet("workers", 3, 4))
    assert '<rect class="fill" x="0" y="3" width="180" height="10"/>' in html
    assert '<line class="captick" x1="240" x2="240"' in html
    assert ">3 / 4</span>" in html and "over cap" not in html


def test_bullet_unknown_cap_is_dashed_open_track_with_words() -> None:
    html = bullet_bar(Bullet("reviewers", 2, None))
    assert re.search(r'class="trk-open"[^>]*stroke-dasharray="3 3"', html)
    assert "cap not reported" in html and "captick" not in html


def test_bullet_over_cap_says_so_in_words() -> None:
    html = bullet_bar(Bullet("runners", 5, 4))
    assert 'class="fill hot"' in html and "over cap" in html


def test_bullet_unmeasured_value_has_no_bar() -> None:
    html = bullet_bar(Bullet("ci", None, 4))
    assert "not measured" in html and "<svg" not in html


def test_bullet_bars_share_one_scale() -> None:
    html = bullet_bars((Bullet("a", 2, 4), Bullet("b", 8, 8)))
    # scale = 8, so a's fill is 2/8 of the track and its cap tick sits at the midpoint.
    assert 'width="60" height="10"' in html
    assert '<line class="captick" x1="120"' in html


def test_bullet_label_is_escaped() -> None:
    assert "&lt;x&gt;" in bullet_bar(Bullet("<x>", 1, 2))


def test_nearest_rank_is_an_observed_sample() -> None:
    assert nearest_rank((1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 90) == 9
    assert nearest_rank((5,), 90) == 5


def test_strip_plot_dots_median_p90_and_sorted_by_median() -> None:
    rows = (
        Distribution("fast", (60, 120, 180)),
        Distribution("slow", (3600, 7200, 10800), approx=True),
    )
    html = strip_plot(rows, "lead time")
    assert html.count('<circle class="sample"') == 6
    assert html.count('<line class="median"') == 2 and html.count('<line class="p90"') == 2
    assert html.index(">slow</text>") < html.index(">fast</text>")
    assert '<g class="dist approx">' in html
    assert "median 2h00m · p90 3h00m · n 3 approx." in html
    assert "median 2m00s · p90 3m00s · n 3" in html
    assert ">1m</text>" in html and ">4h</text>" in html  # log duration ticks
    assert 'role="img"' in html and "<a " not in html


def test_strip_plot_empty_is_an_empty_state() -> None:
    assert "no samples in this window" in strip_plot((Distribution("x", ()),), "t")


def test_strip_plot_escapes_labels() -> None:
    html = strip_plot((Distribution("<r>", (5,)),), "t")
    assert "<r>" not in html and "&lt;r&gt;" in html


def test_charts_css_uses_only_tokens_and_px_text_at_least_11() -> None:
    css = static_asset("charts.css").read_text(encoding="utf-8")
    defined = set(re.findall(r"(--[\w-]+)\s*:", generate_css()))
    used = set(re.findall(r"var\((--[\w-]+)", css))
    assert used and used <= defined, sorted(used - defined)
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b|rgba?\(", css)
    sizes = re.findall(r"font-size:\s*([\d.]+)px", css)
    assert sizes and all(float(s) >= 11 for s in sizes)
