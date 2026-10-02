"""Bullet bars, strip plots and the chart stylesheet: literal structural assertions."""

from __future__ import annotations

import re

from charlie_work.dashboard.charts import (
    Bullet,
    Distribution,
    bullet_bar,
    bullet_bars,
    bullet_svg,
    strip_plot,
)
from charlie_work.dashboard.charts import bullet
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


def test_bullet_svg_draws_the_full_track_and_a_cap_tick() -> None:
    svg = bullet_svg(3, 4, scale=8)
    assert 'viewBox="0 0 240 16"' in svg and 'aria-hidden="true"' in svg
    assert '<rect class="trk" x="0" y="5" width="240" height="6"/>' in svg  # full-width track
    assert '<rect class="fill" x="0" y="3" width="90" height="10"/>' in svg  # 3/8 of 240
    assert '<line class="captick" x1="120" x2="120" y1="0" y2="16"/>' in svg  # 4/8 of 240


def test_bullet_svg_unknown_cap_is_a_fixed_80_percent_open_fill() -> None:
    svg = bullet_svg(2, None)
    # the fill is fixed at 80% of the track regardless of the value (there is no scale
    # to place it on); the rest is a dashed open track and no cap tick is drawn
    assert '<rect class="fill" x="0" y="3" width="192" height="10"/>' in svg
    assert '<line class="trk-open" x1="192" x2="240" y1="8" y2="8" stroke-dasharray="3 3"/>' in svg
    assert "captick" not in svg and 'class="trk"' not in svg
    assert 'width="0"' in bullet_svg(0, None)  # value 0: open track only


def test_bullet_svg_and_bullet_bar_share_the_track_geometry() -> None:
    # one implementation: the row's bar embeds exactly what bullet_svg draws
    assert bullet._track(2, 8, 8) in bullet_svg(2, 8, scale=8)
    assert bullet._track(2, 8, 8) in bullet_bar(Bullet("x", 2, 8), scale=8)
    assert bullet._track(5, 4, 10) == bullet._track(5, 4, 10)  # deterministic


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


def test_small_ratio_ticks_and_values_stay_distinct() -> None:
    from charlie_work.dashboard.charts.line import y_ticks_for
    from charlie_work.dashboard.charts.svg import step_decimals, value_text

    ticks = y_ticks_for([0.0, 0.09])
    labels = [value_text(t, "", step_decimals(ticks)) for t in ticks]
    assert labels == ["0", "0.02", "0.04", "0.06", "0.08", "0.1"]
    assert len(set(labels)) == len(labels)
    assert [value_text(v) for v in (0.09, 0.6047, 0.00001, 2.25, 3.0)] == [
        "0.09", "0.6", "0", "2.2", "3"
    ]  # fmt: skip
    assert [value_text(t, "", step_decimals(y_ticks_for([0, 40]))) for t in (0, 10)] == ["0", "10"]
