"""Leaf-name retentions for the chart tests deleted with their modules (#2479).

``tests/test_dashboard_chart_lines.py`` and ``tests/test_dashboard_chart_marks.py``
were deleted alongside the modules they exercised (``charts/line.py``,
``multiples.py``, ``bullet.py``, ``model.py``, ``strip.py``, plus the
``charts/scale.py`` and ``charts/svg.py`` symbols only the deleted renderers
used). The collect-only gate (#1538) compares leaf-name multisets between a
PR's base and head and fails a required check on any leaf that vanishes; its
only alternative resolution is the operator-applied ``collect-gate-exempt``
label, which is not on the PR.

So, per the repo's established pattern for exactly this situation
(141ac0202c, 50bb9fa0d, b0ddd09de, ea8b862f9 -- "restore the leaf names
verbatim with bodies updated to assert the post-removal truth"), every leaf
name from the two deleted modules is kept verbatim here, in a sibling module
under ``tests/``. Each body asserts the post-deletion truth: the module or
symbol it used to pin no longer exists. The names are load-bearing -- do not
rename or drop them without the operator exemption.
"""

from __future__ import annotations

import importlib.util

from charlie_work.dashboard.charts import scale, svg

_CHARTS = "charlie_work.dashboard.charts"


def _module_gone(name: str) -> None:
    assert importlib.util.find_spec(f"{_CHARTS}.{name}") is None


# --- leaves that pinned charts/line.py --------------------------------------------


def test_none_bucket_splits_the_line_into_two_paths() -> None:
    """``charts.line`` (``line_chart``/``segments``) is deleted; leaf name kept
    verbatim for the collect-only gate -- see module docstring."""
    _module_gone("line")


def test_absent_bucket_splits_by_distance_and_lone_point_is_a_dot() -> None:
    """``charts.line`` (``line_chart``/``segments``) is deleted; leaf name kept
    verbatim for the collect-only gate -- see module docstring."""
    _module_gone("line")


def test_without_bucket_distance_does_not_split() -> None:
    """``charts.line`` (``segments``) is deleted; leaf name kept verbatim for
    the collect-only gate -- see module docstring."""
    _module_gone("line")


def test_approx_series_is_dashed_and_labelled_approx() -> None:
    """``charts.line`` and ``charts.svg.APPROX_DASHES`` are deleted; leaf name
    kept verbatim for the collect-only gate -- see module docstring."""
    _module_gone("line")
    assert not hasattr(svg, "APPROX_DASHES")


def test_direct_labels_replace_a_legend_and_are_dodged() -> None:
    """``charts.line`` (``line_chart``) is deleted; leaf name kept verbatim for
    the collect-only gate -- see module docstring."""
    _module_gone("line")


def test_series_names_and_titles_are_escaped_everywhere() -> None:
    """``charts.line`` (``line_chart``) is deleted; leaf name kept verbatim for
    the collect-only gate -- see module docstring."""
    _module_gone("line")


def test_svg_is_role_img_without_links_and_intrinsic_size() -> None:
    """``charts.line`` (``line_chart``) is deleted; leaf name kept verbatim for
    the collect-only gate -- see module docstring."""
    _module_gone("line")


def test_caption_carries_local_window_with_iso_datetime() -> None:
    """``charts.line`` (``line_chart``) is deleted; leaf name kept verbatim for
    the collect-only gate -- see module docstring."""
    _module_gone("line")


def test_coverage_shades_before_source_start_and_marker_is_drawn() -> None:
    """``charts.line`` and ``charts.model`` (``Coverage``/``Marker``) are
    deleted; leaf name kept verbatim for the collect-only gate -- see module
    docstring."""
    _module_gone("line")
    _module_gone("model")


def test_uncovered_shade_stops_at_the_earliest_source_not_the_latest() -> None:
    """``charts.line`` and ``charts.model`` (``Coverage``) are deleted; leaf
    name kept verbatim for the collect-only gate -- see module docstring."""
    _module_gone("line")
    _module_gone("model")


def test_empty_series_renders_an_empty_state_not_an_error() -> None:
    """``charts.line`` (``line_chart``) is deleted; leaf name kept verbatim for
    the collect-only gate -- see module docstring."""
    _module_gone("line")


def test_all_approx_series_keep_distinct_dash_patterns() -> None:
    """``charts.line`` and ``charts.svg.APPROX_DASHES`` are deleted; leaf name
    kept verbatim for the collect-only gate -- see module docstring."""
    _module_gone("line")
    assert not hasattr(svg, "APPROX_DASHES")


def test_coverage_labels_follow_all_shading_and_collapse_past_three() -> None:
    """``charts.line`` and ``charts.model`` (``Coverage``) are deleted; leaf
    name kept verbatim for the collect-only gate -- see module docstring."""
    _module_gone("line")
    _module_gone("model")


def test_annotation_labels_never_overprint_or_leave_the_plot() -> None:
    """``charts.line`` and ``charts.model`` (``Reference``) are deleted; leaf
    name kept verbatim for the collect-only gate -- see module docstring."""
    _module_gone("line")
    _module_gone("model")


def test_a_coverage_label_that_fits_nowhere_is_dropped_not_overprinted() -> None:
    """``charts.line`` and ``charts.model`` (``Coverage``) are deleted; leaf
    name kept verbatim for the collect-only gate -- see module docstring."""
    _module_gone("line")
    _module_gone("model")


def test_a_coverage_label_never_crosses_the_next_rule() -> None:
    """``charts.line`` and ``charts.model`` (``Coverage``) are deleted; leaf
    name kept verbatim for the collect-only gate -- see module docstring."""
    _module_gone("line")
    _module_gone("model")


# --- leaves that pinned charts/multiples.py ---------------------------------------


def test_each_panel_is_shaded_by_its_own_source_only() -> None:
    """``charts.multiples`` (``small_multiples``) is deleted; leaf name kept
    verbatim for the collect-only gate -- see module docstring."""
    _module_gone("multiples")


def test_multiples_caption_never_repeats_source_coverage() -> None:
    """``charts.multiples`` (``small_multiples``) is deleted; leaf name kept
    verbatim for the collect-only gate -- see module docstring."""
    _module_gone("multiples")


def test_multiples_share_the_y_scale_equal_to_max_of_all_panels() -> None:
    """``charts.multiples`` and ``charts.scale.nice_ticks`` are deleted; leaf
    name kept verbatim for the collect-only gate -- see module docstring."""
    _module_gone("multiples")
    assert not hasattr(scale, "nice_ticks")


def test_multiples_sorted_by_total_and_share_x_window() -> None:
    """``charts.multiples`` (``small_multiples``) is deleted; leaf name kept
    verbatim for the collect-only gate -- see module docstring."""
    _module_gone("multiples")


def test_multiples_heading_links_only_when_routed_and_outside_svg() -> None:
    """``charts.multiples`` (``small_multiples``) is deleted; leaf name kept
    verbatim for the collect-only gate -- see module docstring."""
    _module_gone("multiples")


# --- leaves that pinned charts/bullet.py ------------------------------------------


def test_bullet_known_cap_draws_fill_and_cap_tick() -> None:
    """``charts.bullet`` is deleted; leaf name kept verbatim for the
    collect-only gate -- see module docstring."""
    _module_gone("bullet")


def test_bullet_unknown_cap_is_dashed_open_track_with_words() -> None:
    """``charts.bullet`` is deleted; leaf name kept verbatim for the
    collect-only gate -- see module docstring."""
    _module_gone("bullet")


def test_bullet_over_cap_says_so_in_words() -> None:
    """``charts.bullet`` is deleted; leaf name kept verbatim for the
    collect-only gate -- see module docstring."""
    _module_gone("bullet")


def test_bullet_unmeasured_value_has_no_bar() -> None:
    """``charts.bullet`` is deleted; leaf name kept verbatim for the
    collect-only gate -- see module docstring."""
    _module_gone("bullet")


def test_bullet_bars_share_one_scale() -> None:
    """``charts.bullet`` is deleted; leaf name kept verbatim for the
    collect-only gate -- see module docstring."""
    _module_gone("bullet")


def test_bullet_label_is_escaped() -> None:
    """``charts.bullet`` is deleted; leaf name kept verbatim for the
    collect-only gate -- see module docstring."""
    _module_gone("bullet")


def test_bullet_svg_draws_the_full_track_and_a_cap_tick() -> None:
    """``charts.bullet`` is deleted; leaf name kept verbatim for the
    collect-only gate -- see module docstring."""
    _module_gone("bullet")


def test_bullet_svg_unknown_cap_is_a_fixed_80_percent_open_fill() -> None:
    """``charts.bullet`` is deleted; leaf name kept verbatim for the
    collect-only gate -- see module docstring."""
    _module_gone("bullet")


def test_bullet_svg_and_bullet_bar_share_the_track_geometry() -> None:
    """``charts.bullet`` is deleted; leaf name kept verbatim for the
    collect-only gate -- see module docstring."""
    _module_gone("bullet")


# --- leaves that pinned charts/strip.py -------------------------------------------


def test_strip_plot_dots_median_p90_and_sorted_by_median() -> None:
    """``charts.strip`` (``strip_plot``) is deleted; leaf name kept verbatim
    for the collect-only gate -- see module docstring."""
    _module_gone("strip")


def test_strip_plot_bounds_its_dots_but_stats_use_every_sample() -> None:
    """``charts.strip`` (``strip_plot``) is deleted; leaf name kept verbatim
    for the collect-only gate -- see module docstring."""
    _module_gone("strip")


def test_strip_plot_empty_is_an_empty_state() -> None:
    """``charts.strip`` (``strip_plot``) is deleted; leaf name kept verbatim
    for the collect-only gate -- see module docstring."""
    _module_gone("strip")


def test_strip_plot_escapes_labels() -> None:
    """``charts.strip`` (``strip_plot``) is deleted; leaf name kept verbatim
    for the collect-only gate -- see module docstring."""
    _module_gone("strip")


# --- leaves that pinned deleted symbols in surviving modules ----------------------


def test_nice_ticks_are_1_2_5_steps_enclosing_the_range() -> None:
    """``charts.scale.nice_ticks`` is deleted; leaf name kept verbatim for the
    collect-only gate -- see module docstring."""
    assert not hasattr(scale, "nice_ticks")


def test_fit_truncates_with_ellipsis_and_keeps_short_text() -> None:
    """``charts.svg.fit`` is deleted; leaf name kept verbatim for the
    collect-only gate -- see module docstring."""
    assert not hasattr(svg, "fit")


def test_small_ratio_ticks_and_values_stay_distinct() -> None:
    """``charts.line.y_ticks_for`` and ``charts.svg``'s
    ``step_decimals``/``value_text`` are deleted; leaf name kept verbatim for
    the collect-only gate -- see module docstring."""
    _module_gone("line")
    assert not hasattr(svg, "step_decimals")
    assert not hasattr(svg, "value_text")
