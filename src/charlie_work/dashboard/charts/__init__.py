"""Server-rendered SVG chart primitives for the drill-downs.

Pure functions from frozen value objects to escaped markup strings: no I/O, no clock, no
colour literals (classes map to ``--lj-*`` tokens in ``/static/charts.css``). Times are
drawn in the display zone (host-local by default) with UTC ISO kept in ``<time datetime>``.
"""
