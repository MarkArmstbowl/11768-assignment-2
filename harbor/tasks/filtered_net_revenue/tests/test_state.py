"""Check completed-order totals and the chart that the submitted script renders.

The shared fixture checks file integrity, unchanged input, and reproducible
artifacts. These assertions check the declared and plotted numbers, chart
structure, axis labels, and annotations. They do not measure rendered text
legibility; the illegible mutant deliberately exposes that limitation.
"""

from verifier_support import (
    assert_bar_annotations,
    assert_numbers,
    assert_sidecar,
    bar_series,
    main_axes,
    submission,
)


CSV_NAME = "orders.csv"
EXPECTED_INPUT_SHA256 = "9d8ae4904045a2000e3b2f85529f0d18a6bffb08132f4522fead519cbafe7605"


def test_sidecar_net_revenue_matches_expected(submission):
    # Hand-derived from completed orders, independent of the submitted code.
    assert_sidecar(
        submission["sidecar"],
        {"North": 145, "South": 0, "East": 95, "West": 105},
    )


def test_bars_show_completed_net_revenue(submission):
    ax = main_axes(submission["manifest"], 1)[0]
    bars = bar_series(ax)
    assert_numbers(bars["values"], [145, 0, 95, 105])
    assert ax["xticklabels"] == ["North", "South", "East", "West"], (
        "the four region bars must appear in the requested order"
    )


def test_chart_title_axis_labels_and_zero_baseline(submission):
    ax = main_axes(submission["manifest"], 1)[0]
    assert ax["title"] == "Completed-order net revenue"
    assert ax["xlabel"] == "Region"
    assert ax["ylabel"] == "Net revenue (USD)"
    assert_numbers([ax["ylim"][0]], [0])


def test_every_bar_has_its_integer_annotation_including_zero(submission):
    ax = main_axes(submission["manifest"], 1)[0]
    assert_bar_annotations(ax, ["145", "0", "95", "105"])
