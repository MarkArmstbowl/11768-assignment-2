"""Check monthly pooled percentages and the chart rendered from them.

The shared fixture verifies required files, unchanged input, and reproducible
artifacts. Numeric expectations are independent, hand-derived literals rather
than a second implementation of the submitted aggregation. Artist structure
and text content are checked; rendered text legibility is not measured.
"""

from verifier_support import (
    assert_bar_annotations,
    assert_numbers,
    assert_sidecar,
    line_series,
    main_axes,
    submission,
)


CSV_NAME = "cohorts.csv"
EXPECTED_INPUT_SHA256 = "901ee81ab41730071e408379e561fc1ce1d84d004c0050ae29e186800e3db1db"


def test_sidecar_pooled_success_percentages_match_expected(submission):
    assert_sidecar(
        submission["sidecar"],
        {"Jan": 18, "Feb": 27.5, "Mar": 66, "Apr": 35},
    )


def test_line_shows_one_pooled_percentage_per_month(submission):
    ax = main_axes(submission["manifest"], 1)[0]
    line = line_series(ax)
    assert len(line["x"]) == 4, "one point is required for each month"
    assert_numbers(ax["xticks"], line["x"])
    assert all(a < b for a, b in zip(line["x"], line["x"][1:])), (
        "month positions must increase from Jan through Apr"
    )
    assert_numbers(line["y"], [18, 27.5, 66, 35])
    assert ax["xticklabels"] == ["Jan", "Feb", "Mar", "Apr"], (
        "the four monthly points must appear in the requested order"
    )


def test_chart_title_axis_labels_and_percentage_limits(submission):
    ax = main_axes(submission["manifest"], 1)[0]
    assert ax["title"] == "Monthly pooled success rate"
    assert ax["xlabel"] == "Month"
    assert ax["ylabel"] == "Success rate (%)"
    assert_numbers(ax["ylim"], [0, 100])


def test_every_point_has_its_one_decimal_percentage_annotation(submission):
    ax = main_axes(submission["manifest"], 1)[0]
    assert_bar_annotations(ax, ["18.0%", "27.5%", "66.0%", "35.0%"])
