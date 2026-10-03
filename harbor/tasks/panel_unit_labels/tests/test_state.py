"""Check direct channel values and independent labels on the two panels.

The shared submission fixture checks delivered artifacts, unchanged input,
isolated replay, and image/sidecar reproducibility. These checks inspect the
replayed figure's artists; they do not measure rendered text legibility.
"""

from verifier_support import (
    assert_numbers,
    assert_sidecar,
    bar_series,
    main_axes,
    submission,
)


CSV_NAME = "channels.csv"
EXPECTED_INPUT_SHA256 = "57ac66acff3b8762729723227c31017ff3b5c0c6c5efaae0c5d4cbf366ce2db2"
CHANNELS = ["Retail", "Wholesale", "Online", "Partners"]


def _ordered_panels(submission):
    return sorted(main_axes(submission["manifest"], 2), key=lambda ax: ax["position"][0])


def test_sidecar_channel_values_match_the_input(submission):
    assert_sidecar(
        submission["sidecar"],
        {
            "channels": ["Retail", "Wholesale", "Online", "Partners"],
            "orders": [120, 75, 210, 45],
            "refund_rate_pct": [4, 8, 3, 6],
        },
    )


def test_panels_are_arranged_in_one_row(submission):
    left, right = _ordered_panels(submission)
    lx, ly, lw, lh = left["position"]
    rx, ry, rw, rh = right["position"]
    assert lx + lw <= rx, "the two panels must occupy separate columns"
    vertical_overlap = min(ly + lh, ry + rh) - max(ly, ry)
    assert vertical_overlap > 0.5 * min(lh, rh), "the panels must occupy the same row"


def test_each_panel_plots_its_four_values_in_channel_order(submission):
    left, right = _ordered_panels(submission)
    for ax, expected in ((left, [120, 75, 210, 45]), (right, [4, 8, 3, 6])):
        series = bar_series(ax)
        assert_numbers(series["values"], expected)
        assert ax["xticklabels"] == CHANNELS, "channels must have the requested left-to-right order"
        offsets = series["offsets"]
        assert len(offsets) == 4 and all(a < b for a, b in zip(offsets, offsets[1:])), (
            "the four bar heights must be associated with the channels in left-to-right order"
        )


def test_both_panels_have_their_own_titles_and_axis_labels(submission):
    left, right = _ordered_panels(submission)
    assert left["title"] == "Orders by channel"
    assert left["xlabel"] == "Sales channel"
    assert left["ylabel"] == "Orders (count)"
    assert right["title"] == "Refund rate by channel"
    assert right["xlabel"] == "Sales channel"
    assert right["ylabel"] == "Refund rate (%)"
