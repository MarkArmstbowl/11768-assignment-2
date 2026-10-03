"""Check sensor traces, aligned x scales, and figure-level label scope.

Artist structure and label positions are checked without a font-size threshold.
The shared fixture binds these artists to the reproducible delivered image.
"""

from verifier_support import (
    assert_numbers,
    assert_sidecar,
    main_axes,
    submission,
)


CSV_NAME = "sensors.csv"
EXPECTED_INPUT_SHA256 = "26dc2ca8ac448149723650553647119bdbb173733f42345fddda5f920f120fd0"


def _ordered_panels(submission):
    return sorted(
        main_axes(submission["manifest"], 2),
        key=lambda ax: ax["position"][1],
        reverse=True,
    )


def test_sidecar_contains_both_six_observation_traces(submission):
    assert_sidecar(
        submission["sidecar"],
        {
            "time_s": [0, 1, 2, 3, 4, 5],
            "sensor_a_v": [0.2, 0.6, 1.2, 0.7, 0.1, -0.3],
            "sensor_b_v": [1.2, 0.9, 0.5, 0.1, -0.4, -0.8],
        },
    )


def test_panels_are_arranged_in_two_rows(submission):
    top, bottom = _ordered_panels(submission)
    tx, ty, tw, th = top["position"]
    bx, by, bw, bh = bottom["position"]
    assert by + bh <= ty, "Sensor A and Sensor B must occupy separate rows"
    horizontal_overlap = min(tx + tw, bx + bw) - max(tx, bx)
    assert horizontal_overlap > 0.5 * min(tw, bw), "the panels must occupy the same column"


def test_each_panel_has_its_requested_title_and_sensor_series(submission):
    top, bottom = _ordered_panels(submission)
    for ax, title, expected in (
        (top, "Sensor A", [0.2, 0.6, 1.2, 0.7, 0.1, -0.3]),
        (bottom, "Sensor B", [1.2, 0.9, 0.5, 0.1, -0.4, -0.8]),
    ):
        assert ax["title"] == title
        assert len(ax["lines"]) == 1, "each sensor panel must contain one sensor line"
        assert not ax["containers"] and not ax["collections"] and not ax["images"], (
            "each sensor panel must use a line plot"
        )
        line = ax["lines"][0]
        assert line["type"] == "line"
        assert line["linestyle"].casefold() not in {"none", "", " "}, (
            "the observations must be connected by a line"
        )
        assert line["linewidth"] > 0, "the sensor line must be visible"
        assert_numbers(line["x"], [0, 1, 2, 3, 4, 5])
        assert_numbers(line["y"], expected)


def test_panels_share_the_x_axis_scale(submission):
    top, bottom = _ordered_panels(submission)
    assert top["xscale"] == bottom["xscale"], "both panels must use the same x scale"
    assert_numbers(top["xlim"], bottom["xlim"])
    # Equal limits must also map to aligned horizontal positions in the PNG.
    assert abs(top["position"][0] - bottom["position"][0]) < 0.01
    assert abs(top["position"][2] - bottom["position"][2]) < 0.01


def test_labels_are_shared_once_and_positioned_beside_the_pair(submission):
    manifest = submission["manifest"]
    axes = main_axes(manifest, 2)
    for ax in axes:
        assert ax["xlabel"] == "", "do not add per-panel x-axis labels"
        assert ax["ylabel"] == "", "do not add per-panel y-axis labels"

    texts = manifest["figure_texts"]
    xlabels = [entry for entry in texts if entry["text"] == "Elapsed time (s)"]
    ylabels = [entry for entry in texts if entry["text"] == "Amplitude (V)"]
    assert len(xlabels) == 1, "exactly one figure-level Elapsed time (s) label is required"
    assert len(ylabels) == 1, "exactly one figure-level Amplitude (V) label is required"
    xlabel, ylabel = xlabels[0], ylabels[0]
    assert xlabel["visible"] is True and ylabel["visible"] is True

    left = min(ax["position"][0] for ax in axes)
    right = max(ax["position"][0] + ax["position"][2] for ax in axes)
    bottom = min(ax["position"][1] for ax in axes)
    top = max(ax["position"][1] + ax["position"][3] for ax in axes)
    xx, xy = xlabel["position"]
    yx, yy = ylabel["position"]
    assert left <= xx <= right and 0 <= xy < bottom, (
        "the figure-level x-axis label must sit beneath the pair of panels"
    )
    assert 0 <= yx < left and bottom <= yy <= top, (
        "the figure-level y-axis label must sit alongside the pair of panels"
    )
