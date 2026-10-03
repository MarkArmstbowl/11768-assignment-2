"""Check chronological warm-up observations and semantic quantity/unit labels.

Equivalent clear axis-label wording is accepted. Artifact integrity and replay
are handled by the shared fixture; rendered font size is not graded here.
"""

import re
import unicodedata

from verifier_support import (
    assert_numbers,
    assert_sidecar,
    line_series,
    main_axes,
    submission,
)


CSV_NAME = "warmup.csv"
EXPECTED_INPUT_SHA256 = "04f6c48b846f51dd282fb805641ae873ede9d96884a56603c9317cb7cabda5b3"


def test_sidecar_contains_minutes_and_celsius_observations(submission):
    assert_sidecar(
        submission["sidecar"],
        {
            "elapsed_minutes": [0, 1.5, 3, 4.5, 6, 7.5],
            "temperature_c": [18, 20, 25, 31, 36, 39],
        },
    )


def test_one_line_with_circular_markers_uses_all_observations(submission):
    ax = main_axes(submission["manifest"], 1)[0]
    line = line_series(ax)
    assert_numbers(line["x"], [0, 1.5, 3, 4.5, 6, 7.5])
    assert_numbers(line["y"], [18, 20, 25, 31, 36, 39])
    assert ax["title"] == "Temperature during warm-up"


def test_separate_axis_labels_express_the_required_quantities_and_units(submission):
    ax = main_axes(submission["manifest"], 1)[0]
    # For example, the standard DEGREE CELSIUS symbol "℃" normalizes to "°C".
    xlabel = unicodedata.normalize("NFKC", ax["xlabel"]).casefold()
    ylabel = unicodedata.normalize("NFKC", ax["ylabel"]).casefold()
    assert re.search(r"\b(?:time|elapsed|duration)\b", xlabel), (
        "the separate x-axis label must express elapsed time"
    )
    assert re.search(r"\b(?:mins?|minutes?)\b", xlabel), (
        "the separate x-axis label must express minutes, not seconds"
    )
    assert re.search(r"\b(?:temperature|temp)\b", ylabel), (
        "the separate y-axis label must express temperature"
    )
    assert re.search(r"(?:°\s*c\b|\bc\b|\b(?:celsius|centigrade)\b)", ylabel), (
        "the separate y-axis label must express degrees Celsius"
    )
