"""Offline regression tests for the authored Harbor verifier's task contract.

Capture actual Matplotlib artists in memory, then call the packaged production
assertions. These tests do not run a submitted script, use Docker, make model
calls, or create submitted artifacts. Harbor's replay/artifact fixture remains
covered by the retained Harbor jobs rather than being simulated here.
"""

import copy
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = ROOT / "harbor" / "tasks"
TASK_IDS = (
    "filtered_net_revenue",
    "panel_unit_labels",
    "pooled_success_rates",
    "semantic_unit_labels",
    "shared_axis_scope",
)


def _load_module(name, path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class HarborVerifierContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Keep Matplotlib's font cache outside both the repository and HOME.
        cls.cache = tempfile.TemporaryDirectory(prefix="hw2-mpl-contract-")
        cls.addClassCleanup(cls.cache.cleanup)
        cls.environment = patch.dict(os.environ, {"MPLCONFIGDIR": cls.cache.name})
        cls.environment.start()
        cls.addClassCleanup(cls.environment.stop)
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        cls.Figure = Figure
        cls.FigureCanvasAgg = FigureCanvasAgg

        # Pytest is a Docker-only verifier dependency, not a local dependency.
        # Its fixture decorator is irrelevant when calling the assertions with
        # an already captured submission. Stub only that decorator, and only
        # while importing the production modules; do not change their source.
        pytest_module = sys.modules.get("pytest")
        if pytest_module is None:
            pytest_module = types.ModuleType("pytest")
            pytest_module.fixture = lambda *args, **kwargs: lambda function: function
        with patch.dict(sys.modules, {"pytest": pytest_module}):
            cls.helper = _load_module(
                "hw2_contract_support",
                TASK_ROOT / TASK_IDS[0] / "tests" / "verifier_support.py",
            )
            cls.capture = _load_module(
                "hw2_contract_capture",
                TASK_ROOT / TASK_IDS[0] / "tests" / "figure_manifest.py",
            )
            with patch.dict(sys.modules, {"verifier_support": cls.helper}):
                cls.states = {
                    task_id: _load_module(
                        "hw2_contract_" + task_id,
                        TASK_ROOT / task_id / "tests" / "test_state.py",
                    )
                    for task_id in TASK_IDS
                }

    def figure(self):
        figure = self.Figure(figsize=(8, 6))
        self.FigureCanvasAgg(figure)
        self.addCleanup(figure.clear)
        return figure

    def manifest(self, figure):
        result = self.capture.manifest_from_figure(figure)
        # Match the public metadata added by the packaged sitecustomize hook,
        # without installing its global savefig monkeypatch or writing files.
        result["figure_texts"] = []
        for artist in figure.texts:
            position = figure.transFigure.inverted().transform(
                artist.get_transform().transform(artist.get_position())
            )
            result["figure_texts"].append({
                "text": artist.get_text(),
                "position": [float(value) for value in position],
                "visible": artist.get_visible(),
                "rotation": float(artist.get_rotation()),
            })
        for entry, ax in zip(result["axes"], figure.axes):
            entry["xticks"] = [float(value) for value in ax.get_xticks()]
        return result

    def net_revenue(self, *, separate=False, indices=(0, 1, 2, 3)):
        figure = self.figure()
        ax = figure.subplots()
        values = [145, 0, 95, 105]
        if separate:
            for index in indices:
                ax.bar(index, values[index])
        else:
            ax.bar(list(indices), [values[index] for index in indices])
        ax.set_xticks([0, 1, 2, 3], ["North", "South", "East", "West"])
        ax.set(title="Completed-order net revenue", xlabel="Region", ylabel="Net revenue (USD)")
        ax.set_ylim(0, 160)
        for index in indices:
            ax.text(index, values[index], str(values[index]))
        return figure, ax

    def semantic_units(self, *, xlabel="Elapsed time (min)", ylabel="Temperature (℃)"):
        figure = self.figure()
        ax = figure.subplots()
        ax.plot([0, 1.5, 3, 4.5, 6, 7.5], [18, 20, 25, 31, 36, 39], marker="o")
        ax.set(title="Temperature during warm-up", xlabel=xlabel, ylabel=ylabel)
        return figure, ax

    def shared_sensor_panels(self):
        figure = self.figure()
        top, bottom = figure.subplots(2, 1, sharex=True)
        top.plot([0, 1, 2, 3, 4, 5], [0.2, 0.6, 1.2, 0.7, 0.1, -0.3])
        bottom.plot([0, 1, 2, 3, 4, 5], [1.2, 0.9, 0.5, 0.1, -0.4, -0.8])
        top.set_title("Sensor A")
        bottom.set_title("Sensor B")
        figure.text(0.5, 0.04, "Elapsed time (s)")
        figure.text(0.03, 0.5, "Amplitude (V)", rotation=90)
        return figure, top, bottom

    def assert_net_values(self, figure):
        self.states["filtered_net_revenue"].test_bars_show_completed_net_revenue(
            {"manifest": self.manifest(figure)}
        )

    def test_packaged_helper_copies_remain_identical(self):
        canonical = (TASK_ROOT / TASK_IDS[0] / "tests" / "verifier_support.py").read_bytes()
        for task_id in TASK_IDS[1:]:
            with self.subTest(task=task_id):
                self.assertEqual(
                    (TASK_ROOT / task_id / "tests" / "verifier_support.py").read_bytes(),
                    canonical,
                )

    def test_unspecified_log_scale_keeps_semantic_line_correct(self):
        figure, ax = self.semantic_units()
        ax.set_yscale("log")
        submission = {"manifest": self.manifest(figure)}
        self.assertEqual(self.helper.main_axes(submission["manifest"], 1)[0]["yscale"], "log")
        state = self.states["semantic_unit_labels"]
        state.test_one_line_with_circular_markers_uses_all_observations(submission)
        state.test_separate_axis_labels_express_the_required_quantities_and_units(submission)

    def test_axis_shape_and_extra_axes_checks_are_not_relaxed(self):
        figure, _ = self.semantic_units()
        manifest = self.manifest(figure)
        for field, value in (("n_axes", 2), ("n_axes_raw", 2), ("n_child_axes", 1)):
            with self.subTest(field=field):
                invalid = copy.deepcopy(manifest)
                invalid[field] = value
                with self.assertRaises(AssertionError):
                    self.helper.main_axes(invalid, 1)
        for field, value in (("role", "colorbar"), ("projection", "polar")):
            with self.subTest(field=field):
                invalid = copy.deepcopy(manifest)
                invalid["axes"][0][field] = value
                with self.assertRaises(AssertionError):
                    self.helper.main_axes(invalid, 1)

    def test_matching_nonlinear_shared_scale_is_accepted(self):
        figure, top, _ = self.shared_sensor_panels()
        top.set_xscale("symlog")
        submission = {"manifest": self.manifest(figure)}
        state = self.states["shared_axis_scope"]
        state.test_panels_are_arranged_in_two_rows(submission)
        state.test_each_panel_has_its_requested_title_and_sensor_series(submission)
        state.test_panels_share_the_x_axis_scale(submission)
        state.test_labels_are_shared_once_and_positioned_beside_the_pair(submission)

    def test_shared_scale_mismatches_are_still_rejected(self):
        figure, _, _ = self.shared_sensor_panels()
        manifest = self.manifest(figure)
        for field, value in (("xscale", "symlog"), ("xlim", [0, 10])):
            with self.subTest(field=field):
                invalid = copy.deepcopy(manifest)
                invalid["axes"][1][field] = value
                with self.assertRaises(AssertionError):
                    self.states["shared_axis_scope"].test_panels_share_the_x_axis_scale(
                        {"manifest": invalid}
                    )

    def test_explicit_percentage_axis_limits_are_still_required(self):
        figure = self.figure()
        ax = figure.subplots()
        ax.plot([0, 1, 2, 3], [18, 27.5, 66, 35], marker="o")
        ax.set(title="Monthly pooled success rate", xlabel="Month", ylabel="Success rate (%)")
        ax.set_ylim(0, 100)
        state = self.states["pooled_success_rates"]
        state.test_chart_title_axis_labels_and_percentage_limits({"manifest": self.manifest(figure)})
        ax.set_ylim(0, 80)
        with self.assertRaises(AssertionError):
            state.test_chart_title_axis_labels_and_percentage_limits(
                {"manifest": self.manifest(figure)}
            )

    def test_single_and_four_separate_bar_calls_are_accepted(self):
        for separate in (False, True):
            with self.subTest(separate=separate):
                figure, _ = self.net_revenue(separate=separate)
                self.assert_net_values(figure)

    def test_reverse_creation_order_preserves_values_and_offsets(self):
        figure, _ = self.net_revenue(separate=True, indices=(3, 1, 0, 2))
        manifest = self.manifest(figure)
        bars = self.helper.bar_series(manifest["axes"][0])
        self.helper.assert_numbers(bars["values"], [145, 0, 95, 105])
        self.helper.assert_numbers(bars["offsets"], [-0.4, 0.6, 1.6, 2.6])
        self.helper.assert_numbers(bars["baselines"], [0, 0, 0, 0])
        self.assert_net_values(figure)

    def test_two_panel_task_accepts_separate_calls_in_reverse_creation_order(self):
        figure = self.figure()
        left, right = figure.subplots(1, 2)
        channels = ["Retail", "Wholesale", "Online", "Partners"]
        for ax, title, ylabel, values in (
            (left, "Orders by channel", "Orders (count)", [120, 75, 210, 45]),
            (right, "Refund rate by channel", "Refund rate (%)", [4, 8, 3, 6]),
        ):
            for index in (3, 1, 0, 2):
                ax.bar(index, values[index])
            ax.set_xticks([0, 1, 2, 3], channels)
            ax.set(title=title, xlabel="Sales channel", ylabel=ylabel)
        submission = {"manifest": self.manifest(figure)}
        state = self.states["panel_unit_labels"]
        state.test_panels_are_arranged_in_one_row(submission)
        state.test_each_panel_plots_its_four_values_in_channel_order(submission)
        state.test_both_panels_have_their_own_titles_and_axis_labels(submission)

    def test_duplicate_bar_positions_are_still_rejected(self):
        figure = self.figure()
        ax = figure.subplots()
        for value in (145, 0, 95, 105):
            ax.bar(0, value)
        with self.assertRaises(AssertionError):
            self.helper.bar_series(self.manifest(figure)["axes"][0])

    def test_bar_array_lengths_must_match(self):
        figure, _ = self.net_revenue()
        ax = self.manifest(figure)["axes"][0]
        for field in ("offsets", "values", "baselines"):
            with self.subTest(field=field):
                invalid = copy.deepcopy(ax)
                invalid["containers"][0][field].pop()
                with self.assertRaises(AssertionError):
                    self.helper.bar_series(invalid)

    def test_legitimate_zero_baseline_guide_is_accepted(self):
        figure, ax = self.net_revenue(separate=True)
        ax.axhline(0, color="black", linewidth=0.8)
        self.assert_net_values(figure)

    def test_arbitrary_additional_line_is_rejected(self):
        for kind in ("positive_baseline", "data_line", "zero_markers"):
            with self.subTest(kind=kind):
                figure, ax = self.net_revenue()
                if kind == "positive_baseline":
                    ax.axhline(10)
                elif kind == "data_line":
                    ax.plot([0, 1, 2, 3], [1, 2, 3, 4])
                else:
                    ax.plot([0, 1], [0, 0], marker="o")
                with self.assertRaises(AssertionError):
                    self.assert_net_values(figure)

    def test_wrong_bar_heights_are_rejected_by_task_expected_values(self):
        figure, ax = self.net_revenue()
        ax.containers[0].patches[0].set_height(345)
        with self.assertRaises(AssertionError):
            self.assert_net_values(figure)

    def test_missing_and_extra_bars_are_rejected_by_task_expected_count(self):
        figure, _ = self.net_revenue(separate=True, indices=(0, 1, 2))
        with self.assertRaises(AssertionError):
            self.assert_net_values(figure)
        figure, ax = self.net_revenue()
        ax.bar(4, 10)
        with self.assertRaises(AssertionError):
            self.assert_net_values(figure)

    def test_horizontal_bars_are_still_rejected(self):
        figure = self.figure()
        ax = figure.subplots()
        ax.barh([0, 1, 2, 3], [145, 0, 95, 105])
        with self.assertRaises(AssertionError):
            self.helper.bar_series(self.manifest(figure)["axes"][0])

    def test_nonzero_baseline_remains_rejected_with_separate_containers(self):
        figure, ax = self.net_revenue(separate=True, indices=(3, 1, 0, 2))
        ax.containers[2].patches[0].set_y(10)
        with self.assertRaises(AssertionError):
            self.helper.bar_series(self.manifest(figure)["axes"][0])

    def test_extra_images_and_collections_are_still_rejected(self):
        for kind in ("image", "scatter"):
            with self.subTest(kind=kind):
                figure, ax = self.net_revenue()
                if kind == "image":
                    ax.imshow([[0, 1], [1, 0]])
                else:
                    ax.scatter([0, 1], [10, 20])
                with self.assertRaises(AssertionError):
                    self.assert_net_values(figure)

    def test_equivalent_celsius_symbol_and_wording_are_accepted(self):
        for xlabel, ylabel in (
            ("Elapsed time (min)", "Temperature (℃)"),
            ("Duration in minutes", "Temp [°C]"),
            ("Time (minutes)", "Temperature in degrees Celsius"),
            ("Time (min)", "Temperature (C)"),
        ):
            with self.subTest(xlabel=xlabel, ylabel=ylabel):
                figure, _ = self.semantic_units(xlabel=xlabel, ylabel=ylabel)
                self.states["semantic_unit_labels"].test_separate_axis_labels_express_the_required_quantities_and_units(
                    {"manifest": self.manifest(figure)}
                )

    def test_wrong_units_and_quantities_are_still_rejected(self):
        for xlabel, ylabel in (
            ("Elapsed time (seconds)", "Temperature (℃)"),
            ("Elapsed time (min)", "Temperature (℉)"),
            ("Elapsed time (min)", "Temperature (Kelvin)"),
            ("Elapsed time (min)", "Revenue (C)"),
            ("Distance (min)", "Temperature (℃)"),
        ):
            with self.subTest(xlabel=xlabel, ylabel=ylabel):
                figure, _ = self.semantic_units(xlabel=xlabel, ylabel=ylabel)
                with self.assertRaises(AssertionError):
                    self.states["semantic_unit_labels"].test_separate_axis_labels_express_the_required_quantities_and_units(
                        {"manifest": self.manifest(figure)}
                    )


if __name__ == "__main__":
    unittest.main()
