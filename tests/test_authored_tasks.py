"""Offline checks for authored task inputs and task-derived numerical targets.

No agent/validator calls, candidate labels, reference PNGs, or generated runs.
Expected values are review targets, not predictions about future agent behavior.
"""

import csv
import json
import unittest
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

from workflow.generate import AGENTS, build_request
from workflow.tasks import load_tasks


ROOT = Path(__file__).resolve().parents[1]


class AuthoredTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = load_tasks(ROOT / "tasks")
        cls.by_id = {task.task_id: (task, descriptor) for task, descriptor in cls.tasks}

    def rows(self, task_id):
        task, descriptor = self.by_id[task_id]
        self.assertEqual(len(task.inputs), 1)
        with (descriptor.parent / task.inputs[0].path).open(newline="") as source:
            return list(csv.DictReader(source))

    def test_five_tasks_in_two_classes(self):
        self.assertEqual(len(self.tasks), 5)
        self.assertEqual(
            Counter(task.task_class for task, _ in self.tasks),
            {"scoped_design_labels": 3, "data_design_separation": 2},
        )

    def test_descriptors_have_exact_fields(self):
        for task, descriptor in self.tasks:
            with self.subTest(task=task.task_id):
                document = json.loads(descriptor.read_text())
                self.assertEqual(
                    set(document), {"task_id", "task_class", "instructions", "inputs"}
                )
                self.assertEqual(descriptor.parent.name, task.task_id)

    def test_harbor_only_requirements_not_in_original_prompts(self):
        for task, _ in self.tasks:
            with self.subTest(task=task.task_id):
                self.assertIn("figure.png", task.instructions)
                self.assertNotIn("plotted_values.json", task.instructions)
                self.assertNotIn("plot.py", task.instructions)

    def test_fifteen_requests_use_fixed_agents_and_unchanged_inputs(self):
        expected_models = {
            "qwen": "Qwen/Qwen2.5-Coder-3B-Instruct",
            "ministral": "mistralai/Ministral-3-14B-Instruct-2512",
            "glm": "zai-org/GLM-4.7-Flash",
        }
        self.assertEqual({agent.key: agent.model for agent in AGENTS}, expected_models)
        request_count = 0
        for task, descriptor in self.tasks:
            for agent in AGENTS:
                with self.subTest(task=task.task_id, agent=agent.key):
                    request = build_request(task, descriptor, agent)
                    self.assertEqual(request["model"], expected_models[agent.key])
                    self.assertEqual(request["task"]["instructions"], task.instructions)
                    self.assertEqual(
                        request["task"]["expected_output"],
                        {"path": "figure.png", "format": "png"},
                    )
                    self.assertEqual(
                        set(request["input_files"]), {item.name for item in task.inputs}
                    )
                    for item in task.inputs:
                        self.assertEqual(
                            request["input_files"][item.name],
                            (descriptor.parent / item.path).read_bytes(),
                        )
                    request_count += 1
        self.assertEqual(request_count, 15)

    def test_panel_unit_inputs(self):
        rows = self.rows("panel_unit_labels")
        self.assertEqual(
            [row["channel"] for row in rows], ["Retail", "Wholesale", "Online", "Partners"]
        )
        self.assertEqual([int(row["orders"]) for row in rows], [120, 75, 210, 45])
        self.assertEqual([int(row["refund_rate_pct"]) for row in rows], [4, 8, 3, 6])

    def test_minutes_conversion_and_temperature_inputs(self):
        rows = self.rows("semantic_unit_labels")
        self.assertEqual(
            [float(Fraction(int(row["elapsed_seconds"]), 60)) for row in rows],
            [0, 1.5, 3, 4.5, 6, 7.5],
        )
        self.assertEqual([int(row["temperature_c"]) for row in rows], [18, 20, 25, 31, 36, 39])

    def test_shared_axis_inputs(self):
        rows = self.rows("shared_axis_scope")
        self.assertEqual([int(row["time_s"]) for row in rows], [0, 1, 2, 3, 4, 5])
        self.assertEqual(
            [float(row["sensor_a_v"]) for row in rows], [0.2, 0.6, 1.2, 0.7, 0.1, -0.3]
        )
        self.assertEqual(
            [float(row["sensor_b_v"]) for row in rows], [1.2, 0.9, 0.5, 0.1, -0.4, -0.8]
        )

    def test_filtered_net_targets_include_zero_region(self):
        rows = self.rows("filtered_net_revenue")
        totals = defaultdict(int)
        for row in rows:
            if row["status"] == "completed":
                totals[row["region"]] += int(row["gross"]) - int(row["refund"])
        order = ["North", "South", "East", "West"]
        self.assertEqual([totals[region] for region in order], [145, 0, 95, 105])
        self.assertFalse(
            any(row["region"] == "South" and row["status"] == "completed" for row in rows)
        )
        self.assertEqual(sum(totals.values()), 345)

    def test_unfiltered_net_and_gross_are_distinct_errors(self):
        rows = self.rows("filtered_net_revenue")
        all_status_net = defaultdict(int)
        completed_gross = defaultdict(int)
        for row in rows:
            all_status_net[row["region"]] += int(row["gross"]) - int(row["refund"])
            if row["status"] == "completed":
                completed_gross[row["region"]] += int(row["gross"])
        order = ["North", "South", "East", "West"]
        self.assertEqual([all_status_net[region] for region in order], [345, 225, 140, 105])
        self.assertEqual([completed_gross[region] for region in order], [180, 0, 110, 130])

    def test_pooled_success_targets(self):
        rows = self.rows("pooled_success_rates")
        totals = defaultdict(lambda: [0, 0])
        for row in rows:
            successes, attempts = int(row["successes"]), int(row["attempts"])
            self.assertGreater(attempts, 0)
            self.assertLessEqual(successes, attempts)
            totals[row["month"]][0] += successes
            totals[row["month"]][1] += attempts
        order = ["Jan", "Feb", "Mar", "Apr"]
        self.assertEqual([totals[month][0] for month in order], [18, 22, 33, 7])
        self.assertEqual([totals[month][1] for month in order], [100, 80, 50, 20])
        percentages = [
            float(100 * Fraction(totals[month][0], totals[month][1])) for month in order
        ]
        self.assertEqual(percentages, [18.0, 27.5, 66.0, 35.0])

    def test_unweighted_group_average_is_not_pooled_rate(self):
        rows = self.rows("pooled_success_rates")
        groups = defaultdict(list)
        for row in rows:
            groups[row["month"]].append(
                Fraction(int(row["successes"]), int(row["attempts"]))
            )
        order = ["Jan", "Feb", "Mar", "Apr"]
        unweighted = [
            float(100 * sum(groups[month], Fraction(0)) / len(groups[month]))
            for month in order
        ]
        self.assertEqual(unweighted, [50.0, 45.0, 60.0, 50.0])
        pooled = [18.0, 27.5, 66.0, 35.0]
        self.assertTrue(all(wrong != correct for wrong, correct in zip(unweighted, pooled)))


if __name__ == "__main__":
    unittest.main()
