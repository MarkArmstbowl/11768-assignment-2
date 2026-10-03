"""Offline contracts for bounded Part 4 response repair.

Synthetic model answers test protocol reliability, not judge accuracy. No
evaluation runners, network requests or existing result files are used here.
"""

import base64
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

with patch("dotenv.load_dotenv"):
    from validator import solution
    from validator.prediction import ErrorFamily
    from validator.runner import Run


TASK = "Plot completed orders only.\nInclude North and South in that order."
SOURCES = {"s1": "Plot completed orders only.", "s2": "Include North and South in that order."}
DATA_KINDS = {"wrong_values", "missing_data", "wrong_selection", "wrong_order"}
READ_KINDS = {
    "clipped_text", "overlapping_text", "occlusion", "low_contrast",
    "indistinguishable_data", "squeezed_layout",
}


def document(*, fail=False, family=ErrorFamily.WRONG_DATA):
    kinds = DATA_KINDS if family == ErrorFamily.WRONG_DATA else READ_KINDS
    failed_kind = "wrong_values" if family == ErrorFamily.WRONG_DATA else "clipped_text"
    checks = [
        {"kind": kind, "status": "fail" if fail and kind == failed_kind else "pass",
         "evidence": f"A specific synthetic observation for {kind}."}
        for kind in sorted(kinds)
    ]
    findings = []
    if fail:
        if family == ErrorFamily.WRONG_DATA:
            findings = [{
                "kind": failed_kind, "source_id": "s1", "location": "North bar",
                "expected": "Completed orders total 20 USD.",
                "observed": "The final plotted bar is 30 USD.",
                "evidence": "A pending 10 USD order is included in the final sum.",
            }]
        else:
            findings = [{
                "kind": failed_kind, "location": "left saved image edge",
                "elements": "y-axis label", "observation": "Initial letters are cut off.",
                "impact": "The quantity label cannot be read in full.",
            }]
    return {"checks": checks, "findings": findings}


def response(value, reason="stop"):
    content = value if isinstance(value, str) else json.dumps(value)
    return {"choices": [{"message": {"content": content}, "finish_reason": reason}]}


def compact_response(value, reason="stop"):
    """Adapt a synthetic data fixture to the final compact data protocol."""
    if isinstance(value, dict):
        findings = {item["kind"]: item for item in value["findings"]}
        value = {
            item["kind"]: {
                "status": item["status"], "evidence": item["evidence"],
                "finding": None if item["kind"] not in findings else {
                    key: findings[item["kind"]][key]
                    for key in ("source_id", "location", "expected", "observed")
                },
            }
            for item in value["checks"]
        }
    return response(value, reason)


def contradictory_document():
    value = document(fail=True)
    value["findings"][0]["observed"] = value["findings"][0]["expected"]
    return value


def mismatched_document():
    value = document(fail=True)
    value["findings"].append(copy.deepcopy(value["findings"][0]))
    return value


def budgets(query):
    return [call.args[1] for call in query.call_args_list]


class AdaptiveRepairTests(unittest.TestCase):
    def judge(self, query, *, family=ErrorFamily.WRONG_DATA, stage="initial"):
        return solution._validated_family_query(query, family, SOURCES, stage=stage)

    def test_normal_data_pass_uses_one_base_budget_request(self):
        query = Mock(return_value=response(document()))
        self.assertEqual(self.judge(query)["status"], "pass")
        self.assertEqual(budgets(query), [1536])

    def test_actual_length_finish_reason_increases_only_data_repair_budget(self):
        query = Mock(side_effect=[response(document(), "length"), response(document())])
        self.assertEqual(self.judge(query)["status"], "pass")
        self.assertEqual(budgets(query), [1536, 3072])
        self.assertIn("truncat", query.call_args.args[0].lower())

    def test_parseable_json_with_length_finish_reason_is_not_accepted(self):
        query = Mock(return_value=response(document(), "length"))
        with self.assertRaises(RuntimeError):
            self.judge(query)
        self.assertEqual(budgets(query), [1536, 3072])

    def test_syntax_errors_without_length_do_not_increase_budget(self):
        query = Mock(side_effect=[response('{"checks": ['), response(document())])
        self.assertEqual(self.judge(query)["status"], "pass")
        self.assertEqual(budgets(query), [1536, 1536])

    def test_word_truncated_in_response_content_does_not_trigger_budget_increase(self):
        query = Mock(side_effect=[response("The answer was truncated, allegedly."), response(document())])
        self.assertEqual(self.judge(query)["status"], "pass")
        self.assertEqual(budgets(query), [1536, 1536])

    def test_invalid_schema_without_length_does_not_increase_budget(self):
        invalid = document()
        invalid["checks"].pop()
        query = Mock(side_effect=[response(invalid), response(document())])
        self.assertEqual(self.judge(query)["status"], "pass")
        self.assertEqual(budgets(query), [1536, 1536])

    def test_length_on_second_attempt_has_no_third_attempt(self):
        query = Mock(side_effect=[response("NO"), response(document(), "length")])
        with self.assertRaises(RuntimeError) as raised:
            self.judge(query)
        self.assertEqual(budgets(query), [1536, 1536])
        self.assertIn("output_budget=1536", str(raised.exception))

    def test_schema_error_then_second_truncation_reports_actual_not_future_budget(self):
        invalid = document()
        invalid["checks"].pop()
        query = Mock(side_effect=[response(invalid), response(document(), "length")])
        with self.assertRaises(RuntimeError) as raised:
            self.judge(query)
        self.assertEqual(budgets(query), [1536, 1536])
        self.assertIn("output_budget=1536", str(raised.exception))
        self.assertNotIn("3072", str(raised.exception))

    def test_readability_truncation_keeps_same_readability_budget(self):
        value = document(family=ErrorFamily.HARD_TO_READ)
        query = Mock(side_effect=[response(value, "length"), response(value)])
        self.assertEqual(self.judge(query, family=ErrorFamily.HARD_TO_READ)["status"], "pass")
        self.assertEqual(budgets(query), [2048, 2048])

    def test_identical_data_gets_targeted_retraction_repair_without_larger_budget(self):
        query = Mock(side_effect=[response(contradictory_document()), response(document())])
        self.assertEqual(self.judge(query)["status"], "pass")
        self.assertEqual(budgets(query), [1536, 1536])
        note = query.call_args.args[0].lower()
        self.assertIn("identical", note)
        self.assertIn("mark that check pass and remove its finding", note)
        self.assertIn("invent", note)

    def test_duplicate_finding_gets_synchronized_one_per_kind_repair(self):
        query = Mock(side_effect=[response(mismatched_document()), response(document(fail=True))])
        self.assertEqual(self.judge(query)["status"], "fail")
        self.assertEqual(budgets(query), [1536, 1536])
        note = query.call_args.args[0].lower()
        self.assertIn("exactly one", note)
        self.assertIn("fail", note)
        self.assertIn("finding", note)

    def test_special_parser_errors_remain_value_errors_with_reason(self):
        for value, expected_reason in (
            (contradictory_document(), "identical_data"),
            (mismatched_document(), "finding_mismatch"),
        ):
            with self.subTest(reason=expected_reason), self.assertRaises(ValueError) as raised:
                solution._parse_family_response(json.dumps(value), ErrorFamily.WRONG_DATA, SOURCES)
            self.assertIsInstance(raised.exception, solution._FamilyResponseError)
            self.assertEqual(raised.exception.reason, expected_reason)

    def test_missing_and_unassociated_findings_use_targeted_mismatch_repair(self):
        missing = document(fail=True)
        missing["findings"] = []
        unassociated = document()
        unassociated["findings"] = document(fail=True)["findings"]
        for value in (missing, unassociated):
            with self.subTest(value=value), self.assertRaises(ValueError) as raised:
                solution._parse_family_response(json.dumps(value), ErrorFamily.WRONG_DATA, SOURCES)
            self.assertEqual(raised.exception.reason, "finding_mismatch")

    def test_terminal_diagnostics_report_last_attempt_without_raw_content(self):
        secret = "RAW_RESPONSE_SENTINEL_DO_NOT_LOG"
        query = Mock(return_value=response(secret, "length"))
        with self.assertRaises(RuntimeError) as raised:
            self.judge(query, stage="confirmation")
        message = str(raised.exception)
        for expected in ("confirmation", "finish_reason", "length", "output_budget", "3072", "response_chars", str(len(secret))):
            self.assertIn(expected, message)
        self.assertNotIn(secret, message)
        self.assertEqual(query.call_count, 2)

    def test_initial_terminal_diagnostics_remain_explicit_and_bounded(self):
        query = Mock(return_value=response("not JSON"))
        with self.assertRaises(RuntimeError) as raised:
            self.judge(query)
        self.assertIn("initial", str(raised.exception))
        self.assertIn("1536", str(raised.exception))
        self.assertEqual(query.call_count, 2)

    def test_transport_fault_is_not_format_repaired_or_budget_escalated(self):
        original = RuntimeError("Synthetic transport problem")
        query = Mock(side_effect=original)
        with self.assertRaises(RuntimeError) as raised:
            self.judge(query)
        self.assertIs(raised.exception, original)
        self.assertEqual(budgets(query), [1536])


class ConfirmationRepairTests(unittest.TestCase):
    def confirm(self, query):
        return solution._confirmed_family_decision(query, ErrorFamily.WRONG_DATA, SOURCES)

    def test_all_pass_never_requests_confirmation(self):
        query = Mock(return_value=response(document()))
        self.assertEqual(self.confirm(query)["status"], "pass")
        self.assertEqual(budgets(query), [1536])

    def test_confirmation_starts_base_budget_even_if_initial_needed_larger_repair(self):
        query = Mock(side_effect=[
            response(document(fail=True), "length"),
            response(document(fail=True)),
            response(document()),
        ])
        self.assertEqual(self.confirm(query)["status"], "pass")
        self.assertEqual(budgets(query), [1536, 3072, 1536])

    def test_initial_and_confirmation_can_each_repair_length_once(self):
        value = document(fail=True)
        query = Mock(side_effect=[response(value, "length"), response(value), response(value, "length"), response(value)])
        self.assertEqual(self.confirm(query)["status"], "fail")
        self.assertEqual(budgets(query), [1536, 3072, 1536, 3072])

    def test_repaired_contradiction_that_becomes_pass_needs_no_confirmation(self):
        query = Mock(side_effect=[response(contradictory_document()), response(document())])
        self.assertEqual(self.confirm(query)["status"], "pass")
        self.assertEqual(budgets(query), [1536, 1536])

    def test_repaired_contradiction_that_becomes_fail_is_still_confirmed(self):
        query = Mock(side_effect=[response(contradictory_document()), response(document(fail=True)), response(document())])
        self.assertEqual(self.confirm(query)["status"], "pass")
        self.assertEqual(budgets(query), [1536, 1536, 1536])
        self.assertIn("Provisional assessment", query.call_args.args[0])

    def test_invalid_confirmation_after_its_repair_reports_confirmation_stage(self):
        query = Mock(side_effect=[response(document(fail=True)), response("bad JSON"), response("bad JSON")])
        with self.assertRaises(RuntimeError) as raised:
            self.confirm(query)
        self.assertIn("confirmation", str(raised.exception))
        self.assertEqual(budgets(query), [1536, 1536, 1536])

    def test_confirmation_repair_keeps_provisional_claims_fallible(self):
        value = document(fail=True)
        query = Mock(side_effect=[response(value), response(mismatched_document()), response(value)])
        self.assertEqual(self.confirm(query)["status"], "fail")
        for call in query.call_args_list[1:]:
            note = call.args[0]
            self.assertIn("fallible", note)
            self.assertIn("Provisional assessment", note)
        self.assertEqual(budgets(query), [1536, 1536, 1536])


class RepairEvidenceRoutingTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="part4-repair-offline-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        figure = self.root / "figure.png"
        Image.new("RGB", (120, 120), "white").save(figure)
        data = self.root / "orders.csv"
        # Only temporary, synthetic fixtures are written by this test.
        data.write_text("REGRESSION_INPUT_MARKER\nregion,status,value\nNorth,completed,20\nNorth,pending,10\n")
        messages = [{"role": "tool", "content": "REGRESSION_TRAJECTORY_MARKER: figure delivered"}]
        self.run = Run("unseen-repair-run", "unseen-repair-task", TASK, messages, figure, {"orders.csv": data})
        self.original_image = figure.read_bytes()

    def test_data_initial_repair_and_confirmation_keep_original_evidence(self):
        value = document(fail=True)
        answers = [compact_response(value, "length"), compact_response(value), compact_response(value)]
        with patch("validator.solution.complete", side_effect=answers) as complete:
            errors = solution.judge_data(self.run)
        self.assertEqual(errors[0].family, ErrorFamily.WRONG_DATA)
        self.assertEqual([call.kwargs["max_tokens"] for call in complete.call_args_list], [1536, 3072, 1536])
        for call in complete.call_args_list:
            content = call.args[0][-1]["content"]
            text = "\n".join(item["text"] for item in content if item["type"] == "text")
            self.assertIn(TASK, text)
            self.assertIn("REGRESSION_INPUT_MARKER", text)
            self.assertIn("REGRESSION_TRAJECTORY_MARKER", text)
            url = next(item["image_url"]["url"] for item in content if item["type"] == "image_url")
            self.assertEqual(base64.b64decode(url.split(",", 1)[1]), self.original_image)
            self.assertNotIn("model", call.kwargs)
        self.assertEqual(self.run.figure.read_bytes(), self.original_image)

    def test_readability_transport_accepts_explicit_budget_and_preserves_original_image(self):
        with patch("validator.solution.complete", return_value=response(document(family=ErrorFamily.HARD_TO_READ))) as complete:
            solution._query_readability(self.run, self.original_image, "Synthetic repair", max_tokens=2048)
        self.assertEqual(complete.call_args.kwargs["max_tokens"], 2048)
        content = complete.call_args.args[0][-1]["content"]
        url = next(item["image_url"]["url"] for item in content if item["type"] == "image_url")
        self.assertEqual(base64.b64decode(url.split(",", 1)[1]), self.original_image)


if __name__ == "__main__":
    unittest.main()
