"""Archived offline contracts for the rejected data-only proof-review revision.

All answers are synthetic: these tests measure protocol/control-flow correctness,
not the fixed model's accuracy. No official evaluation, existing artifact changes,
network requests or Docker operations occur here.

This experiment is retained for provenance, not active test discovery. It loads
only the archived proof checkpoint 42353174, not the restored stable validator.
"""

import importlib.util
import itertools
import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

with patch("dotenv.load_dotenv"):
    from validator.prediction import ErrorFamily
    archive = Path(__file__).with_name("solution_proof_42353174.py")
    spec = importlib.util.spec_from_file_location("hw2_archived_proof_solution", archive)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the archived proof-revision source.")
    solution = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(solution)


KINDS = {"wrong_values", "missing_data", "wrong_selection", "wrong_order"}
GATES = {"task_support", "data_difference", "final_support"}
STATUSES = {"supported", "unsupported", "uncertain"}
SOURCES = {"s1": "Plot the final North and South totals in USD."}


def initial_document(*, failed=("wrong_values",), uncertain=()):
    failed, uncertain = set(failed), set(uncertain)
    return {
        kind: {
            "status": "fail" if kind in failed else "uncertain" if kind in uncertain else "pass",
            "evidence": f"Specific evidence for {kind}.",
            "finding": None if kind not in failed else finding(),
        }
        for kind in sorted(KINDS)
    }


def finding():
    return {"source_id": "s1", "location": "North bar",
            "expected": "North total is 20 USD.", "observed": "Final annotation and bar show 30 USD."}


def review_document(kinds=("wrong_values",), *, gate="supported"):
    return {
        kind: {
            **{name: gate for name in GATES},
            "counterevidence": "Checked rounding, numeric equivalence and the last successful output.",
            "evidence": "The final chart represents 30 USD rather than the requested 20 USD.",
            "finding": finding() if gate == "supported" else None,
        }
        for kind in kinds
    }


def response(value, reason="stop"):
    content = value if isinstance(value, str) else json.dumps(value)
    return {"choices": [{"message": {"content": content}, "finish_reason": reason}]}


class DataReviewParserTests(unittest.TestCase):
    def setUp(self):
        self.initial = solution._parse_compact_data_response(
            json.dumps(initial_document()), ErrorFamily.WRONG_DATA, SOURCES,
        )

    def parse(self, value, initial=None, family=ErrorFamily.WRONG_DATA):
        content = value if isinstance(value, str) else json.dumps(value)
        return solution._parse_data_review_response(
            content, family, SOURCES, self.initial if initial is None else initial,
        )

    def test_schema_contains_only_provisional_modes_and_exact_bounded_gate_fields(self):
        spec = solution._data_review_response_format({"wrong_values", "wrong_order"}, SOURCES)
        self.assertEqual(spec["type"], "json_schema")
        self.assertIs(spec["json_schema"]["strict"], True)
        schema = spec["json_schema"]["schema"]
        self.assertEqual(set(schema["properties"]), {"wrong_values", "wrong_order"})
        self.assertEqual(set(schema["required"]), {"wrong_values", "wrong_order"})
        self.assertIs(schema["additionalProperties"], False)
        for item in schema["properties"].values():
            self.assertEqual(set(item["properties"]), GATES | {"counterevidence", "evidence", "finding"})
            self.assertEqual(set(item["required"]), set(item["properties"]))
            self.assertIs(item["additionalProperties"], False)
            for gate in GATES:
                self.assertEqual(set(item["properties"][gate]["enum"]), STATUSES)
            for field in ("evidence", "counterevidence"):
                self.assertEqual(item["properties"][field]["maxLength"], 160)

    def test_review_schema_rejects_empty_or_unknown_mode_sets(self):
        for kinds in (set(), {"wrong_chart"}, {"wrong_values", "extra"}):
            with self.subTest(kinds=kinds), self.assertRaises(ValueError):
                solution._data_review_response_format(kinds, SOURCES)

    def test_supported_final_visual_data_difference_is_retained_without_code_quote_gate(self):
        parsed = self.parse(review_document())
        self.assertEqual(parsed["status"], "fail")
        claim = parsed["findings"][0]
        self.assertEqual(claim["source_quote"], SOURCES["s1"])
        self.assertEqual(claim["location"], "North bar")
        self.assertEqual(claim["evidence"], review_document()["wrong_values"]["evidence"])
        self.assertEqual(parsed["support_review"], review_document())

    def test_each_unsupported_gate_rejects_the_claim_without_erasing_other_pass_checks(self):
        for gate in GATES:
            value = review_document()
            value["wrong_values"][gate] = "unsupported"
            value["wrong_values"]["finding"] = None
            with self.subTest(gate=gate):
                parsed = self.parse(value)
                self.assertEqual(parsed["status"], "pass")
                self.assertEqual(parsed["findings"], [])
                self.assertEqual({item["kind"] for item in parsed["checks"]}, KINDS)
                original_pass = {item["kind"]: item for item in self.initial["checks"] if item["status"] == "pass"}
                for item in parsed["checks"]:
                    if item["kind"] in original_pass:
                        self.assertEqual(item, original_pass[item["kind"]])

    def test_all_27_gate_combinations_follow_unsupported_then_uncertain_then_supported(self):
        gate_names = sorted(GATES)
        for values in itertools.product(sorted(STATUSES), repeat=3):
            value = review_document()
            value["wrong_values"].update(dict(zip(gate_names, values)))
            if any(gate != "supported" for gate in values):
                value["wrong_values"]["finding"] = None
            with self.subTest(gates=values):
                parsed = self.parse(value)
                expected = "pass" if "unsupported" in values else "uncertain" if "uncertain" in values else "fail"
                self.assertEqual(parsed["status"], expected)

    def test_new_extra_missing_or_foreign_modes_cannot_be_introduced_by_review(self):
        variants = [{}, {**review_document(), **review_document(("wrong_order",))}, {"wrong_chart": review_document()["wrong_values"]}]
        for value in variants:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.parse(value)

    def test_duplicate_root_or_nested_json_keys_are_rejected(self):
        text = json.dumps(review_document())
        root_duplicate = text[:-1] + ', "wrong_values": ' + json.dumps(review_document()["wrong_values"]) + '}'
        nested_duplicate = text.replace('"task_support": "supported"', '"task_support": "supported", "task_support": "unsupported"', 1)
        for value in (root_duplicate, nested_duplicate):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.parse(value)

    def test_missing_extra_status_or_nonobject_review_fields_are_rejected(self):
        original = review_document()["wrong_values"]
        variants = [None, [], "supported", {**original, "status": "fail"}]
        variants.extend({key: value for key, value in original.items() if key != field} for field in original)
        for item in variants:
            with self.subTest(item=item), self.assertRaises(ValueError):
                self.parse({"wrong_values": item})

    def test_gate_values_must_be_exact_strings_not_truthy_coercions(self):
        for gate in GATES:
            for malformed in ("SUPPORT", "yes", "fail", None, [], True, 1):
                value = review_document()
                value["wrong_values"][gate] = malformed
                with self.subTest(gate=gate, value=malformed), self.assertRaises(ValueError):
                    self.parse(value)

    def test_counterevidence_and_discrepancy_evidence_have_strict_160_character_limits(self):
        for field in ("evidence", "counterevidence"):
            value = review_document()
            value["wrong_values"][field] = "x" * 160
            self.assertEqual(self.parse(value)["status"], "fail")
            for malformed in ("x" * 161, "", " \n ", None, [], True):
                value = review_document()
                value["wrong_values"][field] = malformed
                with self.subTest(field=field, value=malformed), self.assertRaises(ValueError):
                    self.parse(value)

    def test_finding_is_required_only_when_every_support_gate_is_supported(self):
        all_supported = review_document()
        all_supported["wrong_values"]["finding"] = None
        invalid_unsupported = review_document()
        invalid_unsupported["wrong_values"]["task_support"] = "unsupported"
        invalid_uncertain = review_document()
        invalid_uncertain["wrong_values"]["final_support"] = "uncertain"
        for value in (all_supported, invalid_unsupported, invalid_uncertain):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.parse(value)

    def test_finding_retains_source_and_field_limits_and_rejects_fabricated_fields(self):
        for field, malformed in (("source_id", "s99"), ("location", "x" * 81), ("expected", "x" * 161), ("observed", "x" * 161)):
            value = review_document()
            value["wrong_values"]["finding"][field] = malformed
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.parse(value)
        value = review_document()
        value["wrong_values"]["finding"]["source_quote"] = "Fabricated task text."
        with self.assertRaises(ValueError):
            self.parse(value)

    def test_equal_expected_and_observed_remain_uncertain_not_false_proof_of_pass(self):
        value = review_document()
        value["wrong_values"]["finding"].update(expected=" North\nSouth ", observed="North  South")
        parsed = self.parse(value)
        self.assertEqual(parsed["status"], "uncertain")
        self.assertEqual(parsed["findings"], [])
        self.assertTrue(parsed["rejected_claims"])

    def test_uncertain_initial_mode_can_be_supported_and_other_initial_passes_stay_pass(self):
        initial = solution._parse_compact_data_response(json.dumps(initial_document(failed=(), uncertain=("wrong_values",))), ErrorFamily.WRONG_DATA, SOURCES)
        parsed = self.parse(review_document(), initial=initial)
        self.assertEqual(parsed["status"], "fail")
        self.assertEqual({item["kind"] for item in parsed["findings"]}, {"wrong_values"})

    def test_non_data_family_or_all_pass_initial_cannot_request_data_proof_review(self):
        with self.assertRaises(ValueError):
            self.parse(review_document(), family=ErrorFamily.HARD_TO_READ)
        initial = solution._parse_compact_data_response(json.dumps(initial_document(failed=())), ErrorFamily.WRONG_DATA, SOURCES)
        with self.assertRaises(ValueError):
            self.parse({}, initial=initial)


class DataProofDecisionTests(unittest.TestCase):
    def confirm(self, query, review):
        return solution._confirmed_data_decision(query, review, SOURCES)

    def test_complete_initial_pass_never_calls_review(self):
        query = Mock(return_value=response(initial_document(failed=())))
        review = Mock()
        self.assertEqual(self.confirm(query, review)["status"], "pass")
        query.assert_called_once()
        review.assert_not_called()

    def test_one_review_receives_only_provisional_modes_and_no_additional_model_stage(self):
        query = Mock(return_value=response(initial_document(failed=("wrong_values",), uncertain=("wrong_order",))))
        review = Mock(return_value=response(review_document(("wrong_values", "wrong_order"))))
        parsed = self.confirm(query, review)
        self.assertEqual(parsed["status"], "fail")
        query.assert_called_once()
        review.assert_called_once()
        self.assertEqual(review.call_args.args[2], {"wrong_values", "wrong_order"})
        self.assertEqual(review.call_args.args[1], 1536)
        self.assertIn("fallible", review.call_args.args[0])

    def test_unsupported_review_discards_provisional_claim(self):
        query = Mock(return_value=response(initial_document()))
        review = Mock(return_value=response(review_document(gate="unsupported")))
        parsed = self.confirm(query, review)
        self.assertEqual(parsed["status"], "pass")
        self.assertEqual(parsed["findings"], [])
        self.assertEqual(query.call_count + review.call_count, 2)

    def test_unresolved_review_warns_and_preserves_semantic_uncertainty(self):
        query = Mock(return_value=response(initial_document()))
        review = Mock(return_value=response(review_document(gate="uncertain")))
        with self.assertWarns(RuntimeWarning):
            parsed = self.confirm(query, review)
        self.assertEqual(parsed["status"], "uncertain")
        self.assertEqual(parsed["findings"], [])
        review.assert_called_once()

    def test_equal_comparison_after_supported_gates_warns_and_withholds_claim(self):
        value = review_document()
        value["wrong_values"]["finding"]["observed"] = value["wrong_values"]["finding"]["expected"]
        query = Mock(return_value=response(initial_document()))
        review = Mock(return_value=response(value))
        with self.assertWarns(RuntimeWarning):
            self.assertEqual(self.confirm(query, review)["status"], "uncertain")
        self.assertEqual(query.call_count + review.call_count, 2)

    def test_initial_and_review_format_attempts_still_have_total_four_call_limit(self):
        query = Mock(side_effect=[response("invalid JSON"), response(initial_document())])
        review = Mock(side_effect=[response("invalid JSON"), response(review_document())])
        self.assertEqual(self.confirm(query, review)["status"], "fail")
        self.assertEqual(query.call_count + review.call_count, 4)
        self.assertEqual([call.args[1] for call in query.call_args_list], [1536, 1536])
        self.assertEqual([call.args[1] for call in review.call_args_list], [1536, 1536])
        repair_note = review.call_args.args[0]
        for field in GATES | {"counterevidence", "evidence", "finding"}:
            self.assertIn(field, repair_note)
        self.assertIn("No status field", repair_note)

    def test_review_length_only_repair_increases_budget_once(self):
        query = Mock(return_value=response(initial_document()))
        review = Mock(side_effect=[response(review_document(), "length"), response(review_document())])
        self.assertEqual(self.confirm(query, review)["status"], "fail")
        self.assertEqual([call.args[1] for call in review.call_args_list], [1536, 3072])

    def test_unrecoverable_review_schema_or_length_raises_not_semantic_withholding(self):
        for value, reason in (("invalid JSON", "stop"), (review_document(), "length")):
            query = Mock(return_value=response(initial_document()))
            review = Mock(return_value=response(value, reason))
            with self.subTest(reason=reason), self.assertRaises(RuntimeError):
                self.confirm(query, review)
            self.assertEqual(review.call_count, 2)

    def test_initial_and_review_transport_errors_propagate_without_becoming_labels(self):
        original = RuntimeError("Synthetic endpoint fault")
        for at_review in (False, True):
            query = Mock(return_value=response(initial_document())) if at_review else Mock(side_effect=original)
            review = Mock(side_effect=original)
            with self.subTest(at_review=at_review), self.assertRaises(RuntimeError) as raised:
                self.confirm(query, review)
            self.assertIs(raised.exception, original)
            query.assert_called_once()
            self.assertEqual(review.call_count, int(at_review))


if __name__ == "__main__":
    unittest.main()
