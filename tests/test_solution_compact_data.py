"""Offline contracts for Part 4's bounded, compact data assessment.

These synthetic answers verify schema, evidence routing and uncertainty policy;
they are not measurements of the fixed model's semantic accuracy. No evaluation
runner, model endpoint or existing artifact/prediction file is used here.
"""

import json
import unittest
from unittest.mock import Mock, patch

with patch("dotenv.load_dotenv"):
    from validator import solution
    from validator.prediction import ErrorFamily


KINDS = {"wrong_values", "missing_data", "wrong_selection", "wrong_order"}
SOURCES = {
    "s1": "Use only completed orders and retain North, South order.",
    "s2": "Plot the sum of net revenue in USD for each region.",
}


def document(*, failed=(), uncertain=()):
    failed, uncertain = set(failed), set(uncertain)
    return {
        kind: {
            "status": "fail" if kind in failed else "uncertain" if kind in uncertain else "pass",
            "evidence": f"Specific original-evidence observation for {kind}.",
            "finding": None if kind not in failed else {
                "source_id": "s2", "location": "North bar",
                "expected": "Completed North net revenue is 20 USD.",
                "observed": "Final code and plotted annotation show 30 USD.",
            },
        }
        for kind in sorted(KINDS)
    }


def response(value, reason="stop"):
    content = value if isinstance(value, str) else json.dumps(value)
    return {"choices": [{"message": {"content": content}, "finish_reason": reason}]}


class CompactDataParserTests(unittest.TestCase):
    def parse(self, value, *, family=ErrorFamily.WRONG_DATA, sources=SOURCES):
        content = value if isinstance(value, str) else json.dumps(value)
        return solution._parse_compact_data_response(content, family, sources)

    def test_all_pass_is_explicit_full_coverage_and_has_no_findings(self):
        parsed = self.parse(document())
        self.assertEqual(parsed["status"], "pass")
        self.assertEqual(parsed["findings"], [])
        self.assertEqual({item["kind"] for item in parsed["checks"]}, KINDS)
        self.assertTrue(all(item["status"] == "pass" for item in parsed["checks"]))

    def test_each_mode_attaches_its_own_finding_without_parallel_array_matching(self):
        for kind in sorted(KINDS):
            with self.subTest(kind=kind):
                parsed = self.parse(document(failed=[kind]))
                self.assertEqual(parsed["status"], "fail")
                self.assertEqual(len(parsed["findings"]), 1)
                item = parsed["findings"][0]
                self.assertEqual(item["kind"], kind)
                self.assertEqual(item["source_quote"], SOURCES["s2"])
                self.assertEqual(item["evidence"], document(failed=[kind])[kind]["evidence"])

    def test_all_four_failures_can_be_preserved(self):
        parsed = self.parse(document(failed=KINDS))
        self.assertEqual(parsed["status"], "fail")
        self.assertEqual({item["kind"] for item in parsed["findings"]}, KINDS)

    def test_schema_has_fixed_four_objects_null_union_and_explicit_length_bounds(self):
        spec = solution._compact_data_response_format(SOURCES)
        self.assertEqual(spec["type"], "json_schema")
        self.assertIs(spec["json_schema"]["strict"], True)
        schema = spec["json_schema"]["schema"]
        self.assertEqual(set(schema["properties"]), KINDS)
        self.assertEqual(set(schema["required"]), KINDS)
        self.assertIs(schema["additionalProperties"], False)
        for check in schema["properties"].values():
            self.assertEqual(set(check["properties"]), {"status", "evidence", "finding"})
            self.assertEqual(set(check["required"]), {"status", "evidence", "finding"})
            self.assertEqual(check["properties"]["evidence"]["maxLength"], 160)
            variants = check["properties"]["finding"]["anyOf"]
            self.assertTrue(any(item.get("type") == "null" for item in variants))
            finding = next(item for item in variants if item.get("type") == "object")
            fields = finding["properties"]
            self.assertEqual(set(fields), {"source_id", "location", "expected", "observed"})
            self.assertEqual(fields["source_id"]["enum"], list(SOURCES))
            self.assertEqual(fields["location"]["maxLength"], 80)
            self.assertEqual(fields["expected"]["maxLength"], 160)
            self.assertEqual(fields["observed"]["maxLength"], 160)

    def test_exact_length_boundaries_are_accepted(self):
        value = document(failed=["wrong_values"])
        value["wrong_values"]["evidence"] = "e" * 160
        value["wrong_values"]["finding"].update(location="l" * 80, expected="x" * 160, observed="y" * 160)
        self.assertEqual(self.parse(value)["status"], "fail")

    def test_overlong_evidence_or_finding_fields_are_not_silently_truncated(self):
        for field, length in (("evidence", 161), ("location", 81), ("expected", 161), ("observed", 161)):
            value = document(failed=["wrong_values"])
            target = value["wrong_values"] if field == "evidence" else value["wrong_values"]["finding"]
            target[field] = "x" * length
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.parse(value)

    def test_missing_extra_foreign_or_legacy_root_keys_are_rejected(self):
        missing = document()
        missing.pop("wrong_order")
        variants = [missing, {**document(), "extra": {}}, {"checks": [], "findings": []}, [], None]
        foreign = document()
        foreign["wrong_chart"] = foreign.pop("wrong_order")
        variants.append(foreign)
        for value in variants:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.parse(value)

    def test_duplicate_json_keys_are_rejected_at_both_levels(self):
        value = json.dumps(document())
        duplicate_root = value[:-1] + ', "wrong_values": {"status": "pass", "evidence": "x", "finding": null}}'
        duplicate_status = value.replace('"status": "pass"', '"status": "pass", "status": "fail"', 1)
        for content in (duplicate_root, duplicate_status):
            with self.subTest(content=content), self.assertRaises(ValueError):
                self.parse(content)

    def test_check_fields_types_statuses_and_blank_evidence_remain_strict(self):
        item = document()["wrong_values"]
        bad_items = [None, [], {**item, "extra": "x"}]
        bad_items.extend({key: value for key, value in item.items() if key != field} for field in item)
        bad_items.extend({**item, "status": value} for value in ("PASS", "YES", True, None, []))
        bad_items.extend({**item, "evidence": value} for value in ("", " \n ", True, None, [], 3))
        for malformed in bad_items:
            value = document()
            value["wrong_values"] = malformed
            with self.subTest(item=malformed), self.assertRaises(ValueError):
                self.parse(value)

    def test_fail_needs_finding_but_pass_or_uncertain_must_have_null(self):
        malformed = document(failed=["wrong_values"])
        malformed["wrong_values"]["finding"] = None
        variants = [malformed]
        for status in ("pass", "uncertain"):
            value = document(failed=["wrong_values"])
            value["wrong_values"]["status"] = status
            variants.append(value)
        for value in variants:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.parse(value)

    def test_finding_fields_are_exact_and_source_must_exist(self):
        original = document(failed=["wrong_values"])["wrong_values"]["finding"]
        bad = [[], "failure", {**original, "source_quote": "fabricated"}, {**original, "kind": "wrong_order"}]
        bad.extend({key: value for key, value in original.items() if key != field} for field in original)
        bad.extend({**original, field: value} for field in original for value in (None, [], True, "", " \n "))
        bad.append({**original, "source_id": "s99"})
        for finding in bad:
            value = document(failed=["wrong_values"])
            value["wrong_values"]["finding"] = finding
            with self.subTest(finding=finding), self.assertRaises(ValueError):
                self.parse(value)

    def test_equal_descriptions_are_semantic_uncertainty_not_invalid_json_or_a_pass(self):
        for expected, observed in (("20 USD", "20 USD"), (" North\nSouth ", "North  South")):
            value = document(failed=["wrong_values"])
            value["wrong_values"]["finding"].update(expected=expected, observed=observed)
            with self.subTest(expected=expected):
                parsed = self.parse(value)
                self.assertEqual(parsed["status"], "uncertain")
                self.assertEqual(parsed["findings"], [])
                self.assertEqual(next(check for check in parsed["checks"] if check["kind"] == "wrong_values")["status"], "uncertain")
                self.assertTrue(parsed.get("rejected_claims"))

    def test_equal_claim_does_not_erase_another_distinct_supported_finding(self):
        value = document(failed=["wrong_values", "wrong_order"])
        finding = value["wrong_values"]["finding"]
        finding["observed"] = finding["expected"]
        parsed = self.parse(value)
        self.assertEqual(parsed["status"], "fail")
        self.assertEqual({item["kind"] for item in parsed["findings"]}, {"wrong_order"})
        self.assertTrue(parsed.get("rejected_claims"))

    def test_uncertain_is_not_rewritten_to_pass_and_valid_failure_has_priority(self):
        self.assertEqual(self.parse(document(uncertain=["wrong_values"]))["status"], "uncertain")
        self.assertEqual(self.parse(document(failed=["wrong_order"], uncertain=["wrong_values"]))["status"], "fail")

    def test_compact_protocol_rejects_other_error_family(self):
        for family in (ErrorFamily.HARD_TO_READ, ErrorFamily.WRONG_CHART, ErrorFamily.EXECUTION_FAILURE):
            with self.subTest(family=family), self.assertRaises(ValueError):
                self.parse(document(), family=family)


class CompactDataDecisionTests(unittest.TestCase):
    def confirm(self, query):
        return solution._confirmed_family_decision(
            query, ErrorFamily.WRONG_DATA, SOURCES,
            parse=solution._parse_compact_data_response, compact=True,
            withhold_uncertain=True,
        )

    def test_full_all_pass_takes_one_bounded_call(self):
        query = Mock(return_value=response(document()))
        self.assertEqual(self.confirm(query)["status"], "pass")
        self.assertEqual(query.call_count, 1)
        self.assertEqual(query.call_args.args[1], 1536)

    def test_positive_requires_one_fresh_review_and_can_be_removed(self):
        query = Mock(side_effect=[response(document(failed=["wrong_values"])), response(document())])
        parsed = self.confirm(query)
        self.assertEqual(parsed["status"], "pass")
        self.assertEqual(parsed["findings"], [])
        self.assertEqual(query.call_count, 2)
        self.assertIn("fallible", query.call_args.args[0])

    def test_persistent_semantic_uncertainty_warns_and_remains_uncertain(self):
        query = Mock(return_value=response(document(uncertain=["wrong_values"])))
        with self.assertWarns(RuntimeWarning):
            parsed = self.confirm(query)
        self.assertEqual(parsed["status"], "uncertain")
        self.assertEqual(parsed["findings"], [])
        self.assertEqual(query.call_count, 2)

    def test_equal_claim_triggers_semantic_review_not_schema_repair(self):
        value = document(failed=["wrong_values"])
        value["wrong_values"]["finding"]["observed"] = value["wrong_values"]["finding"]["expected"]
        query = Mock(side_effect=[response(value), response(document())])
        self.assertEqual(self.confirm(query)["status"], "pass")
        self.assertEqual(query.call_count, 2)
        note = query.call_args.args[0]
        self.assertIn(value["wrong_values"]["finding"]["expected"], note)
        self.assertIn("fallible", note)

    def test_valid_positive_survives_other_uncertain_modes(self):
        value = document(failed=["wrong_order"], uncertain=["wrong_values"])
        query = Mock(return_value=response(value))
        with patch("warnings.warn"):
            parsed = self.confirm(query)
        self.assertEqual(parsed["status"], "fail")
        self.assertEqual({item["kind"] for item in parsed["findings"]}, {"wrong_order"})
        self.assertEqual(query.call_count, 2)

    def test_malformed_or_truncated_answers_still_raise_after_two_format_attempts(self):
        for value, reason in (("not JSON", "stop"), (document(), "length")):
            query = Mock(return_value=response(value, reason))
            with self.subTest(reason=reason), self.assertRaises(RuntimeError):
                self.confirm(query)
            self.assertEqual(query.call_count, 2)

    def test_transport_fault_is_not_withheld_as_semantic_uncertainty(self):
        original = RuntimeError("Synthetic network problem")
        query = Mock(side_effect=original)
        with self.assertRaises(RuntimeError) as raised:
            self.confirm(query)
        self.assertIs(raised.exception, original)
        query.assert_called_once()

    def test_readability_persistent_uncertainty_still_raises(self):
        kinds = sorted(solution.READABILITY_KINDS)
        value = {"checks": [{"kind": kind, "status": "uncertain", "evidence": "Specific observation."} for kind in kinds], "findings": []}
        query = Mock(return_value=response(value))
        with self.assertRaises(RuntimeError):
            solution._confirmed_family_decision(query, ErrorFamily.HARD_TO_READ, SOURCES)
        self.assertEqual(query.call_count, 2)


if __name__ == "__main__":
    unittest.main()
