"""Offline checks for task-grounded chart-design requirements and their audit."""

import base64
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from openai import BadRequestError
from PIL import Image

with patch("dotenv.load_dotenv"):
    from validator import solution
    from validator.prediction import Error, ErrorFamily, Prediction
    from validator.runner import Run


INSTRUCTIONS = "Create a polar line chart. Set the radial axis label to 'Magnitude'."


def requirements():
    return [
        {
            "id": "chart-projection",
            "kind": "chart_type",
            "requirement": "Use a polar projection for the line chart.",
            "scope": "main panel",
            "source_quote": "polar line chart",
        },
        {
            "id": "radial-label",
            "kind": "axes",
            "requirement": "Give the radial axis the label Magnitude.",
            "scope": "radial axis",
            "source_quote": "Set the radial axis label to 'Magnitude'.",
        },
    ]


def indexed_requirements():
    return [
        {**{key: value for key, value in item.items() if key != "source_quote"}, "source_id": f"s{index + 1}"}
        for index, item in enumerate(requirements())
    ]


def checks(status="pass"):
    return [
        {"id": item["id"], "status": status, "evidence": f"Evidence for {item['id']}."}
        for item in requirements()
    ]


def model_response(document, finish_reason="stop"):
    content = document if isinstance(document, str) else json.dumps(document)
    return {"choices": [{"message": {"content": content}, "finish_reason": finish_reason}]}


def bad_request(message):
    return BadRequestError(
        message,
        response=httpx.Response(
            400, request=httpx.Request("POST", "https://offline.invalid/v1/chat/completions")
        ),
        body={"error": {"message": message}},
    )


class DesignRequirementParserTests(unittest.TestCase):
    def test_accepts_task_grounded_requirements(self):
        expected = requirements()
        result = solution._parse_design_requirements(
            json.dumps({"requirements": expected}), INSTRUCTIONS
        )
        self.assertEqual(result, expected)

    def test_accepts_empty_requirements(self):
        self.assertEqual(
            solution._parse_design_requirements('{"requirements":[]}', INSTRUCTIONS), []
        )

    def test_source_quote_matching_normalizes_whitespace(self):
        extracted = requirements()[:1]
        extracted[0]["source_quote"] = "Create\n  a\tpolar line chart."
        result = solution._parse_design_requirements(
            json.dumps({"requirements": extracted}), INSTRUCTIONS
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], extracted[0]["id"])

    def test_rejects_invalid_top_level_schema(self):
        for document in (
            [],
            {"requirements": requirements(), "explanation": "extra"},
            {"requirements": "polar chart"},
            {"requirements": None},
            {"other": requirements()},
        ):
            with self.subTest(document=document), self.assertRaises(ValueError):
                solution._parse_design_requirements(json.dumps(document), INSTRUCTIONS)

    def test_rejects_missing_extra_invalid_or_empty_requirement_fields(self):
        valid = requirements()[0]
        invalid = {
            "extra_key": {**valid, "priority": "high"},
            "missing_key": {key: value for key, value in valid.items() if key != "scope"},
            "unknown_kind": {**valid, "kind": "data"},
            "boolean_id": {**valid, "id": True},
            "numeric_id": {**valid, "id": 1},
            "invented_quote": {**valid, "source_quote": "The legend must be outside the plot."},
            "non_string_requirement": {**valid, "requirement": ["polar"]},
            "non_string_scope": {**valid, "scope": False},
            "non_string_quote": {**valid, "source_quote": 1},
        }
        for field in valid:
            invalid[f"empty_{field}"] = {**valid, field: ""}
            invalid[f"whitespace_{field}"] = {**valid, field: " \n\t "}
        for name, item in invalid.items():
            with self.subTest(case=name), self.assertRaises(ValueError):
                solution._parse_design_requirements(
                    json.dumps({"requirements": [item]}), INSTRUCTIONS
                )

    def test_rejects_duplicate_requirement_ids(self):
        repeated = requirements()
        repeated[1]["id"] = repeated[0]["id"]
        with self.assertRaises(ValueError):
            solution._parse_design_requirements(
                json.dumps({"requirements": repeated}), INSTRUCTIONS
            )

    def test_rejects_duplicate_json_keys(self):
        for content in (
            '{"requirements":[],"requirements":[]}',
            '{"requirements":[{"id":"one","id":"two","kind":"chart_type",'
            '"requirement":"Use polar coordinates","scope":"main panel",'
            '"source_quote":"polar line chart"}]}',
        ):
            with self.subTest(content=content), self.assertRaises(ValueError):
                solution._parse_design_requirements(content, INSTRUCTIONS)

    def test_accepts_every_allowed_requirement_kind(self):
        for kind in ("chart_type", "layout", "axes", "legend_annotation", "style"):
            extracted = requirements()[:1]
            extracted[0]["kind"] = kind
            with self.subTest(kind=kind):
                result = solution._parse_design_requirements(
                    json.dumps({"requirements": extracted}), INSTRUCTIONS
                )
                self.assertEqual(result[0]["kind"], kind)

    def test_rejects_non_object_requirements(self):
        for item in (None, "polar chart", 1, ["polar chart"]):
            with self.subTest(item=item), self.assertRaises(ValueError):
                solution._parse_design_requirements(
                    json.dumps({"requirements": [item]}), INSTRUCTIONS
                )


class IndexedRequirementParserTests(unittest.TestCase):
    def setUp(self):
        self.sources = {
            "s1": "Create a polar line chart and add a legend.",
            "s2": "Set the radial axis label to 'Magnitude'.",
        }
        self.instructions = "\n\n".join(self.sources.values())

    def test_restores_the_original_source_line_with_and_and_punctuation(self):
        result = solution._parse_indexed_requirements(
            json.dumps({"requirements": indexed_requirements()}), self.instructions, self.sources
        )
        self.assertEqual([item["source_quote"] for item in result], list(self.sources.values()))
        self.assertTrue(all("source_id" not in item for item in result))

    def test_rejects_unknown_boolean_and_other_invalid_source_ids(self):
        for source_id in ("s99", True, 1, None, "", " ", ["s1"]):
            invalid = indexed_requirements()
            invalid[0]["source_id"] = source_id
            with self.subTest(source_id=source_id), self.assertRaises(ValueError):
                solution._parse_indexed_requirements(
                    json.dumps({"requirements": invalid}), self.instructions, self.sources
                )

    def test_rejects_missing_extra_or_mixed_source_fields(self):
        valid = indexed_requirements()[0]
        for item in (
            {key: value for key, value in valid.items() if key != "scope"},
            {**valid, "source_quote": self.sources["s1"]},
            {**valid, "extra": "not allowed"},
        ):
            with self.subTest(item=item), self.assertRaises(ValueError):
                solution._parse_indexed_requirements(
                    json.dumps({"requirements": [item]}), self.instructions, self.sources
                )

    def test_rejects_duplicate_json_source_keys_and_duplicate_requirement_ids(self):
        duplicate_keys = (
            '{"requirements":[{"id":"one","kind":"chart_type",'
            '"requirement":"Use polar coordinates","scope":"main panel",'
            '"source_id":"s1","source_id":"s2"}]}'
        )
        with self.assertRaises(ValueError):
            solution._parse_indexed_requirements(duplicate_keys, self.instructions, self.sources)
        duplicated = indexed_requirements()
        duplicated[1]["id"] = duplicated[0]["id"]
        with self.assertRaises(ValueError):
            solution._parse_indexed_requirements(
                json.dumps({"requirements": duplicated}), self.instructions, self.sources
            )

    def test_compatible_quotes_still_require_canonical_grounding_and_exact_fields(self):
        self.assertEqual(
            solution._parse_indexed_requirements(
                json.dumps({"requirements": requirements()}), self.instructions, self.sources
            ),
            requirements(),
        )
        for malformed in (
            {**requirements()[0], "source_quote": "A fabricated title is required."},
            {**requirements()[0], "extra": "not allowed"},
        ):
            with self.subTest(item=malformed), self.assertRaises(ValueError):
                solution._parse_indexed_requirements(
                    json.dumps({"requirements": [malformed]}), self.instructions, self.sources
                )


class DesignAuditParserTests(unittest.TestCase):
    def test_accepts_all_three_statuses(self):
        for status in ("pass", "fail", "uncertain"):
            expected = checks(status)
            with self.subTest(status=status):
                self.assertEqual(
                    solution._parse_design_audit(
                        json.dumps({"checks": expected}), requirements()
                    ),
                    expected,
                )

    def test_requires_exact_coverage_of_a_targeted_subset(self):
        subset = requirements()[1:]
        expected = checks()[1:]
        self.assertEqual(
            solution._parse_design_audit(json.dumps({"checks": expected}), subset), expected
        )
        with self.assertRaises(ValueError):
            solution._parse_design_audit(json.dumps({"checks": checks()}), subset)

    def test_rejects_missing_unknown_or_duplicate_ids(self):
        valid = checks()
        invalid = {
            "missing_id": valid[:1],
            "unknown_id": [valid[0], {**valid[1], "id": "not-requested"}],
            "duplicate_id": [valid[0], {**valid[1], "id": valid[0]["id"]}],
            "boolean_id": [valid[0], {**valid[1], "id": True}],
        }
        for name, items in invalid.items():
            with self.subTest(case=name), self.assertRaises(ValueError):
                solution._parse_design_audit(json.dumps({"checks": items}), requirements())

    def test_rejects_invalid_fields_types_or_status(self):
        valid = checks()[0]
        invalid = {
            "extra_key": {**valid, "confidence": 0.8},
            "missing_key": {"id": valid["id"], "status": "pass"},
            "unknown_status": {**valid, "status": "probably_pass"},
            "boolean_status": {**valid, "status": False},
            "empty_evidence": {**valid, "evidence": ""},
            "whitespace_evidence": {**valid, "evidence": " \t\n "},
            "non_string_evidence": {**valid, "evidence": ["looks fine"]},
        }
        for name, item in invalid.items():
            with self.subTest(case=name), self.assertRaises(ValueError):
                solution._parse_design_audit(
                    json.dumps({"checks": [item, checks()[1]]}), requirements()
                )

    def test_rejects_invalid_top_level_schema(self):
        for document in (
            [],
            {"checks": checks(), "summary": "extra"},
            {"checks": "pass"},
            {"checks": None},
            {"other": checks()},
        ):
            with self.subTest(document=document), self.assertRaises(ValueError):
                solution._parse_design_audit(json.dumps(document), requirements())

    def test_rejects_duplicate_json_keys(self):
        for content in (
            '{"checks":[],"checks":[]}',
            '{"checks":[{"id":"chart-projection","status":"fail","status":"pass",'
            '"evidence":"A polar projection is visible."}]}',
        ):
            with self.subTest(content=content), self.assertRaises(ValueError):
                solution._parse_design_audit(content, requirements()[:1])

    def test_rejects_non_object_checks(self):
        for item in (None, "pass", True, ["pass"]):
            with self.subTest(item=item), self.assertRaises(ValueError):
                solution._parse_design_audit(
                    json.dumps({"checks": [item]}), requirements()[:1]
                )


class ImageObservationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="image-observer-test-")
        self.addCleanup(self.directory.cleanup)
        self.figure = Path(self.directory.name) / "figure.png"
        Image.new("RGB", (120, 100), "white").save(self.figure, format="PNG")
        self.image_bytes = self.figure.read_bytes()
        self.description = "One panel is visible. No separate x-axis title can be read."

    def test_observer_receives_only_the_image_without_task_inputs_or_trajectory(self):
        with patch(
            "validator.solution.complete", return_value=model_response(self.description)
        ) as complete, patch.object(Path, "read_text", side_effect=AssertionError("Unexpected input read")), patch.object(
            Path, "read_bytes", side_effect=AssertionError("Unexpected artifact read")
        ):
            self.assertEqual(solution._observe_figure(self.image_bytes), self.description)
        complete.assert_called_once()
        self.assertEqual(complete.call_args.kwargs["max_tokens"], 1024)
        self.assertNotIn("response_format", complete.call_args.kwargs)
        messages = complete.call_args.args[0]
        self.assertEqual(messages[0]["content"], solution.DESIGN_OBSERVATION_POLICY)
        shown = str(messages)
        for excluded in (INSTRUCTIONS, "INPUT_FILE_ONLY_MARKER", "AGENT_CODE_ONLY_MARKER", "AGENT_RESULT_ONLY_MARKER"):
            self.assertNotIn(excluded, shown)
        content = messages[-1]["content"]
        prompt = next(part["text"] for part in content if part["type"] == "text")
        self.assertEqual(prompt.strip(), "Describe this final figure.")
        image_url = next(part["image_url"]["url"] for part in content if part["type"] == "image_url")
        self.assertEqual(base64.b64decode(image_url.split(",", 1)[1]), self.image_bytes)

    def test_empty_observation_is_retried_once(self):
        with patch(
            "validator.solution.complete",
            side_effect=[model_response(" \n\t "), model_response(self.description)],
        ) as complete:
            self.assertEqual(solution._observe_figure(self.image_bytes), self.description)
        self.assertEqual(complete.call_count, 2)
        self.assertIn("nonempty image observation", str(complete.call_args.args[0]))

    def test_non_string_observation_is_retried_instead_of_accepted(self):
        for content in (None, [], {}, False, 0):
            invalid_response = {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}
            with self.subTest(content=content), patch(
                "validator.solution.complete",
                side_effect=[invalid_response, model_response(self.description)],
            ) as complete:
                self.assertEqual(solution._observe_figure(self.image_bytes), self.description)
            self.assertEqual(complete.call_count, 2)

    def test_two_empty_observations_raise_instead_of_inventing_visual_evidence(self):
        with patch(
            "validator.solution.complete", return_value=model_response(" ")
        ) as complete, self.assertRaises(RuntimeError):
            solution._observe_figure(self.image_bytes)
        self.assertEqual(complete.call_count, 2)

    def test_truncated_observation_is_retried_once(self):
        with patch(
            "validator.solution.complete",
            side_effect=[model_response(self.description, "length"), model_response(self.description)],
        ) as complete:
            self.assertEqual(solution._observe_figure(self.image_bytes), self.description)
        self.assertEqual(complete.call_count, 2)

    def test_two_truncated_observations_raise(self):
        with patch(
            "validator.solution.complete", return_value=model_response(self.description, "length")
        ) as complete, self.assertRaises(RuntimeError):
            solution._observe_figure(self.image_bytes)
        self.assertEqual(complete.call_count, 2)

    def test_transport_error_propagates_without_schema_retry_or_downscaling(self):
        error = RuntimeError("Image observation endpoint unavailable")
        with patch("validator.solution.complete", side_effect=error) as complete, patch(
            "validator.solution.baseline._downscale"
        ) as downscale, patch("validator.solution.tokenize") as tokenize:
            with self.assertRaises(RuntimeError) as raised:
                solution._observe_figure(self.image_bytes)
        self.assertIs(raised.exception, error)
        complete.assert_called_once()
        downscale.assert_not_called()
        tokenize.assert_not_called()

    def test_context_overflow_downscales_only_the_in_memory_image(self):
        with patch(
            "validator.solution.complete",
            side_effect=[bad_request("maximum context length exceeded"), model_response(self.description)],
        ) as complete, patch("validator.solution.tokenize") as tokenize, self.assertWarns(UserWarning):
            self.assertEqual(solution._observe_figure(self.image_bytes), self.description)
        self.assertEqual(complete.call_count, 2)
        content = complete.call_args.args[0][-1]["content"]
        image_url = next(part["image_url"]["url"] for part in content if part["type"] == "image_url")
        with Image.open(io.BytesIO(base64.b64decode(image_url.split(",", 1)[1]))) as resized:
            self.assertLess(resized.width, 120)
            self.assertLess(resized.height, 100)
        self.assertEqual(self.figure.read_bytes(), self.image_bytes)
        tokenize.assert_not_called()

    def test_non_context_bad_request_does_not_downscale_or_retry(self):
        error = bad_request("Unsupported request parameter")
        with patch("validator.solution.complete", side_effect=error) as complete, patch(
            "validator.solution.baseline._downscale"
        ) as downscale:
            with self.assertRaises(BadRequestError) as raised:
                solution._observe_figure(self.image_bytes)
        self.assertIs(raised.exception, error)
        complete.assert_called_once()
        downscale.assert_not_called()


class DesignModelInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="design-model-test-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.figure = self.root / "figure.png"
        Image.new("RGB", (120, 100), "white").save(self.figure, format="PNG")
        self.input = self.root / "values.csv"
        self.input.write_text("INPUT_FILE_ONLY_MARKER\nx,y\n1,2\n")
        self.messages = [
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": "design-tool-call",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": json.dumps({"command": "python AGENT_CODE_ONLY_MARKER.py"}),
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "design-tool-call",
                "content": "AGENT_RESULT_ONLY_MARKER: generated the final plot.",
            },
        ]
        self.run = Run(
            "interface-design-run", "interface-design-task", INSTRUCTIONS,
            self.messages, self.figure, {"values.csv": self.input},
        )
        self.observation = "OBSERVATION_ONLY_MARKER: one panel with no visible axis title."
        self.observer_patch = patch("validator.solution._observe_figure", return_value=self.observation)
        self.observer = self.observer_patch.start()
        self.addCleanup(self.observer_patch.stop)

    def query_stage(self, stage):
        if stage == "extraction":
            return solution._extract_design_requirements(self.run)
        return solution._audit_design(self.run, requirements())

    def valid_document(self, stage):
        return {"requirements": requirements()} if stage == "extraction" else {"checks": checks()}

    def test_extraction_schema_requires_only_indexed_fields_and_available_sources(self):
        with patch(
            "validator.solution.complete",
            return_value=model_response({"requirements": indexed_requirements()[:1]}),
        ) as complete:
            solution._extract_design_requirements(self.run)
        format_spec = complete.call_args.kwargs["response_format"]
        self.assertEqual(format_spec["type"], "json_schema")
        self.assertIs(format_spec["json_schema"]["strict"], True)
        schema = format_spec["json_schema"]["schema"]
        self.assertIs(schema["additionalProperties"], False)
        self.assertEqual(schema["required"], ["requirements"])
        self.assertEqual(set(schema["properties"]), {"requirements"})
        items = schema["properties"]["requirements"]["items"]
        fields = {"id", "kind", "requirement", "scope", "source_id"}
        self.assertEqual(set(items["properties"]), fields)
        self.assertEqual(set(items["required"]), fields)
        self.assertIs(items["additionalProperties"], False)
        self.assertEqual(set(items["properties"]["kind"]["enum"]), {
            "chart_type", "layout", "axes", "legend_annotation", "style",
        })
        self.assertEqual(items["properties"]["source_id"]["enum"], ["s1"])
        self.assertTrue(all(value["type"] == "string" for value in items["properties"].values()))

    def test_audit_schema_requires_every_check_and_only_the_supplied_ids(self):
        with patch(
            "validator.solution.complete", return_value=model_response({"checks": checks()})
        ) as complete:
            solution._audit_design(self.run, requirements())
        format_spec = complete.call_args.kwargs["response_format"]
        self.assertEqual(format_spec["type"], "json_schema")
        self.assertIs(format_spec["json_schema"]["strict"], True)
        schema = format_spec["json_schema"]["schema"]
        self.assertIs(schema["additionalProperties"], False)
        self.assertEqual(schema["required"], ["checks"])
        self.assertEqual(set(schema["properties"]), {"checks"})
        array = schema["properties"]["checks"]
        self.assertEqual((array["minItems"], array["maxItems"]), (2, 2))
        items = array["items"]
        self.assertEqual(set(items["properties"]), {"id", "status", "evidence"})
        self.assertEqual(set(items["required"]), {"id", "status", "evidence"})
        self.assertIs(items["additionalProperties"], False)
        self.assertEqual(items["properties"]["id"]["enum"], [item["id"] for item in requirements()])
        self.assertEqual(set(items["properties"]["status"]["enum"]), {"pass", "fail", "uncertain"})

    def test_empty_or_whitespace_only_task_skips_model_and_observer(self):
        for instructions in ("", " \t\n\n "):
            run = Run(
                "empty-task-run", "empty-task", instructions,
                self.messages, self.figure, {"values.csv": self.input},
            )
            with self.subTest(instructions=instructions), patch("validator.solution.complete") as complete:
                self.assertEqual(solution._extract_design_requirements(run), [])
                self.assertEqual(solution.judge_chart(run), [])
            complete.assert_not_called()
        self.observer.assert_not_called()

    def test_extraction_receives_only_the_task_and_never_reads_run_artifacts(self):
        with patch(
            "validator.solution.complete", return_value=model_response({"requirements": requirements()})
        ) as complete, patch.object(Path, "read_bytes", side_effect=AssertionError("Unexpected artifact read")), patch.object(
            Path, "read_text", side_effect=AssertionError("Unexpected input read")
        ):
            self.assertEqual(solution._extract_design_requirements(self.run), requirements())
        complete.assert_called_once()
        messages = complete.call_args.args[0]
        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertTrue(all(isinstance(message["content"], str) for message in messages))
        shown = str(messages)
        self.assertIn(INSTRUCTIONS, messages[-1]["content"])
        for excluded in ("INPUT_FILE_ONLY_MARKER", "AGENT_CODE_ONLY_MARKER", "AGENT_RESULT_ONLY_MARKER"):
            self.assertNotIn(excluded, shown)
        self.assertNotIn("image_url", shown)
        self.assertNotIn("model", complete.call_args.kwargs)

    def test_audit_receives_the_exact_final_png_full_trajectory_and_input_data(self):
        with patch(
            "validator.solution.complete", return_value=model_response({"checks": checks()})
        ) as complete:
            self.assertEqual(solution._audit_design(self.run, requirements()), checks())
        complete.assert_called_once()
        messages = complete.call_args.args[0]
        content = messages[-1]["content"]
        prompt = next(part["text"] for part in content if part["type"] == "text")
        image_url = next(part["image_url"]["url"] for part in content if part["type"] == "image_url")
        self.assertTrue(image_url.startswith("data:image/png;base64,"))
        self.assertEqual(base64.b64decode(image_url.split(",", 1)[1]), self.figure.read_bytes())
        self.assertIn(INSTRUCTIONS, prompt)
        self.assertIn(self.input.read_text(), prompt)
        serialized = prompt.split("Trajectory (including actual tool calls and results):\n", 1)[1]
        shown_messages, _ = json.JSONDecoder().raw_decode(serialized)
        self.assertEqual(shown_messages, self.messages)
        self.assertIn(self.observation, prompt)
        self.observer.assert_called_once_with(self.figure.read_bytes())
        self.assertNotIn("model", complete.call_args.kwargs)

    def test_extraction_numbers_only_nonempty_original_lines_and_restores_them(self):
        lines = ["Create a polar line chart.", "Set the radial axis label to 'Magnitude'."]
        instructions = "  " + lines[0] + "  \n\n \t\n " + lines[1] + "\n"
        run = Run(
            "indexed-task-run", "indexed-task", instructions,
            self.messages, self.figure, {"values.csv": self.input},
        )
        with patch(
            "validator.solution.complete",
            return_value=model_response({"requirements": indexed_requirements()}),
        ) as complete:
            result = solution._extract_design_requirements(run)
        complete.assert_called_once()
        task_prompt = complete.call_args.args[0][-1]["content"]
        self.assertIn(instructions, task_prompt)
        supplied, _ = json.JSONDecoder().raw_decode(task_prompt.split("Verbatim task-line sources:\n", 1)[1])
        self.assertEqual(supplied, {"s1": lines[0], "s2": lines[1]})
        self.assertEqual([item["source_quote"] for item in result], lines)
        schema = complete.call_args.kwargs["response_format"]["json_schema"]["schema"]
        self.assertEqual(
            schema["properties"]["requirements"]["items"]["properties"]["source_id"]["enum"],
            ["s1", "s2"],
        )

    def test_extraction_repairs_an_unknown_source_id_without_guessing_its_quote(self):
        invalid = indexed_requirements()[:1]
        invalid[0]["source_id"] = "s99"
        corrected = indexed_requirements()[:1]
        with patch(
            "validator.solution.complete",
            side_effect=[
                model_response({"requirements": invalid}),
                model_response({"requirements": corrected}),
            ],
        ) as complete:
            result = solution._extract_design_requirements(self.run)
        self.assertEqual(complete.call_count, 2)
        self.assertEqual(result[0]["source_quote"], INSTRUCTIONS)
        self.assertIn("s99", complete.call_args.args[0][-1]["content"])
        self.assertIn("choose a supplied task-line id", complete.call_args.args[0][-1]["content"])

    def test_malformed_json_is_retried_once_for_each_stage(self):
        for stage in ("extraction", "audit"):
            with self.subTest(stage=stage), patch(
                "validator.solution.complete",
                side_effect=[model_response("YES"), model_response(self.valid_document(stage))],
            ) as complete:
                result = self.query_stage(stage)
            self.assertEqual(result, requirements() if stage == "extraction" else checks())
            self.assertEqual(complete.call_count, 2)
            self.assertIn("schema validation", str(complete.call_args.args[0]))

    def test_two_malformed_responses_raise_for_each_stage(self):
        for stage in ("extraction", "audit"):
            with self.subTest(stage=stage), patch(
                "validator.solution.complete", return_value=model_response("not JSON")
            ) as complete, self.assertRaises(RuntimeError):
                self.query_stage(stage)
            self.assertEqual(complete.call_count, 2)

    def test_truncated_valid_json_is_not_accepted_for_either_stage(self):
        for stage in ("extraction", "audit"):
            document = self.valid_document(stage)
            with self.subTest(stage=stage), patch(
                "validator.solution.complete",
                side_effect=[model_response(document, "length"), model_response(document)],
            ) as complete:
                result = self.query_stage(stage)
            self.assertEqual(result, requirements() if stage == "extraction" else checks())
            self.assertEqual(complete.call_count, 2)

    def test_two_truncated_answers_raise_for_each_stage(self):
        for stage in ("extraction", "audit"):
            with self.subTest(stage=stage), patch(
                "validator.solution.complete",
                return_value=model_response(self.valid_document(stage), "length"),
            ) as complete, self.assertRaises(RuntimeError):
                self.query_stage(stage)
            self.assertEqual(complete.call_count, 2)

    def test_transport_exception_is_not_retried_as_a_schema_error(self):
        for stage in ("extraction", "audit"):
            error = RuntimeError("The validator endpoint could not be reached")
            with self.subTest(stage=stage), patch(
                "validator.solution.complete", side_effect=error
            ) as complete, patch("validator.solution.tokenize") as tokenize:
                with self.assertRaises(RuntimeError) as raised:
                    self.query_stage(stage)
            self.assertIs(raised.exception, error)
            complete.assert_called_once()
            tokenize.assert_not_called()

    def test_audit_retries_an_incomplete_checklist_instead_of_treating_it_as_pass(self):
        with patch(
            "validator.solution.complete",
            side_effect=[model_response({"checks": checks()[:1]}), model_response({"checks": checks()})],
        ) as complete:
            self.assertEqual(solution._audit_design(self.run, requirements()), checks())
        self.assertEqual(complete.call_count, 2)

    def test_requirement_extraction_retries_an_invented_task_quote(self):
        invented = requirements()
        invented[0]["source_quote"] = "The plot must use a blue dashed line."
        with patch(
            "validator.solution.complete",
            side_effect=[
                model_response({"requirements": invented}),
                model_response({"requirements": requirements()}),
            ],
        ) as complete:
            self.assertEqual(solution._extract_design_requirements(self.run), requirements())
        self.assertEqual(complete.call_count, 2)

    def test_quote_repair_preserves_and_instead_of_rewriting_it_as_a_period(self):
        instructions = (
            "Set the title to 'Trend' and label the x-axis 'Time' and the y-axis 'Value'."
        )
        original_quote = "label the x-axis 'Time' and the y-axis 'Value'."
        invalid_quote = original_quote.replace(" and ", ". ")
        corrected = [{
            "id": "axis-x",
            "kind": "axes",
            "requirement": "Set the x-axis label to Time.",
            "scope": "x-axis",
            "source_quote": original_quote,
        }]
        malformed = [{**corrected[0], "source_quote": invalid_quote}]
        with self.assertRaisesRegex(ValueError, "axis-x"):
            solution._parse_design_requirements(
                json.dumps({"requirements": malformed}), instructions
            )
        run = Run(
            "quote-repair-run", "quote-repair-task", instructions,
            self.messages, self.figure, {"values.csv": self.input},
        )
        with patch(
            "validator.solution.complete",
            side_effect=[
                model_response({"requirements": malformed}),
                model_response({"requirements": corrected}),
            ],
        ) as complete, patch("validator.solution.tokenize") as tokenize:
            self.assertEqual(solution._extract_design_requirements(run), corrected)
        self.assertEqual(complete.call_count, 2)
        repair_note = complete.call_args.args[0][-1]["content"]
        self.assertIn("axis-x", repair_note)
        self.assertIn(invalid_quote, repair_note)
        self.assertIn("Copy the original task's exact text, including punctuation", repair_note)
        self.assertIn("a shorter exact substring is permitted", repair_note)
        tokenize.assert_not_called()

    def test_uncertainty_rechecks_only_the_targeted_requirement_with_full_evidence(self):
        initial = checks()
        initial[1]["status"] = "uncertain"
        reviewed = [{**initial[1], "status": "pass"}]
        self.observer_patch.stop()
        with patch(
            "validator.solution.complete",
            side_effect=[
                model_response({"requirements": requirements()}),
                model_response("First image-only observation: no separate radial label is visible."),
                model_response({"checks": initial}),
                model_response("Second image-only observation: the radial label is small but visible."),
                model_response({"checks": reviewed}),
            ],
        ) as complete:
            self.assertEqual(solution.judge_chart(self.run), [])
        self.assertEqual(complete.call_count, 5)
        for call in (complete.call_args_list[1], complete.call_args_list[3]):
            shown = str(call.args[0])
            self.assertNotIn(INSTRUCTIONS, shown)
            self.assertNotIn("AGENT_CODE_ONLY_MARKER", shown)
            self.assertNotIn("AGENT_RESULT_ONLY_MARKER", shown)
            self.assertNotIn("INPUT_FILE_ONLY_MARKER", shown)
            self.assertEqual(call.kwargs["max_tokens"], 1024)
        audit_prompts = []
        for call in (complete.call_args_list[2], complete.call_args_list[4]):
            content = call.args[0][-1]["content"]
            prompt = next(part["text"] for part in content if part["type"] == "text")
            audit_prompts.append(prompt)
            image_url = next(part["image_url"]["url"] for part in content if part["type"] == "image_url")
            self.assertEqual(base64.b64decode(image_url.split(",", 1)[1]), self.figure.read_bytes())
            self.assertIn("AGENT_CODE_ONLY_MARKER", prompt)
            self.assertIn("AGENT_RESULT_ONLY_MARKER", prompt)
            self.assertIn("INPUT_FILE_ONLY_MARKER", prompt)
        supplied = [
            json.JSONDecoder().raw_decode(prompt.split("check every id, no new requirements):\n", 1)[1])[0]
            for prompt in audit_prompts
        ]
        self.assertEqual(supplied, [requirements(), requirements()[1:]])
        self.assertIn("First image-only observation", audit_prompts[0])
        self.assertIn("Second image-only observation", audit_prompts[1])
        for call, subset in zip(
            (complete.call_args_list[2], complete.call_args_list[4]),
            (requirements(), requirements()[1:]),
        ):
            array = call.kwargs["response_format"]["json_schema"]["schema"]["properties"]["checks"]
            self.assertEqual((array["minItems"], array["maxItems"]), (len(subset), len(subset)))
            self.assertEqual(array["items"]["properties"]["id"]["enum"], [item["id"] for item in subset])

    def test_audit_context_fallback_budgets_the_design_policy_and_complete_checklist(self):
        original_image = self.figure.read_bytes()
        original_input = self.input.read_text()
        retained = json.dumps(self.messages)
        with patch(
            "validator.solution.complete",
            side_effect=[
                bad_request("maximum context length exceeded"),
                model_response({"checks": checks()}),
            ],
        ) as complete, patch(
            "validator.solution._compact_execution_evidence",
            return_value=(retained, "COMPACTED_INPUT_MARKER"),
        ) as compact, patch("validator.solution.tokenize") as tokenize, self.assertWarns(UserWarning):
            self.assertEqual(solution._audit_design(self.run, requirements()), checks())
        compact.assert_called_once()
        self.assertEqual(compact.call_args.args[0], self.run)
        self.assertEqual(compact.call_args.args[1], "values.csv:\n" + original_input)
        fixed_prompt = compact.call_args.args[2]
        checklist, _ = json.JSONDecoder().raw_decode(
            fixed_prompt.split("check every id, no new requirements):\n", 1)[1]
        )
        self.assertEqual(checklist, requirements())
        self.assertEqual(compact.call_args.kwargs["policy"], solution.DESIGN_AUDIT_POLICY)
        self.assertEqual(compact.call_args.kwargs["max_output_tokens"], 4096)
        self.assertEqual(complete.call_count, 2)
        self.assertEqual(
            complete.call_args_list[0].kwargs["response_format"],
            complete.call_args_list[1].kwargs["response_format"],
        )
        final_messages = complete.call_args.args[0]
        self.assertEqual(final_messages[0]["content"], solution.DESIGN_AUDIT_POLICY)
        content = final_messages[-1]["content"]
        prompt = next(part["text"] for part in content if part["type"] == "text")
        self.assertIn("COMPACTED_INPUT_MARKER", prompt)
        self.assertIn("NOT an agent error", prompt)
        checklist, _ = json.JSONDecoder().raw_decode(
            prompt.split("check every id, no new requirements):\n", 1)[1]
        )
        self.assertEqual(checklist, requirements())
        image_url = next(part["image_url"]["url"] for part in content if part["type"] == "image_url")
        self.assertEqual(base64.b64decode(image_url.split(",", 1)[1]), original_image)
        self.assertTrue(all(call.kwargs["max_tokens"] == 4096 for call in complete.call_args_list))
        self.assertEqual(self.figure.read_bytes(), original_image)
        self.assertEqual(self.input.read_text(), original_input)
        tokenize.assert_not_called()

    def test_audit_image_downscaling_occurs_in_memory_without_changing_the_saved_figure(self):
        original_image = self.figure.read_bytes()
        with patch(
            "validator.solution.complete",
            side_effect=[
                bad_request("maximum context length exceeded"),
                bad_request("maximum context length exceeded"),
                model_response({"checks": checks()}),
            ],
        ) as complete, patch(
            "validator.solution._compact_execution_evidence",
            return_value=(json.dumps(self.messages), ""),
        ), patch("validator.solution.tokenize"), self.assertWarns(UserWarning):
            self.assertEqual(solution._audit_design(self.run, requirements()), checks())
        self.assertEqual(complete.call_count, 3)
        original_format = complete.call_args_list[0].kwargs["response_format"]
        self.assertTrue(all(call.kwargs["response_format"] == original_format for call in complete.call_args_list))
        content = complete.call_args.args[0][-1]["content"]
        image_url = next(part["image_url"]["url"] for part in content if part["type"] == "image_url")
        resized_bytes = base64.b64decode(image_url.split(",", 1)[1])
        with Image.open(io.BytesIO(resized_bytes)) as resized:
            self.assertLess(resized.width, 120)
            self.assertLess(resized.height, 100)
        self.assertEqual(self.figure.read_bytes(), original_image)

    def test_audit_non_context_bad_request_propagates_without_compaction(self):
        error = bad_request("Unsupported request parameter")
        with patch("validator.solution.complete", side_effect=error) as complete, patch(
            "validator.solution._compact_execution_evidence"
        ) as compact, patch("validator.solution.tokenize") as tokenize:
            with self.assertRaises(BadRequestError) as raised:
                solution._audit_design(self.run, requirements())
        self.assertIs(raised.exception, error)
        complete.assert_called_once()
        compact.assert_not_called()
        tokenize.assert_not_called()


class ChartJudgmentTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="design-validator-test-")
        self.addCleanup(self.directory.cleanup)
        self.figure = Path(self.directory.name) / "figure.png"
        Image.new("RGB", (120, 100), "white").save(self.figure, format="PNG")
        self.run = Run(
            "offline-design-run", "offline-design-task", INSTRUCTIONS, [], self.figure, {}
        )

    def test_all_pass_returns_no_chart_error(self):
        with patch("validator.solution._extract_design_requirements", return_value=requirements()), patch(
            "validator.solution._audit_design", return_value=checks()
        ):
            self.assertEqual(solution.judge_chart(self.run), [])

    def test_multiple_failures_merge_into_one_schema_valid_chart_error(self):
        failures = checks("fail")
        failures[0]["evidence"] = "The final panel uses Cartesian rather than polar coordinates."
        failures[1]["evidence"] = "The radial axis has no Magnitude label."
        with patch("validator.solution._extract_design_requirements", return_value=requirements()), patch(
            "validator.solution._audit_design", return_value=failures
        ):
            errors = solution.judge_chart(self.run)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].family, ErrorFamily.WRONG_CHART)
        for requirement, failure in zip(requirements(), failures):
            self.assertIn(requirement["source_quote"], errors[0].evidence)
            self.assertIn(requirement["scope"], errors[0].evidence)
            self.assertIn(failure["evidence"], errors[0].evidence)
        Prediction(run_id=self.run.run_id, errors=errors)

    def test_uncertain_then_confirmed_fail_returns_chart_error(self):
        initial = checks()
        initial[1]["status"] = "uncertain"
        reviewed = [{**initial[1], "status": "fail", "evidence": "The requested radial label is absent."}]
        with patch("validator.solution._extract_design_requirements", return_value=requirements()), patch(
            "validator.solution._audit_design", side_effect=[initial, reviewed]
        ):
            errors = solution.judge_chart(self.run)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].family, ErrorFamily.WRONG_CHART)
        self.assertIn(reviewed[0]["evidence"], errors[0].evidence)

    def test_uncertain_then_pass_returns_no_chart_error(self):
        initial = checks()
        initial[1]["status"] = "uncertain"
        reviewed = [{**initial[1], "status": "pass"}]
        with patch("validator.solution._extract_design_requirements", return_value=requirements()), patch(
            "validator.solution._audit_design", side_effect=[initial, reviewed]
        ):
            self.assertEqual(solution.judge_chart(self.run), [])

    def test_unresolved_uncertainty_does_not_silently_pass(self):
        initial = checks()
        initial[1]["status"] = "uncertain"
        with patch("validator.solution._extract_design_requirements", return_value=requirements()), patch(
            "validator.solution._audit_design", side_effect=[initial, [initial[1]]]
        ):
            with self.assertRaises(RuntimeError):
                solution.judge_chart(self.run)

    def test_confirmed_failure_remains_reportable_with_other_uncertainty(self):
        initial = checks()
        initial[0]["status"] = "fail"
        initial[1]["status"] = "uncertain"
        with patch("validator.solution._extract_design_requirements", return_value=requirements()), patch(
            "validator.solution._audit_design", side_effect=[initial, [initial[1]]]
        ):
            errors = solution.judge_chart(self.run)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].family, ErrorFamily.WRONG_CHART)
        self.assertIn(initial[0]["evidence"], errors[0].evidence)

    def test_no_requirements_skips_the_audit(self):
        with patch("validator.solution._extract_design_requirements", return_value=[]), patch(
            "validator.solution._audit_design"
        ) as audit:
            self.assertEqual(solution.judge_chart(self.run), [])
        audit.assert_not_called()

    def test_shared_global_label_does_not_require_a_set_xlabel_call(self):
        instructions = (
            "Create two vertically stacked line-chart panels with a shared x-axis. "
            "Use a single global x-axis label 'Time (s)' for both panels."
        )
        extracted = [{
            "id": "shared-label",
            "kind": "axes",
            "requirement": "Show one shared Time (s) x-axis label.",
            "scope": "whole figure",
            "source_quote": "Use a single global x-axis label 'Time (s)' for both panels.",
        }]
        run = Run(
            "shared-label-run", "shared-label-task", instructions,
            [{"role": "tool", "content": "fig.supxlabel('Time (s)'); fig.savefig('figure.png')"}],
            self.figure, {},
        )
        audited = [{"id": "shared-label", "status": "pass", "evidence": "A global Time (s) label is visible."}]
        with patch("validator.solution._extract_design_requirements", return_value=extracted), patch(
            "validator.solution._audit_design", return_value=audited
        ):
            self.assertEqual(solution.judge_chart(run), [])

    def test_wrong_data_and_wrong_chart_can_coexist_without_old_combined_checker(self):
        data = [Error(family=ErrorFamily.WRONG_DATA, evidence="The plotted magnitude values are doubled.")]
        chart = [Error(family=ErrorFamily.WRONG_CHART, evidence="The requested polar projection is absent.")]
        with patch("validator.solution.judge_data", return_value=data) as data_check, patch(
            "validator.solution.judge_chart", return_value=chart
        ), patch("validator.solution.baseline.judge_data_and_chart") as old_combined:
            errors = solution.judge_data_and_chart(self.run)
        data_check.assert_called_once_with(self.run)
        old_combined.assert_not_called()
        self.assertEqual(errors, data + chart)
        Prediction(run_id=self.run.run_id, errors=errors)


if __name__ == "__main__":
    unittest.main()
