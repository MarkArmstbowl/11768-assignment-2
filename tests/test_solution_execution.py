"""Offline regression checks for the student's execution-failure gate.

All model responses are mocked. Importing the validator also avoids loading the
local .env, so these checks require neither credentials nor a running endpoint.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from openai import BadRequestError
from PIL import Image, ImageDraw

with patch("dotenv.load_dotenv"):
    from validator import solution
    from validator.prediction import Error, ErrorFamily, Prediction
    from validator.runner import Run


def response(content, finish_reason="stop"):
    return {"choices": [{"message": {"content": content}, "finish_reason": finish_reason}]}


def bad_request(message):
    return BadRequestError(
        message,
        response=httpx.Response(
            400, request=httpx.Request("POST", "https://offline.invalid/v1/chat/completions")
        ),
        body={"error": {"message": message}},
    )


def offline_tokenize(text):
    """Use a conservative character count to exercise fallback without a server."""
    return {"tokens": list(range(len(text))), "max_model_len": 32768}


def final_tool_pair(command="python final_plot.py"):
    return [
        {
            "role": "assistant",
            "tool_calls": [{
                "id": "final-pair-call",
                "type": "function",
                "function": {"name": "bash", "arguments": json.dumps({"command": command})},
            }],
        },
        {
            "role": "tool",
            "tool_call_id": "final-pair-call",
            "content": "The complete final plotting command produced figure.png.",
        },
    ]


def execution_content(failed=False, evidence="Axes and plotted points are visible."):
    return json.dumps(
        {
            "execution_failure": failed,
            "artifact_status": "no_figure_content" if failed else "rendered_figure",
            "evidence": evidence,
        }
    )


class ExecutionResponseTests(unittest.TestCase):
    def test_accepts_both_consistent_decisions(self):
        for failed in (False, True):
            with self.subTest(execution_failure=failed):
                solution._parse_execution_response(execution_content(failed))

    def test_accepts_one_entire_json_code_fence(self):
        solution._parse_execution_response(
            "```json\n" + execution_content(False) + "\n```"
        )

    def test_rejects_invalid_types_keys_and_states(self):
        valid = json.loads(execution_content(False))
        invalid = {
            "string_boolean": {**valid, "execution_failure": "false"},
            "numeric_boolean": {**valid, "execution_failure": 0},
            "empty_evidence": {**valid, "evidence": ""},
            "whitespace_evidence": {**valid, "evidence": " \n\t "},
            "non_string_evidence": {**valid, "evidence": ["visible axes"]},
            "unknown_key": {**valid, "wrong_data": False},
            "missing_key": {key: value for key, value in valid.items() if key != "evidence"},
            "unknown_status": {**valid, "artifact_status": "unknown"},
            "failure_with_rendered_figure": {**valid, "execution_failure": True},
            "success_without_content": {**valid, "artifact_status": "no_figure_content"},
            "array_instead_of_object": [valid],
        }
        for name, document in invalid.items():
            with self.subTest(case=name), self.assertRaises(ValueError):
                solution._parse_execution_response(json.dumps(document))

    def test_rejects_duplicate_json_keys(self):
        content = (
            '{"execution_failure":true,"execution_failure":false,'
            '"artifact_status":"rendered_figure","evidence":"Visible points."}'
        )
        with self.assertRaises(ValueError):
            solution._parse_execution_response(content)

    def test_rejects_prose_or_multiple_objects_around_json(self):
        content = execution_content(False)
        for malformed in (
            "NO: a figure exists.",
            "Here is my answer:\n" + content,
            content + "\nAdditional explanation.",
            content + "\n" + content,
            "```json\n" + content + "\n```\nExtra text.",
        ):
            with self.subTest(content=malformed), self.assertRaises(ValueError):
                solution._parse_execution_response(malformed)


class ExecutionJudgmentTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="execution-validator-test-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.figure = self.root / "figure.png"
        image = Image.new("RGB", (200, 150), "white")
        draw = ImageDraw.Draw(image)
        draw.line([(30, 20), (30, 120), (180, 120)], fill="black", width=2)
        draw.line([(40, 100), (90, 70), (150, 40)], fill="blue", width=3)
        image.save(self.figure, format="PNG")

    def make_run(self, *, figure=None, messages=None, use_default_figure=True):
        return Run(
            run_id="offline-execution-run",
            task_id="offline-execution-task",
            instructions="Plot the three requested points and save figure.png.",
            messages=messages or [],
            figure=self.figure if use_default_figure else figure,
            inputs={},
        )

    def assert_execution_failure(self, errors):
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].family, ErrorFamily.EXECUTION_FAILURE)
        self.assertTrue(errors[0].evidence.strip())
        Prediction(run_id="offline-execution-run", errors=errors)

    def test_valid_png_with_rendered_figure_response_is_acceptable(self):
        with patch(
            "validator.solution.complete", return_value=response(execution_content(False))
        ) as complete:
            self.assertEqual(solution.judge_execution(self.make_run()), [])
        complete.assert_called_once()

    def test_model_can_report_no_figure_content_in_a_decodable_png(self):
        Image.new("RGB", (200, 150), "white").save(self.figure, format="PNG")
        evidence = (
            "The final tool command only saved an empty placeholder image; "
            "the final trajectory contains no plotting action or produced chart."
        )
        messages = [
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": "placeholder-call",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": json.dumps({
                            "command": "python -c \"from PIL import Image; "
                            "Image.new('RGB', (200, 150), 'white').save('figure.png')\""
                        }),
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "placeholder-call",
                "content": "Saved an empty placeholder only; no plot was generated.",
            },
        ]
        with patch(
            "validator.solution.complete",
            return_value=response(execution_content(True, evidence)),
        ) as complete:
            errors = solution.judge_execution(self.make_run(messages=messages))
        self.assert_execution_failure(errors)
        self.assertEqual(errors[0].evidence, evidence)
        self.assertEqual(complete.call_count, 2)
        self.assertIn("provisional execution_failure=true", str(complete.call_args.args[0]))

    def test_blank_pixels_alone_do_not_determine_execution_failure(self):
        Image.new("RGB", (200, 150), "white").save(self.figure, format="PNG")
        with patch(
            "validator.solution.complete", return_value=response(execution_content(False))
        ) as complete:
            self.assertEqual(solution.judge_execution(self.make_run()), [])
        complete.assert_called_once()

    def test_positive_after_bad_json_exhausts_budget_without_confirmation(self):
        with patch(
            "validator.solution.complete",
            side_effect=[response("invalid JSON"), response(execution_content(True))],
        ) as complete:
            with self.assertRaises(RuntimeError):
                solution.judge_execution(self.make_run())
        self.assertEqual(complete.call_count, 2)

    def test_provisional_positive_then_negative_continues_other_checks(self):
        later_errors = [
            Error(family=ErrorFamily.WRONG_CHART, evidence="Requested polar projection is missing.")
        ]
        with patch(
            "validator.solution.complete",
            side_effect=[response(execution_content(True)), response(execution_content(False))],
        ) as complete, patch(
            "validator.solution.judge_data_and_chart", return_value=later_errors
        ) as data_and_chart, patch(
            "validator.solution.judge_readability", return_value=[]
        ) as readability:
            run = self.make_run()
            self.assertEqual(solution.validate(run), later_errors)
        self.assertEqual(complete.call_count, 2)
        data_and_chart.assert_called_once_with(run)
        readability.assert_called_once_with(run)

    def test_absent_figure_does_not_call_the_model(self):
        with patch("validator.solution.complete") as complete:
            errors = solution.judge_execution(self.make_run(use_default_figure=False))
        self.assert_execution_failure(errors)
        complete.assert_not_called()

    def test_missing_figure_path_does_not_call_the_model(self):
        run = self.make_run(figure=self.root / "missing.png", use_default_figure=False)
        with patch("validator.solution.complete") as complete:
            errors = solution.judge_execution(run)
        self.assert_execution_failure(errors)
        complete.assert_not_called()

    def test_corrupt_png_does_not_call_the_model(self):
        self.figure.write_bytes(self.figure.read_bytes()[:45])
        with patch("validator.solution.complete") as complete:
            errors = solution.judge_execution(self.make_run())
        self.assert_execution_failure(errors)
        complete.assert_not_called()

    def test_non_png_with_png_filename_does_not_call_the_model(self):
        Image.new("RGB", (100, 100), "white").save(self.figure, format="JPEG")
        with patch("validator.solution.complete") as complete:
            errors = solution.judge_execution(self.make_run())
        self.assert_execution_failure(errors)
        complete.assert_not_called()

    def test_read_permission_error_is_not_an_agent_failure(self):
        error = PermissionError("Cannot read the local artifact")
        with patch.object(Path, "read_bytes", side_effect=error), patch(
            "validator.solution.complete"
        ) as complete:
            with self.assertRaises(PermissionError) as raised:
                solution.judge_execution(self.make_run())
        self.assertIs(raised.exception, error)
        complete.assert_not_called()

    def test_bad_response_then_good_response_is_retried_once(self):
        with patch(
            "validator.solution.complete",
            side_effect=[response("NO, it looks fine"), response(execution_content(False))],
        ) as complete:
            self.assertEqual(solution.judge_execution(self.make_run()), [])
        self.assertEqual(complete.call_count, 2)

    def test_inconsistent_status_is_retried_instead_of_becoming_a_label(self):
        inconsistent = json.loads(execution_content(False))
        inconsistent["execution_failure"] = True
        with patch(
            "validator.solution.complete",
            side_effect=[response(json.dumps(inconsistent)), response(execution_content(False))],
        ) as complete:
            self.assertEqual(solution.judge_execution(self.make_run()), [])
        self.assertEqual(complete.call_count, 2)

    def test_two_bad_responses_raise_instead_of_guessing_an_agent_label(self):
        with patch(
            "validator.solution.complete", return_value=response("not valid JSON")
        ) as complete:
            with self.assertRaises(RuntimeError):
                solution.judge_execution(self.make_run())
        self.assertEqual(complete.call_count, 2)

    def test_truncated_valid_json_is_retried_instead_of_accepted(self):
        with patch(
            "validator.solution.complete",
            side_effect=[
                response(execution_content(False), finish_reason="length"),
                response(execution_content(False)),
            ],
        ) as complete:
            self.assertEqual(solution.judge_execution(self.make_run()), [])
        self.assertEqual(complete.call_count, 2)

    def test_two_truncated_responses_raise_instead_of_returning_a_label(self):
        with patch(
            "validator.solution.complete",
            return_value=response(execution_content(False), finish_reason="length"),
        ) as complete:
            with self.assertRaises(RuntimeError):
                solution.judge_execution(self.make_run())
        self.assertEqual(complete.call_count, 2)

    def test_transport_error_propagates_instead_of_becoming_an_agent_failure(self):
        error = RuntimeError("Model endpoint unavailable")
        with patch("validator.solution.complete", side_effect=error) as complete:
            with self.assertRaises(RuntimeError) as raised:
                solution.judge_execution(self.make_run())
        self.assertIs(raised.exception, error)
        complete.assert_called_once()

    def test_old_traceback_does_not_override_a_successful_final_figure(self):
        messages = [
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": "failed-call",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": json.dumps({"command": "python original_plot.py"}),
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "failed-call",
                "content": "Traceback (most recent call last): ValueError: wrong array shape",
            },
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": "repaired-call",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": json.dumps({"command": "python repaired_plot.py"}),
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "repaired-call",
                "content": "Repair completed; figure.png was saved.",
            },
        ]
        with patch(
            "validator.solution.complete", return_value=response(execution_content(False))
        ) as complete:
            self.assertEqual(solution.judge_execution(self.make_run(messages=messages)), [])
        shown_messages = str(complete.call_args.args[0])
        self.assertIn("Traceback", shown_messages)
        self.assertIn("original_plot.py", shown_messages)
        self.assertIn("repaired_plot.py", shown_messages)
        self.assertIn("Repair completed", shown_messages)
        self.assertIn("image_url", shown_messages)

    def test_context_fallback_preserves_complete_final_tool_messages(self):
        old_messages = [
            {"role": "tool", "content": "old-evidence-" + "x" * 10000}
            for _ in range(13)
        ]
        final_messages = [
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": "final-call",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": json.dumps({"command": "python complete_final_plot.py"}),
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "final-call",
                "content": "Full final result: figure.png generated; checks complete.",
            },
            {"role": "assistant", "content": "The final figure is ready."},
        ]
        run = self.make_run(messages=old_messages + final_messages)
        with patch(
            "validator.solution.complete",
            side_effect=[
                bad_request("maximum context length exceeded"),
                response(execution_content(False)),
            ],
        ) as complete, patch(
            "validator.solution.tokenize", side_effect=offline_tokenize
        ) as tokenize, self.assertWarns(UserWarning):
            self.assertEqual(solution.judge_execution(run), [])
        self.assertEqual(complete.call_count, 2)
        self.assertGreater(tokenize.call_count, 0)
        content = complete.call_args.args[0][-1]["content"]
        prompt = next(part["text"] for part in content if part["type"] == "text")
        serialized = prompt.split("Trajectory (including actual tool calls and results):\n", 1)[1]
        retained, _ = json.JSONDecoder().raw_decode(serialized)
        self.assertGreaterEqual(len(retained), len(final_messages))
        self.assertEqual(retained[-len(final_messages):], final_messages)
        self.assertEqual(retained, run.messages[-len(retained):])
        self.assertIn("Retained messages are complete", prompt)
        self.assertIn("NOT an agent error", prompt)

    def test_compaction_keeps_final_pair_when_three_messages_do_not_fit(self):
        pair = final_tool_pair()
        run = self.make_run(messages=[{"role": "tool", "content": "x" * 20000}] + pair)
        with patch("validator.solution.tokenize", side_effect=offline_tokenize):
            trajectory, shown_inputs = solution._compact_execution_evidence(run, "", "Task.")
        self.assertEqual(json.loads(trajectory), pair)
        self.assertEqual(shown_inputs, "")

    def test_compaction_rejects_an_orphan_result_when_the_pair_does_not_fit(self):
        run = self.make_run(messages=final_tool_pair(command="x" * 20000))
        with patch(
            "validator.solution.complete",
            side_effect=bad_request("maximum context length exceeded"),
        ) as complete, patch(
            "validator.solution.tokenize", side_effect=offline_tokenize
        ), self.assertWarns(UserWarning), self.assertRaises(RuntimeError):
            solution.judge_execution(run)
        complete.assert_called_once()

    def test_compaction_preserves_final_tool_pair_with_a_trailing_exit(self):
        for trailing_count in (1, 13):
            with self.subTest(trailing_messages=trailing_count):
                final_messages = final_tool_pair() + [
                    {"role": "assistant", "content": "exit"} for _ in range(trailing_count)
                ]
                run = self.make_run(
                    messages=[{"role": "tool", "content": "x" * 20000}] + final_messages
                )
                with patch("validator.solution.tokenize", side_effect=offline_tokenize):
                    trajectory, _ = solution._compact_execution_evidence(run, "", "Task.")
                self.assertEqual(json.loads(trajectory), final_messages)

    def test_compaction_cannot_replace_a_large_final_pair_with_only_a_trailing_exit(self):
        for trailing_count in (1, 13):
            with self.subTest(trailing_messages=trailing_count):
                messages = final_tool_pair(command="x" * 20000) + [
                    {"role": "assistant", "content": "exit"} for _ in range(trailing_count)
                ]
                with patch(
                    "validator.solution.tokenize", side_effect=offline_tokenize
                ), self.assertRaises(RuntimeError):
                    solution._compact_execution_evidence(self.make_run(messages=messages), "", "Task.")

    def test_non_context_bad_request_propagates_without_compaction(self):
        error = bad_request("Invalid request parameter: unsupported image format")
        with patch("validator.solution.complete", side_effect=error) as complete, patch(
            "validator.solution.tokenize"
        ) as tokenize:
            with self.assertRaises(BadRequestError) as raised:
                solution.judge_execution(self.make_run())
        self.assertIs(raised.exception, error)
        complete.assert_called_once()
        tokenize.assert_not_called()

    def test_invalid_tokenizer_context_is_an_environment_error(self):
        for context in (None, 0, -1, True, "32768"):
            with self.subTest(context=context), patch(
                "validator.solution.complete",
                side_effect=bad_request("maximum context length exceeded"),
            ) as complete, patch(
                "validator.solution.tokenize", return_value={"tokens": [], "max_model_len": context}
            ), self.assertWarns(UserWarning), self.assertRaises(RuntimeError):
                solution.judge_execution(self.make_run())
            complete.assert_called_once()


class ValidationGateTests(unittest.TestCase):
    def setUp(self):
        self.run = Run("gate-run", "gate-task", "Draw the requested chart.", [], None, {})

    def test_execution_failure_stops_the_remaining_judges(self):
        errors = [Error(family=ErrorFamily.EXECUTION_FAILURE, evidence="No figure was saved.")]
        with patch("validator.solution.judge_execution", return_value=errors), patch(
            "validator.solution.judge_data_and_chart"
        ) as data_and_chart, patch("validator.solution.judge_readability") as readability:
            result = solution.validate(self.run)
        self.assertEqual(result, errors)
        data_and_chart.assert_not_called()
        readability.assert_not_called()
        self.assertEqual(len(Prediction(run_id=self.run.run_id, errors=result).errors), 1)

    def test_valid_execution_preserves_all_three_later_error_families(self):
        data_errors = [
            Error(family=ErrorFamily.WRONG_DATA, evidence="The final point is 9 instead of 3."),
            Error(family=ErrorFamily.WRONG_CHART, evidence="Requested axis title is missing."),
        ]
        readability_errors = [
            Error(family=ErrorFamily.HARD_TO_READ, evidence="Adjacent tick labels overlap.")
        ]
        with patch("validator.solution.judge_execution", return_value=[]), patch(
            "validator.solution.judge_data_and_chart", return_value=data_errors
        ) as data_and_chart, patch(
            "validator.solution.judge_readability", return_value=readability_errors
        ) as readability:
            result = solution.validate(self.run)
        data_and_chart.assert_called_once_with(self.run)
        readability.assert_called_once_with(self.run)
        prediction = Prediction(run_id=self.run.run_id, errors=result)
        self.assertEqual(prediction.errors, data_errors + readability_errors)

    def test_all_clear_remains_an_empty_error_list(self):
        with patch("validator.solution.judge_execution", return_value=[]), patch(
            "validator.solution.judge_data_and_chart", return_value=[]
        ), patch("validator.solution.judge_readability", return_value=[]):
            self.assertEqual(solution.validate(self.run), [])

    def test_execution_judge_exception_does_not_call_later_judges(self):
        with patch("validator.solution.judge_execution", side_effect=RuntimeError("Unavailable")), patch(
            "validator.solution.judge_data_and_chart"
        ) as data_and_chart, patch("validator.solution.judge_readability") as readability:
            with self.assertRaises(RuntimeError):
                solution.validate(self.run)
        data_and_chart.assert_not_called()
        readability.assert_not_called()


if __name__ == "__main__":
    unittest.main()
