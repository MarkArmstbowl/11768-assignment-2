"""Read-only artifact audit, optionally followed by selected live execution checks.

Default: inspect the 110 released artifacts locally; no model requests.
With --live: check ONLY execution for explicitly selected runs. Empty errors
here do not mean the entire visualization task is acceptable. Print results;
do not overwrite baseline or the official full-seed prediction files.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

from validator import solution
from validator.prediction import Error, Prediction
from validator.runner import Run, runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--run-ids", nargs="+")
    args = parser.parse_args()
    if args.live and not args.run_ids:
        parser.error("--live requires an explicit --run-ids subset")
    root = Path(__file__).resolve().parents[1]
    all_runs = runs(root / "artifacts", root)
    absent_or_invalid = []
    decoded = []
    for run in all_runs:
        inspection = solution._inspect_figure(run)
        (absent_or_invalid if isinstance(inspection, Error) else decoded).append(run.run_id)
    result = {
        "scope": "execution_gate_only_not_full_validator_predictions",
        "offline_artifact_check": {
            "runs": len(all_runs),
            "missing_or_invalid_run_ids": absent_or_invalid,
            "decoded_png_count": len(decoded),
        },
        "live_execution_checks": [],
    }
    by_id = {run.run_id: run for run in all_runs}
    if args.live:
        unknown = sorted(set(args.run_ids) - set(by_id))
        if unknown:
            parser.error(f"unknown run IDs: {unknown}")
        original_complete = solution.complete
        for run_id in args.run_ids:
            run = by_id[run_id]
            calls = []

            def capture(messages, **generation):
                started = time.monotonic()
                response = original_complete(messages, **generation)
                choice = response["choices"][0]
                calls.append({
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "finish_reason": choice.get("finish_reason"),
                    "content": choice["message"]["content"],
                    "usage": response.get("usage"),
                })
                return response

            print(f"Checking execution gate for {run_id}...", file=sys.stderr, flush=True)
            with patch.object(solution, "complete", side_effect=capture):
                errors = solution.judge_execution(run)
            prediction = Prediction(run_id=run_id, errors=errors)
            result["live_execution_checks"].append({
                **prediction.model_dump(mode="json"),
                "model_call_count": len(calls),
                "judge_responses": calls,
            })
            print(
                f"{run_id}: execution_failure={bool(errors)}, model_calls={len(calls)}",
                file=sys.stderr, flush=True,
            )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
