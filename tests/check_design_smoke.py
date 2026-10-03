"""Selected live DESIGN checks, not full validator predictions or MCC evaluation.

Requires explicit run IDs and --live; leaves all supplied artifacts unchanged.
Prints diagnostic JSON, never overwrites official prediction files. Call logs
contain judge responses/usage, not credentials or base64 request image data.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

from validator import solution
from validator.prediction import Prediction
from validator.runner import runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--run-ids", nargs="+")
    args = parser.parse_args()
    if not args.live or not args.run_ids:
        parser.error("requires --live and an explicit --run-ids subset")
    root = Path(__file__).resolve().parents[1]
    by_id = {run.run_id: run for run in runs(root / "artifacts", root)}
    unknown = sorted(set(args.run_ids) - set(by_id))
    if unknown:
        parser.error(f"unknown run IDs: {unknown}")
    result = {
        "scope": "design_gate_only_not_full_validator_predictions",
        "solution_sha256": hashlib.sha256(Path(solution.__file__).read_bytes()).hexdigest(),
        "live_design_checks": [],
    }
    original_complete = solution.complete
    for run_id in args.run_ids:
        calls = []

        def capture(messages, **generation):
            stage = {
                solution.DESIGN_REQUIREMENTS_POLICY: "requirements",
                solution.DESIGN_OBSERVATION_POLICY: "image_only_observation",
                solution.DESIGN_AUDIT_POLICY: "audit",
            }.get(messages[0]["content"], "unknown")
            started = time.monotonic()
            response = original_complete(messages, **generation)
            choice = response["choices"][0]
            calls.append({
                "stage": stage,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "finish_reason": choice.get("finish_reason"),
                "content": choice["message"]["content"],
                "usage": response.get("usage"),
            })
            print(
                f"{run_id}: completed {stage} call {len(calls)}",
                file=sys.stderr, flush=True,
            )
            return response

        print(f"Checking design gate for {run_id}...", file=sys.stderr, flush=True)
        record = {"run_id": run_id}
        try:
            with patch.object(solution, "complete", side_effect=capture):
                errors = solution.judge_chart(by_id[run_id])
            record.update(Prediction(run_id=run_id, errors=errors).model_dump(mode="json"))
            print(
                f"{run_id}: wrong_chart={bool(errors)}, model_calls={len(calls)}",
                file=sys.stderr, flush=True,
            )
        except Exception as error:
            # Retain observed responses even when one case cannot be judged.
            # Do not invent a negative label for a validator-side failure.
            record["validator_error"] = f"{type(error).__name__}: {error}"
            print(f"{run_id}: {record['validator_error']}", file=sys.stderr, flush=True)
        record.update(model_call_count=len(calls), judge_responses=calls)
        result["live_design_checks"].append(record)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if any("validator_error" in item for item in result["live_design_checks"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
