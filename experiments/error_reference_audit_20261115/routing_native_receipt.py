"""File-bound CPU routing replay receipt; exact negative results are retained too."""

import argparse
import importlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path


def make_receipt(runner, old, routing_root):
    result = runner.rescore(routing_root)
    if (
        result.get("exact_match") is not True
        or type(result.get("native_forward_count")) is not int
        or not 32 <= result["native_forward_count"] <= 112
        or result.get("confirmation_calls") != 0
    ):
        raise ValueError(
            "Only a complete exact in-scope routing replay can be attested"
        )
    study = Path(routing_root)
    root = study / "routing"
    paths = {
        "registration_sha256": study / "REGISTRATION.json",
        "source_manifest_sha256": study / "SOURCES.json",
        "plan_sha256": study / "PLAN.json",
        "inputs_sha256": study / "INPUTS.json",
        "selected_plan_sha256": root / "SELECTED_PLAN.json",
        "custody_sha256": root / "CUSTODY.json",
        "result_sha256": root / "RESULT.json",
        "raw_index_sha256": root / "RAW_INDEX.json",
        "natural_sha256": root / "NATURAL.pt",
        "interventions_sha256": root / "INTERVENTIONS.pt",
        "capture_summaries_sha256": root / "CAPTURE_SUMMARIES.json",
        "counts_sha256": root / "COUNTS.json",
    }
    # A scientific no-go is not an audit failure: never filter on its assessment.
    return {
        **result,
        **{name: old.verify_file(path) for name, path in paths.items()},
        "audited_utc": datetime.now(UTC).isoformat(),
        "receipt_source_sha256": old.sha256(Path(__file__)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routing-root", required=True, type=Path)
    parser.add_argument("--routing-source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise FileExistsError("Routing audit receipt output is single-use")
    import torch

    torch.set_num_threads(1)  # Replay the registered CPU FP32 delta arithmetic exactly.
    sys.path.insert(0, str(args.routing_source_root.resolve()))
    runner = importlib.import_module("routing_run")
    if Path(runner.__file__).resolve().parent != args.routing_source_root.resolve():
        raise ValueError("Routing audit imported an unexpected runner")
    old = runner.check(args.routing_root)[4]
    receipt = make_receipt(runner, old, args.routing_root)
    old.seal_json(args.output, receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
