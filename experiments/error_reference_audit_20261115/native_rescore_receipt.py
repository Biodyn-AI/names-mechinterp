"""Seal a file-bound receipt only after the complete native-output rescore."""

import argparse
import importlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path


def make_receipt(runner, old, study_root, model):
    result = runner.rescore(study_root, model)
    if (
        result.get("exact_match") is not True
        or type(result.get("native_steps")) is not int
        or result["native_steps"] < 3072
        or result.get("confirmation_calls") != 0
        or result.get("causal_calls") != 0
    ):
        raise ValueError(
            "Only a complete successful development rescore can be attested"
        )
    study = Path(study_root)
    paths = {
        "registration_sha256": study / "REGISTRATION.json",
        "custody_sha256": study / model / "CUSTODY.json",
        "result_sha256": study / model / "RESULT.json",
        "raw_index_sha256": study / model / "RAW_INDEX.json",
        "generations_sha256": study / model / "GENERATIONS.json",
        "source_manifest_sha256": study / "CODE.json",
    }
    return {
        **result,
        "model": model,
        **{name: old.verify_file(path) for name, path in paths.items()},
        "audited_utc": datetime.now(UTC).isoformat(),
        "receipt_source_sha256": old.sha256(Path(__file__)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--model", choices=("qwen", "phi"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or args.output.with_name(args.output.name + ".sha256").exists()
    ):
        raise FileExistsError("Native audit receipt output is single-use")
    sys.path.insert(0, str(args.source_root.resolve()))
    runner = importlib.import_module("error_run")
    if Path(runner.__file__).resolve().parent != args.source_root.resolve():
        raise ValueError("Native audit imported an unexpected runner")
    old, _ = runner.check(args.study_root)
    receipt = make_receipt(runner, old, args.study_root, args.model)
    old.seal_json(args.output, receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
