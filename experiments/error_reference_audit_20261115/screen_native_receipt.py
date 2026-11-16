"""Produce an anchored receipt after exact saved-logit screen verification."""

import argparse
import importlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path


def make_receipt(screen, old, root):
    result = screen.rescore(root)
    if (
        result.get("exact_match") is not True
        or result.get("confirmation_calls") != 0
        or not isinstance(result.get("physical_counts"), dict)
        or result.get("complete_answer_repair_established") is not False
    ):
        raise ValueError(
            "Only an exact, complete saved discovery-screen audit is eligible"
        )
    root = Path(root)
    paths = {
        "registration_sha256": root / "REGISTRATION.json",
        "custody_sha256": root / "screen" / "CUSTODY.json",
        "result_sha256": root / "screen" / "RESULT.json",
        "raw_index_sha256": root / "screen" / "RAW_INDEX.json",
        "source_manifest_sha256": root / "SOURCES.json",
        "plan_sha256": root / "PLAN.json",
        "inputs_sha256": root / "INPUTS.json",
        "prefixes_sha256": root / "screen" / "PREFIXES.pt",
    }
    return {
        **result,
        **{name: old.verify_file(path) for name, path in paths.items()},
        "audited_utc": datetime.now(UTC).isoformat(),
        "receipt_source_sha256": old.sha256(Path(__file__)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-root", required=True, type=Path)
    parser.add_argument("--screen-source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if (
        args.output.exists()
        or args.output.with_name(args.output.name + ".sha256").exists()
    ):
        raise FileExistsError("Anchored screen audit output is single-use")
    sys.path.insert(0, str(args.screen_source_root.resolve()))
    screen = importlib.import_module("causal_run")
    if Path(screen.__file__).resolve().parent != args.screen_source_root.resolve():
        raise ValueError("Unexpected screen source module")
    _, _, _, old, *_ = screen.check(args.screen_root)
    receipt = make_receipt(screen, old, args.screen_root)
    old.seal_json(args.output, receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
