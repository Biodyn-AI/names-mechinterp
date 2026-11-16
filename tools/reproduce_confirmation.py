"""Recompute confirmation results from complete text/JSON, entirely offline.

Requires this repository's frozen analysis and independent printed-prompt parser.
This is not native-score, hidden-state or kernel replay and loads no model.
"""

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FROZEN = REPO / "experiments/error_reference_confirmation_20261116"
PROTOCOL_COMMIT = "f1d7836c653a82c0183c8bfed59c2c1153f73e42"
SOURCE_PINS = {
    "confirmation_plan": "fed362b9ba425a569b5c4b4cd8202cf4fbb2a1989863147a5c2872d3630334a6",
    "literal_audit": "c07e02d7fe24486881a1f3d02ae2b67def9f68d82a52d17d97387708a6041e4a",
}


def sources():
    modules = []
    for name, expected in SOURCE_PINS.items():
        path = FROZEN / f"{name}.py"
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen confirmation analysis changed: {name}")
        spec = importlib.util.spec_from_file_location(f"_reproduction_{name}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    return tuple(modules)


def compare_saved(expected, saved, path="result"):
    """Counts are exact; only floating statistics permit tiny libm differences."""
    if isinstance(expected, dict):
        valid = isinstance(saved, dict) and set(saved) == set(expected)
        if valid:
            for key in expected:
                compare_saved(expected[key], saved[key], f"{path}.{key}")
    elif isinstance(expected, list):
        valid = isinstance(saved, list) and len(saved) == len(expected)
        if valid:
            for index, value in enumerate(expected):
                compare_saved(value, saved[index], f"{path}[{index}]")
    elif type(expected) is float:
        valid = (
            type(saved) in (int, float)
            and math.isfinite(expected)
            and math.isfinite(saved)
            and math.isclose(expected, saved, rel_tol=0.0, abs_tol=1e-12)
        )
    else:
        valid = type(saved) is type(expected) and saved == expected
    if not valid:
        raise ValueError(f"Saved result differs from recomputation: {path}")


def reproduce(data_root):
    root, fingerprints = Path(data_root), {}

    def load(name):
        content = (root / name).read_bytes()
        fingerprints[name] = hashlib.sha256(content).hexdigest()
        return json.loads(content)

    plan, literal = sources()
    rows, worlds = load("ROWS.json"), load("WORLDS.json")
    behavior, independent, predictions, verified = {}, {}, {}, []
    for model in ("qwen", "phi"):
        predictions[model] = load(f"{model}/GENERATIONS.json")
        behavior[model] = plan.score_behavior(rows, predictions[model], worlds)
        independent[model] = literal.reconstruct_behavior(
            rows, predictions[model], REPO
        )
        literal.compare_behavior(independent[model], behavior[model])
    cases, generations = load("repair/CASES.json"), load("repair/GENERATIONS.json")
    if cases != plan.select_cases(rows, predictions["qwen"], worlds):
        raise ValueError("Saved repair cases differ from complete-cohort eligibility")
    repair = plan.assess_repair(cases, rows, predictions["qwen"], generations)
    independent["repair"] = literal.reconstruct_repair(
        cases, rows, predictions["qwen"], generations, REPO
    )
    literal.compare_repair(independent["repair"], repair)
    for stage, report in (*behavior.items(), ("repair", repair)):
        name = f"{stage}/RESULT.json"
        if (root / name).is_file():
            compare_saved(report, load(name), stage)
            verified.append(stage)
    return {
        "schema": "names-mechinterp-confirmation-cpu-reproduction-v1",
        "protocol_commit": PROTOCOL_COMMIT,
        "analysis_source_sha256": dict(SOURCE_PINS),
        "input_sha256": fingerprints,
        "independent_worlds": 64,
        "questions_per_model": 6144,
        "same_worlds_for_both_models": True,
        "behavior": behavior,
        "repair": repair,
        "independent_literal_reconstructions": independent,
        "saved_results_verified": verified,
        "native_logit_replay_performed": False,
        "new_model_forward_count": 0,
        "scope": "CPU-only recomputation of complete decoded text outcomes and frozen statistics. Both models share the same 64 worlds. No native-score argmax, hidden-state, kernel, custody or cloud-execution audit is performed; those require the separate audit records and full native artifacts.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    report = reproduce(args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "worlds": 64,
                "eligible_repair_worlds": report["repair"]["eligible_worlds"],
                "native_logit_replay_performed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
