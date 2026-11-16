"""Distinguish known lexical and binding-line invariance signatures."""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "error_reference_baselines_20261115"))
sys.path.insert(0, str(ROOT / "error_reference_choice_baselines_20261115"))
import retrieval_baselines as original
from choice_baselines import paired_interval


def classify(answers):
    """Order: original line0/line1, swapped line0/line1 (name order reverses)."""
    if len(answers) != 4:
        raise ValueError("Require the complete two-name by two-location table")
    first, second, swapped_first, swapped_second = answers
    if len(set(answers)) == 1:
        return "all_same"
    if first == swapped_second and second == swapped_first:
        return "name_aligned"
    if first == swapped_first and second == swapped_second:
        return "line_aligned"
    return "mixed"


def analyze(rows, worlds, predictions):
    protocol = original.load_protocol(ROOT / "error_reference_study_20261114")
    original.validate_cases(rows, predictions, worlds, protocol)
    lookup = {
        (
            row["world_id"],
            row["focus_condition"],
            row["field"],
            row["name_swap"],
            row["query_alias_index"],
        ): row
        for row in rows
        if row["probe_kind"] == "fact"
    }
    predicted = {prediction["id"]: prediction for prediction in predictions}
    reports = {}
    for focus in ("none", "distractor"):
        tables, per_world = [], []
        for world in sorted(worlds, key=lambda item: item["world_id"]):
            counts = Counter()
            for field in ("color", "shape", "home"):
                group = [
                    lookup[(world["world_id"], focus, field, swap, alias)]
                    for swap in (0, 1)
                    for alias in (0, 1)
                ]
                names = [row["aliases"][row["query_alias_index"]] for row in group]
                if (
                    names[0] != names[3]
                    or names[1] != names[2]
                    or names[0] == names[1]
                    or len({row["expected_answer"] for row in group}) != 1
                ):
                    raise ValueError("Same-name and same-meaning crossing changed")
                answers = []
                for row in group:
                    prediction = predicted[row["id"]]
                    finished = original._finished(prediction)
                    answers.append(
                        ("finished:" if finished else "truncated:")
                        + original.normalize(prediction["generated_text"])
                    )
                category = classify(answers)
                counts[category] += 1
                tables.append(
                    {
                        "world_id": world["world_id"],
                        "field": field,
                        "row_ids": [row["id"] for row in group],
                        "query_names": names,
                        "answers": answers,
                        "category": category,
                    }
                )
            per_world.append({"world_id": world["world_id"], "counts": dict(counts)})
        totals = Counter(table["category"] for table in tables)
        reports[focus] = {
            "table_count": len(tables),
            "world_count": len(per_world),
            "counts": dict(totals),
            "name_minus_line_fraction": paired_interval(
                [
                    (
                        world["counts"].get("name_aligned", 0)
                        - world["counts"].get("line_aligned", 0)
                    )
                    / 3
                    for world in per_world
                ]
            ),
            "per_world": per_world,
            "tables": tables,
        }
    return {
        "schema": "name-position-invariance-20261115-v1",
        "reports": reports,
        "confirmation_outcomes_read": False,
        "interpretation": "Exploratory behavioral signature of known retrieval routes, not causal identification or a novel mechanism.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--model", choices=("qwen", "phi"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = [
        args.study_dir / f"TOKENS-{args.model}.json",
        args.study_dir / "WORLDS.json",
        args.study_dir / args.model / "GENERATIONS.json",
    ]
    tokens, worlds, predictions = map(original.sealed_json, inputs)
    report = analyze(
        [row for row in tokens["rows"] if row["split"] == "development"],
        [world for world in worlds if world["split"] == "development"],
        predictions,
    )
    report["model"] = args.model
    sources = [
        Path(__file__),
        Path(original.__file__),
        ROOT / "error_reference_choice_baselines_20261115/choice_baselines.py",
        Path(__file__).with_name("PLAN.json"),
    ]
    report["input_sha256"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in inputs + sources
    }
    payload = (
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    with args.output.open("xb") as output:
        output.write(payload)
    with Path(str(args.output) + ".sha256").open("x") as sidecar:
        sidecar.write(hashlib.sha256(payload).hexdigest() + "\n")
    print(
        json.dumps(
            {
                "model": args.model,
                "reports": {
                    key: {
                        name: value
                        for name, value in section.items()
                        if name not in ("tables", "per_world")
                    }
                    for key, section in report["reports"].items()
                },
            }
        )
    )


if __name__ == "__main__":
    main()
