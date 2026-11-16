"""Whole-world predictive controls, not a causal or novelty test."""

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "error_reference_baselines_20261115"))
import retrieval_baselines as original

ADDED_COMPONENTS = (
    "constant_A",
    "constant_B",
    "constant_C",
    "constant_D",
    "second_record",
    "third_record",
)
MIXTURES = (
    "original_global_mixture",
    "expanded_global_mixture",
    "expanded_field_mixture",
    "expanded_field_format_mixture",
)


def expanded_components(row, world):
    """Retain the old cues and add object-ID and middle-position preferences."""
    choices = list(range(4)) + [
        "ABCD".index(world["object_order"][position]) for position in (1, 2)
    ]
    return np.concatenate(
        (
            original.component_distributions(row, world),
            original.smooth(np.eye(5)[choices]),
        )
    )


def value_prior(training, evaluation, by_alias):
    """Fit lexical output counts only on training worlds, not evaluation labels.

    We condition on the four words actually available in each test registry.
    This deliberately gives the lexical baseline knowledge that listed answers
    are likely, but never tells it which of those answers is the correct one.
    """
    field_counts, alias_counts = defaultdict(Counter), defaultdict(Counter)
    for row, world, choice in training:
        word = None if choice == 4 else world["objects"]["ABCD"[choice]][row["field"]]
        field_counts[row["field"]][word] += 1
        alias = row["aliases"][row["query_alias_index"]]
        alias_counts[(alias, row["field"])][word] += 1
    probabilities = []
    for row, world in evaluation:
        alias = row["aliases"][row["query_alias_index"]]
        counts = (
            alias_counts.get((alias, row["field"]), field_counts[row["field"]])
            if by_alias
            else field_counts[row["field"]]
        )
        words = [world["objects"][obj][row["field"]] for obj in "ABCD"] + [None]
        frequencies = np.array([counts[word] + 1 for word in words], dtype=np.float64)
        probabilities.append(frequencies / frequencies.sum())
    return np.asarray(probabilities)


def paired_interval(world_differences):
    """A descriptive paired bootstrap; 32 worlds, never thousands of questions."""
    differences = np.asarray(world_differences, dtype=np.float64)
    if differences.shape != (32,) or not np.isfinite(differences).all():
        raise ValueError("Require one finite paired difference per development world")
    generator = np.random.default_rng(20261115)
    means = differences[generator.integers(0, 32, size=(10000, 32))].mean(axis=1)
    return {
        "mean": float(differences.mean()),
        "percentile_95": np.quantile(means, [0.025, 0.975]).tolist(),
    }


def grouping(row, name):
    if name == "expanded_field_format_mixture":
        return row["field"], row["template"]
    if name == "expanded_field_mixture":
        return (row["field"],)
    return ()


def analyze(rows, predictions, worlds, model):
    protocol = original.load_protocol(ROOT / "error_reference_study_20261114")
    original.validate_cases(rows, predictions, worlds, protocol)
    world_lookup = {world["world_id"]: world for world in worlds}
    prediction_lookup = {prediction["id"]: prediction for prediction in predictions}
    facts = sorted(
        (row for row in rows if row["probe_kind"] == "fact"), key=lambda row: row["id"]
    )
    cases = [(row, world_lookup[row["world_id"]]) for row in facts]
    observed = np.array(
        [
            original.observed_outcome(row, prediction_lookup[row["id"]], world)["class"]
            for row, world in cases
        ]
    )
    binding = np.array(
        [
            "ABCD".index(world["alias_targets"][row["query_alias_index"]])
            for row, world in cases
        ]
    )
    components = np.array([expanded_components(row, world) for row, world in cases])
    folds = original.world_folds(list(world_lookup))
    fold_ids = np.array([folds[row["world_id"]] for row in facts])
    probabilities = {
        name: np.zeros((len(facts), 5))
        for name in (*MIXTURES, "field_value_prior", "alias_field_value_prior")
    }
    fits = []
    for fold in range(4):
        training, evaluation = fold_ids != fold, fold_ids == fold
        for name in MIXTURES:
            selected_components = (
                components[:, :8] if name == MIXTURES[0] else components
            )
            groups = sorted({grouping(row, name) for row in facts})
            for group in groups:
                members = np.array([grouping(row, name) == group for row in facts])
                fit_indices, test_indices = members & training, members & evaluation
                if not fit_indices.any() or not test_indices.any():
                    raise ValueError(
                        "Every fitted context must occur in both training and evaluation worlds"
                    )
                fitted = original.fit_mixture(
                    selected_components[fit_indices], observed[fit_indices]
                )
                probabilities[name][test_indices] = original.predict(
                    selected_components[test_indices], fitted["weights"]
                )
                fits.append(
                    {
                        "fold": fold,
                        "model": name,
                        "group": group,
                        "training_questions": int(fit_indices.sum()),
                        "evaluation_questions": int(test_indices.sum()),
                        **fitted,
                    }
                )
        training_cases = [
            (row, world, int(observed[index]))
            for index, (row, world) in enumerate(cases)
            if training[index]
        ]
        evaluation_cases = [
            case for index, case in enumerate(cases) if evaluation[index]
        ]
        for name, by_alias in (
            ("field_value_prior", False),
            ("alias_field_value_prior", True),
        ):
            probabilities[name][evaluation] = value_prior(
                training_cases, evaluation_cases, by_alias
            )
    metrics = {
        name: original.metrics(values, observed, binding)
        for name, values in probabilities.items()
    }
    per_world = []
    for identity in sorted(world_lookup):
        indices = np.array([row["world_id"] == identity for row in facts])
        per_world.append(
            {
                "world_id": identity,
                "fold": folds[identity],
                "metrics": {
                    name: original.metrics(
                        values[indices], observed[indices], binding[indices]
                    )
                    for name, values in probabilities.items()
                },
            }
        )
    reference = "original_global_mixture"
    comparisons = {
        name: paired_interval(
            [
                world["metrics"][reference]["nll"] - world["metrics"][name]["nll"]
                for world in per_world
            ]
        )
        for name in probabilities
        if name != reference
    }
    return {
        "schema": "expanded-choice-baselines-20261115-v1",
        "research_model": model,
        "classes": ["A", "B", "C", "D", "malformed_or_truncated"],
        "components": list(original.COMPONENTS) + list(ADDED_COMPONENTS),
        "metrics": metrics,
        "per_world": per_world,
        "fits": fits,
        "nll_improvement_over_original": comparisons,
        "folds": folds,
        "confirmation_outcomes_read": False,
        "interpretation": "Exploratory predictive comparison on previously inspected development outcomes. Correlated cue weights are not identified causal pathways; no model selection or novel-mechanism conclusion.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", required=True, type=Path)
    parser.add_argument("--model", required=True, choices=("qwen", "phi"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    inputs = [
        args.study_dir / f"TOKENS-{args.model}.json",
        args.study_dir / "WORLDS.json",
        args.study_dir / args.model / "GENERATIONS.json",
    ]
    tokens, worlds, predictions = map(original.sealed_json, inputs)
    rows = [row for row in tokens["rows"] if row["split"] == "development"]
    worlds = [world for world in worlds if world["split"] == "development"]
    report = analyze(rows, predictions, worlds, args.model)
    source_paths = [
        Path(__file__),
        Path(original.__file__),
        Path(__file__).with_name("PLAN.json"),
    ]
    report["input_sha256"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in inputs + source_paths
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
                "metrics": report["metrics"],
                "nll_improvement_over_original": report[
                    "nll_improvement_over_original"
                ],
                "unconverged_fits": sum(
                    not fitted["converged"] for fitted in report["fits"]
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
