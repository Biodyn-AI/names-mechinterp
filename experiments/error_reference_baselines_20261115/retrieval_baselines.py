"""Small development-only retrieval mixtures; predictive comparisons, not mechanisms."""

import argparse
import hashlib
import importlib.util
import json
import string
from pathlib import Path

import numpy as np

PROTOCOL_SHA256 = "186e73abf1e541aaa1a23bc7728f3a4ccd61cde95fb319e23bc00e45bf669200"
FOLD_SEED = "reference-retrieval-baselines-20261115-v1"
SMOOTHING = 0.001  # Add this fixed pseudocount to every categorical outcome.
WEIGHT_PSEUDOCOUNT = 0.001  # Symmetric EM/MAP regularization; never selected on data.
COMPONENTS = (
    "binding",
    "focus",
    "alias_line_record_position",
    "alias_line_stable_id_ordinal",
    "first_record",
    "last_record",
    "uniform_objects",
    "malformed_policy",
)


def normalize(text):
    return " ".join(text.strip().casefold().split()).rstrip(string.punctuation).rstrip()


def load_protocol(directory):
    """Import only the hash-pinned model-free generator; no inference dependencies."""
    path = Path(directory) / "error_protocol.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("Frozen protocol source changed")
    spec = importlib.util.spec_from_file_location("baseline_frozen_protocol", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _finished(prediction):
    terminated, truncated = prediction.get("terminated"), prediction.get("truncated")
    if (
        type(terminated) is not bool
        or type(truncated) is not bool
        or terminated == truncated
        or not isinstance(prediction.get("generated_text"), str)
    ):
        raise ValueError("Invalid complete-generation termination metadata")
    return terminated and not truncated


def observed_outcome(row, prediction, world):
    """Class 4 groups errors for fitting, while preserving distinct failure flags."""
    finished = _finished(prediction)
    answer = normalize(prediction["generated_text"])
    matches = [
        i
        for i, obj in enumerate("ABCD")
        if normalize(world["objects"][obj][row["field"]]) == answer
    ]
    if len(matches) > 1:
        raise ValueError("Ambiguous field values violate the generator")
    return {
        "class": matches[0] if finished and matches else 4,
        "malformed": finished and not matches,
        "truncated": prediction["truncated"],
    }


def cue_classes(row, world):
    """Return object indices; None predicts uniform A-D, never a guessed binding."""
    alias = row["query_alias_index"]
    position = world["alias_order"].index(alias)
    focus = {"none": None, "coref_0": 0, "coref_1": 1, "distractor": 2}[
        row["focus_condition"]
    ]
    objects = [
        world["alias_targets"][alias],
        None if focus is None else world["alias_targets"][focus],
        world["object_order"][position],
        "ABCD"[position],
        world["object_order"][0],
        world["object_order"][-1],
        None,
    ]
    return [None if obj is None else "ABCD".index(obj) for obj in objects] + [4]


def smooth(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if (
        probabilities.shape[-1] != 5
        or not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or not np.allclose(probabilities.sum(axis=-1), 1)
    ):
        raise ValueError("Components must be five-class probability distributions")
    return (probabilities + SMOOTHING) / (1 + 5 * SMOOTHING)


def component_distributions(row, world):
    probabilities = np.zeros((len(COMPONENTS), 5))
    for index, choice in enumerate(cue_classes(row, world)):
        if choice is None:
            probabilities[index, :4] = 0.25
        else:
            probabilities[index, choice] = 1
    return smooth(probabilities)


def predict(components, weights):
    weights = np.asarray(weights, dtype=np.float64)
    if (
        weights.shape != (components.shape[1],)
        or not np.isfinite(weights).all()
        or (weights < 0).any()
        or not np.isclose(weights.sum(), 1)
    ):
        raise ValueError("Mixture weights must be finite, nonnegative and sum to one")
    return np.einsum("nko,k->no", components, weights)


def fit_mixture(components, observed):
    """Fit one global mixture using observed object choices, not expected labels."""
    components, observed = np.asarray(components), np.asarray(observed)
    if (
        components.ndim != 3
        or components.shape[2] != 5
        or len(observed) == 0
        or observed.shape != (len(components),)
        or observed.dtype.kind not in "iu"
        or (observed < 0).any()
        or (observed > 4).any()
        or not np.isfinite(components).all()
        or (components <= 0).any()
        or not np.allclose(components.sum(axis=2), 1)
    ):
        raise ValueError(
            "EM needs positive categorical components and observed five-class labels"
        )
    likelihood = components[np.arange(len(observed)), :, observed]
    weights = np.full(components.shape[1], 1 / components.shape[1])

    def objective(values):
        return float(
            np.log(likelihood @ values).sum()
            + WEIGHT_PSEUDOCOUNT * np.log(values).sum()
        )

    initial = previous = objective(weights)
    converged = False
    for iteration in range(1, 2001):
        responsibilities = likelihood * weights
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        weights = responsibilities.sum(axis=0) + WEIGHT_PSEUDOCOUNT
        weights /= weights.sum()
        current = objective(weights)
        if current < previous - 1e-8:
            raise ArithmeticError("EM regularized objective decreased")
        if abs(current - previous) < 1e-9:
            converged = True
            break
        previous = current
    return {
        "weights": weights.tolist(),
        "iterations": iteration,
        "converged": converged,
        "objective_initial": initial,
        "objective_final": current,
    }


def world_folds(world_ids):
    if len(world_ids) != 32 or len(set(world_ids)) != 32:
        raise ValueError("Exactly 32 distinct development worlds are required")
    ordered = sorted(
        world_ids,
        key=lambda identity: (
            hashlib.sha256(f"{FOLD_SEED}:{identity}".encode()).hexdigest(),
            identity,
        ),
    )
    return {identity: rank % 4 for rank, identity in enumerate(ordered)}


def metrics(probabilities, observed, binding):
    """Wrong-object conditioning excludes correct binding AND the malformed class."""
    probabilities, observed, binding = map(
        np.asarray, (probabilities, observed, binding)
    )
    chosen = probabilities.argmax(axis=1)  # Fixed A,B,C,D,malformed tie order.
    wrong = (observed < 4) & (observed != binding)
    conditional = probabilities[:, :4].copy()
    conditional[np.arange(len(binding)), binding] = 0
    conditional /= conditional.sum(axis=1, keepdims=True)
    return {
        "count": len(observed),
        "nll": float(-np.log(probabilities[np.arange(len(observed)), observed]).mean()),
        "choice_accuracy": float((chosen == observed).mean()),
        "model_binding_accuracy": float((observed == binding).mean()),
        "malformed_or_truncated_count": int((observed == 4).sum()),
        "wrong_object_count": int(wrong.sum()),
        "wrong_object_exact_accuracy": float((chosen[wrong] == observed[wrong]).mean())
        if wrong.any()
        else None,
        "wrong_object_conditional_accuracy": float(
            (conditional[wrong].argmax(axis=1) == observed[wrong]).mean()
        )
        if wrong.any()
        else None,
        "wrong_object_conditional_nll": float(
            -np.log(conditional[np.flatnonzero(wrong), observed[wrong]]).mean()
        )
        if wrong.any()
        else None,
    }


def validate_cases(rows, predictions, worlds, protocol):
    """Require the entire development grid and recompute every fact from its source."""
    study = protocol.build_study()
    expected_rows = {r["id"]: r for r in study["rows"] if r["split"] == "development"}
    expected_worlds = {
        w["world_id"]: w for w in study["worlds"] if w["split"] == "development"
    }
    row_lookup, prediction_lookup = (
        {r["id"]: r for r in rows},
        {p["id"]: p for p in predictions},
    )
    if (
        len(row_lookup) != len(rows)
        or len(prediction_lookup) != len(predictions)
        or set(row_lookup) != set(expected_rows)
        or set(prediction_lookup) != set(expected_rows)
        or len(worlds) != len(expected_worlds)
        or {w["world_id"]: w for w in worlds} != expected_worlds
    ):
        raise ValueError("Complete development cases differ from the frozen generator")
    for identity, expected in expected_rows.items():
        if any(
            row_lookup[identity].get(key) != value for key, value in expected.items()
        ):
            raise ValueError("Row facts or text differ from the frozen generator")
        prediction = prediction_lookup[identity]
        finished = _finished(prediction)
        correct = finished and normalize(prediction["generated_text"]) == normalize(
            expected["expected_answer"]
        )
        if (
            prediction.get("expected_answer", expected["expected_answer"])
            != expected["expected_answer"]
            or prediction.get("correct", correct) != correct
        ):
            raise ValueError("Saved expected answer or correctness changed")


def analyze(rows, predictions, worlds, protocol, model_name):
    validate_cases(rows, predictions, worlds, protocol)
    world_lookup, prediction_lookup = (
        {w["world_id"]: w for w in worlds},
        {p["id"]: p for p in predictions},
    )
    facts = sorted(
        (r for r in rows if r["probe_kind"] == "fact"), key=lambda r: r["id"]
    )
    outcomes = [
        observed_outcome(r, prediction_lookup[r["id"]], world_lookup[r["world_id"]])
        for r in facts
    ]
    observed = np.array([o["class"] for o in outcomes])
    binding = np.array(
        [
            "ABCD".index(
                world_lookup[r["world_id"]]["alias_targets"][r["query_alias_index"]]
            )
            for r in facts
        ]
    )
    components = np.array(
        [component_distributions(r, world_lookup[r["world_id"]]) for r in facts]
    )
    fold_map = world_folds(list(world_lookup))
    fold_ids = np.array([fold_map[r["world_id"]] for r in facts])
    probabilities = {
        name: components[:, i, :].copy() for i, name in enumerate(COMPONENTS)
    }
    probabilities.update(
        mixture=np.zeros((len(facts), 5)), majority_class=np.zeros((len(facts), 5))
    )
    folds = []
    for fold in range(4):
        training, evaluation = fold_ids != fold, fold_ids == fold
        fitted = fit_mixture(components[training], observed[training])
        probabilities["mixture"][evaluation] = predict(
            components[evaluation], fitted["weights"]
        )
        majority = int(np.bincount(observed[training], minlength=5).argmax())
        probabilities["majority_class"][evaluation] = smooth(np.eye(5)[majority])
        folds.append(
            {
                "fold": fold,
                "fit_worlds": sorted(w for w, f in fold_map.items() if f != fold),
                "evaluation_worlds": sorted(
                    w for w, f in fold_map.items() if f == fold
                ),
                "mixture_fit": fitted,
                "training_majority_class": majority,
            }
        )
    cv = {
        name: metrics(values, observed, binding)
        for name, values in probabilities.items()
    }
    per_world = []
    for identity in sorted(world_lookup):
        indices = np.array(
            [i for i, row in enumerate(facts) if row["world_id"] == identity]
        )
        per_world.append(
            {
                "world_id": identity,
                "fold": fold_map[identity],
                "metrics": {
                    name: metrics(values[indices], observed[indices], binding[indices])
                    for name, values in probabilities.items()
                },
                "outcomes": [
                    {
                        "id": facts[i]["id"],
                        **outcomes[i],
                        "binding_class": int(binding[i]),
                        "mixture_probabilities": probabilities["mixture"][i].tolist(),
                        "mixture_choice_class": int(
                            probabilities["mixture"][i].argmax()
                        ),
                    }
                    for i in indices
                ],
            }
        )
    return {
        "schema": "reference-retrieval-baselines-20261115-v1",
        "model": model_name,
        "protocol_sha256": PROTOCOL_SHA256,
        "components": list(COMPONENTS),
        "classes": ["A", "B", "C", "D", "malformed_or_truncated"],
        "fact_count": len(facts),
        "class_pseudocount": SMOOTHING,
        "weight_pseudocount": WEIGHT_PSEUDOCOUNT,
        "fold_seed": FOLD_SEED,
        "fold_rule": "SHA256(seed:world_id) sort, rank modulo four; whole worlds only",
        "folds": folds,
        "cv_metrics": cv,
        "per_world": per_world,
        "full_development_fit": fit_mixture(components, observed),
        "nll_improvement_over": {
            name: cv[name]["nll"] - cv["mixture"]["nll"]
            for name in ("binding", "majority_class")
        },
        "confirmation_outputs_read": False,
        "formal_novelty_inference": False,
        "limitations": [
            "Predictive baseline, not causal identification; correlated component weights are not uniquely identifiable.",
            "These eight cue components do not cover learned lexical value/name priors or all known retrieval mechanisms; outperforming this baseline is not proof of a novel mechanism.",
            "Global shared weights only; no learned per-name, per-field or per-world parameters.",
            "Conditional wrong-object metrics are descriptive subsets, not a separately fit error model.",
            "Use alongside native-logit custody/rescoring; this module does not independently audit raw model outputs.",
        ],
    }


def sealed_json(path):
    path = Path(path)
    content = path.read_bytes()
    if (
        hashlib.sha256(content).hexdigest()
        != Path(str(path) + ".sha256").read_text().strip()
    ):
        raise ValueError(f"Input checksum mismatch: {path}")
    return json.loads(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", required=True, type=Path)
    parser.add_argument("--model", required=True, choices=("qwen", "phi"))
    parser.add_argument(
        "--protocol-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "error_reference_study_20261114",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    protocol = load_protocol(args.protocol_dir)
    rows = [
        r
        for r in sealed_json(args.study_dir / f"TOKENS-{args.model}.json")["rows"]
        if r["split"] == "development"
    ]
    worlds = [
        w
        for w in sealed_json(args.study_dir / "WORLDS.json")
        if w["split"] == "development"
    ]
    predictions = sealed_json(args.study_dir / args.model / "GENERATIONS.json")
    report = analyze(rows, predictions, worlds, protocol, args.model)
    inputs = [
        args.study_dir / f"TOKENS-{args.model}.json",
        args.study_dir / "WORLDS.json",
        args.study_dir / args.model / "GENERATIONS.json",
    ]
    report["input_sha256"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs
    }
    report["analysis_source_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    with args.output.open("x") as output:
        json.dump(report, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
