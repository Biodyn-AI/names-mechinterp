"""Recompute paper-facing counts from released text/JSON, entirely offline.

This entry point does not import a model, Torch, a tokenizer or network client.
Historical native/tensor receipts remain evidence, not newly repeated audits.
"""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from . import _behavior, _bootstrap, _decomposition, _literal_audit, _validation


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def finished(item):
    if type(item.get("terminated")) is not bool or type(item.get("truncated")) is not bool or item["terminated"] == item["truncated"]:
        raise ValueError("explicit consistent completion metadata is required")
    return item["terminated"] and not item["truncated"]


def validate_literal_rows(rows):
    parsed = {}
    for row in rows:
        if row["id"] in parsed:
            raise ValueError("duplicate question identity")
        value = _literal_audit.literal_answer(row["text"])
        if value[0] != row["expected_answer"] or row["probe_kind"] == "fact" and value[2] != row["field"]:
            raise ValueError("saved truth differs from the printed registry")
        parsed[row["id"]] = value
    return parsed


def name_position(rows, predictions):
    """Recompute the known lexical-versus-line invariance signature, not causality."""
    lookup = {(r["world_id"], r["focus_condition"], r["field"], r["name_swap"], r["query_alias_index"]): r for r in rows if r["probe_kind"] == "fact"}
    predicted = {p["id"]: p for p in predictions}
    reports = {}
    for focus in ("none", "distractor"):
        counts, per_world = Counter(), {}
        for world in sorted({r["world_id"] for r in rows}):
            local = Counter()
            for field in ("color", "shape", "home"):
                group = [lookup[(world, focus, field, swap, alias)] for swap in (0, 1) for alias in (0, 1)]
                names = [r["aliases"][r["query_alias_index"]] for r in group]
                if names[0] != names[3] or names[1] != names[2] or names[0] == names[1]:
                    raise ValueError("the consistent name-swap crossing changed")
                answers = [("finished:" if finished(predicted[r["id"]]) else "truncated:") + _literal_audit._clean(predicted[r["id"]]["generated_text"]) for r in group]
                a, b, c, d = answers
                category = "all_same" if len(set(answers)) == 1 else "name_aligned" if a == d and b == c else "line_aligned" if a == c and b == d else "mixed"
                counts[category] += 1
                local[category] += 1
            per_world[world] = dict(local)
        reports[focus] = {"table_count": sum(counts.values()), "world_count": len(per_world), "counts": dict(counts), "per_world": per_world}
    return {"reports": reports, "scope": "Descriptive known retrieval signatures, not causal identification or novelty."}


def check_intervention_truth(planned, rows, old_predictions, *, counterfactual=False):
    """Re-derive literal E and old-correct status, never trust saved success flags."""
    parsed = validate_literal_rows(rows)
    old = {p["id"]: p for p in old_predictions}
    cases = {c["case_id"]: c for c in planned["cases"]}
    for job in planned["generation_jobs"]:
        expected, obj, field, facts = parsed[job["row_id"]]
        task = job["task_answer"] if counterfactual else job["expected_answer"]
        if task != expected or field != job["field"]:
            raise ValueError("intervention truth differs from the printed registry")
        protected = finished(old[job["row_id"]]) and _literal_audit._clean(old[job["row_id"]]["generated_text"]) == _literal_audit._clean(expected)
        if not counterfactual and job["baseline_correct"] != protected:
            raise ValueError("decomposition baseline correctness changed")
        if not counterfactual:
            continue
        condition = job["condition"]
        if condition == "selfRfield":
            target = old[job["row_id"]]["generated_text"]
        elif condition == "cyclicDcolor":
            target = facts["BCDA"["ABCD".index(obj)]][field]
        elif condition == "reverseRscreen":
            case = cases[job["case_id"]]
            wrong = old[case["screen_row_id"]]
            matches = [o for o in "ABCD" if _literal_audit._clean(facts[o][case["screen_field"]]) == _literal_audit._clean(wrong["generated_text"])]
            if len(matches) != 1 or matches[0] == obj:
                raise ValueError("the fixed old wrong-object target is ambiguous")
            target = facts[matches[0]][field]
        else:
            target = expected
        if _literal_audit._clean(job["predicted_answer"]) != _literal_audit._clean(target):
            raise ValueError("a counterfactual steering target changed after observation")


def reproduce(data_root, original_root=None):
    data = Path(data_root).resolve()
    original = Path(original_root).resolve() if original_root else data.parent / "original"
    rows, worlds = load(data / "prompts.json"), load(data / "worlds.json")
    validate_literal_rows(rows)
    if len(rows) != 3072 or len(worlds) != 32 or {r["split"] for r in rows} != {"development"}:
        raise ValueError("exactly the released 32-world development cohort is required")
    digest = hashlib.sha256((data / "prompts.json").read_bytes()).hexdigest()
    behavior, literal, names, predictions = {}, {}, {}, {}
    for model in ("qwen", "phi"):
        predictions[model] = load(data / f"{model}_generations.json")
        behavior[model] = _behavior.score(rows, predictions[model], worlds)
        literal[model] = _literal_audit.audit(data / "prompts.json", predictions[model], digest)
        names[model] = name_position(rows, predictions[model])
        if behavior[model]["fact_errors"]["wrong_object"] != literal[model]["counts"]["wrong_object"] or behavior[model]["coreference"]["discordant_wrong_object"] != literal[model]["eligible_wrong_object_coref_pairs"]:
            raise ValueError("independent printed-label and registered counts disagree")
    decomposition_plan = load(data / "decomposition" / "PLAN.json")
    decomposition_inputs = load(data / "decomposition" / "INPUTS.json")
    check_intervention_truth(decomposition_plan, decomposition_inputs["rows"], decomposition_inputs["predictions"])
    decomposition = _decomposition.grade(decomposition_plan, load(data / "decomposition" / "stage" / "GENERATIONS.json"))
    validation_root = original / "cloud_reference_validation_metadata_20261115" / "validation"
    validation = {"recomputed": False, "reason": "validation metadata not supplied"}
    if (validation_root / "SOURCES_SELECTED.json").is_file():
        selected = load(validation_root / "SOURCES_SELECTED.json")
        selected_rows = {j["row_id"] for j in selected["generation_jobs"]}
        # Reverse targets also refer to the case's old recipient screen question.
        selected_rows.update(c["screen_row_id"] for c in selected["cases"])
        check_intervention_truth(selected, [r for r in rows if r["id"] in selected_rows], predictions["qwen"], counterfactual=True)
        validation = _validation.assess(selected, load(validation_root / "GENERATIONS.json"))
    return {"schema": "names-mechinterp-cpu-reproduction-v1", "independent_worlds": 32, "questions_per_model": 3072, "repeated_questions_per_world": 96,
            "behavior": behavior, "independent_literal_audits": literal, "name_position_signatures": names, "retrospective_descriptive_world_intervals": _bootstrap.intervals(behavior),
            "validation": validation, "decomposition": decomposition, "native_logit_replay_performed": False, "new_model_forward_count": 0,
            "scope": "Offline recomputation from released complete text outcomes. Repeated prompts are not independent samples. Native-logit and hidden-state integrity are separate optional audits; no new mechanism or confirmation is claimed."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[2] / "data" / "processed")
    parser.add_argument("--original-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = reproduce(args.data_root, args.original_root)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"behavior_wrong_object_facts": {m: result["behavior"][m]["fact_errors"]["wrong_object"] for m in ("qwen", "phi")}, "eligible_wrong_object_coref_pairs": {m: result["behavior"][m]["coreference"]["discordant_wrong_object"] for m in ("qwen", "phi")}, "decomposition_all_fixed_branches_failed": result["decomposition"]["all_fixed_branches_failed"], "native_logit_replay_performed": False}, sort_keys=True))


if __name__ == "__main__":
    main()
