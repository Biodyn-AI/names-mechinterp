"""Synthetic printed registries test literal grading without experimental graders."""

import copy

import literal_receipt as receipt
import pytest


def fixture():
    facts = {
        obj: dict(zip(("color", "shape", "home"), values))
        for obj, values in zip(
            "ABCD",
            (
                ("red", "circle", "forest"),
                ("blue", "square", "castle"),
                ("green", "triangle", "river"),
                ("yellow", "star", "tower"),
            ),
        )
    }
    rows, old, cases, jobs, new = [], [], [], [], []
    for number, entity in enumerate("CCCAAB"):
        world, cid = f"development:{number}", f"case{number}"
        prefix = "\n".join(
            f"Object {obj} | nameplate Name{obj} | color {f['color']} | shape {f['shape']} | home {f['home']}."
            for obj, f in facts.items()
        )
        prefix += f"\nAlias apple -> Object {entity}.\nAlias bat -> Object {entity}.\nAlias anchor -> Object A.\nAlias date -> Object B.\n"
        case = {
            "case_id": cid,
            "world_id": world,
            "recipient_row_ids": {},
            "donor_row_ids": {},
        }
        for role, alias in (("donor", "bat"), ("recipient", "apple")):
            for field in receipt.FIELDS:
                row_id = f"{cid}:{role}:{field}"
                label = facts[entity][field]
                correct = role == "donor" or not (
                    field == "home" or field == "shape" and number < 2
                )
                answer = label if correct else facts["D"][field]
                rows.append(
                    {
                        "id": row_id,
                        "world_id": world,
                        "field": field,
                        "text": prefix
                        + f"Requested alias: {alias}.\n[ALIAS_END]\nRequested field: {field}.\nAnswer:",
                        "expected_answer": label,
                    }
                )
                original = {
                    "id": row_id,
                    "generated_text": answer,
                    "generated_ids": [1 if correct else 2, 99],
                    "terminated": True,
                    "truncated": False,
                    "stop_reason": "eos",
                }
                old.append(original)
                case[f"{role}_row_ids"][field] = row_id
                if role == "donor":
                    continue
                for condition in receipt.CONDITIONS:
                    identity = f"{cid}:{field}:{condition}"
                    endpoints = [f"{identity}:step{i}" for i in range(8)]
                    jobs.append(
                        {
                            "id": identity,
                            "case_id": cid,
                            "world_id": world,
                            "condition": condition,
                            "field": field,
                            "row_id": row_id,
                            "expected_answer": label,
                            "baseline_correct": correct,
                            "greedy_endpoint_ids": endpoints,
                        }
                    )
                    prediction = {
                        **original,
                        "id": identity,
                        "expected_answer": label,
                        "step_endpoint_ids": endpoints[:2],
                    }
                    if condition == "oldDcolorMass" and number == 0 and field == "home":
                        prediction.update(generated_text=label, generated_ids=[1, 99])
                    new.append(prediction)
        cases.append(case)
    plan = {
        "cases": cases,
        "generation_jobs": jobs,
        "baseline": {
            "recipient_wrong": 8,
            "recipient_protected": 10,
            "donor_correct": 18,
        },
    }
    return plan, {"rows": rows, "predictions": old}, new


def test_literal_baseline_whole_answers_and_negative_gates_are_independent():
    report = receipt.reconstruct(*fixture())
    assert report["baseline"] == {"recipient_wrong": 8, "recipient_protected": 10}
    assessment = report["assessment"]
    assert assessment["all_fixed_branches_failed"] is True
    assert assessment["conditions"]["oldDcolorMass"]["repaired"] == 1
    assert assessment["conditions"]["sameFieldFullHead"]["repaired"] == 0
    assert all(c["protected_damaged"] == 0 for c in assessment["conditions"].values())
    assert receipt.compare_assessment(assessment, copy.deepcopy(assessment)) is True
    forged = copy.deepcopy(assessment)
    forged["conditions"]["sameFieldFullHead"]["repaired"] = 4
    with pytest.raises(ValueError, match="assessment"):
        receipt.compare_assessment(assessment, forged)


@pytest.mark.parametrize(
    "mutation",
    ["truth", "baseline", "field", "world", "duplicate", "self_ids", "termination"],
)
def test_forged_labels_metadata_grid_and_self_control_are_rejected(mutation):
    plan, inputs, generations = fixture()
    if mutation == "truth":
        plan["generation_jobs"][0]["expected_answer"] = "invented"
    elif mutation == "baseline":
        plan["generation_jobs"][0]["baseline_correct"] = False
    elif mutation == "field":
        plan["generation_jobs"][0]["field"] = "home"
    elif mutation == "world":
        plan["generation_jobs"][0]["world_id"] = "foreign"
    elif mutation == "duplicate":
        generations[-1] = copy.deepcopy(generations[0])
    elif mutation == "self_ids":
        next(g for g in generations if g["id"].endswith(":self"))["generated_ids"] = [
            123,
            99,
        ]
    else:
        generations[0]["terminated"] = 1
    with pytest.raises(ValueError):
        receipt.reconstruct(plan, inputs, generations)


def test_extra_explanations_and_truncated_correct_prefixes_are_not_repairs():
    plan, inputs, generations = fixture()
    job = next(
        j
        for j in plan["generation_jobs"]
        if j["condition"] == "sameFieldMass" and not j["baseline_correct"]
    )
    prediction = next(g for g in generations if g["id"] == job["id"])
    prediction["generated_text"] = job["expected_answer"] + "\nExplanation."
    assert (
        receipt.reconstruct(plan, inputs, generations)["assessment"]["conditions"][
            "sameFieldMass"
        ]["repaired"]
        == 0
    )
    prediction.update(
        generated_text=job["expected_answer"],
        generated_ids=list(range(8)),
        step_endpoint_ids=job["greedy_endpoint_ids"],
        terminated=False,
        truncated=True,
        stop_reason="max_tokens",
    )
    assert (
        receipt.reconstruct(plan, inputs, generations)["assessment"]["conditions"][
            "sameFieldMass"
        ]["truncated"]
        == 1
    )
