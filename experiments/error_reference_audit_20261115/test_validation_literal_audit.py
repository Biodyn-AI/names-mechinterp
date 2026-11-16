"""Independent literal E/W/cyclic targets on synthetic whole answers only."""

import copy

import pytest
import validation_literal_audit as audit


def fixture():
    objects = {
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
    cases, rows, old, jobs, new = [], [], [], [], []
    for number, entity in enumerate("CCCAAB"):
        cid, wrong = f"case{number}", "D"
        prefix = "\n".join(
            f"Object {obj} | nameplate Name{obj} | color {v['color']} | shape {v['shape']} | home {v['home']}."
            for obj, v in objects.items()
        )
        prefix += f"\nAlias apple -> Object {entity}.\nAlias bat -> Object {entity}.\nAlias anchor -> Object A.\nAlias date -> Object B.\n"
        case = {
            "case_id": cid,
            "world_id": f"development:{number}",
            "donor_row_ids": {},
            "recipient_row_ids": {},
        }
        old_by_row = {}
        for role, alias in (("donor", "bat"), ("recipient", "apple")):
            for field in audit.FIELDS:
                identity = f"{cid}:{role}:{field}"
                text = (
                    prefix
                    + f"Requested alias: {alias}.\n[ALIAS_END]\nRequested field: {field}.\nAnswer:"
                )
                answer = objects[entity][field]
                selected = (
                    wrong
                    if role == "recipient"
                    and (field == "home" or field == "shape" and number < 2)
                    else entity
                )
                prediction = {
                    "id": identity,
                    "generated_text": objects[selected][field],
                    "generated_ids": ["ABCD".index(selected) + 1, 99],
                    "terminated": True,
                    "truncated": False,
                    "stop_reason": "eos",
                }
                case[f"{role}_row_ids"][field] = identity
                rows.append({"id": identity, "text": text, "expected_answer": answer})
                old.append(prediction)
                old_by_row[identity] = prediction
        case["screen_row_id"] = case["recipient_row_ids"]["home"]
        cases.append(case)
        for condition in audit.CONDITIONS:
            role = (
                "donor"
                if condition in ("reverseRscreen", "cyclicDcolor")
                else "recipient"
            )
            for field in audit.FIELDS:
                identity = f"{cid}:{condition}:{field}"
                row_id = case[f"{role}_row_ids"][field]
                truth = objects[entity][field]
                target = (
                    objects[wrong][field]
                    if condition == "reverseRscreen"
                    else objects["BCDA"["ABCD".index(entity)]][field]
                    if condition == "cyclicDcolor"
                    else old_by_row[row_id]["generated_text"]
                    if condition == "selfRfield"
                    else truth
                )
                jobs.append(
                    {
                        "id": identity,
                        "case_id": cid,
                        "condition": condition,
                        "field": field,
                        "row_id": row_id,
                        "task_answer": truth,
                        "predicted_answer": target,
                    }
                )
                result = copy.deepcopy(old_by_row[row_id])
                result["id"] = identity
                if condition in ("primaryDcolor", "reverseRscreen", "cyclicDcolor"):
                    result["generated_text"] = target
                    result["generated_ids"] = [10, 99]
                result.update(task_answer=truth, predicted_target_answer=target)
                new.append(result)
    return {"cases": cases, "generation_jobs": jobs}, rows, old, new


def test_literal_truth_countertargets_baselines_and_independent_counts():
    values = fixture()
    report = audit.audit(*values)
    assert report["baseline"] == {"recipient_wrong": 8, "recipient_protected": 10}
    assert report["independent_worlds"] == 6
    assert report["by_condition"]["reverseRscreen"]["task_correct"] == 0
    assert report["by_condition"]["reverseRscreen"]["predicted_target_hits"] == 18
    assert report["by_condition"]["cyclicDcolor"]["predicted_target_hits"] == 18
    assert report["by_condition"]["selfRfield"]["predicted_target_hits"] == 18
    assert report["by_condition"]["primaryDcolor"]["repaired_initial_errors"] == 8
    assert report["strong_candidate_passed"] is True
    assert report["practical_counts"]["cyclic_success_object_count"] == 3


def test_forged_truth_countertargets_old_screen_and_incomplete_grid_are_rejected():
    for mutation in ("truth", "countertarget", "oldscreen", "missing", "duplicate"):
        plan, rows, old, new = fixture()
        if mutation == "truth":
            plan["generation_jobs"][0]["task_answer"] = "invented"
        elif mutation == "countertarget":
            next(
                j for j in plan["generation_jobs"] if j["condition"] == "reverseRscreen"
            )["predicted_answer"] = "invented"
        elif mutation == "oldscreen":
            identity = plan["cases"][0]["screen_row_id"]
            next(p for p in old if p["id"] == identity)["generated_text"] = "river"
        elif mutation == "missing":
            new.pop()
        else:
            new[-1] = copy.deepcopy(new[0])
        with pytest.raises(ValueError):
            audit.audit(plan, rows, old, new)


def test_whole_answer_strictness_truncation_and_exact_self_trajectory_metadata():
    plan, rows, old, new = fixture()
    primary = [g for g in new if ":primaryDcolor:" in g["id"]]
    primary[0]["generated_text"] += "\nExplanation."
    primary[1].update(terminated=False, truncated=True, stop_reason="max_tokens")
    report = audit.audit(plan, rows, old, new)
    assert report["by_condition"]["primaryDcolor"]["malformed_or_truncated"] == 2
    self_answer = next(g for g in new if ":selfRfield:" in g["id"])
    self_answer["generated_ids"] = [123, 99]
    with pytest.raises(ValueError, match="Self"):
        audit.audit(plan, rows, old, new)
