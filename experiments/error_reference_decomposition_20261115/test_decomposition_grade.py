"""Whole-answer grading and fixed gates, defined before experimental outcomes."""

import copy

import pytest
from decomposition_grade import CONDITIONS, grade


def fixture():
    jobs, generations = [], []
    for condition in CONDITIONS:
        for index in range(18):
            world, field = f"world{index // 3}", ("color", "shape", "home")[index % 3]
            identity = f"{condition}:{world}:{field}"
            endpoints = [f"{identity}:step{step}" for step in range(8)]
            jobs.append(
                {
                    "id": identity,
                    "case_id": world,
                    "world_id": world,
                    "field": field,
                    "condition": condition,
                    "row_id": f"row:{world}:{field}",
                    "expected_answer": "red",
                    "baseline_correct": index >= 8,
                    "original_query_position": 100,
                    "greedy_endpoint_ids": endpoints,
                }
            )
            generations.append(
                {
                    "id": identity,
                    "expected_answer": "red",
                    "generated_ids": [11, 2],
                    "generated_text": "red" if index >= 8 else "blue",
                    "terminated": True,
                    "truncated": False,
                    "stop_reason": "eos",
                    "step_endpoint_ids": endpoints[:2],
                }
            )
    return {
        "generation_jobs": jobs,
        "baseline": {"recipient_wrong": 8, "recipient_protected": 10},
        "cases": [f"world{i}" for i in range(6)],
    }, generations


def repair(generations, condition, indices):
    selected = [item for item in generations if item["id"].startswith(condition + ":")]
    for index in indices:
        selected[index]["generated_text"] = " RED. "


def test_all_failed_stops_but_fixed_factor_can_pass_when_full_head_fails():
    planned, generations = fixture()
    planned["baseline"].update(donor_correct=18, recipient_fields=18)
    assert grade(planned, generations)["all_fixed_branches_failed"]
    repair(generations, "sameFieldRecordContribution", [0, 3, 6, 7])
    report = grade(planned, generations)
    assert report["conditions"]["sameFieldRecordContribution"]["diagnostic_passed"]
    assert not report["conditions"]["sameFieldFullHead"]["diagnostic_passed"]
    assert not report["all_fixed_branches_failed"]
    assert report["confirmation_authorized"] is False


def test_repair_worlds_and_protected_damage_are_part_of_the_gate():
    planned, generations = fixture()
    repair(generations, "sameFieldMass", [0, 1, 2, 3])
    assert not grade(planned, generations)["conditions"]["sameFieldMass"][
        "diagnostic_passed"
    ]
    repair(generations, "sameFieldMass", [6])
    selected = [item for item in generations if item["id"].startswith("sameFieldMass:")]
    selected[8]["generated_text"] = selected[9]["generated_text"] = "blue"
    assert not grade(planned, generations)["conditions"]["sameFieldMass"][
        "diagnostic_passed"
    ]


def test_full_head_random_comparison_is_scoped_and_not_automatically_confirmed():
    planned, generations = fixture()
    repair(generations, "sameFieldFullHead", [0, 3, 6, 7])
    repair(generations, "normMatchedRandomFullHead", [0, 3, 6])
    report = grade(planned, generations)
    assert report["conditions"]["sameFieldFullHead"]["diagnostic_passed"]
    assert not report["full_head_beats_matched_random_by_two_repairs"]
    assert "sameFieldFullHead" not in report["branches_for_possible_new_plan"]


def test_prefix_and_truncation_are_not_correct_completed_answers():
    planned, generations = fixture()
    selected = [item for item in generations if item["id"].startswith("sameFieldMass:")]
    selected[0]["generated_text"] = "Answer: red"
    selected[1].update(
        generated_text="red",
        generated_ids=list(range(8)),
        terminated=False,
        truncated=True,
        stop_reason="max_tokens",
        step_endpoint_ids=planned["generation_jobs"][37]["greedy_endpoint_ids"],
    )
    report = grade(planned, generations)
    assert report["conditions"]["sameFieldMass"]["repaired"] == 0
    assert report["conditions"]["sameFieldMass"]["truncated"] == 1


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "label", "termination", "endpoint", "baseline"]
)
def test_corrupt_or_incomplete_metadata_fails_closed(mutation):
    planned, generations = fixture()
    generations = copy.deepcopy(generations)
    if mutation == "missing":
        generations.pop()
    elif mutation == "duplicate":
        generations[-1] = generations[0]
    elif mutation == "label":
        generations[0]["expected_answer"] = "blue"
    elif mutation == "termination":
        generations[0]["terminated"] = 1
    elif mutation == "endpoint":
        generations[0]["step_endpoint_ids"][0] = "different"
    else:
        planned["generation_jobs"][0]["baseline_correct"] = True
    with pytest.raises(ValueError):
        grade(planned, generations)
