"""Strict whole-answer grading for a fixed, reopened development diagnosis."""

import string
from collections import Counter

CONDITIONS = (
    "self",
    "oldDcolorMass",
    "sameFieldMass",
    "sameFieldWithinOnly",
    "sameFieldProfileWithin",
    "sameFieldRecordContribution",
    "sameFieldNonrecordResidual",
    "sameFieldFullHead",
    "normMatchedRandomFullHead",
)


def normalize(text):
    return " ".join(text.strip().casefold().split()).rstrip(string.punctuation).rstrip()


def grade(planned, generations):
    """Do not treat first-token success, answer prefixes or truncation as repair.

    Literal truth is the original registry answer under every intervention.
    Native-token/EOS reconstruction and authentication belong to the executor's
    separate rescore; this function validates whole-answer metadata and gates.
    """
    jobs = planned["generation_jobs"]
    lookup = {item["id"]: item for item in generations}
    if (
        len(jobs) != 162
        or len({job["id"] for job in jobs}) != 162
        or len(generations) != 162
        or len(lookup) != 162
        or set(lookup) != {job["id"] for job in jobs}
    ):
        raise ValueError(
            "Exactly all 162 planned unique complete generations are required"
        )
    if any(
        planned["baseline"].get(key) != value
        for key, value in {"recipient_wrong": 8, "recipient_protected": 10}.items()
    ) or Counter(job["condition"] for job in jobs) != Counter(
        {name: 18 for name in CONDITIONS}
    ):
        raise ValueError("The fixed nine-cell cohort and baseline changed")
    if (
        len({job["world_id"] for job in jobs}) != 6
        or len({job["case_id"] for job in jobs}) != 6
    ):
        raise ValueError("Exactly six independent development worlds/cases required")
    targets = {}
    for job in jobs:
        key = (job["case_id"], job["field"])
        value = (
            job["row_id"],
            job["world_id"],
            job["expected_answer"],
            job["baseline_correct"],
        )
        if (
            type(job["baseline_correct"]) is not bool
            or targets.setdefault(key, value) != value
        ):
            raise ValueError("Conditions must share the same literal task and baseline")
    if (
        len(targets) != 18
        or len({(job["condition"], job["case_id"], job["field"]) for job in jobs})
        != 162
    ):
        raise ValueError("Every condition must cover exactly three fields in each case")
    for case in {job["case_id"] for job in jobs}:
        selected = [job for job in jobs if job["case_id"] == case]
        if {job["field"] for job in selected} != {"color", "shape", "home"} or len(
            {job["world_id"] for job in selected}
        ) != 1:
            raise ValueError(
                "A case must belong to one world and contain all three fields"
            )
    reports, per_question = {}, []
    for condition in CONDITIONS:
        selected = [job for job in jobs if job["condition"] == condition]
        if sum(not job["baseline_correct"] for job in selected) != 8:
            raise ValueError("Each condition must have exactly eight baseline errors")
        counts, repaired_worlds = Counter(), set()
        for job in selected:
            prediction = lookup[job["id"]]
            ids, endpoints = (
                prediction["generated_ids"],
                prediction["step_endpoint_ids"],
            )
            terminated, truncated = prediction["terminated"], prediction["truncated"]
            if (
                prediction["expected_answer"] != job["expected_answer"]
                or not isinstance(prediction["generated_text"], str)
                or type(terminated) is not bool
                or type(truncated) is not bool
                or terminated == truncated
                or not 1 <= len(ids) <= 8
                or any(type(token) is not int or token < 0 for token in ids)
                or len(job["greedy_endpoint_ids"]) != 8
                or len(set(job["greedy_endpoint_ids"])) != 8
                or endpoints != job["greedy_endpoint_ids"][: len(ids)]
                or prediction["stop_reason"] != ("eos" if terminated else "max_tokens")
                or truncated
                and len(ids) != 8
            ):
                raise ValueError(
                    "Generation truth, coverage or explicit termination metadata changed"
                )
            correct = terminated and normalize(
                prediction["generated_text"]
            ) == normalize(job["expected_answer"])
            repaired = not job["baseline_correct"] and correct
            damaged = job["baseline_correct"] and not correct
            counts.update(
                {
                    "correct": int(correct),
                    "repaired": int(repaired),
                    "protected_damaged": int(damaged),
                    "truncated": int(truncated),
                }
            )
            if repaired:
                repaired_worlds.add(job["world_id"])
            per_question.append(
                {
                    "id": job["id"],
                    "condition": condition,
                    "world_id": job["world_id"],
                    "field": job["field"],
                    "correct": correct,
                    "repaired": repaired,
                    "protected_damaged": damaged,
                }
            )
        reports[condition] = {
            **dict(counts),
            "count": 18,
            "initial_errors": 8,
            "initial_protected": 10,
            "repair_worlds": sorted(repaired_worlds),
            "diagnostic_passed": counts["repaired"] >= 4
            and len(repaired_worlds) >= 3
            and counts["protected_damaged"] <= 1,
        }
    branches = [
        name for name in CONDITIONS if name not in ("self", "normMatchedRandomFullHead")
    ]
    advantage = (
        reports["sameFieldFullHead"]["repaired"]
        - reports["normMatchedRandomFullHead"]["repaired"]
    )
    return {
        "conditions": reports,
        "per_question": per_question,
        "all_fixed_branches_failed": all(
            not reports[name]["diagnostic_passed"] for name in branches
        ),
        "full_head_beats_matched_random_by_two_repairs": advantage >= 2,
        "full_head_repair_advantage_over_matched_random": advantage,
        "branches_for_possible_new_plan": [
            name
            for name in branches
            if reports[name]["diagnostic_passed"]
            and (name != "sameFieldFullHead" or advantage >= 2)
        ],
        "confirmation_authorized": False,
        "new_entity_mechanism_established": False,
        "scope": "Six previously opened development worlds; exploratory diagnosis only. A full natural-head transfer is a comparator, not a mathematical performance bound or proof of entity abstraction. Random control matches only the full-head delta norm, not factor doses.",
    }
