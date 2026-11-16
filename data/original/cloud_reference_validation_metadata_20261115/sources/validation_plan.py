"""Six-world conditional development validation; never automatic confirmation.

Source profiles are fixed across fields, but each field/condition has its own
first-step delta. That delta is frozen across greedy steps at the ORIGINAL query
position. Color sources see their field. File and native-receipt authentication
belong to the caller; all selection below is model-free and deterministic.
"""

import hashlib
import json
import math
import re
import string
from collections import Counter

SEED = "reference-causal-discovery-20261115-v1"
FIELDS = ("color", "shape", "home")
FOCUS = ("none", "coref_0", "coref_1", "distractor")
CONDITIONS = (
    "primaryDcolor",
    "selfRfield",
    "frozenDiscoveryMean",
    "Rcolor",
    "reverseRscreen",
    "cyclicDcolor",
)
CAP = 1_000_000_000
PRACTICAL_TARGETS = {
    "initial_error_repairs_at_least": 4,
    "repair_worlds_at_least": 3,
    "protected_damage_at_most": 1,
    "repair_advantage_over_frozen_mean_at_least": 2,
    "reverse_specific_hits_at_least": 9,
    "reverse_success_worlds_at_least": 3,
    "cyclic_specific_hits_at_least": 9,
    "cyclic_success_worlds_at_least": 3,
    "cyclic_success_object_ids_at_least": 2,
}


def _hash(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def normalize(value):
    return (
        " ".join(value.strip().casefold().split()).rstrip(string.punctuation).rstrip()
    )


def _profile(values):
    if (
        len(values) != 4
        or any(
            type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values
        )
        or sum(values) > 1.000001
        or sum(values) < 1e-4
    ):
        raise ValueError("Unsupported source record mass; no epsilon rescue")
    return [v / sum(values) for v in values]


def _cohort(inputs):
    rows, worlds, predictions = inputs["rows"], inputs["worlds"], inputs["predictions"]
    row_ids, predicted = {r["id"] for r in rows}, {p["id"]: p for p in predictions}
    audit = inputs["behavioral_rescore"]
    if (
        len(rows) != 3072
        or len(row_ids) != 3072
        or len(worlds) != 32
        or len({w["world_id"] for w in worlds}) != 32
        or len(predicted) != len(predictions)
        or set(predicted) != row_ids
        or any(r["split"] != "development" for r in rows)
        or any(w["split"] != "development" for w in worlds)
        or set(Counter(r["world_id"] for r in rows).values()) != {96}
        or audit.get("exact_match") is not True
        or audit.get("confirmation_calls") != 0
        or audit.get("causal_calls") != 0
        or audit.get("native_steps")
        != sum(len(p["generated_ids"]) for p in predictions)
    ):
        raise ValueError(
            "Complete independently rescored old development behavior is required"
        )
    lookup = {
        (
            r["world_id"],
            r["name_swap"],
            r["focus_condition"],
            r["probe_kind"],
            r["query_alias_index"],
            r["field"],
        ): r
        for r in rows
    }

    def correct(row):
        p = predicted[row["id"]]
        if (
            type(p.get("terminated")) is not bool
            or type(p.get("truncated")) is not bool
            or p["terminated"] == p["truncated"]
        ):
            raise ValueError("Explicit old completion metadata is required")
        return p["terminated"] and normalize(p["generated_text"]) == normalize(
            row["expected_answer"]
        )

    eligible = {}
    for world in worlds:
        for swap in (0, 1):
            for focus in FOCUS:
                for recipient in (0, 1):
                    donor, group = 1 - recipient, (world["world_id"], swap, focus)
                    d = {f: lookup[group + ("fact", donor, f)] for f in FIELDS}
                    r = {f: lookup[group + ("fact", recipient, f)] for f in FIELDS}
                    if not all(
                        correct(row)
                        for row in [
                            *d.values(),
                            lookup[group + ("identity", donor, None)],
                        ]
                    ):
                        continue
                    wrong = {}
                    for field, row in r.items():
                        p = predicted[row["id"]]
                        matches = [
                            o
                            for o in "ABCD"
                            if normalize(world["objects"][o][field])
                            == normalize(p["generated_text"])
                        ]
                        if not correct(row) and p["terminated"] and len(matches) == 1:
                            wrong[field] = matches[0]
                    if not wrong:
                        continue
                    identity = (
                        f"{world['world_id']}:swap{swap}:{focus}:recipient{recipient}"
                    )
                    screen = min(wrong, key=lambda f: _hash([SEED, identity, f]))
                    case = {
                        "case_id": identity,
                        "world_id": world["world_id"],
                        "correct_object": world["base_object_id"],
                        "screen_field": screen,
                        "reverse_object": wrong[screen],
                        "wrong_object_by_field": wrong,
                        "initial_wrong_fields": [
                            f for f in FIELDS if not correct(r[f])
                        ],
                        "protected_correct_fields": [
                            f for f in FIELDS if correct(r[f])
                        ],
                        "donor_row_ids": {f: row["id"] for f, row in d.items()},
                        "recipient_row_ids": {f: row["id"] for f, row in r.items()},
                        "screen_row_id": r[screen]["id"],
                    }
                    eligible.setdefault(world["world_id"], []).append(case)
    ordered = sorted(eligible, key=lambda w: _hash([SEED, "world", w]))
    chosen = [
        min(eligible[w], key=lambda c: _hash([SEED, "condition", c["case_id"]]))
        for w in ordered
    ]
    base = inputs["base_plan"]
    if (
        len(chosen) != 14
        or [c["case_id"] for c in chosen[:8]] != [c["case_id"] for c in base["cases"]]
        or ordered[8:] != base["eligibility"]["unused_eligible_world_ids"]
    ):
        raise ValueError(
            "Old hash cohort must reproduce the same first eight and six unused worlds"
        )
    return chosen[8:]


def record_regions(row):
    text, rendered = row["text"], row["rendered_prompt"]
    if rendered.count(text) != 1 or len(row["input_ids"]) != len(row["offset_mapping"]):
        raise ValueError("Invalid rendered token geometry")
    shift, regions = rendered.index(text), {}
    for m in re.finditer(r"(?m)^Object ([A-D])(?: \| | has )[^\n]*", text):
        positions = [
            i
            for i, (left, right) in enumerate(row["offset_mapping"])
            if left < shift + m.end() and right > shift + m.start()
        ]
        if m[1] in regions or not positions or m.end() > row["alias_end_span"][0]:
            raise ValueError("Invalid or duplicate literal record span")
        regions[m[1]] = positions
    all_positions = [p for values in regions.values() for p in values]
    if set(regions) != set("ABCD") or len(all_positions) != len(set(all_positions)):
        raise ValueError("Four disjoint record masks are required")
    return {o: regions[o] for o in "ABCD"}


def plan(inputs):
    """Freeze the head, six old-behavior cases, source mean and endpoint bounds."""
    result, receipt = inputs["discovery_result"], inputs["discovery_rescore"]
    if (
        receipt.get("exact_match") is not True
        or receipt.get("confirmation_calls") != 0
        or receipt.get("assessment") != result["assessment"]
        or receipt.get("native_forward_count") != result["native_forward_count"]
        or not inputs["provenance"]
        or any(
            not isinstance(v, str)
            or len(v) != 64
            or any(c not in "0123456789abcdef" for c in v)
            for v in inputs["provenance"].values()
        )
    ):
        raise ValueError("Exact authenticated routing discovery rescore is required")
    passing = []
    for head in result["assessment"]["head_results"]:
        valid = (
            head["newly_correct"] >= 2
            and head["mean_gains"]["Dcolor"] >= 0.5
            and head["primary_minus_LOOWmean_gain"] >= 0.25
        )
        if valid != head["passed"]:
            raise ValueError("Discovery practical gate metadata changed")
        if valid:
            passing.append(head)
    if not passing:
        raise ValueError("No discovery head passes the previously frozen gate")
    head = min(
        passing,
        key=lambda h: (
            -h["primary_minus_LOOWmean_gain"],
            -h["mean_gains"]["Dcolor"],
            -h["newly_correct"],
            h["layer_index"],
            h["head_index"],
        ),
    )
    selected = {k: head[k] for k in ("layer_index", "head_index")}
    discovery = inputs["discovery_selected_plan"]
    if [c["case_id"] for c in discovery["cases"]] != [
        c["case_id"] for c in inputs["base_plan"]["cases"]
    ]:
        raise ValueError("Routing discovery did not retain the original eight cases")
    summaries = {c["endpoint_id"]: c for c in inputs["discovery_capture_summaries"]}
    if len(summaries) != 32 or set(summaries) != {
        j["endpoint_id"] for j in discovery["capture_jobs"]
    }:
        raise ValueError("All 32 sealed discovery captures are required")
    profiles = [
        _profile(
            summaries[j["endpoint_id"]]["record_masses"][selected["layer_index"]][
                selected["head_index"]
            ]
        )
        for j in discovery["capture_jobs"]
        if j["role"] == "Dcolor"
    ]
    if len(profiles) != 8:
        raise ValueError(
            "Freeze the donor-color mean across exactly eight discovery worlds"
        )
    frozen_mean = [sum(p[i] for p in profiles) / 8 for i in range(4)]
    cases, model = _cohort(inputs), inputs["base_plan"]["model"]
    rows, worlds = (
        {r["id"]: r for r in inputs["rows"]},
        {w["world_id"]: w for w in inputs["worlds"]},
    )
    predictions = {p["id"]: p for p in inputs["predictions"]}
    natural, generations, longest = [], [], 0
    for case in cases:
        geometry = []
        for role in ("donor", "recipient"):
            for field in FIELDS:
                row = rows[case[f"{role}_row_ids"][field]]
                regions = record_regions(row)
                union = sorted(p for values in regions.values() for p in values)
                geometry.append(
                    (
                        regions,
                        [row["input_ids"][p] for p in union],
                        len(row["input_ids"]),
                    )
                )
                longest = max(longest, len(row["input_ids"]))
                natural.append(
                    {
                        "endpoint_id": f"natural::{case['case_id']}::{role}::{field}",
                        "case_id": case["case_id"],
                        "role": role,
                        "field": field,
                        "row_id": row["id"],
                        "record_regions": regions,
                        "original_query_position": len(row["input_ids"]) - 1,
                        "baseline_correct": role == "donor"
                        or field in case["protected_correct_fields"],
                        "expected_label_token_ids": inputs["base_plan"][
                            "label_token_ids"
                        ][field][row["expected_answer"]],
                        "baseline_prediction_id": row["id"],
                    }
                )
        if any(value != geometry[0] for value in geometry[1:]):
            raise ValueError(
                "All six natural contexts need identical record tokens, masks and lengths"
            )
        for condition in CONDITIONS:
            role = (
                "donor"
                if condition in ("reverseRscreen", "cyclicDcolor")
                else "recipient"
            )
            obj = (
                case["reverse_object"]
                if condition == "reverseRscreen"
                else "BCDA"["ABCD".index(case["correct_object"])]
                if condition == "cyclicDcolor"
                else case["correct_object"]
            )
            for field in FIELDS:
                row = rows[case[f"{role}_row_ids"][field]]
                identity = f"{case['case_id']}::{condition}::{field}"
                answers = {
                    o: worlds[case["world_id"]]["objects"][o][field] for o in "ABCD"
                }
                predicted_answer = (
                    predictions[row["id"]]["generated_text"]
                    if condition == "selfRfield"
                    else answers[obj]
                )
                predicted_object = (
                    next(
                        (
                            o
                            for o, answer in answers.items()
                            if normalize(answer) == normalize(predicted_answer)
                        ),
                        None,
                    )
                    if condition == "selfRfield"
                    else obj
                )
                generations.append(
                    {
                        "id": identity,
                        "case_id": case["case_id"],
                        "field": field,
                        "condition": condition,
                        "destination_role": role,
                        "row_id": row["id"],
                        "natural_endpoint_id": f"natural::{case['case_id']}::{role}::{field}",
                        "original_query_position": len(row["input_ids"]) - 1,
                        "operator_endpoint_id": f"operator::{identity}",
                        "greedy_endpoint_ids": [
                            f"greedy::{identity}::step{step:02d}" for step in range(8)
                        ],
                        "task_answer": answers[case["correct_object"]],
                        "baseline_generated_ids": predictions[row["id"]][
                            "generated_ids"
                        ],
                        "baseline_step_endpoint_ids": predictions[row["id"]][
                            "step_endpoint_ids"
                        ],
                        "predicted_answer": predicted_answer,
                        "predicted_object": predicted_object,
                        "object_answers": answers,
                        "protected_before": role == "recipient"
                        and field in case["protected_correct_fields"],
                    }
                )
    if (
        sum(len(c["initial_wrong_fields"]) for c in cases) != 8
        or sum(len(c["protected_correct_fields"]) for c in cases) != 10
    ):
        raise ValueError(
            "The frozen six-world cohort must retain eight errors and ten protected facts"
        )
    estimated = int(
        1.05
        * (
            1008 * model["vocab_size"] * 4
            + 144 * model["layers"] * model["heads"] * longest * 4
            + 108 * longest * model["head_dim"] * 4
            + 864 * model["head_dim"] * 4
            + 64 * 1024**2
        )
    )
    if estimated > CAP:
        raise ValueError("Prospective complete validation evidence exceeds one GB")
    return {
        "study_id": "six-world-routing-validation-20261115-v1",
        "input_sha256": _hash(inputs),
        "provenance": inputs["provenance"],
        "selected_head": selected,
        "model": model,
        "cases": cases,
        "frozen_discovery_mean_Dcolor": frozen_mean,
        "natural_jobs": natural,
        "generation_jobs": generations,
        "baseline": {
            "recipient_wrong": sum(len(c["initial_wrong_fields"]) for c in cases),
            "recipient_protected": sum(
                len(c["protected_correct_fields"]) for c in cases
            ),
            "recipient_fields": 18,
            "donor_correct": 18,
        },
        "target_object_counts": dict(Counter(c["correct_object"] for c in cases)),
        "practical_targets": dict(PRACTICAL_TARGETS),
        "numeric": {
            "max_query_head_drift": 2e-5,
            "first_step_bitwise_equivalence": True,
            "self_whole_generation_bitwise": True,
        },
        "storage": {
            "cap_bytes": CAP,
            "estimated_bytes": estimated,
            "max_native_endpoints": 1008,
            "natural_endpoints": 36,
            "operator_endpoints": 108,
            "greedy_endpoints": 864,
        },
        "scope": "Six conditional development worlds, not 64-world confirmation. Profiles see color; deltas are per-field and frozen across steps at the original query position. No unrelated-alias preservation claim.",
        "execution_authorized": False,
        "confirmation_calls_authorized": False,
    }


def select_sources(planned, captures):
    """Freeze profiles/support for all fields before any intervention outcomes."""
    expected = {j["endpoint_id"]: j for j in planned["natural_jobs"]}
    found = {c["endpoint_id"]: c for c in captures}
    if len(found) != len(captures) or set(found) != set(expected):
        raise ValueError("Exactly all 36 natural validation captures are required")
    by_key, hashes = {}, {}
    for identity, captured in found.items():
        job = expected[identity]
        if (
            type(captured.get("argmax_correct")) is not bool
            or captured["argmax_correct"] != job["baseline_correct"]
        ):
            raise ValueError(
                "Natural first-token qualification changed; no replacement cases"
            )
        values = captured["record_masses"]
        if (
            len(values) != 4
            or any(
                type(v) not in (int, float) or not math.isfinite(v) or v < 0
                for v in values
            )
            or sum(values) > 1.000001
        ):
            raise ValueError("Invalid natural record masses")
        digest = captured["record_value_hashes"]
        if set(digest) != {str(i) for i in range(planned["model"]["layers"])} or any(
            not isinstance(v, str)
            or len(v) != 64
            or any(c not in "0123456789abcdef" for c in v)
            for v in digest.values()
        ):
            raise ValueError("Complete record-V hash coverage is required")
        if job["case_id"] in hashes and hashes[job["case_id"]] != digest:
            raise ValueError(
                "All natural field/alias contexts must have exactly unchanged record V"
            )
        hashes[job["case_id"]] = digest
        by_key[(job["case_id"], job["role"], job["field"])] = captured
    supported, unsupported = [], []
    for case in planned["cases"]:
        cid = case["case_id"]
        for condition in CONDITIONS:
            jobs = [
                j
                for j in planned["generation_jobs"]
                if j["case_id"] == cid and j["condition"] == condition
            ]
            role = jobs[0]["destination_role"]
            if any(
                sum(by_key[(cid, role, f)]["record_masses"]) < 1e-4
                or min(by_key[(cid, role, f)]["record_masses"]) < 1e-8
                for f in FIELDS
            ):
                unsupported.append(
                    {
                        "case_id": cid,
                        "condition": condition,
                        "reason": "recipient_all_three_field_support_failed",
                        "fields_not_run": list(FIELDS),
                    }
                )
                continue
            source_role, source_field = (
                ("recipient", case["screen_field"])
                if condition == "reverseRscreen"
                else ("recipient", "color")
                if condition == "Rcolor"
                else ("donor", "color")
            )
            source_mass = by_key[(cid, source_role, source_field)]["record_masses"]
            if (
                condition not in ("selfRfield", "frozenDiscoveryMean")
                and sum(source_mass) < 1e-4
            ):
                unsupported.append(
                    {
                        "case_id": cid,
                        "condition": condition,
                        "reason": "source_support_failed",
                        "fields_not_run": list(FIELDS),
                    }
                )
                continue
            for job in jobs:
                if condition == "selfRfield":
                    profile = _profile(
                        by_key[(cid, "recipient", job["field"])]["record_masses"]
                    )
                elif condition == "frozenDiscoveryMean":
                    profile = planned["frozen_discovery_mean_Dcolor"]
                else:
                    profile = _profile(source_mass)
                    if condition == "cyclicDcolor":
                        profile = [profile[3], profile[0], profile[1], profile[2]]
                supported.append({**job, "desired_profile": profile})
    return {
        **planned,
        "all_generation_jobs": planned["generation_jobs"],
        "supported_generation_jobs": supported,
        "generation_jobs": [] if unsupported else supported,
        "unsupported_conditions": unsupported,
        "status": "stopped_unsupported_sources"
        if unsupported
        else "sources_frozen_for_root_audit",
        "execution_authorized": False,
        "confirmation_calls_authorized": False,
    }


def practical_flags(values, *, complete):
    """Prospective engineering targets, not significance/publication criteria."""
    if not complete:
        return {
            "repair_replication_passed": None,
            "address_specificity_passed": None,
            "strong_candidate_passed": None,
            "practical_status": "not_evaluable_incomplete_or_unsupported",
        }
    repair = (
        values["primary_repaired"] >= 4
        and values["primary_repair_worlds"] >= 3
        and values["protected_damaged"] <= 1
        and values["primary_repaired"] - values["mean_repaired"] >= 2
    )
    address = (
        values["reverse_hits"] >= 9
        and values["reverse_worlds"] >= 3
        and values["cyclic_hits"] >= 9
        and values["cyclic_worlds"] >= 3
        and values["cyclic_success_object_count"] >= 2
    )
    return {
        "repair_replication_passed": repair,
        "address_specificity_passed": address,
        "strong_candidate_passed": repair and address,
        "practical_status": "evaluated_development_targets_only",
    }


def assess(planned, generations):
    """Describe whole answers and specific steering, never authorize confirmation."""
    jobs = {j["id"]: j for j in planned["generation_jobs"]}
    found = {g["id"]: g for g in generations}
    if len(found) != len(generations) or set(found) != set(jobs):
        raise ValueError("Exactly all supported generation records are required")
    intended = planned.get("all_generation_jobs", planned["generation_jobs"])
    counts = {
        c: Counter(
            {
                "intended_fields": 18,
                "observed_fields": 0,
                "task_correct": 0,
                "predicted_target_hits": 0,
                "repaired_initial_errors": 0,
                "protected_damaged": 0,
                "other_listed_wrong": 0,
                "malformed_or_truncated": 0,
            }
        )
        for c in CONDITIONS
    }
    outcomes = {}
    repaired_worlds, reverse_worlds, cyclic_worlds, cyclic_objects = (
        set(),
        set(),
        set(),
        set(),
    )
    for identity, g in found.items():
        j = jobs[identity]
        if (
            "task_answer" in g
            and g["task_answer"] != j["task_answer"]
            or "predicted_target_answer" in g
            and g["predicted_target_answer"] != j["predicted_answer"]
        ):
            raise ValueError(
                "Task truth E and steering predictions must not be changed or conflated"
            )
        if (
            type(g.get("terminated")) is not bool
            or type(g.get("truncated")) is not bool
            or g["terminated"] == g["truncated"]
        ):
            raise ValueError(
                "Every complete answer needs explicit consistent completion metadata"
            )
        finished = g["terminated"] and not g["truncated"]
        answer = normalize(g["generated_text"])
        correct = finished and answer == normalize(j["task_answer"])
        hit = finished and answer == normalize(j["predicted_answer"])
        for key, value in (("correct_literal", correct), ("hit_predicted_target", hit)):
            if key in g and g[key] != value:
                raise ValueError(
                    "Saved grading differs from strict whole-answer scoring"
                )
        if j["condition"] == "selfRfield" and (
            not hit or g["generated_ids"] != j["baseline_generated_ids"]
        ):
            raise ValueError(
                "Self generation did not reproduce the old complete answer and IDs"
            )
        c = counts[j["condition"]]
        c["observed_fields"] += 1
        c["task_correct"] += correct
        c["predicted_target_hits"] += hit
        c["repaired_initial_errors"] += (
            j["destination_role"] == "recipient"
            and not j["protected_before"]
            and correct
        )
        c["protected_damaged"] += j["protected_before"] and not correct
        listed = finished and answer in {
            normalize(v) for v in j["object_answers"].values()
        }
        c["other_listed_wrong"] += listed and not correct and not hit
        c["malformed_or_truncated"] += not listed
        if j["condition"] == "primaryDcolor" and not j["protected_before"] and correct:
            repaired_worlds.add(j["case_id"])
        if j["condition"] == "reverseRscreen" and hit:
            reverse_worlds.add(j["case_id"])
        if j["condition"] == "cyclicDcolor" and hit:
            cyclic_worlds.add(j["case_id"])
            cyclic_objects.add(j["predicted_object"])
        outcomes[(j["case_id"], j["field"], j["condition"])] = {
            "task_correct": bool(correct),
            "predicted_hit": bool(hit),
        }
    report = {}
    for condition, c in counts.items():
        observed, correct = c["observed_fields"], c["task_correct"]
        report[condition] = {
            **dict(c),
            "observed_task_accuracy": correct / observed if observed else None,
            "unconditional_task_accuracy": correct / 18 if observed == 18 else None,
            "unconditional_task_accuracy_bounds": [
                correct / 18,
                (correct + 18 - observed) / 18,
            ],
            "three_field_joint_predicted_worlds": sum(
                all(
                    outcomes.get((case["case_id"], f, condition), {}).get(
                        "predicted_hit", False
                    )
                    for f in FIELDS
                )
                for case in planned["cases"]
            ),
        }
    complete = len(found) == len(intended) == 108
    practical_values = {
        "primary_repaired": counts["primaryDcolor"]["repaired_initial_errors"],
        "primary_repair_worlds": len(repaired_worlds),
        "protected_damaged": counts["primaryDcolor"]["protected_damaged"],
        "mean_repaired": counts["frozenDiscoveryMean"]["repaired_initial_errors"],
        "reverse_hits": counts["reverseRscreen"]["predicted_target_hits"],
        "reverse_worlds": len(reverse_worlds),
        "cyclic_hits": counts["cyclicDcolor"]["predicted_target_hits"],
        "cyclic_worlds": len(cyclic_worlds),
        "cyclic_success_object_count": len(cyclic_objects),
    }
    return {
        "by_condition": report,
        "intended_worlds": 6,
        "intended_generations": len(intended),
        "observed_generations": len(found),
        "unsupported_conditions": planned.get("unsupported_conditions", []),
        "practical_counts": practical_values,
        "practical_targets": dict(PRACTICAL_TARGETS),
        "cyclic_success_object_ids": sorted(cyclic_objects),
        **practical_flags(practical_values, complete=complete),
        "primary_minus_frozen_mean_task_accuracy": (
            counts["primaryDcolor"]["task_correct"]
            - counts["frozenDiscoveryMean"]["task_correct"]
        )
        / 18
        if complete
        else None,
        "scope": "Conditional six-world development validation only. Reverse/cyclic hits are specific newly induced records, not generic damage. Unrelated unpatched aliases are not preservation evidence.",
        "mechanism_identified": False,
        "confirmation_calls_authorized": False,
    }
