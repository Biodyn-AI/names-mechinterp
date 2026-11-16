"""Finite answer-time record-allocation test; pure planning, never execution.

The caller authenticates prior files and native rescores. Head selection uses
gold object labels on development data; profile interventions themselves do not.
Color donors see a field, so this is not a field-blind/entity-code experiment.
"""

import hashlib
import json
import math
import re
from collections import Counter

OBJECTS = "ABCD"
ROLES = ("Dcolor", "Dscreen", "Rcolor", "Rscreen")
CONDITIONS = ("Dcolor", "selfRscreen", "Dscreen", "Rcolor", "LOOWmeanDcolor")
CAP = 500_000_000
TOTAL_FLOOR, RECORD_FLOOR = 1e-4, 1e-8


class QualificationError(ValueError):
    """Frozen natural first-token eligibility failed, not routing absence."""


def _digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _sha(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def record_regions(row):
    """Map the four literal printed record lines to disjoint token positions."""
    text, rendered = row["text"], row["rendered_prompt"]
    if rendered.count(text) != 1 or len(row["offset_mapping"]) != len(row["input_ids"]):
        raise ValueError("Rendered text and token offsets must match exactly")
    shift, regions = rendered.index(text), {}
    for match in re.finditer(r"(?m)^Object ([A-D])(?: \| | has )[^\n]*", text):
        obj = match[1]
        if obj in regions or match.end() > row["alias_end_span"][0]:
            raise ValueError("Record lines must be unique and precede the query marker")
        positions = [
            i
            for i, (left, right) in enumerate(row["offset_mapping"])
            if left < shift + match.end() and right > shift + match.start()
        ]
        if not positions:
            raise ValueError("Each record needs nonempty token positions")
        regions[obj] = positions
    union = [p for positions in regions.values() for p in positions]
    if set(regions) != set(OBJECTS) or len(union) != len(set(union)):
        raise ValueError("Exactly four nonoverlapping record masks are required")
    return {obj: regions[obj] for obj in OBJECTS}


def plan(inputs):
    """Retain the previously frozen eight cases, without inspecting new attention."""
    base, audit = inputs["base_plan"], inputs["prior_audit"]
    if (
        audit.get("exact_match") is not True
        or audit.get("confirmation_calls") != 0
        or audit.get("physical_counts", {}).get("native", 0) <= 0
        or not inputs["provenance"]
        or not all(_sha(v) for v in inputs["provenance"].values())
    ):
        raise ValueError(
            "An authenticated exact prior audit and hash anchors are required"
        )
    cases, model = base["cases"], base["model"]
    if (
        len(cases) != 8
        or len({c["world_id"] for c in cases}) != 8
        or len({c["case_id"] for c in cases}) != 8
        or any(c["screen_field"] not in ("shape", "home") for c in cases)
        or model["model_id"] != "Qwen/Qwen3-4B-Instruct-2507"
        or any(
            type(model[k]) is not int or model[k] <= 0
            for k in ("layers", "heads", "head_dim", "vocab_size")
        )
    ):
        raise ValueError(
            "Exactly the eight distinct pinned-model cross-field cases are required"
        )
    rows = {r["id"]: r for r in inputs["rows"]}
    worlds = {w["world_id"]: w for w in inputs["worlds"]}
    if len(rows) != len(inputs["rows"]) or len(worlds) != len(inputs["worlds"]):
        raise ValueError("Duplicate rows or worlds are forbidden")
    jobs, longest, screen_labels = [], 0, {}
    for case in cases:
        same, geometry = [], []
        source_rows = (
            case["donor_row_ids"]["color"],
            case["donor_row_ids"][case["screen_field"]],
            case["recipient_row_ids"]["color"],
            case["screen_row_id"],
        )
        world = worlds[case["world_id"]]
        if (
            world["split"] != "development"
            or case["correct_object"] not in OBJECTS
            or case["wrong_object_by_field"][case["screen_field"]] not in OBJECTS
            or case["correct_object"]
            == case["wrong_object_by_field"][case["screen_field"]]
        ):
            raise ValueError(
                "Only previously selected development wrong-object cases may continue"
            )
        field, wrong = (
            case["screen_field"],
            case["wrong_object_by_field"][case["screen_field"]],
        )
        correct_ids = base["label_token_ids"][field][
            world["objects"][case["correct_object"]][field]
        ]
        wrong_ids = base["label_token_ids"][field][world["objects"][wrong][field]]
        if (
            not correct_ids
            or not wrong_ids
            or set(correct_ids).intersection(wrong_ids)
            or any(
                type(i) is not int or not 0 <= i < model["vocab_size"]
                for i in correct_ids + wrong_ids
            )
        ):
            raise ValueError("Distinct complete-label token sets are required")
        screen_labels[case["case_id"]] = {"correct": correct_ids, "wrong": wrong_ids}
        for role, identity in zip(ROLES, source_rows):
            row = rows[identity]
            if (
                row["world_id"] != case["world_id"]
                or row["split"] != "development"
                or row["probe_kind"] != "fact"
                or row["field"]
                != ("color" if role.endswith("color") else case["screen_field"])
                or row["expected_answer"]
                != world["objects"][case["correct_object"]][row["field"]]
            ):
                raise ValueError(
                    "A source row differs from its frozen case or expected object"
                )
            regions = record_regions(row)
            union = sorted(p for positions in regions.values() for p in positions)
            if any(
                type(i) is not int or not 0 <= i < model["vocab_size"]
                for i in row["input_ids"]
            ):
                raise ValueError("Invalid token IDs")
            same.append((regions, [row["input_ids"][i] for i in union]))
            geometry.append(len(row["input_ids"]))
            longest = max(longest, len(row["input_ids"]))
            jobs.append(
                {
                    "endpoint_id": f"capture::{case['case_id']}::{role}",
                    "case_id": case["case_id"],
                    "role": role,
                    "row_id": identity,
                    "record_regions": regions,
                    "expected_label_token_ids": base["label_token_ids"][row["field"]][
                        row["expected_answer"]
                    ],
                }
            )
        if any(value != same[0] for value in same[1:]) or len(set(geometry)) != 1:
            raise ValueError(
                "All four contexts need identical record tokens, positions and full lengths"
            )
    endpoints = 32 + 2 * 8 * 5
    estimated = int(
        1.05
        * (
            endpoints * model["vocab_size"] * 4
            + endpoints * model["layers"] * model["heads"] * longest * 4
            + endpoints * longest * model["head_dim"] * 4
            + 64 * 1024**2
        )
    )
    if estimated > CAP:
        raise ValueError("Prospective complete evidence exceeds the 500-MB cap")
    return {
        "study_id": "answer-time-record-routing-20261115-v1",
        "input_sha256": _digest(inputs),
        "provenance": inputs["provenance"],
        "model": model,
        "cases": cases,
        "capture_jobs": jobs,
        "label_token_ids": base["label_token_ids"],
        "screen_label_token_ids": screen_labels,
        "target_object_counts": dict(Counter(c["correct_object"] for c in cases)),
        "record_mass_floors": {
            "all_four_context_totals": TOTAL_FLOOR,
            "each_recipient_screen_record": RECORD_FLOOR,
        },
        "storage": {
            "cap_bytes": CAP,
            "estimated_bytes": estimated,
            "max_native_endpoints": endpoints,
            "capture_endpoints": 32,
        },
        "continuation_gates": {
            "newly_correct_at_least": 2,
            "primary_mean_gain_at_least": 0.5,
            "primary_minus_LOOWmean_gain_at_least": 0.25,
            "Rcolor_or_Dscreen_superiority_required": False,
        },
        "limitations": "Eight development cases are target-ID imbalanced; no C target. Selection is gold-informed, but the full four-record profile operation is gold-blind. Color donors see a field. This tests record allocation, not a field-blind or abstract entity variable. Unsupported heads do not establish absence of routing.",
        "execution_authorized": False,
        "further_validation_authorized": False,
        "confirmation_calls_authorized": False,
    }


def _validated(planned, captures):
    expected = {j["endpoint_id"]: j for j in planned["capture_jobs"]}
    found = {c["endpoint_id"]: c for c in captures}
    if len(found) != len(captures) or set(found) != set(expected):
        raise ValueError("Exactly all 32 registered natural captures are required")
    result = {}
    model = planned["model"]
    for identity, captured in found.items():
        hashes = captured["record_value_hashes"]
        masses = captured["record_masses"]
        if (
            set(hashes) != {str(l) for l in range(model["layers"])}
            or not all(_sha(value) for value in hashes.values())
            or len(masses) != model["layers"]
            or any(len(layer) != model["heads"] for layer in masses)
        ):
            raise ValueError(
                "Incomplete natural layer/head captures or record-value hashes"
            )
        for layer in masses:
            for values in layer:
                if (
                    len(values) != 4
                    or any(
                        type(v) not in (int, float) or not math.isfinite(v) or v < 0
                        for v in values
                    )
                    or sum(values) > 1.000001
                ):
                    raise ValueError(
                        "Record masses must be finite nonnegative native probabilities"
                    )
        job = expected[identity]
        if type(captured.get("argmax_correct")) is not bool:
            raise ValueError("Every capture needs native-argmax correctness metadata")
        if (
            job["role"] in ("Dcolor", "Dscreen") and not captured["argmax_correct"]
        ) or (job["role"] == "Rscreen" and captured["argmax_correct"]):
            raise QualificationError(
                "Natural first-token eligibility failed; do not interpret as absence of routing or replace cases"
            )
        result[(job["case_id"], job["role"])] = captured
    for case in planned["cases"]:
        values = [
            result[(case["case_id"], role)]["record_value_hashes"] for role in ROLES
        ]
        if any(value != values[0] for value in values[1:]):
            raise ValueError(
                "All record V vectors must be bitwise invariant across the four contexts"
            )
    return result


def _normalize(values):
    total = sum(values)
    if total <= 0:
        raise ValueError("Zero mass is unsupported; no epsilon or uniform rescue")
    return [value / total for value in values]


def desired_profiles(planned, captures, layer_index, head_index):
    """Construct source profiles with no correct/wrong-object labels as inputs."""
    records = _validated(planned, captures)
    profiles = {
        c["case_id"]: {
            role: _normalize(
                records[(c["case_id"], role)]["record_masses"][layer_index][head_index]
            )
            for role in ROLES
        }
        for c in planned["cases"]
    }
    return {
        identity: {
            "Dcolor": source["Dcolor"],
            "selfRscreen": source["Rscreen"],
            "Dscreen": source["Dscreen"],
            "Rcolor": source["Rcolor"],
            "LOOWmeanDcolor": _normalize(
                [
                    sum(
                        other["Dcolor"][j]
                        for other_id, other in profiles.items()
                        if other_id != identity
                    )
                    / 7
                    for j in range(4)
                ]
            ),
        }
        for identity, source in profiles.items()
    }


def select_heads(planned, captures):
    records, candidates, unsupported = _validated(planned, captures), [], 0
    model = planned["model"]
    for layer in range(model["layers"]):
        for head in range(model["heads"]):
            scores, supported = [], True
            for case in planned["cases"]:
                masses = {
                    role: records[(case["case_id"], role)]["record_masses"][layer][head]
                    for role in ROLES
                }
                if (
                    any(sum(values) < TOTAL_FLOOR for values in masses.values())
                    or min(masses["Rscreen"]) < RECORD_FLOOR
                ):
                    supported = False
                    break
                donor, recipient = (
                    _normalize(masses["Dscreen"]),
                    _normalize(masses["Rscreen"]),
                )
                e, w = (
                    OBJECTS.index(case["correct_object"]),
                    OBJECTS.index(case["wrong_object_by_field"][case["screen_field"]]),
                )
                scores.append((donor[e] - donor[w]) - (recipient[e] - recipient[w]))
            if not supported:
                unsupported += 1
            elif sum(scores) / 8 > 0:
                candidates.append(
                    {
                        "layer_index": layer,
                        "head_index": head,
                        "mean_alias_contrast": sum(scores) / 8,
                    }
                )
    selected = sorted(
        candidates,
        key=lambda h: (-h["mean_alias_contrast"], h["layer_index"], h["head_index"]),
    )[:2]
    jobs = []
    for head in selected:
        profiles = desired_profiles(
            planned, captures, head["layer_index"], head["head_index"]
        )
        for case in planned["cases"]:
            wrong = case["wrong_object_by_field"][case["screen_field"]]
            for condition in CONDITIONS:
                jobs.append(
                    {
                        "endpoint_id": f"route::{case['case_id']}::l{head['layer_index']:02d}::h{head['head_index']:02d}::{condition}",
                        "case_id": case["case_id"],
                        "row_id": case["screen_row_id"],
                        "layer_index": head["layer_index"],
                        "head_index": head["head_index"],
                        "condition": condition,
                        "desired_profile": profiles[case["case_id"]][condition],
                        "correct_object": case["correct_object"],
                        "wrong_object": wrong,
                        "correct_token_ids": planned["screen_label_token_ids"][
                            case["case_id"]
                        ]["correct"],
                        "wrong_token_ids": planned["screen_label_token_ids"][
                            case["case_id"]
                        ]["wrong"],
                    }
                )
    return {
        **planned,
        "selected_heads": selected,
        "intervention_jobs": jobs,
        "unsupported_heads": unsupported,
        "recipient_color_argmax_correct": {
            c["case_id"]: records[(c["case_id"], "Rcolor")]["argmax_correct"]
            for c in planned["cases"]
        },
        "status": "selected_for_root_audit"
        if selected
        else "no_supported_positive_same_field_head",
    }


def assess(planned, measurements):
    """Practical discovery gate only; never authorize further/held-out calls."""
    jobs = {j["endpoint_id"]: j for j in planned["intervention_jobs"]}
    found = {m["endpoint_id"]: m for m in measurements}
    if len(found) != len(measurements) or set(found) != set(jobs):
        raise ValueError(
            "Every registered intervention endpoint must be assessed exactly once"
        )
    grouped = {}
    for identity, value in found.items():
        gain = value["correct_minus_wrong_gain"]
        if (
            type(gain) not in (int, float)
            or not math.isfinite(gain)
            or type(value["argmax_correct"]) is not bool
            or value.get("clean_argmax_correct") is not False
        ):
            raise ValueError(
                "Finite gains, explicit correctness and eight initially wrong answers are required"
            )
        job = jobs[identity]
        if job["condition"] == "selfRscreen" and (
            abs(gain) > 4e-4 or value["argmax_correct"]
        ):
            raise ValueError("Self-transfer failed its fixed numerical/argmax oracle")
        grouped[
            (job["layer_index"], job["head_index"], job["case_id"], job["condition"])
        ] = value
    summaries = []
    for head in planned["selected_heads"]:
        key = (head["layer_index"], head["head_index"])
        by_condition = {
            condition: [
                grouped[key + (case["case_id"], condition)] for case in planned["cases"]
            ]
            for condition in CONDITIONS
        }
        means = {
            condition: sum(v["correct_minus_wrong_gain"] for v in values) / 8
            for condition, values in by_condition.items()
        }
        repaired = sum(v["argmax_correct"] for v in by_condition["Dcolor"])
        advantage = means["Dcolor"] - means["LOOWmeanDcolor"]
        summaries.append(
            {
                **head,
                "newly_correct": repaired,
                "case_count": 8,
                "mean_gains": means,
                "primary_minus_LOOWmean_gain": advantage,
                "primary_minus_Rcolor_gain_descriptive": means["Dcolor"]
                - means["Rcolor"],
                "primary_minus_Dscreen_gain_descriptive": means["Dcolor"]
                - means["Dscreen"],
                "Rcolor_source_correct_count": sum(
                    planned["recipient_color_argmax_correct"].values()
                ),
                "passed": repaired >= 2
                and means["Dcolor"] >= 0.5
                and advantage >= 0.25,
            }
        )
    return {
        "head_results": summaries,
        "passed": any(s["passed"] for s in summaries),
        "scope": "Development first-token feasibility only, not complete-answer repair or confirmation.",
        "further_validation_authorized": False,
        "confirmation_calls_authorized": False,
    }
