"""Synthetic complete old behavior, not validation-model outcomes."""

import copy
import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "error_reference_study_20261114")
)
import error_protocol
import validation_plan as validation


def _hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def fixture():
    study = error_protocol.build_study()
    rows = [r for r in study["rows"] if r["split"] == "development"]
    worlds = [w for w in study["worlds"] if w["split"] == "development"]
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
    by_world = {w["world_id"]: w for w in worlds}
    eligible_worlds = {w["world_id"] for w in worlds[:14]}
    ordered = sorted(
        eligible_worlds, key=lambda w: _hash([validation.SEED, "world", w])
    )
    extra_shape_worlds = set(ordered[8:10])
    predictions = []
    for row in rows:
        matches = list(
            re.finditer(r"(?m)^Object ([A-D])(?: \| | has )[^\n]*", row["text"])
        )
        end = matches[-1].end()
        offsets = [[m.start(), m.end()] for m in matches] + [
            [end, end + 1],
            [end + 1, len(row["text"])],
        ]
        row.update(
            rendered_prompt=row["text"],
            offset_mapping=offsets,
            input_ids=[10 + "ABCD".index(m[1]) for m in matches]
            + [
                20 + (row["query_alias_index"] or 0),
                30 + (validation.FIELDS.index(row["field"]) if row["field"] else 3),
            ],
        )
        answer = row["expected_answer"]
        if (
            row["world_id"] in eligible_worlds
            and row["probe_kind"] == "fact"
            and row["query_alias_index"] == 0
            and (
                row["field"] == "home"
                or row["field"] == "shape"
                and row["world_id"] in extra_shape_worlds
            )
        ):
            world = by_world[row["world_id"]]
            answer = world["objects"][world["alias_targets"][2]][row["field"]]
        predictions.append(
            {
                "id": row["id"],
                "generated_text": answer,
                "generated_ids": [100, 2],
                "step_endpoint_ids": [row["id"] + "::step00", row["id"] + "::step01"],
                "terminated": True,
                "truncated": False,
            }
        )
    ordered = sorted(
        eligible_worlds, key=lambda w: _hash([validation.SEED, "world", w])
    )
    cases = []
    for wid in ordered:
        case_id = min(
            (
                f"{wid}:swap{swap}:{focus}:recipient0"
                for swap in (0, 1)
                for focus in validation.FOCUS
            ),
            key=lambda cid: _hash([validation.SEED, "condition", cid]),
        )
        middle = case_id.split(":")
        group = (wid, int(middle[2][-1]), middle[3])
        cases.append(
            {
                "case_id": case_id,
                "world_id": wid,
                "correct_object": by_world[wid]["base_object_id"],
                "screen_field": "home",
                "donor_row_ids": {
                    f: lookup[group + ("fact", 1, f)]["id"] for f in validation.FIELDS
                },
                "recipient_row_ids": {
                    f: lookup[group + ("fact", 0, f)]["id"] for f in validation.FIELDS
                },
            }
        )
    model = {
        "layers": 3,
        "heads": 2,
        "head_dim": 4,
        "vocab_size": 1000,
        "model_id": "Qwen/Qwen3-4B-Instruct-2507",
        "revision": "fixture",
    }
    base = {
        "cases": cases[:8],
        "model": model,
        "label_token_ids": {
            f: {v: [100 + i] for i, v in enumerate(error_protocol.BANKS[f])}
            for f in validation.FIELDS
        },
        "eligibility": {"unused_eligible_world_ids": ordered[8:]},
    }
    capture_jobs, summaries = [], []
    for case in cases[:8]:
        for role in ("Dcolor", "Dscreen", "Rcolor", "Rscreen"):
            identity = f"capture::{case['case_id']}::{role}"
            masses = [[[0.1] * 4 for _ in range(2)] for _ in range(3)]
            masses[1][0]["ABCD".index(case["correct_object"])] = 0.4
            capture_jobs.append(
                {"endpoint_id": identity, "case_id": case["case_id"], "role": role}
            )
            summaries.append({"endpoint_id": identity, "record_masses": masses})
    heads = [
        {
            "layer_index": 1,
            "head_index": 0,
            "newly_correct": 4,
            "mean_gains": {"Dcolor": 1.0},
            "primary_minus_LOOWmean_gain": 0.5,
            "passed": True,
        },
        {
            "layer_index": 2,
            "head_index": 1,
            "newly_correct": 3,
            "mean_gains": {"Dcolor": 0.8},
            "primary_minus_LOOWmean_gain": 0.3,
            "passed": True,
        },
    ]
    assessment = {"head_results": heads, "passed": True}
    return {
        "base_plan": base,
        "rows": rows,
        "worlds": worlds,
        "predictions": predictions,
        "behavioral_rescore": {
            "exact_match": True,
            "confirmation_calls": 0,
            "causal_calls": 0,
            "native_steps": 6144,
        },
        "discovery_selected_plan": {"cases": cases[:8], "capture_jobs": capture_jobs},
        "discovery_result": {"assessment": assessment, "native_forward_count": 112},
        "discovery_rescore": {
            "exact_match": True,
            "confirmation_calls": 0,
            "native_forward_count": 112,
            "assessment": copy.deepcopy(assessment),
        },
        "discovery_capture_summaries": summaries,
        "provenance": {"behavior": "a" * 64, "routing": "b" * 64},
    }


def captures(planned):
    result = []
    for job in planned["natural_jobs"]:
        index = validation.FIELDS.index(job["field"])
        result.append(
            {
                "endpoint_id": job["endpoint_id"],
                "record_masses": [0.1 + 0.01 * index, 0.2, 0.1, 0.1],
                "argmax_correct": job["baseline_correct"],
                "record_value_hashes": {
                    str(l): _hash([job["case_id"], l]) for l in range(3)
                },
            }
        )
    return result


def test_same_old_hash_cohort_one_head_all_fields_and_bounded_endpoints():
    inputs = fixture()
    planned = validation.plan(inputs)
    assert planned == validation.plan(inputs)
    assert planned["selected_head"] == {"layer_index": 1, "head_index": 0}
    assert [c["world_id"] for c in planned["cases"]] == inputs["base_plan"][
        "eligibility"
    ]["unused_eligible_world_ids"]
    assert len(planned["natural_jobs"]) == 36 and len(planned["generation_jobs"]) == 108
    assert (
        planned["baseline"]["recipient_wrong"] == 8
        and planned["baseline"]["recipient_protected"] == 10
    )
    assert planned["storage"]["max_native_endpoints"] == 1008
    assert planned["storage"]["estimated_bytes"] < 1_000_000_000
    for job in planned["generation_jobs"]:
        assert len(job["greedy_endpoint_ids"]) == 8
        assert job["original_query_position"] == 5
        assert job["destination_role"] == (
            "donor"
            if job["condition"] in ("reverseRscreen", "cyclicDcolor")
            else "recipient"
        )
    assert planned["execution_authorized"] is False


def test_gate_audit_or_old_cohort_changes_are_not_rescued():
    for mutation in ("audit", "cohort", "native", "budget"):
        inputs = fixture()
        if mutation == "audit":
            inputs["discovery_rescore"]["exact_match"] = False
        elif mutation == "cohort":
            inputs["base_plan"]["eligibility"]["unused_eligible_world_ids"].reverse()
        elif mutation == "native":
            inputs["behavioral_rescore"]["native_steps"] = 1
        else:
            inputs["base_plan"]["model"]["vocab_size"] = 10_000_000
        with pytest.raises(ValueError):
            validation.plan(inputs)


def test_prospective_head_tiebreak_uses_gain_before_repair_count():
    inputs = fixture()
    other = inputs["discovery_result"]["assessment"]["head_results"][1]
    other["primary_minus_LOOWmean_gain"] = 0.5
    other["mean_gains"]["Dcolor"] = 1.1
    other["newly_correct"] = 2
    inputs["discovery_rescore"]["assessment"] = copy.deepcopy(
        inputs["discovery_result"]["assessment"]
    )
    assert validation.plan(inputs)["selected_head"] == {
        "layer_index": 2,
        "head_index": 1,
    }


def test_self_trajectory_object_matches_old_answer_not_always_task_truth():
    planned = validation.plan(fixture())
    wrong_self = [
        j
        for j in planned["generation_jobs"]
        if j["condition"] == "selfRfield" and not j["protected_before"]
    ]
    assert len(wrong_self) == 8
    for job in wrong_self:
        assert job["predicted_object"] is not None
        assert validation.normalize(
            job["object_answers"][job["predicted_object"]]
        ) == validation.normalize(job["predicted_answer"])
        assert validation.normalize(job["predicted_answer"]) != validation.normalize(
            job["task_answer"]
        )


def test_profiles_are_frozen_across_fields_and_cyclic_targets_clean_donor():
    planned = validation.plan(fixture())
    selected = validation.select_sources(planned, captures(planned))
    assert len(selected["generation_jobs"]) == 108
    for case in planned["cases"]:
        jobs = [
            j for j in selected["generation_jobs"] if j["case_id"] == case["case_id"]
        ]
        for condition in (
            "primaryDcolor",
            "Rcolor",
            "reverseRscreen",
            "frozenDiscoveryMean",
            "cyclicDcolor",
        ):
            profiles = [
                j["desired_profile"] for j in jobs if j["condition"] == condition
            ]
            assert profiles[0] == profiles[1] == profiles[2]
        primary = next(
            j["desired_profile"] for j in jobs if j["condition"] == "primaryDcolor"
        )
        cyclic = next(
            j["desired_profile"] for j in jobs if j["condition"] == "cyclicDcolor"
        )
        assert cyclic == [primary[3], primary[0], primary[1], primary[2]]
        selfs = [j["desired_profile"] for j in jobs if j["condition"] == "selfRfield"]
        assert selfs[0] != selfs[1] != selfs[2]


def test_any_unsupported_condition_stops_all_greedy_without_replacement():
    planned = validation.plan(fixture())
    natural = captures(planned)
    natural[0]["record_masses"][0] = 0.0
    selected = validation.select_sources(planned, natural)
    assert selected["generation_jobs"] == []
    assert len(selected["all_generation_jobs"]) == 108
    assert selected["unsupported_conditions"]
    assert selected["status"] == "stopped_unsupported_sources"
    result = validation.assess(selected, [])
    assert (
        result["by_condition"]["primaryDcolor"]["unconditional_task_accuracy"] is None
    )
    assert result["by_condition"]["primaryDcolor"][
        "unconditional_task_accuracy_bounds"
    ] == [0, 1]


def test_all_natural_record_values_and_native_eligibility_are_verified():
    planned = validation.plan(fixture())
    for mutation in ("hash", "correctness", "missing"):
        natural = captures(planned)
        if mutation == "hash":
            natural[1]["record_value_hashes"]["0"] = "c" * 64
        elif mutation == "correctness":
            natural[0]["argmax_correct"] = not natural[0]["argmax_correct"]
        else:
            natural.pop()
        with pytest.raises(ValueError):
            validation.select_sources(planned, natural)


def test_prospective_practical_thresholds_are_separate_and_reject_D_only_default():
    values = {
        "primary_repaired": 4,
        "primary_repair_worlds": 3,
        "protected_damaged": 1,
        "mean_repaired": 2,
        "reverse_hits": 9,
        "reverse_worlds": 3,
        "cyclic_hits": 9,
        "cyclic_worlds": 3,
        "cyclic_success_object_count": 2,
    }
    assert (
        validation.practical_flags(values, complete=True)["strong_candidate_passed"]
        is True
    )
    for key, value in [
        ("primary_repaired", 3),
        ("primary_repair_worlds", 2),
        ("protected_damaged", 2),
        ("mean_repaired", 3),
        ("reverse_hits", 8),
        ("cyclic_hits", 8),
        ("cyclic_success_object_count", 1),
    ]:
        assert (
            validation.practical_flags({**values, key: value}, complete=True)[
                "strong_candidate_passed"
            ]
            is False
        )
    assert (
        validation.practical_flags(values, complete=False)["strong_candidate_passed"]
        is None
    )


def test_complete_answer_scoring_and_specific_steering_not_generic_damage():
    planned = validation.select_sources(
        validation.plan(fixture()), captures(validation.plan(fixture()))
    )
    generations = [
        {
            "id": j["id"],
            "generated_text": j["predicted_answer"],
            "generated_ids": j["baseline_generated_ids"]
            if j["condition"] == "selfRfield"
            else [100, 2],
            "terminated": True,
            "truncated": False,
        }
        for j in planned["generation_jobs"]
    ]
    assessed = validation.assess(planned, generations)
    assert assessed["by_condition"]["primaryDcolor"]["task_correct"] == 18
    assert assessed["by_condition"]["primaryDcolor"]["repaired_initial_errors"] == 8
    assert assessed["by_condition"]["reverseRscreen"]["predicted_target_hits"] == 18
    assert assessed["by_condition"]["cyclicDcolor"]["predicted_target_hits"] == 18
    assert assessed["confirmation_calls_authorized"] is False
    broken = copy.deepcopy(generations)
    job = next(
        j for j in planned["generation_jobs"] if j["condition"] == "reverseRscreen"
    )
    output = next(g for g in broken if g["id"] == job["id"])
    output["generated_text"] += "\nExplanation."
    assert (
        validation.assess(planned, broken)["by_condition"]["reverseRscreen"][
            "predicted_target_hits"
        ]
        == 17
    )
