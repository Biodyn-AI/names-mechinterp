"""Model-free tests precede the new, development-only causal planner."""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "error_reference_study_20261114")
)
import causal_plan
import error_analysis
import error_protocol


def _inputs():
    study = error_protocol.build_study()
    worlds = [w for w in study["worlds"] if w["split"] == "development"]
    rows = [r for r in study["rows"] if r["split"] == "development"]
    labels = {
        field: {value: [100 + index] for index, value in enumerate(bank)}
        for field, bank in error_protocol.BANKS.items()
        if field in error_protocol.FIELDS
    }
    by_world = {w["world_id"]: w for w in worlds}
    predictions = []
    for row in rows:
        # Synthetic token geometry is deliberately not a real tokenizer. The
        # planner consumes already verified token records; it runs no tokenizer.
        marker_start, marker_end = row.get("alias_end_span", [0, 0])
        row.update(
            input_ids=[10, 20 + (row["query_alias_index"] or 0), 30, 40, 50, 99],
            alias_end_position=3,
            rendered_prompt=row["text"],
            offset_mapping=[
                [0, 1],
                [1, 2],
                [marker_start, marker_end - 1],
                [marker_end - 1, row["field_start"]],
                [row["field_start"], row["field_start"] + 1],
                [0, 0],
            ],
        )
        answer = row["expected_answer"]
        if row["probe_kind"] == "fact" and row["query_alias_index"] == 0:
            world = by_world[row["world_id"]]
            answer = world["objects"][world["alias_targets"][2]][row["field"]]
        predictions.append(
            {
                "id": row["id"],
                "generated_text": answer,
                "generated_ids": [100, 2],
                "terminated": True,
                "truncated": False,
            }
        )
    return {
        "model": {
            "model_id": "fixture",
            "revision": "fixed",
            "vocab_size": 1000,
            "layers": 6,
            "heads": 4,
            "head_dim": 8,
            "o_proj_in_features": 32,
        },
        "rows": rows,
        "worlds": worlds,
        "predictions": predictions,
        "behavioral_result": error_analysis.score(rows, predictions, worlds),
        "custody": {
            "completed_rows": 3072,
            "physical_forward_count": 6144,
            "registration_sha256": "a" * 64,
        },
        "independent_rescore": {
            "exact_match": True,
            "native_steps": 6144,
            "confirmation_calls": 0,
            "causal_calls": 0,
        },
        "provenance": {
            name: char * 64
            for name, char in [
                ("registration_sha256", "a"),
                ("custody_sha256", "b"),
                ("rescore_sha256", "c"),
            ]
        },
        "label_token_ids": labels,
    }


def test_plan_is_reproducible_prefield_and_discovery_only():
    inputs = _inputs()
    planned = causal_plan.plan(inputs)
    assert planned == causal_plan.plan(inputs)
    assert len(planned["cases"]) == 8
    assert len({c["world_id"] for c in planned["cases"]}) == 8
    assert planned["eligibility"]["eligible_worlds"] == 32
    assert len(planned["layer_jobs"]) == 8 * 5
    assert len(planned["layer_self_jobs"]) == 8 * 5
    assert len({j["endpoint_id"] for j in planned["layer_self_jobs"]}) == 8 * 5
    assert {j["case_id"] for j in planned["layer_self_jobs"]} == {
        c["case_id"] for c in planned["cases"]
    }
    assert planned["storage"]["reserved_self_oracles"] == 8 * 5 + 2 * 4
    assert len(planned["clean_jobs"]) == 8
    assert max(j["layer_index"] for j in planned["layer_jobs"]) == 4
    assert planned["execution_authorized"] is False
    assert planned["confirmation_calls_authorized"] is False
    assert planned["storage"]["estimated_bytes"] <= 1_000_000_000
    assert all(
        len(c["donor_prefix_ids"]) == c["alias_end_position"] + 1
        for c in planned["cases"]
    )


def test_failed_or_unverified_behavior_is_rejected():
    inputs = _inputs()
    for field, value in [
        ("exact_match", False),
        ("native_steps", 1),
        ("confirmation_calls", 1),
        ("causal_calls", 1),
    ]:
        broken = copy.deepcopy(inputs)
        broken["independent_rescore"][field] = value
        with pytest.raises(ValueError):
            causal_plan.plan(broken)
    broken = copy.deepcopy(inputs)
    broken["behavioral_result"]["feasibility"]["passed"] = False
    with pytest.raises(ValueError):
        causal_plan.plan(broken)


def test_unbound_registration_geometry_and_raw_budget_are_rejected():
    for mutation in ("registration", "geometry", "raw_budget"):
        inputs = _inputs()
        if mutation == "registration":
            inputs["custody"]["registration_sha256"] = "d" * 64
        elif mutation == "geometry":
            inputs["model"]["o_proj_in_features"] = 64
        else:
            inputs["model"]["vocab_size"] = 10_000_000
        with pytest.raises(ValueError):
            causal_plan.plan(inputs)


def test_stronger_donor_requirement_is_reported_not_rescued():
    inputs = _inputs()
    for row, prediction in zip(inputs["rows"], inputs["predictions"]):
        if row["probe_kind"] == "identity" and row["query_alias_index"] == 1:
            prediction["generated_text"] = "not an object"
    inputs["behavioral_result"] = error_analysis.score(
        inputs["rows"], inputs["predictions"], inputs["worlds"]
    )
    assert inputs["behavioral_result"]["feasibility"]["passed"] is True
    planned = causal_plan.plan(inputs)
    assert planned["status"] == "insufficient_eligible_worlds"
    assert planned["eligibility"]["eligible_worlds"] == 0
    assert planned["eligibility"]["rejections"]["donor_not_all_correct"] > 0
    assert planned["layer_jobs"] == []


def test_future_field_or_geometry_leak_is_rejected():
    for mutation in ("prefix", "position", "offset"):
        inputs = _inputs()
        row = next(
            r
            for r in inputs["rows"]
            if r["probe_kind"] == "fact"
            and r["query_alias_index"] == 1
            and r["field"] == "shape"
        )
        if mutation == "prefix":
            row["input_ids"][0] = 999
        elif mutation == "position":
            row["alias_end_position"] = 4
        else:
            row["offset_mapping"][3][1] += 1
        with pytest.raises(ValueError):
            causal_plan.plan(inputs)


def test_ambiguous_noninteger_or_missing_label_tokens_fail_closed():
    for mutation in ("empty", "overlap", "noninteger", "out_of_vocabulary"):
        inputs = _inputs()
        bank = inputs["label_token_ids"]["color"]
        names = list(bank)
        if mutation == "empty":
            bank[names[0]] = []
        elif mutation == "overlap":
            bank[names[0]] = bank[names[1]]
        elif mutation == "noninteger":
            bank[names[0]] = [True]
        else:
            bank[names[0]] = [1000]
        with pytest.raises(ValueError):
            causal_plan.plan(inputs)


def test_screen_refinements_require_complete_endpoints_and_keep_bounds():
    planned = causal_plan.plan(_inputs())
    measured = [
        {"endpoint_id": j["endpoint_id"], "correct_minus_wrong_gain": j["layer_index"]}
        for j in planned["layer_jobs"]
    ]
    heads = causal_plan.select_layers(planned, measured)
    assert heads["selected_layers"] == [4, 3]
    assert len(heads["head_jobs"]) == 8 * 2 * 4
    assert len(heads["head_self_jobs"]) == 2 * 4
    with pytest.raises(ValueError):
        causal_plan.select_layers(planned, measured[:-1])
    scores = [
        {
            "endpoint_id": j["endpoint_id"],
            "correct_minus_wrong_gain": j["head_index"] + 1,
        }
        for j in heads["head_jobs"]
    ]
    final = causal_plan.select_heads(heads, scores)
    assert len(final["selected_heads"]) == 2
    assert final["execution_authorized"] is False
    assert final["confirmation_calls_authorized"] is False
    with pytest.raises(ValueError):
        causal_plan.select_heads(heads, scores + scores[:1])
    scores[0]["correct_minus_wrong_gain"] = float("nan")
    with pytest.raises(ValueError):
        causal_plan.select_heads(heads, scores)


def test_nonpositive_screening_stops_and_only_positive_candidates_continue():
    planned = causal_plan.plan(_inputs())
    layer_scores = [
        {"endpoint_id": j["endpoint_id"], "correct_minus_wrong_gain": 0.0}
        for j in planned["layer_jobs"]
    ]
    stopped = causal_plan.select_layers(planned, layer_scores)
    assert stopped["status"] == "stopped_no_positive_layer_gain"
    assert stopped["selected_layers"] == []
    assert stopped["head_jobs"] == [] and stopped["head_self_jobs"] == []
    for job, measured in zip(planned["layer_jobs"], layer_scores):
        measured["correct_minus_wrong_gain"] = 1.0 if job["layer_index"] == 2 else -1.0
    heads = causal_plan.select_layers(planned, layer_scores)
    assert heads["selected_layers"] == [2]
    assert len(heads["head_jobs"]) == 8 * 4
    head_scores = [
        {"endpoint_id": j["endpoint_id"], "correct_minus_wrong_gain": -0.5}
        for j in heads["head_jobs"]
    ]
    stopped = causal_plan.select_heads(heads, head_scores)
    assert stopped["status"] == "stopped_no_positive_head_gain"
    assert stopped["selected_heads"] == []
    for job, measured in zip(heads["head_jobs"], head_scores):
        measured["correct_minus_wrong_gain"] = 0.5 if job["head_index"] == 1 else 0.0
    selected = causal_plan.select_heads(heads, head_scores)
    assert selected["selected_heads"] == [{"layer_index": 2, "head_index": 1}]


def test_mixed_field_errors_do_not_define_a_shared_wrong_record():
    inputs = _inputs()
    worlds = {w["world_id"]: w for w in inputs["worlds"]}
    for row, prediction in zip(inputs["rows"], inputs["predictions"]):
        if (
            row["probe_kind"] == "fact"
            and row["query_alias_index"] == 0
            and row["field"] == "shape"
        ):
            world = worlds[row["world_id"]]
            prediction["generated_text"] = world["objects"][
                world["unaliased_object_id"]
            ]["shape"]
    inputs["behavioral_result"] = error_analysis.score(
        inputs["rows"], inputs["predictions"], inputs["worlds"]
    )
    planned = causal_plan.plan(inputs)
    for case in planned["cases"]:
        assert case["observed_wrong_objects_coherent"] is False
        assert (
            case["reverse_predicted_object"]
            == case["wrong_object_by_field"][case["screen_field"]]
        )
        assert case["reverse_predicted_answers"] == {
            field: worlds[case["world_id"]]["objects"][
                case["reverse_predicted_object"]
            ][field]
            for field in error_protocol.FIELDS
        }
