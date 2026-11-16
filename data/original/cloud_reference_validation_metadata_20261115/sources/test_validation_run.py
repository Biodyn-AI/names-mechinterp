"""Model-free fixed-query generation and source-authentication contracts."""

import copy
import hashlib
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import validation_run


class Reader:
    def __init__(self, arrays):
        self.arrays = arrays
        self.index = {"endpoints": {identity: {} for identity in arrays}}

    def read(self, identity):
        return self.arrays[identity]


def fixture():
    job = {
        "id": "case:condition:color",
        "condition": "selfRfield",
        "original_query_position": 5,
        "operator_endpoint_id": "operator",
        "greedy_endpoint_ids": [f"g:{i}" for i in range(8)],
        "task_answer": "red",
        "predicted_answer": "red",
    }
    delta, reference = torch.zeros(4), torch.ones(4)
    metadata = {
        "delta": delta,
        "reference_natural_head": reference,
        "query_position": 5,
        "layer_index": 2,
        "head_index": 1,
        "head_dim": 4,
        "attention_width": 8,
        "drift_tolerance": 2e-5,
        "zero_delta": True,
        "delta_norm": 0.0,
        "observations": [
            {
                "sequence_length": 6 + i,
                "query_position": 5,
                "natural_head": reference.clone(),
                "max_abs_drift": 0.0,
            }
            for i in range(2)
        ],
    }
    prediction = {
        "id": job["id"],
        "task_answer": "red",
        "predicted_target_answer": "red",
        "generated_ids": [0, 3],
        "generated_text": "red",
        "terminated": True,
        "truncated": False,
        "stop_reason": "eos",
        "step_endpoint_ids": ["g:0", "g:1"],
        "operator_endpoint_id": "operator",
    }
    arrays = {
        "operator": torch.tensor([3.0, 0.0, 0.0, 0.0]),
        "g:0": torch.tensor([3.0, 0.0, 0.0, 0.0]),
        "g:1": torch.tensor([0.0, 0.0, 0.0, 3.0]),
    }
    old_prediction = {
        "generated_ids": [0, 3],
        "generated_text": "red",
        "terminated": True,
        "truncated": False,
        "stop_reason": "eos",
        "step_endpoint_ids": ["old:0", "old:1"],
    }
    old_arrays = {"old:0": arrays["g:0"], "old:1": arrays["g:1"]}
    operator = {
        "intervention": {"delta_head_output": delta, "old_head_output": reference}
    }
    planned = {
        "selected_head": {"layer_index": 2, "head_index": 1},
        "model": {"heads": 2, "head_dim": 4},
        "numeric": {"max_query_head_drift": 2e-5},
    }

    class Tokenizer:
        def decode(self, ids, *, skip_special_tokens):
            assert not skip_special_tokens
            return "".join(
                {0: "red", 1: "<special>", 2: " ", 3: "<eos>"}[i] for i in ids
            )

    behavior = SimpleNamespace(
        decode_answer=lambda tokenizer, ids, eos: tokenizer.decode(
            ids[:-1] if ids and ids[-1] in eos else ids, skip_special_tokens=False
        )
    )
    return (
        job,
        prediction,
        metadata,
        operator,
        Reader(arrays),
        old_prediction,
        Reader(old_arrays),
        planned,
        behavior,
        Tokenizer(),
    )


def test_fixed_query_delta_generation_replays_all_steps_and_whole_self_answer():
    args = fixture()
    assert validation_run.audit_generation(*args, eos={3})[
        "self_whole_trajectory_equal"
    ]


@pytest.mark.parametrize(
    "mutation",
    ["position", "drift", "delta", "first_native", "self_later_native", "decode"],
)
def test_query_position_drift_delta_native_and_decode_mismatches_fail_closed(mutation):
    values = list(fixture())
    if mutation == "position":
        values[2]["observations"][1]["query_position"] = 6
    elif mutation == "drift":
        values[2]["observations"][1]["natural_head"][0] += 3e-5
        values[2]["observations"][1]["max_abs_drift"] = float(
            (values[2]["observations"][1]["natural_head"] - 1).abs().max()
        )
    elif mutation == "delta":
        values[2]["delta"][0] = 1
        values[3] = copy.deepcopy(values[3])
        values[3]["intervention"]["delta_head_output"] = torch.zeros(4)
    elif mutation == "first_native":
        values[4].arrays["operator"] = values[4].arrays["operator"] + 1e-7
    elif mutation == "self_later_native":
        values[6].arrays["old:1"] = values[6].arrays["old:1"] + 1e-7
    else:
        values[1]["generated_text"] = "red explanation"
    with pytest.raises(ValueError):
        validation_run.audit_generation(*values, eos={3})


def test_routing_audit_receipt_anchors_every_source_and_tensor_artifact():
    anchors = {key: f"{i:064x}" for i, key in enumerate(validation_run.ROUTING_ANCHORS)}
    receipt = {
        **anchors,
        "exact_match": True,
        "confirmation_calls": 0,
        "assessment": {"passed": True},
        "native_forward_count": 112,
    }
    validation_run.verify_routing_receipt(
        receipt, anchors, {"assessment": {"passed": True}, "native_forward_count": 112}
    )
    for key in anchors:
        with pytest.raises(ValueError):
            validation_run.verify_routing_receipt(
                {**receipt, key: "f" * 64},
                anchors,
                {"assessment": {"passed": True}, "native_forward_count": 112},
            )


def full_fixture():
    torch.set_num_threads(1)
    sys.path.insert(
        0, str(Path(__file__).resolve().parents[1] / "error_reference_routing_20261115")
    )
    routing = importlib.import_module("routing_run")
    mass_hooks = importlib.import_module("record_routing_hooks")
    planner = importlib.import_module("validation_plan")
    inputs = importlib.import_module("test_validation_plan").fixture()
    fields = ("color", "shape", "home")
    labels = inputs["base_plan"]["label_token_ids"]
    words = {}
    for field_index, field in enumerate(fields):
        for index, word in enumerate(labels[field]):
            labels[field][word] = [100 * (field_index + 1) + index]
            words[labels[field][word][0]] = word
    rows = {r["id"]: r for r in inputs["rows"]}
    for old in inputs["predictions"]:
        row = rows[old["id"]]
        token = (
            labels[row["field"]][old["generated_text"]][0]
            if row["probe_kind"] == "fact"
            else 1
        )
        old.update(generated_ids=[token, 999], stop_reason="eos")
    planned = planner.plan(inputs)
    raw, arrays, old_arrays = {}, {}, {}
    old_predictions = {p["id"]: p for p in inputs["predictions"]}
    layer_index = planned["selected_head"]["layer_index"]
    head_index = planned["selected_head"]["head_index"]
    for job in planned["natural_jobs"]:
        masses = [0.1 + 0.01 * fields.index(job["field"]), 0.2, 0.1, 0.1]
        weights = torch.full((6,), (1 - sum(masses)) / 2, dtype=torch.float32)
        for index, obj in enumerate("ABCD"):
            weights[job["record_regions"][obj][0]] = masses[index]
        layers = {
            layer: {
                "attention_rows": torch.full((2, 6), 1 / 6, dtype=torch.float32),
                "record_value_sha256": hashlib.sha256(
                    f"{job['case_id']}:{layer}".encode()
                ).hexdigest(),
                "record_positions": [0, 1, 2, 3],
                "head_dim": 4,
                "num_key_value_heads": 1,
            }
            for layer in range(3)
        }
        layers[layer_index]["attention_rows"][head_index] = weights
        raw[job["endpoint_id"]] = {"layers": layers, "intervention": None}
        old = old_predictions[job["row_id"]]
        for step, token in enumerate(old["generated_ids"]):
            native = torch.zeros(1000, dtype=torch.float32)
            native[token] = 3
            old_arrays[old["step_endpoint_ids"][step]] = native
        arrays[job["endpoint_id"]] = old_arrays[old["step_endpoint_ids"][0]].clone()
    summaries = validation_run.natural_summaries(
        routing, planned, inputs, raw, Reader(arrays)
    )
    selected = planner.select_sources(planned, summaries)
    natural_jobs = {j["endpoint_id"]: j for j in planned["natural_jobs"]}
    operators, traces, generations = {}, {}, []
    for job in selected["generation_jobs"]:
        baseline = natural_jobs[job["natural_endpoint_id"]]
        recipient = raw[baseline["endpoint_id"]]
        proxy = validation_run.proxy_operator_job(job, planned)
        profile = routing.effective_profile(
            proxy, recipient, baseline["record_regions"], mass_hooks
        )
        weights = recipient["layers"][layer_index]["attention_rows"][head_index]
        metadata = mass_hooks.route_record_weights(
            weights,
            torch.arange(24, dtype=torch.float32).reshape(6, 4),
            baseline["record_regions"],
            profile,
        )
        metadata.update(
            layer_index=layer_index,
            head_index=head_index,
            old_head_output=torch.zeros(4),
            new_head_output=metadata["delta_head_output"].clone(),
        )
        captured = copy.deepcopy(recipient)
        captured["intervention"] = metadata
        operators[job["id"]] = captured
        old = old_predictions[job["row_id"]]
        answer = (
            old["generated_text"]
            if job["condition"] in ("selfRfield", "frozenDiscoveryMean")
            else job["predicted_answer"]
        )
        token = labels[job["field"]][answer][0]
        native = torch.zeros(1000, dtype=torch.float32)
        native[token] = 3
        end = torch.zeros(1000, dtype=torch.float32)
        end[999] = 3
        arrays[job["operator_endpoint_id"]] = native
        arrays[job["greedy_endpoint_ids"][0]] = native.clone()
        arrays[job["greedy_endpoint_ids"][1]] = end
        generations.append(
            {
                "id": job["id"],
                "task_answer": job["task_answer"],
                "predicted_target_answer": job["predicted_answer"],
                "generated_ids": [token, 999],
                "generated_text": answer,
                "terminated": True,
                "truncated": False,
                "stop_reason": "eos",
                "step_endpoint_ids": job["greedy_endpoint_ids"][:2],
                "operator_endpoint_id": job["operator_endpoint_id"],
            }
        )
        delta = metadata["delta_head_output"]
        traces[job["id"]] = {
            "delta": delta,
            "reference_natural_head": torch.zeros(4),
            "query_position": 5,
            "layer_index": layer_index,
            "head_index": head_index,
            "head_dim": 4,
            "attention_width": 8,
            "drift_tolerance": 2e-5,
            "zero_delta": bool((delta == 0).all()),
            "delta_norm": float(torch.linalg.vector_norm(delta.double())),
            "observations": [
                {
                    "sequence_length": 6 + step,
                    "query_position": 5,
                    "natural_head": torch.zeros(4),
                    "max_abs_drift": 0.0,
                }
                for step in range(2)
            ],
        }

    class Tokenizer:
        def decode(self, ids, *, skip_special_tokens):
            assert not skip_special_tokens
            return "".join(words.get(token, "<eos>") for token in ids)

    behavior = SimpleNamespace(
        decode_answer=lambda tokenizer, ids, eos: tokenizer.decode(
            ids[:-1] if ids and ids[-1] in eos else ids, skip_special_tokens=False
        )
    )
    return (
        Reader(arrays),
        Reader(old_arrays),
        raw,
        operators,
        traces,
        generations,
        planned,
        inputs,
        routing,
        planner,
        mass_hooks,
        behavior,
        Tokenizer(),
        {999},
    )


def test_all_108_generations_replay_without_confusing_steering_with_literal_answers():
    values = full_fixture()
    _, selected, result = validation_run.evaluate(*values)
    assert len(selected["generation_jobs"]) == 108
    assert result["native_forward_count"] == 360
    assert result["operator_count"] == 108 and result["greedy_step_count"] == 216
    assert len(result["oracles"]) == 108
    assert sum(o["self_whole_trajectory_equal"] for o in result["oracles"]) == 18
    assert result["assessment"]["by_condition"]["reverseRscreen"]["task_correct"] == 0
    assert (
        result["assessment"]["by_condition"]["reverseRscreen"]["predicted_target_hits"]
        == 18
    )


def test_any_missing_support_stops_all_108_after_exactly_36_natural_captures():
    values = list(full_fixture())
    planned, raw = values[6], values[2]
    job = planned["natural_jobs"][0]
    head = planned["selected_head"]
    weights = raw[job["endpoint_id"]]["layers"][head["layer_index"]]["attention_rows"][
        head["head_index"]
    ]
    position = job["record_regions"]["A"][0]
    weights[4] += weights[position]
    weights[position] = 0
    natural_ids = {j["endpoint_id"] for j in planned["natural_jobs"]}
    values[0] = Reader({k: v for k, v in values[0].arrays.items() if k in natural_ids})
    values[3:6] = [{}, {}, []]
    _, selected, result = validation_run.evaluate(*values)
    assert selected["generation_jobs"] == []
    assert result["status"] == "stopped_unsupported_sources"
    assert result["native_forward_count"] == 36
    assert result["operator_count"] == result["greedy_step_count"] == 0
