"""Complete finite plan/native/delta replay, using only synthetic CPU tensors."""

import copy
import hashlib
from types import SimpleNamespace

import decomposition_run as runner
import pytest
import torch
from test_decomposition_math import fixture as operands


class Reader:
    def __init__(self, arrays):
        self.arrays = arrays
        self.index = {"endpoints": {key: {} for key in arrays}}

    def read(self, identity):
        return self.arrays[identity]


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return set(a) == set(b) and all(
            equal(value, b[key]) for key, value in a.items()
        )
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def fixture():
    r, d, color, regions = operands()
    wrong = {0, 1, 3, 4, 6, 9, 12, 15}
    rows, cases, natural_jobs, captures, old_natural, originals, predictions = (
        [],
        [],
        [],
        {},
        {},
        {},
        [],
    )
    old_colors = {}
    for index in range(6):
        cid = f"case{index}"
        case = {
            "case_id": cid,
            "world_id": f"world{index}",
            "donor_row_ids": {},
            "recipient_row_ids": {},
            "protected_correct_fields": [],
        }
        masses = torch.stack(
            [color["weights"][regions[obj]].double().sum() for obj in "ABCD"]
        )
        old_colors[cid] = (masses / masses.sum()).tolist()
        for field_index, field in enumerate(("color", "shape", "home")):
            if index * 3 + field_index not in wrong:
                case["protected_correct_fields"].append(field)
            for role in ("donor", "recipient"):
                rid = f"{cid}:{role}:{field}"
                case[f"{role}_row_ids"][field] = rid
                rows.append(
                    {"id": rid, "expected_answer": "red", "input_ids": [0] * 10}
                )
                nid = f"natural::{cid}::{role}::{field}"
                natural_jobs.append(
                    {
                        "endpoint_id": nid,
                        "case_id": cid,
                        "role": role,
                        "field": field,
                        "row_id": rid,
                        "record_regions": regions,
                        "original_query_position": 9,
                    }
                )
                selected = (
                    r if role == "recipient" else color if field == "color" else d
                )
                item = copy.deepcopy(selected)
                record_hash = hashlib.sha256(
                    item["values"][:8].contiguous().numpy().tobytes()
                ).hexdigest()
                item.update(
                    record_V_sha256=record_hash,
                    record_positions=list(range(8)),
                    router_head_output=item["head_output"].clone(),
                    router_delta=torch.zeros(4),
                )
                item["trace"] = {
                    "zero_delta": True,
                    "query_position": 9,
                    "observations": [
                        {
                            "sequence_length": 10,
                            "natural_head": item["head_output"].clone(),
                            "max_abs_drift": 0.0,
                        }
                    ],
                }
                captures[nid] = item
                old_rows = torch.zeros((32, 10))
                old_rows[0] = item["weights"]
                old_natural[nid] = {
                    "layers": {
                        28: {
                            "attention_rows": old_rows,
                            "record_value_sha256": record_hash,
                        }
                    }
                }
                correct = role == "donor" or index * 3 + field_index not in wrong
                token = 0 if correct else 1
                predicted = {
                    "id": rid,
                    "generated_ids": [token, 3],
                    "generated_text": "red" if correct else "blue",
                    "terminated": True,
                    "truncated": False,
                    "stop_reason": "eos",
                    "step_endpoint_ids": [f"{rid}::step00", f"{rid}::step01"],
                }
                predictions.append(predicted)
                for step, tid in enumerate(predicted["generated_ids"]):
                    native = torch.zeros(4)
                    native[tid] = 3
                    originals[predicted["step_endpoint_ids"][step]] = native
        cases.append(case)
    old_plan = {
        "selected_head": runner.HEAD,
        "cases": cases,
        "natural_jobs": natural_jobs,
        "baseline": {
            "recipient_wrong": 8,
            "recipient_protected": 10,
            "recipient_fields": 18,
            "donor_correct": 18,
        },
        "model": {"vocab_size": 4, "heads": 32, "head_dim": 4, "layers": 36},
    }
    inputs = {
        "old_plan": old_plan,
        "rows": rows,
        "old_color_profiles": old_colors,
        "predictions": predictions,
        "old_validation_result": {"assessment": {"strong_candidate_passed": False}},
    }
    planned = runner.plan(inputs)
    computed = runner.rebuild_kernels(planned, inputs, captures, old_natural)
    old_predictions = {p["id"]: p for p in predictions}
    arrays, old_arrays, old_generations, traces, generations = {}, {}, {}, {}, []
    for job in natural_jobs:
        arrays[job["endpoint_id"]] = originals[
            old_predictions[job["row_id"]]["step_endpoint_ids"][0]
        ].clone()
    for job in planned["generation_jobs"]:
        old = old_predictions[job["row_id"]]
        correct = (
            job["condition"] in ("sameFieldProfileWithin", "sameFieldFullHead")
            or old["generated_text"] == "red"
        )
        token = 0 if correct else 1
        prediction = {
            "id": job["id"],
            "expected_answer": "red",
            "generated_ids": [token, 3],
            "generated_text": "red" if correct else "blue",
            "terminated": True,
            "truncated": False,
            "stop_reason": "eos",
            "step_endpoint_ids": job["greedy_endpoint_ids"][:2],
        }
        generations.append(prediction)
        for step, tid in enumerate(prediction["generated_ids"]):
            native = torch.zeros(4)
            native[tid] = 3
            arrays[prediction["step_endpoint_ids"][step]] = native
        if job["condition"] == "oldDcolorMass":
            oid = f"{job['case_id']}::primaryDcolor::{job['field']}"
            old_gen = {**old, "step_endpoint_ids": [oid + "::step0", oid + "::step1"]}
            old_generations[oid] = old_gen
            for step, tid in enumerate(old_gen["generated_ids"]):
                native = torch.zeros(4)
                native[tid] = 3
                old_arrays[old_gen["step_endpoint_ids"][step]] = native
        delta = computed[f"{job['case_id']}::{job['field']}"]["conditions"][
            job["condition"]
        ]["delta"]
        reference = r["head_output"]
        traces[job["id"]] = {
            "delta": delta,
            "reference_natural_head": reference,
            "layer_index": 28,
            "head_index": 0,
            "query_position": 9,
            "head_dim": 4,
            "attention_width": 128,
            "drift_tolerance": 2e-5,
            "delta_norm": float(torch.linalg.vector_norm(delta.double())),
            "zero_delta": bool((delta == 0).all()),
            "observations": [
                {
                    "sequence_length": 10 + step,
                    "query_position": 9,
                    "natural_head": reference.clone(),
                    "max_abs_drift": 0.0,
                }
                for step in range(2)
            ],
        }

    class Tokenizer:
        def decode(self, ids, *, skip_special_tokens):
            assert not skip_special_tokens
            return "".join({0: "red", 1: "blue", 3: "<eos>"}[i] for i in ids)

    behavior = SimpleNamespace(
        decode_answer=lambda tokenizer, ids, eos: tokenizer.decode(
            ids[:-1] if ids and ids[-1] in eos else ids, skip_special_tokens=False
        )
    )
    return (
        Reader(arrays),
        Reader(originals),
        Reader(old_arrays),
        captures,
        computed,
        traces,
        generations,
        planned,
        inputs,
        old_natural,
        old_generations,
        behavior,
        Tokenizer(),
        {3},
        equal,
    )


def test_complete162_job_native_kernel_and_literal_grading_replay():
    result = runner.evaluate(*fixture())
    assert result["native_forward_count"] == 360
    assert result["natural_count"] == 36 and result["generations"] == 162
    assert result["assessment"]["conditions"]["self"]["repaired"] == 0
    assert result["assessment"]["conditions"]["oldDcolorMass"]["repaired"] == 0
    assert result["assessment"]["conditions"]["sameFieldProfileWithin"][
        "diagnostic_passed"
    ]
    assert (
        result["old_validation_result_preserved"]["assessment"][
            "strong_candidate_passed"
        ]
        is False
    )
    assert result["automatic_followup_authorized"] is False


@pytest.mark.parametrize(
    "mutation", ["delta", "width", "dtype", "position", "native", "negative_control"]
)
def test_saved_delta_geometry_query_native_and_old_controls_fail_closed(mutation):
    values = list(fixture())
    job = values[7]["generation_jobs"][0]
    if mutation == "delta":
        values[5][job["id"]] = copy.deepcopy(values[5][job["id"]])
        values[5][job["id"]]["delta"][0] = 1
    elif mutation == "width":
        values[5][job["id"]]["attention_width"] = 4
    elif mutation == "dtype":
        values[5][job["id"]]["observations"][0]["natural_head"] = values[5][job["id"]][
            "observations"
        ][0]["natural_head"].double()
    elif mutation == "position":
        values[5][job["id"]]["observations"][1]["query_position"] = 10
    elif mutation == "native":
        values[0].arrays[job["greedy_endpoint_ids"][1]][2] = 1e-7
    else:
        old = next(iter(values[10].values()))
        values[2].arrays[old["step_endpoint_ids"][1]][2] = 1e-7
    with pytest.raises(ValueError):
        runner.evaluate(*values)
