"""Model-free evidence and replay contracts for the bounded routing screen."""

import copy
from types import SimpleNamespace

import pytest
import record_routing_hooks as hooks
import routing_plan
import routing_run
import torch
from test_routing_plan import _fixture


def raw_captures(planned):
    raw = {}
    for job in planned["capture_jobs"]:
        length = 6
        layers = {}
        for layer in range(planned["model"]["layers"]):
            layers[layer] = {
                "attention_rows": torch.full(
                    (2, length), 1 / length, dtype=torch.float32
                ),
                "record_value_sha256": f"{layer + 1:064x}",
                "record_positions": [0, 1, 2, 3],
                "head_dim": planned["model"]["head_dim"],
                "num_key_value_heads": 1,
            }
        raw[job["endpoint_id"]] = {"layers": layers, "intervention": None}
    return raw


def test_summaries_are_rebuilt_from_all_raw_attention_rows_and_hashes():
    planned = routing_plan.plan(_fixture())
    raw = raw_captures(planned)
    summaries = routing_run.capture_summaries(planned, raw)
    assert len(summaries) == 32
    assert summaries[0]["record_masses"][0][0] == pytest.approx([1 / 6] * 4)
    assert summaries[0]["record_value_hashes"]["0"] == f"{1:064x}"
    broken = copy.deepcopy(raw)
    first = next(iter(broken.values()))
    first["layers"][0]["attention_rows"][0, 0] = float("nan")
    with pytest.raises(ValueError):
        routing_run.capture_summaries(planned, broken)


def test_every_case_context_must_have_the_same_record_values():
    planned = routing_plan.plan(_fixture())
    raw = raw_captures(planned)
    first = planned["capture_jobs"][0]
    other = next(
        j
        for j in planned["capture_jobs"]
        if j["case_id"] == first["case_id"] and j["endpoint_id"] != first["endpoint_id"]
    )
    raw[other["endpoint_id"]]["layers"][0]["record_value_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="record value"):
        routing_run.capture_summaries(planned, raw)


def test_unnormalized_unselected_head_is_rejected_before_selection():
    planned = routing_plan.plan(_fixture())
    raw = raw_captures(planned)
    first = next(iter(raw.values()))
    # This non-selected layer/head still must be a complete probability row.
    first["layers"][2]["attention_rows"][1] *= 0.5
    with pytest.raises(ValueError, match="probability"):
        routing_run.capture_summaries(planned, raw)


def test_tensor_metadata_comparison_is_exact_and_recursive():
    record = {"tensor": torch.tensor([1.0, 2.0]), "nested": [1, {"s": "same"}]}
    assert routing_run.exact_equal(record, copy.deepcopy(record))
    changed = copy.deepcopy(record)
    changed["tensor"][0] += 1e-7
    assert not routing_run.exact_equal(record, changed)


def test_prior_receipt_requires_all_original_screen_anchors_and_counts():
    anchors = {name: str(i) * 64 for i, name in enumerate(routing_run.PRIOR_ANCHORS)}
    receipt = {
        **anchors,
        "exact_match": True,
        "confirmation_calls": 0,
        "physical_counts": {"native": 1144, "backbone_prefix": 16},
    }
    routing_run.verify_prior_receipt(receipt, anchors, receipt["physical_counts"])
    for key in anchors:
        changed = {**receipt, key: "f" * 64}
        with pytest.raises(ValueError):
            routing_run.verify_prior_receipt(
                changed, anchors, receipt["physical_counts"]
            )


def test_bindings_and_focus_prefix_must_match_not_only_printed_records():
    inputs = _fixture()
    planned = routing_plan.plan(inputs)
    for row in inputs["rows"]:
        row["input_ids"] = row["input_ids"][:4] + [99] + row["input_ids"][4:]
        row["marker_positions"] = {"[FOCUS_END]": [4]}
    routing_run.validate_shared_context(planned, inputs)
    identity = planned["capture_jobs"][0]["row_id"]
    row = next(row for row in inputs["rows"] if row["id"] == identity)
    row["input_ids"][4] = 98
    with pytest.raises(ValueError, match="bindings"):
        routing_run.validate_shared_context(planned, inputs)


class Reader:
    def __init__(self, arrays):
        self.arrays = arrays
        self.index = {"endpoints": {key: {} for key in arrays}}

    def read(self, identity):
        return self.arrays[identity]


def complete_fixture():
    torch.set_num_threads(1)
    inputs = _fixture()
    planned = routing_plan.plan(inputs)
    raw, arrays, original = raw_captures(planned), {}, {}
    cases = {case["case_id"]: case for case in planned["cases"]}
    for job in planned["capture_jobs"]:
        case = cases[job["case_id"]]
        weights = torch.tensor([0.05] * 4 + [0.25, 0.25], dtype=torch.float32)
        favored_object = (
            case["correct_object"]
            if job["role"] in ("Dcolor", "Dscreen")
            else case["wrong_object_by_field"][case["screen_field"]]
        )
        weights[job["record_regions"][favored_object][0]] = 0.35
        for layer in raw[job["endpoint_id"]]["layers"].values():
            layer["attention_rows"][:] = torch.tensor([0.1] * 4 + [0.3, 0.3])
        raw[job["endpoint_id"]]["layers"][0]["attention_rows"][0] = weights
        native = torch.zeros(1000, dtype=torch.float32)
        if job["role"] in ("Dcolor", "Dscreen", "Rcolor"):
            native[job["expected_label_token_ids"][0]] = 2
        else:
            native[planned["screen_label_token_ids"][case["case_id"]]["wrong"][0]] = 2
        arrays[job["endpoint_id"]] = native
        original[job["row_id"] + "::step00"] = native.clone()
    summaries = routing_run.capture_summaries(planned, raw, Reader(arrays))
    selected = routing_plan.select_heads(planned, summaries)
    assert len(selected["selected_heads"]) == 1
    recipients = {
        job["case_id"]: job
        for job in planned["capture_jobs"]
        if job["role"] == "Rscreen"
    }
    for job in selected["intervention_jobs"]:
        base = recipients[job["case_id"]]
        recipient = raw[base["endpoint_id"]]
        profile = routing_run.effective_profile(
            job, recipient, base["record_regions"], hooks
        )
        weights = recipient["layers"][job["layer_index"]]["attention_rows"][
            job["head_index"]
        ]
        values = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        metadata = hooks.route_record_weights(
            weights, values, base["record_regions"], profile
        )
        metadata.update(
            layer_index=job["layer_index"],
            head_index=job["head_index"],
            old_head_output=torch.zeros(4),
            new_head_output=metadata["delta_head_output"].clone(),
        )
        captured = copy.deepcopy(recipient)
        captured["intervention"] = metadata
        raw[job["endpoint_id"]] = captured
        native = arrays[base["endpoint_id"]].clone()
        if job["condition"] not in ("selfRscreen", "LOOWmeanDcolor"):
            native[job["correct_token_ids"][0]] = 3
        arrays[job["endpoint_id"]] = native

    def margin(native, correct, wrong):
        return float(
            torch.logsumexp(native[correct].double(), 0)
            - torch.logsumexp(native[wrong].double(), 0)
        )

    return planned, inputs, raw, arrays, original, SimpleNamespace(label_margin=margin)


def test_complete_native_and_operation_replay_reproduces_selection_and_gate():
    planned, inputs, raw, arrays, original, screen = complete_fixture()
    _, selected, result = routing_run.evaluate(
        Reader(arrays),
        Reader(original),
        raw,
        planned,
        inputs,
        screen,
        routing_plan,
        hooks,
    )
    assert len(selected["intervention_jobs"]) == 40
    assert result["native_forward_count"] == 72
    assert len(result["self_oracles"]) == 8
    assert result["assessment"]["passed"]
    assert result["mechanism_identified"] is False


def test_operator_delta_tampering_and_nonzero_self_native_are_rejected():
    planned, inputs, raw, arrays, original, screen = complete_fixture()
    selected = routing_plan.select_heads(
        planned, routing_run.capture_summaries(planned, raw, Reader(arrays))
    )
    job = next(j for j in selected["intervention_jobs"] if j["condition"] == "Dcolor")
    broken = copy.deepcopy(raw)
    broken[job["endpoint_id"]]["intervention"]["delta_weights"][0] += 1e-6
    with pytest.raises(ValueError, match="CPU replay"):
        routing_run.evaluate(
            Reader(arrays),
            Reader(original),
            broken,
            planned,
            inputs,
            screen,
            routing_plan,
            hooks,
        )
    self_job = next(
        j for j in selected["intervention_jobs"] if j["condition"] == "selfRscreen"
    )
    broken_native = {key: value.clone() for key, value in arrays.items()}
    broken_native[self_job["endpoint_id"]][0] += 1e-7
    with pytest.raises(ValueError, match="bitwise native"):
        routing_run.evaluate(
            Reader(broken_native),
            Reader(original),
            raw,
            planned,
            inputs,
            screen,
            routing_plan,
            hooks,
        )


def test_no_supported_positive_head_is_a_stop_not_mechanism_absence():
    planned, inputs, raw, arrays, original, screen = complete_fixture()
    ids = {job["endpoint_id"] for job in planned["capture_jobs"]}
    raw = {key: value for key, value in raw.items() if key in ids}
    arrays = {key: value for key, value in arrays.items() if key in ids}
    for captured in raw.values():
        captured["layers"][0]["attention_rows"][0] = torch.tensor(
            [0.1] * 4 + [0.3, 0.3]
        )
    _, selected, result = routing_run.evaluate(
        Reader(arrays),
        Reader(original),
        raw,
        planned,
        inputs,
        screen,
        routing_plan,
        hooks,
    )
    assert selected["selected_heads"] == []
    assert result["native_forward_count"] == 32
    assert result["assessment"]["passed"] is False
    assert result["mechanism_identified"] is False
