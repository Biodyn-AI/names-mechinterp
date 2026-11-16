"""Model-free contracts for a bounded, conditional first-token screen."""

import copy
import math
from types import SimpleNamespace

import causal_plan
import causal_run
import pytest
import torch
from test_causal_plan import _inputs


def test_label_ids_are_exhaustive_whole_label_decodes_not_first_word_matches():
    class Tokenizer:
        def decode(self, ids, *, skip_special_tokens):
            assert not skip_special_tokens
            return ["red", " Red.", "red explanation", "<eos>", "blue", "RED\n"][ids[0]]

    labels = causal_run.exhaustive_label_ids(Tokenizer(), 6, {"color": ["red", "blue"]})
    assert labels == {"color": {"red": [0, 1, 5], "blue": [4]}}


def test_label_sets_must_cover_each_declared_value():
    class Tokenizer:
        def decode(self, ids, *, skip_special_tokens):
            return "red"

    with pytest.raises(ValueError, match="one-token"):
        causal_run.exhaustive_label_ids(Tokenizer(), 1, {"color": ["red", "blue"]})


def test_native_metric_uses_all_accepted_labels_without_vocabulary_renormalization():
    native = torch.tensor([1.0, 2.0, 3.0, 100.0], dtype=torch.float32)
    assert causal_run.label_margin(native, [0, 1], [2]) == pytest.approx(
        math.log(math.exp(1) + math.exp(2)) - 3
    )


def test_self_oracle_requires_both_numeric_and_argmax_agreement():
    clean = torch.tensor([0.0, 1.0], dtype=torch.float32)
    assert causal_run.oracle_difference(clean, clean)["passed"]
    assert not causal_run.oracle_difference(torch.tensor([0.0, 1.001]), clean)["passed"]
    assert not causal_run.oracle_difference(
        torch.tensor([1.00001, 1.0]), torch.tensor([1.0, 1.00001])
    )["passed"]


def fixture():
    inputs = _inputs()
    planned = causal_plan.plan(inputs)
    cases = {c["case_id"]: c for c in planned["cases"]}
    worlds = {w["world_id"]: w for w in inputs["worlds"]}
    rows = {r["id"]: r for r in inputs["rows"]}
    arrays = {}

    def add(job, gain=0):
        native = torch.zeros(inputs["model"]["vocab_size"], dtype=torch.float32)
        case, row = cases[job["case_id"]], rows[job["row_id"]]
        label = worlds[case["world_id"]]["objects"][case["correct_object"]][
            row["field"]
        ]
        native[inputs["label_token_ids"][row["field"]][label][0]] = gain
        arrays[job["endpoint_id"]] = native

    for job in planned["clean_jobs"] + planned["layer_self_jobs"]:
        add(job)
    for job in planned["layer_jobs"]:
        add(job, job["layer_index"] + 1)
    return inputs, planned, arrays, add


class Reader:
    def __init__(self, arrays):
        self.arrays = arrays
        self.index = {"endpoints": {identity: {} for identity in arrays}}

    def read(self, identity):
        return self.arrays[identity]


def test_saved_native_endpoints_reproduce_layer_and_head_selection():
    inputs, planned, arrays, add = fixture()
    layer_scores = causal_run.measure_jobs(
        Reader(arrays), planned["layer_jobs"], planned, inputs
    )
    refined = causal_plan.select_layers(planned, layer_scores)
    for job in refined["head_self_jobs"]:
        add(job)
    for job in refined["head_jobs"]:
        add(job, job["head_index"] + 1)
    original = {
        c["screen_row_id"] + "::step00": arrays[f"clean::{c['case_id']}"]
        for c in planned["cases"]
    }
    result = causal_run.evaluate_screen(
        Reader(arrays), Reader(original), planned, inputs
    )
    assert result["selected_layers"] == [4, 3]
    assert result["selected_heads"] == [
        {"layer_index": 3, "head_index": 3},
        {"layer_index": 4, "head_index": 3},
    ]
    assert result["native_forward_count"] == len(arrays)
    assert result["layer_ranking"][0] == {"layer_index": 4, "mean_gain": 5.0}
    assert result["head_ranking"][0] == {
        "layer_index": 3,
        "head_index": 3,
        "mean_gain": 4.0,
    }
    assert result["complete_answer_repair_established"] is False


def test_failed_layer_self_oracle_stops_before_any_screen_job():
    inputs, planned, arrays, _ = fixture()
    allowed = {
        j["endpoint_id"] for j in planned["clean_jobs"] + planned["layer_self_jobs"]
    }
    arrays = {
        identity: native for identity, native in arrays.items() if identity in allowed
    }
    # A discrepancy in the eighth world is as disqualifying as one in the first.
    assert {job["case_id"] for job in planned["layer_self_jobs"]} == {
        case["case_id"] for case in planned["cases"]
    }
    assert len(planned["layer_self_jobs"]) == 8 * 5
    arrays[planned["layer_self_jobs"][-1]["endpoint_id"]][0] = 1
    original = {
        c["screen_row_id"] + "::step00": arrays[f"clean::{c['case_id']}"]
        for c in planned["cases"]
    }
    result = causal_run.evaluate_screen(
        Reader(arrays), Reader(original), planned, inputs
    )
    assert result["status"] == "stopped_layer_self_oracle"
    assert result["selected_layers"] == [] and result["selected_heads"] == []


def test_failed_head_self_oracle_stops_before_any_head_screen_job():
    inputs, planned, arrays, add = fixture()
    layer_scores = causal_run.measure_jobs(
        Reader(arrays), planned["layer_jobs"], planned, inputs
    )
    refined = causal_plan.select_layers(planned, layer_scores)
    for job in refined["head_self_jobs"]:
        add(job)
    arrays[refined["head_self_jobs"][-1]["endpoint_id"]][0] = 1
    original = {
        c["screen_row_id"] + "::step00": arrays[f"clean::{c['case_id']}"]
        for c in planned["cases"]
    }
    result = causal_run.evaluate_screen(
        Reader(arrays), Reader(original), planned, inputs
    )
    assert result["status"] == "stopped_head_self_oracle"
    assert result["selected_layers"] == [4, 3]
    assert result["selected_heads"] == result["head_measurements"] == []


def test_clean_replay_must_be_bitwise_identical_to_original_behavior():
    inputs, planned, arrays, _ = fixture()
    original = {
        c["screen_row_id"] + "::step00": arrays[f"clean::{c['case_id']}"] + 1e-7
        for c in planned["cases"]
    }
    with pytest.raises(ValueError, match="bitwise"):
        causal_run.evaluate_screen(Reader(arrays), Reader(original), planned, inputs)


def test_rescore_receipt_is_bound_to_every_required_behavior_artifact():
    receipt = {
        "model": "qwen",
        "exact_match": True,
        "native_steps": 10,
        "confirmation_calls": 0,
        "causal_calls": 0,
    }
    anchors = {
        name: str(index) * 64 for index, name in enumerate(causal_run.RECEIPT_ANCHORS)
    }
    receipt.update(anchors)
    causal_run.verify_receipt(receipt, anchors, 10)
    for key in (
        "model",
        "registration_sha256",
        "custody_sha256",
        "raw_index_sha256",
        "source_manifest_sha256",
    ):
        broken = copy.deepcopy(receipt)
        broken[key] = "wrong"
        with pytest.raises(ValueError):
            causal_run.verify_receipt(broken, anchors, 10)


def test_prefix_receipt_requires_every_strict_capture_and_finite_cpu_vectors():
    inputs, planned, _, _ = fixture()
    prefixes = {}
    for job in planned["prefix_capture_jobs"]:
        prefixes[job["endpoint_id"]] = {
            "input_ids": job["input_ids"],
            "prefix_length": len(job["input_ids"]),
            "alias_end_position": job["alias_end_position"],
            "forward_scope": "backbone_prefix_only_no_logits",
            "num_attention_heads": inputs["model"]["heads"],
            "head_dim": inputs["model"]["head_dim"],
            "layers": {
                layer: torch.zeros(inputs["model"]["o_proj_in_features"])
                for layer in job["layers"]
            },
        }
    assert causal_run.validate_prefixes(prefixes, planned) == 16 * 5 * 32 * 4
    broken = copy.deepcopy(prefixes)
    broken.pop(next(iter(broken)))
    with pytest.raises(ValueError, match="coverage"):
        causal_run.validate_prefixes(broken, planned)
    broken = copy.deepcopy(prefixes)
    next(iter(broken.values()))["layers"][0][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        causal_run.validate_prefixes(broken, planned)


def test_nonpositive_layer_screen_stops_without_any_head_calls():
    inputs, planned, arrays, _ = fixture()
    for job in planned["layer_jobs"]:
        arrays[job["endpoint_id"]].zero_()
    original = {
        c["screen_row_id"] + "::step00": arrays[f"clean::{c['case_id']}"]
        for c in planned["cases"]
    }
    result = causal_run.evaluate_screen(
        Reader(arrays), Reader(original), planned, inputs
    )
    assert result["status"] == "stopped_no_positive_layer_gain"
    assert result["selected_layers"] == result["selected_heads"] == []
    assert result["head_jobs"] == result["head_measurements"] == []


def test_native_forward_retains_the_full_vocabulary_with_no_cache():
    class FakeModel:
        def parameters(self):
            yield torch.zeros(1)

        def __call__(self, **kwargs):
            assert kwargs["input_ids"].tolist() == [[10, 11]]
            assert kwargs["use_cache"] is False and kwargs["logits_to_keep"] == 1
            return SimpleNamespace(
                logits=torch.arange(7, dtype=torch.float32).reshape(1, 1, 7)
            )

    assert torch.equal(
        causal_run.native_forward(FakeModel(), {"input_ids": [10, 11]}, 7),
        torch.arange(7, dtype=torch.float32),
    )
    with pytest.raises(ValueError, match="complete native"):
        causal_run.native_forward(FakeModel(), {"input_ids": [10, 11]}, 6)
