"""Saved-native-only readout tests; synthetic logits, never model inference."""

import copy

import pytest
import screen_effect_readout as readout
import torch


class Tokenizer:
    def decode(self, tokens, *, skip_special_tokens):
        assert skip_special_tokens is False
        return ["red", "Red.", "blue", "green", "yellow", "<other>"][tokens[0]]


class Reader:
    def __init__(self, native):
        self.native = native

    def read(self, identity):
        return self.native[identity]


def fixture():
    labels = {"red": [0, 1], "blue": [2], "green": [3], "yellow": [4]}
    facts = {obj: {"color": label} for obj, label in zip("ABCD", labels)}
    inputs = {
        "model": {"vocab_size": 6},
        "rows": [],
        "worlds": [],
        "label_token_ids": {"color": labels},
    }
    plan = {"cases": [], "layer_jobs": []}
    result = {
        "selected_layers": [0, 1],
        "selected_heads": [
            {"layer_index": 0, "head_index": 0},
            {"layer_index": 1, "head_index": 1},
        ],
        "head_jobs": [],
        "layer_measurements": [],
        "head_measurements": [],
    }
    native = {}
    for number in range(8):
        identity = f"case{number}"
        case = {
            "case_id": identity,
            "world_id": f"world{number}",
            "screen_field": "color",
            "screen_row_id": f"row{number}",
            "correct_object": "A",
            "reverse_predicted_object": "B",
        }
        plan["cases"].append(case)
        inputs["rows"].append({"id": case["screen_row_id"], "expected_answer": "red"})
        inputs["worlds"].append({"world_id": case["world_id"], "objects": facts})
        clean = torch.tensor([0.0, -2.0, 4.0, 1.0, -1.0, -3.0], dtype=torch.float32)
        native[f"clean::{identity}"] = clean
        for kind, layer, head in [
            ("layer", 0, None),
            ("layer", 1, None),
            ("head", 0, 0),
            ("head", 1, 1),
        ]:
            endpoint = f"{kind}::{identity}::{layer}::{head}"
            job = {
                "endpoint_id": endpoint,
                "case_id": identity,
                "layer_index": layer,
                "head_index": head,
            }
            (plan["layer_jobs"] if kind == "layer" else result["head_jobs"]).append(job)
            patched = clean.clone()
            patched[0] += 4.2 if number == 0 and layer == 0 else 0.1
            native[endpoint] = patched
            gain = float(
                (torch.logsumexp(patched.double()[[0, 1]], 0) - patched.double()[2])
                - (torch.logsumexp(clean.double()[[0, 1]], 0) - clean.double()[2])
            )
            result[kind + "_measurements"].append(
                {"endpoint_id": endpoint, "correct_minus_wrong_gain": gain}
            )
    return Reader(native), plan, inputs, result


def test_float64_complete_label_mass_and_argmax_object():
    native = torch.tensor([0.0, -2.0, 4.0, 1.0, -1.0, -3.0], dtype=torch.float32)
    probabilities = torch.softmax(native.double(), 0)
    stats = readout.token_stats(
        native,
        [0, 1],
        [2],
        {"A": "red", "B": "blue", "C": "green", "D": "yellow"},
        Tokenizer(),
        6,
    )
    assert stats["correct_label_probability"] == float(probabilities[[0, 1]].sum())
    assert stats["argmax_token_id"] == 2
    assert stats["argmax_decoded_token"] == "blue"
    assert stats["argmax_listed_object"] == "B"
    assert stats["correct_first_token"] is False
    with pytest.raises(ValueError):
        readout.token_stats(native.double(), [0, 1], [2], {}, Tokenizer(), 6)


def test_selected_targets_report_eight_cases_and_any_screen_flips():
    reader, plan, inputs, result = fixture()
    report = readout.screen_readout(reader, plan, inputs, result, Tokenizer())
    assert len(report["selected_targets"]) == 4
    first = report["selected_targets"][0]
    assert first["summary"]["case_count"] == 8
    assert first["summary"]["clean_correct_first_token_count"] == 0
    assert first["summary"]["patched_correct_first_token_count"] == 1
    assert first["summary"]["argmax_token_changed_count"] == 1
    assert first["summary"]["mean_correct_label_probability_change"] > 0
    assert report["all_screen_argmax_changes"]["layer"]["job_count"] == 16
    assert report["all_screen_argmax_changes"]["head"]["argmax_changed_count"] == 1
    assert report["complete_answer_repair_established"] is False
    assert any("selection" in text.lower() for text in report["caveats"])


def test_saved_margin_measurement_must_match_native_recomputation():
    reader, plan, inputs, result = fixture()
    forged = copy.deepcopy(result)
    forged["layer_measurements"][0]["correct_minus_wrong_gain"] += 0.01
    with pytest.raises(ValueError, match="margin"):
        readout.screen_readout(reader, plan, inputs, forged, Tokenizer())
