"""Token qualification and held-out-blind alignment selection."""

import re
from copy import deepcopy

import pytest
import torch

import e30_14_alignment as alignment
import e30_14_capture as capture
import e30_14_protocol as protocol


class CharacterTokenizer:
    chat_template = "synthetic-standard-user-chat"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert not tokenize and add_generation_prompt
        assert messages[0]["role"] == "user"
        return "<user>" + messages[0]["content"] + "<assistant>\n"

    def __call__(self, text, *, add_special_tokens, return_offsets_mapping):
        assert not add_special_tokens and return_offsets_mapping
        parts = list(re.finditer(r"\?\n|[\s\S]", text))
        return {
            "input_ids": [
                500 if part.group() == "?\n" else ord(part.group()) + 1
                for part in parts
            ],
            "offset_mapping": [part.span() for part in parts],
        }

    def decode(self, ids, **kwargs):
        return "".join("?\n" if token == 500 else chr(token - 1) for token in ids)


def facts_record():
    return {
        "population_reference": "2020_census",
        "facts": {
            row["entity_id"]: {"capital": "Alpha City", "largest_city": "Beta Town"}
            for row in protocol.build_entity_panel()
        },
    }


@pytest.mark.parametrize("chat_model", [False, True])
def test_token_preparation_tracks_complete_name_marker_and_answer_prefix(chat_model):
    prompts = protocol.build_prompts()
    rows = capture.prepare_token_rows(
        CharacterTokenizer(),
        prompts,
        chat_model=chat_model,
        facts=facts_record()["facts"],
    )
    assert len(rows) == 224
    assert {row["marker_token_id"] for row in rows} == {500}
    for row in rows:
        assert row["marker_position"] == row["entity_positions"][-1] + 1
        assert row["query_position"] == len(row["input_ids"]) - 1
        expected_first = (
            ord("A" if row["relation_id"] == "capital" else "B") + 1
            if chat_model
            else ord(" ") + 1
        )
        assert row["answer_first_token_id"] == expected_first
        assert row["answer_ids"]
        assert row["answer_prefix"] == ("" if chat_model else " ")
        assert row["position_ids"] == list(range(len(row["input_ids"])))
    assert any(len(row["entity_positions"]) > 1 for row in rows)


def test_wrong_donors_are_chosen_from_token_geometry_without_model_outputs():
    rows = capture.prepare_token_rows(
        CharacterTokenizer(),
        protocol.build_prompts(),
        chat_model=False,
        facts=facts_record()["facts"],
    )
    cells = capture.qualify_comparison_cells(rows, facts_record())
    assert len(cells) == 640
    assert all(type(cell["first_token_distinct_stratum"]) is bool for cell in cells)
    available = [
        cell for cell in cells if cell["length_matched_wrong_entity_id"] is not None
    ]
    assert available
    lookup = {row["prompt_id"]: row for row in rows}
    for cell in available:
        assert cell["length_matched_wrong_entity_id"] not in (
            cell["base_entity_id"],
            cell["donor_entity_id"],
        )
        assert capture.length_signature(
            lookup[cell["source_donor_prompt_id"]]
        ) == capture.length_signature(
            lookup[cell["length_matched_wrong_source_donor_prompt_id"]]
        )
        assert capture.length_signature(
            lookup[cell["donor_prompt_id"]]
        ) == capture.length_signature(
            lookup[cell["length_matched_wrong_target_donor_prompt_id"]]
        )


def test_model_manifest_and_combined_disk_plan_are_json_metadata_only():
    rows = capture.prepare_token_rows(
        CharacterTokenizer(),
        protocol.build_prompts(),
        chat_model=False,
        facts=facts_record()["facts"],
    )
    manifest = capture.make_token_manifest(
        rows,
        facts_record(),
        model_id="openai-community/gpt2-medium",
        model_config={"vocab_size": 50257, "n_embd": 1024, "n_layer": 24},
    )
    assert manifest["prompt_count"] == 224 and len(manifest["comparison_cells"]) == 640
    estimate = capture.estimate_storage(
        {
            "openai-community/gpt2-medium": {
                "vocab_size": 50257,
                "n_embd": 1024,
                "n_layer": 24,
            },
            "Qwen/Qwen3-4B-Instruct-2507": {
                "vocab_size": 151936,
                "hidden_size": 2560,
                "num_hidden_layers": 36,
            },
        }
    )
    assert estimate["within_limit"] and estimate["estimated_total_bytes"] < 15000000000
    with pytest.raises(ValueError, match="storage"):
        capture.estimate_storage(
            {
                "Qwen/Qwen3-4B-Instruct-2507": {
                    "vocab_size": 1000000,
                    "hidden_size": 2560,
                    "num_hidden_layers": 36,
                }
            }
        )


def test_answer_boundary_merge_fails_before_any_model_call():
    class MergingTokenizer(CharacterTokenizer):
        def __call__(self, text, **kwargs):
            result = super().__call__(text, **kwargs)
            if text.endswith(("Alpha City", "Beta Town")):
                result["input_ids"][0] += 1
            return result

    with pytest.raises(ValueError, match="prefix"):
        capture.prepare_token_rows(
            MergingTokenizer(),
            protocol.build_prompts(),
            chat_model=True,
            facts=facts_record()["facts"],
        )


def synthetic_training_capture():
    rows = [
        row for row in protocol.build_prompts() if row["split"] in ("fit", "validation")
    ]
    generator = torch.Generator().manual_seed(14)
    entities = protocol.build_splits()["fit"] + protocol.build_splits()["validation"]
    values = {
        entity: torch.randn(4, dtype=torch.float64, generator=generator)
        for entity in entities
    }
    rotation = torch.linalg.qr(
        torch.randn(4, 4, dtype=torch.float64, generator=generator)
    ).Q
    states = {}
    for row in rows:
        value = values[row["entity_id"]]
        if row["form"] == "abbr":
            value = value @ rotation
        states[row["prompt_id"]] = {}
        for layer in range(2):
            for site in ("marker", "query"):
                state = value.clone()
                if layer == 1 and row["split"] == "validation" and row["frame"] == "B":
                    state = state + torch.randn(
                        4, dtype=torch.float64, generator=generator
                    )
                states[row["prompt_id"]][f"residual::{layer}::{site}"] = state
    record = protocol.build_protocol()
    record["rank_grid"], record["ridge_fraction_grid"] = [2, 4], [0.01, 0.1]
    record["shuffles"], record["shuffle_count"] = record["shuffles"][:2], 2
    return states, rows, record


def test_selection_uses_validation_only_and_nulls_retune_at_fixed_layer():
    states, rows, record = synthetic_training_capture()
    selected = capture.select_maps(states, rows, protocol_record=record)
    assert set(selected) == {"entity_marker", "query"}
    for family in selected.values():
        assert family["layer_index"] == 0 and family["rank"] == 4
        assert family["ridge_fraction"] == 0.01
        assert len(family["shuffle_selection"]) == 2
        for direction in family["directions"].values():
            assert (
                direction["ridge"]["training_entity_ids"]
                == protocol.build_splits()["fit"]
            )
            assert len(direction["shuffles"]) == 2
            assert direction["orthogonal"]["method"] == "orthogonal"
        assert all(
            null["layer_index"] == family["layer_index"]
            for null in family["shuffle_selection"]
        )
    metadata = capture.selection_metadata(selected)
    assert metadata["test_vectors_or_outcomes_used"] is False


def test_test_vectors_cannot_enter_training_capture_or_selection():
    states, rows, record = synthetic_training_capture()
    test = next(row for row in protocol.build_prompts() if row["split"] == "test")
    with pytest.raises(ValueError, match="fit.*validation"):
        capture.select_maps(
            {**states, test["prompt_id"]: next(iter(states.values()))},
            rows + [test],
            protocol_record=record,
        )
    with pytest.raises(ValueError, match="fit.*validation"):
        capture.capture_stage(
            None, rows + [test], model_id="openai-community/gpt2-medium"
        )


def test_final_block_marker_is_ineligible_even_if_geometry_would_prefer_it():
    states, rows, record = synthetic_training_capture()
    for row in rows:
        if row["split"] == "validation" and row["frame"] == "B":
            for site in ("marker", "query"):
                first = states[row["prompt_id"]][f"residual::0::{site}"]
                last = states[row["prompt_id"]][f"residual::1::{site}"]
                states[row["prompt_id"]][f"residual::0::{site}"] = last
                states[row["prompt_id"]][f"residual::1::{site}"] = first
    selected = capture.select_maps(states, rows, protocol_record=record)
    assert selected["entity_marker"]["layer_index"] == 0
    assert selected["query"]["layer_index"] == 1
    assert {
        row["layer_index"] for row in selected["entity_marker"]["candidate_diagnostics"]
    } == {0}


def test_identity_plus_correction_preserves_unseen_components_when_y_equals_x():
    states, rows, record = synthetic_training_capture()
    original = deepcopy(states)
    source_ids = {
        row["entity_id"]: row["prompt_id"]
        for row in rows
        if row["form"] == "full" and row["frame"] == "A"
    }
    for row in rows:
        states[row["prompt_id"]] = deepcopy(original[source_ids[row["entity_id"]]])
    selected = capture.select_maps(states, rows, protocol_record=record)
    test_ids = protocol.build_splits()["test"][:2]
    delta = torch.tensor([1.0, -2.0, 3.0, -4.0], dtype=torch.float64)
    for family in selected.values():
        metadata = family["identity_correction_selection"]
        assert metadata["layer_index"] == family["layer_index"]
        assert metadata["diagnostic_only_not_confirmatory"]
        assert metadata["validation"]["squared_error_sum"] == 0
        assert metadata["validation"]["target_difference_energy_sum"] > 0
        assert metadata["validation"]["relative_squared_error"] == 0
        for bundle in family["directions"].values():
            corrected = delta + alignment.apply_alignment(
                bundle["identity_correction"],
                delta,
                base_entity_id=test_ids[0],
                donor_entity_id=test_ids[1],
            )
            torch.testing.assert_close(corrected, delta, atol=0, rtol=0)


def test_actual_random_tiny_qwen3_capture_and_patch_contract():
    transformers = pytest.importorskip("transformers")
    with torch.random.fork_rng():
        torch.manual_seed(14)
        config = transformers.Qwen3Config(
            vocab_size=1024,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=4,
            max_position_embeddings=512,
            attention_dropout=0,
        )
        config._attn_implementation = "eager"
        model = (
            transformers.Qwen3ForCausalLM(config).double().eval().requires_grad_(False)
        )
    prepared = capture.prepare_token_rows(
        CharacterTokenizer(),
        protocol.build_prompts(),
        chat_model=True,
        facts=facts_record()["facts"],
    )
    training_rows = [row for row in prepared if row["split"] in ("fit", "validation")]
    executor, captured = capture.capture_stage(
        model,
        training_rows,
        model_id="Qwen/Qwen3-4B-Instruct-2507",
        control_tolerance=1e-10,
    )
    assert len(captured["states"]) == 64
    assert captured["physical_forward_count"] == 64
    identities = [row["prompt_id"] for row in training_rows[:2]]
    checks = executor.qualify_oracles(sites=[(0, "marker")], prompt_ids=identities)
    assert (
        checks["self_patch_max_abs_error"]
        == checks["final_query_matched_max_abs_error"]
        == 0
    )
    result = executor.run_patch(
        identities[0], layer_index=0, token_site="marker", delta=torch.ones(8)
    )
    assert result.shape == (1024,) and bool(torch.isfinite(result).all())
    assert all(not module._forward_hooks for module in model.modules())
