"""Fixed-prefix additive interventions on synthetic projections and random Qwen3."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import query_delta_hooks as hooks
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM


def fake_model():
    model = torch.nn.Module()
    model.config = SimpleNamespace(
        model_type="qwen3",
        _attn_implementation="eager",
        head_dim=3,
        num_attention_heads=4,
        hidden_size=8,
    )
    model.model = torch.nn.Module()
    layer = torch.nn.Module()
    layer.self_attn = torch.nn.Module()
    layer.self_attn.o_proj = torch.nn.Linear(12, 8, bias=False)
    model.model.layers = torch.nn.ModuleList([layer])
    return model.float().eval().requires_grad_(False), layer.self_attn.o_proj


def test_only_original_query_position_and_one_head_change_as_sequence_grows():
    model, projection = fake_model()
    original = torch.arange(48, dtype=torch.float32).reshape(1, 4, 12)
    longer = torch.cat((original, torch.ones((1, 2, 12))), dim=1)
    delta = torch.tensor([0.2, -0.3, 0.5], dtype=torch.float32)
    reference = original[0, 1, 6:9].clone()
    seen = []
    with hooks.add_query_delta(model, 0, 2, 1, delta, reference) as metadata:
        observer = projection.register_forward_pre_hook(
            lambda module, inputs: seen.append(inputs[0].clone())
        )
        try:
            projection(original)
            projection(longer)
        finally:
            observer.remove()
    for actual, natural in zip(seen, (original, longer)):
        expected = natural.clone()
        expected[0, 1, 6:9] += delta
        assert torch.equal(actual, expected)
    assert metadata["attention_width"] == 12 != model.config.hidden_size
    assert [o["sequence_length"] for o in metadata["observations"]] == [4, 6]
    assert [o["query_position"] for o in metadata["observations"]] == [1, 1]
    assert all(o["max_abs_drift"] == 0 for o in metadata["observations"])


def test_zero_delta_is_an_exact_noop_and_removes_hook():
    model, projection = fake_model()
    original = torch.randn((1, 4, 12))
    reference = original[0, 3, :3].clone()
    before = projection(original)
    with hooks.add_query_delta(model, 0, 0, 3, torch.zeros(3), reference):
        after = projection(original)
    assert torch.equal(before, after)
    assert not projection._forward_pre_hooks


def test_drift_failure_and_external_exception_both_cleanup():
    model, projection = fake_model()
    original = torch.zeros((1, 4, 12))
    changed = original.clone()
    changed[0, 1, 3] = 3e-5
    with (
        pytest.raises(RuntimeError, match="drift"),
        hooks.add_query_delta(model, 0, 1, 1, torch.ones(3), torch.zeros(3)),
    ):
        projection(changed)
    assert not projection._forward_pre_hooks
    with (
        pytest.raises(RuntimeError, match="synthetic"),
        hooks.add_query_delta(model, 0, 1, 1, torch.ones(3), torch.zeros(3)),
    ):
        raise RuntimeError("synthetic")
    assert not projection._forward_pre_hooks


@pytest.mark.parametrize(
    "delta",
    [
        torch.zeros(2),
        torch.zeros(3, dtype=torch.float64),
        torch.tensor([0.0, float("nan"), 0.0]),
    ],
)
def test_invalid_delta_shape_dtype_or_norm_is_refused(delta):
    model, projection = fake_model()
    with (
        pytest.raises(ValueError),
        hooks.add_query_delta(model, 0, 0, 0, delta, torch.zeros(3)),
    ):
        projection(torch.zeros((1, 4, 12)))
    assert not projection._forward_pre_hooks


def test_tolerance_cannot_be_relaxed_and_model_must_be_frozen():
    model, projection = fake_model()
    with (
        pytest.raises(ValueError),
        hooks.add_query_delta(
            model, 0, 0, 0, torch.zeros(3), torch.zeros(3), drift_tolerance=3e-5
        ),
    ):
        pass
    model.requires_grad_(True)
    with (
        pytest.raises(ValueError),
        hooks.add_query_delta(model, 0, 0, 0, torch.zeros(3), torch.zeros(3)),
    ):
        pass
    assert not projection._forward_pre_hooks


def test_tiny_real_qwen_matches_global_operator_and_keeps_prefix_under_growth():
    sys.path.insert(
        0, str(Path(__file__).resolve().parents[1] / "error_reference_routing_20261115")
    )
    from record_routing_hooks import route_attention

    torch.manual_seed(65115)
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=12,
        max_position_embeddings=32,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    model = Qwen3ForCausalLM(config).float().eval().requires_grad_(False)
    ids = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]])
    regions = {obj: [position] for position, obj in enumerate("ABCD")}

    def forward(tokens):
        with torch.inference_mode():
            return (
                model(
                    input_ids=tokens,
                    attention_mask=torch.ones_like(tokens),
                    use_cache=False,
                    logits_to_keep=1,
                )
                .logits.detach()
                .clone()
            )

    with route_attention(
        model,
        regions,
        intervention={
            "layer_index": 0,
            "head_index": 1,
            "desired_profile": [0.1, 0.2, 0.3, 0.4],
        },
    ) as global_capture:
        global_native = forward(ids)
    surgery = global_capture["intervention"]
    with hooks.add_query_delta(
        model, 0, 1, 7, surgery["delta_head_output"], surgery["old_head_output"]
    ) as metadata:
        additive_native = forward(ids)
        forward(torch.cat((ids, torch.tensor([[10, 11]])), dim=1))
    assert (
        model.model.layers[0].self_attn.o_proj.in_features == 24 != config.hidden_size
    )
    assert torch.equal(global_native, additive_native)
    assert [o["query_position"] for o in metadata["observations"]] == [7, 7]
    assert all(o["max_abs_drift"] <= 2e-5 for o in metadata["observations"])
