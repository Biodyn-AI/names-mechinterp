"""Real dispatch integration on an UNTRAINED tiny CPU Qwen, never research weights."""

import importlib.util
from pathlib import Path

import decomposition_run as runner
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.models.qwen3 import modeling_qwen3


def load_hook_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_actual_fixed_head_capture_is_bitwise_read_only_with_unequal_width_gqa():
    torch.set_num_threads(1)
    torch.manual_seed(74115)
    base = Path(__file__).resolve().parents[1]
    mass_hooks = load_hook_file(
        "capture_test_mass_hooks",
        base / "error_reference_routing_20261115" / "record_routing_hooks.py",
    )
    query_hooks = load_hook_file(
        "capture_test_query_hooks",
        base / "error_reference_validation_20261115" / "query_delta_hooks.py",
    )
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=2,
        max_position_embeddings=32,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    model = Qwen3ForCausalLM(config).float().eval().requires_grad_(False)
    ids = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]])
    regions = {obj: [position] for position, obj in enumerate("ABCD")}
    projection = model.model.layers[28].self_attn.o_proj
    reference = []
    original_eager = modeling_qwen3.eager_attention_forward

    def observe(module, inputs):
        reference.append(inputs[0][0, -1, :2].detach().cpu().clone())

    def forward(identity):
        assert isinstance(identity, str)
        with torch.inference_mode():
            return (
                model(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    use_cache=False,
                    logits_to_keep=1,
                )
                .logits[0, 0]
                .detach()
                .cpu()
                .clone()
            )

    handle = projection.register_forward_pre_hook(observe)
    try:
        with mass_hooks.route_attention(model, regions) as old_capture:
            clean = forward("clean")
    finally:
        handle.remove()
    assert len(reference) == 1
    job = {
        "endpoint_id": "new-natural",
        "record_regions": regions,
        "original_query_position": len(ids[0]) - 1,
    }
    native, captured = runner.capture(
        model, job, old_capture, reference[0], mass_hooks, query_hooks, forward
    )
    assert projection.in_features == 64 != config.hidden_size
    assert torch.equal(native, clean)
    assert torch.equal(captured["head_output"], reference[0])
    assert torch.equal(
        captured["weights"], old_capture["layers"][28]["attention_rows"][0]
    )
    assert captured["values"].shape == (8, 2)
    assert (
        captured["record_V_sha256"] == old_capture["layers"][28]["record_value_sha256"]
    )
    assert captured["record_positions"] == [0, 1, 2, 3]
    assert captured["trace"]["zero_delta"] is True
    assert len(captured["trace"]["observations"]) == 1
    assert captured["trace"]["observations"][0]["max_abs_drift"] == 0
    assert not projection._forward_pre_hooks
    assert modeling_qwen3.eager_attention_forward is original_eager
