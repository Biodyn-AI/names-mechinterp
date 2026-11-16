"""Compatibility test on a tiny, random CPU model, not research-model inference."""

import torch
from record_routing_hooks import record_mass_profile, route_attention
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.models.qwen3 import modeling_qwen3


def test_real_qwen_dispatch_gqa_unequal_width_self_and_causal_values():
    torch.manual_seed(42)
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=12,
        max_position_embeddings=32,
        bos_token_id=0,
        eos_token_id=1,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    model = Qwen3ForCausalLM(config).float().eval().requires_grad_(False)
    ids = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]])
    regions = {obj: [position] for position, obj in enumerate("ABCD")}
    original = modeling_qwen3.eager_attention_forward

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

    with route_attention(model, regions) as natural:
        clean = forward(ids)
    assert set(natural["layers"]) == {0, 1}
    assert (
        model.model.layers[0].self_attn.o_proj.in_features == 24 != config.hidden_size
    )
    assert natural["layers"][0]["attention_rows"].shape == (2, 8)
    profile = record_mass_profile(natural["layers"][0]["attention_rows"][1], regions)[
        "profile"
    ]
    with route_attention(
        model,
        regions,
        intervention={
            "layer_index": 0,
            "head_index": 1,
            "desired_profile": profile,
        },
    ) as sham:
        repeated = forward(ids)
    assert torch.equal(clean, repeated)
    assert sham["intervention"]["self_zero_delta"]
    changed_ids = ids.clone()
    changed_ids[0, -1] = 10
    with route_attention(model, regions) as other_query:
        forward(changed_ids)
    for layer in range(2):
        assert (
            natural["layers"][layer]["record_value_sha256"]
            == other_query["layers"][layer]["record_value_sha256"]
        )
    assert modeling_qwen3.eager_attention_forward is original
