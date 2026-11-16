"""Prefix-only capture and surgical attention-input replacement contracts."""

from types import SimpleNamespace

import prefix_hooks
import pytest
import torch
from torch import nn


class FakeAttention(nn.Module):
    def __init__(self, width, head_dim):
        super().__init__()
        self.head_dim = head_dim
        self.o_proj = nn.Linear(width, width, bias=False)
        with torch.no_grad():
            self.o_proj.weight.copy_(torch.eye(width))


class FakeLayer(nn.Module):
    def __init__(self, width, head_dim):
        super().__init__()
        self.self_attn = FakeAttention(width, head_dim)


class FakeBackbone(nn.Module):
    def __init__(self, width, head_dim):
        super().__init__()
        self.layers = nn.ModuleList([FakeLayer(width, head_dim) for _ in range(3)])
        self.seen = []
        self.fail = False

    def forward(self, *, input_ids, attention_mask, use_cache, return_dict):
        assert use_cache is False and return_dict
        assert torch.equal(attention_mask, torch.ones_like(input_ids))
        self.seen.append(input_ids.detach().cpu().tolist())
        width = self.layers[0].self_attn.o_proj.in_features
        hidden = input_ids.to(torch.float32).unsqueeze(-1).repeat(1, 1, width)
        for layer in self.layers:
            hidden = layer.self_attn.o_proj(hidden)
            if self.fail:
                raise RuntimeError("synthetic failure")
        return SimpleNamespace(last_hidden_state=hidden)


class FakeModel(nn.Module):
    def __init__(self, heads=2, head_dim=3):
        super().__init__()
        self.config = SimpleNamespace(
            num_attention_heads=heads, hidden_size=5, vocab_size=1000
        )
        self.model = FakeBackbone(heads * head_dim, head_dim)
        self.eval().requires_grad_(False)


def hook_counts(model):
    return [
        len(layer.self_attn.o_proj._forward_pre_hooks) for layer in model.model.layers
    ]


def test_capture_never_forwards_tokens_after_marker_and_retains_cpu_fp32():
    model = FakeModel()
    result = prefix_hooks.capture_prefix(model, [10, 11, 999], 1, [0, 1])
    assert model.model.seen == [[[10, 11]]]
    assert result["input_ids"] == [10, 11]
    assert result["prefix_length"] == 2 and result["alias_end_position"] == 1
    assert result["forward_scope"] == "backbone_prefix_only_no_logits"
    assert result["num_attention_heads"] == 2 and result["head_dim"] == 3
    assert set(result["layers"]) == {0, 1}
    for vector in result["layers"].values():
        assert vector.dtype == torch.float32 and vector.device.type == "cpu"
        assert torch.equal(vector, torch.full((6,), 11.0))
    assert hook_counts(model) == [0, 0, 0]


def test_capture_removes_every_hook_after_forward_error():
    model = FakeModel()
    model.model.fail = True
    with pytest.raises(RuntimeError, match="synthetic"):
        prefix_hooks.capture_prefix(model, [10, 11, 12], 1, [0, 1])
    assert hook_counts(model) == [0, 0, 0]


def test_capture_rejects_final_layer_and_bad_marker_before_forward():
    model = FakeModel()
    for layers, position in (
        ([2], 1),
        ([0, 0], 1),
        ([-1], 1),
        ([0], -1),
        ([0], 3),
        ([0], True),
    ):
        with pytest.raises((TypeError, ValueError)):
            prefix_hooks.capture_prefix(model, [10, 11, 12], position, layers)
    assert model.model.seen == [] and hook_counts(model) == [0, 0, 0]


def test_replacement_modifies_only_one_marker_head_and_never_input_storage():
    model = FakeModel()
    original = torch.arange(18, dtype=torch.float32).reshape(1, 3, 6)
    preserved = original.clone()
    projection = model.model.layers[0].self_attn.o_proj
    with prefix_hooks.replace_at_marker(
        model, 0, 1, torch.tensor([100.0, 101.0, 102.0]), head_index=1
    ):
        actual = projection(original)
        expected = original.clone()
        expected[0, 1, 3:] = torch.tensor([100.0, 101.0, 102.0])
        assert torch.equal(actual, expected)
        assert torch.equal(original, preserved)
    assert hook_counts(model) == [0, 0, 0]
    assert torch.equal(projection(original), original)


def test_self_replacement_is_bitwise_identity_and_reused_on_recomputed_steps():
    model = FakeModel()
    projection = model.model.layers[1].self_attn.o_proj
    first = torch.arange(18, dtype=torch.float32).reshape(1, 3, 6)
    donor = first[0, 1].clone()
    with prefix_hooks.replace_at_marker(model, 1, 1, donor):
        assert torch.equal(projection(first), first)
        extended = torch.cat([first + 50, torch.zeros((1, 1, 6))], dim=1)
        actual = projection(extended)
        assert torch.equal(actual[0, 1], donor)
        assert torch.equal(actual[0, 0], extended[0, 0])
        assert torch.equal(actual[0, 2:], extended[0, 2:])
    assert hook_counts(model) == [0, 0, 0]


def test_replacement_cleans_up_when_body_or_hook_raises():
    model = FakeModel()
    with (
        pytest.raises(RuntimeError, match="body"),
        prefix_hooks.replace_at_marker(model, 0, 1, torch.zeros(6)),
    ):
        raise RuntimeError("body")
    assert hook_counts(model) == [0, 0, 0]
    with (
        pytest.raises(ValueError, match="position"),
        prefix_hooks.replace_at_marker(model, 0, 9, torch.zeros(6)),
    ):
        model.model.layers[0].self_attn.o_proj(torch.zeros((1, 3, 6)))
    assert hook_counts(model) == [0, 0, 0]


@pytest.mark.parametrize(
    "donor",
    [
        torch.zeros(6, dtype=torch.float64),
        torch.full((6,), float("nan")),
        torch.zeros((1, 6)),
        torch.zeros(5),
    ],
)
def test_replacement_rejects_invalid_donor(donor):
    model = FakeModel()
    with (
        pytest.raises((TypeError, ValueError)),
        prefix_hooks.replace_at_marker(model, 0, 1, donor),
    ):
        pass
    assert hook_counts(model) == [0, 0, 0]


def test_real_qwen_style_head_width_need_not_equal_hidden_width():
    class IdentityProjection(nn.Module):
        in_features = 4096

        def forward(self, value):
            return value

    # Qwen3-4B has 32 heads × 128 dimensions despite hidden_size=2560.
    # Identity projections avoid allocating three large matrices in this test.
    model = FakeModel()
    model.config.num_attention_heads = 32
    model.config.hidden_size = 2560
    model.register_parameter(
        "device_anchor", nn.Parameter(torch.zeros(1), requires_grad=False)
    )
    for layer in model.model.layers:
        layer.self_attn.head_dim = 128
        layer.self_attn.o_proj = IdentityProjection()
    model.eval()
    captured = prefix_hooks.capture_prefix(model, [10, 11], 1, [0])
    assert captured["layers"][0].shape == (4096,)
    model.model.layers[0].self_attn.head_dim = 127
    with pytest.raises(ValueError, match="width"):
        prefix_hooks.capture_prefix(model, [10, 11], 1, [0])


def test_capture_rejects_non_integer_ids_and_nonfinite_projection_input():
    model = FakeModel()
    with pytest.raises((TypeError, ValueError)):
        prefix_hooks.capture_prefix(model, torch.tensor([10.0, 11.0]), 1, [0])
    with prefix_hooks.replace_at_marker(model, 0, 1, torch.zeros(6)):
        for bad in (
            torch.zeros((2, 3, 6)),
            torch.zeros((1, 3, 6), dtype=torch.float64),
            torch.full((1, 3, 6), float("inf")),
        ):
            with pytest.raises((TypeError, ValueError)):
                model.model.layers[0].self_attn.o_proj(bad)
    assert hook_counts(model) == [0, 0, 0]


def test_replacement_rejects_last_layer_invalid_head_or_marker():
    model = FakeModel()
    for layer, marker, head in (
        (2, 1, None),
        (0, -1, None),
        (0, 1, -1),
        (0, 1, 2),
        (0, 1, True),
    ):
        with (
            pytest.raises(ValueError),
            prefix_hooks.replace_at_marker(
                model, layer, marker, torch.zeros(6), head_index=head
            ),
        ):
            pass
    assert hook_counts(model) == [0, 0, 0]


def test_donor_is_copied_at_context_entry_and_training_mode_is_forbidden():
    model = FakeModel()
    donor = torch.ones(6)
    with prefix_hooks.replace_at_marker(model, 0, 1, donor):
        donor.zero_()
        actual = model.model.layers[0].self_attn.o_proj(torch.zeros((1, 3, 6)))
        assert torch.equal(actual[0, 1], torch.ones(6))
    model.train()
    with pytest.raises(ValueError, match="evaluation"):
        prefix_hooks.capture_prefix(model, [10, 11], 1, [0])
    assert hook_counts(model) == [0, 0, 0]
