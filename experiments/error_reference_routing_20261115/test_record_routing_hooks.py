"""Routing surgery specifications using synthetic Q/K/V, never pretrained models."""

import hashlib
from types import SimpleNamespace

import pytest
import record_routing_hooks as routing
import torch
from transformers.models.qwen3 import modeling_qwen3 as qwen

REGIONS = {"A": [0, 1], "B": [2], "C": [3], "D": [4]}


def fake_model():
    config = SimpleNamespace(
        model_type="qwen3",
        _attn_implementation="eager",
        head_dim=3,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    modules = [
        SimpleNamespace(
            num_key_value_groups=2, training=False, head_dim=3, config=config
        )
        for _ in range(2)
    ]
    model = SimpleNamespace(
        config=config,
        training=False,
        model=SimpleNamespace(layers=[SimpleNamespace(self_attn=m) for m in modules]),
    )
    rng = torch.Generator().manual_seed(11215)
    query = torch.randn((1, 4, 7, 3), generator=rng)
    key = torch.randn((1, 2, 7, 3), generator=rng)
    value = torch.randn((1, 2, 7, 3), generator=rng)
    return model, modules, query, key, value


def test_capture_original_function_is_bitwise_unchanged_and_hashes_original_kv():
    model, modules, query, key, value = fake_model()
    original = qwen.eager_attention_forward
    expected, expected_weights = original(
        modules[0], query, key, value, None, scaling=0.5
    )
    with routing.route_attention(model, REGIONS) as captured:
        actual, weights = qwen.eager_attention_forward(
            modules[0], query, key, value, None, scaling=0.5
        )
        assert torch.equal(actual, expected)
        assert torch.equal(weights, expected_weights)
        layer = captured["layers"][0]
        assert torch.equal(layer["attention_rows"], expected_weights[0, :, -1, :])
        assert (
            layer["record_value_sha256"]
            == hashlib.sha256(
                value[:, :, :5, :].contiguous().numpy().tobytes()
            ).hexdigest()
        )
        assert layer["record_positions"] == [0, 1, 2, 3, 4]
        assert layer["num_key_value_heads"] == 2
        assert captured["intervention"] is None
    assert qwen.eager_attention_forward is original


def test_operation_is_gold_blind_preserves_nonrecord_mass_and_within_record_pattern():
    weights = torch.tensor(
        [0.10, 0.15, 0.20, 0.05, 0.10, 0.15, 0.25], dtype=torch.float32
    )
    values = torch.arange(21, dtype=torch.float32).reshape(7, 3)
    desired = [0.1, 0.2, 0.3, 0.4]
    changed = routing.route_record_weights(weights, values, REGIONS, desired)
    assert torch.equal(changed["new_weights"][5:], weights[5:])
    assert torch.equal(changed["value_matrix"], values)
    assert changed["old_record_total"] == pytest.approx(
        changed["new_record_total"], abs=1e-7
    )
    assert changed["new_weights"][:5].sum() == pytest.approx(
        float(weights[:5].sum()), abs=1e-7
    )
    assert float(
        changed["new_weights"][0] / changed["new_weights"][1]
    ) == pytest.approx(float(weights[0] / weights[1]), abs=1e-6)
    torch.testing.assert_close(
        changed["new_record_profile"],
        torch.tensor(desired, dtype=torch.float64),
        atol=1e-7,
        rtol=0,
    )
    assert torch.equal(changed["delta_head_output"], changed["delta_weights"] @ values)
    assert torch.equal(weights, changed["old_weights"])
    with pytest.raises(TypeError):
        routing.route_record_weights(
            weights, values, REGIONS, desired, correct_object="D"
        )


def test_one_head_last_query_only_gqa_value_mapping_and_offline_replay():
    model, modules, query, key, value = fake_model()
    original = qwen.eager_attention_forward
    expected, old_weights = original(modules[0], query, key, value, None, scaling=0.5)
    intervention = {
        "layer_index": 0,
        "head_index": 3,
        "desired_profile": [0.1, 0.2, 0.3, 0.4],
    }
    with routing.route_attention(model, REGIONS, intervention=intervention) as captured:
        changed, new_weights = qwen.eager_attention_forward(
            modules[0], query, key, value, None, scaling=0.5
        )
    metadata = captured["intervention"]
    assert torch.equal(
        metadata["value_matrix"], value[0, 1]
    )  # GQA heads2/3 share KVhead1.
    replay = routing.route_record_weights(
        metadata["old_weights"],
        metadata["value_matrix"],
        REGIONS,
        metadata["desired_profile"],
    )
    for name in ("new_weights", "delta_weights", "delta_head_output"):
        assert torch.equal(replay[name], metadata[name])
    difference = changed - expected
    difference[0, -1, 3] = 0
    assert torch.equal(difference, torch.zeros_like(difference))
    assert torch.equal(new_weights[0, 3, -1, 5:], old_weights[0, 3, -1, 5:])
    assert torch.equal(
        metadata["new_head_output"],
        metadata["old_head_output"] + metadata["delta_head_output"],
    )


def test_self_profile_is_bitwise_zero_and_cleanup_on_exception():
    model, modules, query, key, value = fake_model()
    original = qwen.eager_attention_forward
    expected, weights = original(modules[0], query, key, value, None, scaling=0.5)
    profile = routing.record_mass_profile(weights[0, 2, -1], REGIONS)["profile"]
    with routing.route_attention(
        model,
        REGIONS,
        intervention={"layer_index": 0, "head_index": 2, "desired_profile": profile},
    ) as captured:
        actual, actual_weights = qwen.eager_attention_forward(
            modules[0], query, key, value, None, scaling=0.5
        )
    assert torch.equal(actual, expected)
    assert torch.equal(actual_weights, weights)
    assert captured["intervention"]["self_zero_delta"] is True
    with (
        pytest.raises(RuntimeError, match="synthetic failure"),
        routing.route_attention(model, REGIONS),
    ):
        raise RuntimeError("synthetic failure")
    assert qwen.eager_attention_forward is original


@pytest.mark.parametrize(
    "weights,profile",
    [
        ([0.1, 0.1, 0.2, 0.2, 0.0, 0.2, 0.2], [0.1, 0.2, 0.3, 0.4]),
        ([0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5], [0.1, 0.2, 0.3, 0.4]),
        ([0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.2], [-0.1, 0.2, 0.3, 0.6]),
        ([0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.2], [0.1, 0.2, 0.3, 0.5]),
    ],
)
def test_invalid_support_or_profiles_are_refused_without_epsilon(weights, profile):
    with pytest.raises(ValueError):
        routing.route_record_weights(
            torch.tensor(weights, dtype=torch.float32),
            torch.zeros((7, 3)),
            REGIONS,
            profile,
        )


def test_overlapping_regions_and_unsupported_family_are_refused():
    model, _, _, _, _ = fake_model()
    with (
        pytest.raises(ValueError),
        routing.route_attention(model, {**REGIONS, "D": [3, 4]}),
    ):
        pass
    model.config.model_type = "other"
    with (
        pytest.raises(ValueError, match="Qwen"),
        routing.route_attention(model, REGIONS),
    ):
        pass


def test_zero_desired_regions_are_allowed_when_recipient_support_exists():
    weights = torch.tensor(
        [0.10, 0.15, 0.20, 0.05, 0.10, 0.15, 0.25], dtype=torch.float32
    )
    changed = routing.route_record_weights(
        weights, torch.zeros((7, 3)), REGIONS, [1.0, 0.0, 0.0, 0.0]
    )
    assert torch.equal(changed["new_weights"][2:5], torch.zeros(3))
    assert changed["new_record_profile"][0] == 1


def test_nested_wrapper_and_intervention_failure_restore_original():
    model, modules, query, key, value = fake_model()
    original = qwen.eager_attention_forward
    with (
        routing.route_attention(model, REGIONS),
        pytest.raises(ValueError, match="source"),
        routing.route_attention(model, REGIONS),
    ):
        pass
    assert qwen.eager_attention_forward is original
    with (
        pytest.raises(ValueError),
        routing.route_attention(
            model,
            REGIONS,
            intervention={
                "layer_index": 0,
                "head_index": 0,
                "desired_profile": [1.0, 1.0, 1.0, 1.0],
            },
        ),
    ):
        qwen.eager_attention_forward(modules[0], query, key, value, None, scaling=0.5)
    assert qwen.eager_attention_forward is original


def test_achieved_profile_error_over_fixed_fp32_tolerance_is_refused(monkeypatch):
    original = routing.record_mass_profile

    def inaccurate_profile(weights, regions, *, require_support=True):
        measured = original(weights, regions, require_support=require_support)
        if not require_support:
            measured["profile"] = measured["profile"].clone()
            measured["profile"][0] += 2e-6
        return measured

    monkeypatch.setattr(routing, "record_mass_profile", inaccurate_profile)
    weights = torch.tensor(
        [0.10, 0.15, 0.20, 0.05, 0.10, 0.15, 0.25], dtype=torch.float32
    )
    with pytest.raises(ArithmeticError, match="profile"):
        routing.route_record_weights(
            weights, torch.zeros((7, 3)), REGIONS, [0.1, 0.2, 0.3, 0.4]
        )


def test_finite_native_values_cannot_produce_unchecked_nonfinite_delta_output():
    weights = torch.tensor(
        [0.005, 0.005, 0.3, 0.3, 0.38, 0.005, 0.005], dtype=torch.float32
    )
    values = torch.full((7, 3), -3e38, dtype=torch.float32)
    values[:2] = 3e38
    with pytest.raises(ArithmeticError, match="finite"):
        routing.route_record_weights(weights, values, REGIONS, [1.0, 0.0, 0.0, 0.0])
