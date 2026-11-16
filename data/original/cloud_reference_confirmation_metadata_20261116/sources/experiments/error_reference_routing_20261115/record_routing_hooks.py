"""Gold-blind, record-mass-only surgery for pinned Qwen3 eager attention.

The installed eager function is always called unchanged. Natural captures retain
its original last-query rows, not surgically modified rows. Only one selected
head's last-query output can change. CPU routing arithmetic is explicit so the
saved native row, recipient V matrix and desired profile suffice for replay.
This process-global wrapper is deliberately single-forward and non-reentrant.
"""

import hashlib
import inspect
from contextlib import contextmanager

import torch
from transformers.models.qwen3 import modeling_qwen3 as qwen

SOURCE_HASHES = {
    "eager_attention_forward": "cb34ccea28710ae8d5d94a241b704ea3d974048f85372f0b1b3c2a8a2ef1ee20",
    "attention_forward": "6c8fb032ff3742438fb80dab4427c11ef04e12fdc02f08abd11ff5adaa0fd1c3",
    "repeat_kv": "e9b1ad4a8df3194d5b12f94567c3a2dfed2c0d9833af1b8f775e5875575ff9c4",
}


def _regions(record_regions, length=None):
    if set(record_regions) != set("ABCD"):
        raise ValueError("Exactly four opaque record regions A-D are required")
    regions = {obj: list(record_regions[obj]) for obj in "ABCD"}
    positions = [position for region in regions.values() for position in region]
    if (
        any(not region for region in regions.values())
        or len(set(positions)) != len(positions)
        or any(
            type(position) is not int
            or position < 0
            or (length is not None and position >= length)
            for position in positions
        )
    ):
        raise ValueError(
            "Record regions must be nonempty, disjoint, unique valid positions"
        )
    return regions, sorted(positions)


def record_mass_profile(weights, record_regions, *, require_support=True):
    """Compute CPU float64 masses without smoothing or repairing missing support."""
    if (
        weights.dtype != torch.float32
        or weights.ndim != 1
        or not bool(torch.isfinite(weights).all())
        or bool((weights < 0).any())
    ):
        raise ValueError("Attention weights must be a finite nonnegative float32 row")
    regions, _ = _regions(record_regions, len(weights))
    row = weights.detach().cpu().to(torch.float64)
    if abs(float(row.sum()) - 1) > 1e-6:
        raise ValueError("Attention row must sum to one within FP32 tolerance")
    masses = torch.stack([row[regions[obj]].sum() for obj in "ABCD"])
    total = float(masses.sum())
    if total <= 0 or (
        require_support and (total < 1e-4 or bool((masses < 1e-8).any()))
    ):
        raise ValueError("Insufficient record support; no epsilon fallback is allowed")
    return {"masses": masses, "total": total, "profile": masses / total}


def _desired_profile(profile):
    desired = (
        torch.as_tensor(profile, dtype=torch.float64, device="cpu").detach().clone()
    )
    if (
        tuple(desired.shape) != (4,)
        or not bool(torch.isfinite(desired).all())
        or bool((desired < 0).any())
        or abs(float(desired.sum()) - 1) > 1e-10
    ):
        raise ValueError(
            "Desired profile must be nonnegative and normalized within 1e-10"
        )
    return desired


def route_record_weights(old_weights, value_matrix, record_regions, desired_profile):
    """Pure, answer-blind operation and offline replay from saved native tensors.

    Float64 coefficients are formed on CPU, then delta weights and their product
    with original recipient V are float32 CPU tensors. No labels, target object,
    donor values or facts are accepted. FP32 rounding errors are measured, never
    concealed by renormalizing the new row or adding support epsilon.
    """
    if (
        value_matrix.dtype != torch.float32
        or value_matrix.ndim != 2
        or value_matrix.shape[0] != len(old_weights)
        or not bool(torch.isfinite(value_matrix).all())
    ):
        raise ValueError(
            "Recipient V must be the complete finite float32 S-by-D matrix"
        )
    regions, positions = _regions(record_regions, len(old_weights))
    desired = _desired_profile(desired_profile)
    original = old_weights.detach().cpu().clone()
    values = value_matrix.detach().cpu().clone()
    before = record_mass_profile(original, regions)
    delta64 = torch.zeros(len(original), dtype=torch.float64)
    for index, obj in enumerate("ABCD"):
        region = regions[obj]
        within = original[region].to(torch.float64) / before["masses"][index]
        delta64[region] = (
            (desired[index] - before["profile"][index]) * before["total"] * within
        )
    delta = delta64.to(torch.float32)
    new = original + delta
    after = record_mass_profile(new, regions, require_support=False)
    if abs(after["total"] - before["total"]) > 1e-7:
        raise ArithmeticError("FP32 routing unexpectedly changed total record mass")
    profile_error = float((after["profile"] - desired).abs().max())
    if profile_error > 1e-6:
        raise ArithmeticError(
            "FP32 routing exceeded the fixed achieved-profile tolerance"
        )
    delta_output = delta @ values
    if not bool(torch.isfinite(delta_output).all()):
        raise ArithmeticError("Routing delta head output must remain finite")
    return {
        "old_weights": original,
        "new_weights": new,
        "delta_weights": delta,
        "value_matrix": values,
        "delta_head_output": delta_output,
        "old_record_masses": before["masses"],
        "new_record_masses": after["masses"],
        "old_record_profile": before["profile"],
        "new_record_profile": after["profile"],
        "desired_profile": desired,
        "old_record_total": before["total"],
        "new_record_total": after["total"],
        "record_regions": regions,
        "record_positions": positions,
        "record_mass_error": abs(after["total"] - before["total"]),
        "old_attention_total": float(original.to(torch.float64).sum()),
        "new_attention_total": float(new.to(torch.float64).sum()),
        "new_profile_max_abs_error": profile_error,
        "self_zero_delta": bool((delta == 0).all()),
        "delta_computation": "CPU float64 masses/coefficients, then CPU float32 delta_weights @ recipient_V",
        "cpu_threads": torch.get_num_threads(),
    }


def _verify_installed_source():
    functions = {
        "eager_attention_forward": qwen.eager_attention_forward,
        "attention_forward": qwen.Qwen3Attention.forward,
        "repeat_kv": qwen.repeat_kv,
    }
    try:
        hashes = {
            name: hashlib.sha256(inspect.getsource(function).encode()).hexdigest()
            for name, function in functions.items()
        }
    except (OSError, TypeError) as error:
        raise ValueError("Cannot verify installed Qwen3 eager source") from error
    if (
        hashes != SOURCE_HASHES
        or qwen.Qwen3Attention.forward.__globals__ is not qwen.__dict__
        or qwen.ALL_ATTENTION_FUNCTIONS.get_interface(
            "eager", qwen.eager_attention_forward
        )
        is not qwen.eager_attention_forward
    ):
        raise ValueError("Installed Qwen3 eager source or global dispatch changed")
    return hashes


@contextmanager
def route_attention(model, record_regions, *, intervention=None):
    """Yield layers[L]: original CPU [H,S] rows + aggregate native record-V hash.

    intervention, if present, has exactly layer_index, head_index and a four-mass
    desired_profile. All saved intervention tensors are CPU native float32,
    except the explicitly named float64 record masses/profiles.
    """
    source_hashes = _verify_installed_source()
    regions, positions = _regions(record_regions)
    config = model.config
    if (
        config.model_type != "qwen3"
        or config._attn_implementation != "eager"
        or model.training
    ):
        raise ValueError("Only evaluation-mode Qwen3 eager attention is supported")
    modules = [layer.self_attn for layer in model.model.layers]
    mapping = {id(module): index for index, module in enumerate(modules)}
    if len(mapping) != len(modules):
        raise ValueError("Attention module identities must be unique")
    if intervention is not None and (
        set(intervention) != {"layer_index", "head_index", "desired_profile"}
        or type(intervention["layer_index"]) is not int
        or not 0 <= intervention["layer_index"] < len(modules)
        or type(intervention["head_index"]) is not int
        or not 0 <= intervention["head_index"] < config.num_attention_heads
    ):
        raise ValueError(
            "Intervention must select exactly one valid layer/head and profile"
        )
    if intervention is not None:
        intervention = {
            **intervention,
            "desired_profile": _desired_profile(intervention["desired_profile"]),
        }
    capture = {
        "layers": {},
        "intervention": None,
        "installed_source_sha256": source_hashes,
    }
    original = qwen.eager_attention_forward

    def wrapped(
        module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs
    ):
        if id(module) not in mapping:
            raise ValueError("Concurrent foreign attention call is not supported")
        layer = mapping[id(module)]
        if (
            layer in capture["layers"]
            or dropout != 0
            or module.training
            or query.dtype != torch.float32
            or key.dtype != torch.float32
            or value.dtype != torch.float32
            or query.ndim != 4
            or key.ndim != 4
            or value.ndim != 4
            or query.shape[0] != 1
            or query.device != key.device
            or query.device != value.device
            or query.shape
            != (1, config.num_attention_heads, value.shape[2], config.head_dim)
            or key.shape != value.shape
            or value.shape[:2] != (1, config.num_key_value_heads)
            or value.shape[3] != config.head_dim
            or module.head_dim != config.head_dim
            or module.num_key_value_groups * config.num_key_value_heads
            != config.num_attention_heads
        ):
            raise ValueError(
                "One complete B1 FP32 no-cache eager pass with actual head_dim is required"
            )
        _regions(
            regions, value.shape[2] - 1
        )  # Printed records must precede the final query.
        output, weights = original(
            module,
            query,
            key,
            value,
            attention_mask,
            scaling=scaling,
            dropout=dropout,
            **kwargs,
        )
        if tuple(output.shape) != (
            1,
            query.shape[2],
            config.num_attention_heads,
            config.head_dim,
        ) or tuple(weights.shape) != (
            1,
            config.num_attention_heads,
            query.shape[2],
            value.shape[2],
        ):
            raise ValueError("Installed eager output geometry changed")
        record_values = value[:, :, positions, :].detach().cpu().contiguous()
        capture["layers"][layer] = {
            "attention_rows": weights[0, :, -1, :].detach().cpu().clone(),
            "record_value_sha256": hashlib.sha256(
                record_values.numpy().tobytes()
            ).hexdigest(),
            "record_positions": positions,
            "num_key_value_heads": value.shape[1],
            "head_dim": value.shape[3],
        }
        if intervention is None or layer != intervention["layer_index"]:
            return output, weights
        head = intervention["head_index"]
        recipient_v = qwen.repeat_kv(value, module.num_key_value_groups)[0, head]
        metadata = route_record_weights(
            weights[0, head, -1], recipient_v, regions, intervention["desired_profile"]
        )
        metadata.update(
            layer_index=layer,
            head_index=head,
            old_head_output=output[0, -1, head].detach().cpu().clone(),
        )
        if not metadata["self_zero_delta"]:
            output, weights = output.clone(), weights.clone()
            output[0, -1, head] += metadata["delta_head_output"].to(output.device)
            weights[0, head, -1] = metadata["new_weights"].to(weights.device)
        metadata["new_head_output"] = output[0, -1, head].detach().cpu().clone()
        capture["intervention"] = metadata
        return output, weights

    qwen.eager_attention_forward = wrapped
    try:
        yield capture
        if intervention is not None and capture["intervention"] is None:
            raise ValueError("The selected attention module was never reached")
    finally:
        qwen.eager_attention_forward = original
