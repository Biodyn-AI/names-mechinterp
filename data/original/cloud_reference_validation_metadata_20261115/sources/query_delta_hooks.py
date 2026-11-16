"""Hold a first-step additive head delta at the ORIGINAL query position.

Callers must recompute the whole prompt plus generated tokens with use_cache=False.
The same delta is reused; neither its position nor its value follows generation.
This helper loads no model and accepts no facts, answer labels, or donor values.
"""

import math
from contextlib import contextmanager

import torch


@contextmanager
def add_query_delta(
    model,
    layer,
    head,
    original_query_position,
    delta,
    reference_natural_head,
    *,
    drift_tolerance=2e-5,
):
    """Observe unmodified o_proj input, check prefix drift, then add one fixed delta.

    All saved vectors are CPU native float32 clones. A zero delta returns None
    from the prehook, leaving even the original input object untouched. The
    preset drift tolerance may be tightened but never relaxed above 2e-5.
    """
    config = model.config
    if (
        config.model_type != "qwen3"
        or config._attn_implementation != "eager"
        or any(module.training for module in model.modules())
        or any(
            parameter.requires_grad or parameter.dtype != torch.float32
            for parameter in model.parameters()
        )
    ):
        raise ValueError(
            "A frozen float32 evaluation-mode eager Qwen3 model is required"
        )
    layers = model.model.layers
    head_dim, heads = config.head_dim, config.num_attention_heads
    if (
        type(head_dim) is not int
        or head_dim <= 0
        or type(heads) is not int
        or heads <= 0
        or type(layer) is not int
        or not 0 <= layer < len(layers)
        or type(head) is not int
        or not 0 <= head < heads
        or type(original_query_position) is not int
        or original_query_position < 0
    ):
        raise ValueError("Valid layer/head and original query position are required")
    if (
        type(drift_tolerance) not in (int, float)
        or not math.isfinite(drift_tolerance)
        or not 0 <= drift_tolerance <= 2e-5
    ):
        raise ValueError("The fixed drift tolerance cannot be relaxed above 2e-5")
    projection = layers[layer].self_attn.o_proj
    width = heads * head_dim  # Qwen attention width need not equal hidden_size.
    if projection.in_features != width:
        raise ValueError(
            "Projection width differs from actual head-count times head_dim"
        )
    for name, vector in (("delta", delta), ("reference", reference_natural_head)):
        if (
            not isinstance(vector, torch.Tensor)
            or vector.dtype != torch.float32
            or tuple(vector.shape) != (head_dim,)
            or not bool(torch.isfinite(vector).all())
        ):
            raise ValueError(
                f"{name} must be a finite native float32 head-dimension vector"
            )
    fixed_delta = delta.detach().cpu().clone()
    reference = reference_natural_head.detach().cpu().clone()
    zero = bool((fixed_delta == 0).all())
    start, stop = head * head_dim, (head + 1) * head_dim
    metadata = {
        "layer_index": layer,
        "head_index": head,
        "query_position": original_query_position,
        "head_dim": head_dim,
        "attention_width": width,
        "delta": fixed_delta,
        "reference_natural_head": reference,
        "delta_norm": float(torch.linalg.vector_norm(fixed_delta.to(torch.float64))),
        "drift_tolerance": drift_tolerance,
        "zero_delta": zero,
        "observations": [],
    }

    def intervene(module, inputs):
        if len(inputs) != 1:
            raise ValueError("Exactly one native projection input is required")
        hidden = inputs[0]
        if (
            hidden.dtype != torch.float32
            or hidden.ndim != 3
            or hidden.shape[0] != 1
            or hidden.shape[2] != width
            or original_query_position >= hidden.shape[1]
        ):
            raise ValueError(
                "A complete B1 FP32 projection input containing the original query is required"
            )
        natural = hidden[0, original_query_position, start:stop].detach().cpu().clone()
        drift = float(
            (natural.to(torch.float64) - reference.to(torch.float64)).abs().max()
        )
        metadata["observations"].append(
            {
                "sequence_length": hidden.shape[1],
                "query_position": original_query_position,
                "natural_head": natural,
                "max_abs_drift": drift,
            }
        )
        if not math.isfinite(drift) or drift > drift_tolerance:
            raise RuntimeError(
                "Natural original-query head drift exceeds the frozen tolerance"
            )
        if zero:
            return None
        modified = hidden.clone()
        modified[0, original_query_position, start:stop] += fixed_delta.to(
            hidden.device
        )
        return (modified,)

    handle = projection.register_forward_pre_hook(intervene)
    try:
        yield metadata
        if not metadata["observations"]:
            raise RuntimeError("The selected projection was never called")
    finally:
        handle.remove()
