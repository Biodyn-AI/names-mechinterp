"""Field-blind backbone captures and single-site attention-input replacement.

These helpers contain no model loading, resource provisioning, or saved-logit
executor. Prefix captures are backbone-only forwards, not vocabulary endpoints.
Replacement callers must recompute complete sequences with use_cache=False.
"""

from contextlib import contextmanager

import torch


def _position(value, length=None):
    if type(value) is not int or value < 0 or (length is not None and value >= length):
        raise ValueError("marker position must be a valid nonnegative integer")


def _target(model, layer_index):
    """Use actual attention-head width, which can differ from hidden_size."""
    layers = getattr(getattr(model, "model", None), "layers", ())
    if type(layer_index) is not int or not 0 <= layer_index < len(layers) - 1:
        raise ValueError("layer index must precede the final decoder layer")
    if any(module.training for module in model.modules()):
        raise ValueError("prefix hooks require evaluation mode")
    attention = layers[layer_index].self_attn
    projection = attention.o_proj
    heads, head_dim, width = (
        model.config.num_attention_heads,
        attention.head_dim,
        projection.in_features,
    )
    if (
        any(type(value) is not int or value <= 0 for value in (heads, head_dim, width))
        or width != heads * head_dim
    ):
        raise ValueError("attention input width must equal head count times head_dim")
    return projection, heads, head_dim, width


def _projection_input(arguments, width, position):
    if not arguments or not isinstance(arguments[0], torch.Tensor):
        raise TypeError("attention projection requires a tensor positional input")
    value = arguments[0]
    if (
        value.dtype != torch.float32
        or value.ndim != 3
        or value.shape[0] != 1
        or value.shape[2] != width
    ):
        raise ValueError(
            "attention input must have float32 B1/sequence/head-width geometry"
        )
    _position(position, value.shape[1])
    if not bool(torch.isfinite(value).all()):
        raise ValueError("attention projection input must be finite")
    return value


def capture_prefix(model, input_ids, alias_end_position, layer_indices):
    """Capture pre-o_proj concatenated heads after strictly cutting the future.

    The caller supplies a tokenizer-verified marker boundary. The requested
    attribute and every later token are excluded from the backbone forward.
    """
    if isinstance(input_ids, torch.Tensor):
        if (
            input_ids.dtype != torch.long
            or input_ids.ndim not in (1, 2)
            or (input_ids.ndim == 2 and input_ids.shape[0] != 1)
        ):
            raise ValueError("input IDs must be a long vector or B1 matrix")
        input_ids = input_ids.detach().cpu().reshape(-1).tolist()
    if (
        not isinstance(input_ids, (list, tuple))
        or not input_ids
        or any(type(token) is not int or token < 0 for token in input_ids)
    ):
        raise ValueError("input IDs must contain nonnegative integers")
    _position(alias_end_position, len(input_ids))
    indices = list(layer_indices)
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("capture layers must be nonempty and distinct")
    targets = {index: _target(model, index) for index in indices}
    geometries = {values[1:] for values in targets.values()}
    if len(geometries) != 1:
        raise ValueError("requested layers have inconsistent attention geometry")
    heads, head_dim, width = next(iter(geometries))
    cut_ids, captured, handles = list(input_ids[: alias_end_position + 1]), {}, []
    parameter = next(model.parameters(), None)
    if parameter is None or parameter.dtype != torch.float32:
        raise ValueError("backbone parameters must have an explicit float32 device")
    prefix_tensor = torch.tensor([cut_ids], dtype=torch.long, device=parameter.device)
    try:
        for index, (projection, _, _, _) in targets.items():

            def save(module, arguments, layer=index):
                if layer in captured:
                    raise RuntimeError("capture projection executed more than once")
                value = _projection_input(arguments, width, alias_end_position)
                captured[layer] = value[0, alias_end_position].detach().cpu().clone()

            handles.append(projection.register_forward_pre_hook(save))
        with torch.inference_mode():
            model.model(
                input_ids=prefix_tensor,
                attention_mask=torch.ones_like(prefix_tensor),
                use_cache=False,
                return_dict=True,
            )
        if set(captured) != set(indices):
            raise RuntimeError("not every declared projection executed")
    finally:
        for handle in handles:
            handle.remove()
    return {
        "input_ids": cut_ids,
        "alias_end_position": alias_end_position,
        "prefix_length": len(cut_ids),
        "num_attention_heads": heads,
        "head_dim": head_dim,
        "layers": captured,
        "forward_scope": "backbone_prefix_only_no_logits",
    }


@contextmanager
def replace_at_marker(
    model, layer_index, marker_position, donor_vector, head_index=None
):
    """Replace exactly one token's full attention input or one head slice.

    A copied donor is reused on every full-sequence recomputation in the context.
    Inputs outside the selected token/head remain untouched, including storage.
    """
    projection, heads, head_dim, width = _target(model, layer_index)
    _position(marker_position)
    if head_index is not None and (
        type(head_index) is not int or not 0 <= head_index < heads
    ):
        raise ValueError("head index lies outside the attention head count")
    expected_width = width if head_index is None else head_dim
    if (
        not isinstance(donor_vector, torch.Tensor)
        or donor_vector.dtype != torch.float32
        or tuple(donor_vector.shape) != (expected_width,)
        or not bool(torch.isfinite(donor_vector).all())
    ):
        raise ValueError(
            "donor must be a finite float32 vector of the exact selected width"
        )
    donor = donor_vector.detach().clone()
    start = 0 if head_index is None else head_index * head_dim

    def replace(module, arguments):
        original = _projection_input(arguments, width, marker_position)
        changed = original.clone()
        changed[0, marker_position, start : start + expected_width] = donor.to(
            device=original.device
        )
        return (changed, *arguments[1:])

    handle = projection.register_forward_pre_hook(replace)
    try:
        yield
    finally:
        handle.remove()
