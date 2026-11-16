"""Train-only low-rank coordinate maps and loader-independent causal patches.

No tokenizer, model weights, network access, or artifact writing lives here.
Entity identities are split before fitting; each training row is one entity,
not an inflated collection of dependent ordered pairs. All numerical fitting
uses detached CPU float64 vectors. Production callers must seal the returned
map records and the validation-selected configuration before touching test
outcomes. The patch executor uses one prompt per forward and qualifies exact
mathematical controls before permitting a substantive intervention.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from copy import deepcopy

import torch


def _ids(values, label):
    result = list(values)
    if (
        not result
        or any(not isinstance(value, str) or not value for value in result)
        or len(set(result)) != len(result)
    ):
        raise ValueError(f"{label} must contain distinct nonempty entity IDs")
    return result


def _vector(value, label):
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 1
        or not value.is_floating_point()
    ):
        raise ValueError(f"{label} must be a floating vector")
    value = value.detach().to(device="cpu", dtype=torch.float64).clone()
    if not value.numel() or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{label} must be finite and nonempty")
    return value


def _matrix(states, entities, label):
    if not isinstance(states, Mapping) or set(states) != set(entities):
        raise ValueError(f"{label} keys must equal exactly the declared entity set")
    vectors = [_vector(states[entity], label) for entity in entities]
    if len({tuple(value.shape) for value in vectors}) != 1:
        raise ValueError(f"{label} vector dimensions differ")
    return torch.stack(vectors)


def _tensor_digest(value):
    encoded = value.detach().cpu().contiguous()
    digest = hashlib.sha256(str((str(encoded.dtype), tuple(encoded.shape))).encode())
    digest.update(encoded.numpy().tobytes())
    return digest.hexdigest()


def alignment_digest(record):
    """A serializable map's exact factors and provenance have one stable digest."""
    projection = {
        key: {"tensor_sha256": _tensor_digest(value)}
        if isinstance(value, torch.Tensor)
        else value
        for key, value in record.items()
        if key != "alignment_sha256"
    }
    return hashlib.sha256(
        json.dumps(
            projection, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def fit_alignment(
    source_train,
    target_train,
    *,
    training_entity_ids,
    evaluation_entity_ids,
    rank,
    ridge_fraction,
    method="ridge",
    target_permutation=None,
):
    """Fit a map using ONLY exact training dictionaries, without answer labels.

    Ridge minimizes ||X W - Y||² + alpha ||W||² after centering training rows.
    alpha is the fixed fraction times the mean squared singular value across
    min(n-1, source_dimension) training modes, independent of retained rank.
    The orthogonal alternative is a truncated partial Procrustes map confined
    to the observed training subspaces: it never completes arbitrary ambient
    null directions into a full orthogonal matrix.
    """
    train = _ids(training_entity_ids, "training entities")
    evaluation = _ids(evaluation_entity_ids, "evaluation entities")
    if set(train) & set(evaluation):
        raise ValueError("training and evaluation entity sets must be disjoint")
    if len(train) < 2:
        raise ValueError("at least two training entities are required")
    if method not in ("ridge", "orthogonal"):
        raise ValueError("unknown alignment method")
    if (
        isinstance(ridge_fraction, bool)
        or not isinstance(ridge_fraction, (float, int))
        or not math.isfinite(ridge_fraction)
        or ridge_fraction < 0
        or (method == "ridge" and ridge_fraction == 0)
    ):
        raise ValueError("ridge fraction must be finite and positive for ridge")
    source = _matrix(source_train, train, "source training states")
    target = _matrix(target_train, train, "target training states")
    original_target_sha = _tensor_digest(target)
    maximum_rank = min(len(train) - 1, source.shape[1], target.shape[1])
    if type(rank) is not int or not 1 <= rank <= maximum_rank:
        raise ValueError("rank must fit the entity-centered training subspace")
    permutation = list(train if target_permutation is None else target_permutation)
    if len(permutation) != len(train) or set(permutation) != set(train):
        raise ValueError(
            "target permutation must bijectively permute training entities"
        )
    if target_permutation is not None:
        target = _matrix(target_train, permutation, "permuted target training states")
    source_mean, target_mean = source.mean(0), target.mean(0)
    x, y = source - source_mean, target - target_mean
    u, singular, vh = torch.linalg.svd(x, full_matrices=False)
    modes = min(len(train) - 1, source.shape[1])
    scale = float(singular[:modes].square().mean())
    alpha = float(ridge_fraction) * scale if method == "ridge" else 0.0
    if method == "ridge":
        source_basis = vh[:rank].T.contiguous()
        s = singular[:rank]
        denominator = s.square() + alpha
        multipliers = torch.where(denominator > 0, s / denominator, torch.zeros_like(s))
        target_factor = (multipliers[:, None] * (u[:, :rank].T @ y)).contiguous()
        retained_rank = rank
    else:
        uy, sy, vhy = torch.linalg.svd(y, full_matrices=False)
        source_modes, target_modes = modes, min(len(train) - 1, target.shape[1])
        middle = (
            singular[:source_modes, None]
            * (u[:, :source_modes].T @ uy[:, :target_modes])
            * sy[None, :target_modes]
        )
        left, values, right = torch.linalg.svd(middle, full_matrices=False)
        tolerance = (
            torch.finfo(torch.float64).eps * max(middle.shape) * float(values.max())
        )
        retained_rank = min(rank, int((values > tolerance).sum()))
        source_basis = (vh[:source_modes].T @ left[:, :retained_rank]).contiguous()
        target_factor = (right[:retained_rank] @ vhy[:target_modes]).contiguous()
    record = {
        "schema_version": "e30_14_train_only_alignment_v1",
        "method": method,
        "training_entity_ids": train,
        "evaluation_entity_ids": evaluation,
        "training_target_entity_order": permutation,
        "training_entity_count": len(train),
        "training_rows_are_entities_not_ordered_pairs": True,
        "source_dimension": source.shape[1],
        "target_dimension": target.shape[1],
        "requested_rank": rank,
        "retained_rank": retained_rank,
        "ridge_fraction": float(ridge_fraction),
        "ridge_alpha": alpha,
        "ridge_scale": scale,
        "ridge_scale_mode_count": modes,
        "ridge_scale_rule": "mean squared singular value of all min(n-1,Dsource) centered training modes",
        "source_training_matrix_sha256": _tensor_digest(source),
        "target_training_matrix_sha256": _tensor_digest(target),
        "unpermuted_target_training_matrix_sha256": original_target_sha,
        "source_mean": source_mean,
        "target_mean": target_mean,
        "source_basis": source_basis,
        "target_factor": target_factor,
        "training_source_singular_values": singular,
        "arbitrary_ambient_nullspace_completion": False,
        "answer_tokens_or_correctness_labels_used": False,
    }
    record["alignment_sha256"] = alignment_digest(record)
    return record


def apply_alignment(fit, source_delta, *, base_entity_id, donor_entity_id):
    """Map one held-out entity difference; an intercept necessarily cancels."""
    evaluation = set(fit["evaluation_entity_ids"])
    if (
        base_entity_id == donor_entity_id
        or {base_entity_id, donor_entity_id} - evaluation
    ):
        raise ValueError(
            "difference endpoints must be distinct declared evaluation entities"
        )
    if {base_entity_id, donor_entity_id} & set(fit["training_entity_ids"]):
        raise ValueError("training entities cannot be reported as held-out endpoints")
    delta = _vector(source_delta, "source difference")
    if delta.numel() != fit["source_dimension"]:
        raise ValueError("source difference dimension changed")
    result = (delta @ fit["source_basis"]) @ fit["target_factor"]
    if result.shape != (fit["target_dimension"],) or not bool(
        torch.isfinite(result).all()
    ):
        raise ValueError("invalid aligned difference")
    return result


def score_validation(
    fit, source_validation, target_validation, *, validation_entity_ids, test_entity_ids
):
    """Report the fixed geometric validation objective, never test accuracy."""
    validation = _ids(validation_entity_ids, "validation entities")
    test = _ids(test_entity_ids, "test entities")
    if set(validation) & set(test) or (set(validation) | set(test)) & set(
        fit["training_entity_ids"]
    ):
        raise ValueError("fit, validation, and test entity sets must be disjoint")
    if set(validation) | set(test) != set(fit["evaluation_entity_ids"]):
        raise ValueError(
            "validation/test sets must partition declared evaluation entities"
        )
    source = _matrix(source_validation, validation, "source validation states")
    target = _matrix(target_validation, validation, "target validation states")
    error = energy = 0.0
    for base, base_id in enumerate(validation):
        for donor, donor_id in enumerate(validation):
            if base == donor:
                continue
            prediction = apply_alignment(
                fit,
                source[donor] - source[base],
                base_entity_id=base_id,
                donor_entity_id=donor_id,
            )
            expected = target[donor] - target[base]
            error += float((prediction - expected).square().sum())
            energy += float(expected.square().sum())
    return {
        "scored_entity_ids": validation,
        "ordered_pair_count": len(validation) * (len(validation) - 1),
        "squared_error_sum": error,
        "target_difference_energy_sum": energy,
        "relative_squared_error": None if energy == 0 else error / energy,
        "test_outcomes_used": False,
    }


def make_intervention(
    condition,
    *,
    fit,
    source_delta,
    base_entity_id,
    donor_entity_id,
    same_form_delta=None,
    wrong_source_delta=None,
    wrong_donor_entity_id=None,
    permutation_fit=None,
    random_seed=0,
    gain=1.0,
):
    """Build a target-coordinate perturbation with explicit, matched doses."""
    if (
        isinstance(gain, bool)
        or not isinstance(gain, (float, int))
        or not math.isfinite(gain)
        or gain <= 0
    ):
        raise ValueError("gain must be positive and finite")
    aligned = apply_alignment(
        fit,
        source_delta,
        base_entity_id=base_entity_id,
        donor_entity_id=donor_entity_id,
    )
    desired_norm = float(torch.linalg.vector_norm(aligned))
    norm_match = condition in ("random", "wrong_donor", "permuted_dosematched")
    if condition == "aligned":
        delta = aligned
    elif condition == "reverse":
        delta = -aligned
    elif condition == "unmapped":
        delta = _vector(source_delta, "unmapped difference")
    elif condition == "same_form":
        delta = _vector(same_form_delta, "same-form difference")
    elif condition in ("permuted", "permuted_dosematched"):
        fixed_fields = (
            "training_entity_ids",
            "evaluation_entity_ids",
            "method",
            "source_training_matrix_sha256",
            "source_dimension",
            "target_dimension",
            "unpermuted_target_training_matrix_sha256",
        )
        if permutation_fit is None or any(
            permutation_fit[key] != fit[key] for key in fixed_fields
        ):
            raise ValueError(
                "permutation control must use the same training data, panel, method and dimensions"
            )
        delta = apply_alignment(
            permutation_fit,
            source_delta,
            base_entity_id=base_entity_id,
            donor_entity_id=donor_entity_id,
        )
    elif condition == "wrong_donor":
        if wrong_donor_entity_id in (base_entity_id, donor_entity_id):
            raise ValueError("wrong donor must differ from both intended endpoints")
        delta = apply_alignment(
            fit,
            wrong_source_delta,
            base_entity_id=base_entity_id,
            donor_entity_id=wrong_donor_entity_id,
        )
    elif condition == "random":
        if type(random_seed) is not int or not 0 <= random_seed < 2**63:
            raise ValueError("random control requires a registered integer seed")
        delta = torch.randn(
            aligned.shape,
            dtype=torch.float64,
            generator=torch.Generator(device="cpu").manual_seed(random_seed),
        )
    else:
        raise ValueError("unknown intervention condition")
    if delta.shape != aligned.shape:
        raise ValueError("intervention target dimension differs")
    original_norm = float(torch.linalg.vector_norm(delta))
    degenerate = norm_match and (original_norm == 0 or desired_norm == 0)
    if norm_match and original_norm > 0:
        delta = delta * (desired_norm / original_norm)
    delta = delta * float(gain)
    return {
        "condition": condition,
        "delta": delta,
        "gain": float(gain),
        "aligned_unit_gain_norm": desired_norm,
        "actual_norm": float(torch.linalg.vector_norm(delta)),
        "dose_matched_to_aligned": norm_match,
        "degenerate_control": degenerate,
        "oracle_target_state_used_to_rescale": False,
    }


class ResidualPatchExecutor:
    """Capture/patch supplied decoder blocks, with exact controls first.

    GPT-2 and Qwen-style models use different module paths; callers supply the
    ordered blocks explicitly. Multi-token names are represented by a verified
    common marker token AFTER the entire name, not by an arbitrary subword.
    Batch size is always one. Keeping only the final LM-head position also
    avoids head matrix-shape changes when donor/base prompt lengths differ.
    """

    def __init__(
        self, model, block_modules, *, control_tolerance=1e-4, forward_kwargs=None
    ):
        self.model, self.blocks = model, list(block_modules)
        if not self.blocks or len({id(block) for block in self.blocks}) != len(
            self.blocks
        ):
            raise ValueError("ordered decoder blocks must be distinct and nonempty")
        modules = {id(module) for module in model.modules()}
        if any(id(block) not in modules for block in self.blocks):
            raise ValueError("decoder blocks must belong to the supplied model")
        if (
            isinstance(control_tolerance, bool)
            or not isinstance(control_tolerance, (float, int))
            or not math.isfinite(control_tolerance)
            or control_tolerance <= 0
        ):
            raise ValueError("control tolerance must be finite and positive")
        self.tolerance = float(control_tolerance)
        self.forward_kwargs = (
            {"logits_to_keep": 1} if forward_kwargs is None else dict(forward_kwargs)
        )
        if set(self.forward_kwargs) & {
            "input_ids",
            "attention_mask",
            "use_cache",
            "return_dict",
        }:
            raise ValueError("fixed execution arguments cannot be overridden")
        self._prompts, self._states, self._clean_logits = {}, {}, {}
        self._qualified_pairs = set()
        self._capture_started = self._capture_complete = self._oracle_failed = False
        self.physical_forward_count = 0
        self._require_model()
        parameter = next(model.parameters())
        self.device, self.dtype = parameter.device, parameter.dtype
        self._parameter_fingerprint = self._fingerprint()

    def _fingerprint(self):
        return tuple(
            (name, id(value), value._version, tuple(value.shape))
            for name, value in self.model.named_parameters()
        )

    def _require_model(self):
        if self.model is None or any(
            module.training for module in self.model.modules()
        ):
            raise ValueError("model must be in evaluation mode")
        parameters = list(self.model.parameters())
        if not parameters:
            raise ValueError("model requires frozen parameters")
        device, dtype = parameters[0].device, parameters[0].dtype
        if (device.type, dtype) not in (
            ("cpu", torch.float64),
            ("cuda", torch.float32),
        ):
            raise ValueError("model must be frozen CPU float64 or CUDA float32")
        if any(
            value.requires_grad or value.device != device or value.dtype != dtype
            for value in parameters
        ):
            raise ValueError("parameters must be frozen on one device/dtype")
        for value in self.model.buffers():
            if value.device != device or (
                value.is_floating_point() and value.dtype != dtype
            ):
                raise ValueError("model buffers must share parameter device/dtype")
        for name in (
            "_global_forward_hooks",
            "_global_forward_pre_hooks",
            "_global_backward_hooks",
            "_global_backward_pre_hooks",
        ):
            if getattr(torch.nn.modules.module, name, {}):
                raise RuntimeError("global hooks are prohibited")
        for module in self.model.modules():
            if any(
                getattr(module, name, {})
                for name in (
                    "_forward_hooks",
                    "_forward_pre_hooks",
                    "_backward_hooks",
                    "_backward_pre_hooks",
                )
            ):
                raise RuntimeError("preexisting model hooks are prohibited")
        if (
            hasattr(self, "_parameter_fingerprint")
            and self._fingerprint() != self._parameter_fingerprint
        ):
            raise RuntimeError("model parameters changed")

    def _site(self, layer_index, token_site):
        if (
            type(layer_index) is not int
            or not 0 <= layer_index < len(self.blocks)
            or token_site not in ("marker", "query")
        ):
            raise ValueError("invalid residual site")
        return f"residual::{layer_index}::{token_site}"

    def _forward(self, row, *, capture=False, patch=None):
        self._require_model()
        handles, states, calls = [], {}, {}
        layers = range(len(self.blocks)) if capture else (patch["layer_index"],)
        try:
            for layer in layers:
                calls[layer] = 0

                def hook(_module, _inputs, output, *, layer=layer):
                    calls[layer] += 1
                    hidden = output[0] if isinstance(output, tuple) else output
                    if (
                        not isinstance(hidden, torch.Tensor)
                        or hidden.ndim != 3
                        or tuple(hidden.shape[:2]) != (1, len(row["input_ids"]))
                        or hidden.device != self.device
                        or hidden.dtype != self.dtype
                        or not bool(torch.isfinite(hidden).all())
                        or calls[layer] != 1
                    ):
                        raise RuntimeError("invalid supplied decoder-block boundary")
                    if capture:
                        for token_site in ("marker", "query"):
                            states[self._site(layer, token_site)] = (
                                hidden[0, row[token_site + "_position"]]
                                .detach()
                                .cpu()
                                .clone()
                            )
                        return output
                    position = row[patch["token_site"] + "_position"]
                    value = (
                        patch["replacement"]
                        if patch["replacement"] is not None
                        else hidden[0, position]
                        + patch["delta"].to(device=self.device, dtype=self.dtype)
                    )
                    value = value.to(device=self.device, dtype=self.dtype)
                    if value.shape != hidden[0, position].shape or not bool(
                        torch.isfinite(value).all()
                    ):
                        raise ValueError("invalid replacement residual vector")
                    changed = hidden.clone()
                    changed[0, position] = value
                    return (
                        (changed, *output[1:]) if isinstance(output, tuple) else changed
                    )

                handles.append(self.blocks[layer].register_forward_hook(hook))
            with torch.inference_mode():
                input_ids = torch.tensor(
                    [row["input_ids"]], dtype=torch.long, device=self.device
                )
                self.physical_forward_count += 1
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    use_cache=False,
                    return_dict=True,
                    **self.forward_kwargs,
                )
                logits = (
                    output.logits if hasattr(output, "logits") else output["logits"]
                )
                if (
                    not isinstance(logits, torch.Tensor)
                    or logits.ndim != 3
                    or logits.shape[0] != 1
                    or logits.shape[1] not in (1, len(row["input_ids"]))
                    or logits.dtype != self.dtype
                    or logits.device != self.device
                    or not bool(torch.isfinite(logits).all())
                ):
                    raise RuntimeError("invalid first-answer-token logits")
                if (
                    self.forward_kwargs.get("logits_to_keep") == 1
                    and logits.shape[1] != 1
                ):
                    raise RuntimeError(
                        "model ignored the required single-position LM-head argument"
                    )
                endpoint = logits[0, -1].detach().cpu().clone()
            if any(value != 1 for value in calls.values()):
                raise RuntimeError(
                    "selected decoder block did not execute exactly once"
                )
        finally:
            for handle in handles:
                handle.remove()
        self._require_model()
        return endpoint, states

    def capture_prompts(self, prompts):
        if self._capture_started:
            raise RuntimeError("clean capture is one-shot")
        qualified, marker_ids = {}, set()
        for original in prompts:
            row = deepcopy(dict(original))
            identity, ids = row["prompt_id"], row["input_ids"]
            marker, query, marker_id = (
                row["marker_position"],
                row["query_position"],
                row["marker_token_id"],
            )
            if (
                not isinstance(identity, str)
                or not identity
                or identity in qualified
                or not ids
                or any(type(token) is not int or token < 0 for token in ids)
            ):
                raise ValueError("invalid prompt identity or input IDs")
            if (
                type(marker) is not int
                or not 0 <= marker < len(ids) - 1
                or query != len(ids) - 1
                or type(marker_id) is not int
                or ids[marker] != marker_id
            ):
                raise ValueError(
                    "marker/query positions do not match verified prompt tokens"
                )
            marker_ids.add(marker_id)
            qualified[identity] = row
        if not qualified or len(marker_ids) != 1:
            raise ValueError("all prompts need one shared verified marker token")
        self._prompts, self._capture_started = qualified, True
        for identity, row in self._prompts.items():
            endpoint, states = self._forward(row, capture=True)
            self._clean_logits[identity], self._states[identity] = endpoint, states
        self._capture_complete = True
        return {
            "states": deepcopy(self._states),
            "clean_logits": deepcopy(self._clean_logits),
            "physical_forward_count": self.physical_forward_count,
        }

    def qualify_oracles(self, *, sites, prompt_ids=None, donor_pairs=None):
        if self._oracle_failed:
            raise RuntimeError("a failed oracle phase cannot be resumed")
        if not self._capture_complete:
            raise RuntimeError("clean capture must precede oracle qualification")
        identities = list(self._prompts if prompt_ids is None else prompt_ids)
        if (
            not identities
            or len(set(identities)) != len(identities)
            or set(identities) - set(self._prompts)
        ):
            raise ValueError("oracle prompt panel changed")
        selected = {(layer, token_site) for layer, token_site in sites}
        if not selected:
            raise ValueError("at least one oracle site is required")
        for layer, token_site in selected:
            self._site(layer, token_site)
        pairs = (
            list(zip(identities, identities[1:] + identities[:1]))
            if donor_pairs is None
            else list(donor_pairs)
        )
        if not pairs or any(
            base == donor or base not in identities or donor not in identities
            for base, donor in pairs
        ):
            raise ValueError(
                "oracle donor pairs must belong to the qualified prompt panel"
            )
        before = self.physical_forward_count
        self_error = final_error = 0.0
        try:
            for identity in identities:
                for layer, token_site in sorted(selected):
                    site = self._site(layer, token_site)
                    endpoint, _ = self._forward(
                        self._prompts[identity],
                        patch={
                            "layer_index": layer,
                            "token_site": token_site,
                            "replacement": self._states[identity][site],
                            "delta": None,
                        },
                    )
                    self_error = max(
                        self_error,
                        float(
                            (endpoint.double() - self._clean_logits[identity].double())
                            .abs()
                            .max()
                        ),
                    )
            layer = len(self.blocks) - 1
            for base, donor in pairs:
                endpoint, _ = self._forward(
                    self._prompts[base],
                    patch={
                        "layer_index": layer,
                        "token_site": "query",
                        "replacement": self._states[donor][self._site(layer, "query")],
                        "delta": None,
                    },
                )
                final_error = max(
                    final_error,
                    float(
                        (endpoint.double() - self._clean_logits[donor].double())
                        .abs()
                        .max()
                    ),
                )
            if max(self_error, final_error) > self.tolerance:
                raise RuntimeError(
                    f"mathematical oracle failed: self={self_error}, final_query={final_error}, tolerance={self.tolerance}"
                )
        except BaseException:
            self._oracle_failed = True
            raise
        self._qualified_pairs |= {
            (identity, layer, token_site)
            for identity in identities
            for layer, token_site in selected
        }
        return {
            "passed": True,
            "self_patch_max_abs_error": self_error,
            "final_query_matched_max_abs_error": final_error,
            "control_tolerance": self.tolerance,
            "physical_forward_count": self.physical_forward_count - before,
            "prompt_ids": identities,
            "sites": [list(site) for site in sorted(selected)],
            "final_query_replacement_is_structural_not_semantic": True,
        }

    def run_patch(
        self, prompt_id, *, layer_index, token_site, delta=None, replacement=None
    ):
        self._site(layer_index, token_site)
        if (
            self._oracle_failed
            or (prompt_id, layer_index, token_site) not in self._qualified_pairs
        ):
            raise RuntimeError(
                "matching mathematical oracles must pass before any causal patch"
            )
        if (delta is None) == (replacement is None):
            raise ValueError("supply exactly one of delta or replacement")
        patch = {
            "layer_index": layer_index,
            "token_site": token_site,
            "delta": None if delta is None else _vector(delta, "patch difference"),
            "replacement": None
            if replacement is None
            else _vector(replacement, "replacement state"),
        }
        endpoint, _ = self._forward(self._prompts[prompt_id], patch=patch)
        return endpoint

    def close(self):
        self._states.clear()
        self._clean_logits.clear()
        self._prompts.clear()
        self.model = None
