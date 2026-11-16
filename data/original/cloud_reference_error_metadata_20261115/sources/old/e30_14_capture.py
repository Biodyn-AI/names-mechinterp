"""Model-free token qualification and test-blind fit/validation capture.

Loading is local-only and happens only when a caller explicitly invokes the
loader. Token preparation and disk planning never load weights. Capture and
map selection reject every test row; callers must publish the selected maps
before constructing a separate executor for the held-out causal experiments.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path

import torch

import e30_14_alignment as alignment
import e30_14_protocol as protocol

MODEL_IDS = ("openai-community/gpt2-medium", "Qwen/Qwen3-4B-Instruct-2507")
DIRECTIONS = (("full", "abbr"), ("abbr", "full"))


def _digest(record):
    return hashlib.sha256(
        json.dumps(
            record, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _encode(tokenizer, text):
    value = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(value["input_ids"])
    offsets = [list(pair) for pair in value["offset_mapping"]]
    if (
        not ids
        or len(ids) != len(offsets)
        or any(type(token) is not int or token < 0 for token in ids)
    ):
        raise ValueError("invalid tokenizer IDs or offsets")
    return ids, offsets


def prepare_token_rows(tokenizer, prompts, *, chat_model, facts):
    """Freeze all tokens, complete-name spans, marker positions and answers."""
    if type(chat_model) is not bool:
        raise ValueError("chat_model must explicitly select the rendering contract")
    rows, identities, marker_ids = [], set(), set()
    for original in prompts:
        row = deepcopy(dict(original))
        identity, name = row["prompt_id"], row["entity_text"]
        if identity in identities:
            raise ValueError("duplicated prompt identity")
        answer = facts[row["entity_id"]][row["relation_id"]]
        if (
            not isinstance(answer, str)
            or not answer
            or row.get("answer") not in (None, answer)
        ):
            raise ValueError("prompt answer differs from the fixed canonical facts")
        if chat_model:
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            rendered = row["prompt"]
        if not isinstance(rendered, str):
            raise TypeError("chat template must return the exact rendered text")
        matches = list(re.finditer(r"(?<!\w)" + re.escape(name) + r"(?!\w)", rendered))
        if len(matches) != 1:
            raise ValueError("complete entity name must occur exactly once")
        start, end = matches[0].span()
        ids, offsets = _encode(tokenizer, rendered)
        positions = [
            index
            for index, (left, right) in enumerate(offsets)
            if left < end and right > start
        ]
        if (
            not positions
            or positions != list(range(positions[0], positions[-1] + 1))
            or offsets[positions[-1]][1] != end
            or rendered[offsets[positions[0]][0] : end].strip() != name
        ):
            raise ValueError("token offsets do not cover the complete entity name")
        marker = positions[-1] + 1
        marker_text = row.get("entity_marker_text", "?")
        if (
            marker >= len(ids) - 1
            or offsets[marker][0] != end
            or not rendered[offsets[marker][0] : offsets[marker][1]].startswith(
                marker_text
            )
        ):
            raise ValueError(
                "a separate common marker token must immediately follow the entire name"
            )
        prefix = "" if chat_model else " "
        appended_ids, _ = _encode(tokenizer, rendered + prefix + answer)
        if appended_ids[: len(ids)] != ids or len(appended_ids) <= len(ids):
            raise ValueError("answer appending changed the frozen prompt-token prefix")
        answer_ids = appended_ids[len(ids) :]
        row.update(
            {
                "rendered_prompt": rendered,
                "rendered_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
                "chat_model": chat_model,
                "chat_template_sha256": _digest(
                    getattr(tokenizer, "chat_template", None)
                )
                if chat_model
                else None,
                "input_ids": ids,
                "offset_mapping": offsets,
                "position_ids": list(range(len(ids))),
                "entity_char_span": [start, end],
                "entity_positions": positions,
                "entity_token_ids": [ids[index] for index in positions],
                "marker_position": marker,
                "query_position": len(ids) - 1,
                "marker_token_id": ids[marker],
                "marker_token_text": tokenizer.decode(
                    [ids[marker]], clean_up_tokenization_spaces=False
                ),
                "answer": answer,
                "answer_prefix": prefix,
                "answer_ids": answer_ids,
                "answer_first_token_id": answer_ids[0],
            }
        )
        marker_ids.add(ids[marker])
        identities.add(identity)
        rows.append(row)
    if not rows or len(marker_ids) != 1:
        raise ValueError(
            "all prompts must share exactly the same post-name marker token"
        )
    return rows


def length_signature(row):
    return (
        len(row["entity_positions"]),
        row["marker_position"],
        row["query_position"],
        len(row["input_ids"]),
    )


def qualify_comparison_cells(rows, facts_record):
    """Resolve length nulls and factual/token strata without model outputs."""
    lookup = {row["prompt_id"]: row for row in rows}
    by_role = {
        (
            row["split"],
            row["frame"],
            row["form"],
            row["relation_id"],
            row["entity_id"],
        ): row
        for row in rows
    }
    matched = {}
    matched_rows = {}
    for cell in protocol.build_comparison_cells():
        source = lookup[cell["source_donor_prompt_id"]]
        target = lookup[cell["donor_prompt_id"]]
        selected = None
        for entity in sorted(protocol.build_splits()["test"]):
            if entity in (cell["base_entity_id"], cell["donor_entity_id"]):
                continue
            wrong_source = by_role[
                ("test", "A", cell["source_form"], cell["source_relation_id"], entity)
            ]
            wrong_target = by_role[
                ("test", "C", cell["target_form"], cell["target_relation_id"], entity)
            ]
            if length_signature(wrong_source) == length_signature(
                source
            ) and length_signature(wrong_target) == length_signature(target):
                selected = entity
                matched_rows[cell["cell_id"]] = (
                    wrong_source["prompt_id"],
                    wrong_target["prompt_id"],
                )
                break
        matched[cell["cell_id"]] = selected
    annotated = protocol.annotate_comparison_cells(
        facts_record,
        answer_first_token_ids={
            identity: row["answer_first_token_id"] for identity, row in lookup.items()
        },
        length_matched_wrong_entities=matched,
    )
    for cell in annotated:
        pair = matched_rows.get(cell["cell_id"])
        cell["length_matched_wrong_source_donor_prompt_id"] = (
            None if pair is None else pair[0]
        )
        cell["length_matched_wrong_target_donor_prompt_id"] = (
            None if pair is None else pair[1]
        )
        cell["length_matched_wrong_available"] = pair is not None
        cell["source_target_positions_equal"] = length_signature(
            lookup[cell["source_base_prompt_id"]]
        ) == length_signature(
            lookup[cell["source_donor_prompt_id"]]
        ) and length_signature(lookup[cell["base_prompt_id"]]) == length_signature(
            lookup[cell["donor_prompt_id"]]
        )
    return annotated


def _dimensions(config):
    config = config.to_dict() if hasattr(config, "to_dict") else dict(config)
    values = (
        config.get("vocab_size"),
        config.get("hidden_size", config.get("n_embd")),
        config.get("num_hidden_layers", config.get("n_layer")),
    )
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError(
            "model config needs explicit positive vocabulary, width and layer counts"
        )
    return values


def make_token_manifest(rows, facts_record, *, model_id, model_config):
    """Return JSON metadata only; the caller publishes it before weight calls."""
    if model_id not in MODEL_IDS:
        raise ValueError("model is outside the fixed prospective panel")
    expected = {row["prompt_id"] for row in protocol.build_prompts()}
    if not expected <= {row["prompt_id"] for row in rows}:
        raise ValueError("token manifest must contain all 224 experimental prompts")
    vocab, width, layers = _dimensions(model_config)
    if any(max(row["input_ids"] + row["answer_ids"]) >= vocab for row in rows):
        raise ValueError("token ID exceeds the registered LM-head vocabulary")
    cells = qualify_comparison_cells(rows, facts_record)
    record = {
        "schema_version": "e30_14_model_free_tokens_v1",
        "model_id": model_id,
        "prompt_count": len(rows),
        "rows": deepcopy(list(rows)),
        "vocab_size": vocab,
        "hidden_size": width,
        "layer_count": layers,
        "marker_token_id": rows[0]["marker_token_id"],
        "comparison_cells": cells,
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "primary_question_arm_count": sum(
            cell["primary_question_arm"] for cell in cells
        ),
        "primary_fact_distinct_count": sum(cell["primary_stratum"] for cell in cells),
        "primary_first_token_distinct_count": sum(
            cell["first_token_distinct_stratum"] for cell in cells
        ),
        "length_matched_wrong_available_count": sum(
            cell["length_matched_wrong_available"] for cell in cells
        ),
        "model_outputs_used": False,
    }
    json.dumps(record, allow_nan=False)
    return record


def estimate_storage(model_configs, *, protocol_record=None):
    """Conservative native-logit/state/map plan, before deduplication savings."""
    design = protocol.build_protocol() if protocol_record is None else protocol_record
    components = {}
    for model_id, config in model_configs.items():
        if model_id not in MODEL_IDS:
            raise ValueError("storage plan model is outside the fixed panel")
        vocab, width, layers = _dimensions(config)
        prompts = design["prompt_count"] + design["qualification_prompt_count"]
        maps = (
            len(design["primary_site_families"])
            * len(DIRECTIONS)
            * (3 + design["shuffle_count"])
        )
        components[model_id] = {
            "logical_endpoint_bytes": design["logical_endpoint_count_per_model"]
            * vocab
            * 4,
            "operand_plan_bytes_upper_bound": design["logical_endpoint_count_per_model"]
            * width
            * 8,
            "clean_prompt_logits_bytes": prompts * vocab * 4,
            "all_clean_residual_bytes": prompts * 2 * layers * width * 4,
            "selected_map_bytes_upper_bound": maps
            * (
                2 * width * max(design["rank_grid"]) * 8
                + 2 * width * 8
                + len(design["splits"]["fit"]) * 8
            ),
            "metadata_allowance_bytes": 10 * 1024**2,
        }
    subtotal = sum(sum(values.values()) for values in components.values())
    total = math.ceil(subtotal * 1.05)
    limit = design["raw_storage_limit_bytes"]
    if total > limit:
        raise ValueError(
            f"planned storage {total} bytes exceeds the registered {limit}-byte storage limit"
        )
    return {
        "models": components,
        "subtotal_bytes": subtotal,
        "serialization_margin_fraction": 0.05,
        "estimated_total_bytes": total,
        "limit_bytes": limit,
        "within_limit": True,
        "deduplication_savings_assumed": False,
    }


def model_blocks(model, model_id):
    if model_id == MODEL_IDS[0]:
        blocks = list(model.transformer.h)
    elif model_id == MODEL_IDS[1]:
        blocks = list(model.model.layers)
    else:
        raise ValueError("model is outside the fixed prospective panel")
    if not blocks:
        raise ValueError("model has no decoder blocks")
    return blocks


def load_local_model(snapshot, model_id, *, device="cuda:0"):
    """Explicit caller-only weight loading; no download or remote code."""
    if model_id not in MODEL_IDS:
        raise ValueError("model is outside the fixed prospective panel")
    from transformers import AutoModelForCausalLM

    model = (
        AutoModelForCausalLM.from_pretrained(
            str(Path(snapshot)),
            local_files_only=True,
            token=False,
            trust_remote_code=False,
            use_safetensors=True,
            dtype=torch.float32,
            attn_implementation="eager",
        )
        .to(device=device, dtype=torch.float32)
        .eval()
        .requires_grad_(False)
    )
    model.config.use_cache = False
    model_blocks(model, model_id)
    return model


def _training_rows(rows):
    expected = {
        row["prompt_id"]: row
        for row in protocol.build_prompts()
        if row["split"] in ("fit", "validation")
    }
    actual = {row["prompt_id"]: row for row in rows}
    if len(actual) != len(rows) or set(actual) != set(expected):
        raise ValueError(
            "capture/selection accepts exactly the fit and validation rows, never test rows"
        )
    for identity, row in actual.items():
        for key in (
            "split",
            "entity_id",
            "entity_text",
            "form",
            "frame",
            "relation_id",
            "prompt",
        ):
            if row[key] != expected[identity][key]:
                raise ValueError("fit/validation prompt semantics changed")
    return actual


def capture_stage(model, rows, *, model_id, control_tolerance=1e-4):
    _training_rows(rows)
    executor = alignment.ResidualPatchExecutor(
        model, model_blocks(model, model_id), control_tolerance=control_tolerance
    )
    return executor, executor.capture_prompts(rows)


def _combine_validation(results):
    error = sum(result["squared_error_sum"] for result in results.values())
    energy = sum(result["target_difference_energy_sum"] for result in results.values())
    return {
        "squared_error_sum": error,
        "target_difference_energy_sum": energy,
        "relative_squared_error": None if energy == 0 else error / energy,
        "directions": results,
    }


def select_maps(captured_states, rows, *, protocol_record=None):
    """Select on six validation states; test activations cannot enter this API."""
    lookup = _training_rows(rows)
    if set(captured_states) != set(lookup):
        raise ValueError(
            "captured states must contain exactly fit and validation prompts"
        )
    design = protocol.build_protocol() if protocol_record is None else protocol_record
    splits = protocol.build_splits()
    if design["splits"] != splits:
        raise ValueError("predeclared entity splits changed")
    site_ids = set(next(iter(captured_states.values())))
    if any(set(states) != site_ids for states in captured_states.values()):
        raise ValueError("captured layer/site panels differ across prompts")
    layers = sorted({int(site.split("::")[1]) for site in site_ids})
    expected_sites = {
        f"residual::{layer}::{site}" for layer in layers for site in ("marker", "query")
    }
    if layers != list(range(len(layers))) or site_ids != expected_sites:
        raise ValueError("capture must cover every post-block marker/query site")
    role = {
        (row["split"], row["frame"], row["form"], row["entity_id"]): identity
        for identity, row in lookup.items()
    }

    def candidate(
        layer,
        site,
        rank,
        fraction,
        permutation=None,
        method="ridge",
        identity_correction=False,
    ):
        fits, scores = {}, {}
        for source_form, target_form in DIRECTIONS:
            name = f"{source_form}_to_{target_form}"
            key = f"residual::{layer}::{site}"

            def vectors(split, frame, form, site_key=key):
                return {
                    entity: captured_states[role[(split, frame, form, entity)]][
                        site_key
                    ]
                    for entity in splits[split]
                }

            fit_source = vectors("fit", "A", source_form)
            fit_target = vectors("fit", "A", target_form)
            validation_source = vectors("validation", "A", source_form)
            validation_target = vectors("validation", "B", target_form)
            if identity_correction:
                fit_target = {
                    entity: fit_target[entity].double() - fit_source[entity].double()
                    for entity in splits["fit"]
                }
                scored_target = {
                    entity: validation_target[entity].double()
                    - validation_source[entity].double()
                    for entity in splits["validation"]
                }
            else:
                scored_target = validation_target
            fitted = alignment.fit_alignment(
                fit_source,
                fit_target,
                training_entity_ids=splits["fit"],
                evaluation_entity_ids=splits["validation"] + splits["test"],
                rank=rank,
                ridge_fraction=fraction,
                method=method,
                target_permutation=permutation,
            )
            if identity_correction:
                fitted["diagnostic_identity_correction"] = True
                fitted["target_representation"] = (
                    "target_minus_source; add predicted correction to the original source difference"
                )
                fitted["alignment_sha256"] = alignment.alignment_digest(fitted)
            fits[name] = fitted
            scores[name] = alignment.score_validation(
                fitted,
                validation_source,
                scored_target,
                validation_entity_ids=splits["validation"],
                test_entity_ids=splits["test"],
            )
            if identity_correction:
                # The residual-error numerator equals the complete corrected-
                # map error. Normalize by FULL target energy, including when
                # the ideal correction itself is exactly zero.
                full_energy = sum(
                    float(
                        (
                            validation_target[donor].double()
                            - validation_target[base].double()
                        )
                        .square()
                        .sum()
                    )
                    for base in splits["validation"]
                    for donor in splits["validation"]
                    if base != donor
                )
                scores[name]["target_difference_energy_sum"] = full_energy
                scores[name]["relative_squared_error"] = (
                    None
                    if full_energy == 0
                    else scores[name]["squared_error_sum"] / full_energy
                )
                scores[name]["normalization_scope"] = (
                    "full target difference energy, not correction energy"
                )
        return fits, _combine_validation(scores)

    def choose(candidate_layers, site, permutation=None, identity_correction=False):
        best = None
        diagnostics = []
        for layer in candidate_layers:
            for rank in design["rank_grid"]:
                for fraction in design["ridge_fraction_grid"]:
                    fits, validation = candidate(
                        layer,
                        site,
                        rank,
                        fraction,
                        permutation,
                        identity_correction=identity_correction,
                    )
                    entry = {
                        "layer_index": layer,
                        "rank": rank,
                        "ridge_fraction": fraction,
                        "validation": validation,
                        "alignment_sha256": {
                            direction: fit["alignment_sha256"]
                            for direction, fit in fits.items()
                        },
                    }
                    diagnostics.append(entry)
                    ratio = validation["relative_squared_error"]
                    if ratio is None:
                        continue
                    tie_key = (ratio, rank, -fraction, layer)
                    if best is None or tie_key < best[0]:
                        best = (tie_key, fits, entry)
        if best is None:
            raise RuntimeError(
                "validation target differences contain no selection signal"
            )
        return best[1], best[2], diagnostics

    selected = {}
    for family, site in (("entity_marker", "marker"), ("query", "query")):
        eligible_layers = protocol.eligible_layer_indices(family, len(layers))
        if not eligible_layers:
            raise ValueError(
                "marker selection requires a later decoder block to affect the query"
            )
        fitted, choice, diagnostics = choose(eligible_layers, site)
        orthogonal, _ = candidate(
            choice["layer_index"], site, choice["rank"], 0, method="orthogonal"
        )
        correction, correction_choice, correction_diagnostics = choose(
            [choice["layer_index"]],
            site,
            identity_correction=True,
        )
        directions = {
            name: {
                "ridge": fit,
                "orthogonal": orthogonal[name],
                "identity_correction": correction[name],
                "shuffles": [],
            }
            for name, fit in fitted.items()
        }
        shuffles = []
        for shuffle in design["shuffles"]:
            null_fits, null_choice, null_candidates = choose(
                [choice["layer_index"]], site, shuffle["target_entity_ids"]
            )
            for name, fit in null_fits.items():
                directions[name]["shuffles"].append(fit)
            shuffles.append(
                {
                    "shuffle_index": shuffle["shuffle_index"],
                    **null_choice,
                    "candidate_diagnostics": null_candidates,
                    "layer_fixed_from_real_selection": True,
                }
            )
        selected[family] = {
            **choice,
            "candidate_diagnostics": diagnostics,
            "directions": directions,
            "shuffle_selection": shuffles,
            "identity_correction_selection": {
                **correction_choice,
                "candidate_diagnostics": correction_diagnostics,
                "fixed_from_primary_selected_physical_layer": True,
                "diagnostic_only_not_confirmatory": True,
            },
            "test_vectors_or_outcomes_used": False,
        }
    return selected


def selection_metadata(selected):
    """JSON-only receipt, paired with a separately sealed tensor-map artifact."""
    result = {"test_vectors_or_outcomes_used": False, "sites": {}}
    for family, value in selected.items():
        metadata = {
            key: deepcopy(item) for key, item in value.items() if key != "directions"
        }
        metadata["directions"] = {}
        for direction, maps in value["directions"].items():
            receipt = {}
            for name, fitted in maps.items():
                fits = fitted if isinstance(fitted, list) else [fitted]
                for fit in fits:
                    if alignment.alignment_digest(fit) != fit["alignment_sha256"]:
                        raise ValueError("selected map factors or provenance changed")
                receipt[name] = (
                    [fit["alignment_sha256"] for fit in fits]
                    if isinstance(fitted, list)
                    else fitted["alignment_sha256"]
                )
            metadata["directions"][direction] = receipt
        result["sites"][family] = metadata
    json.dumps(result, allow_nan=False)
    return result
