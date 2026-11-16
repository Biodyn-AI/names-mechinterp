"""Held-out causal jobs and independent float64 outcome scoring.

Mappings are supplied only after their train/validation selection is sealed.
No choices here use target-delta norms or model outcomes. Exact duplicate
operands can share a raw endpoint, while every logical condition stays recorded.
"""

import hashlib
import json

import torch

from e30_14_alignment import apply_alignment


def tensor_hash(value):
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256(str((str(value.dtype), tuple(value.shape))).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def norm_match(value, reference):
    value, reference = value.double(), reference.double()
    length, requested = float(value.norm()), float(reference.norm())
    degenerate = length == 0 or requested == 0
    return (value * (requested / length) if length else value.clone()), degenerate


def js_divergence(first, second):
    """Natural-log Jensen-Shannon divergence from native full-vocabulary logits."""
    first, second = first.double(), second.double()
    if first.ndim != 1 or first.shape != second.shape:
        raise ValueError("Both native endpoints must have the same vocabulary")
    if not bool(torch.isfinite(first).all() and torch.isfinite(second).all()):
        raise ValueError("Nonfinite native logits")
    if torch.equal(first, second):
        return 0.0
    log_first, log_second = first.log_softmax(-1), second.log_softmax(-1)
    log_middle = torch.logaddexp(log_first, log_second) - 0.6931471805599453
    divergence = 0.5 * (
        (log_first.exp() * (log_first - log_middle)).sum()
        + (log_second.exp() * (log_second - log_middle)).sum()
    )
    # Identical inputs are exactly zero; tiny negative roundoff is not evidence.
    return max(0.0, float(divergence))


def aggregate_recovery(rows, effect_floor=1e-6):
    rows = list(rows)
    denominator = sum(row["denominator"] for row in rows)
    error = sum(row["error"] for row in rows)
    return {
        "cell_count": len(rows),
        "denominator_sum": denominator,
        "error_sum": error,
        "recovery": None if denominator < effect_floor else 1 - error / denominator,
    }


def score_cell(endpoints):
    """Never average local and whole-natural references into one number."""
    references = {"local": endpoints["matched"], "natural": endpoints["natural_donor"]}
    denominators = {
        scope: js_divergence(endpoints["base"], reference)
        for scope, reference in references.items()
    }
    scored = {}
    for condition, endpoint in endpoints.items():
        scored[condition] = {}
        for scope, reference in references.items():
            scored[condition][scope] = {
                "denominator": denominators[scope],
                "error": js_divergence(endpoint, reference),
            }
    return scored


def build_jobs(cells, token_rows, states, selected_maps):
    """Build the complete operand plan before executing any held-out patch.

    selected_maps has one layer and two directional map bundles per site.
    Each bundle contains ridge, orthogonal, and 32 validation-selected shuffles.
    Missing length controls remain explicit missing conditions, not substituted.
    """
    rows = {row["prompt_id"]: row for row in token_rows}
    jobs, logical_cells, deduplicated = [], [], {}

    def add_job(prompt_id, layer, token_site, operation, vector):
        vector = vector.detach().cpu().double().contiguous()
        identity = (prompt_id, layer, token_site, operation, tensor_hash(vector))
        if identity not in deduplicated:
            job_id = f"patch_{len(jobs):06d}"
            deduplicated[identity] = job_id
            jobs.append(
                {
                    "endpoint_id": job_id,
                    "prompt_id": prompt_id,
                    "layer_index": layer,
                    "token_site": token_site,
                    "operation": operation,
                    "operand_sha256": identity[-1],
                    "operand": vector,
                }
            )
        return deduplicated[identity]

    for cell in cells:
        selected = selected_maps[cell["site_family"]]
        layer = selected["layer_index"]
        token_site = "marker" if cell["site_family"] == "entity_marker" else "query"
        site = f"residual::{layer}::{token_site}"
        bundle = selected["directions"][
            cell["source_form"] + "_to_" + cell["target_form"]
        ]
        base_id, donor_id = cell["base_entity_id"], cell["donor_entity_id"]
        base_prompt, donor_prompt = cell["base_prompt_id"], cell["donor_prompt_id"]
        source_base = states[cell["source_base_prompt_id"]][site].double()
        source_donor = states[cell["source_donor_prompt_id"]][site].double()
        source_delta = source_donor - source_base

        def mapped(fit, delta=source_delta, donor=donor_id, base=base_id):
            return apply_alignment(
                fit, delta, base_entity_id=base, donor_entity_id=donor
            )

        aligned = mapped(bundle["ridge"])
        control_doses = {}

        def matched_control(
            condition, vector, reference_delta=aligned, dose_diagnostics=control_doses
        ):
            changed, degenerate = norm_match(vector, reference_delta)
            dose_diagnostics[condition] = {
                "original_norm": float(vector.norm()),
                "actual_norm": float(changed.norm()),
                "aligned_norm": float(reference_delta.norm()),
                "degenerate_control": degenerate,
            }
            return changed

        wrong_prompt = cell["wrong_source_donor_prompt_id"]
        wrong_id = rows[wrong_prompt]["entity_id"]
        wrong = matched_control(
            "wrong_donor",
            mapped(
                bundle["ridge"],
                states[wrong_prompt][site].double() - source_base,
                wrong_id,
            ),
        )
        seed = int(
            hashlib.sha256(
                ("e30_14_noise_v1::" + cell["cell_id"]).encode()
            ).hexdigest()[:15],
            16,
        )
        noise = torch.randn(
            aligned.shape,
            dtype=torch.float64,
            generator=torch.Generator().manual_seed(seed),
        )
        noise = matched_control("random", noise)
        deltas = {
            "unmapped_delta": source_delta,
            "mapped_ridge": aligned,
            "mapped_orthogonal": mapped(bundle["orthogonal"]),
            # Diagnostic only: unlike a direct low-rank map, this retains
            # source components outside the learned training subspace.
            "identity_plus_correction": source_delta
            + mapped(bundle["identity_correction"]),
            "reverse": -aligned,
            "random": noise,
            "wrong_donor": wrong,
            "delta_same_form": states[cell["same_form_source_donor_prompt_id"]][
                site
            ].double()
            - states[cell["same_form_source_base_prompt_id"]][site].double(),
        }
        condition_ids = {
            "base": "clean::" + base_prompt,
            "natural_donor": "clean::" + donor_prompt,
        }
        for condition, value in (
            ("matched", states[donor_prompt][site]),
            ("self", states[base_prompt][site]),
            ("replacement", source_donor),
        ):
            condition_ids[condition] = add_job(
                base_prompt, layer, token_site, "replacement", value
            )
        missing = []
        length_wrong_id = cell.get("length_matched_wrong_entity_id")
        if length_wrong_id is not None:
            length_prompt = next(
                row["prompt_id"]
                for row in token_rows
                if row["entity_id"] == length_wrong_id
                and row["frame"] == "A"
                and row["form"] == cell["source_form"]
                and row["relation_id"] == cell["source_relation_id"]
            )
            deltas["length_matched_wrong_donor"] = matched_control(
                "length_matched_wrong_donor",
                mapped(
                    bundle["ridge"],
                    states[length_prompt][site].double() - source_base,
                    length_wrong_id,
                ),
            )
        else:
            missing.append("length_matched_wrong_donor")
        if cell["primary_question_arm"]:
            deltas["mapped_ridge_dose_0.5"] = aligned * 0.5
            deltas["mapped_ridge_dose_2.0"] = aligned * 2.0
            for index, fit in enumerate(bundle["shuffles"]):
                condition = f"shuffle_{index:02d}"
                deltas[condition] = matched_control(condition, mapped(fit))
        for condition, value in deltas.items():
            condition_ids[condition] = add_job(
                base_prompt, layer, token_site, "delta", value
            )
        # Target-state geometry is diagnostic only: never fed back into operands.
        target_delta = (
            states[donor_prompt][site].double() - states[base_prompt][site].double()
        )
        norm_product = float(aligned.norm() * target_delta.norm())
        geometry = {
            "source_norm": float(source_delta.norm()),
            "mapped_norm": float(aligned.norm()),
            "target_norm": float(target_delta.norm()),
            "squared_error": float((aligned - target_delta).square().sum()),
            "target_energy": float(target_delta.square().sum()),
            "cosine": None
            if norm_product == 0
            else float(aligned @ target_delta) / norm_product,
            "source_train_subspace_energy": float(
                (source_delta @ bundle["ridge"]["source_basis"]).square().sum()
            ),
            "source_total_energy": float(source_delta.square().sum()),
        }
        logical_cells.append(
            {
                **cell,
                "layer_index": layer,
                "conditions": condition_ids,
                "missing_conditions": missing,
                "geometry": geometry,
                "random_seed": seed,
                "control_dose_diagnostics": control_doses,
                "degenerate_wrong_control": control_doses["wrong_donor"][
                    "degenerate_control"
                ],
                "degenerate_random_control": control_doses["random"][
                    "degenerate_control"
                ],
            }
        )
    return jobs, logical_cells


def plan_digest(jobs, cells):
    """Bind the operand plan without copying tensors into JSON."""
    metadata = [
        {key: value for key, value in job.items() if key != "operand"} for job in jobs
    ]
    return hashlib.sha256(
        json.dumps(
            {"jobs": metadata, "cells": cells},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
