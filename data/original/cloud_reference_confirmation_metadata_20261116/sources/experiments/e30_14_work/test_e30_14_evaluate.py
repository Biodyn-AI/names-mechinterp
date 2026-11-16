"""Test causal planning/scoring without trained models or cloud access."""

import pytest
import torch

from e30_14_alignment import fit_alignment
from e30_14_evaluate import (
    aggregate_recovery,
    build_jobs,
    js_divergence,
    norm_match,
    score_cell,
)
from e30_14_protocol import build_comparison_cells, build_prompts, build_splits


def test_js_is_symmetric_and_zero_for_identical_logits():
    first = torch.tensor([1.0, -1.0, 0.0])
    second = torch.tensor([-2.0, 1.0, 0.0])
    assert js_divergence(first, first) == 0.0
    assert js_divergence(first, second) == pytest.approx(js_divergence(second, first))
    assert 0 < js_divergence(first, second) < 0.693148


def test_aggregate_uses_ratio_of_sums_and_retains_negative_recovery():
    scored = [{"denominator": 0.3, "error": 0.1}, {"denominator": 0.1, "error": 0.7}]
    assert aggregate_recovery(scored)["recovery"] == pytest.approx(-1.0)
    assert aggregate_recovery([{"denominator": 0, "error": 0}])["recovery"] is None


def test_score_keeps_local_and_natural_references_separate():
    base = torch.tensor([5.0, 0.0, 0.0])
    matched = torch.tensor([0.0, 5.0, 0.0])
    natural = torch.tensor([0.0, 0.0, 5.0])
    scored = score_cell(
        {
            "base": base,
            "matched": matched,
            "natural_donor": natural,
            "mapped_ridge": matched,
        }
    )
    assert scored["mapped_ridge"]["local"]["error"] == 0
    assert scored["mapped_ridge"]["natural"]["error"] > 0


def test_norm_matching_uses_only_the_given_reference_and_records_degeneracy():
    matched, degenerate = norm_match(torch.tensor([3.0, 4.0]), torch.tensor([0.0, 2.0]))
    assert float(matched.norm()) == pytest.approx(2)
    assert not degenerate
    zero, degenerate = norm_match(torch.zeros(2), torch.tensor([0.0, 2.0]))
    assert degenerate and torch.equal(zero, torch.zeros(2))


def test_operand_plan_retains_logical_nulls_without_target_norm_oracle():
    splits = build_splits()
    generator = torch.Generator().manual_seed(813)
    training = {
        entity: torch.randn(5, generator=generator).double() for entity in splits["fit"]
    }
    fit = fit_alignment(
        training,
        training,
        training_entity_ids=splits["fit"],
        evaluation_entity_ids=splits["validation"] + splits["test"],
        rank=2,
        ridge_fraction=0.1,
    )
    orthogonal = fit_alignment(
        training,
        training,
        training_entity_ids=splits["fit"],
        evaluation_entity_ids=splits["validation"] + splits["test"],
        rank=2,
        ridge_fraction=0.1,
        method="orthogonal",
    )
    rows = build_prompts()
    states = {
        row["prompt_id"]: {
            "residual::0::marker": torch.randn(5, generator=generator).double()
        }
        for row in rows
    }
    cell = next(
        cell for cell in build_comparison_cells() if cell["primary_question_arm"]
    )
    selected = {
        "entity_marker": {
            "layer_index": 0,
            "directions": {
                "full_to_abbr": {
                    "ridge": fit,
                    "orthogonal": orthogonal,
                    "identity_correction": fit,
                    "shuffles": [fit] * 32,
                }
            },
        }
    }
    jobs, cells = build_jobs([cell], rows, states, selected)
    assert (
        len([key for key in cells[0]["conditions"] if key.startswith("shuffle_")]) == 32
    )
    assert len(cells[0]["control_dose_diagnostics"]) == 34
    for dose in cells[0]["control_dose_diagnostics"].values():
        assert not dose["degenerate_control"]
        assert dose["actual_norm"] == pytest.approx(dose["aligned_norm"])
    assert cells[0]["missing_conditions"] == ["length_matched_wrong_donor"]
    by_id = {job["endpoint_id"]: job for job in jobs}
    mapped = by_id[cells[0]["conditions"]["mapped_ridge"]]["operand"].clone()
    # A target delta is used for descriptive geometry, never for causal dose.
    states[cell["donor_prompt_id"]]["residual::0::marker"] *= 100
    later_jobs, later_cells = build_jobs([cell], rows, states, selected)
    later_by_id = {job["endpoint_id"]: job for job in later_jobs}
    assert torch.equal(
        mapped, later_by_id[later_cells[0]["conditions"]["mapped_ridge"]]["operand"]
    )
