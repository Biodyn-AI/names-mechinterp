"""Independent operand invariance checks; synthetic tensors only."""

from copy import deepcopy

import pytest
import torch

from e30_14_alignment import apply_alignment, fit_alignment
from e30_14_evaluate import build_jobs, norm_match
from e30_14_protocol import build_comparison_cells, build_prompts, build_splits


def fixture(source_form="full"):
    splits = build_splits()
    generator = torch.Generator().manual_seed(49018)
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
    rows = build_prompts()
    states = {
        row["prompt_id"]: {
            "residual::0::marker": torch.randn(5, generator=generator).double()
        }
        for row in rows
    }
    cell = next(
        c
        for c in build_comparison_cells()
        if c["primary_question_arm"] and c["source_form"] == source_form
    )
    wrong = next(
        e
        for e in splits["test"]
        if e not in (cell["base_entity_id"], cell["donor_entity_id"])
    )
    cell["length_matched_wrong_entity_id"] = wrong
    direction = cell["source_form"] + "_to_" + cell["target_form"]
    selected = {
        "entity_marker": {
            "layer_index": 0,
            "directions": {
                direction: {
                    "ridge": fit,
                    "orthogonal": fit,
                    "identity_correction": fit,
                    "shuffles": [fit] * 32,
                }
            },
        }
    }
    return cell, rows, states, selected, fit


def operands(jobs, logical):
    by_id = {job["endpoint_id"]: job for job in jobs}
    return {
        condition: by_id[identity]["operand"]
        for condition, identity in logical["conditions"].items()
        if identity in by_id and by_id[identity]["operation"] == "delta"
    }


@pytest.mark.parametrize("source_form", ["full", "abbr"])
def test_no_delta_or_null_operand_depends_on_true_target_norm(source_form):
    cell, rows, states, selected, _ = fixture(source_form)
    jobs, logical = build_jobs([cell], rows, states, selected)
    before = operands(jobs, logical[0])
    changed = deepcopy(states)
    for row in rows:
        if row["frame"] == "C":
            changed[row["prompt_id"]]["residual::0::marker"] *= 1000000
    jobs_after, logical_after = build_jobs([cell], rows, changed, selected)
    after = operands(jobs_after, logical_after[0])
    assert set(before) == set(after)
    assert all(torch.equal(before[name], after[name]) for name in before)
    assert (
        logical[0]["geometry"]["target_norm"]
        != logical_after[0]["geometry"]["target_norm"]
    )


def test_all_normalized_nulls_match_source_only_aligned_dose_and_keep_logical_ids():
    cell, rows, states, selected, _ = fixture()
    jobs, logical = build_jobs([cell], rows, states, selected)
    values = operands(jobs, logical[0])
    length = float(values["mapped_ridge"].norm())
    names = ["random", "wrong_donor", "length_matched_wrong_donor"] + [
        f"shuffle_{i:02d}" for i in range(32)
    ]
    for name in names:
        assert float(values[name].norm()) == pytest.approx(length)
    assert (
        len([name for name in logical[0]["conditions"] if name.startswith("shuffle_")])
        == 32
    )
    assert logical[0]["missing_conditions"] == []


def test_length_control_uses_wrong_entity_in_source_form_and_question():
    cell, rows, states, selected, fit = fixture("abbr")
    jobs, logical = build_jobs([cell], rows, states, selected)
    actual = operands(jobs, logical[0])
    wrong_id = cell["length_matched_wrong_entity_id"]
    wrong_prompt = next(
        r["prompt_id"]
        for r in rows
        if r["entity_id"] == wrong_id
        and r["frame"] == "A"
        and r["form"] == cell["source_form"]
        and r["relation_id"] == cell["source_relation_id"]
    )
    difference = (
        states[wrong_prompt]["residual::0::marker"]
        - states[cell["source_base_prompt_id"]]["residual::0::marker"]
    )
    expected = apply_alignment(
        fit, difference, base_entity_id=cell["base_entity_id"], donor_entity_id=wrong_id
    )
    expected, _ = norm_match(expected, actual["mapped_ridge"])
    assert torch.equal(actual["length_matched_wrong_donor"], expected)


def test_diagnostic_identity_correction_preserves_raw_source_component():
    cell, rows, states, selected, fit = fixture()
    jobs, logical = build_jobs([cell], rows, states, selected)
    actual = operands(jobs, logical[0])
    source = (
        states[cell["source_donor_prompt_id"]]["residual::0::marker"]
        - states[cell["source_base_prompt_id"]]["residual::0::marker"]
    )
    correction = apply_alignment(
        fit,
        source,
        base_entity_id=cell["base_entity_id"],
        donor_entity_id=cell["donor_entity_id"],
    )
    assert torch.equal(actual["identity_plus_correction"], source + correction)
