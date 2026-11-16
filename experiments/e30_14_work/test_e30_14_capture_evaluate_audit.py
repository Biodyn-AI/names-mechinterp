"""Independent whole-grid interface and held-out-target dose-leakage audit."""

import json
from copy import deepcopy
from pathlib import Path

import torch

import e30_14_capture as capture
import e30_14_evaluate as evaluate
import e30_14_protocol as protocol
from test_e30_14_capture import CharacterTokenizer, synthetic_training_capture


def prepared_problem():
    training_states, training_rows, design = synthetic_training_capture()
    design["rank_grid"], design["ridge_fraction_grid"] = [2], [0.1]
    design["shuffles"] = protocol.build_shuffles()
    design["shuffle_count"] = 32
    selected = capture.select_maps(
        training_states, training_rows, protocol_record=design
    )
    facts = json.loads(Path(__file__).with_name("us_state_facts_2020.json").read_text())
    rows = capture.prepare_token_rows(
        CharacterTokenizer(),
        protocol.build_prompts(),
        chat_model=True,
        facts=facts["facts"],
    )
    cells = capture.qualify_comparison_cells(rows, facts)
    test_rows = [row for row in rows if row["split"] == "test"]
    generator = torch.Generator().manual_seed(1401)
    values = {
        entity: torch.randn(4, dtype=torch.float64, generator=generator)
        for entity in protocol.build_splits()["test"]
    }
    states = {}
    for row in test_rows:
        value = values[row["entity_id"]].clone()
        if row["form"] == "abbr":
            value = value.flip(0)
        if row["relation_id"] == "largest_city":
            value = value + torch.tensor([0.0, 0.1, 0.2, 0.3])
        states[row["prompt_id"]] = {
            f"residual::{layer}::{site}": value.float().clone()
            for layer in range(2)
            for site in ("marker", "query")
        }
    return cells, test_rows, states, selected


def test_all_640_cells_accept_actual_selected_map_records_and_score():
    cells, rows, states, selected = prepared_problem()
    jobs, logical_cells = evaluate.build_jobs(cells, rows, states, selected)
    assert len(logical_cells) == 640
    assert len({job["endpoint_id"] for job in jobs}) == len(jobs)
    assert sum(cell["primary_question_arm"] for cell in logical_cells) == 160
    assert len(evaluate.plan_digest(jobs, logical_cells)) == 64
    for cell in logical_cells:
        assert len(
            [name for name in cell["conditions"] if name.startswith("shuffle_")]
        ) == (32 if cell["primary_question_arm"] else 0)
        assert ("mapped_ridge_dose_0.5" in cell["conditions"]) == cell[
            "primary_question_arm"
        ]
        assert ("mapped_ridge_dose_2.0" in cell["conditions"]) == cell[
            "primary_question_arm"
        ]
    generator = torch.Generator().manual_seed(1402)
    head = torch.randn(4, 16, dtype=torch.float64, generator=generator)
    clean = {
        "clean::" + row["prompt_id"]: (
            states[row["prompt_id"]]["residual::0::query"].double() @ head
        ).float()
        for row in rows
    }
    endpoints = dict(clean)
    for job in jobs:
        assert job["operand"].shape == (4,) and job["operand"].dtype == torch.float64
        assert job["operand_sha256"] == evaluate.tensor_hash(job["operand"])
        current = states[job["prompt_id"]][
            f"residual::{job['layer_index']}::{job['token_site']}"
        ].double()
        changed = (
            job["operand"]
            if job["operation"] == "replacement"
            else current + job["operand"]
        )
        endpoints[job["endpoint_id"]] = (changed @ head).float()
    for cell in logical_cells:
        score = evaluate.score_cell(
            {
                condition: endpoints[identity]
                for condition, identity in cell["conditions"].items()
            }
        )
        assert score["self"]["local"] == score["base"]["local"]
        assert score["matched"]["local"]["error"] == 0


def test_changing_heldout_target_states_cannot_change_any_delta_operand():
    cells, rows, states, selected = prepared_problem()
    first_jobs, first_cells = evaluate.build_jobs(cells, rows, states, selected)
    changed = deepcopy(states)
    for row in rows:
        if row["frame"] == "C":
            changed[row["prompt_id"]] = {
                key: value * 17 + 100
                for key, value in changed[row["prompt_id"]].items()
            }
    second_jobs, second_cells = evaluate.build_jobs(cells, rows, changed, selected)
    first_lookup = {job["endpoint_id"]: job for job in first_jobs}
    second_lookup = {job["endpoint_id"]: job for job in second_jobs}
    for first, second in zip(first_cells, second_cells):
        assert first["cell_id"] == second["cell_id"]
        for condition, identity in first["conditions"].items():
            if (
                identity.startswith("clean::")
                or first_lookup[identity]["operation"] != "delta"
            ):
                continue
            next_identity = second["conditions"][condition]
            torch.testing.assert_close(
                first_lookup[identity]["operand"],
                second_lookup[next_identity]["operand"],
                atol=0,
                rtol=0,
            )
