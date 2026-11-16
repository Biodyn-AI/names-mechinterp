"""Synthetic complete-grid tests; these are not new model observations."""

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "reproduce_confirmation", ROOT / "tools/reproduce_confirmation.py"
)
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture(scope="module")
def synthetic_data(tmp_path_factory):
    """One eligible world, with an explicitly invented successful primary repair."""
    target = tmp_path_factory.mktemp("confirmation-synthetic")
    plan, _ = helper.sources()
    study = plan._study()
    rows = [r for r in study["rows"] if r["split"] == "confirmation"]
    worlds = [w for w in study["worlds"] if w["split"] == "confirmation"]
    first = worlds[0]
    qwen, phi = [], []
    for row in rows:
        prediction = {
            "id": row["id"],
            "row_id": row["id"],
            "expected_answer": row["expected_answer"],
            "generated_text": row["expected_answer"],
            "generated_ids": [2, 9],
            "step_endpoint_ids": [f"{row['id']}::step00", f"{row['id']}::step01"],
            "terminated": True,
            "truncated": False,
            "stop_reason": "eos",
        }
        phi.append(copy.deepcopy(prediction))
        if (
            row["world_id"] == first["world_id"]
            and row["probe_kind"] == "fact"
            and row["query_alias_index"] == 0
            and row["field"] == "home"
        ):
            prediction["generated_text"] = first["objects"][first["alias_targets"][2]][
                "home"
            ]
        qwen.append(prediction)
    cases = plan.select_cases(rows, qwen, worlds)
    assert len(cases) == 1
    by_row, old = {r["id"]: r for r in rows}, {p["id"]: p for p in qwen}
    generations = []
    for case in cases:
        for condition in plan.CONDITIONS:
            for field in plan.FIELDS:
                row_id = case["recipient_row_ids"][field]
                identity = f"repair::{case['case_id']}::{condition}::{field}"
                prediction = copy.deepcopy(old[row_id])
                prediction.update(
                    id=identity,
                    case_id=case["case_id"],
                    world_id=case["world_id"],
                    row_id=row_id,
                    condition=condition,
                    field=field,
                    step_endpoint_ids=[identity + "::step00", identity + "::step01"],
                )
                if condition == "primaryDcolor":
                    prediction["generated_text"] = by_row[row_id]["expected_answer"]
                generations.append(prediction)
    for name, value in (
        ("ROWS.json", rows),
        ("WORLDS.json", worlds),
        ("qwen/GENERATIONS.json", qwen),
        ("phi/GENERATIONS.json", phi),
        ("repair/CASES.json", cases),
        ("repair/GENERATIONS.json", generations),
    ):
        write(target / name, value)
    reports = {
        "qwen": plan.score_behavior(rows, qwen, worlds),
        "phi": plan.score_behavior(rows, phi, worlds),
        "repair": plan.assess_repair(cases, rows, qwen, generations),
    }
    for stage, report in reports.items():
        write(target / stage / "RESULT.json", report)
    return target


@pytest.fixture
def mutable_data(synthetic_data, tmp_path):
    import shutil

    shutil.copytree(synthetic_data, tmp_path / "study")
    return tmp_path / "study"


def test_complete_grid_recomputed_offline_with_independent_printed_labels(
    synthetic_data,
):
    before_imports = set(sys.modules)
    report = helper.reproduce(synthetic_data)
    assert report["independent_worlds"] == 64
    assert report["questions_per_model"] == 6144
    assert report["behavior"]["qwen"]["fact_errors"]["wrong_object"] == 8
    assert report["behavior"]["phi"]["by_probe_kind"]["fact"]["correct"] == 4608
    assert report["repair"]["eligible_worlds"] == 1
    assert (
        report["repair"]["by_condition"]["primaryDcolor"]["repaired_initial_errors"]
        == 1
    )
    assert report["native_logit_replay_performed"] is False
    assert report["new_model_forward_count"] == 0
    assert report["saved_results_verified"] == ["qwen", "phi", "repair"]
    assert not {"torch", "transformers"} & (set(sys.modules) - before_imports)


@pytest.mark.parametrize("stage", ["qwen", "repair"])
def test_saved_integer_counts_cannot_disagree_with_recomputed_answers(
    mutable_data, stage
):
    path = mutable_data / stage / "RESULT.json"
    report = json.loads(path.read_text())
    if stage == "qwen":
        report["by_probe_kind"]["fact"]["correct"] += 1
    else:
        report["by_condition"]["primaryDcolor"]["repaired_initial_errors"] += 1
    write(path, report)
    with pytest.raises(ValueError, match="Saved result differs"):
        helper.reproduce(mutable_data)


@pytest.mark.parametrize("name", ["qwen/GENERATIONS.json", "repair/GENERATIONS.json"])
def test_missing_question_or_repair_grid_is_not_a_completed_result(mutable_data, name):
    path = mutable_data / name
    records = json.loads(path.read_text())
    write(path, records[:-1])
    with pytest.raises(ValueError):
        helper.reproduce(mutable_data)


def test_independent_parser_detects_a_main_analysis_count_error(
    synthetic_data, monkeypatch
):
    plan, literal = helper.sources()

    def mistaken(rows, predictions, worlds):
        report = plan.score_behavior(rows, predictions, worlds)
        report["fact_errors"]["wrong_object"] += 1
        return report

    proxy = SimpleNamespace(
        score_behavior=mistaken,
        select_cases=plan.select_cases,
        assess_repair=plan.assess_repair,
    )
    monkeypatch.setattr(helper, "sources", lambda: (proxy, literal))
    with pytest.raises(ValueError, match="literal|counts differ"):
        helper.reproduce(synthetic_data)


def test_saved_statistic_rounding_tolerance_never_applies_to_integer_counts():
    helper.compare_saved({"mean": 0.5}, {"mean": 0.5 + 5e-13})
    with pytest.raises(ValueError, match="Saved result differs"):
        helper.compare_saved({"mean": 0.5}, {"mean": 0.5 + 2e-12})
    with pytest.raises(ValueError, match="Saved result differs"):
        helper.compare_saved({"count": 2}, {"count": 2.0})


def test_output_is_exclusive_and_existing_file_survives(synthetic_data, tmp_path):
    output = tmp_path / "recomputed.json"
    helper.main(["--data-root", str(synthetic_data), "--output", str(output)])
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        helper.main(["--data-root", str(synthetic_data), "--output", str(output)])
    assert output.read_bytes() == original


def test_saved_statistics_are_optional_and_never_claimed_verified_when_absent(
    mutable_data,
):
    for stage in ("qwen", "phi", "repair"):
        (mutable_data / stage / "RESULT.json").unlink()
    report = helper.reproduce(mutable_data)
    assert report["saved_results_verified"] == []
    assert report["repair"]["eligible_worlds"] == 1


def test_frozen_analysis_hash_is_checked_before_import(monkeypatch):
    monkeypatch.setattr(helper, "SOURCE_PINS", {"confirmation_plan": "0" * 64})
    with pytest.raises(ValueError, match="Frozen confirmation analysis changed"):
        helper.sources()
