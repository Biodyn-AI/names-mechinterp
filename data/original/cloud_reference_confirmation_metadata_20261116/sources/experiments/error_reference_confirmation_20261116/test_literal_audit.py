"""Independent printed-prompt checks fixed before new model answers exist."""

import copy
import importlib.util
from pathlib import Path

import pytest

import literal_audit as audit

REPO = Path(__file__).resolve().parents[2]


def fixture(error_worlds=0):
    source = REPO / "experiments/error_reference_study_20261114/error_protocol.py"
    spec = importlib.util.spec_from_file_location("literal_test_generator", source)
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    study = generator.build_study()
    rows = [row for row in study["rows"] if row["split"] == "confirmation"]
    worlds = {
        world["world_id"]: world
        for world in study["worlds"]
        if world["split"] == "confirmation"
    }
    errors = set(list(worlds)[:error_worlds])
    predictions = []
    for row in rows:
        answer = row["expected_answer"]
        if (
            row["world_id"] in errors
            and row["probe_kind"] == "fact"
            and row["query_alias_index"] == 0
            and row["field"] == "home"
        ):
            world = worlds[row["world_id"]]
            answer = world["objects"][world["alias_targets"][2]]["home"]
        predictions.append(
            {
                "id": row["id"],
                "expected_answer": row["expected_answer"],
                "generated_text": answer,
                "generated_ids": [2, 9],
                "step_endpoint_ids": [f"{row['id']}::step00", f"{row['id']}::step01"],
                "terminated": True,
                "truncated": False,
                "stop_reason": "eos",
            }
        )
    return rows, predictions


def repair_generations(cases, rows, predictions):
    lookup = {row["id"]: row for row in rows}
    old = {prediction["id"]: prediction for prediction in predictions}
    records = []
    for case in cases:
        for condition in audit.CONDITIONS:
            for field in audit.FIELDS:
                row = lookup[case["recipient_row_ids"][field]]
                identity = f"repair::{case['case_id']}::{condition}::{field}"
                record = copy.deepcopy(old[row["id"]])
                record.update(
                    id=identity,
                    case_id=case["case_id"],
                    world_id=case["world_id"],
                    row_id=row["id"],
                    condition=condition,
                    field=field,
                    step_endpoint_ids=[f"{identity}::step00", f"{identity}::step01"],
                )
                records.append(record)
    return records


def test_full_confirmation_is_graded_from_printed_text():
    rows, predictions = fixture()
    report = audit.reconstruct_behavior(rows, predictions, REPO)
    assert report["worlds"] == 64
    assert report["overall"] == {"correct": 6144, "count": 6144}
    assert report["by_probe_kind"]["fact"] == {"correct": 4608, "count": 4608}
    assert report["coreference"] == {
        "pair_count": 1536,
        "discordant_wrong_object": 0,
        "discordant_one_correct": 0,
        "different_answers": 0,
        "both_correct": 1536,
        "both_wrong": 0,
    }
    assert report["wrong_object_coreference_worlds"] == 0


@pytest.mark.parametrize(
    "change", ["row_truth", "prediction_truth", "field", "literal_fact"]
)
def test_tampered_truth_or_literal_prompt_is_not_silently_accepted(change):
    rows, predictions = fixture()
    if change == "row_truth":
        rows[0]["expected_answer"] = "tampered"
    elif change == "prediction_truth":
        predictions[0]["expected_answer"] = "tampered"
    elif change == "field":
        rows[0]["field"] = "home"
    else:
        rows[0]["text"] = rows[0]["text"].replace("Object A", "Object B", 1)
    with pytest.raises(ValueError):
        audit.reconstruct_behavior(rows, predictions, REPO)


@pytest.mark.parametrize("change", ["missing", "duplicate", "foreign", "row_duplicate"])
def test_behavior_requires_complete_unique_historical_confirmation_coverage(change):
    rows, predictions = fixture()
    if change == "missing":
        predictions.pop()
    elif change == "duplicate":
        predictions[-1] = predictions[0]
    elif change == "foreign":
        predictions[0]["id"] = "development:000"
    else:
        rows[-1] = rows[0]
    with pytest.raises(ValueError, match="coverage|generator"):
        audit.reconstruct_behavior(rows, predictions, REPO)


def test_finished_wrong_object_is_not_confused_with_formatting_or_a_prefix():
    rows, predictions = fixture(1)
    report = audit.reconstruct_behavior(rows, predictions, REPO)
    assert report["fact_errors"] == {"wrong_object": 8, "unlisted_or_malformed": 0}
    assert report["coreference"]["discordant_wrong_object"] == 8
    assert (
        report["wrong_object_worlds"] == report["wrong_object_coreference_worlds"] == 1
    )
    for row, prediction in zip(rows, predictions):
        if (
            row["probe_kind"] == "fact"
            and row["field"] == "home"
            and row["query_alias_index"] == 0
        ):
            prediction["generated_text"] = (
                "The answer is " + prediction["generated_text"]
            )
    changed = audit.reconstruct_behavior(rows, predictions, REPO)
    assert changed["fact_errors"]["wrong_object"] == 0
    assert changed["coreference"]["discordant_wrong_object"] == 0


def test_unfinished_correct_text_is_an_error_and_bad_completion_metadata_fails():
    rows, predictions = fixture()
    predictions[0].update(
        terminated=False,
        truncated=True,
        stop_reason="max_tokens",
        generated_ids=[2] * 8,
        step_endpoint_ids=[f"{rows[0]['id']}::step{i:02d}" for i in range(8)],
    )
    report = audit.reconstruct_behavior(rows, predictions, REPO)
    assert report["by_probe_kind"]["fact"]["correct"] == 4607
    assert report["fact_errors"]["unlisted_or_malformed"] == 1
    predictions[0]["truncated"] = False
    with pytest.raises(ValueError, match="completion"):
        audit.reconstruct_behavior(rows, predictions, REPO)


def test_case_selection_includes_one_context_in_every_eligible_world():
    rows, predictions = fixture(3)
    cases = audit.independently_select_cases(rows, predictions, REPO)
    assert len(cases) == len({case["world_id"] for case in cases}) == 3
    assert all(case["initial_wrong_fields"] == ["home"] for case in cases)
    assert all(case["protected_correct_fields"] == ["color", "shape"] for case in cases)
    assert (
        audit.independently_select_cases(rows[::-1], predictions[::-1], REPO) == cases
    )
    for row, prediction in zip(rows, predictions):
        if row["probe_kind"] == "identity" and row["query_alias_index"] == 1:
            prediction["generated_text"] = "Object " + prediction["generated_text"]
    assert audit.independently_select_cases(rows, predictions, REPO) == []


def test_repair_grid_counts_entire_answers_and_protected_damage_separately():
    rows, predictions = fixture(2)
    cases = audit.independently_select_cases(rows, predictions, REPO)
    outputs = repair_generations(cases, rows, predictions)
    for prediction in outputs:
        if prediction["condition"] == "primaryDcolor":
            prediction["generated_text"] = (
                prediction["expected_answer"]
                if prediction["field"] == "home"
                else "extra explanation"
            )
    report = audit.reconstruct_repair(cases, rows, predictions, outputs, REPO)
    assert report["baseline"] == {
        "recipient_wrong": 2,
        "recipient_protected": 4,
        "recipient_fields": 6,
        "donor_correct": 6,
    }
    assert report["by_condition"]["primaryDcolor"]["repaired_initial_errors"] == 2
    assert report["by_condition"]["primaryDcolor"]["protected_damaged"] == 4
    assert report["by_condition"]["selfRfield"]["repaired_initial_errors"] == 0


@pytest.mark.parametrize(
    "change", ["missing", "duplicate", "condition", "case", "row", "label", "self"]
)
def test_repair_rejects_case_reselection_incomplete_grid_and_metadata_tampering(change):
    rows, predictions = fixture(2)
    cases = audit.independently_select_cases(rows, predictions, REPO)
    outputs = repair_generations(cases, rows, predictions)
    if change == "missing":
        outputs.pop()
    elif change == "duplicate":
        outputs[-1] = outputs[0]
    elif change == "condition":
        outputs[-1]["condition"] = "new_favorable_repair"
    elif change == "case":
        cases[0]["protected_correct_fields"] = []
    elif change == "row":
        outputs[0]["row_id"] = cases[1]["recipient_row_ids"]["home"]
    elif change == "label":
        outputs[0]["expected_answer"] = "changed"
    else:
        outputs[0]["generated_text"] = "changed self answer"
    with pytest.raises(ValueError):
        audit.reconstruct_repair(cases, rows, predictions, outputs, REPO)


def test_no_eligible_cases_is_a_complete_empty_grid_not_a_positive_result():
    rows, predictions = fixture()
    report = audit.reconstruct_repair([], rows, predictions, [], REPO)
    assert report["eligible_worlds"] == report["generations"] == 0
    assert report["baseline"]["recipient_wrong"] == 0


def reported_behavior(independent):
    return {
        "independent_worlds": 64,
        "prompts": 6144,
        **{
            key: copy.deepcopy(independent[key])
            for key in (
                "by_probe_kind",
                "fact_errors",
                "coreference",
                "wrong_object_worlds",
                "wrong_object_coreference_worlds",
                "world_summaries",
            )
        },
    }


def reported_repair(independent):
    baseline = independent["baseline"]
    return {
        "eligible_worlds": independent["eligible_worlds"],
        "by_condition": {
            condition: {
                "complete_answers": counts["count"],
                "correct": counts["correct"],
                "repaired_initial_errors": counts["repaired_initial_errors"],
                "protected_damaged": counts["protected_damaged"],
                "initial_errors": baseline["recipient_wrong"],
                "protected_facts": baseline["recipient_protected"],
                "by_world": copy.deepcopy(counts["by_world"]),
            }
            for condition, counts in independent["by_condition"].items()
        },
    }


def test_reported_aggregate_and_per_world_counts_must_match_independent_answers():
    rows, predictions = fixture(2)
    behavior = audit.reconstruct_behavior(rows, predictions, REPO)
    reported = reported_behavior(behavior)
    assert audit.compare_behavior(behavior, reported)
    reported["world_summaries"][next(iter(reported["world_summaries"]))][
        "wrong_object"
    ] += 1
    with pytest.raises(ValueError, match="behavioral counts"):
        audit.compare_behavior(behavior, reported)
    cases = audit.independently_select_cases(rows, predictions, REPO)
    repair = audit.reconstruct_repair(
        cases, rows, predictions, repair_generations(cases, rows, predictions), REPO
    )
    reported = reported_repair(repair)
    assert audit.compare_repair(repair, reported)
    reported["by_condition"]["primaryDcolor"]["protected_damaged"] += 1
    with pytest.raises(ValueError, match="repair counts"):
        audit.compare_repair(repair, reported)


def seal(path, value):
    import hashlib
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    content = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    path.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    Path(str(path) + ".sha256").write_text(digest + "\n")
    return digest


def stage_fixture(root, name, registration, *, native_custody=None):
    stage = root / name
    artifact_names = ["GENERATIONS.json", "RESULT.json"]
    if name == "repair":
        artifact_names += ["CASES.json", "EXECUTION.json"]
    digests = {
        artifact: seal(
            stage / artifact,
            [] if artifact in ("GENERATIONS.json", "CASES.json") else {},
        )
        for artifact in artifact_names
    }
    custody = seal(
        stage / "CUSTODY.json",
        {"registration_sha256": registration, "artifact_sha256": digests},
    )
    native = {
        "exact_match": True,
        "generations": 0,
        "custody_sha256": native_custody or custody,
        **(
            {"cases": 0}
            if name == "repair"
            else {"required_rows": 6144, "required_worlds": 64}
        ),
    }
    seal(stage / "NATIVE_AUDIT.json", native)
    return stage


def test_stage_receipt_authentication_rejects_changed_custody_and_native_binding(
    tmp_path,
):
    stage = stage_fixture(tmp_path, "repair", "registration")
    assert (
        audit.verified_stage(tmp_path, "repair", "registration")[0]["CASES.json"] == []
    )
    with pytest.raises(ValueError, match="registration"):
        audit.verified_stage(tmp_path, "repair", "another registration")
    seal(stage / "GENERATIONS.json", [{"changed": True}])
    with pytest.raises(ValueError, match="Custody hash"):
        audit.verified_stage(tmp_path, "repair", "registration")
    stage_fixture(tmp_path, "qwen", "registration", native_custody="another custody")
    with pytest.raises(ValueError, match="native audit"):
        audit.verified_stage(tmp_path, "qwen", "registration")


def test_output_and_checksum_are_write_once_even_before_any_input_is_read(tmp_path):
    target = tmp_path / "receipt.json"
    target.write_text("old receipt")
    with pytest.raises(FileExistsError, match="single-use"):
        audit.run_audit(tmp_path / "missing", REPO, target)
    target.unlink()
    Path(str(target) + ".sha256").write_text("old checksum")
    with pytest.raises(FileExistsError, match="single-use"):
        audit.run_audit(tmp_path / "missing", REPO, target)


def test_complete_receipt_plumbing_with_synthetic_answers_and_native_receipts(tmp_path):
    """Synthetic native receipts test file bindings, not real tensor proof."""
    import hashlib
    import shutil

    rows, predictions = fixture(2)
    behavior = audit.reconstruct_behavior(rows, predictions, REPO)
    cases = audit.independently_select_cases(rows, predictions, REPO)
    outputs = repair_generations(cases, rows, predictions)
    repair = audit.reconstruct_repair(cases, rows, predictions, outputs, REPO)
    relative = "experiments/error_reference_confirmation_20261116/literal_audit.py"
    frozen_source = tmp_path / "sources" / relative
    frozen_source.parent.mkdir(parents=True)
    shutil.copyfile(REPO / relative, frozen_source)
    inputs = {
        "PLAN.json": {},
        "ROWS.json": rows,
        "WORLDS.json": [],
        "TOKENS-qwen.json": {},
        "TOKENS-phi.json": {},
        "CODE.json": {
            relative: hashlib.sha256((REPO / relative).read_bytes()).hexdigest()
        },
    }
    registration_hash = seal(
        tmp_path / "REGISTRATION.json",
        {
            "protocol_commit": "f" * 40,
            "input_sha256": {
                name: seal(tmp_path / name, value) for name, value in inputs.items()
            },
        },
    )

    def save_stage(name, records):
        root = tmp_path / name
        custody_hash = seal(
            root / "CUSTODY.json",
            {
                "registration_sha256": registration_hash,
                "artifact_sha256": {
                    artifact: seal(root / artifact, value)
                    for artifact, value in records.items()
                },
            },
        )
        return seal(
            root / "NATIVE_AUDIT.json",
            {
                "exact_match": True,
                "custody_sha256": custody_hash,
                "generations": len(records["GENERATIONS.json"]),
                **(
                    {"cases": len(cases)}
                    if name == "repair"
                    else {"required_rows": 6144, "required_worlds": 64}
                ),
            },
        )

    baseline_audit = save_stage(
        "qwen",
        {"GENERATIONS.json": predictions, "RESULT.json": reported_behavior(behavior)},
    )
    save_stage(
        "phi",
        {"GENERATIONS.json": predictions, "RESULT.json": reported_behavior(behavior)},
    )
    save_stage(
        "repair",
        {
            "GENERATIONS.json": outputs,
            "RESULT.json": reported_repair(repair),
            "CASES.json": cases,
            "EXECUTION.json": {"baseline_audit_sha256": baseline_audit},
        },
    )
    output = tmp_path / "literal_receipt.json"
    receipt = audit.run_audit(tmp_path, REPO, output)
    assert receipt["all_reported_counts_match"] is True
    assert receipt["repair"]["eligible_worlds"] == 2
    assert receipt["native_proof_claimed"] is False
    assert audit.verified_json(output)[0] == receipt
    with pytest.raises(FileExistsError, match="single-use"):
        audit.run_audit(tmp_path, REPO, output)
