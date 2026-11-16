"""Model-free tests fixed before the new confirmation answers are observed."""

import copy
import importlib.util
import json
import math
from pathlib import Path

import pytest

import confirmation_plan as plan

REPO = Path(__file__).resolve().parents[2]


def test_public_protocol_snapshot_matches_the_executable_choices():
    public_protocol = Path(__file__).with_name("PLAN.json")
    assert json.loads(public_protocol.read_text()) == plan.build_plan()


def fixture():
    source = REPO / "experiments/error_reference_study_20261114/error_protocol.py"
    spec = importlib.util.spec_from_file_location("confirmation_test_generator", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    study = module.build_study()
    rows = [row for row in study["rows"] if row["split"] == "confirmation"]
    worlds = [world for world in study["worlds"] if world["split"] == "confirmation"]
    predictions = [
        {
            "id": row["id"],
            "generated_text": row["expected_answer"],
            "terminated": True,
            "truncated": False,
        }
        for row in rows
    ]
    return rows, predictions, worlds


def errors_in_worlds(count):
    rows, predictions, worlds = fixture()
    selected = {world["world_id"] for world in worlds[:count]}
    lookup = {world["world_id"]: world for world in worlds}
    for row, prediction in zip(rows, predictions):
        if (
            row["world_id"] in selected
            and row["probe_kind"] == "fact"
            and row["query_alias_index"] == 0
            and row["field"] == "home"
        ):
            world = lookup[row["world_id"]]
            other = world["alias_targets"][2]
            prediction["generated_text"] = world["objects"][other]["home"]
    return rows, predictions, worlds


def generations(cases, rows, predictions, *, repair=False, damage=False):
    row_lookup = {row["id"]: row for row in rows}
    predicted = {prediction["id"]: prediction for prediction in predictions}
    records = []
    for case in cases:
        for condition in plan.build_plan()["conditions"]:
            for field in plan.FIELDS:
                row = row_lookup[case["recipient_row_ids"][field]]
                record = dict(predicted[row["id"]])
                record.update(
                    case_id=case["case_id"],
                    world_id=case["world_id"],
                    row_id=row["id"],
                    condition=condition,
                    field=field,
                )
                if condition == "primaryDcolor" and repair:
                    record["generated_text"] = row["expected_answer"]
                if (
                    condition == "primaryDcolor"
                    and damage
                    and field in case["protected_correct_fields"]
                ):
                    record["generated_text"] = "not a listed value"
                records.append(record)
    return records


def test_plan_is_narrow_and_keeps_old_failure_closed():
    specification = plan.build_plan()
    assert specification["selected_head"] == {"layer_index": 28, "head_index": 0}
    assert specification["conditions"] == [
        "selfRfield",
        "primaryDcolor",
        "frozenDiscoveryMean",
        "Rcolor",
    ]
    assert specification["numeric"]["max_query_head_drift"] == 2e-5
    assert specification["old_positive_mechanism_gate_passed"] is False
    assert specification["minimum_eligible_worlds_for_usefulness_inference"] == 20
    assert specification["useful_world_balanced_repair_rate"] == 0.5
    assert specification["frozen_discovery_mean_Dcolor"] == [
        0.17506461211422675,
        0.6395382948200473,
        0.024789365765038294,
        0.16060772730068765,
    ]


def test_historical_inputs_are_authenticated_and_confirmation_is_complete():
    loaded = plan.confirmation_inputs(REPO)
    assert len(loaded["worlds"]) == 64
    assert len(loaded["rows"]) == 6144
    assert all(row["split"] == "confirmation" for row in loaded["rows"])
    assert len(loaded["tokens"]["qwen"]["rows"]) == 9216
    assert len(loaded["tokens"]["phi"]["rows"]) == 9216
    assert loaded["input_sha256"]["WORLDS.json"] == (
        "bf97679eef752f00430c165a3595fb04e490a3ec28a956f84bf92fc1335a88b7"
    )


def test_perfect_behavior_has_nonzero_uncertainty_and_no_eligible_repairs():
    rows, predictions, worlds = fixture()
    assessment = plan.score_behavior(rows, predictions, worlds)
    assert assessment["by_probe_kind"]["fact"] == {"correct": 4608, "count": 4608, "accuracy": 1.0}
    assert assessment["coreference"]["pair_count"] == 1536
    assert assessment["coreference"]["discordant_wrong_object"] == 0
    assert assessment["world_prevalence_intervals"]["wrong_object_coreference"]["upper"] > 0
    assert plan.select_cases(rows, predictions, worlds) == []


def test_whole_answer_scoring_rejects_extra_text_and_unfinished_correct_prefix():
    rows, predictions, worlds = fixture()
    predictions[0]["generated_text"] += "\nAn explanation."
    predictions[1]["generated_text"] = " " + rows[1]["expected_answer"].upper() + ". "
    predictions[2].update(terminated=False, truncated=True)
    assessment = plan.score_behavior(rows, predictions, worlds)
    assert assessment["by_probe_kind"]["fact"]["correct"] == 4606
    assert assessment["fact_errors"]["unlisted_or_malformed"] == 2


def test_wrong_object_pairs_and_worlds_are_not_independent_questions():
    rows, predictions, worlds = errors_in_worlds(1)
    assessment = plan.score_behavior(rows, predictions, worlds)
    assert assessment["fact_errors"]["wrong_object"] == 8
    assert assessment["coreference"]["discordant_wrong_object"] == 8
    assert assessment["wrong_object_worlds"] == 1
    assert assessment["wrong_object_coreference_worlds"] == 1


@pytest.mark.parametrize("failure", ["missing", "duplicate", "foreign", "termination", "facts"])
def test_broken_complete_grid_or_completion_metadata_is_rejected(failure):
    rows, predictions, worlds = fixture()
    if failure == "missing":
        predictions.pop()
    elif failure == "duplicate":
        predictions[-1] = predictions[0]
    elif failure == "foreign":
        predictions[0]["id"] = "foreign"
    elif failure == "termination":
        predictions[0]["terminated"] = False
    else:
        worlds[0]["objects"]["A"]["home"] = "changed"
    with pytest.raises(ValueError):
        plan.score_behavior(rows, predictions, worlds)


def test_selection_uses_every_eligible_world_and_is_order_invariant():
    rows, predictions, worlds = errors_in_worlds(24)
    selected = plan.select_cases(rows, predictions, worlds)
    assert len(selected) == 24
    assert {case["world_id"] for case in selected} == {world["world_id"] for world in worlds[:24]}
    assert all(case["initial_wrong_fields"] == ["home"] for case in selected)
    assert all(case["protected_correct_fields"] == ["color", "shape"] for case in selected)
    assert plan.select_cases(rows[::-1], predictions[::-1], worlds[::-1]) == selected


def test_bad_donor_identity_or_unfinished_wrong_answer_does_not_qualify():
    rows, predictions, worlds = errors_in_worlds(1)
    for row, prediction in zip(rows, predictions):
        if row["world_id"] != worlds[0]["world_id"]:
            continue
        if row["probe_kind"] == "identity" and row["query_alias_index"] == 1:
            prediction["generated_text"] = "Object " + row["expected_answer"]
    assert plan.select_cases(rows, predictions, worlds) == []
    rows, predictions, worlds = errors_in_worlds(1)
    for row, prediction in zip(rows, predictions):
        if row["world_id"] == worlds[0]["world_id"] and row["field"] == "home":
            prediction.update(terminated=False, truncated=True)
    assert plan.select_cases(rows, predictions, worlds) == []


def test_no_repair_has_conservative_nonzero_upper_bound():
    rows, predictions, worlds = errors_in_worlds(24)
    cases = plan.select_cases(rows, predictions, worlds)
    assessment = plan.assess_repair(cases, rows, predictions, generations(cases, rows, predictions))
    primary = assessment["by_condition"]["primaryDcolor"]
    assert primary["repaired_initial_errors"] == 0
    assert primary["world_balanced_repair"]["mean"] == 0
    assert primary["world_balanced_repair"]["ci95"][1] == pytest.approx(
        math.sqrt(math.log(40) / 48)
    )
    assert assessment["usefulness"]["conclusion"] == "useful_benefit_excluded"
    assert primary["protected_world_damage"]["upper_one_sided_95"] == pytest.approx(
        1 - math.pow(0.05, 1 / 24)
    )
    assert primary["protected_world_damage"]["risk_below_10_percent_demonstrated"] is False


def test_low_eligibility_is_imprecise_not_a_success_or_failure_gate():
    rows, predictions, worlds = errors_in_worlds(6)
    cases = plan.select_cases(rows, predictions, worlds)
    assessment = plan.assess_repair(cases, rows, predictions, generations(cases, rows, predictions))
    assert assessment["usefulness"]["conclusion"] == "imprecise_low_eligible_world_count"


def test_world_balanced_repair_is_not_pooled_question_accuracy():
    rows, predictions, worlds = errors_in_worlds(24)
    three_error_worlds = {world["world_id"] for world in worlds[12:24]}
    world_lookup = {world["world_id"]: world for world in worlds}
    for row, prediction in zip(rows, predictions):
        if (
            row["world_id"] in three_error_worlds
            and row["probe_kind"] == "fact"
            and row["query_alias_index"] == 0
        ):
            world = world_lookup[row["world_id"]]
            prediction["generated_text"] = world["objects"][world["alias_targets"][2]][row["field"]]
    cases = plan.select_cases(rows, predictions, worlds)
    outputs = generations(cases, rows, predictions)
    row_lookup = {row["id"]: row for row in rows}
    for output in outputs:
        if output["condition"] == "primaryDcolor" and output["field"] == "home":
            output["generated_text"] = row_lookup[output["row_id"]]["expected_answer"]
    primary = plan.assess_repair(cases, rows, predictions, outputs)["by_condition"]["primaryDcolor"]
    assert primary["initial_errors"] == 48
    assert primary["repaired_initial_errors"] == 24
    assert primary["pooled_repair_rate"] == 0.5
    assert primary["world_balanced_repair"]["mean"] == pytest.approx(2 / 3)


def test_malformed_initial_fact_counts_after_wrong_object_eligibility():
    rows, predictions, worlds = errors_in_worlds(1)
    for row, prediction in zip(rows, predictions):
        if (
            row["world_id"] == worlds[0]["world_id"]
            and row["probe_kind"] == "fact"
            and row["query_alias_index"] == 0
            and row["field"] == "color"
        ):
            prediction["generated_text"] = row["expected_answer"] + " with an explanation"
    case = plan.select_cases(rows, predictions, worlds)[0]
    assert case["initial_wrong_fields"] == ["color", "home"]
    assert case["wrong_object_by_field"].keys() == {"home"}
    assert case["protected_correct_fields"] == ["shape"]


def test_complete_repairs_damage_and_paired_mean_control_are_separate():
    rows, predictions, worlds = errors_in_worlds(40)
    cases = plan.select_cases(rows, predictions, worlds)
    assessment = plan.assess_repair(
        cases, rows, predictions, generations(cases, rows, predictions, repair=True, damage=True)
    )
    primary = assessment["by_condition"]["primaryDcolor"]
    assert primary["repaired_initial_errors"] == 40
    assert primary["protected_damaged"] == 80
    assert primary["protected_world_damage"]["damaged_worlds"] == 40
    assert assessment["usefulness"]["conclusion"] == "repair_benefit_only_above_threshold"
    assert assessment["primary_minus_frozen_mean"]["mean"] == 1
    assert assessment["usefulness"]["preservation_is_separate"] is True
    assert assessment["overall_repair_utility"]["status"] == "not_established_by_separate_intervals"
    assert (
        assessment["overall_repair_utility"]["point_estimates_meet_all_practical_targets"] is False
    )


def test_benefit_advantage_and_preservation_have_distinct_point_requirements():
    rows, predictions, worlds = errors_in_worlds(40)
    cases = plan.select_cases(rows, predictions, worlds)
    assessment = plan.assess_repair(
        cases, rows, predictions, generations(cases, rows, predictions, repair=True)
    )
    practical = assessment["overall_repair_utility"]
    assert practical["minimum_advantage_over_frozen_mean"] == 0.25
    assert practical["point_estimates_meet_all_practical_targets"] is True
    assert practical["status"] == "not_established_by_separate_intervals"
    assert practical["simultaneous_95_percent_claim"] is False


def test_binary_bounds_remain_nonzero_after_no_events():
    interval = plan._binary_interval(0, 64)
    assert interval["lower"] == 0
    assert interval["upper"] == pytest.approx(1 - math.pow(0.025, 1 / 64))
    interval = plan._binary_interval(64, 64)
    assert interval["lower"] == pytest.approx(math.pow(0.025, 1 / 64))
    assert interval["upper"] == 1


def test_no_cases_and_incomplete_generation_grid_are_explicit():
    rows, predictions, worlds = fixture()
    empty = plan.assess_repair([], rows, predictions, [])
    assert empty["usefulness"]["conclusion"] == "no_eligible_cases"
    rows, predictions, worlds = errors_in_worlds(1)
    cases = plan.select_cases(rows, predictions, worlds)
    outputs = generations(cases, rows, predictions)
    for broken in (outputs[:-1], outputs + outputs[:1]):
        with pytest.raises(ValueError, match="generation grid"):
            plan.assess_repair(cases, rows, predictions, broken)
    outputs[0]["generated_text"] = "changed self output"
    with pytest.raises(ValueError, match="self replay"):
        plan.assess_repair(cases, rows, predictions, outputs)


def test_cases_cannot_be_replaced_after_seeing_repair_answers():
    rows, predictions, worlds = errors_in_worlds(2)
    cases = plan.select_cases(rows, predictions, worlds)
    tampered = copy.deepcopy(cases)
    tampered[0]["protected_correct_fields"] = []
    with pytest.raises(ValueError, match="selected cases"):
        plan.assess_repair(tampered, rows, predictions, generations(cases, rows, predictions))
