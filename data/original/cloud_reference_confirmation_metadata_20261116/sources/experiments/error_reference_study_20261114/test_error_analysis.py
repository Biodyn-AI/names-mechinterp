"""Tests specify strict scoring and world-level feasibility before analysis."""

import copy

import pytest

import error_analysis as analysis
import error_protocol as protocol


def _cases():
    study = protocol.build_study()
    rows = [r for r in study["rows"] if r["split"] == "development"]
    worlds = [w for w in study["worlds"] if w["split"] == "development"]
    predictions = [
        {
            "id": r["id"],
            "generated_text": r["expected_answer"],
            "terminated": True,
            "truncated": False,
        }
        for r in rows
    ]
    return rows, predictions, worlds


def test_perfect_model_is_no_go_for_error_mechanism():
    rows, predictions, worlds = _cases()
    result = analysis.score(rows, predictions, worlds)
    assert result["overall"] == {"correct": 3072, "count": 3072, "accuracy": 1.0}
    assert result["feasibility"]["passed"] is False
    assert result["fact_errors"]["wrong_object"] == 0
    assert result["coreference"]["pair_count"] == 768
    assert result["coreference"]["discordant_one_correct"] == 0
    assert result["world_intervals"]["fact_accuracy"]["ci95"] == [1.0, 1.0]
    assert result["causal_calls_authorized"] is False


def test_strict_entire_answer_and_truncation():
    rows, predictions, worlds = _cases()
    predictions[0]["generated_text"] += "\nThis is my answer."
    predictions[1]["generated_text"] = "  " + rows[1]["expected_answer"].upper() + ".\n"
    predictions[2]["terminated"] = False
    predictions[2]["truncated"] = True
    result = analysis.score(rows, predictions, worlds)
    assert result["overall"]["correct"] == 3070
    assert result["fact_errors"]["wrong_object"] == 0
    assert result["fact_errors"]["unlisted_or_malformed"] == 2


def test_feasibility_counts_independent_worlds_not_repeats():
    rows, predictions, worlds = _cases()
    by_world = {w["world_id"]: w for w in worlds}
    for row, prediction in zip(rows, predictions):
        if row["probe_kind"] == "fact" and row["query_alias_index"] == 0:
            world = by_world[row["world_id"]]
            other = world["alias_targets"][2]
            prediction["generated_text"] = world["objects"][other][row["field"]]
    result = analysis.score(rows, predictions, worlds)
    assert result["fact_errors"]["wrong_object"] == 768
    assert result["feasibility"]["wrong_object_worlds"] == 32
    assert result["coreference"]["discordant_one_correct"] == 768
    assert result["feasibility"]["passed"] is True
    assert result["world_intervals"]["fact_accuracy"]["mean"] == pytest.approx(2 / 3)
    # Thousands of repeated errors from one world still fail the world gate.
    predictions = [
        {
            "id": r["id"],
            "generated_text": r["expected_answer"],
            "terminated": True,
            "truncated": False,
        }
        for r in rows
    ]
    for row, prediction in zip(rows, predictions):
        if row["world_id"] == worlds[0]["world_id"] and row["probe_kind"] == "fact":
            world = by_world[row["world_id"]]
            target = world["alias_targets"][row["query_alias_index"]]
            other = next(obj for obj in "ABCD" if obj != target)
            prediction["generated_text"] = world["objects"][other][row["field"]]
    assert analysis.score(rows, predictions, worlds)["feasibility"]["passed"] is False


def test_duplicate_missing_or_foreign_predictions_are_rejected():
    rows, predictions, worlds = _cases()
    for broken in (predictions[:-1], predictions + predictions[:1]):
        with pytest.raises(ValueError):
            analysis.score(rows, broken, worlds)
    broken = copy.deepcopy(predictions)
    broken[0]["id"] = "foreign"
    with pytest.raises(ValueError):
        analysis.score(rows, broken, worlds)
    with pytest.raises(ValueError):
        analysis.score(rows, predictions, worlds[:-1])


def test_missing_or_contradictory_termination_metadata_is_rejected():
    rows, predictions, worlds = _cases()
    del predictions[0]["terminated"]
    with pytest.raises(ValueError, match="termination metadata"):
        analysis.score(rows, predictions, worlds)
    predictions[0]["terminated"] = False
    with pytest.raises(ValueError, match="termination metadata"):
        analysis.score(rows, predictions, worlds)


def test_malformed_coreference_disagreement_cannot_qualify_wrong_object_study():
    rows, predictions, worlds = _cases()
    by_world = {w["world_id"]: w for w in worlds}
    for row, prediction in zip(rows, predictions):
        if row["probe_kind"] != "fact":
            continue
        if row["query_alias_index"] == 0:
            prediction["generated_text"] = "I do not know."
        elif row["query_alias_index"] == 2:
            world = by_world[row["world_id"]]
            prediction["generated_text"] = world["objects"][world["base_object_id"]][
                row["field"]
            ]
    result = analysis.score(rows, predictions, worlds)
    assert result["feasibility"]["wrong_object_worlds"] == 32
    assert result["coreference"]["discordant_one_correct"] == 768
    assert result["coreference"]["discordant_wrong_object"] == 0
    assert result["feasibility"]["passed"] is False


def test_focus_changes_and_cue_matches_are_descriptive_not_causes():
    rows, predictions, worlds = _cases()
    by_world = {w["world_id"]: w for w in worlds}
    for row, prediction in zip(rows, predictions):
        if (
            row["focus_condition"] == "distractor"
            and row["probe_kind"] == "fact"
            and row["query_alias_index"] == 0
        ):
            world = by_world[row["world_id"]]
            prediction["generated_text"] = world["objects"][world["alias_targets"][2]][
                row["field"]
            ]
    result = analysis.score(rows, predictions, worlds)
    assert result["focus_comparisons"]["distractor"]["changed"] == 192
    assert result["focus_comparisons"]["distractor"]["correct_to_wrong"] == 192
    assert result["cue_matches_on_wrong_object"]["focus_object"] == 192
    assert "not causal" in result["cue_match_interpretation"]


def test_bootstrap_is_reproducible_and_resamples_worlds():
    first = analysis.world_interval([0.0, 0.5, 1.0])
    assert first == analysis.world_interval([0.0, 0.5, 1.0])
    assert first["worlds"] == 3 and first["resamples"] == 2000
    assert first["mean"] == 0.5
