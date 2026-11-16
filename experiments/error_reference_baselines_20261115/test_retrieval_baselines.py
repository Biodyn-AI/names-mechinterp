"""Model-free specifications: cue predictions, strict outcomes and world-held-out fits."""

import copy
from pathlib import Path

import numpy as np
import pytest
import retrieval_baselines as baseline

PROTOCOL_DIR = Path(__file__).resolve().parents[1] / "error_reference_study_20261114"


@pytest.fixture(scope="module")
def cases():
    protocol = baseline.load_protocol(PROTOCOL_DIR)
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
    return protocol, rows, predictions, worlds


def test_cues_use_semantic_binding_and_physical_alias_line():
    world = {
        "object_order": ["D", "B", "A", "C"],
        "alias_order": [2, 0, 3, 1],
        "alias_targets": ["A", "A", "C", "D"],
    }
    row = {"query_alias_index": 1, "focus_condition": "distractor"}
    choices = baseline.cue_classes(row, world)
    assert choices == [0, 2, 2, 3, 3, 2, None, 4]
    row["focus_condition"] = "none"
    assert baseline.cue_classes(row, world)[1] is None
    distribution = baseline.component_distributions(row, world)
    assert distribution.shape == (8, 5)
    assert np.all(distribution > 0)
    np.testing.assert_allclose(distribution.sum(axis=1), 1)
    np.testing.assert_allclose(distribution[1, :4], distribution[1, 0])
    assert distribution[1, 4] < distribution[1, 0]


def test_outcomes_keep_completed_malformed_and_truncation_distinct(cases):
    _, rows, predictions, worlds = cases
    row = rows[0]
    world = {w["world_id"]: w for w in worlds}[row["world_id"]]
    other = next(obj for obj in "ABCD" if obj != world["alias_targets"][0])
    prediction = dict(
        predictions[0], generated_text=world["objects"][other][row["field"]]
    )
    assert baseline.observed_outcome(row, prediction, world)["class"] == "ABCD".index(
        other
    )
    prediction["generated_text"] += "\nAn explanation."
    assert baseline.observed_outcome(row, prediction, world)["malformed"] is True
    prediction.update(
        generated_text=row["expected_answer"], terminated=False, truncated=True
    )
    outcome = baseline.observed_outcome(row, prediction, world)
    assert outcome == {"class": 4, "malformed": False, "truncated": True}
    prediction.update(terminated=True, truncated=True)
    with pytest.raises(ValueError, match="termination"):
        baseline.observed_outcome(row, prediction, world)


def test_em_recovers_two_component_weights_and_finite_probabilities():
    components = np.tile(np.eye(5)[[0, 1]], (100, 1, 1))
    components = baseline.smooth(components)
    observed = np.array([0] * 70 + [1] * 30)
    fitted = baseline.fit_mixture(components, observed)
    assert fitted["converged"] is True
    assert fitted["weights"][0] == pytest.approx(0.7, abs=0.002)
    assert sum(fitted["weights"]) == pytest.approx(1)
    assert min(fitted["weights"]) > 0
    assert fitted["objective_final"] >= fitted["objective_initial"]
    assert np.isfinite(baseline.predict(components, fitted["weights"])).all()
    with pytest.raises(ValueError):
        baseline.fit_mixture(components, np.array([5] * 100))


def test_exact_balanced_whole_world_folds(cases):
    _, _, _, worlds = cases
    identities = [w["world_id"] for w in worlds]
    folds = baseline.world_folds(identities)
    assert folds == baseline.world_folds(list(reversed(identities)))
    assert [list(folds.values()).count(i) for i in range(4)] == [8] * 4
    with pytest.raises(ValueError):
        baseline.world_folds(identities[:-1])


def test_conditioned_wrong_object_metrics_do_not_reward_binding_majority():
    probability = np.array([[0.8, 0.1, 0.06, 0.03, 0.01], [0.8, 0.06, 0.1, 0.03, 0.01]])
    metrics = baseline.metrics(probability, np.array([1, 2]), np.array([0, 0]))
    assert metrics["choice_accuracy"] == 0
    assert metrics["wrong_object_count"] == 2
    assert metrics["wrong_object_exact_accuracy"] == 0
    assert metrics["wrong_object_conditional_accuracy"] == 1
    assert np.isfinite(metrics["wrong_object_conditional_nll"])


def test_complete_validation_rejects_forged_worlds_rows_and_missing_predictions(cases):
    protocol, rows, predictions, worlds = cases
    baseline.validate_cases(rows, predictions, worlds, protocol)
    for broken in (predictions[:-1], predictions + predictions[:1]):
        with pytest.raises(ValueError):
            baseline.validate_cases(rows, broken, worlds, protocol)
    forged = copy.deepcopy(worlds)
    forged[0]["objects"]["A"]["color"] = "invented"
    with pytest.raises(ValueError, match="generator"):
        baseline.validate_cases(rows, predictions, forged, protocol)
    forged_rows = copy.deepcopy(rows)
    forged_rows[0]["expected_answer"] = "invented"
    with pytest.raises(ValueError, match="generator"):
        baseline.validate_cases(forged_rows, predictions, worlds, protocol)


def test_cross_validation_uses_only_training_worlds_and_records_every_fact(cases):
    protocol, rows, predictions, worlds = cases
    report = baseline.analyze(
        rows, predictions, worlds, protocol, model_name="synthetic"
    )
    assert report["fact_count"] == 2304
    assert report["cv_metrics"]["binding"]["choice_accuracy"] == 1
    assert len(report["per_world"]) == 32
    assert sum(len(w["outcomes"]) for w in report["per_world"]) == 2304
    for fold in report["folds"]:
        assert len(fold["fit_worlds"]) == 24
        assert len(fold["evaluation_worlds"]) == 8
        assert not set(fold["fit_worlds"]) & set(fold["evaluation_worlds"])
        assert sum(fold["mixture_fit"]["weights"]) == pytest.approx(1)
    assert report["confirmation_outputs_read"] is False
    assert report["formal_novelty_inference"] is False
    assert any(
        "do not cover learned lexical value/name priors" in limitation
        and "not proof of a novel mechanism" in limitation
        for limitation in report["limitations"]
    )


def test_changing_evaluation_outcomes_cannot_change_that_folds_fit(cases):
    protocol, rows, predictions, worlds = cases
    before = baseline.analyze(
        rows, predictions, worlds, protocol, model_name="synthetic"
    )
    held_out = set(before["folds"][0]["evaluation_worlds"])
    changed = copy.deepcopy(predictions)
    for row, prediction in zip(rows, changed):
        if row["world_id"] in held_out and row["probe_kind"] == "fact":
            prediction["generated_text"] = "unlisted"
    after = baseline.analyze(rows, changed, worlds, protocol, model_name="synthetic")
    assert before["folds"][0]["mixture_fit"] == after["folds"][0]["mixture_fit"]
    assert (
        before["folds"][0]["training_majority_class"]
        == after["folds"][0]["training_majority_class"]
    )
    assert before["folds"][1]["mixture_fit"] != after["folds"][1]["mixture_fit"]
