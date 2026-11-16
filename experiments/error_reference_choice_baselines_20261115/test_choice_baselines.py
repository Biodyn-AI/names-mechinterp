"""Tests written before running expanded controls on research outcomes."""

import numpy as np
import pytest
from choice_baselines import expanded_components, paired_interval, value_prior


def fixture():
    world = {
        "alias_targets": ["B", "B", "D", "A"],
        "alias_order": [2, 0, 3, 1],
        "object_order": ["D", "B", "A", "C"],
        "objects": {
            obj: {"color": val}
            for obj, val in zip("ABCD", ("red", "blue", "green", "gray"))
        },
    }
    row = {
        "query_alias_index": 0,
        "focus_condition": "none",
        "field": "color",
        "aliases": ["cat", "bat", "date", "fig"],
    }
    return row, world


def test_added_components_include_constant_ids_and_middle_positions():
    row, world = fixture()
    probabilities = expanded_components(row, world)
    assert probabilities.shape == (14, 5)
    assert np.allclose(probabilities.sum(axis=1), 1)
    assert probabilities[8:].argmax(axis=1).tolist() == [0, 1, 2, 3, 1, 0]


def test_value_prior_uses_training_only_and_maps_words_not_object_ids():
    row, world = fixture()
    training = [(row, world, 1), (row, world, 1), (row, world, 4)]
    probabilities = value_prior(training, [(row, world)], by_alias=True)
    assert probabilities.shape == (1, 5)
    assert np.allclose(probabilities.sum(axis=1), 1)
    assert probabilities[0, 1] > probabilities[0, 4] > probabilities[0, 0]
    reversed_world = {
        **world,
        "objects": {**world["objects"], "A": {"color": "blue"}, "B": {"color": "red"}},
    }
    moved = value_prior(training, [(row, reversed_world)], by_alias=True)
    assert moved[0, 0] == probabilities[0, 1]


def test_unseen_alias_falls_back_to_training_field_prior():
    row, world = fixture()
    unseen = {**row, "aliases": ["apple", "bat", "date", "fig"]}
    training = [(row, world, 1)]
    assert np.array_equal(
        value_prior(training, [(unseen, world)], True),
        value_prior(training, [(row, world)], False),
    )


def test_paired_uncertainty_resamples_worlds():
    summary = paired_interval([1.0] * 32)
    assert summary["mean"] == 1
    assert summary["percentile_95"] == [1, 1]
    with pytest.raises(ValueError):
        paired_interval([1.0] * 31)
