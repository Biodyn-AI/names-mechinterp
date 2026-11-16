"""Retrospective descriptive intervals retain whole-world repeated measures."""

import copy

import pytest
from descriptive_world_bootstrap import intervals


def fixture():
    summaries = {
        f"world{i:02d}": {
            "fact_accuracy": {"count": 72},
            "wrong_object": 18,
            "wrong_object_coref": 2,
            "coref_pairs": 24,
        }
        for i in range(32)
    }
    return {
        model: {"world_summaries": copy.deepcopy(summaries)}
        for model in ("qwen", "phi")
    }


def test_constant_world_rates_have_deterministic_degenerate_intervals():
    first = intervals(fixture())
    assert first == intervals(fixture())
    assert first["models"]["qwen"]["wrong_object_fact_rate"] == {
        "mean": 0.25,
        "ci95": [0.25, 0.25],
        "worlds": 32,
    }
    assert first["models"]["phi"]["eligible_wrong_object_coref_rate"][
        "mean"
    ] == pytest.approx(2 / 24)
    assert first["retrospective_descriptive"] is True
    assert first["p_values_computed"] is False


def test_world_coverage_and_denominators_are_checked():
    for change in ("world", "facts", "pairs"):
        values = fixture()
        if change == "world":
            values["phi"]["world_summaries"].pop("world00")
        elif change == "facts":
            values["phi"]["world_summaries"]["world00"]["fact_accuracy"]["count"] = 71
        else:
            values["phi"]["world_summaries"]["world00"]["coref_pairs"] = 23
        with pytest.raises(ValueError):
            intervals(values)
