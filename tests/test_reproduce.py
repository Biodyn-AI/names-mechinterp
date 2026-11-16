"""Regression tests against released scientific data, not model inference."""

import copy
import json
import socket
from pathlib import Path

import pytest

from names_mechinterp import reproduce

DATA = Path(__file__).resolve().parents[1] / "data" / "processed"


@pytest.fixture(scope="module")
def report():
    return reproduce.reproduce(DATA)


def test_behavior_literal_counts_keep_worlds_and_repeated_questions_distinct(report):
    qwen, phi = report["behavior"]["qwen"], report["behavior"]["phi"]
    assert qwen["by_probe_kind"]["fact"]["count"] == 2304
    assert qwen["fact_errors"] == {"wrong_object": 647, "unlisted_or_malformed": 0}
    assert phi["fact_errors"] == {"wrong_object": 260, "unlisted_or_malformed": 64}
    assert qwen["coreference"]["discordant_wrong_object"] == 117
    assert phi["coreference"]["discordant_wrong_object"] == 69
    assert phi["coreference"]["discordant_one_correct"] == 87
    assert report["independent_worlds"] == 32 and report["questions_per_model"] == 3072


def test_decomposition_keeps_all_negative_gates_and_prior_result(report):
    counts = report["decomposition"]["conditions"]
    assert counts["oldDcolorMass"]["repaired"] == 1
    assert all(value["repaired"] == 0 for key, value in counts.items() if key != "oldDcolorMass")
    assert all(value["protected_damaged"] == 0 for value in counts.values())
    assert report["decomposition"]["all_fixed_branches_failed"]
    assert report["decomposition"]["new_entity_mechanism_established"] is False
    assert report["native_logit_replay_performed"] is False


def test_ported_behavior_and_additional_intervals_match_frozen_outputs(report):
    original = DATA.parent / "original" / "cloud_reference_error_metadata_20261115"
    for model in ("qwen", "phi"):
        assert report["behavior"][model] == json.loads((original / model / "RESULT.json").read_text())
    evidence = json.loads((DATA / "evidence_digest.json").read_text())
    assert report["retrospective_descriptive_world_intervals"] == evidence["additional_retrospective_world_intervals"]


def test_literal_parser_rejects_a_silently_changed_truth_label():
    rows = json.loads((DATA / "prompts.json").read_text())
    changed = copy.deepcopy(rows[0])
    changed["expected_answer"] = "invented"
    with pytest.raises(ValueError, match="printed registry"):
        reproduce.validate_literal_rows([changed])


def test_default_suite_has_an_active_network_guard():
    with pytest.raises(AssertionError, match="Network access"):
        socket.create_connection(("example.com", 443))
