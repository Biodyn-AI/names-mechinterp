"""CPU-only canonical first-token effects, never a substitute for whole answers."""

import math

import pytest
import torch

import effect_readout as effects


def test_scalar_margin_and_probability_use_the_complete_saved_vocabulary():
    baseline = torch.tensor([0.0, 2.0, -1.0], dtype=torch.float32)
    native = torch.tensor([1.0, 1.0, -1.0], dtype=torch.float32)
    measured = effects.vector_effect(native, baseline, 0, 1)
    assert measured["canonical_E_minus_W_before"] == -2
    assert measured["canonical_E_minus_W_after"] == 0
    assert measured["canonical_E_minus_W_change"] == 2
    assert measured["canonical_E_probability_before"] == pytest.approx(
        1 / (1 + math.exp(2) + math.exp(-1))
    )
    assert measured["canonical_E_probability_after"] == pytest.approx(
        math.exp(1) / (2 * math.exp(1) + math.exp(-1))
    )
    assert measured["native_vector_changed"] and measured["argmax_changed"]


def test_identical_vectors_have_exactly_zero_changes_and_no_wrong_margin_for_protected():
    native = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    measured = effects.vector_effect(native, native.clone(), 1, None)
    assert measured["canonical_E_probability_change"] == 0
    assert measured["canonical_E_minus_W_change"] is None
    assert measured["native_max_abs_change"] == 0
    assert not measured["native_vector_changed"] and not measured["argmax_changed"]


def test_bitwise_change_distinguishes_signed_zero_without_inventing_numeric_change():
    native = torch.tensor([-0.0, 1.0], dtype=torch.float32)
    baseline = torch.tensor([0.0, 1.0], dtype=torch.float32)
    measured = effects.vector_effect(native, baseline, 0, None)
    assert measured["native_vector_changed"]
    assert measured["native_max_abs_change"] == 0


def test_multi_token_and_noninteger_canonical_labels_are_rejected():
    assert effects.canonical_id({"expected_answer_ids": [4]}) == 4
    for value in ([4, 5], [], [True], [-1]):
        with pytest.raises(ValueError):
            effects.canonical_id({"expected_answer_ids": value})


def test_only_finished_one_value_plus_eos_old_errors_supply_the_fixed_wrong_token():
    old = {
        "generated_ids": [5, 9],
        "terminated": True,
        "truncated": False,
        "stop_reason": "eos",
    }
    assert effects.wrong_token(old, 4, False) == 5
    assert effects.wrong_token(old, 4, True) is None
    for changed in (
        {**old, "generated_ids": [5, 6, 9]},
        {**old, "generated_ids": [4, 9]},
        {**old, "truncated": True},
    ):
        with pytest.raises(ValueError):
            effects.wrong_token(changed, 4, False)


@pytest.mark.parametrize("mutation", ["dtype", "nan", "shape", "bounds"])
def test_malformed_native_vectors_and_token_ids_fail_closed(mutation):
    baseline = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32)
    native, token = baseline.clone(), 0
    if mutation == "dtype":
        native = native.double()
    elif mutation == "nan":
        native[0] = float("nan")
    elif mutation == "shape":
        native = native.reshape(1, 3)
    else:
        token = 3
    with pytest.raises(ValueError):
        effects.vector_effect(native, baseline, token, 1)


def test_summary_keeps_conditional_error_margin_separate_from_all_question_probability():
    first = effects.vector_effect(
        torch.tensor([1.0, 1.0, 0.0]), torch.tensor([0.0, 2.0, 0.0]), 0, 1
    )
    second = effects.vector_effect(
        torch.tensor([1.0, 1.0, 0.0]), torch.tensor([1.0, 1.0, 0.0]), 0, None
    )
    result = effects.summarize([first, second])
    assert result["questions"] == 2 and result["old_error_questions"] == 1
    assert result["mean_canonical_E_minus_W_change_on_old_errors"] == 2
    assert result["native_vectors_changed"] == 1
    assert result["argmax_changes"] == 1
