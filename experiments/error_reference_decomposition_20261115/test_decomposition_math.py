"""Label-blind nine-cell decomposition, written before the kernels."""

import copy

import decomposition_math as math
import pytest
import torch


def fixture():
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(12)
    values = torch.randn(10, 4, generator=generator)
    regions = {obj: [2 * index, 2 * index + 1] for index, obj in enumerate("ABCD")}
    r = torch.tensor(
        [0.05, 0.1, 0.04, 0.06, 0.1, 0.05, 0.05, 0.05, 0.2, 0.3], dtype=torch.float32
    )
    d = torch.tensor(
        [0.1, 0.05, 0.1, 0.05, 0.05, 0.05, 0.1, 0.1, 0.2, 0.2], dtype=torch.float32
    )
    c = torch.tensor(
        [0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2], dtype=torch.float32
    )
    donor_values = values.clone()
    donor_values[9] += 0.5
    recipient = {"weights": r, "values": values, "head_output": r @ values}
    donor = {"weights": d, "values": donor_values, "head_output": d @ donor_values}
    color = {"weights": c, "values": values.clone(), "head_output": c @ values}
    return recipient, donor, color, regions


def test_nine_cells_self_exact_and_profile_within_total_factorization():
    r, d, c, regions = fixture()
    result = math.decompose(r, d, c, regions, case_id="case", field="shape")
    assert tuple(result["conditions"]) == math.CONDITIONS
    assert len(math.CONDITIONS) == 9
    assert torch.equal(result["conditions"]["self"]["delta"], torch.zeros(4))
    old = r["weights"].double()
    for index, obj in enumerate("ABCD"):
        positions = regions[obj]
        new = (
            old + result["conditions"]["sameFieldWithinOnly"]["delta_weights"].double()
        )
        assert float(new[positions].sum()) == pytest.approx(
            float(old[positions].sum()), abs=2e-8
        )
        expected_within = (
            d["weights"][positions].double() / d["weights"][positions].double().sum()
        )
        assert torch.allclose(
            new[positions] / new[positions].sum(), expected_within, atol=2e-7, rtol=0
        )
    assert result["audit"]["record_delta_equivalence_max_abs"] <= 2e-5
    assert result["audit"]["full_reconstruction_max_abs"] <= 2e-5


def test_actual_head_record_and_nonrecord_residuals_are_separately_preserved():
    r, d, c, regions = fixture()
    result = math.decompose(r, d, c, regions, case_id="case", field="home")
    record = result["conditions"]["sameFieldRecordContribution"]["delta"]
    nonrecord = result["conditions"]["sameFieldNonrecordResidual"]["delta"]
    full = result["conditions"]["sameFieldFullHead"]["delta"]
    assert torch.equal(
        record,
        result["donor_record_contribution"] - result["recipient_record_contribution"],
    )
    assert torch.equal(
        nonrecord,
        result["donor_nonrecord_residual"] - result["recipient_nonrecord_residual"],
    )
    assert torch.allclose(record + nonrecord, full, atol=2e-5, rtol=0)
    assert torch.allclose(r["head_output"] + full, d["head_output"], atol=2e-5, rtol=0)


def test_random_is_reproducible_and_matched_only_to_full_head_norm():
    args = fixture()
    first = math.decompose(*args, case_id="case", field="color")
    second = math.decompose(*args, case_id="case", field="color")
    random = first["conditions"]["normMatchedRandomFullHead"]
    assert torch.equal(
        random["delta"], second["conditions"]["normMatchedRandomFullHead"]["delta"]
    )
    assert random["relative_norm_error"] <= 1e-5
    assert random["matched_condition"] == "sameFieldFullHead"


@pytest.mark.parametrize(
    "mutation",
    ["record_values", "donor_support", "head_reconstruction", "row_sum", "nonfinite"],
)
def test_bad_record_values_support_reconstruction_or_native_rows_fail_closed(mutation):
    r, d, c, regions = fixture()
    if mutation == "record_values":
        d["values"][0, 0] += 1e-5
    elif mutation == "donor_support":
        d["weights"][8] += d["weights"][:2].sum()
        d["weights"][:2] = 0
        d["head_output"] = d["weights"] @ d["values"]
    elif mutation == "head_reconstruction":
        r["head_output"][0] += 1e-3
    elif mutation == "row_sum":
        r["weights"] *= 0.5
    else:
        r["values"][0, 0] = float("nan")
    with pytest.raises(ValueError):
        math.decompose(r, d, c, regions, case_id="case", field="home")


def test_identical_donor_has_exact_zero_full_and_random_deltas():
    r, _, _, regions = fixture()
    result = math.decompose(
        r, copy.deepcopy(r), copy.deepcopy(r), regions, case_id="same", field="shape"
    )
    for condition in ("self", "sameFieldFullHead", "normMatchedRandomFullHead"):
        assert torch.equal(result["conditions"][condition]["delta"], torch.zeros(4))
