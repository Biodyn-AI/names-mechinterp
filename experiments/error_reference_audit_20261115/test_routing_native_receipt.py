"""A receipt attests exact replay, including negative scientific outcomes."""

from types import SimpleNamespace

import pytest
from routing_native_receipt import make_receipt


def test_exact_positive_and_negative_results_both_bind_all_artifacts():
    for passed, count in ((True, 112), (False, 32)):
        calls, verified = [], []
        result = {
            "exact_match": True,
            "native_forward_count": count,
            "confirmation_calls": 0,
            "assessment": {"practical_gate_passed": passed},
            "further_validation_authorized": False,
        }
        runner = SimpleNamespace(
            rescore=lambda root, seen=calls, value=result: seen.append(root) or value
        )
        old = SimpleNamespace(
            verify_file=lambda path, seen=verified: seen.append(str(path)) or "a" * 64,
            sha256=lambda path: "b" * 64,
        )
        receipt = make_receipt(runner, old, "/routing-study")
        assert calls == ["/routing-study"]
        assert receipt["assessment"]["practical_gate_passed"] is passed
        assert len(verified) == 12
        assert "/routing-study/routing/SELECTED_PLAN.json" in verified
        assert "/routing-study/routing/INTERVENTIONS.pt" in verified
        assert receipt["source_manifest_sha256"] == "a" * 64
        assert receipt["natural_sha256"] == "a" * 64
        assert receipt["capture_summaries_sha256"] == "a" * 64
        assert receipt["receipt_source_sha256"] == "b" * 64


def test_failed_incomplete_or_out_of_scope_rescores_cannot_be_attested():
    good = {"exact_match": True, "native_forward_count": 112, "confirmation_calls": 0}
    invalid = [{**good, "exact_match": False}, {**good, "confirmation_calls": 1}]
    invalid += [
        {**good, "native_forward_count": count} for count in (31, 113, True, 32.0)
    ]
    old = SimpleNamespace(
        verify_file=lambda path: pytest.fail("Invalid audit must not be anchored")
    )
    for result in invalid:
        with pytest.raises(ValueError):
            make_receipt(
                SimpleNamespace(rescore=lambda *args, value=result: value),
                old,
                "/routing-study",
            )
