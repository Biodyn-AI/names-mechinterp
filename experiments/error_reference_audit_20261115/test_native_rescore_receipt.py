"""Receipt wrapper must bind an actual successful native audit to all inputs."""

from types import SimpleNamespace

import pytest
from native_rescore_receipt import make_receipt


def test_successful_receipt_binds_every_required_anchor():
    calls = []
    runner = SimpleNamespace(
        rescore=lambda path, model: (
            calls.append((path, model))
            or {
                "exact_match": True,
                "native_steps": 6144,
                "confirmation_calls": 0,
                "causal_calls": 0,
            }
        )
    )
    old = SimpleNamespace(
        verify_file=lambda path: "f" * 64, sha256=lambda path: "a" * 64
    )
    receipt = make_receipt(runner, old, "/study", "qwen")
    assert calls == [("/study", "qwen")]
    assert receipt["native_steps"] == 6144
    for key in (
        "registration_sha256",
        "custody_sha256",
        "result_sha256",
        "raw_index_sha256",
        "generations_sha256",
        "source_manifest_sha256",
    ):
        assert receipt[key] == "f" * 64
    assert receipt["receipt_source_sha256"] == "a" * 64


def test_unsuccessful_or_out_of_scope_native_audit_cannot_be_attested():
    old = SimpleNamespace(
        verify_file=lambda path: "f" * 64, sha256=lambda path: "a" * 64
    )
    for result in (
        {"exact_match": False},
        {
            "exact_match": True,
            "native_steps": 6144,
            "confirmation_calls": 1,
            "causal_calls": 0,
        },
    ):
        with pytest.raises(ValueError):
            make_receipt(
                SimpleNamespace(
                    rescore=lambda *args, receipt_result=result: receipt_result
                ),
                old,
                "/study",
                "qwen",
            )
