"""Bind saved-screen rescoring to its separate registration and raw inputs."""

from types import SimpleNamespace

import pytest

from screen_native_receipt import make_receipt


def test_screen_receipt_anchors_every_required_artifact():
    screen = SimpleNamespace(
        rescore=lambda root: {
            "exact_match": True,
            "confirmation_calls": 0,
            "physical_counts": {"native": 1144, "backbone_prefix": 16},
            "complete_answer_repair_established": False,
        }
    )
    old = SimpleNamespace(
        verify_file=lambda path: "f" * 64, sha256=lambda path: "a" * 64
    )
    receipt = make_receipt(screen, old, "/screen")
    for key in (
        "registration_sha256",
        "custody_sha256",
        "result_sha256",
        "raw_index_sha256",
        "source_manifest_sha256",
        "plan_sha256",
        "inputs_sha256",
        "prefixes_sha256",
    ):
        assert receipt[key] == "f" * 64
    assert receipt["receipt_source_sha256"] == "a" * 64


def test_failed_or_out_of_scope_screen_cannot_be_attested():
    old = SimpleNamespace(
        verify_file=lambda path: "f" * 64, sha256=lambda path: "a" * 64
    )
    with pytest.raises(ValueError):
        make_receipt(
            SimpleNamespace(rescore=lambda root: {"exact_match": False}), old, "/screen"
        )
