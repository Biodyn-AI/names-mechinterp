"""Cross-check the independent literal parser against every frozen question."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "error_reference_study_20261114")
)
import error_protocol
from audit_behavior import audit, literal_answer


def test_every_literal_label_in_both_wordings_and_splits():
    for row in error_protocol.build_study()["rows"]:
        assert literal_answer(row["text"])[0] == row["expected_answer"]


def test_ambiguous_binding_and_records_are_rejected():
    text = error_protocol.build_study()["rows"][0]["text"]
    with pytest.raises(ValueError):
        literal_answer(text + "\nAlias hidden -> Object A.")
    with pytest.raises(ValueError):
        literal_answer(
            text + "\nObject A | nameplate X | color blue | shape star | home river."
        )


def test_complete_audit_rejects_duplicates_and_wrong_registered_bytes(tmp_path):
    rows = error_protocol.build_study()["rows"]
    content = json.dumps(rows).encode()
    path = tmp_path / "rows.json"
    path.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    predictions = [
        {
            "id": row["id"],
            "expected_answer": row["expected_answer"],
            "generated_text": row["expected_answer"],
            "terminated": True,
            "truncated": False,
        }
        for row in rows
        if row["split"] == "development"
    ]
    report = audit(path, predictions, digest)
    assert report["counts"]["correct"] == report["counts"]["count"] == 3072
    assert report["feasibility_passed"] is False
    with pytest.raises(ValueError, match="coverage"):
        audit(path, predictions[:-1] + predictions[:1], digest)
    with pytest.raises(ValueError, match="registration"):
        audit(path, predictions, "0" * 64)
