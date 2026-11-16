"""Receipt integrity tests use synthetic files and no research-model execution."""

import hashlib
import json
from types import SimpleNamespace

import audit_receipt
import pytest


def _seal(path, value):
    payload = json.dumps(value, sort_keys=True).encode()
    path.write_bytes(payload)
    path.with_name(path.name + ".sha256").write_text(
        hashlib.sha256(payload).hexdigest() + "\n"
    )
    return hashlib.sha256(payload).hexdigest()


def fixture(tmp_path):
    run, source = tmp_path / "run", tmp_path / "source"
    run.mkdir()
    source.mkdir()
    (run / "sources").mkdir()
    stage = run / "decomposition"
    stage.mkdir()
    closure = {}
    for name in ("decomposition_run.py", *[f"source{i}.py" for i in range(6)]):
        payload = b"# synthetic frozen source\n"
        (source / name).write_bytes(payload)
        (run / "sources" / name).write_bytes(payload)
        closure[name] = hashlib.sha256(payload).hexdigest()
    registered = {
        name: _seal(run / name, value)
        for name, value in [
            ("SOURCES.json", {"new": closure}),
            ("INPUTS.json", {}),
            ("PLAN.json", {}),
        ]
    }
    registration = _seal(run / "REGISTRATION.json", {"input_sha256": registered})
    result = {
        "status": "completed_fixed_head_decomposition",
        "native_forward_count": 360,
        "assessment": {"all_fixed_branches_failed": True},
    }
    artifacts = {}
    for name in audit_receipt.STAGE_FILES:
        value = (
            result
            if name == "RESULT.json"
            else {"attempted_native": 360, "saved_native": 360}
            if name == "COUNTS.json"
            else {"endpoints": {str(i): {} for i in range(360)}}
            if name == "RAW_INDEX.json"
            else {}
        )
        artifacts[name] = _seal(stage / name, value)
    _seal(
        stage / "CUSTODY.json",
        {
            "registration_sha256": registration,
            "native_forward_count": 360,
            "artifact_sha256": artifacts,
        },
    )
    answer = {
        "exact_match": True,
        **result,
        "confirmation_calls": 0,
        "automatic_followup_authorized": False,
    }
    return run, source, answer


def test_negative_science_can_have_a_successful_anchored_audit(tmp_path, monkeypatch):
    run, source, answer = fixture(tmp_path)
    monkeypatch.setattr(
        audit_receipt,
        "_load_runner",
        lambda path: SimpleNamespace(rescore=lambda root: answer),
    )
    receipt_path = tmp_path / "receipt.json"
    receipt = audit_receipt.run_audit(run, source, receipt_path)
    assert receipt["exact_match"] is True
    assert receipt["assessment"]["all_fixed_branches_failed"] is True
    assert receipt["native_forward_count"] == 360
    assert len(receipt["runtime_source_sha256"]) == 7
    assert receipt["automatic_followup_authorized"] is False
    assert (
        receipt_path.exists() and receipt_path.with_name("receipt.json.sha256").exists()
    )
    with pytest.raises(FileExistsError):
        audit_receipt.run_audit(run, source, receipt_path)


def test_corrupt_artifact_is_rejected_before_rescore(tmp_path, monkeypatch):
    run, source, _ = fixture(tmp_path)
    (run / "decomposition/NATURAL.pt").write_bytes(b"changed")
    monkeypatch.setattr(
        audit_receipt,
        "_load_runner",
        lambda path: pytest.fail("Imported before hash checks"),
    )
    with pytest.raises(ValueError):
        audit_receipt.run_audit(run, source, tmp_path / "receipt.json")


def test_source_changes_or_rescore_mismatch_cannot_produce_receipt(
    tmp_path, monkeypatch
):
    run, source, answer = fixture(tmp_path)

    def changed(_root):
        (source / "source0.py").write_text("# changed after audit began\n")
        return answer

    monkeypatch.setattr(
        audit_receipt, "_load_runner", lambda path: SimpleNamespace(rescore=changed)
    )
    with pytest.raises(ValueError):
        audit_receipt.run_audit(run, source, tmp_path / "receipt.json")
    assert not (tmp_path / "receipt.json").exists()


def test_receipt_cannot_mutate_sealed_source_or_evidence_namespace(tmp_path):
    run, source, _ = fixture(tmp_path)
    for target in (run / "AUDIT.json", source / "AUDIT.json"):
        with pytest.raises(ValueError):
            audit_receipt.run_audit(run, source, target)


@pytest.mark.parametrize(
    "mutation", ["exact_match", "native_forward_count", "assessment"]
)
def test_failed_or_inconsistent_rescore_leaves_no_success_receipt(
    tmp_path, monkeypatch, mutation
):
    run, source, answer = fixture(tmp_path)
    answer[mutation] = (
        False
        if mutation == "exact_match"
        else 359
        if mutation == "native_forward_count"
        else {"all_fixed_branches_failed": False}
    )
    monkeypatch.setattr(
        audit_receipt,
        "_load_runner",
        lambda path: SimpleNamespace(rescore=lambda root: answer),
    )
    target = tmp_path / "receipt.json"
    with pytest.raises(ValueError):
        audit_receipt.run_audit(run, source, target)
    assert not target.exists()
