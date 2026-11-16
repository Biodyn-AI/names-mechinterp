"""Byte-integrity tests only; no tensors are loaded or interpreted."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_confirmation_native", ROOT / "tools/verify_confirmation_native.py"
)
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def seal(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = content if isinstance(content, bytes) else json.dumps(content).encode()
    path.write_bytes(payload)
    value = hashlib.sha256(payload).hexdigest()
    path.with_name(path.name + ".sha256").write_text(value + "\n")
    return value


@pytest.fixture
def folders(tmp_path):
    metadata, native = tmp_path / "metadata", tmp_path / "native"
    for stage in verifier.STAGES:
        sha = seal(
            native / stage / "raw-0000.safetensors", b"synthetic bytes, not a tensor"
        )
        index = {
            "shards": [
                {"file": "raw-0000.safetensors", "sha256": sha, "endpoint_count": 1}
            ],
            "endpoints": {
                "one": {
                    "file": "raw-0000.safetensors",
                    "key": "one",
                    "dtype": "float32",
                    "shape": [2],
                }
            },
        }
        index_sha = seal(metadata / stage / "RAW_INDEX.json", index)
        artifacts = {"RAW_INDEX.json": index_sha, "raw-0000.safetensors": sha}
        if stage == "repair":
            artifacts["case-000.pt"] = seal(
                native / stage / "case-000.pt", b"synthetic capture, never loaded"
            )
        seal(metadata / stage / "CUSTODY.json", {"artifact_sha256": artifacts})
    return metadata, native


def test_complete_integrity_is_distinct_from_native_mathematical_replay(folders):
    report = verifier.verify(*folders)
    assert report["complete"] and report["present_files_verified"]
    assert report["expected_native_shards"] == report["present_native_shards"] == 3
    assert report["expected_capture_files"] == report["present_capture_files"] == 1
    assert report["native_mathematical_replay_performed"] is False
    assert report["new_model_forward_count"] == 0


def test_missing_payload_never_becomes_a_complete_deposit(folders, tmp_path):
    metadata, native = folders
    (native / "qwen/raw-0000.safetensors").unlink()
    report = verifier.verify(metadata, native)
    assert not report["complete"] and len(report["missing"]) == 1
    args = ["--metadata-root", str(metadata), "--native-root", str(native)]
    assert verifier.main(args) == 1
    assert verifier.main([*args, "--allow-missing"]) == 0


@pytest.mark.parametrize(
    "problem", ["bytes", "metadata_sidecar", "payload_sidecar", "symlink"]
)
def test_changed_bytes_sidecars_and_escaped_paths_fail_closed(
    folders, tmp_path, problem
):
    metadata, native = folders
    path = native / "qwen/raw-0000.safetensors"
    if problem == "bytes":
        path.write_bytes(b"a different synthetic file")
    elif problem == "metadata_sidecar":
        (metadata / "qwen/RAW_INDEX.json.sha256").write_text("0" * 64)
    elif problem == "payload_sidecar":
        path.with_name(path.name + ".sha256").write_text("0" * 64)
    else:
        outside = tmp_path / "outside"
        outside.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(outside)
    with pytest.raises(ValueError):
        verifier.verify(metadata, native)


def test_output_is_exclusive(folders, tmp_path):
    output = tmp_path / "receipt.json"
    args = [
        "--metadata-root",
        str(folders[0]),
        "--native-root",
        str(folders[1]),
        "--output",
        str(output),
    ]
    assert verifier.main(args) == 0
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        verifier.main(args)
    assert output.read_bytes() == before


def test_actual_confirmation_metadata_index_contains_620_shards_and_30_captures():
    metadata = ROOT / "data/original/cloud_reference_confirmation_metadata_20261116"
    records, _ = verifier.requirements(metadata)
    assert sum(r["kind"] == "native_shard" for r in records) == 620
    assert sum(r["kind"] == "case_capture" for r in records) == 30
