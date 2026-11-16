"""Integrity is offline and never turns missing native data into a replay claim."""

import gzip
import hashlib
import json
import runpy
from pathlib import Path

import pytest

from names_mechinterp.verify import STAGES, digest, native_availability, verify_manifest

ROOT = Path(__file__).resolve().parents[1]


def test_available_tensor_captures_are_distinguished_from_missing_native_logits():
    result = native_availability(ROOT / "data" / "original")
    assert result["native_shards_complete"] is False
    assert result["present_native_shards"] == 0
    assert result["expected_native_shards"] > 0
    assert result["stages"]["decomposition"]["native_endpoint_count"] == 360
    assert len(result["included_tensor_artifacts"]) == 5
    assert result["native_tensor_rescore_performed"] is False


def test_manifest_detects_tampering_and_verifies_exact_gzip_original(tmp_path):
    (tmp_path / "data").mkdir()
    artifact = tmp_path / "data" / "original.json.gz"
    original = b'{"original":true}\n'
    artifact.write_bytes(gzip.compress(original, mtime=0))
    record = {"path": "data/original.json.gz", "size_bytes": artifact.stat().st_size, "sha256": digest(artifact), "uncompressed_sha256": hashlib.sha256(original).hexdigest()}
    (tmp_path / "data" / "MANIFEST.json").write_text(json.dumps({"files": [record]}))
    assert verify_manifest(tmp_path, verify_compressed=True)["verified"]
    artifact.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_manifest(tmp_path)


def test_manifest_rejects_paths_outside_release(tmp_path):
    (tmp_path / "data").mkdir()
    record = {"path": "../../outside", "size_bytes": 0, "sha256": "0" * 64}
    (tmp_path / "data" / "MANIFEST.json").write_text(json.dumps({"files": [record]}))
    with pytest.raises(ValueError, match="escaped"):
        verify_manifest(tmp_path)


def test_manifest_builder_excludes_generated_install_and_cache_files(tmp_path):
    """An editable install must not invalidate scientific artifact hashes."""
    import zipfile

    builder = runpy.run_path(str(ROOT / "experiments" / "release_tools" / "build_manifest.py"))
    original = tmp_path / "data" / "original"
    archive = original / "supporting_data_archive" / "S1_Data.zip"
    archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive, "w"):
        pass
    for relative in STAGES.values():
        folder = original / relative
        folder.mkdir(parents=True)
        (folder / "RAW_INDEX.json").write_text('{"shards": [], "endpoints": {}}')
        (folder / "CUSTODY.json").write_text('{"artifact_sha256": {}}')
    included = tmp_path / "src" / "names_mechinterp" / "scientific.py"
    included.parent.mkdir(parents=True)
    included.write_text("# Scientific release code.\n")
    generated = ["src/names_mechinterp.egg-info/SOURCES.txt", "src/example.dist-info/METADATA",
                 "src/names_mechinterp/__pycache__/scientific.cpython-312.pyc",
                 "tests/.pytest_cache/state", "experiments/.ruff_cache/state",
                 "src/.venv/pyvenv.cfg", "src/.DS_Store"]
    for relative in generated:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Generated, not scientific.\n")
    manifest = builder["build"](tmp_path, tmp_path / "historical")
    paths = {record["path"] for record in manifest["files"]}
    assert "src/names_mechinterp/scientific.py" in paths
    assert not paths.intersection(generated)
    for relative in generated:
        (tmp_path / relative).write_text("Changed by a new installation.\n")
    assert verify_manifest(tmp_path)["verified"]
