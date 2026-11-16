"""Synthetic native-file archives: integrity, allowlists, safe paths and no overwrite."""

import hashlib
import importlib.util
import json
import tarfile
from pathlib import Path

import pytest

from names_mechinterp.verify import STAGES

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "package_native_release", ROOT / "tools/package_native_release.py"
)
packaging = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(packaging)


def digest(content):
    return hashlib.sha256(content).hexdigest()


def seal(path, value):
    content = (json.dumps(value, sort_keys=True) + "\n").encode()
    path.write_bytes(content)
    path.with_name(path.name + ".sha256").write_text(digest(content) + "\n")
    return digest(content)


def fixture(tmp_path, *, phase="confirmation", stage="qwen"):
    relative = stage if phase == "confirmation" else STAGES[stage]
    metadata, native = tmp_path / "metadata", tmp_path / "native"
    folder, payloads = metadata / relative, native / relative
    folder.mkdir(parents=True)
    payloads.mkdir(parents=True)
    contents = {
        "raw-0000.safetensors": b"alpha",
        "raw-0001.safetensors": b"bravo",
        "capture.pt": b"tensor",
    }
    for name, content in contents.items():
        (payloads / name).write_bytes(content)
    shards = [
        {"file": name, "sha256": digest(content), "endpoint_count": 1}
        for name, content in contents.items()
        if name.endswith(".safetensors")
    ]
    index_sha = seal(
        folder / "RAW_INDEX.json",
        {
            "shards": shards,
            "endpoints": {
                f"endpoint{i}": {"file": shard["file"]} for i, shard in enumerate(shards)
            },
        },
    )
    custody = {
        "artifact_sha256": {
            **{name: digest(content) for name, content in contents.items()},
            "RAW_INDEX.json": index_sha,
            "MODEL_FILES.json": "0" * 64,
        }
    }
    seal(folder / "CUSTODY.json", custody)
    # Unregistered files, including model-like files, must never enter an archive.
    (payloads / "weights.pt").write_bytes(b"not in custody")
    (payloads / "secret.pem").write_bytes(b"not research data")
    return metadata, native, folder, payloads, contents


def test_planning_uses_actual_files_and_splits_only_between_files(tmp_path):
    metadata, native, _, _, _ = fixture(tmp_path)
    parts = packaging.plan_parts("confirmation", "qwen", metadata, native, limit=10)
    assert [part["uncompressed_bytes"] for part in parts] == [6, 10]
    assert [part["part_index"] for part in parts] == [1, 2]
    assert parts[0]["asset_name"] == "names-confirmation-qwen-part001.tar.gz"
    assert [member["archive_path"] for member in parts[1]["members"]] == [
        "confirmation/qwen/raw-0000.safetensors",
        "confirmation/qwen/raw-0001.safetensors",
    ]
    assert not (tmp_path / "output").exists()


def test_archive_and_manifest_are_deterministic_and_allowlisted(tmp_path):
    metadata, native, _, _, contents = fixture(tmp_path)
    manifests = []
    for folder in ("first", "second"):
        manifests.append(
            packaging.package_part(
                "confirmation", "qwen", metadata, native, tmp_path / folder, 2, limit=10
            )
        )
    name = manifests[0]["asset_name"]
    archive = tmp_path / "first" / name
    assert archive.read_bytes() == (tmp_path / "second" / name).read_bytes()
    assert manifests[0] == manifests[1]
    assert manifests[0]["archive_sha256"] == digest(archive.read_bytes())
    assert json.loads(Path(str(archive) + ".manifest.json").read_text()) == manifests[0]
    with tarfile.open(archive) as packaged:
        assert packaged.getnames() == [
            "confirmation/qwen/raw-0000.safetensors",
            "confirmation/qwen/raw-0001.safetensors",
        ]
        for member in packaged.getmembers():
            assert member.isfile() and (member.uid, member.gid, member.mtime, member.mode) == (
                0,
                0,
                0,
                0o644,
            )
            assert member.uname == member.gname == ""
            assert packaged.extractfile(member).read() == contents[Path(member.name).name]


def test_development_paths_follow_the_published_stage_mapping(tmp_path):
    metadata, native, _, _, _ = fixture(tmp_path, phase="development", stage="routing")
    parts = packaging.plan_parts("development", "routing", metadata, native)
    assert all(
        member["archive_path"].startswith("development/" + STAGES["routing"] + "/")
        for member in parts[0]["members"]
    )


@pytest.mark.parametrize(
    "problem",
    [
        "missing",
        "payload_hash",
        "index_sidecar",
        "custody_binding",
        "duplicate_shard",
        "path_escape",
        "payload_symlink",
        "stage_symlink",
        "oversize",
    ],
)
def test_invalid_payloads_and_metadata_are_rejected_before_archive(tmp_path, problem):
    metadata, native, folder, payloads, _ = fixture(tmp_path)
    limit = 10
    if problem == "missing":
        (payloads / "capture.pt").unlink()
    elif problem == "payload_hash":
        (payloads / "raw-0000.safetensors").write_bytes(b"changed")
    elif problem == "index_sidecar":
        (folder / "RAW_INDEX.json.sha256").write_text("0" * 64)
    elif problem == "custody_binding":
        custody = json.loads((folder / "CUSTODY.json").read_text())
        custody["artifact_sha256"]["RAW_INDEX.json"] = "0" * 64
        seal(folder / "CUSTODY.json", custody)
    elif problem in ("duplicate_shard", "path_escape"):
        index = json.loads((folder / "RAW_INDEX.json").read_text())
        if problem == "duplicate_shard":
            index["shards"].append(index["shards"][0])
        else:
            index["shards"][0]["file"] = "../raw-0000.safetensors"
        index_sha = seal(folder / "RAW_INDEX.json", index)
        custody = json.loads((folder / "CUSTODY.json").read_text())
        custody["artifact_sha256"]["RAW_INDEX.json"] = index_sha
        seal(folder / "CUSTODY.json", custody)
    elif problem == "payload_symlink":
        (payloads / "raw-0000.safetensors").unlink()
        (payloads / "raw-0000.safetensors").symlink_to(payloads / "raw-0001.safetensors")
    elif problem == "stage_symlink":
        payloads.rename(native / "other")
        payloads.symlink_to(native / "other", target_is_directory=True)
    else:
        limit = 4
    with pytest.raises((ValueError, FileNotFoundError)):
        packaging.package_part(
            "confirmation", "qwen", metadata, native, tmp_path / "output", 1, limit=limit
        )
    assert not (tmp_path / "output").exists()


def test_existing_archive_or_manifest_is_never_overwritten(tmp_path):
    metadata, native, _, _, _ = fixture(tmp_path)
    output = tmp_path / "output"
    manifest = packaging.package_part("confirmation", "qwen", metadata, native, output, 1, limit=10)
    archive = output / manifest["asset_name"]
    original = archive.read_bytes()
    with pytest.raises(FileExistsError):
        packaging.package_part("confirmation", "qwen", metadata, native, output, 1, limit=10)
    assert archive.read_bytes() == original


def test_existing_manifest_alone_prevents_creating_an_archive(tmp_path):
    metadata, native, _, _, _ = fixture(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    receipt = output / "names-confirmation-qwen-part001.tar.gz.manifest.json"
    receipt.write_text("User-owned existing manifest")
    with pytest.raises(FileExistsError):
        packaging.package_part("confirmation", "qwen", metadata, native, output, 1)
    assert receipt.read_text() == "User-owned existing manifest"
    assert not (output / "names-confirmation-qwen-part001.tar.gz").exists()


@pytest.mark.parametrize("problem", ["duplicate_json", "metadata_symlink", "root_symlink"])
def test_metadata_cannot_hide_duplicate_keys_or_symbolic_links(tmp_path, problem):
    metadata, native, folder, _, _ = fixture(tmp_path)
    if problem == "duplicate_json":
        content = b'{"artifact_sha256": {}, "artifact_sha256": {}}'
        (folder / "CUSTODY.json").write_bytes(content)
        (folder / "CUSTODY.json.sha256").write_text(digest(content))
    elif problem == "metadata_symlink":
        original = folder / "RAW_INDEX.json"
        original.rename(folder / "other.json")
        original.symlink_to(folder / "other.json")
    else:
        link = tmp_path / "linked-metadata"
        link.symlink_to(metadata, target_is_directory=True)
        metadata = link
    with pytest.raises(ValueError):
        packaging.plan_parts("confirmation", "qwen", metadata, native)


def test_compressed_public_asset_limit_is_checked_before_success_manifest(tmp_path, monkeypatch):
    metadata, native, _, _, _ = fixture(tmp_path)
    monkeypatch.setattr(packaging, "MAX_ARCHIVE_BYTES", 1)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="compressed"):
        packaging.package_part("confirmation", "qwen", metadata, native, output, 1)
    assert (output / "names-confirmation-qwen-part001.tar.gz").is_file()
    assert not (output / "names-confirmation-qwen-part001.tar.gz.manifest.json").exists()


def test_payload_change_during_packaging_retains_failed_archive(tmp_path, monkeypatch):
    metadata, native, _, payloads, _ = fixture(tmp_path)
    original_plan = packaging.plan_parts

    def changed_after_plan(*args, **kwargs):
        parts = original_plan(*args, **kwargs)
        (payloads / "capture.pt").write_bytes(b"mutate")
        return parts

    monkeypatch.setattr(packaging, "plan_parts", changed_after_plan)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="changed"):
        packaging.package_part("confirmation", "qwen", metadata, native, output, 1, limit=10)
    assert (output / "names-confirmation-qwen-part001.tar.gz").is_file()
    assert not (output / "names-confirmation-qwen-part001.tar.gz.manifest.json").exists()


@pytest.mark.parametrize("part_index", [0, -1, 3])
def test_part_index_is_one_based_and_cannot_invent_missing_parts(tmp_path, part_index):
    metadata, native, _, _, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match="part"):
        packaging.package_part(
            "confirmation", "qwen", metadata, native, tmp_path / "output", part_index, limit=10
        )
