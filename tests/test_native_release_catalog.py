"""Released archive inventories must cover the original scientific custody exactly."""

import hashlib
import json
from pathlib import Path, PurePosixPath

from names_mechinterp.verify import STAGES

ROOT = Path(__file__).resolve().parents[1]


def test_public_native_release_covers_every_declared_scientific_payload():
    publication = json.loads((ROOT / "data/NATIVE_RELEASE.json").read_text())
    assert publication["public_release_verified"] is True
    assert publication["native_mathematical_replay_performed_during_publication"] is False
    assert publication["new_model_forward_count_during_publication"] == 0
    expected, observed, archive_records = {}, {}, []
    maps = {"development": STAGES, "confirmation": {name: name for name in ("qwen", "phi", "repair")}}
    for phase, stages in maps.items():
        metadata = ROOT / "data/original"
        if phase == "confirmation":
            metadata /= "cloud_reference_confirmation_metadata_20261116"
        for relative in stages.values():
            custody = json.loads((metadata / relative / "CUSTODY.json").read_text())["artifact_sha256"]
            for name, sha in custody.items():
                if name.endswith((".safetensors", ".pt")):
                    expected[f"{phase}/{relative}/{name}"] = sha
        catalog_path = ROOT / "data/native_archive_catalogs" / f"names-{phase}-archive-catalog.json"
        catalog_bytes = catalog_path.read_bytes()
        assert hashlib.sha256(catalog_bytes).hexdigest() == publication["catalog_sha256"][phase]
        for archive in json.loads(catalog_bytes)["archives"]:
            name = archive["name"] if phase == "development" else archive["asset_name"]
            sha = archive["sha256"] if phase == "development" else archive["archive_sha256"]
            size = archive["bytes"] if phase == "development" else archive["archive_size_bytes"]
            assert Path(name).name == name and 0 < size < 2 * 1024**3
            archive_records.append({"name": name, "sha256": sha, "size_bytes": size})
            for record in archive["members"]:
                member = record["member"] if phase == "development" else record["archive_path"]
                path = PurePosixPath(member)
                assert not path.is_absolute() and ".." not in path.parts and member.startswith(phase + "/")
                assert member not in observed
                observed[member] = record
    payloads = {name: value for name, value in observed.items() if not name.endswith(".sha256")}
    sidecars = {name: value for name, value in observed.items() if name.endswith(".sha256")}
    assert {name: record["sha256"] for name, record in payloads.items()} == expected
    assert len(payloads) == 1069
    assert sum(name.endswith(".safetensors") for name in payloads) == 1030
    assert sum(name.endswith(".pt") for name in payloads) == 39
    assert len(sidecars) == 419
    for name, record in sidecars.items():
        source = name.removesuffix(".sha256")
        assert source.startswith("development/") and source in payloads
        assert record["size"] == 65
        assert record["sha256"] == hashlib.sha256((expected[source] + "\n").encode()).hexdigest()
    size = sum(record.get("size", record.get("size_bytes")) for record in payloads.values())
    assert size == publication["coverage"]["scientific_payload_bytes"] == 16814720490
    assert sorted(archive_records, key=lambda value: value["name"]) == publication["archives"]
    assert len(archive_records) == 11
    assert sum(value["size_bytes"] for value in archive_records) == 15164230779
