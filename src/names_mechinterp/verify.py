"""Offline release-file hashes and optional native-file integrity, not model rescore."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

STAGES = {
    "behavior_qwen": "cloud_reference_error_metadata_20261115/qwen",
    "behavior_phi": "cloud_reference_error_metadata_20261115/phi",
    "prefix_screen": "cloud_reference_causal_metadata_20261115/screen",
    "routing": "cloud_reference_routing_metadata_20261115/routing",
    "validation": "cloud_reference_validation_metadata_20261115/validation",
    "decomposition": "cloud_reference_decomposition_metadata_20261115/decomposition",
}


def digest(path, *, decompress=False):
    result = hashlib.sha256()
    opener = gzip.open if decompress else open
    with opener(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def verify_manifest(repo_root, *, verify_compressed=False):
    root = Path(repo_root).resolve()
    manifest = json.loads((root / "data" / "MANIFEST.json").read_text())
    seen = set()
    total = 0
    for record in manifest["files"]:
        path = root / record["path"]
        if path.resolve().is_relative_to(root) is False or record["path"] in seen:
            raise ValueError("manifest path escaped the release or was duplicated")
        seen.add(record["path"])
        if not path.is_file() or path.stat().st_size != record["size_bytes"] or digest(path) != record["sha256"]:
            raise ValueError(f"release artifact hash mismatch: {record['path']}")
        if verify_compressed and "uncompressed_sha256" in record and digest(path, decompress=True) != record["uncompressed_sha256"]:
            raise ValueError(f"compressed original differs from its original checksum: {record['path']}")
        total += record["size_bytes"]
    return {"verified": True, "file_count": len(seen), "total_bytes": total, "compressed_originals_checked": verify_compressed, "manifest_sha256": digest(root / "data" / "MANIFEST.json")}


def native_availability(metadata_root, *, native_root=None, verify_sha256=False):
    metadata = Path(metadata_root)
    supplied = Path(native_root) if native_root else metadata
    stages, total_expected, total_present, missing_tensors, tensors = {}, 0, 0, [], []
    for stage, relative in STAGES.items():
        folder = metadata / relative
        index = json.loads((folder / "RAW_INDEX.json").read_text())
        shards = index["shards"]
        names = {s["file"] for s in shards}
        if len(names) != len(shards) or any(Path(name).name != name for name in names):
            raise ValueError("native shard names are duplicated or escape their stage")
        endpoint_counts = {name: 0 for name in names}
        for value in index["endpoints"].values():
            if value["file"] not in names:
                raise ValueError("native endpoint references an undeclared shard")
            endpoint_counts[value["file"]] += 1
        present, missing = [], []
        for shard in shards:
            if endpoint_counts[shard["file"]] != shard["endpoint_count"]:
                raise ValueError("native shard endpoint count disagrees with its index")
            path = supplied / relative / shard["file"]
            record = {"path": f"{relative}/{shard['file']}", "sha256": shard["sha256"], "endpoint_count": shard["endpoint_count"]}
            if path.is_file():
                if verify_sha256 and digest(path) != shard["sha256"]:
                    raise ValueError(f"native shard digest mismatch: {record['path']}")
                present.append(record)
            else:
                missing.append(record)
        for path in sorted(folder.glob("*.pt")):
            tensors.append(path.relative_to(metadata).as_posix())
        custody = json.loads((folder / "CUSTODY.json").read_text())
        for name, sha in custody["artifact_sha256"].items():
            if name.endswith(".pt") and not (folder / name).is_file():
                missing_tensors.append({"stage": stage, "path": f"{relative}/{name}", "sha256": sha})
        stages[stage] = {"native_endpoint_count": len(index["endpoints"]), "expected_shards": len(shards), "present_shards": len(present), "present": present, "missing": missing}
        total_expected += len(shards)
        total_present += len(present)
    return {"schema": "names-mechinterp-artifact-availability-v1", "native_shards_complete": total_expected == total_present, "expected_native_shards": total_expected, "present_native_shards": total_present, "native_file_sha256_checked": verify_sha256,
            "stages": stages, "included_tensor_artifacts": tensors, "missing_non_native_tensor_artifacts": missing_tensors,
            "native_tensor_rescore_performed": False, "new_model_forward_count": 0,
            "scope": "File availability and optional SHA-256 integrity only, not newly repeated native argmax, tokenizer decoding, mathematical tensor rescore or whole-answer evaluation. Native roots mirror metadata-relative stage paths."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--native-root", type=Path)
    parser.add_argument("--verify-native", action="store_true")
    parser.add_argument("--require-native", action="store_true")
    parser.add_argument("--verify-compressed", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {"release": verify_manifest(args.repo_root, verify_compressed=args.verify_compressed), "native": native_availability(args.repo_root / "data" / "original", native_root=args.native_root, verify_sha256=args.verify_native)}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"release_verified": True, "native_shards_complete": result["native"]["native_shards_complete"], "present_native_shards": result["native"]["present_native_shards"], "expected_native_shards": result["native"]["expected_native_shards"]}, sort_keys=True))
    return 1 if args.require_native and not result["native"]["native_shards_complete"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
