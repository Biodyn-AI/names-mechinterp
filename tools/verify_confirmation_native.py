"""Check downloaded confirmation payload bytes; never load or replay tensors.

Expected layout: NATIVE_ROOT/{qwen,phi,repair}/raw-*.safetensors and
NATIVE_ROOT/repair/case-*.pt. Use metadata from the matching repository snapshot.
Missing files produce an incomplete report and nonzero exit unless explicitly
allowed. Neither a matching hash nor this receipt is mathematical native replay.
"""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

STAGES = ("qwen", "phi", "repair")
DEFAULT_METADATA = (
    Path(__file__).resolve().parents[1]
    / "data/original/cloud_reference_confirmation_metadata_20261116"
)
NATIVE_NAME = re.compile(r"raw-[0-9]{4}\.safetensors")
CAPTURE_NAME = re.compile(r"case-[0-9]{3}\.pt")


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def safe_path(root, stage, name):
    root = Path(root).resolve()
    path = root / stage / name
    if (
        stage not in STAGES
        or Path(name).name != name
        or path.is_symlink()
        or not path.resolve().is_relative_to(root)
    ):
        raise ValueError("Artifact path escapes the specified root or uses a symlink")
    return path


def sidecar(path, expected):
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("Invalid declared SHA-256")
    adjacent = path.with_name(path.name + ".sha256")
    if adjacent.exists() and (
        adjacent.is_symlink() or adjacent.read_text().strip() != expected
    ):
        raise ValueError(f"Checksum sidecar disagrees: {path.name}")


def requirements(metadata_root):
    records, fingerprints = [], {}
    for stage in STAGES:
        custody_path = safe_path(metadata_root, stage, "CUSTODY.json")
        custody_sha = digest(custody_path)
        sidecar(custody_path, custody_sha)
        custody = json.loads(custody_path.read_text())["artifact_sha256"]
        index_path = safe_path(metadata_root, stage, "RAW_INDEX.json")
        index_sha = digest(index_path)
        sidecar(index_path, index_sha)
        if custody.get("RAW_INDEX.json") != index_sha:
            raise ValueError("Native index differs from its custody record")
        fingerprints.update(
            {f"{stage}/CUSTODY.json": custody_sha, f"{stage}/RAW_INDEX.json": index_sha}
        )
        index = json.loads(index_path.read_text())
        shards = {item["file"]: item for item in index["shards"]}
        if len(shards) != len(index["shards"]) or any(
            not NATIVE_NAME.fullmatch(name) for name in shards
        ):
            raise ValueError(
                "Native shard names are duplicated or outside the whitelist"
            )
        counts, tensor_bytes = Counter(), Counter()
        for identity, endpoint in index["endpoints"].items():
            shape = endpoint["shape"]
            if (
                endpoint["file"] not in shards
                or endpoint["key"] != identity
                or endpoint["dtype"] != "float32"
                or len(shape) != 1
                or type(shape[0]) is not int
                or shape[0] <= 0
            ):
                raise ValueError("Native index endpoint geometry or membership changed")
            counts[endpoint["file"]] += 1
            tensor_bytes[endpoint["file"]] += shape[0] * 4
        if {name for name in custody if NATIVE_NAME.fullmatch(name)} != set(shards):
            raise ValueError("Custody and native index file coverage disagree")
        for name, item in sorted(shards.items()):
            if (
                counts[name] != item["endpoint_count"]
                or custody.get(name) != item["sha256"]
            ):
                raise ValueError("Native index count or hash differs from custody")
            sidecar(safe_path(metadata_root, stage, name), item["sha256"])
            records.append(
                {
                    "stage": stage,
                    "name": name,
                    "kind": "native_shard",
                    "sha256": item["sha256"],
                    "shape_data_bytes": tensor_bytes[name],
                }
            )
        for name, sha in sorted(custody.items()):
            if name.endswith(".pt"):
                if stage != "repair" or not CAPTURE_NAME.fullmatch(name):
                    raise ValueError("Capture path outside the fixed repair whitelist")
                sidecar(safe_path(metadata_root, stage, name), sha)
                records.append(
                    {
                        "stage": stage,
                        "name": name,
                        "kind": "case_capture",
                        "sha256": sha,
                        "shape_data_bytes": None,
                    }
                )
    return records, fingerprints


def verify(metadata_root, native_root):
    expected, fingerprints = requirements(metadata_root)
    present, missing = [], []
    for item in expected:
        relative = f"{item['stage']}/{item['name']}"
        path = safe_path(native_root, item["stage"], item["name"])
        if not path.exists():
            missing.append(relative)
            continue
        if not path.is_file() or digest(path) != item["sha256"]:
            raise ValueError(f"Downloaded payload hash mismatch: {relative}")
        sidecar(path, item["sha256"])
        if (
            item["shape_data_bytes"] is not None
            and path.stat().st_size < item["shape_data_bytes"] + 8
        ):
            raise ValueError("Native file is shorter than its declared tensor data")
        present.append({**item, "path": relative, "size_bytes": path.stat().st_size})
    return {
        "schema": "names-mechinterp-confirmation-native-integrity-v1",
        "complete": not missing,
        "present_files_verified": bool(present),
        "expected_native_shards": sum(i["kind"] == "native_shard" for i in expected),
        "present_native_shards": sum(i["kind"] == "native_shard" for i in present),
        "expected_capture_files": sum(i["kind"] == "case_capture" for i in expected),
        "present_capture_files": sum(i["kind"] == "case_capture" for i in present),
        "metadata_sha256": fingerprints,
        "files": present,
        "missing": missing,
        "native_mathematical_replay_performed": False,
        "new_model_forward_count": 0,
        "scope": "Complete-file SHA-256 and index/custody consistency only. Shape-derived data sizes are lower bounds, not encoded-file sizes. This does not check native argmax, token decoding, head arithmetic, model execution or hidden-state semantics.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    report = verify(args.metadata_root, args.native_root)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "complete",
                    "expected_native_shards",
                    "present_native_shards",
                    "expected_capture_files",
                    "present_capture_files",
                    "native_mathematical_replay_performed",
                )
            },
            sort_keys=True,
        )
    )
    return 0 if report["complete"] or args.allow_missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
