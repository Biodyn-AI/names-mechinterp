"""Package one checksum-verified native-data asset; never upload or delete files.

Only RAW_INDEX-listed score shards and custody-listed .pt measurements enter an
archive. The public metadata supplies their original checksum sidecars. Failed
outputs are retained without a success manifest and must not be published.
"""

import argparse
import gzip
import hashlib
import json
import re
import sys
import tarfile
from collections import Counter
from pathlib import Path

# Standalone runs use this checkout's canonical development-stage mapping.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from names_mechinterp.verify import STAGES, digest  # noqa: E402

MAX_BYTES = 1_800_000_000
MAX_ARCHIVE_BYTES = 2 * 1024**3
CONFIRMATION = {stage: stage for stage in ("qwen", "phi", "repair")}


def _file(root, relative):
    current = root
    for part in Path(relative).parts:
        if part in (".", ".."):
            raise ValueError("File path escaped its declared root")
        current = current / part
        if current.is_symlink():
            raise ValueError("Symlinks are not release payloads")
    if not current.is_file() or not current.resolve().is_relative_to(root):
        raise FileNotFoundError(f"Missing regular release file: {relative}")
    return current


def _unique(pairs):
    values = dict(pairs)
    if len(values) != len(pairs):
        raise ValueError("Duplicate JSON keys in native metadata")
    return values


def _metadata(root, relative):
    path = _file(root, relative)
    actual = digest(path)
    if _file(root, relative + ".sha256").read_text().strip() != actual:
        raise ValueError(f"Metadata checksum mismatch: {relative}")
    return json.loads(path.read_text(), object_pairs_hook=_unique), actual


def plan_parts(phase, stage, metadata_root, native_root, *, limit=MAX_BYTES):
    """Plan from a complete selected stage, using observed sizes and exact hashes."""
    if (
        phase not in ("confirmation", "development")
        or type(limit) is not int
        or not 0 < limit <= MAX_BYTES
    ):
        raise ValueError("Unknown phase or invalid part byte limit")
    mapping = CONFIRMATION if phase == "confirmation" else STAGES
    if stage not in mapping:
        raise ValueError("Unknown native stage")
    if Path(metadata_root).is_symlink() or Path(native_root).is_symlink():
        raise ValueError("Symlink roots are not permitted")
    metadata, native, relative = (
        Path(metadata_root).resolve(strict=True),
        Path(native_root).resolve(strict=True),
        mapping[stage],
    )
    index, index_sha = _metadata(metadata, relative + "/RAW_INDEX.json")
    custody, custody_sha = _metadata(metadata, relative + "/CUSTODY.json")
    artifacts = custody["artifact_sha256"]
    if any(
        not isinstance(name, str) or Path(name).name != name or "\\" in name for name in artifacts
    ):
        raise ValueError("Custody artifact path escaped its stage")
    if artifacts.get("RAW_INDEX.json") != index_sha:
        raise ValueError("Raw index does not match its custody binding")
    allowed = {}
    for shard in index["shards"]:
        name = shard["file"]
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"raw-\d+\.safetensors", name)
            or name in allowed
        ):
            raise ValueError("Native shard path escaped its stage or was duplicated")
        if artifacts.get(name) != shard["sha256"]:
            raise ValueError("Raw shard checksum differs from custody")
        allowed[name] = shard["sha256"]
    counts = Counter(endpoint["file"] for endpoint in index["endpoints"].values())
    if set(counts) - set(allowed) or any(
        counts[shard["file"]] != shard["endpoint_count"] for shard in index["shards"]
    ):
        raise ValueError("Raw index endpoint coverage is inconsistent")
    allowed.update({name: sha for name, sha in artifacts.items() if name.endswith(".pt")})
    parts, members, total = [], [], 0
    for name, expected in sorted(allowed.items()):
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("Invalid native file checksum")
        path = _file(native, relative + "/" + name)
        size = path.stat().st_size
        if size > limit:
            raise ValueError(f"A whole native file exceeds the part byte limit: {name}")
        if digest(path) != expected:
            raise ValueError(f"Native payload checksum mismatch: {name}")
        if members and total + size > limit:
            parts.append((members, total))
            members, total = [], 0
        members.append(
            {
                "archive_path": f"{phase}/{relative}/{name}",
                "source_relative_path": f"{relative}/{name}",
                "size_bytes": size,
                "sha256": expected,
            }
        )
        total += size
    if members:
        parts.append((members, total))
    return [
        {
            "phase": phase,
            "stage": stage,
            "part_index": number,
            "part_count": len(parts),
            "asset_name": f"names-{phase}-{stage}-part{number:03d}.tar.gz",
            "uncompressed_bytes": size,
            "part_limit_bytes": limit,
            "metadata_sha256": {"RAW_INDEX.json": index_sha, "CUSTODY.json": custody_sha},
            "members": group,
        }
        for number, (group, size) in enumerate(parts, 1)
    ]


class _CheckedReader:
    """Hash the exact bytes copied into tar, detecting changes since planning."""

    def __init__(self, stream):
        self.stream, self.sha256, self.size = stream, hashlib.sha256(), 0

    def read(self, size):
        content = self.stream.read(size)
        self.sha256.update(content)
        self.size += len(content)
        return content


def package_part(
    phase, stage, metadata_root, native_root, output_dir, part_index, *, limit=MAX_BYTES
):
    parts = plan_parts(phase, stage, metadata_root, native_root, limit=limit)
    if type(part_index) is not int or not 1 <= part_index <= len(parts):
        raise ValueError("Part index must identify one complete planned part, starting at one")
    part, output = parts[part_index - 1], Path(output_dir)
    if output.is_symlink():
        raise ValueError("Symlink output directories are not permitted")
    archive = output / part["asset_name"]
    manifest_path = archive.with_name(archive.name + ".manifest.json")
    if (
        archive.exists()
        or archive.is_symlink()
        or manifest_path.exists()
        or manifest_path.is_symlink()
    ):
        raise FileExistsError("Archive or manifest already exists; refusing to overwrite")
    output.mkdir(parents=True, exist_ok=True)
    with (
        archive.open("xb") as outgoing,
        gzip.GzipFile(
            filename="", mode="wb", fileobj=outgoing, mtime=0, compresslevel=6
        ) as compressed,
        tarfile.open(fileobj=compressed, mode="w|", format=tarfile.USTAR_FORMAT) as packaged,
    ):
        for member in part["members"]:
            source = _file(Path(native_root).resolve(strict=True), member["source_relative_path"])
            header = tarfile.TarInfo(member["archive_path"])
            header.size, header.mode = member["size_bytes"], 0o644
            header.uid = header.gid = header.mtime = 0
            header.uname = header.gname = ""
            with source.open("rb") as incoming:
                checked = _CheckedReader(incoming)
                packaged.addfile(header, checked)
                if (
                    checked.size != member["size_bytes"]
                    or checked.sha256.hexdigest() != member["sha256"]
                    or incoming.read(1)
                ):
                    raise ValueError(
                        "Native payload changed during packaging; failed archive retained"
                    )
    archive_size = archive.stat().st_size
    if archive_size >= MAX_ARCHIVE_BYTES:
        raise ValueError("The compressed asset exceeds the public file limit; archive retained")
    manifest = {**part, "archive_sha256": digest(archive), "archive_size_bytes": archive_size}
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("confirmation", "development"), required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--part-index", type=int)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.list:
        receipt = plan_parts(args.phase, args.stage, args.metadata_root, args.native_root)
    else:
        if args.part_index is None or args.output_dir is None:
            parser.error("Packaging requires --part-index and --output-dir")
        receipt = package_part(
            args.phase,
            args.stage,
            args.metadata_root,
            args.native_root,
            args.output_dir,
            args.part_index,
        )
    print(json.dumps(receipt, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
