"""Import verified confirmation metadata without native tensors or credentials.

Large token JSON files use deterministic gzip; their historical SHA-256 sidecars
continue to describe the decompressed bytes. Existing targets are never reused.
"""

import argparse
import gzip
import hashlib
import json
import re
import shutil
from pathlib import Path

FORBIDDEN = re.compile(
    rb"/Users/|/home/ihor|-----BEGIN (?:OPENSSH|RSA|EC) PRIVATE KEY|(?:rpa_|rps_)[A-Za-z0-9]{15}"
)


def checked_digest(path):
    digest, overlap = hashlib.sha256(), b""
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            if FORBIDDEN.search(overlap + chunk):
                raise ValueError(
                    f"Personal path or credential marker in metadata: {path.name}"
                )
            digest.update(chunk)
            overlap = chunk[-80:]
    return digest.hexdigest()


def import_metadata(source, target):
    source, target = Path(source).resolve(), Path(target)
    if target.exists():
        raise FileExistsError(f"Refusing to reuse metadata namespace: {target}")
    files, checked = sorted(p for p in source.rglob("*") if p.is_file()), {}
    if not files:
        raise ValueError("Metadata source is empty")
    for path in files:
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(source)
            or path.suffix not in (".json", ".py", ".sha256")
        ):
            raise ValueError(f"Not a regular allowed metadata file: {path.name}")
        actual = checked_digest(path)
        if path.suffix != ".sha256":
            sidecar = path.with_name(path.name + ".sha256")
            if not sidecar.is_file() or sidecar.read_text().strip() != actual:
                raise ValueError(f"Unverified metadata bytes: {path.name}")
        checked[path] = actual
    target.mkdir(parents=True)
    receipt = {
        "schema": "names-mechinterp-confirmation-metadata-import-v1",
        "source_description": "Completed, checksum-verified cloud confirmation metadata; no native payloads imported",
        "files": {},
    }
    for path, original_sha256 in checked.items():
        relative = path.relative_to(source)
        compressed = relative.as_posix() in ("TOKENS-qwen.json", "TOKENS-phi.json")
        destination = target / (str(relative) + ".gz" if compressed else relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if compressed:
            with (
                path.open("rb") as incoming,
                destination.open("xb") as outgoing,
                gzip.GzipFile(
                    filename="", mode="wb", fileobj=outgoing, mtime=0, compresslevel=9
                ) as archive,
            ):
                shutil.copyfileobj(incoming, archive)
        else:
            with path.open("rb") as incoming, destination.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing)
        copied_sha256 = checked_digest(destination)
        if compressed:
            restored = hashlib.sha256()
            with gzip.open(destination, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    restored.update(chunk)
            identical = restored.hexdigest() == original_sha256
        else:
            identical = copied_sha256 == original_sha256
        if not identical:
            raise ValueError(
                "Metadata bytes changed during import; incomplete target retained"
            )
        receipt["files"][destination.relative_to(target).as_posix()] = {
            "original_relative_path": relative.as_posix(),
            "original_sha256": original_sha256,
            "sha256": copied_sha256,
            "original_size_bytes": path.stat().st_size,
            "size_bytes": destination.stat().st_size,
            "representation": "deterministic gzip of exact original bytes"
            if compressed
            else "byte-identical metadata copy",
        }
    with (target / "IMPORT_RECEIPT.json").open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    receipt = import_metadata(args.source, args.target)
    print(
        json.dumps(
            {
                "imported_metadata_files": len(receipt["files"]),
                "native_payloads_imported": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
