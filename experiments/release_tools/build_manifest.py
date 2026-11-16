"""Author-side release manifest builder; never reads credentials or cloud state.

The historical workspace is used only for byte provenance. Its absolute path
is not written into public records. This program generates manifests; it does
not alter scientific originals or rewrite historical custody assertions.
"""

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from names_mechinterp.verify import digest, native_availability

PORTS = {
    "_literal_audit.py": "error_reference_audit_20261115/audit_behavior.py",
    "_protocol.py": "error_reference_study_20261114/error_protocol.py",
    "_behavior.py": "error_reference_study_20261114/error_analysis.py",
    "_decomposition.py": "error_reference_decomposition_20261115/decomposition_grade.py",
    "_validation.py": "error_reference_validation_20261115/validation_plan.py",
    "_bootstrap.py": "paper_draft_20261115/descriptive_world_bootstrap.py",
}

# These are installation/build products, not scientific inputs. In particular,
# editable installs rewrite egg-info/SOURCES.txt whenever the file tree changes.
GENERATED_DIRECTORIES = {
    "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".cache",
    ".tox", ".nox", ".hypothesis", ".ipynb_checkpoints", "venv", ".eggs",
    "build", "dist", "htmlcov", "node_modules", "model-cache", "models",
    "runs", "outputs", "results", "artifacts",
}


def release_file(path, root):
    """Reject generated/ignored products even when an install created them."""
    relative = path.relative_to(root)
    return (path.is_file() and relative.as_posix() != "data/MANIFEST.json"
            and not any(part in GENERATED_DIRECTORIES or part.startswith(".venv")
                        or part.endswith((".egg-info", ".dist-info")) for part in relative.parts)
            and path.name not in {".DS_Store", ".coverage", ".env"}
            and not path.name.startswith(".env.")
            and not path.name.endswith((".pyc", ".pyo", ".pyd", ".pem", ".key")))


def build(repo_root, source_workspace):
    root, source = Path(repo_root).resolve(), Path(source_workspace).resolve()
    confirmation_prefix = "data/original/cloud_reference_confirmation_metadata_20261116/"
    confirmation_root = root / confirmation_prefix
    import_receipt_path = confirmation_root / "IMPORT_RECEIPT.json"
    import_receipt = json.loads(import_receipt_path.read_text()) if import_receipt_path.is_file() else None
    availability = native_availability(root / "data" / "original")
    (root / "data" / "AVAILABILITY.json").write_text(json.dumps(availability, sort_keys=True, indent=2) + "\n")
    original_zip = root / "data" / "original" / "supporting_data_archive" / "S1_Data.zip"
    archive_sha = digest(original_zip)
    records = []
    with zipfile.ZipFile(original_zip) as archive:
        for directory in ("data", "experiments", "src", "tests"):
            for path in sorted((root / directory).rglob("*")):
                if not release_file(path, root):
                    continue
                relative = path.relative_to(root).as_posix()
                record = {"path": relative, "size_bytes": path.stat().st_size, "sha256": digest(path)}
                origin = None
                if relative.startswith(confirmation_prefix):
                    if import_receipt is None:
                        raise ValueError("confirmation metadata lack their import receipt")
                    record["confirmation_import_receipt_sha256"] = digest(import_receipt_path)
                    short = relative.removeprefix(confirmation_prefix)
                    if short == "IMPORT_RECEIPT.json":
                        record["adaptation"] = "Release-side checksum record of byte-identical metadata copies and deterministic token-JSON compression"
                    else:
                        imported = import_receipt["files"][short]
                        if record["sha256"] != imported["sha256"]:
                            raise ValueError("confirmation metadata changed after verified import")
                        record.update(original_sha256=imported["original_sha256"], original_relative_path=imported["original_relative_path"], adaptation=imported["representation"])
                        if short.endswith(".json.gz"):
                            if digest(path, decompress=True) != imported["original_sha256"]:
                                raise ValueError("compressed confirmation metadata differ from original bytes")
                            record.update(uncompressed_sha256=imported["original_sha256"], uncompressed_size_bytes=imported["original_size_bytes"])
                        else:
                            record["byte_identical"] = record["sha256"] == imported["original_sha256"]
                elif relative.startswith("data/processed/confirmation/"):
                    member = relative.removeprefix("data/processed/confirmation/")
                    if member == "CPU_REPRODUCTION.json":
                        report = json.loads(path.read_text())
                        if report["native_logit_replay_performed"] is not False or report["new_model_forward_count"] != 0:
                            raise ValueError("processed confirmation report overstates its replay scope")
                        record.update(adaptation="Standard-library recomputation of complete confirmation text outcomes and frozen statistics", analysis_source_sha256=report["analysis_source_sha256"])
                    else:
                        original_path = confirmation_root / member
                        if not original_path.is_file() or digest(original_path) != record["sha256"]:
                            raise ValueError("processed confirmation input differs from imported metadata")
                        record.update(original_sha256=record["sha256"], byte_identical=True, source_metadata_path=original_path.relative_to(root).as_posix(), confirmation_import_receipt_sha256=digest(import_receipt_path))
                elif relative.startswith("experiments/"):
                    short = relative.removeprefix("experiments/")
                    origin = source / ("paper_draft_20261115/" + path.name if short.startswith("paper_analysis/") else short)
                elif relative.startswith("data/original/"):
                    short = relative.removeprefix("data/original/")
                    if short.startswith("supporting_data_archive/"):
                        if path.name == "S1_Data.zip":
                            origin = source / "paper_draft_20261115/submission_package/S1_Data.zip"
                    else:
                        origin = source / (short[:-3] if short.endswith(".json.gz") else short)
                elif relative.startswith("src/names_mechinterp/") and path.name in PORTS:
                    origin = source / PORTS[path.name]
                    record["adaptation"] = "one import changed to a package-relative protocol import" if path.name == "_behavior.py" else "byte-identical original module, renamed for a self-contained CPU package"
                elif relative.startswith("data/processed/"):
                    member = relative.removeprefix("data/processed/")
                    original = archive.read(member)
                    record.update(source_archive_sha256=archive_sha, source_archive_member=member, original_sha256=hashlib.sha256(original).hexdigest(), byte_identical=original == path.read_bytes())
                    if member == "package_scope.json":
                        before, after = json.loads(original), json.loads(path.read_text())
                        changed = {key for key in before if before[key] != after.get(key)}
                        expected_changes = {"scope", "raw_native_deposit", "license", "reserved_confirmation_outcomes_included"} if import_receipt is not None else {"scope", "raw_native_deposit", "license"}
                        added = set(after) - set(before)
                        expected_added = {"development_worlds", "confirmation_worlds", "confirmation_questions_per_model", "confirmation_directory", "confirmation_publication_status"} if import_receipt is not None else set()
                        if changed != expected_changes or added != expected_added:
                            raise ValueError("processed scope has an undeclared scientific-data change")
                        if import_receipt is not None and (after["worlds"] != 32 or after["development_worlds"] != 32 or after["confirmation_worlds"] != 64 or after["confirmation_questions_per_model"] != 6144 or after["confirmation_directory"] != "confirmation/" or after["reserved_confirmation_outcomes_included"] is not True):
                            raise ValueError("development and confirmation sample scopes were mixed")
                        record["adaptation"] = "Release scope, native-data limitations and license metadata; completed confirmation is separately indexed without changing development observations"
                    elif not record["byte_identical"]:
                        raise ValueError(f"processed supporting data changed: {member}")
                if origin is not None and origin.is_file():
                    record["historical_relative_path"] = origin.relative_to(source).as_posix()
                    record["original_sha256"] = digest(origin)
                    if path.name.endswith(".json.gz"):
                        uncompressed = digest(path, decompress=True)
                        if uncompressed != record["original_sha256"]:
                            raise ValueError("compressed original does not preserve exact bytes")
                        record.update(uncompressed_sha256=uncompressed, uncompressed_size_bytes=origin.stat().st_size, adaptation="deterministic gzip -n representation of the exact original JSON")
                    else:
                        record["byte_identical"] = record["sha256"] == record["original_sha256"]
                        if not record["byte_identical"] and path.name != "_behavior.py":
                            raise ValueError(f"unexplained historical source/data modification: {relative}")
                records.append(record)
    result = {"schema": "names-mechinterp-code-data-manifest-v1", "supporting_archive_sha256": archive_sha, "files": records,
              "licenses": {"code": "MIT", "data": "CC BY 4.0"}, "scientific_originals_mutated": False,
              "historical_paths": "Original provenance fields may retain author or cloud absolute paths; they are preserved records, not required portable runtime paths.",
              "excluded_categories": ["credentials and private keys", "cloud access guides", "operational spending ledgers", "machine/project/runtime archives", "model weights", "local virtual environments", "unrelated earlier experiment results"],
              "native_data_scope": "Large native-score and measurement files are separate public release assets, not Git-checkout payloads. NATIVE_RELEASE.json and the two native_archive_catalogs files describe complete development and confirmation coverage. AVAILABILITY.json reports checkout inventory; CONFIRMATION_AVAILABILITY.json distinguishes that inventory from the separate public deposit. SHA integrity and mathematical replay remain distinct."}
    (root / "data" / "MANIFEST.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--source-workspace", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.repo_root, args.source_workspace)
    print(json.dumps({"files": len(result["files"]), "scientific_originals_mutated": False}))
