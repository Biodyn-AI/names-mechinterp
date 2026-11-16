"""CPU-only numerical audit receipt; scientific negatives are valid outcomes.

This wrapper never provisions resources or calls a research model. It imports
only the authenticated frozen runner and invokes its CPU rescore entry point.
Receipts must live outside both the sealed source and scientific run directory.
"""

import argparse
import hashlib
import importlib
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

STAGE_FILES = (
    "RESULT.json",
    "RAW_INDEX.json",
    "NATURAL.pt",
    "KERNELS.pt",
    "TRACES.pt",
    "GENERATIONS.json",
    "COUNTS.json",
)
ANCHOR_KEYS = {
    "REGISTRATION.json": "registration_sha256",
    "SOURCES.json": "source_manifest_sha256",
    "INPUTS.json": "inputs_sha256",
    "PLAN.json": "plan_sha256",
}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def verified(path, expected=None):
    path = Path(path)
    actual = digest(path)
    if (
        actual != Path(str(path) + ".sha256").read_text().strip()
        or expected is not None
        and actual != expected
    ):
        raise ValueError(f"Hash mismatch: {path.name}")
    return actual


def snapshot(run, source, expected_registration=None, expected_sources=None):
    anchors = {
        key: verified(
            run / name,
            expected_registration
            if name == "REGISTRATION.json"
            else expected_sources
            if name == "SOURCES.json"
            else None,
        )
        for name, key in ANCHOR_KEYS.items()
    }
    registration = json.loads((run / "REGISTRATION.json").read_text())
    if set(registration["input_sha256"]) != {
        "SOURCES.json",
        "INPUTS.json",
        "PLAN.json",
    }:
        raise ValueError("Registered input closure changed")
    for name, expected in registration["input_sha256"].items():
        verified(run / name, expected)
    closure = json.loads((run / "SOURCES.json").read_text())["new"]
    if len(closure) != 7 or set(closure) != {p.name for p in source.glob("*.py")}:
        raise ValueError("Exactly the seven frozen runtime Python sources are required")
    for name, expected in closure.items():
        if Path(name).name != name or not name.endswith(".py"):
            raise ValueError("Source name escaped its declared root")
        if (
            digest(source / name) != expected
            or digest(run / "sources" / name) != expected
        ):
            raise ValueError("Current runtime source differs from its registered copy")
    stage = run / "decomposition"
    anchors["custody_sha256"] = verified(stage / "CUSTODY.json")
    custody = json.loads((stage / "CUSTODY.json").read_text())
    if custody["registration_sha256"] != anchors["registration_sha256"]:
        raise ValueError("Custody is bound to another registration")
    for name in STAGE_FILES:
        anchors[name.split(".")[0].lower() + "_sha256"] = verified(
            stage / name, custody["artifact_sha256"][name]
        )
    return {"anchors": anchors, "runtime_source_sha256": closure}


def _load_runner(source):
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source))
    runner = importlib.import_module("decomposition_run")
    if Path(runner.__file__).resolve().parent != source:
        raise ValueError("Imported runner escaped the exact frozen source root")
    return runner


def run_audit(
    run_root,
    source_root,
    receipt_path,
    *,
    expected_registration=None,
    expected_sources=None,
):
    run, source, target = (
        Path(run_root).resolve(),
        Path(source_root).resolve(),
        Path(receipt_path).resolve(),
    )
    if target.is_relative_to(run) or target.is_relative_to(source):
        raise ValueError("Receipt must not alter sealed scientific evidence or sources")
    sidecar = Path(str(target) + ".sha256")
    if target.exists() or sidecar.exists():
        raise FileExistsError("Audit receipt namespace is single-use")
    start = time.perf_counter()
    before = snapshot(run, source, expected_registration, expected_sources)
    # Process-local restriction, set before the frozen runner imports torch.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.dont_write_bytecode = True
    runner = _load_runner(source)
    torch = sys.modules.get("torch")
    if torch is not None:
        if torch.cuda.is_initialized():
            raise ValueError("Use a fresh CPU-only audit process")
        torch.set_num_threads(1)
    result = runner.rescore(run)
    after = snapshot(run, source, expected_registration, expected_sources)
    if before != after:
        raise ValueError("Scientific inputs or runtime sources changed during audit")
    if torch is not None and torch.cuda.is_initialized():
        raise ValueError("The CPU audit unexpectedly initialized CUDA")
    stage = run / "decomposition"
    saved = json.loads((stage / "RESULT.json").read_text())
    custody = json.loads((stage / "CUSTODY.json").read_text())
    counts = json.loads((stage / "COUNTS.json").read_text())
    index = json.loads((stage / "RAW_INDEX.json").read_text())
    native = result.get("native_forward_count")
    if (
        result.get("exact_match") is not True
        or result.get("confirmation_calls") != 0
        or result.get("automatic_followup_authorized") is not False
        or result.get("status") != "completed_fixed_head_decomposition"
        or result.get("status") != saved["status"]
        or result.get("assessment") != saved["assessment"]
        or type(native) is not int
        or not 36 <= native <= 1332
        or native != saved["native_forward_count"]
        or native != custody["native_forward_count"]
        or native != counts["saved_native"]
        or native != counts["attempted_native"]
        or native != len(index["endpoints"])
    ):
        raise ValueError("Exact CPU rescore, scope, assessment or native counts differ")
    receipt = {
        **result,
        **before["anchors"],
        "runtime_source_sha256": before["runtime_source_sha256"],
        "audit_source_sha256": digest(Path(__file__)),
        "audited_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": time.perf_counter() - start,
        "audit_scope": "CPU-only frozen native/tensor rescore; not scientific-success qualification or execution authority",
        "scientific_success_required": False,
        "automatic_followup_authorized": False,
    }
    payload = (
        json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    with target.open("xb") as stream:
        stream.write(payload)
    with sidecar.open("x") as stream:
        stream.write(hashlib.sha256(payload).hexdigest() + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--expected-registration-sha256")
    parser.add_argument("--expected-sources-sha256")
    args = parser.parse_args()
    receipt = run_audit(
        args.run_root,
        args.source_root,
        args.receipt,
        expected_registration=args.expected_registration_sha256,
        expected_sources=args.expected_sources_sha256,
    )
    print(
        json.dumps(
            {
                "exact_match": receipt["exact_match"],
                "native_forward_count": receipt["native_forward_count"],
                "receipt_sha256": digest(args.receipt),
                "scientific_success_required": False,
            }
        )
    )
