"""Bounded answer-query record-mass routing discovery, not confirmation.

Natural color donors see their requested field. Only their four record masses
are transferred: recipient values, within-record weights and non-record weights
stay recipient-owned. This does not establish an abstract entity mechanism or
complete-answer repair. Failed support checks stop; no epsilon rescue or later
GPU experiment is authorized. Resource/deletion guards belong to the caller.
"""

import argparse
import importlib
import importlib.metadata
import json
import sys
import traceback
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
CAP = 500_000_000
PRIOR_ANCHORS = (
    "registration_sha256",
    "custody_sha256",
    "result_sha256",
    "raw_index_sha256",
    "source_manifest_sha256",
    "plan_sha256",
    "inputs_sha256",
    "prefixes_sha256",
)


def modules(source_root):
    source_root = Path(source_root).resolve()
    sys.path.insert(0, str(source_root))
    screen = importlib.import_module("causal_run")
    if Path(screen.__file__).resolve().parent != source_root:
        raise ValueError("old screen escaped its explicit frozen source root")
    planner, hooks = (
        importlib.import_module("routing_plan"),
        importlib.import_module("record_routing_hooks"),
    )
    if any(
        Path(module.__file__).resolve().parent != HERE for module in (planner, hooks)
    ):
        raise ValueError("routing code escaped its frozen source root")
    return screen, planner, hooks


def verify_prior_receipt(receipt, anchors, counts):
    if (
        receipt.get("exact_match") is not True
        or receipt.get("confirmation_calls") != 0
        or receipt.get("physical_counts") != counts
        or any(receipt.get(name) != digest for name, digest in anchors.items())
    ):
        raise ValueError("prior native audit must match every frozen screen artifact")


def verified_prior(screen_root, source_root, receipt_path):
    screen, planner, hooks = modules(source_root)
    behavior, _, _, old, tokens, registration, old_inputs, base_plan = screen.check(
        screen_root
    )
    screen_root, receipt_path = Path(screen_root), Path(receipt_path)
    root = screen_root / "screen"
    paths = (
        screen_root / "REGISTRATION.json",
        root / "CUSTODY.json",
        root / "RESULT.json",
        root / "RAW_INDEX.json",
        screen_root / "SOURCES.json",
        screen_root / "PLAN.json",
        screen_root / "INPUTS.json",
        root / "PREFIXES.pt",
    )
    anchors = {key: old.verify_file(path) for key, path in zip(PRIOR_ANCHORS, paths)}
    custody = old.read_json(root / "CUSTODY.json")
    for name, digest in custody["artifact_sha256"].items():
        if Path(name).name != name:
            raise ValueError("prior custody escaped its namespace")
        old.verify_file(root / name, digest)
    old.verify_file(receipt_path)
    receipt = old.read_json(receipt_path)
    verify_prior_receipt(receipt, anchors, custody["physical_counts"])
    if (
        old.read_json(root / "RESULT.json")["status"] != "completed_discovery_screen"
        or receipt.get("status") != "completed_discovery_screen"
    ):
        raise ValueError("the prior bounded screen must be complete and audited")
    inputs = {
        "base_plan": base_plan,
        "rows": old_inputs["rows"],
        "worlds": old_inputs["worlds"],
        "prior_audit": receipt,
        "provenance": {
            "marker_plan_sha256": anchors["plan_sha256"],
            "marker_rescore_sha256": old.verify_file(receipt_path),
            "marker_registration_sha256": anchors["registration_sha256"],
            "marker_custody_sha256": anchors["custody_sha256"],
        },
    }
    return screen, planner, hooks, behavior, old, tokens, registration, inputs


def source_manifest(old, screen_root):
    return {
        "new": {path.name: old.sha256(path) for path in sorted(HERE.glob("*.py"))},
        "old_screen": old.read_json(Path(screen_root) / "SOURCES.json"),
    }


def validate_shared_context(planned, inputs):
    """Registry, bindings and Focus must match, not just printed record tokens."""
    rows, prefixes = {row["id"]: row for row in inputs["rows"]}, {}
    for job in planned["capture_jobs"]:
        row = rows[job["row_id"]]
        positions = row["marker_positions"]["[FOCUS_END]"]
        if (
            not positions
            or type(positions[-1]) is not int
            or not 0 <= positions[-1] < len(row["input_ids"])
        ):
            raise ValueError("the shared registry/Focus boundary is invalid")
        prefix = row["input_ids"][: positions[-1] + 1]
        if job["case_id"] in prefixes and prefixes[job["case_id"]] != prefix:
            raise ValueError("registry, bindings or Focus differ across four contexts")
        prefixes[job["case_id"]] = prefix


def prepare(output, *, screen_root, screen_source_root, screen_rescore_receipt):
    screen, _, _ = modules(screen_source_root)
    prior_registration = json.loads(
        (Path(screen_root) / "REGISTRATION.json").read_text()
    )
    behavior, _, _ = screen.modules(prior_registration["behavior_source_root"])
    old = behavior.helpers(
        json.loads(
            (
                Path(prior_registration["behavior_root"]) / "REGISTRATION.json"
            ).read_text()
        )["dependency_root"]
    )
    output = Path(output)
    output.mkdir()
    try:
        _, planner, _, behavior, old, _, _, inputs = verified_prior(
            screen_root, screen_source_root, screen_rescore_receipt
        )
        planned, code = planner.plan(inputs), source_manifest(old, screen_root)
        validate_shared_context(planned, inputs)
        if len(planned["capture_jobs"]) != 32 or planned["storage"]["cap_bytes"] != CAP:
            raise ValueError("routing grid or bounded storage changed")
        (output / "sources").mkdir()
        for name in code["new"]:
            old.write_bytes(output / "sources" / name, (HERE / name).read_bytes())
        records = {
            "SOURCES.json": code,
            "INPUTS.json": inputs,
            "PLAN.json": planned,
            "PRIOR_RECEIPT.json": inputs["prior_audit"],
        }
        for name, value in records.items():
            old.seal_json(output / name, value)
        registration = {
            "screen_root": str(Path(screen_root).resolve()),
            "screen_source_root": str(Path(screen_source_root).resolve()),
            "screen_rescore_receipt": str(Path(screen_rescore_receipt).resolve()),
            "input_sha256": {name: old.verify_file(output / name) for name in records},
            "runtime_python": str(Path(sys.executable).absolute()),
            "python_version": sys.version,
            "runtime_versions": {
                name: importlib.metadata.version(name) for name in behavior.PACKAGES
            },
            "cap_bytes": CAP,
            "stage": "answer_query_record_mass_routing_only",
            "confirmation_calls_authorized": False,
            "execution_scope": {
                "device": "cuda:0",
                "dtype": "float32",
                "attention": "eager",
                "batch_size": 1,
                "lm_head_positions": 1,
                "tf32": False,
            },
        }
        if code != source_manifest(old, screen_root):
            raise ValueError("source changed during routing preparation")
        old.seal_json(output / "REGISTRATION.json", registration)
        return {"prepared": True, "natural_jobs": 32, "storage": planned["storage"]}
    except BaseException:
        old.seal_json(
            output / "failure.json",
            {"phase": "prepare", "traceback": traceback.format_exc()},
        )
        raise


def check(output):
    output = Path(output)
    registration = json.loads((output / "REGISTRATION.json").read_text())
    if (
        set(registration["input_sha256"])
        != {"SOURCES.json", "INPUTS.json", "PLAN.json", "PRIOR_RECEIPT.json"}
        or registration["cap_bytes"] != CAP
        or registration["stage"] != "answer_query_record_mass_routing_only"
        or registration["confirmation_calls_authorized"]
    ):
        raise ValueError("routing scope or complete input closure changed")
    context = verified_prior(
        registration["screen_root"],
        registration["screen_source_root"],
        registration["screen_rescore_receipt"],
    )
    _, planner, _, _, old, _, _, inputs = context
    old.verify_file(output / "REGISTRATION.json")
    for name, digest in registration["input_sha256"].items():
        old.verify_file(output / name, digest)
    planned, code = (
        old.read_json(output / "PLAN.json"),
        old.read_json(output / "SOURCES.json"),
    )
    validate_shared_context(planned, inputs)
    if (
        inputs != old.read_json(output / "INPUTS.json")
        or planned != planner.plan(inputs)
        or code != source_manifest(old, registration["screen_root"])
    ):
        raise ValueError("old evidence, routing plan or source closure changed")
    for name, digest in code["new"].items():
        old.verify_file(output / "sources" / name, digest)
    return (*context, registration, planned)


def exact_equal(first, second):
    if isinstance(first, torch.Tensor):
        return (
            isinstance(second, torch.Tensor)
            and first.dtype == second.dtype
            and first.shape == second.shape
            and torch.equal(first, second)
        )
    if isinstance(first, dict):
        return (
            isinstance(second, dict)
            and set(first) == set(second)
            and all(exact_equal(value, second[key]) for key, value in first.items())
        )
    if isinstance(first, (list, tuple)):
        return (
            type(first) is type(second)
            and len(first) == len(second)
            and all(exact_equal(a, b) for a, b in zip(first, second))
        )
    return type(first) is type(second) and first == second


def capture_summaries(planned, raw, reader=None):
    summaries, reference = [], {}
    for job in planned["capture_jobs"]:
        captured, model = raw[job["endpoint_id"]], planned["model"]
        if (
            set(captured["layers"]) != set(range(model["layers"]))
            or captured["intervention"] is not None
        ):
            raise ValueError("natural routing capture coverage or pass-through changed")
        masses, hashes = [], {}
        regions = job["record_regions"]
        for layer in range(model["layers"]):
            item = captured["layers"][layer]
            weights = item["attention_rows"]
            if (
                weights.dtype != torch.float32
                or weights.device.type != "cpu"
                or weights.ndim != 2
                or weights.shape[0] != model["heads"]
                or not bool(torch.isfinite(weights).all())
                or bool((weights < 0).any())
                or item["head_dim"] != model["head_dim"]
                or type(item["num_key_value_heads"]) is not int
                or item["num_key_value_heads"] <= 0
                or model["heads"] % item["num_key_value_heads"] != 0
                or item["record_positions"]
                != sorted(
                    position for positions in regions.values() for position in positions
                )
            ):
                raise ValueError(
                    "routing rows must be complete finite nonnegative CPU float32"
                )
            if bool((weights.to(torch.float64).sum(dim=1) - 1).abs().gt(1e-6).any()):
                raise ValueError(
                    "every attention head must retain a normalized probability row"
                )
            row_masses = torch.stack(
                [weights[:, regions[obj]].to(torch.float64).sum(1) for obj in "ABCD"],
                dim=1,
            )
            masses.append(row_masses.tolist())
            digest = item["record_value_sha256"]
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
            ):
                raise ValueError("record value hash is invalid")
            key = (job["case_id"], layer)
            if key in reference and reference[key] != digest:
                raise ValueError(
                    "record values differ across the four natural contexts"
                )
            reference[key] = digest
            hashes[str(layer)] = digest
        summary = {
            "endpoint_id": job["endpoint_id"],
            "record_masses": masses,
            "record_value_hashes": hashes,
        }
        if reader is not None:
            summary["argmax_correct"] = (
                int(reader.read(job["endpoint_id"]).argmax())
                in job["expected_label_token_ids"]
            )
        summaries.append(summary)
    return summaries


def effective_profile(job, recipient, regions, hooks):
    if job["condition"] != "selfRscreen":
        return job["desired_profile"]
    row = recipient["layers"][job["layer_index"]]["attention_rows"][job["head_index"]]
    profile = hooks.record_mass_profile(row, regions)["profile"]
    if not torch.allclose(
        profile,
        torch.tensor(job["desired_profile"], dtype=torch.float64),
        atol=1e-14,
        rtol=0,
    ):
        raise ValueError(
            "canonical self profile differs from the registered semantic target"
        )
    return profile.tolist()


def replay_operation(job, captured, recipient, regions, hooks):
    metadata = captured["intervention"]
    profile = effective_profile(job, recipient, regions, hooks)
    layer, head = job["layer_index"], job["head_index"]
    natural_row = recipient["layers"][layer]["attention_rows"][head]
    dim = recipient["layers"][layer]["head_dim"]
    if metadata is not None and (
        tuple(metadata["value_matrix"].shape) != (len(natural_row), dim)
        or any(
            metadata[key].dtype != torch.float32 or tuple(metadata[key].shape) != (dim,)
            for key in ("old_head_output", "new_head_output")
        )
    ):
        raise ValueError("routing value/head dimensions differ from actual head_dim")
    if (
        metadata is None
        or metadata["layer_index"] != layer
        or metadata["head_index"] != head
        or not torch.equal(metadata["old_weights"], natural_row)
    ):
        raise ValueError("operation does not use the selected natural recipient row")
    replayed = hooks.route_record_weights(
        metadata["old_weights"], metadata["value_matrix"], regions, profile
    )
    if any(
        not exact_equal(value, metadata.get(key)) for key, value in replayed.items()
    ):
        raise ValueError("saved routing operation differs from exact CPU replay")
    if not torch.equal(
        metadata["new_head_output"],
        metadata["old_head_output"] + metadata["delta_head_output"],
    ):
        raise ValueError(
            "saved head output does not equal its recorded recipient-plus-delta"
        )
    if job["condition"] == "selfRscreen" and not metadata["self_zero_delta"]:
        raise ValueError("self profile must produce an exact zero delta")
    for index, item in captured["layers"].items():
        if (
            item["record_value_sha256"]
            != recipient["layers"][index]["record_value_sha256"]
        ):
            raise ValueError("an intervention changed recipient record values")
    return profile


def evaluate(reader, original_reader, raw, planned, inputs, screen, planner, hooks):
    torch.set_num_threads(1)
    rows = {row["id"]: row for row in inputs["rows"]}
    for job in planned["capture_jobs"]:
        native = reader.read(job["endpoint_id"])
        if not torch.equal(native, original_reader.read(job["row_id"] + "::step00")):
            raise ValueError("natural routing capture is not a bitwise original replay")
        if any(
            layer["attention_rows"].shape[1] != len(rows[job["row_id"]]["input_ids"])
            for layer in raw[job["endpoint_id"]]["layers"].values()
        ):
            raise ValueError("capture does not retain the complete final attention row")
    summaries = capture_summaries(planned, raw, reader)
    selected = planner.select_heads(planned, summaries)
    natural_jobs = {
        (job["case_id"], job["role"]): job for job in planned["capture_jobs"]
    }
    measurements, self_oracles = [], []
    for job in selected["intervention_jobs"]:
        baseline = natural_jobs[(job["case_id"], "Rscreen")]
        recipient, captured = raw[baseline["endpoint_id"]], raw[job["endpoint_id"]]
        # Reuse the natural-row contract; operation metadata is replayed below.
        probe = {**baseline, "endpoint_id": job["endpoint_id"]}
        capture_summaries(
            {"model": planned["model"], "capture_jobs": [probe]},
            {job["endpoint_id"]: {"layers": captured["layers"], "intervention": None}},
        )
        profile = replay_operation(
            job, captured, recipient, baseline["record_regions"], hooks
        )
        native, clean = (
            reader.read(job["endpoint_id"]),
            reader.read(baseline["endpoint_id"]),
        )
        if job["condition"] == "selfRscreen":
            if not torch.equal(native, clean):
                raise ValueError("self routing failed the bitwise native oracle")
            self_oracles.append(
                {
                    "endpoint_id": job["endpoint_id"],
                    "bitwise_equal": True,
                    "effective_profile": profile,
                }
            )
        gain = screen.label_margin(
            native, job["correct_token_ids"], job["wrong_token_ids"]
        ) - screen.label_margin(clean, job["correct_token_ids"], job["wrong_token_ids"])
        measurements.append(
            {
                "endpoint_id": job["endpoint_id"],
                "correct_minus_wrong_gain": gain,
                "argmax_correct": int(native.argmax()) in job["correct_token_ids"],
                "clean_argmax_correct": int(clean.argmax()) in job["correct_token_ids"],
            }
        )
    expected = {
        j["endpoint_id"]
        for j in planned["capture_jobs"] + selected["intervention_jobs"]
    }
    if set(reader.index["endpoints"]) != expected or set(raw) != expected:
        raise ValueError("routing native or tensor endpoint coverage changed")
    assessment = planner.assess(selected, measurements)
    result = {
        "status": "completed_routing_discovery"
        if selected["selected_heads"]
        else selected["status"],
        "selected_heads": selected["selected_heads"],
        "assessment": assessment,
        "measurements": measurements,
        "self_oracles": self_oracles,
        "native_forward_count": len(expected),
        "natural_capture_count": 32,
        "intervention_count": len(selected["intervention_jobs"]),
        "complete_answer_repair_established": False,
        "mechanism_identified": False,
        "further_validation_authorized": False,
        "confirmation_calls": 0,
        "limitations": planned["limitations"],
    }
    return summaries, selected, result


def run(output, *, deadline):
    output = Path(output)
    registration = json.loads((output / "REGISTRATION.json").read_text())
    screen, _, _ = modules(registration["screen_source_root"])
    prior_reg = json.loads(
        (Path(registration["screen_root"]) / "REGISTRATION.json").read_text()
    )
    behavior, _, _ = screen.modules(prior_reg["behavior_source_root"])
    behavior.validate_deadline(deadline)
    (
        screen,
        planner,
        hooks,
        behavior,
        old,
        tokens,
        prior_reg,
        inputs,
        registration,
        planned,
    ) = check(output)
    root = output / "routing"
    root.mkdir()
    writer, raw = None, {}
    counts = {"attempted_native": 0, "saved_native": 0}
    try:
        old.seal_json(
            root / "EXECUTION.json",
            {
                "deadline_utc": deadline,
                "stage": registration["stage"],
                "confirmation_calls_authorized": False,
            },
        )
        old.seal_json(root / "RUNTIME.json", old.configure_runtime(registration))
        old.check_deadline(deadline)
        model = behavior.load_model(old, behavior.MODELS["qwen"], tokens, root)
        if any(
            p.requires_grad or p.dtype != torch.float32 for p in model.parameters()
        ) or any(module.training for module in model.modules()):
            raise ValueError("routing requires frozen FP32 evaluation mode")
        rows, writer = {r["id"]: r for r in inputs["rows"]}, old.RawWriter(root)
        original_root = Path(prior_reg["behavior_root"]) / "qwen"
        original_reader = old.RawReader(
            original_root, old.read_json(original_root / "RAW_INDEX.json")
        )

        def execute(job, regions, intervention=None):
            old.check_deadline(deadline)
            if counts["saved_native"] >= planned["storage"]["max_native_endpoints"]:
                raise RuntimeError("routing native endpoint bound exceeded")
            counts["attempted_native"] += 1
            with hooks.route_attention(
                model, regions, intervention=intervention
            ) as captured:
                native = screen.native_forward(
                    model, rows[job["row_id"]], tokens["vocab_size"]
                )
            raw[job["endpoint_id"]] = captured
            writer.add(job["endpoint_id"], native)
            counts["saved_native"] += 1
            return native

        for job in planned["capture_jobs"]:
            native = execute(job, job["record_regions"])
            if not torch.equal(
                native, original_reader.read(job["row_id"] + "::step00")
            ):
                raise ValueError(
                    "natural routing capture is not a bitwise original replay"
                )
        writer.flush()
        old.save_pt(root / "NATURAL.pt", dict(raw))
        current = old.RawReader(root, writer.index)
        summaries = capture_summaries(planned, raw, current)
        selected = planner.select_heads(planned, summaries)
        old.seal_json(root / "CAPTURE_SUMMARIES.json", summaries)
        old.seal_json(root / "SELECTED_PLAN.json", selected)
        recipients = {
            j["case_id"]: j for j in planned["capture_jobs"] if j["role"] == "Rscreen"
        }
        # All exact self oracles precede every non-self intervention.
        jobs = sorted(
            selected["intervention_jobs"],
            key=lambda job: job["condition"] != "selfRscreen",
        )
        for job in jobs:
            baseline = recipients[job["case_id"]]
            recipient = raw[baseline["endpoint_id"]]
            profile = effective_profile(
                job, recipient, baseline["record_regions"], hooks
            )
            native = execute(
                job,
                baseline["record_regions"],
                {
                    "layer_index": job["layer_index"],
                    "head_index": job["head_index"],
                    "desired_profile": profile,
                },
            )
            replay_operation(
                job,
                raw[job["endpoint_id"]],
                recipient,
                baseline["record_regions"],
                hooks,
            )
            if job["condition"] == "selfRscreen" and not torch.equal(
                native, current.read(baseline["endpoint_id"])
            ):
                raise ValueError("self routing failed the bitwise native oracle")
            if counts["saved_native"] % 8 == 0:
                print(json.dumps(counts), flush=True)
        natural_ids = {j["endpoint_id"] for j in planned["capture_jobs"]}
        old.save_pt(
            root / "INTERVENTIONS.pt",
            {k: v for k, v in raw.items() if k not in natural_ids},
        )
        writer.finish()
        summaries, selected, result = evaluate(
            old.RawReader(root, old.read_json(root / "RAW_INDEX.json")),
            original_reader,
            raw,
            planned,
            inputs,
            screen,
            planner,
            hooks,
        )
        if (
            counts["saved_native"] != result["native_forward_count"]
            or counts["attempted_native"] != counts["saved_native"]
        ):
            raise ValueError(
                "physical routing count differs from saved native endpoints"
            )
        old.seal_json(root / "RESULT.json", result)
        old.seal_json(root / "COUNTS.json", counts)
        check(output)
        if (
            sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
            > CAP - 1024**2
        ):
            raise RuntimeError("complete routing evidence exceeds the 500-MB ceiling")
        old.seal_json(
            root / "CUSTODY.json",
            {
                "registration_sha256": old.verify_file(output / "REGISTRATION.json"),
                "artifact_sha256": {
                    p.name: old.sha256(p)
                    for p in root.iterdir()
                    if p.is_file() and not p.name.endswith(".sha256")
                },
                "native_forward_count": counts["saved_native"],
            },
        )
        return result
    except BaseException:
        trace, flush_error = traceback.format_exc(), None
        if writer is not None and not writer.closed:
            try:
                writer.finish()
            except (OSError, RuntimeError, ValueError) as error:
                flush_error = {"type": type(error).__name__, "message": str(error)}
        if raw:
            old.save_pt(root / "PARTIAL_CAPTURES.pt", raw)
        old.seal_json(
            root / "failure.json",
            {
                "phase": "routing",
                "traceback": trace,
                "flush_error": flush_error,
                "counts": counts,
            },
        )
        raise


def rescore(output):
    output = Path(output)
    if sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) > CAP:
        raise ValueError("routing evidence exceeds the 500-MB ceiling")
    (
        screen,
        planner,
        hooks,
        _,
        old,
        tokens,
        prior_reg,
        inputs,
        _registration,
        planned,
    ) = check(output)
    root = output / "routing"
    old.verify_file(root / "CUSTODY.json")
    custody = old.read_json(root / "CUSTODY.json")
    if custody["registration_sha256"] != old.verify_file(output / "REGISTRATION.json"):
        raise ValueError("routing custody changed registration")
    for name, digest in custody["artifact_sha256"].items():
        if Path(name).name != name:
            raise ValueError("routing custody escaped its namespace")
        old.verify_file(root / name, digest)
    natural = torch.load(root / "NATURAL.pt", map_location="cpu", weights_only=True)
    patched = torch.load(
        root / "INTERVENTIONS.pt", map_location="cpu", weights_only=True
    )
    if set(natural).intersection(patched):
        raise ValueError("natural/intervention tensor namespaces overlap")
    raw = {**natural, **patched}
    reader = old.RawReader(root, old.read_json(root / "RAW_INDEX.json"))
    for identity in reader.index["endpoints"]:
        native = reader.read(identity)
        if (
            native.dtype != torch.float32
            or tuple(native.shape) != (tokens["vocab_size"],)
            or not bool(torch.isfinite(native).all())
        ):
            raise ValueError("routing native endpoints must be complete finite FP32")
    original_root = Path(prior_reg["behavior_root"]) / "qwen"
    original_reader = old.RawReader(
        original_root, old.read_json(original_root / "RAW_INDEX.json")
    )
    summaries, selected, result = evaluate(
        reader, original_reader, raw, planned, inputs, screen, planner, hooks
    )
    counts = old.read_json(root / "COUNTS.json")
    if (
        summaries != old.read_json(root / "CAPTURE_SUMMARIES.json")
        or selected != old.read_json(root / "SELECTED_PLAN.json")
        or result != old.read_json(root / "RESULT.json")
        or counts["attempted_native"] != counts["saved_native"]
        or counts["saved_native"] != result["native_forward_count"]
        or custody["native_forward_count"] != counts["saved_native"]
    ):
        raise ValueError("exact routing native/tensor replay or counts differ")
    return {
        "exact_match": True,
        "native_forward_count": counts["saved_native"],
        "status": result["status"],
        "assessment": result["assessment"],
        "confirmation_calls": 0,
        "further_validation_authorized": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "check", "run", "rescore"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--screen-root", type=Path)
    parser.add_argument("--screen-source-root", type=Path)
    parser.add_argument("--screen-rescore-receipt", type=Path)
    parser.add_argument("--deadline")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(
            args.output,
            screen_root=args.screen_root,
            screen_source_root=args.screen_source_root,
            screen_rescore_receipt=args.screen_rescore_receipt,
        )
    elif args.command == "check":
        check(args.output)
        result = {"registration_valid": True}
    else:
        result = (
            run(args.output, deadline=args.deadline)
            if args.command == "run"
            else rescore(args.output)
        )
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
