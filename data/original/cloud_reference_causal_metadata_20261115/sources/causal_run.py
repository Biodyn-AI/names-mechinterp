"""Conditional Qwen-only discovery screen, never a confirmation experiment.

Only a passed, independently audited development result can prepare this stage.
One field per selected world screens layers/heads using complete native logits.
A pre-field donor can contain a bundle of facts; this screen does not identify
an entity variable, demonstrate complete-answer repair, establish preservation,
or authorize later cross-attribute/reverse/confirmation experiments.
"""

import argparse
import importlib
import importlib.metadata
import json
import sys
import traceback
from contextlib import nullcontext
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
RAW_CAP, METADATA_ALLOWANCE, ORACLE_TOLERANCE = 1_000_000_000, 64 * 1024**2, 2e-4
RECEIPT_ANCHORS = (
    "registration_sha256",
    "custody_sha256",
    "result_sha256",
    "raw_index_sha256",
    "generations_sha256",
    "source_manifest_sha256",
)


def modules(source_root):
    source_root = Path(source_root).resolve()
    sys.path.insert(0, str(source_root))
    behavior = importlib.import_module("error_run")
    for name in ("error_run", "error_analysis", "error_protocol"):
        if Path(importlib.import_module(name).__file__).resolve().parent != source_root:
            raise ValueError("behavior module escaped the explicit frozen source root")
    planner, hooks = (
        importlib.import_module("causal_plan"),
        importlib.import_module("prefix_hooks"),
    )
    if any(
        Path(module.__file__).resolve().parent != HERE for module in (planner, hooks)
    ):
        raise ValueError("causal module escaped its source root")
    return behavior, planner, hooks


def verify_receipt(receipt, anchors, native_steps):
    if (
        receipt.get("model") != "qwen"
        or receipt.get("exact_match") is not True
        or receipt.get("native_steps") != native_steps
        or receipt.get("confirmation_calls") != 0
        or receipt.get("causal_calls") != 0
    ):
        raise ValueError("a complete, exact Qwen development rescore is required")
    if any(receipt.get(name) != digest for name, digest in anchors.items()):
        raise ValueError("rescore receipt is not bound to every behavioral artifact")


def verified_behavior(behavior_root, source_root, receipt_path):
    behavior, planner, hooks = modules(source_root)
    old, _ = behavior.check(behavior_root)
    behavior_root, receipt_path = Path(behavior_root), Path(receipt_path)
    root = behavior_root / "qwen"
    anchors = {
        name: old.verify_file(path)
        for name, path in zip(
            RECEIPT_ANCHORS,
            (
                behavior_root / "REGISTRATION.json",
                root / "CUSTODY.json",
                root / "RESULT.json",
                root / "RAW_INDEX.json",
                root / "GENERATIONS.json",
                behavior_root / "CODE.json",
            ),
        )
    }
    receipt_hash = old.verify_file(receipt_path)
    receipt, custody = old.read_json(receipt_path), old.read_json(root / "CUSTODY.json")
    for name, digest in custody["artifact_sha256"].items():
        if Path(name).name != name:
            raise ValueError("behavioral custody path escaped its namespace")
        old.verify_file(root / name, digest)
    predictions = old.read_json(root / "GENERATIONS.json")
    verify_receipt(
        receipt, anchors, sum(len(item["generated_ids"]) for item in predictions)
    )
    tokens = behavior.checked_tokens(old, behavior_root, "qwen")
    inputs = {
        "rows": behavior.development_rows(tokens, "development"),
        "worlds": behavior.development_worlds(
            old.read_json(behavior_root / "WORLDS.json")
        ),
        "predictions": predictions,
        "behavioral_result": old.read_json(root / "RESULT.json"),
        "custody": custody,
        "independent_rescore": receipt,
        "provenance": {
            "registration_sha256": anchors["registration_sha256"],
            "custody_sha256": anchors["custody_sha256"],
            "rescore_sha256": receipt_hash,
        },
    }
    rebuilt = importlib.import_module("error_analysis").score(
        inputs["rows"], predictions, inputs["worlds"]
    )
    if rebuilt != inputs["behavioral_result"] or not rebuilt["feasibility"]["passed"]:
        raise ValueError("the unchanged behavioral feasibility gate must pass")
    return behavior, planner, hooks, old, tokens, inputs


def exhaustive_label_ids(tokenizer, vocab_size, banks):
    normalize = importlib.import_module("error_analysis").normalize
    result = {field: {label: [] for label in labels} for field, labels in banks.items()}
    lookup = {
        normalize(label): (field, label)
        for field, labels in banks.items()
        for label in labels
    }
    for token in range(vocab_size):
        label = normalize(tokenizer.decode([token], skip_special_tokens=False))
        if label in lookup:
            field, value = lookup[label]
            result[field][value].append(token)
    if any(not values for bank in result.values() for values in bank.values()):
        raise ValueError(
            "every declared value needs exhaustive whole-label one-token IDs"
        )
    return result


def architecture(tokens, model_info):
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        tokens["snapshot"], local_files_only=True, token=False, trust_remote_code=False
    )
    if (
        config.model_type != "qwen3"
        or model_info["model_id"] != "Qwen/Qwen3-4B-Instruct-2507"
    ):
        raise ValueError("this screen is restricted to the fixed Qwen model")
    head_dim = getattr(config, "head_dim", None)
    if type(head_dim) is not int or head_dim <= 0:
        raise ValueError("the pinned config must explicitly state head_dim")
    return {
        "model_id": model_info["model_id"],
        "revision": model_info["revision"],
        "vocab_size": config.vocab_size,
        "layers": config.num_hidden_layers,
        "heads": config.num_attention_heads,
        "head_dim": head_dim,
        "o_proj_in_features": config.num_attention_heads * head_dim,
    }


def source_manifest(behavior, old, behavior_registration):
    return {
        "causal": {path.name: old.sha256(path) for path in sorted(HERE.glob("*.py"))},
        "behavior": behavior.source_manifest(
            old, behavior_registration["dependency_root"]
        ),
    }


def prepare(output, *, behavior_root, source_root, receipt_path):
    behavior, _, _ = modules(source_root)
    old = behavior.helpers(
        json.loads((Path(behavior_root) / "REGISTRATION.json").read_text())[
            "dependency_root"
        ]
    )
    output = Path(output)
    output.mkdir()
    try:
        behavior, planner, _, old, tokens, inputs = verified_behavior(
            behavior_root, source_root, receipt_path
        )
        inputs["model"] = architecture(tokens, behavior.MODELS["qwen"])
        protocol = importlib.import_module("error_protocol")
        inputs["label_token_ids"] = exhaustive_label_ids(
            old.load_tokenizer(tokens["snapshot"]),
            tokens["vocab_size"],
            {field: protocol.BANKS[field] for field in protocol.FIELDS},
        )
        planned = planner.plan(inputs)
        if planned["status"] != "ready_for_root_audit" or len(planned["cases"]) != 8:
            raise ValueError(
                "eight independently eligible worlds are required; no rescue"
            )
        behavior_registration = old.read_json(Path(behavior_root) / "REGISTRATION.json")
        code = source_manifest(behavior, old, behavior_registration)
        sources = output / "sources"
        sources.mkdir()
        for name in code["causal"]:
            old.write_bytes(sources / name, (HERE / name).read_bytes())
        records = {
            "SOURCES.json": code,
            "INPUTS.json": inputs,
            "PLAN.json": planned,
            "RESCORE_RECEIPT.json": old.read_json(receipt_path),
        }
        for name, record in records.items():
            old.seal_json(output / name, record)
        registration = {
            "behavior_root": str(Path(behavior_root).resolve()),
            "behavior_source_root": str(Path(source_root).resolve()),
            "verified_rescore_path": str(Path(receipt_path).resolve()),
            "input_sha256": {name: old.verify_file(output / name) for name in records},
            "runtime_python": str(Path(sys.executable).absolute()),
            "python_version": sys.version,
            "runtime_versions": {
                name: importlib.metadata.version(name) for name in behavior.PACKAGES
            },
            "raw_cap_bytes": RAW_CAP,
            "oracle_tolerance": ORACLE_TOLERANCE,
            "model": "qwen",
            "stage": "development_first_token_screen_only",
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
        if code != source_manifest(behavior, old, behavior_registration):
            raise ValueError("source changed during preparation")
        old.seal_json(output / "REGISTRATION.json", registration)
        return {
            "prepared": True,
            "cases": 8,
            "storage": planned["storage"],
            "confirmation_calls_authorized": False,
        }
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
        != {"SOURCES.json", "INPUTS.json", "PLAN.json", "RESCORE_RECEIPT.json"}
        or registration["stage"] != "development_first_token_screen_only"
    ):
        raise ValueError("registered input closure or first-token stage changed")
    behavior, planner, hooks, old, tokens, base = verified_behavior(
        registration["behavior_root"],
        registration["behavior_source_root"],
        registration["verified_rescore_path"],
    )
    old.verify_file(output / "REGISTRATION.json")
    for name, digest in registration["input_sha256"].items():
        old.verify_file(output / name, digest)
    inputs, planned, code = (
        old.read_json(output / name)
        for name in ("INPUTS.json", "PLAN.json", "SOURCES.json")
    )
    behavior_registration = old.read_json(
        Path(registration["behavior_root"]) / "REGISTRATION.json"
    )
    if (
        any(inputs.get(name) != value for name, value in base.items())
        or inputs["model"] != architecture(tokens, behavior.MODELS["qwen"])
        or planned != planner.plan(inputs)
    ):
        raise ValueError("behavioral inputs, geometry or deterministic plan changed")
    if (
        code != source_manifest(behavior, old, behavior_registration)
        or registration["model"] != "qwen"
        or registration["raw_cap_bytes"] != RAW_CAP
        or registration["oracle_tolerance"] != ORACLE_TOLERANCE
        or registration["confirmation_calls_authorized"]
        or planned["status"] != "ready_for_root_audit"
    ):
        raise ValueError("frozen source or first-token-only scope changed")
    for name, digest in code["causal"].items():
        old.verify_file(output / "sources" / name, digest)
    return behavior, planner, hooks, old, tokens, registration, inputs, planned


def label_margin(native, correct_ids, wrong_ids):
    native = native.to(torch.float64)
    return float(
        torch.logsumexp(native[correct_ids], 0) - torch.logsumexp(native[wrong_ids], 0)
    )


def oracle_difference(native, clean):
    difference = float((native.to(torch.float64) - clean.to(torch.float64)).abs().max())
    same_argmax = int(native.argmax()) == int(clean.argmax())
    return {
        "max_abs_diff": difference,
        "same_argmax": same_argmax,
        "passed": difference <= ORACLE_TOLERANCE and same_argmax,
    }


def _labels(case, row, inputs):
    world = next(w for w in inputs["worlds"] if w["world_id"] == case["world_id"])
    field = case["screen_field"]
    wrong = world["objects"][case["reverse_predicted_object"]][field]
    return inputs["label_token_ids"][field][row["expected_answer"]], inputs[
        "label_token_ids"
    ][field][wrong]


def measure_jobs(reader, jobs, planned, inputs):
    cases, rows = (
        {c["case_id"]: c for c in planned["cases"]},
        {r["id"]: r for r in inputs["rows"]},
    )
    measured = []
    for job in jobs:
        case, row = cases[job["case_id"]], rows[job["row_id"]]
        correct, wrong = _labels(case, row, inputs)
        clean = reader.read(f"clean::{case['case_id']}")
        gain = label_margin(
            reader.read(job["endpoint_id"]), correct, wrong
        ) - label_margin(clean, correct, wrong)
        measured.append(
            {"endpoint_id": job["endpoint_id"], "correct_minus_wrong_gain": gain}
        )
    return measured


def evaluate_screen(reader, original_reader, planned, inputs):
    planner = importlib.import_module("causal_plan")
    expected, clean_results, layer_oracles, head_oracles, layer_scores, head_scores = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    cases = {c["case_id"]: c for c in planned["cases"]}
    for job in planned["clean_jobs"]:
        native = reader.read(job["endpoint_id"])
        original = original_reader.read(job["row_id"] + "::step00")
        if not torch.equal(native, original):
            raise ValueError("clean replay differs from bitwise original native logits")
        clean_results.append({"endpoint_id": job["endpoint_id"], "bitwise_equal": True})
        expected.append(job["endpoint_id"])

    def oracles(jobs):
        values = []
        for job in jobs:
            clean = reader.read(f"clean::{cases[job['case_id']]['case_id']}")
            values.append(
                {
                    "endpoint_id": job["endpoint_id"],
                    **oracle_difference(reader.read(job["endpoint_id"]), clean),
                }
            )
            expected.append(job["endpoint_id"])
        return values

    layer_oracles = oracles(planned["layer_self_jobs"])
    final, status = (
        {**planned, "selected_layers": [], "selected_heads": []},
        "stopped_layer_self_oracle",
    )
    if all(item["passed"] for item in layer_oracles):
        layer_scores = measure_jobs(reader, planned["layer_jobs"], planned, inputs)
        expected += [j["endpoint_id"] for j in planned["layer_jobs"]]
        final = planner.select_layers(planned, layer_scores)
        final["selected_heads"] = []
        status = final["status"]
        if final["selected_layers"]:
            head_oracles = oracles(final["head_self_jobs"])
            status = "stopped_head_self_oracle"
            if all(item["passed"] for item in head_oracles):
                head_scores = measure_jobs(reader, final["head_jobs"], final, inputs)
                expected += [j["endpoint_id"] for j in final["head_jobs"]]
                final = planner.select_heads(final, head_scores)
                status = (
                    "completed_discovery_screen"
                    if final["selected_heads"]
                    else final["status"]
                )
    if len(expected) != len(set(expected)) or set(expected) != set(
        reader.index["endpoints"]
    ):
        raise ValueError(
            "native endpoint coverage differs from the fixed adaptive screen"
        )
    return {
        "status": status,
        "selected_layers": final["selected_layers"],
        "selected_heads": final["selected_heads"],
        "head_jobs": final.get("head_jobs", []),
        "head_self_jobs": final.get("head_self_jobs", []),
        "clean_replays": clean_results,
        "layer_self_oracles": layer_oracles,
        "head_self_oracles": head_oracles,
        "layer_measurements": layer_scores,
        "head_measurements": head_scores,
        # Retain every rank using exactly the frozen planner's selection rule.
        "layer_ranking": [
            {"layer_index": key[0], "mean_gain": gain}
            for key, gain in planner._ranking(
                planned["layer_jobs"], layer_scores, ("layer_index",)
            )
        ]
        if layer_scores
        else [],
        "head_ranking": [
            {"layer_index": key[0], "head_index": key[1], "mean_gain": gain}
            for key, gain in planner._ranking(
                final["head_jobs"], head_scores, ("layer_index", "head_index")
            )
        ]
        if head_scores
        else [],
        "native_forward_count": len(expected),
        "backbone_prefix_forward_count": len(planned["prefix_capture_jobs"]),
        "complete_answer_repair_established": False,
        "mechanism_identified": False,
        "confirmation_calls": 0,
        "scope": planned["scope"],
    }


def validate_prefixes(prefixes, planned):
    expected = {job["endpoint_id"]: job for job in planned["prefix_capture_jobs"]}
    if set(prefixes) != set(expected):
        raise ValueError("prefix capture coverage changed")
    size = 0
    for identity, captured in prefixes.items():
        job, model = expected[identity], planned["model"]
        if (
            captured["input_ids"] != job["input_ids"]
            or captured["prefix_length"] != len(job["input_ids"])
            or captured["alias_end_position"] != job["alias_end_position"]
            or captured["forward_scope"] != "backbone_prefix_only_no_logits"
            or captured["num_attention_heads"] != model["heads"]
            or captured["head_dim"] != model["head_dim"]
            or set(captured["layers"]) != set(job["layers"])
        ):
            raise ValueError("prefix geometry, cut IDs or forward scope changed")
        for vector in captured["layers"].values():
            if (
                vector.dtype != torch.float32
                or vector.device.type != "cpu"
                or tuple(vector.shape) != (model["o_proj_in_features"],)
                or not bool(torch.isfinite(vector).all())
            ):
                raise ValueError("prefix vectors must be complete finite CPU float32")
            size += vector.numel() * 4
    return size


def native_forward(model, row, vocab_size):
    ids = torch.tensor(
        [row["input_ids"]], dtype=torch.long, device=next(model.parameters()).device
    )
    with torch.inference_mode():
        logits = model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            use_cache=False,
            return_dict=True,
            logits_to_keep=1,
        ).logits
    if (
        logits.dtype != torch.float32
        or tuple(logits.shape) != (1, 1, vocab_size)
        or not bool(torch.isfinite(logits).all())
    ):
        raise ValueError(
            "screen outputs must be complete native finite float32 B1/head1"
        )
    return logits[0, 0].detach().cpu()


def run(output, *, deadline):
    output = Path(output)
    # This deadline only stops model calls. The caller must separately install
    # and verify the paid machine's exact-ID automatic deletion guard.
    registration = json.loads((output / "REGISTRATION.json").read_text())
    behavior, _, _ = modules(registration["behavior_source_root"])
    behavior.validate_deadline(deadline)
    behavior, planner, hooks, old, tokens, registration, inputs, planned = check(output)
    root = output / "screen"
    root.mkdir()
    prefixes, writer = {}, None
    counts = {
        "native_attempts": 0,
        "native_saved": 0,
        "prefix_attempts": 0,
        "prefix_saved": 0,
        "raw_bytes": 0,
    }
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
            parameter.requires_grad or parameter.dtype != torch.float32
            for parameter in model.parameters()
        ) or any(module.training for module in model.modules()):
            raise ValueError("screen model must be frozen float32 evaluation mode")
        for job in planned["prefix_capture_jobs"]:
            old.check_deadline(deadline)
            counts["prefix_attempts"] += 1
            prefixes[job["endpoint_id"]] = hooks.capture_prefix(
                model, job["input_ids"], job["alias_end_position"], job["layers"]
            )
            counts["prefix_saved"] += 1
        prefix_bytes = validate_prefixes(prefixes, planned)
        old.save_pt(root / "PREFIXES.pt", prefixes)
        writer = old.RawWriter(root)
        rows = {row["id"]: row for row in inputs["rows"]}
        original_root = Path(registration["behavior_root"]) / "qwen"
        original_reader = old.RawReader(
            original_root, old.read_json(original_root / "RAW_INDEX.json")
        )

        def execute(jobs, *, clean=False, oracle=False):
            comparisons = []
            for job in jobs:
                old.check_deadline(deadline)
                if (
                    counts["raw_bytes"] + tokens["vocab_size"] * 4 + prefix_bytes
                    > RAW_CAP - METADATA_ALLOWANCE
                ):
                    raise RuntimeError(
                        "screen native storage ceiling would be exceeded"
                    )
                context = nullcontext()
                if "layer_index" in job:
                    vector = prefixes[
                        f"capture::{job['case_id']}::{job['donor_role']}"
                    ]["layers"][job["layer_index"]]
                    if job["head_index"] is not None:
                        start = job["head_index"] * planned["model"]["head_dim"]
                        vector = vector[start : start + planned["model"]["head_dim"]]
                    case = next(
                        case
                        for case in planned["cases"]
                        if case["case_id"] == job["case_id"]
                    )
                    context = hooks.replace_at_marker(
                        model,
                        job["layer_index"],
                        case["alias_end_position"],
                        vector,
                        head_index=job["head_index"],
                    )
                counts["native_attempts"] += 1
                with context:
                    native = native_forward(
                        model, rows[job["row_id"]], tokens["vocab_size"]
                    )
                writer.add(job["endpoint_id"], native)
                counts["native_saved"] += 1
                counts["raw_bytes"] += native.numel() * 4
                if clean and not torch.equal(
                    native, original_reader.read(job["row_id"] + "::step00")
                ):
                    raise ValueError(
                        "clean replay differs from bitwise original native logits"
                    )
                if oracle:
                    # Clean rows were flushed before any self-oracle execution.
                    current = old.RawReader(root, writer.index)
                    comparisons.append(
                        oracle_difference(
                            native, current.read(f"clean::{job['case_id']}")
                        )
                    )
            writer.flush()
            print(
                json.dumps(
                    {
                        "native_saved": counts["native_saved"],
                        "prefix_saved": counts["prefix_saved"],
                    }
                ),
                flush=True,
            )
            return all(item["passed"] for item in comparisons)

        execute(planned["clean_jobs"], clean=True)
        if execute(planned["layer_self_jobs"], oracle=True):
            execute(planned["layer_jobs"])
            scores = measure_jobs(
                old.RawReader(root, writer.index),
                planned["layer_jobs"],
                planned,
                inputs,
            )
            refined = planner.select_layers(planned, scores)
            if refined["selected_layers"] and execute(
                refined["head_self_jobs"], oracle=True
            ):
                execute(refined["head_jobs"])
        writer.finish()
        result = evaluate_screen(
            old.RawReader(root, old.read_json(root / "RAW_INDEX.json")),
            original_reader,
            planned,
            inputs,
        )
        if (
            result["native_forward_count"] != counts["native_saved"]
            or result["backbone_prefix_forward_count"] != counts["prefix_saved"]
        ):
            raise ValueError(
                "physical forward counts differ from complete saved endpoints"
            )
        old.seal_json(root / "RESULT.json", result)
        old.seal_json(root / "FORWARD_COUNTS.json", counts)
        check(output)
        if (
            sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
            > RAW_CAP - 1024**2
        ):
            raise RuntimeError(
                "combined screening evidence exceeded its one-GB ceiling"
            )
        artifacts = {
            path.name: old.sha256(path)
            for path in root.iterdir()
            if path.is_file() and not path.name.endswith(".sha256")
        }
        old.seal_json(
            root / "CUSTODY.json",
            {
                "registration_sha256": old.verify_file(output / "REGISTRATION.json"),
                "artifact_sha256": artifacts,
                "physical_counts": {
                    "native": counts["native_saved"],
                    "backbone_prefix": counts["prefix_saved"],
                },
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
        if prefixes and not (root / "PREFIXES.pt").exists():
            old.save_pt(root / "PARTIAL_PREFIXES.pt", prefixes)
        old.seal_json(
            root / "failure.json",
            {
                "phase": "screen",
                "traceback": trace,
                "flush_error": flush_error,
                "counts": counts,
            },
        )
        raise


def rescore(output):
    output = Path(output)
    if (
        sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
        > RAW_CAP
    ):
        raise ValueError(
            "combined screening evidence exceeds the registered one-GB ceiling"
        )
    _, _, _, old, tokens, registration, inputs, planned = check(output)
    root = output / "screen"
    old.verify_file(root / "CUSTODY.json")
    custody = old.read_json(root / "CUSTODY.json")
    if custody["registration_sha256"] != old.verify_file(output / "REGISTRATION.json"):
        raise ValueError("screen custody is bound to a different registration")
    for name, digest in custody["artifact_sha256"].items():
        if Path(name).name != name:
            raise ValueError("screen custody escaped its namespace")
        old.verify_file(root / name, digest)
    prefixes = torch.load(root / "PREFIXES.pt", map_location="cpu", weights_only=True)
    validate_prefixes(prefixes, planned)
    reader = old.RawReader(root, old.read_json(root / "RAW_INDEX.json"))
    for identity in reader.index["endpoints"]:
        native = reader.read(identity)
        if (
            native.dtype != torch.float32
            or tuple(native.shape) != (tokens["vocab_size"],)
            or not bool(torch.isfinite(native).all())
        ):
            raise ValueError("saved screen endpoint is not complete native float32")
    original_root = Path(registration["behavior_root"]) / "qwen"
    original_reader = old.RawReader(
        original_root, old.read_json(original_root / "RAW_INDEX.json")
    )
    result = evaluate_screen(reader, original_reader, planned, inputs)
    counts = old.read_json(root / "FORWARD_COUNTS.json")
    expected_counts = {
        "native": result["native_forward_count"],
        "backbone_prefix": result["backbone_prefix_forward_count"],
    }
    if (
        result != old.read_json(root / "RESULT.json")
        or custody["physical_counts"] != expected_counts
        or counts["native_attempts"] != counts["native_saved"]
        or counts["native_saved"] != expected_counts["native"]
        or counts["prefix_attempts"] != counts["prefix_saved"]
        or counts["prefix_saved"] != expected_counts["backbone_prefix"]
        or counts["raw_bytes"] != expected_counts["native"] * tokens["vocab_size"] * 4
    ):
        raise ValueError("exact native rescore, selection or forward counts differ")
    return {
        "exact_match": True,
        "status": result["status"],
        "physical_counts": expected_counts,
        "selected_layers": result["selected_layers"],
        "selected_heads": result["selected_heads"],
        "confirmation_calls": 0,
        "complete_answer_repair_established": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "check", "rescore"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--behavior-root", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--verified-rescore-receipt", type=Path)
    parser.add_argument("--deadline")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(
            args.output,
            behavior_root=args.behavior_root,
            source_root=args.source_root,
            receipt_path=args.verified_rescore_receipt,
        )
    elif args.command == "run":
        result = run(args.output, deadline=args.deadline)
    elif args.command == "check":
        check(args.output)
        result = {"registration_valid": True}
    else:
        result = rescore(args.output)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
