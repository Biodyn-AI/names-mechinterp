"""New independent evaluation of fixed behavioral and repair claims.

Archived development plans remain immutable and failed. This runner never
selects another head, changes a strength, provisions resources, or resumes a
failed namespace. Resource approval and an exact-machine deletion guard belong
to the caller, separately from the model-call deadline checked here.
"""

import argparse
import importlib
import importlib.metadata
import json
import re
import shutil
import sys
import traceback
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SOURCE_DIRS = (
    "experiments/error_reference_confirmation_20261116",
    "experiments/error_reference_study_20261114",
    "experiments/error_reference_validation_20261115",
    "experiments/error_reference_routing_20261115",
    "experiments/e30_14_work",
)
FIELDS = ("color", "shape", "home")
CONDITIONS = ("selfRfield", "primaryDcolor", "frozenDiscoveryMean", "Rcolor")
MAX_TOKENS = 8


def module_at(name, directory):
    """Refuse Python's cached module if it belongs to a different source tree."""
    directory = Path(directory).resolve()
    sys.path.insert(0, str(directory))
    module = importlib.import_module(name)
    if Path(module.__file__).resolve().parent != directory:
        raise ValueError(f"{name} escaped the declared source tree")
    return module


def modules(repo=REPO, *, repair=False):
    repo = Path(repo).resolve()
    plan = module_at("confirmation_plan", repo / SOURCE_DIRS[0])
    behavior = module_at("error_run", repo / SOURCE_DIRS[1])
    old = behavior.helpers(repo / SOURCE_DIRS[-1])
    if not repair:
        return plan, behavior, old
    routing = module_at("routing_run", repo / SOURCE_DIRS[3])
    mass = module_at("record_routing_hooks", repo / SOURCE_DIRS[3])
    query = module_at("query_delta_hooks", repo / SOURCE_DIRS[2])
    geometry = module_at("validation_plan", repo / SOURCE_DIRS[2])
    return plan, behavior, old, routing, mass, query, geometry


def confirmation_grid(rows, worlds):
    if (
        len(rows) != 6144
        or len({r["id"] for r in rows}) != 6144
        or len(worlds) != 64
        or len({w["world_id"] for w in worlds}) != 64
        or any(r["split"] != "confirmation" for r in rows)
        or any(w["split"] != "confirmation" for w in worlds)
        or set(Counter(r["world_id"] for r in rows).values()) != {96}
        or {r["world_id"] for r in rows} != {w["world_id"] for w in worlds}
    ):
        raise ValueError("exactly the complete 64-world confirmation grid is required")
    return rows


def reattach_tokens(archived, current):
    """Retain every archived token; replace only machine-specific cache paths."""
    if any(
        archived[key] != current[key]
        for key in ("asset_sha256", "vocab_size", "eos_token_ids")
    ):
        raise ValueError("runtime token assets differ from the archived input")
    return {
        **archived,
        "snapshot": current["snapshot"],
        "cache_root": current["cache_root"],
    }


def behavior_coverage(rows, predictions):
    if (
        len(predictions) != len(rows)
        or len({p["id"] for p in predictions}) != len(rows)
        or {p["id"] for p in predictions} != {r["id"] for r in rows}
        or any(p.get("row_id") != p["id"] for p in predictions)
    ):
        raise ValueError(
            "every behavioral generation must bind its own literal row exactly once"
        )


def source_manifest(old, repo):
    paths = [p for directory in SOURCE_DIRS for p in (repo / directory).glob("*.py")]
    paths.append(repo / SOURCE_DIRS[-1] / "us_state_facts_2020.json")
    paths.append(repo / SOURCE_DIRS[0] / "PLAN.json")
    return {str(path.relative_to(repo)): old.sha256(path) for path in sorted(paths)}


def prepare(output, *, cache_root, protocol_commit, repo=REPO):
    if not re.fullmatch(r"[0-9a-f]{40}", protocol_commit):
        raise ValueError("the publicly frozen protocol commit is required")
    repo, output = Path(repo).resolve(), Path(output)
    plan, behavior, old = modules(repo)
    output.mkdir()
    try:
        inputs, frozen = plan.confirmation_inputs(repo), plan.build_plan()
        if old.read_json(repo / SOURCE_DIRS[0] / "PLAN.json") != frozen:
            raise ValueError(
                "public plan snapshot differs from the executable protocol"
            )
        confirmation_grid(inputs["rows"], inputs["worlds"])
        closure = source_manifest(old, repo)
        for name in closure:
            target = output / "sources" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            old.write_bytes(target, (repo / name).read_bytes())
        records = {
            "PLAN.json": frozen,
            "ROWS.json": inputs["rows"],
            "WORLDS.json": inputs["worlds"],
            "CODE.json": closure,
        }
        for name, value in records.items():
            old.seal_json(output / name, value)
        for model_name, info in behavior.MODELS.items():
            current, _ = behavior.token_assets(old, info, cache_root)
            records[f"TOKENS-{model_name}.json"] = reattach_tokens(
                inputs["tokens"][model_name], current
            )
            old.seal_json(
                output / f"TOKENS-{model_name}.json",
                records[f"TOKENS-{model_name}.json"],
            )
        if source_manifest(old, repo) != closure:
            raise ValueError("source changed during preparation")
        registration = {
            "repo_root": str(repo),
            "protocol_commit": protocol_commit,
            "input_sha256": {name: old.verify_file(output / name) for name in records},
            "historical_input_sha256": inputs["input_sha256"],
            "runtime_python": str(Path(sys.executable).absolute()),
            "python_version": sys.version,
            "runtime_versions": {
                name: importlib.metadata.version(name) for name in behavior.PACKAGES
            },
            "scope": "New independent evaluation; archived failed mechanism gates remain failed.",
        }
        old.seal_json(output / "REGISTRATION.json", registration)
        return registration
    except BaseException:
        old.seal_json(
            output / "failure.json",
            {"phase": "prepare", "traceback": traceback.format_exc()},
        )
        raise


def check(output):
    output = Path(output)
    registration = json.loads((output / "REGISTRATION.json").read_text())
    repo = Path(registration["repo_root"])
    plan, behavior, old = modules(repo)
    old.verify_file(output / "REGISTRATION.json")
    for name, digest in registration["input_sha256"].items():
        if Path(name).name != name:
            raise ValueError("registered artifact escaped the output namespace")
        old.verify_file(output / name, digest)
    closure, inputs = (
        old.read_json(output / "CODE.json"),
        plan.confirmation_inputs(repo),
    )
    if (
        closure != source_manifest(old, repo)
        or old.read_json(output / "PLAN.json") != plan.build_plan()
    ):
        raise ValueError("frozen protocol or source closure changed")
    if inputs["input_sha256"] != registration["historical_input_sha256"]:
        raise ValueError("historical input provenance changed")
    for name, digest in closure.items():
        old.verify_file(output / "sources" / name, digest)
    for name, key in (("ROWS.json", "rows"), ("WORLDS.json", "worlds")):
        if old.read_json(output / name) != inputs[key]:
            raise ValueError("confirmation facts or questions changed")
    confirmation_grid(inputs["rows"], inputs["worlds"])
    for model_name in behavior.MODELS:
        archived, tokens = (
            inputs["tokens"][model_name],
            old.read_json(output / f"TOKENS-{model_name}.json"),
        )
        if {k: v for k, v in tokens.items() if k not in ("snapshot", "cache_root")} != {
            k: v for k, v in archived.items() if k not in ("snapshot", "cache_root")
        }:
            raise ValueError("archived token rows or metadata changed")
    return (
        plan,
        behavior,
        old,
        registration,
        inputs,
        old.read_json(output / "PLAN.json"),
    )


def native_forward(model, ids, vocab_size):
    import torch

    tensor = torch.tensor(
        [ids], dtype=torch.long, device=next(model.parameters()).device
    )
    with torch.inference_mode():
        logits = model(
            input_ids=tensor,
            attention_mask=torch.ones_like(tensor),
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
            "a complete finite native float32 output vocabulary is required"
        )
    return logits[0, 0].detach().cpu()


def checked_forward(
    model, ids, writer, identity, *, vocab_size, deadline, raw_cap, counter
):
    if datetime.now(UTC) >= datetime.fromisoformat(deadline):
        raise RuntimeError("execution deadline has passed")
    if counter["bytes"] + vocab_size * 4 > raw_cap:
        raise RuntimeError("raw native-logit ceiling exceeded")
    native = native_forward(model, ids, vocab_size)
    writer.add(identity, native)
    counter["bytes"] += native.numel() * 4
    return native


def generate(
    model,
    tokenizer,
    row,
    identity,
    writer,
    *,
    behavior,
    vocab_size,
    deadline,
    raw_cap,
    counter,
):
    prediction = {
        "id": identity,
        "row_id": row["id"],
        "expected_answer": row["expected_answer"],
        "generated_ids": [],
        "generated_text": "",
        "step_endpoint_ids": [],
        "terminated": False,
        "truncated": False,
        "stop_reason": None,
    }
    eos = behavior.eos_ids(model, tokenizer)
    for step in range(MAX_TOKENS):
        endpoint = identity + f"::step{step:02d}"
        native = checked_forward(
            model,
            row["input_ids"] + prediction["generated_ids"],
            writer,
            endpoint,
            vocab_size=vocab_size,
            deadline=deadline,
            raw_cap=raw_cap,
            counter=counter,
        )
        token = int(native.argmax())
        prediction["generated_ids"].append(token)
        prediction["step_endpoint_ids"].append(endpoint)
        prediction["generated_text"] = behavior.decode_answer(
            tokenizer, prediction["generated_ids"], eos
        )
        if token in eos:
            prediction.update(terminated=True, stop_reason="eos")
            return prediction
    prediction.update(truncated=True, stop_reason="max_tokens")
    return prediction


def audit_predictions(
    rows,
    predictions,
    reader,
    tokenizer,
    eos,
    decode,
    *,
    vocab_size,
    additional_endpoints=(),
):
    """CPU-only reconstruction of every greedy choice and complete decoded answer."""
    import torch

    torch.set_num_threads(1)
    lookup, seen, endpoints = (
        {r["id"]: r for r in rows},
        set(),
        set(additional_endpoints),
    )
    for prediction in predictions:
        identity, row_id = prediction["id"], prediction.get("row_id", prediction["id"])
        if identity in seen or row_id not in lookup:
            raise ValueError("duplicate generation or unknown literal row")
        seen.add(identity)
        row, ids, keys = (
            lookup[row_id],
            prediction["generated_ids"],
            prediction["step_endpoint_ids"],
        )
        if not 1 <= len(ids) == len(keys) <= MAX_TOKENS or keys != [
            identity + f"::step{i:02d}" for i in range(len(ids))
        ]:
            raise ValueError("generation endpoint coverage changed")
        if prediction["expected_answer"] != row["expected_answer"]:
            raise ValueError("saved expected answer changed")
        for token, endpoint in zip(ids, keys):
            native = reader.read(endpoint)
            if (
                native.dtype != torch.float32
                or tuple(native.shape) != (vocab_size,)
                or not bool(torch.isfinite(native).all())
                or int(native.argmax()) != token
            ):
                raise ValueError(
                    "saved token is not the native full-vocabulary greedy choice"
                )
            if endpoint in endpoints:
                raise ValueError("endpoint reused")
            endpoints.add(endpoint)
        eos_steps = [i for i, token in enumerate(ids) if token in eos]
        terminated = bool(eos_steps)
        if (
            (eos_steps and eos_steps != [len(ids) - 1])
            or type(prediction["terminated"]) is not bool
            or type(prediction["truncated"]) is not bool
            or prediction["terminated"] != terminated
            or prediction["truncated"] != (not terminated)
            or prediction["stop_reason"] != ("eos" if terminated else "max_tokens")
            or (not terminated and len(ids) != MAX_TOKENS)
            or prediction["generated_text"] != decode(tokenizer, ids, eos)
        ):
            raise ValueError("whole-answer termination or decoding changed")
    if set(reader.index["endpoints"]) != endpoints:
        raise ValueError("raw native endpoint coverage changed")
    return {
        "native_steps": sum(len(p["generated_ids"]) for p in predictions),
        "generations": len(predictions),
        "exact_match": True,
    }


def seal_custody(old, root, registration_digest):
    old.seal_json(
        root / "CUSTODY.json",
        {
            "registration_sha256": registration_digest,
            "artifact_sha256": {
                p.name: old.sha256(p)
                for p in sorted(root.iterdir())
                if p.is_file() and not p.name.endswith(".sha256")
            },
        },
    )


def verify_custody(old, output, root):
    old.verify_file(root / "CUSTODY.json")
    custody = old.read_json(root / "CUSTODY.json")
    if custody["registration_sha256"] != old.verify_file(output / "REGISTRATION.json"):
        raise ValueError("custody refers to another registration")
    for name, digest in custody["artifact_sha256"].items():
        if Path(name).name != name:
            raise ValueError("custody artifact escaped its namespace")
        old.verify_file(root / name, digest)
    return custody


def load_runtime(output, model_name, root, registration, behavior, old):
    import torch

    old.seal_json(root / "RUNTIME.json", old.configure_runtime(registration))
    tokens = behavior.checked_tokens(old, output, model_name)
    model = behavior.load_model(old, behavior.MODELS[model_name], tokens, root)
    if any(
        p.requires_grad or p.dtype != torch.float32 for p in model.parameters()
    ) or any(m.training for m in model.modules()):
        raise ValueError("a frozen float32 evaluation-mode model is required")
    tokenizer = old.load_tokenizer(tokens["snapshot"])
    if behavior.eos_ids(model, tokenizer) != set(tokens["eos_token_ids"]):
        raise ValueError("runtime termination tokens differ from registration")
    return model, tokenizer, tokens


def failed(old, root, writer):
    trace, flush_error = traceback.format_exc(), None
    if writer is not None and not writer.closed:
        try:
            writer.finish()
        except (OSError, RuntimeError, ValueError) as error:
            flush_error = str(error)
    old.seal_json(
        root / "failure.json", {"traceback": trace, "flush_error": flush_error}
    )


def run_behavior(output, model_name, *, deadline):
    output = Path(output)
    plan, behavior, old, registration, inputs, frozen = check(output)
    behavior.validate_deadline(deadline)
    if model_name not in behavior.MODELS:
        raise ValueError("model outside the frozen panel")
    cap = frozen["model_limits_bytes"][model_name]
    if shutil.disk_usage(output).free < cap:
        raise RuntimeError(
            "insufficient disk for the registered worst-case model output"
        )
    root, writer, predictions = output / model_name, None, []
    root.mkdir()
    try:
        old.seal_json(
            root / "EXECUTION.json",
            {
                "deadline_utc": deadline,
                "model": model_name,
                "scope": "all64confirmationworlds",
            },
        )
        model, tokenizer, tokens = load_runtime(
            output, model_name, root, registration, behavior, old
        )
        # Loading can download missing weights into the same persistent drive.
        if shutil.disk_usage(output).free < cap:
            raise RuntimeError(
                "insufficient disk after model loading; no forward authorized"
            )
        rows = confirmation_grid(
            [r for r in tokens["rows"] if r["split"] == "confirmation"],
            inputs["worlds"],
        )
        writer, counter, current_world = old.RawWriter(root), {"bytes": 0}, []
        for index, row in enumerate(rows):
            prediction = generate(
                model,
                tokenizer,
                row,
                row["id"],
                writer,
                behavior=behavior,
                vocab_size=tokens["vocab_size"],
                deadline=deadline,
                raw_cap=cap - 256_000_000,
                counter=counter,
            )
            predictions.append(prediction)
            current_world.append(prediction)
            if (index + 1) % 96 == 0:
                writer.flush()
                old.seal_json(
                    root / f"world-{(index + 1) // 96 - 1:03d}.json", current_world
                )
                current_world = []
                print(
                    json.dumps(
                        {
                            "model": model_name,
                            "completed_rows": index + 1,
                            "required_rows": 6144,
                        }
                    ),
                    flush=True,
                )
        writer.finish()
        old.seal_json(root / "GENERATIONS.json", predictions)
        old.seal_json(
            root / "RESULT.json",
            plan.score_behavior(rows, predictions, inputs["worlds"]),
        )
        if sum(p.stat().st_size for p in root.iterdir() if p.is_file()) > cap:
            raise RuntimeError(
                "combined behavioral artifacts exceeded the storage ceiling"
            )
        check(output)
        seal_custody(old, root, old.verify_file(output / "REGISTRATION.json"))
    except BaseException:
        old.seal_json(root / "PARTIAL_GENERATIONS.json", predictions)
        failed(old, root, writer)
        raise


def audit_behavior(output, model_name):
    output = Path(output)
    plan, behavior, old, _, inputs, _ = check(output)
    root = output / model_name
    custody = verify_custody(old, output, root)
    tokens = behavior.checked_tokens(old, output, model_name)
    rows = confirmation_grid(
        [r for r in tokens["rows"] if r["split"] == "confirmation"], inputs["worlds"]
    )
    predictions = old.read_json(root / "GENERATIONS.json")
    behavior_coverage(rows, predictions)
    audit = audit_predictions(
        rows,
        predictions,
        old.RawReader(root, old.read_json(root / "RAW_INDEX.json")),
        old.load_tokenizer(tokens["snapshot"]),
        set(tokens["eos_token_ids"]),
        behavior.decode_answer,
        vocab_size=tokens["vocab_size"],
    )
    if plan.score_behavior(rows, predictions, inputs["worlds"]) != old.read_json(
        root / "RESULT.json"
    ):
        raise ValueError("saved behavioral analysis differs from CPU reconstruction")
    audit.update(
        custody_sha256=old.verify_file(root / "CUSTODY.json"),
        generation_sha256=custody["artifact_sha256"]["GENERATIONS.json"],
        required_worlds=64,
        required_rows=6144,
    )
    old.seal_json(root / "NATIVE_AUDIT.json", audit)
    return audit


def compact_capture(capture, layer):
    """Retain the selected rows and every record-value hash needed for replay."""
    return {
        **capture,
        "layers": {
            index: {
                key: value
                for key, value in values.items()
                if key != "attention_rows" or index == layer
            }
            for index, values in capture["layers"].items()
        },
    }


def repair_profile(condition, field, captures, frozen, mass):
    head = frozen["selected_head"]
    layer, index = head["layer_index"], head["head_index"]
    recipient = captures[("recipient", field)]
    regions = recipient["regions"]
    # Destination support is mandatory even for the zero-change comparison.
    mass.record_mass_profile(
        recipient["capture"]["layers"][layer]["attention_rows"][index], regions
    )
    if condition == "frozenDiscoveryMean":
        return frozen["frozen_discovery_mean_Dcolor"]
    role, source_field = (
        ("recipient", field)
        if condition == "selfRfield"
        else ("recipient", "color")
        if condition == "Rcolor"
        else ("donor", "color")
    )
    source = captures[(role, source_field)]
    source_profile = mass.record_mass_profile(
        source["capture"]["layers"][layer]["attention_rows"][index],
        source["regions"],
        require_support=False,
    )
    if source_profile["total"] < 1e-4:
        raise ValueError("unsupported source record mass; no replacement case")
    return source_profile["profile"].tolist()


def run_repair(output, *, deadline):
    import torch

    output = Path(output)
    plan, behavior, old, registration, inputs, frozen = check(output)
    _, _, _, routing, mass, query, geometry = modules(
        registration["repo_root"], repair=True
    )
    behavior.validate_deadline(deadline)
    if tuple(frozen["conditions"]) != CONDITIONS:
        raise ValueError("only the four separately frozen comparisons are supported")
    baseline_root = output / "qwen"
    verify_custody(old, output, baseline_root)
    old.verify_file(baseline_root / "NATIVE_AUDIT.json")
    audit = old.read_json(baseline_root / "NATIVE_AUDIT.json")
    if (
        audit.get("exact_match") is not True
        or audit.get("custody_sha256")
        != old.verify_file(baseline_root / "CUSTODY.json")
        or audit.get("required_rows") != 6144
    ):
        raise ValueError("complete independently audited Qwen behavior is required")
    predictions = old.read_json(baseline_root / "GENERATIONS.json")
    cases = plan.select_cases(inputs["rows"], predictions, inputs["worlds"])
    cap = frozen["model_limits_bytes"]["repair"]
    if shutil.disk_usage(output).free < cap:
        raise RuntimeError("insufficient disk for the bounded repair output")
    root, writer, generations = output / "repair", None, []
    root.mkdir()
    try:
        old.seal_json(
            root / "EXECUTION.json",
            {
                "deadline_utc": deadline,
                "baseline_audit_sha256": old.verify_file(
                    baseline_root / "NATIVE_AUDIT.json"
                ),
            },
        )
        old.seal_json(root / "CASES.json", cases)
        model, tokenizer, tokens = load_runtime(
            output, "qwen", root, registration, behavior, old
        )
        if shutil.disk_usage(output).free < cap:
            raise RuntimeError(
                "insufficient disk after model loading; no forward authorized"
            )
        rows = {r["id"]: r for r in tokens["rows"] if r["split"] == "confirmation"}
        baseline = {p["id"]: p for p in predictions}
        original = old.RawReader(
            baseline_root, old.read_json(baseline_root / "RAW_INDEX.json")
        )
        writer, counter = old.RawWriter(root), {"bytes": 0}
        head, natural_ids, operator_ids = frozen["selected_head"], [], []
        for case_number, case in enumerate(cases):
            captures, operators, traces = {}, {}, {}
            for role in ("donor", "recipient"):
                for field in FIELDS:
                    row = rows[case[f"{role}_row_ids"][field]]
                    regions, identity = (
                        geometry.record_regions(row),
                        f"natural::{case['case_id']}::{role}::{field}",
                    )
                    with mass.route_attention(model, regions) as capture:
                        native = checked_forward(
                            model,
                            row["input_ids"],
                            writer,
                            identity,
                            vocab_size=tokens["vocab_size"],
                            deadline=deadline,
                            raw_cap=cap - 300_000_000,
                            counter=counter,
                        )
                    if not torch.equal(native, original.read(row["id"] + "::step00")):
                        raise ValueError(
                            "fresh natural capture differs from audited baseline"
                        )
                    natural_ids.append(identity)
                    captures[(role, field)] = {
                        "capture": compact_capture(capture, head["layer_index"]),
                        "regions": regions,
                        "row_id": row["id"],
                        "endpoint_id": identity,
                    }
            # Freeze and validate every comparison profile before any repair outcome.
            profiles = {
                (condition, field): repair_profile(
                    condition, field, captures, frozen, mass
                )
                for condition in CONDITIONS
                for field in FIELDS
            }
            hashes = [
                {
                    layer: entry["record_value_sha256"]
                    for layer, entry in item["capture"]["layers"].items()
                }
                for item in captures.values()
            ]
            if any(value != hashes[0] for value in hashes[1:]):
                raise ValueError(
                    "unchanged record values differ across name/property contexts"
                )
            for condition in CONDITIONS:
                for field in FIELDS:
                    row = rows[case["recipient_row_ids"][field]]
                    natural = captures[("recipient", field)]
                    identity = f"repair::{case['case_id']}::{condition}::{field}"
                    operator_identity = identity + "::operator"
                    profile = profiles[(condition, field)]
                    with mass.route_attention(
                        model,
                        natural["regions"],
                        intervention={**head, "desired_profile": profile},
                    ) as captured:
                        operator_native = checked_forward(
                            model,
                            row["input_ids"],
                            writer,
                            operator_identity,
                            vocab_size=tokens["vocab_size"],
                            deadline=deadline,
                            raw_cap=cap - 300_000_000,
                            counter=counter,
                        )
                    proxy = {
                        **head,
                        "condition": "selfRscreen"
                        if condition == "selfRfield"
                        else condition,
                        "desired_profile": profile,
                    }
                    routing.replay_operation(
                        proxy, captured, natural["capture"], natural["regions"], mass
                    )
                    metadata = captured["intervention"]
                    with query.add_query_delta(
                        model,
                        head["layer_index"],
                        head["head_index"],
                        len(row["input_ids"]) - 1,
                        metadata["delta_head_output"],
                        metadata["old_head_output"],
                        drift_tolerance=frozen["numeric"]["max_query_head_drift"],
                    ) as trace:
                        prediction = generate(
                            model,
                            tokenizer,
                            row,
                            identity,
                            writer,
                            behavior=behavior,
                            vocab_size=tokens["vocab_size"],
                            deadline=deadline,
                            raw_cap=cap - 300_000_000,
                            counter=counter,
                        )
                    writer.flush()
                    reader = old.RawReader(root, writer.index)
                    if not torch.equal(
                        reader.read(prediction["step_endpoint_ids"][0]), operator_native
                    ):
                        raise ValueError(
                            "operator and additive first-step outputs differ"
                        )
                    if condition == "selfRfield":
                        if (
                            not metadata["self_zero_delta"]
                            or prediction["generated_ids"]
                            != baseline[row["id"]]["generated_ids"]
                        ):
                            raise ValueError(
                                "zero-change complete answer differs from baseline"
                            )
                        for step, endpoint in enumerate(
                            prediction["step_endpoint_ids"]
                        ):
                            if not torch.equal(
                                reader.read(endpoint),
                                original.read(row["id"] + f"::step{step:02d}"),
                            ):
                                raise ValueError(
                                    "zero-change native trajectory differs from baseline"
                                )
                    prediction.update(
                        case_id=case["case_id"],
                        world_id=case["world_id"],
                        condition=condition,
                        field=field,
                        operator_endpoint_id=operator_identity,
                    )
                    generations.append(prediction)
                    operators[identity], traces[identity] = (
                        compact_capture(captured, head["layer_index"]),
                        trace,
                    )
                    operator_ids.append(operator_identity)
            old.save_pt(
                root / f"case-{case_number:03d}.pt",
                {
                    "case": case,
                    "natural": captures,
                    "operators": operators,
                    "traces": traces,
                    "profiles": profiles,
                },
            )
            old.seal_json(root / f"case-{case_number:03d}.json", generations[-12:])
            if (
                sum(p.stat().st_size for p in root.iterdir() if p.is_file())
                + counter["bytes"]
                - sum(p.stat().st_size for p in root.glob("raw-*.safetensors"))
                > cap
            ):
                raise RuntimeError(
                    "combined repair artifacts exceeded the storage ceiling"
                )
            print(
                json.dumps(
                    {
                        "repair_worlds_completed": case_number + 1,
                        "required_eligible_worlds": len(cases),
                    }
                ),
                flush=True,
            )
        writer.finish()
        old.seal_json(root / "GENERATIONS.json", generations)
        old.seal_json(
            root / "ENDPOINTS.json", {"natural": natural_ids, "operators": operator_ids}
        )
        old.seal_json(
            root / "RESULT.json",
            plan.assess_repair(cases, inputs["rows"], predictions, generations),
        )
        if sum(p.stat().st_size for p in root.iterdir() if p.is_file()) > cap:
            raise RuntimeError("combined repair artifacts exceeded the storage ceiling")
        check(output)
        seal_custody(old, root, old.verify_file(output / "REGISTRATION.json"))
    except BaseException:
        old.seal_json(root / "PARTIAL_GENERATIONS.json", generations)
        failed(old, root, writer)
        raise


def audit_repair(output):
    """Reconstruct every saved operation and generation without loading a model."""
    import torch

    output = Path(output)
    plan, behavior, old, registration, inputs, frozen = check(output)
    _, _, _, routing, mass, _, geometry = modules(
        registration["repo_root"], repair=True
    )
    root, baseline_root = output / "repair", output / "qwen"
    verify_custody(old, output, root)
    verify_custody(old, output, baseline_root)
    execution = old.read_json(root / "EXECUTION.json")
    if execution["baseline_audit_sha256"] != old.verify_file(
        baseline_root / "NATIVE_AUDIT.json"
    ):
        raise ValueError("repair execution refers to another baseline native audit")
    tokens = behavior.checked_tokens(old, output, "qwen")
    rows = {r["id"]: r for r in tokens["rows"] if r["split"] == "confirmation"}
    baseline = old.read_json(baseline_root / "GENERATIONS.json")
    cases, generations = (
        old.read_json(root / "CASES.json"),
        old.read_json(root / "GENERATIONS.json"),
    )
    if cases != plan.select_cases(inputs["rows"], baseline, inputs["worlds"]):
        raise ValueError("fresh eligibility or case selection changed")
    endpoints, reader = (
        old.read_json(root / "ENDPOINTS.json"),
        old.RawReader(root, old.read_json(root / "RAW_INDEX.json")),
    )
    expected_natural = [
        f"natural::{case['case_id']}::{role}::{field}"
        for case in cases
        for role in ("donor", "recipient")
        for field in FIELDS
    ]
    expected_operators = [
        f"repair::{case['case_id']}::{condition}::{field}::operator"
        for case in cases
        for condition in CONDITIONS
        for field in FIELDS
    ]
    if endpoints != {"natural": expected_natural, "operators": expected_operators}:
        raise ValueError("natural or operator endpoint grid changed")
    tokenizer = old.load_tokenizer(tokens["snapshot"])
    audit = audit_predictions(
        list(rows.values()),
        generations,
        reader,
        tokenizer,
        set(tokens["eos_token_ids"]),
        behavior.decode_answer,
        vocab_size=tokens["vocab_size"],
        additional_endpoints=endpoints["natural"] + endpoints["operators"],
    )
    original = old.RawReader(
        baseline_root, old.read_json(baseline_root / "RAW_INDEX.json")
    )
    by_id = {p["id"]: p for p in generations}
    head = frozen["selected_head"]
    for number, case in enumerate(cases):
        saved = old.read_json(root / f"case-{number:03d}.json")
        old.verify_file(root / f"case-{number:03d}.pt")
        record = torch.load(
            root / f"case-{number:03d}.pt", map_location="cpu", weights_only=True
        )
        expected_ids = [
            f"repair::{case['case_id']}::{condition}::{field}"
            for condition in CONDITIONS
            for field in FIELDS
        ]
        if (
            record["case"] != case
            or saved != [by_id[identity] for identity in expected_ids]
            or set(record["operators"]) != set(expected_ids)
            or set(record["traces"]) != set(expected_ids)
        ):
            raise ValueError("case artifact coverage changed")
        if set(record["natural"]) != {
            (role, field) for role in ("donor", "recipient") for field in FIELDS
        }:
            raise ValueError("canonical six natural captures are required")
        for (role, field), item in record["natural"].items():
            row_id = case[f"{role}_row_ids"][field]
            if (
                item["row_id"] != row_id
                or item["endpoint_id"] != f"natural::{case['case_id']}::{role}::{field}"
                or item["regions"] != geometry.record_regions(rows[row_id])
                or item["capture"]["installed_source_sha256"] != mass.SOURCE_HASHES
            ):
                raise ValueError(
                    "natural capture differs from its canonical literal input"
                )
            if not torch.equal(
                reader.read(item["endpoint_id"]),
                original.read(item["row_id"] + "::step00"),
            ):
                raise ValueError("natural replay differs from original baseline")
        for prediction in saved:
            identity, condition, field = (
                prediction["id"],
                prediction["condition"],
                prediction["field"],
            )
            natural = record["natural"][("recipient", field)]
            profile = repair_profile(condition, field, record["natural"], frozen, mass)
            captured, trace = record["operators"][identity], record["traces"][identity]
            if (
                record["profiles"][(condition, field)] != profile
                or captured["installed_source_sha256"] != mass.SOURCE_HASHES
            ):
                raise ValueError(
                    "saved repair profile or installed source hashes changed"
                )
            routing.replay_operation(
                {
                    **head,
                    "condition": "selfRscreen"
                    if condition == "selfRfield"
                    else condition,
                    "desired_profile": profile,
                },
                captured,
                natural["capture"],
                natural["regions"],
                mass,
            )
            metadata, observations = captured["intervention"], trace["observations"]
            if (
                not torch.equal(trace["delta"], metadata["delta_head_output"])
                or not torch.equal(
                    trace["reference_natural_head"], metadata["old_head_output"]
                )
                or trace["query_position"]
                != len(rows[prediction["row_id"]]["input_ids"]) - 1
                or len(observations) != len(prediction["generated_ids"])
                or trace["layer_index"] != head["layer_index"]
                or trace["head_index"] != head["head_index"]
                or trace["drift_tolerance"] != frozen["numeric"]["max_query_head_drift"]
                or trace["delta_norm"]
                != float(torch.linalg.vector_norm(trace["delta"].double()))
                or trace["zero_delta"] != bool((trace["delta"] == 0).all())
            ):
                raise ValueError("fixed original-query delta or trace coverage changed")
            for step, observation in enumerate(observations):
                drift = float(
                    (
                        observation["natural_head"].double()
                        - trace["reference_natural_head"].double()
                    )
                    .abs()
                    .max()
                )
                if (
                    observation["natural_head"].dtype != torch.float32
                    or observation["natural_head"].shape
                    != metadata["old_head_output"].shape
                    or drift != observation["max_abs_drift"]
                    or drift > frozen["numeric"]["max_query_head_drift"]
                    or observation["query_position"] != trace["query_position"]
                    or observation["sequence_length"]
                    != trace["query_position"] + 1 + step
                ):
                    raise ValueError("original-query head drift changed")
            if not torch.equal(
                reader.read(prediction["step_endpoint_ids"][0]),
                reader.read(prediction["operator_endpoint_id"]),
            ):
                raise ValueError("first-step operation/addition equivalence changed")
            if condition == "selfRfield":
                if not metadata["self_zero_delta"]:
                    raise ValueError("self comparison has nonzero delta")
                for step, endpoint in enumerate(prediction["step_endpoint_ids"]):
                    if not torch.equal(
                        reader.read(endpoint),
                        original.read(prediction["row_id"] + f"::step{step:02d}"),
                    ):
                        raise ValueError(
                            "self trajectory differs from original baseline"
                        )
    if plan.assess_repair(
        cases, inputs["rows"], baseline, generations
    ) != old.read_json(root / "RESULT.json"):
        raise ValueError("repair assessment differs from CPU reconstruction")
    audit.update(
        custody_sha256=old.verify_file(root / "CUSTODY.json"),
        cases=len(cases),
        original_query_replay=True,
    )
    old.seal_json(root / "NATIVE_AUDIT.json", audit)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    preparation = sub.add_parser("prepare")
    preparation.add_argument("--output", type=Path, required=True)
    preparation.add_argument("--cache-root", required=True)
    preparation.add_argument("--protocol-commit", required=True)
    for command in ("check", "behavior", "audit", "repair", "audit-repair"):
        child = sub.add_parser(command)
        child.add_argument("--output", type=Path, required=True)
        if command in ("behavior", "audit"):
            child.add_argument("--model", choices=("qwen", "phi"), required=True)
        if command in ("behavior", "repair"):
            child.add_argument("--deadline", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(
            args.output,
            cache_root=args.cache_root,
            protocol_commit=args.protocol_commit,
        )
    elif args.command == "check":
        check(args.output)
        result = {"checked": True}
    elif args.command == "behavior":
        result = run_behavior(args.output, args.model, deadline=args.deadline)
    elif args.command == "audit":
        result = audit_behavior(args.output, args.model)
    elif args.command == "repair":
        result = run_repair(args.output, deadline=args.deadline)
    else:
        result = audit_repair(args.output)
    if result is not None:
        print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
