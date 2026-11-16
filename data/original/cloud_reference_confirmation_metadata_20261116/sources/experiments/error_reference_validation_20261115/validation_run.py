"""Six unused development worlds: fixed-query, whole-answer validation only.

These worlds were behavior-screened already, not untouched confirmation data.
Profiles transfer across fields, but each field/condition has its own first-step
delta, held fixed across greedy steps at the ORIGINAL query token. Reverse and
cyclic outputs are counterfactual steering, never factual accuracy. Unsupported
sources stop the entire study; no case/head replacement or automatic next study.
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
CAP = 1_000_000_000
ROUTING_ANCHORS = (
    "registration_sha256",
    "source_manifest_sha256",
    "plan_sha256",
    "selected_plan_sha256",
    "inputs_sha256",
    "custody_sha256",
    "result_sha256",
    "raw_index_sha256",
    "natural_sha256",
    "interventions_sha256",
    "capture_summaries_sha256",
    "counts_sha256",
)


def modules(source_root):
    source_root = Path(source_root).resolve()
    sys.path.insert(0, str(source_root))
    routing = importlib.import_module("routing_run")
    if Path(routing.__file__).resolve().parent != source_root:
        raise ValueError("routing module escaped its frozen source root")
    planner, query_hooks = (
        importlib.import_module("validation_plan"),
        importlib.import_module("query_delta_hooks"),
    )
    if any(
        Path(module.__file__).resolve().parent != HERE
        for module in (planner, query_hooks)
    ):
        raise ValueError("validation module escaped its source root")
    return routing, planner, query_hooks


def verify_routing_receipt(receipt, anchors, result):
    if (
        receipt.get("exact_match") is not True
        or receipt.get("confirmation_calls") != 0
        or receipt.get("assessment") != result["assessment"]
        or not result["assessment"]["passed"]
        or receipt.get("native_forward_count") != result["native_forward_count"]
        or any(receipt.get(key) != value for key, value in anchors.items())
    ):
        raise ValueError(
            "complete, passed and artifact-anchored routing audit required"
        )


def verified_routing(routing_root, source_root, receipt_path):
    routing, planner, query_hooks = modules(source_root)
    (
        screen,
        _,
        mass_hooks,
        behavior,
        old,
        tokens,
        prior_reg,
        routing_inputs,
        _routing_reg,
        _,
    ) = routing.check(routing_root)
    routing_root, receipt_path = Path(routing_root), Path(receipt_path)
    root = routing_root / "routing"
    names = (
        routing_root / "REGISTRATION.json",
        routing_root / "SOURCES.json",
        routing_root / "PLAN.json",
        root / "SELECTED_PLAN.json",
        routing_root / "INPUTS.json",
        root / "CUSTODY.json",
        root / "RESULT.json",
        root / "RAW_INDEX.json",
        root / "NATURAL.pt",
        root / "INTERVENTIONS.pt",
        root / "CAPTURE_SUMMARIES.json",
        root / "COUNTS.json",
    )
    anchors = {key: old.verify_file(path) for key, path in zip(ROUTING_ANCHORS, names)}
    custody = old.read_json(root / "CUSTODY.json")
    for name, digest in custody["artifact_sha256"].items():
        if Path(name).name != name:
            raise ValueError("routing custody escaped its namespace")
        old.verify_file(root / name, digest)
    old.verify_file(receipt_path)
    receipt, result = old.read_json(receipt_path), old.read_json(root / "RESULT.json")
    verify_routing_receipt(receipt, anchors, result)
    behavior_root = Path(prior_reg["behavior_root"])
    inputs = {
        "base_plan": routing_inputs["base_plan"],
        "rows": routing_inputs["rows"],
        "worlds": routing_inputs["worlds"],
        "predictions": old.read_json(behavior_root / "qwen" / "GENERATIONS.json"),
        "behavioral_rescore": old.read_json(prior_reg["verified_rescore_path"]),
        "discovery_selected_plan": old.read_json(root / "SELECTED_PLAN.json"),
        "discovery_result": result,
        "discovery_rescore": receipt,
        "discovery_capture_summaries": old.read_json(root / "CAPTURE_SUMMARIES.json"),
        "provenance": {
            "routing_registration_sha256": anchors["registration_sha256"],
            "routing_custody_sha256": anchors["custody_sha256"],
            "routing_rescore_sha256": old.verify_file(receipt_path),
            "routing_result_sha256": anchors["result_sha256"],
        },
    }
    return (
        routing,
        planner,
        query_hooks,
        screen,
        mass_hooks,
        behavior,
        old,
        tokens,
        prior_reg,
        inputs,
    )


def source_manifest(old, routing_root):
    return {
        "new": {
            p.name: old.sha256(p)
            for p in sorted([*HERE.glob("*.py"), *HERE.glob("*.json")])
        },
        "routing": old.read_json(Path(routing_root) / "SOURCES.json"),
    }


def proxy_capture_plan(planned, inputs):
    rows = {r["id"]: r for r in inputs["rows"]}
    labels = inputs["base_plan"]["label_token_ids"]
    jobs = [
        {
            **job,
            "expected_label_token_ids": labels[job["field"]][
                rows[job["row_id"]]["expected_answer"]
            ],
        }
        for job in planned["natural_jobs"]
    ]
    return {"model": planned["model"], "capture_jobs": jobs}


def prepare(output, *, routing_root, routing_source_root, routing_rescore_receipt):
    routing, _, _ = modules(routing_source_root)
    ancestor = json.loads((Path(routing_root) / "REGISTRATION.json").read_text())
    screen, _, _ = routing.modules(ancestor["screen_source_root"])
    screen_reg = json.loads(
        (Path(ancestor["screen_root"]) / "REGISTRATION.json").read_text()
    )
    behavior, _, _ = screen.modules(screen_reg["behavior_source_root"])
    old = behavior.helpers(
        json.loads(
            (Path(screen_reg["behavior_root"]) / "REGISTRATION.json").read_text()
        )["dependency_root"]
    )
    output = Path(output)
    output.mkdir()
    try:
        routing, planner, _, _, _, behavior, old, _, _, inputs = verified_routing(
            routing_root, routing_source_root, routing_rescore_receipt
        )
        planned, code = planner.plan(inputs), source_manifest(old, routing_root)
        routing.validate_shared_context(proxy_capture_plan(planned, inputs), inputs)
        if (
            planned["storage"]["cap_bytes"] != CAP
            or planned["storage"]["max_native_endpoints"] != 1008
            or len(planned["natural_jobs"]) != 36
            or len(planned["generation_jobs"]) != 108
            or planned["baseline"]["recipient_wrong"] != 8
            or planned["baseline"]["recipient_protected"] != 10
        ):
            raise ValueError(
                "frozen six-world cohort, baseline counts or bounds changed"
            )
        (output / "sources").mkdir()
        for name in code["new"]:
            old.write_bytes(output / "sources" / name, (HERE / name).read_bytes())
        records = {
            "SOURCES.json": code,
            "INPUTS.json": inputs,
            "PLAN.json": planned,
            "ROUTING_RECEIPT.json": inputs["discovery_rescore"],
        }
        for name, value in records.items():
            old.seal_json(output / name, value)
        registration = {
            "routing_root": str(Path(routing_root).resolve()),
            "routing_source_root": str(Path(routing_source_root).resolve()),
            "routing_rescore_receipt": str(Path(routing_rescore_receipt).resolve()),
            "input_sha256": {name: old.verify_file(output / name) for name in records},
            "runtime_python": str(Path(sys.executable).absolute()),
            "python_version": sys.version,
            "runtime_versions": {
                name: importlib.metadata.version(name) for name in behavior.PACKAGES
            },
            "cap_bytes": CAP,
            "stage": "six_behavior_screened_development_worlds_only",
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
        if code != source_manifest(old, routing_root):
            raise ValueError("validation source changed during preparation")
        old.seal_json(output / "REGISTRATION.json", registration)
        return {
            "prepared": True,
            "head": planned["selected_head"],
            "baseline": planned["baseline"],
            "storage": planned["storage"],
        }
    except BaseException:
        old.seal_json(
            output / "failure.json",
            {"phase": "prepare", "traceback": traceback.format_exc()},
        )
        raise


def check(output):
    output = Path(output)
    reg = json.loads((output / "REGISTRATION.json").read_text())
    if (
        set(reg["input_sha256"])
        != {"SOURCES.json", "INPUTS.json", "PLAN.json", "ROUTING_RECEIPT.json"}
        or reg["stage"] != "six_behavior_screened_development_worlds_only"
        or reg["cap_bytes"] != CAP
        or reg["confirmation_calls_authorized"]
    ):
        raise ValueError("frozen validation scope changed")
    context = verified_routing(
        reg["routing_root"], reg["routing_source_root"], reg["routing_rescore_receipt"]
    )
    routing, planner, _, _, _, _, old, _, _, inputs = context
    old.verify_file(output / "REGISTRATION.json")
    for name, digest in reg["input_sha256"].items():
        old.verify_file(output / name, digest)
    planned, code = (
        old.read_json(output / "PLAN.json"),
        old.read_json(output / "SOURCES.json"),
    )
    if (
        inputs != old.read_json(output / "INPUTS.json")
        or planned != planner.plan(inputs)
        or code != source_manifest(old, reg["routing_root"])
    ):
        raise ValueError("validation data, plan or source closure changed")
    routing.validate_shared_context(proxy_capture_plan(planned, inputs), inputs)
    for name, digest in code["new"].items():
        old.verify_file(output / "sources" / name, digest)
    return (*context, reg, planned)


def audit_generation(
    job,
    prediction,
    trace,
    operator,
    reader,
    old_prediction,
    old_reader,
    planned,
    behavior,
    tokenizer,
    *,
    eos,
):
    ids, endpoints = prediction["generated_ids"], prediction["step_endpoint_ids"]
    head, dim = planned["selected_head"], planned["model"]["head_dim"]
    delta, reference = (
        operator["intervention"]["delta_head_output"],
        operator["intervention"]["old_head_output"],
    )
    if (
        not 1 <= len(ids) == len(endpoints) == len(trace["observations"]) <= 8
        or endpoints != job["greedy_endpoint_ids"][: len(ids)]
        or prediction["id"] != job["id"]
        or prediction["task_answer"] != job["task_answer"]
        or prediction["predicted_target_answer"] != job["predicted_answer"]
        or prediction["operator_endpoint_id"] != job["operator_endpoint_id"]
    ):
        raise ValueError(
            "generation identity, target semantics or endpoint coverage changed"
        )
    if (
        trace["layer_index"] != head["layer_index"]
        or trace["head_index"] != head["head_index"]
        or trace["query_position"] != job["original_query_position"]
        or trace["head_dim"] != dim
        or trace["attention_width"] != dim * planned["model"]["heads"]
        or trace["drift_tolerance"] != planned["numeric"]["max_query_head_drift"]
        or not torch.equal(trace["delta"], delta)
        or not torch.equal(trace["reference_natural_head"], reference)
        or trace["delta_norm"] != float(torch.linalg.vector_norm(delta.double()))
        or trace["zero_delta"] != bool((delta == 0).all())
    ):
        raise ValueError("query position, reference or fixed delta changed")
    for step, (token, endpoint, observation) in enumerate(
        zip(ids, endpoints, trace["observations"])
    ):
        native = reader.read(endpoint)
        if (
            native.dtype != torch.float32
            or native.ndim != 1
            or not bool(torch.isfinite(native).all())
            or int(native.argmax()) != token
        ):
            raise ValueError("generated token is not the complete native greedy argmax")
        natural = observation["natural_head"]
        drift = float((natural.double() - reference.double()).abs().max())
        if (
            natural.dtype != torch.float32
            or tuple(natural.shape) != (dim,)
            or observation["sequence_length"]
            != job["original_query_position"] + 1 + step
            or observation["query_position"] != job["original_query_position"]
            or observation["max_abs_drift"] != drift
            or not drift <= planned["numeric"]["max_query_head_drift"]
        ):
            raise ValueError("unmodified original-query head drift or position changed")
        if step == 0 and not torch.equal(
            native, reader.read(job["operator_endpoint_id"])
        ):
            raise ValueError("operator/additive first-step native vectors differ")
        if job["condition"] == "selfRfield" and (
            step >= len(old_prediction["step_endpoint_ids"])
            or not torch.equal(
                native, old_reader.read(old_prediction["step_endpoint_ids"][step])
            )
        ):
            raise ValueError(
                "self whole native trajectory differs from original behavior"
            )
    eos_steps = [i for i, token in enumerate(ids) if token in eos]
    terminated = bool(eos_steps)
    if (
        (eos_steps and eos_steps != [len(ids) - 1])
        or type(prediction["terminated"]) is not bool
        or type(prediction["truncated"]) is not bool
        or prediction["terminated"] != terminated
        or prediction["truncated"] != (not terminated)
        or prediction["stop_reason"] != ("eos" if terminated else "max_tokens")
        or (not terminated and len(ids) != 8)
        or prediction["generated_text"] != behavior.decode_answer(tokenizer, ids, eos)
    ):
        raise ValueError("whole-answer decode, EOS or truncation contract changed")
    if job["condition"] == "selfRfield" and (
        not trace["zero_delta"]
        or any(
            prediction[key] != old_prediction[key]
            for key in (
                "generated_ids",
                "generated_text",
                "terminated",
                "truncated",
                "stop_reason",
            )
        )
    ):
        raise ValueError("self whole answer differs from original behavior")
    return {
        "id": job["id"],
        "first_step_operator_equal": True,
        "self_whole_trajectory_equal": job["condition"] == "selfRfield",
        "max_query_head_drift": max(o["max_abs_drift"] for o in trace["observations"]),
    }


def proxy_operator_job(job, planned):
    return {
        **job,
        **planned["selected_head"],
        "condition": "selfRscreen"
        if job["condition"] == "selfRfield"
        else job["condition"],
    }


def natural_summaries(routing, planned, inputs, raw, reader):
    full = routing.capture_summaries(proxy_capture_plan(planned, inputs), raw, reader)
    layer, head = (
        planned["selected_head"]["layer_index"],
        planned["selected_head"]["head_index"],
    )
    return [
        {**item, "record_masses": item["record_masses"][layer][head]} for item in full
    ]


def evaluate(
    reader,
    original_reader,
    natural,
    operators,
    traces,
    generations,
    planned,
    inputs,
    routing,
    planner,
    mass_hooks,
    behavior,
    tokenizer,
    eos,
):
    torch.set_num_threads(1)
    for job in planned["natural_jobs"]:
        if not torch.equal(
            reader.read(job["endpoint_id"]),
            original_reader.read(job["row_id"] + "::step00"),
        ):
            raise ValueError(
                "natural validation capture is not a bitwise original replay"
            )
    summaries = natural_summaries(routing, planned, inputs, natural, reader)
    selected = planner.select_sources(planned, summaries)
    if len(selected["generation_jobs"]) not in (0, 108):
        raise ValueError("unsupported source groups must stop the entire validation")
    jobs = {job["id"]: job for job in selected["generation_jobs"]}
    predicted = {p["id"]: p for p in generations}
    if (
        len(predicted) != len(generations)
        or set(predicted) != set(jobs)
        or set(operators) != set(jobs)
        or set(traces) != set(jobs)
    ):
        raise ValueError("whole generation, operator or trace coverage changed")
    natural_jobs = {j["endpoint_id"]: j for j in planned["natural_jobs"]}
    old_predictions = {p["id"]: p for p in inputs["predictions"]}
    expected, oracles = {j["endpoint_id"] for j in planned["natural_jobs"]}, []
    for identity, job in jobs.items():
        base = natural_jobs[job["natural_endpoint_id"]]
        recipient = natural[job["natural_endpoint_id"]]
        captured = operators[identity]
        probe = {**base, "endpoint_id": job["operator_endpoint_id"]}
        routing.capture_summaries(
            {"model": planned["model"], "capture_jobs": [probe]},
            {
                probe["endpoint_id"]: {
                    "layers": captured["layers"],
                    "intervention": None,
                }
            },
        )
        routing.replay_operation(
            proxy_operator_job(job, planned),
            captured,
            recipient,
            base["record_regions"],
            mass_hooks,
        )
        oracles.append(
            audit_generation(
                job,
                predicted[identity],
                traces[identity],
                captured,
                reader,
                old_predictions[job["row_id"]],
                original_reader,
                planned,
                behavior,
                tokenizer,
                eos=eos,
            )
        )
        expected.add(job["operator_endpoint_id"])
        expected.update(predicted[identity]["step_endpoint_ids"])
    if set(reader.index["endpoints"]) != expected:
        raise ValueError("native validation endpoint coverage changed")
    assessment = planner.assess(selected, generations)
    result = {
        "status": "completed_six_world_validation"
        if jobs
        else "stopped_unsupported_sources",
        "selected_head": planned["selected_head"],
        "baseline": planned["baseline"],
        "assessment": assessment,
        "oracles": oracles,
        "native_forward_count": len(expected),
        "natural_count": 36,
        "operator_count": len(jobs),
        "greedy_step_count": sum(len(p["generated_ids"]) for p in generations),
        "unsupported_conditions": selected["unsupported_conditions"],
        "confirmation_calls": 0,
        "further_studies_authorized": False,
        "scope": planned["scope"],
        "counterfactual_metric_warning": "Task answers always remain the literal original E. Reverse/cyclic target hits are steering success, not factual accuracy.",
    }
    return summaries, selected, result


def run(output, *, deadline):
    output = Path(output)
    registration = json.loads((output / "REGISTRATION.json").read_text())
    routing, _, _ = modules(registration["routing_source_root"])
    ancestor = json.loads(
        (Path(registration["routing_root"]) / "REGISTRATION.json").read_text()
    )
    screen, _, _ = routing.modules(ancestor["screen_source_root"])
    screen_reg = json.loads(
        (Path(ancestor["screen_root"]) / "REGISTRATION.json").read_text()
    )
    behavior, _, _ = screen.modules(screen_reg["behavior_source_root"])
    behavior.validate_deadline(deadline)
    (
        routing,
        planner,
        query_hooks,
        screen,
        mass_hooks,
        behavior,
        old,
        tokens,
        prior_reg,
        inputs,
        registration,
        planned,
    ) = check(output)
    root = output / "validation"
    root.mkdir()
    natural, operators, traces, generations, writer = {}, {}, {}, [], None
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
            raise ValueError("validation requires frozen FP32 evaluation mode")
        tokenizer, eos = (
            old.load_tokenizer(tokens["snapshot"]),
            set(tokens["eos_token_ids"]),
        )
        if behavior.eos_ids(model, tokenizer) != eos:
            raise ValueError(
                "runtime EOS tokens differ from the registered tokenizer/config union"
            )
        rows = {r["id"]: r for r in inputs["rows"]}
        old_predictions = {p["id"]: p for p in inputs["predictions"]}
        original_root = Path(prior_reg["behavior_root"]) / "qwen"
        original_reader = old.RawReader(
            original_root, old.read_json(original_root / "RAW_INDEX.json")
        )
        writer = old.RawWriter(root)

        def forward(identity, row):
            old.check_deadline(deadline)
            if counts["saved_native"] >= planned["storage"]["max_native_endpoints"]:
                raise RuntimeError("validation native endpoint ceiling exceeded")
            counts["attempted_native"] += 1
            native = screen.native_forward(model, row, tokens["vocab_size"])
            writer.add(identity, native)
            counts["saved_native"] += 1
            return native

        for job in planned["natural_jobs"]:
            with mass_hooks.route_attention(model, job["record_regions"]) as captured:
                native = forward(job["endpoint_id"], rows[job["row_id"]])
            natural[job["endpoint_id"]] = captured
            if not torch.equal(
                native, original_reader.read(job["row_id"] + "::step00")
            ):
                raise ValueError(
                    "natural validation capture is not a bitwise original replay"
                )
        writer.flush()
        old.save_pt(root / "NATURAL.pt", natural)
        current = old.RawReader(root, writer.index)
        summaries = natural_summaries(routing, planned, inputs, natural, current)
        selected = planner.select_sources(planned, summaries)
        if len(selected["generation_jobs"]) not in (0, 108):
            raise ValueError(
                "unsupported source groups must stop the entire validation"
            )
        old.seal_json(root / "NATURAL_SUMMARIES.json", summaries)
        old.seal_json(root / "SOURCES_SELECTED.json", selected)
        natural_jobs = {job["endpoint_id"]: job for job in planned["natural_jobs"]}
        jobs = sorted(
            selected["generation_jobs"],
            key=lambda job: job["condition"] != "selfRfield",
        )
        head = planned["selected_head"]
        for job in jobs:
            row = rows[job["row_id"]]
            if (
                row["expected_answer"] != job["task_answer"]
                or job["original_query_position"] != len(row["input_ids"]) - 1
            ):
                raise ValueError(
                    "literal task answer or original query position changed"
                )
            baseline = natural_jobs[job["natural_endpoint_id"]]
            recipient = natural[job["natural_endpoint_id"]]
            proxy = proxy_operator_job(job, planned)
            profile = routing.effective_profile(
                proxy, recipient, baseline["record_regions"], mass_hooks
            )
            with mass_hooks.route_attention(
                model,
                baseline["record_regions"],
                intervention={**head, "desired_profile": profile},
            ) as captured:
                operator_native = forward(job["operator_endpoint_id"], row)
            operators[job["id"]] = captured
            routing.replay_operation(
                proxy, captured, recipient, baseline["record_regions"], mass_hooks
            )
            delta = captured["intervention"]["delta_head_output"]
            reference = captured["intervention"]["old_head_output"]
            prediction = {
                "id": job["id"],
                "task_answer": job["task_answer"],
                "predicted_target_answer": job["predicted_answer"],
                "generated_ids": [],
                "generated_text": "",
                "step_endpoint_ids": [],
                "terminated": False,
                "truncated": False,
                "stop_reason": None,
                "operator_endpoint_id": job["operator_endpoint_id"],
            }
            generations.append(prediction)
            with query_hooks.add_query_delta(
                model,
                head["layer_index"],
                head["head_index"],
                job["original_query_position"],
                delta,
                reference,
                drift_tolerance=planned["numeric"]["max_query_head_drift"],
            ) as trace:
                traces[job["id"]] = trace
                for step in range(8):
                    native = forward(
                        job["greedy_endpoint_ids"][step],
                        {"input_ids": row["input_ids"] + prediction["generated_ids"]},
                    )
                    token = int(native.argmax())
                    prediction["generated_ids"].append(token)
                    prediction["step_endpoint_ids"].append(
                        job["greedy_endpoint_ids"][step]
                    )
                    prediction["generated_text"] = behavior.decode_answer(
                        tokenizer, prediction["generated_ids"], eos
                    )
                    if step == 0 and not torch.equal(native, operator_native):
                        raise ValueError(
                            "operator/additive first-step native vectors differ"
                        )
                    if job["condition"] == "selfRfield":
                        old_prediction = old_predictions[job["row_id"]]
                        if step >= len(
                            old_prediction["step_endpoint_ids"]
                        ) or not torch.equal(
                            native,
                            original_reader.read(
                                old_prediction["step_endpoint_ids"][step]
                            ),
                        ):
                            raise ValueError(
                                "self whole native trajectory differs from old behavior"
                            )
                    if token in eos:
                        prediction.update(terminated=True, stop_reason="eos")
                        break
                else:
                    prediction.update(truncated=True, stop_reason="max_tokens")
            writer.flush()
            audit_generation(
                job,
                prediction,
                trace,
                captured,
                old.RawReader(root, writer.index),
                old_predictions[job["row_id"]],
                original_reader,
                planned,
                behavior,
                tokenizer,
                eos=eos,
            )
            if len(generations) % 6 == 0:
                print(
                    json.dumps({**counts, "completed_generations": len(generations)}),
                    flush=True,
                )
        old.save_pt(root / "OPERATORS.pt", operators)
        old.save_pt(root / "GREEDY_TRACES.pt", traces)
        old.seal_json(root / "GENERATIONS.json", generations)
        writer.finish()
        summaries, selected, result = evaluate(
            old.RawReader(root, old.read_json(root / "RAW_INDEX.json")),
            original_reader,
            natural,
            operators,
            traces,
            generations,
            planned,
            inputs,
            routing,
            planner,
            mass_hooks,
            behavior,
            tokenizer,
            eos,
        )
        if (
            result["native_forward_count"] != counts["saved_native"]
            or counts["attempted_native"] != counts["saved_native"]
        ):
            raise ValueError("native physical counts differ from saved endpoints")
        old.seal_json(root / "RESULT.json", result)
        old.seal_json(root / "COUNTS.json", counts)
        check(output)
        if (
            sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
            > CAP - 1024**2
        ):
            raise RuntimeError("complete validation evidence exceeds one GB")
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
        for name, values in (
            ("NATURAL", natural),
            ("OPERATORS", operators),
            ("GREEDY_TRACES", traces),
        ):
            if values and not (root / f"{name}.pt").exists():
                old.save_pt(root / f"PARTIAL_{name}.pt", values)
        if not (root / "GENERATIONS.json").exists():
            old.seal_json(root / "PARTIAL_GENERATIONS.json", generations)
        old.seal_json(
            root / "failure.json",
            {
                "phase": "validation",
                "traceback": trace,
                "flush_error": flush_error,
                "counts": counts,
            },
        )
        raise


def rescore(output):
    output = Path(output)
    if sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) > CAP:
        raise ValueError("validation evidence exceeds one GB")
    (
        routing,
        planner,
        _,
        _,
        mass_hooks,
        behavior,
        old,
        tokens,
        prior_reg,
        inputs,
        _registration,
        planned,
    ) = check(output)
    root = output / "validation"
    old.verify_file(root / "CUSTODY.json")
    custody = old.read_json(root / "CUSTODY.json")
    if custody["registration_sha256"] != old.verify_file(output / "REGISTRATION.json"):
        raise ValueError("validation custody changed registration")
    for name, digest in custody["artifact_sha256"].items():
        if Path(name).name != name:
            raise ValueError("validation custody escaped its namespace")
        old.verify_file(root / name, digest)
    natural, operators, traces = (
        torch.load(root / name, map_location="cpu", weights_only=True)
        for name in ("NATURAL.pt", "OPERATORS.pt", "GREEDY_TRACES.pt")
    )
    reader = old.RawReader(root, old.read_json(root / "RAW_INDEX.json"))
    for identity in reader.index["endpoints"]:
        native = reader.read(identity)
        if (
            native.dtype != torch.float32
            or tuple(native.shape) != (tokens["vocab_size"],)
            or not bool(torch.isfinite(native).all())
        ):
            raise ValueError("validation native endpoints must be complete finite FP32")
    original_root = Path(prior_reg["behavior_root"]) / "qwen"
    original_reader = old.RawReader(
        original_root, old.read_json(original_root / "RAW_INDEX.json")
    )
    tokenizer, eos = (
        old.load_tokenizer(tokens["snapshot"]),
        set(tokens["eos_token_ids"]),
    )
    generations = old.read_json(root / "GENERATIONS.json")
    summaries, selected, result = evaluate(
        reader,
        original_reader,
        natural,
        operators,
        traces,
        generations,
        planned,
        inputs,
        routing,
        planner,
        mass_hooks,
        behavior,
        tokenizer,
        eos,
    )
    counts = old.read_json(root / "COUNTS.json")
    if (
        summaries != old.read_json(root / "NATURAL_SUMMARIES.json")
        or selected != old.read_json(root / "SOURCES_SELECTED.json")
        or result != old.read_json(root / "RESULT.json")
        or counts["attempted_native"] != counts["saved_native"]
        or counts["saved_native"] != result["native_forward_count"]
        or custody["native_forward_count"] != counts["saved_native"]
    ):
        raise ValueError(
            "exact whole-answer, native, source, delta or drift rescore differs"
        )
    return {
        "exact_match": True,
        "native_forward_count": counts["saved_native"],
        "status": result["status"],
        "assessment": result["assessment"],
        "confirmation_calls": 0,
        "further_studies_authorized": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "check", "run", "rescore"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--routing-root", type=Path)
    parser.add_argument("--routing-source-root", type=Path)
    parser.add_argument("--routing-rescore-receipt", type=Path)
    parser.add_argument("--deadline")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(
            args.output,
            routing_root=args.routing_root,
            routing_source_root=args.routing_source_root,
            routing_rescore_receipt=args.routing_rescore_receipt,
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
