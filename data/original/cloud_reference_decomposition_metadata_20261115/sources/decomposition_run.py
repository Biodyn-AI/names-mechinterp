"""Finite diagnosis on the SAME six already-opened worlds, fixed layer28/head0.

The old failed validation remains failed. Nine preregistered component swaps do
not establish a new abstract entity representation. All deltas are label-blind,
per-field, fixed at the original query through greedy decoding. Whole-head is
not a performance upper bound: components can cancel. Random norm matches only
the whole-head delta. No confirmation or automatic further study is authorized.
"""

import argparse
import importlib
import importlib.metadata
import json
import sys
import traceback
from pathlib import Path

import decomposition_math as kernels
import torch

HERE = Path(__file__).resolve().parent
if Path(kernels.__file__).resolve().parent != HERE:
    raise ValueError("decomposition kernels escaped the frozen source root")
CAP, MAX_NATIVE = 1_000_000_000, 1332
HEAD = {"layer_index": 28, "head_index": 0}


def foundation(root, source_root):
    """Read/authenticate ancestors once; never require old scientific success."""
    source_root, root = Path(source_root).resolve(), Path(root)
    sys.path.insert(0, str(source_root))
    validation = importlib.import_module("validation_run")
    if Path(validation.__file__).resolve().parent != source_root:
        raise ValueError("validation source escaped its frozen root")
    context = validation.check(root)
    (
        routing,
        _,
        query_hooks,
        screen,
        mass_hooks,
        behavior,
        old,
        tokens,
        prior_reg,
        old_inputs,
        _,
        old_plan,
    ) = context
    stage = root / "validation"
    anchors = {
        "registration": old.verify_file(root / "REGISTRATION.json"),
        "sources": old.verify_file(root / "SOURCES.json"),
        "plan": old.verify_file(root / "PLAN.json"),
        "custody": old.verify_file(stage / "CUSTODY.json"),
    }
    custody = old.read_json(stage / "CUSTODY.json")
    if custody["registration_sha256"] != anchors["registration"]:
        raise ValueError("old validation custody changed registration")
    for name, digest in custody["artifact_sha256"].items():
        if Path(name).name != name:
            raise ValueError("old validation custody escaped its namespace")
        old.verify_file(stage / name, digest)
    result = old.read_json(stage / "RESULT.json")
    if (
        result["status"] != "completed_six_world_validation"
        or result["native_forward_count"] != custody["native_forward_count"]
    ):
        raise ValueError("completed, preserved old validation evidence is required")
    anchors.update(
        {
            name: old.verify_file(stage / name)
            for name in (
                "RESULT.json",
                "RAW_INDEX.json",
                "NATURAL.pt",
                "OPERATORS.pt",
                "GENERATIONS.json",
                "SOURCES_SELECTED.json",
            )
        }
    )
    rows_by_id = {r["id"]: r for r in old_inputs["rows"]}
    row_ids = [job["row_id"] for job in old_plan["natural_jobs"]]
    case_worlds = {c["world_id"] for c in old_plan["cases"]}
    source_profiles = old.read_json(stage / "SOURCES_SELECTED.json")
    inputs = {
        "old_plan": old_plan,
        "rows": [rows_by_id[i] for i in row_ids],
        "worlds": [w for w in old_inputs["worlds"] if w["world_id"] in case_worlds],
        "predictions": [
            p for p in old_inputs["predictions"] if p["id"] in set(row_ids)
        ],
        "old_validation_result": result,
        "old_color_profiles": {
            c["case_id"]: next(
                j["desired_profile"]
                for j in source_profiles["generation_jobs"]
                if j["case_id"] == c["case_id"] and j["condition"] == "primaryDcolor"
            )
            for c in old_plan["cases"]
        },
        "provenance": anchors,
    }
    return (
        validation,
        routing,
        query_hooks,
        screen,
        mass_hooks,
        behavior,
        old,
        tokens,
        prior_reg,
        inputs,
    )


def plan(inputs):
    old = inputs["old_plan"]
    if (
        old["selected_head"] != HEAD
        or len(old["cases"]) != 6
        or len({c["world_id"] for c in old["cases"]}) != 6
        or len(inputs["rows"]) != 36
        or len({r["id"] for r in inputs["rows"]}) != 36
        or old["baseline"]["recipient_wrong"] != 8
        or old["baseline"]["recipient_protected"] != 10
    ):
        raise ValueError("the exact fixed head and same six-world cohort are required")
    rows = {r["id"]: r for r in inputs["rows"]}
    jobs = []
    for case in old["cases"]:
        for field in ("color", "shape", "home"):
            row = rows[case["recipient_row_ids"][field]]
            for condition in kernels.CONDITIONS:
                identity = f"decompose::{case['case_id']}::{field}::{condition}"
                jobs.append(
                    {
                        "id": identity,
                        "case_id": case["case_id"],
                        "world_id": case["world_id"],
                        "field": field,
                        "condition": condition,
                        "row_id": row["id"],
                        "expected_answer": row["expected_answer"],
                        "baseline_correct": field in case["protected_correct_fields"],
                        "original_query_position": len(row["input_ids"]) - 1,
                        "greedy_endpoint_ids": [
                            f"{identity}::step{i:02d}" for i in range(8)
                        ],
                    }
                )
    return {
        "study_id": kernels.SEED,
        "head": HEAD,
        "model": old["model"],
        "cases": old["cases"],
        "natural_jobs": old["natural_jobs"],
        "generation_jobs": jobs,
        "conditions": list(kernels.CONDITIONS),
        "baseline": old["baseline"],
        "numeric": {
            "head_reconstruction_max_abs": 2e-5,
            "query_drift_max_abs": 2e-5,
            "random_full_norm_relative_error": 1e-5,
        },
        "storage": {"cap_bytes": CAP, "max_native_endpoints": MAX_NATIVE},
        "scope": "Same already-opened six development worlds; diagnostic only, no abstract-entity or generalization claim; random controls full-head norm only; full-head is not a performance bound.",
        "confirmation_calls_authorized": False,
    }


def source_manifest(old, root):
    return {
        "new": {p.name: old.sha256(p) for p in sorted(HERE.glob("*.py"))},
        "validation": old.read_json(Path(root) / "SOURCES.json"),
    }


def prepare(output, *, validation_root, validation_source_root):
    context = foundation(validation_root, validation_source_root)
    _validation, routing, _, _, _, behavior, old, _, _, inputs = context
    output = Path(output)
    output.mkdir()
    try:
        planned, code = plan(inputs), source_manifest(old, validation_root)
        routing.validate_shared_context(
            {"capture_jobs": planned["natural_jobs"]}, inputs
        )
        (output / "sources").mkdir()
        for name in code["new"]:
            old.write_bytes(output / "sources" / name, (HERE / name).read_bytes())
        for name, record in (
            ("SOURCES.json", code),
            ("INPUTS.json", inputs),
            ("PLAN.json", planned),
        ):
            old.seal_json(output / name, record)
        actual_prepared_bytes = sum(
            p.stat().st_size for p in output.rglob("*") if p.is_file()
        )
        longest, dim = (
            max(len(r["input_ids"]) for r in inputs["rows"]),
            planned["model"]["head_dim"],
        )
        tensor_bound = (
            36 * (longest * dim * 4 + longest * 16 + dim * 32 + 16384)
            + 18 * (9 * dim * 16 + longest * 96 + 32768)
            + 162 * 8 * (dim * 8 + 4096)
        )
        estimate = (
            actual_prepared_bytes
            + MAX_NATIVE * planned["model"]["vocab_size"] * 4
            + tensor_bound
            + 64 * 1024**2
        )
        if estimate > CAP:
            raise ValueError(
                "worst-case native/tensor/actual prepared storage exceeds one GB"
            )
        registration = {
            "validation_root": str(Path(validation_root).resolve()),
            "validation_source_root": str(Path(validation_source_root).resolve()),
            "input_sha256": {
                name: old.verify_file(output / name)
                for name in ("SOURCES.json", "INPUTS.json", "PLAN.json")
            },
            "runtime_python": str(Path(sys.executable).absolute()),
            "python_version": sys.version,
            "runtime_versions": {
                n: importlib.metadata.version(n) for n in behavior.PACKAGES
            },
            "storage_estimate_bytes": estimate,
            "actual_prepared_bytes": actual_prepared_bytes,
            "stage": "same_six_worlds_fixed_head_finite_decomposition",
            "cap_bytes": CAP,
            "old_validation_success_required": False,
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
        if code != source_manifest(old, validation_root):
            raise ValueError("decomposition source changed during preparation")
        old.seal_json(output / "REGISTRATION.json", registration)
        return {
            "prepared": True,
            "natural_jobs": 36,
            "generations": 162,
            "storage_estimate_bytes": estimate,
            "old_validation_success_required": False,
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
        != {"SOURCES.json", "INPUTS.json", "PLAN.json"}
        or registration["stage"] != "same_six_worlds_fixed_head_finite_decomposition"
        or registration["old_validation_success_required"]
        or registration["confirmation_calls_authorized"]
        or registration["cap_bytes"] != CAP
    ):
        raise ValueError("finite decomposition scope changed")
    context = foundation(
        registration["validation_root"], registration["validation_source_root"]
    )
    _, _, _, _, _, _, old, _, _, inputs = context
    old.verify_file(output / "REGISTRATION.json")
    for name, digest in registration["input_sha256"].items():
        old.verify_file(output / name, digest)
    planned = old.read_json(output / "PLAN.json")
    code = old.read_json(output / "SOURCES.json")
    if (
        inputs != old.read_json(output / "INPUTS.json")
        or planned != plan(inputs)
        or code != source_manifest(old, registration["validation_root"])
    ):
        raise ValueError("old evidence, fixed plan or decomposition source changed")
    for name, digest in code["new"].items():
        old.verify_file(output / "sources" / name, digest)
    return (*context, registration, planned)


def capture(model, job, old_capture, reference, mass_hooks, query_hooks, forward):
    """Two already-tested zero-delta contexts expose actual head/V unchanged."""
    layer, head = HEAD["layer_index"], HEAD["head_index"]
    old_row = old_capture["layers"][layer]["attention_rows"][head]
    self_profile = mass_hooks.record_mass_profile(old_row, job["record_regions"])[
        "profile"
    ].tolist()
    with (
        mass_hooks.route_attention(
            model,
            job["record_regions"],
            intervention={**HEAD, "desired_profile": self_profile},
        ) as raw,
        query_hooks.add_query_delta(
            model,
            layer,
            head,
            job["original_query_position"],
            torch.zeros_like(reference),
            reference,
        ) as trace,
    ):
        native = forward(job["endpoint_id"])
    metadata = raw["intervention"]
    actual = trace["observations"][0]["natural_head"]
    if (
        len(trace["observations"]) != 1
        or not metadata["self_zero_delta"]
        or not torch.equal(actual, metadata["old_head_output"])
        or not torch.equal(actual, metadata["new_head_output"])
        or not bool((metadata["delta_head_output"] == 0).all())
    ):
        raise ValueError("natural capture was not an exact read-only self pass")
    return native, {
        "weights": metadata["old_weights"],
        "values": metadata["value_matrix"],
        "head_output": actual,
        "record_V_sha256": raw["layers"][layer]["record_value_sha256"],
        "record_positions": raw["layers"][layer]["record_positions"],
        "trace": trace,
        "router_head_output": metadata["old_head_output"],
        "router_delta": metadata["delta_head_output"],
    }


def rebuild_kernels(planned, inputs, captures, old_natural):
    jobs = {job["endpoint_id"]: job for job in planned["natural_jobs"]}
    if set(captures) != set(jobs):
        raise ValueError("all36 natural captures are required")
    rows, by_key, hashes = {r["id"]: r for r in inputs["rows"]}, {}, {}
    for identity, item in captures.items():
        job = jobs[identity]
        old_item = old_natural[identity]["layers"][28]
        if (
            not torch.equal(item["weights"], old_item["attention_rows"][0])
            or item["record_V_sha256"] != old_item["record_value_sha256"]
            or item["record_positions"]
            != sorted(p for region in job["record_regions"].values() for p in region)
            or not torch.equal(item["router_head_output"], item["head_output"])
            or not bool((item["router_delta"] == 0).all())
        ):
            raise ValueError(
                "natural selected-head capture did not exactly replay old state"
            )
        trace = item["trace"]
        if (
            not trace["zero_delta"]
            or len(trace["observations"]) != 1
            or trace["query_position"] != job["original_query_position"]
            or trace["observations"][0]["sequence_length"]
            != len(rows[job["row_id"]]["input_ids"])
            or not torch.equal(
                trace["observations"][0]["natural_head"], item["head_output"]
            )
            or not trace["observations"][0]["max_abs_drift"] <= 2e-5
        ):
            raise ValueError(
                "actual o_proj head capture or original-query boundary changed"
            )
        if (
            job["case_id"] in hashes
            and hashes[job["case_id"]] != item["record_V_sha256"]
        ):
            raise ValueError("record V changed across natural field/alias contexts")
        hashes[job["case_id"]] = item["record_V_sha256"]
        by_key[(job["case_id"], job["role"], job["field"])] = (
            item,
            job["record_regions"],
        )
    results = {}
    for case in planned["cases"]:
        cid = case["case_id"]
        color, color_regions = by_key[(cid, "donor", "color")]
        for field in ("color", "shape", "home"):
            recipient, regions = by_key[(cid, "recipient", field)]
            donor, donor_regions = by_key[(cid, "donor", field)]
            if regions != donor_regions or regions != color_regions:
                raise ValueError("all six contexts must use identical physical records")
            results[f"{cid}::{field}"] = kernels.decompose(
                recipient,
                donor,
                color,
                regions,
                case_id=cid,
                field=field,
                old_color_profile=inputs["old_color_profiles"][cid],
            )
    return results


def audit_greedy(
    job,
    prediction,
    trace,
    delta,
    reference,
    reader,
    comparison,
    comparison_reader,
    planned,
    behavior,
    tokenizer,
    eos,
):
    ids, endpoints = prediction["generated_ids"], prediction["step_endpoint_ids"]
    if (
        prediction["id"] != job["id"]
        or prediction["expected_answer"] != job["expected_answer"]
        or not 1 <= len(ids) == len(endpoints) == len(trace["observations"]) <= 8
        or endpoints != job["greedy_endpoint_ids"][: len(ids)]
        or not torch.equal(trace["delta"], delta)
        or not torch.equal(trace["reference_natural_head"], reference)
        or trace["query_position"] != job["original_query_position"]
        or trace["layer_index"] != 28
        or trace["head_index"] != 0
        or trace["head_dim"] != planned["model"]["head_dim"]
        or trace["attention_width"]
        != planned["model"]["heads"] * planned["model"]["head_dim"]
        or trace["drift_tolerance"] != 2e-5
        or trace["delta_norm"] != float(torch.linalg.vector_norm(delta.double()))
        or trace["zero_delta"] != bool((delta == 0).all())
    ):
        raise ValueError(
            "fixed decomposition query, delta, generation or trace changed"
        )
    for step, (token, endpoint, observation) in enumerate(
        zip(ids, endpoints, trace["observations"])
    ):
        native = reader.read(endpoint)
        observed = observation["natural_head"]
        if (
            observed.dtype != torch.float32
            or observed.device.type != "cpu"
            or tuple(observed.shape) != (planned["model"]["head_dim"],)
            or not bool(torch.isfinite(observed).all())
        ):
            raise ValueError(
                "saved natural head observation is not finite native CPU float32"
            )
        drift = float(
            (observation["natural_head"].double() - reference.double()).abs().max()
        )
        if (
            native.dtype != torch.float32
            or tuple(native.shape) != (planned["model"]["vocab_size"],)
            or not bool(torch.isfinite(native).all())
            or int(native.argmax()) != token
            or observation["query_position"] != job["original_query_position"]
            or observation["sequence_length"]
            != job["original_query_position"] + 1 + step
            or observation["max_abs_drift"] != drift
            or not drift <= 2e-5
        ):
            raise ValueError("native argmax, original query or numerical drift changed")
        if comparison is not None and (
            step >= len(comparison["step_endpoint_ids"])
            or not torch.equal(
                native, comparison_reader.read(comparison["step_endpoint_ids"][step])
            )
        ):
            raise ValueError("self or old-color whole native trajectory changed")
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
        raise ValueError("strict whole-answer/EOS decoding changed")
    if comparison is not None and any(
        prediction[key] != comparison[key]
        for key in (
            "generated_ids",
            "generated_text",
            "terminated",
            "truncated",
            "stop_reason",
        )
    ):
        raise ValueError("self or old-color complete answer changed")
    return {
        "id": job["id"],
        "delta_norm": trace["delta_norm"],
        "max_query_head_drift": max(o["max_abs_drift"] for o in trace["observations"]),
        "old_trajectory_replayed": comparison is not None,
    }


def comparison_for(job, old_predictions, old_generations, original_reader, old_reader):
    if job["condition"] == "self":
        return old_predictions[job["row_id"]], original_reader
    if job["condition"] == "oldDcolorMass":
        identity = f"{job['case_id']}::primaryDcolor::{job['field']}"
        return old_generations[identity], old_reader
    return None, None


def evaluate(
    reader,
    original_reader,
    old_reader,
    captures,
    saved_kernels,
    traces,
    generations,
    planned,
    inputs,
    old_natural,
    old_generations,
    behavior,
    tokenizer,
    eos,
    equal,
):
    torch.set_num_threads(1)
    rebuilt = rebuild_kernels(planned, inputs, captures, old_natural)
    if not equal(rebuilt, saved_kernels):
        raise ValueError(
            "saved factor deltas differ from label-blind CPU recomputation"
        )
    for job in planned["natural_jobs"]:
        if not torch.equal(
            reader.read(job["endpoint_id"]),
            original_reader.read(job["row_id"] + "::step00"),
        ):
            raise ValueError("natural complete-vocabulary baseline replay changed")
    by_id = {g["id"]: g for g in generations}
    jobs = {job["id"]: job for job in planned["generation_jobs"]}
    if (
        len(by_id) != len(generations)
        or set(by_id) != set(jobs)
        or set(traces) != set(jobs)
    ):
        raise ValueError("complete162-generation or trace coverage changed")
    natural = {
        (job["case_id"], job["role"], job["field"]): captures[job["endpoint_id"]]
        for job in planned["natural_jobs"]
    }
    old_predictions = {p["id"]: p for p in inputs["predictions"]}
    expected, oracles = {j["endpoint_id"] for j in planned["natural_jobs"]}, []
    for identity, job in jobs.items():
        delta = rebuilt[f"{job['case_id']}::{job['field']}"]["conditions"][
            job["condition"]
        ]["delta"]
        reference = natural[(job["case_id"], "recipient", job["field"])]["head_output"]
        comparison, compared_reader = comparison_for(
            job, old_predictions, old_generations, original_reader, old_reader
        )
        oracles.append(
            audit_greedy(
                job,
                by_id[identity],
                traces[identity],
                delta,
                reference,
                reader,
                comparison,
                compared_reader,
                planned,
                behavior,
                tokenizer,
                eos,
            )
        )
        expected.update(by_id[identity]["step_endpoint_ids"])
    if set(reader.index["endpoints"]) != expected:
        raise ValueError("full native endpoint coverage differs from the finite plan")
    grader = importlib.import_module("decomposition_grade")
    if Path(grader.__file__).resolve().parent != HERE:
        raise ValueError("decomposition grader escaped the frozen source root")
    if grader.CONDITIONS != kernels.CONDITIONS:
        raise ValueError("grader and kernel condition sets differ")
    return {
        "status": "completed_fixed_head_decomposition",
        "assessment": grader.grade(planned, generations),
        "native_forward_count": len(expected),
        "natural_count": 36,
        "generations": 162,
        "oracles": oracles,
        "old_validation_result_preserved": inputs["old_validation_result"],
        "confirmation_calls": 0,
        "automatic_followup_authorized": False,
        "scope": planned["scope"],
    }


def run(output, *, deadline):
    output = Path(output)
    registration = json.loads((output / "REGISTRATION.json").read_text())
    sys.path.insert(0, registration["validation_source_root"])
    validation_module = importlib.import_module("validation_run")
    # Frozen validation loader exposes the behavioral deadline validator without
    # provisioning resources; authenticate all ancestors once below.
    old_config = json.loads(
        (Path(registration["validation_root"]) / "REGISTRATION.json").read_text()
    )
    routing, _, _ = validation_module.modules(old_config["routing_source_root"])
    route_config = json.loads(
        (Path(old_config["routing_root"]) / "REGISTRATION.json").read_text()
    )
    screen, _, _ = routing.modules(route_config["screen_source_root"])
    screen_config = json.loads(
        (Path(route_config["screen_root"]) / "REGISTRATION.json").read_text()
    )
    behavior, _, _ = screen.modules(screen_config["behavior_source_root"])
    behavior.validate_deadline(deadline)
    (
        _validation,
        routing,
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
    root = output / "decomposition"
    root.mkdir()
    captures, computed, traces, generations, writer = {}, {}, {}, [], None
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
        tokenizer, eos = (
            old.load_tokenizer(tokens["snapshot"]),
            set(tokens["eos_token_ids"]),
        )
        if behavior.eos_ids(model, tokenizer) != eos:
            raise ValueError("runtime termination tokens changed")
        rows = {r["id"]: r for r in inputs["rows"]}
        original_root = Path(prior_reg["behavior_root"]) / "qwen"
        original_reader = old.RawReader(
            original_root, old.read_json(original_root / "RAW_INDEX.json")
        )
        old_root = Path(registration["validation_root"]) / "validation"
        old_reader = old.RawReader(old_root, old.read_json(old_root / "RAW_INDEX.json"))
        old_natural = torch.load(
            old_root / "NATURAL.pt", map_location="cpu", weights_only=True
        )
        old_operators = torch.load(
            old_root / "OPERATORS.pt", map_location="cpu", weights_only=True
        )
        old_jobs = old.read_json(old_root / "SOURCES_SELECTED.json")["generation_jobs"]
        old_generations = {
            g["id"]: g for g in old.read_json(old_root / "GENERATIONS.json")
        }
        old_references = {}
        for job in old_jobs:
            reference = old_operators[job["id"]]["intervention"]["old_head_output"]
            if job["row_id"] in old_references and not torch.equal(
                old_references[job["row_id"]], reference
            ):
                raise ValueError(
                    "old natural head references disagree across conditions"
                )
            old_references[job["row_id"]] = reference
        writer = old.RawWriter(root)

        def forward(identity, row):
            old.check_deadline(deadline)
            if counts["saved_native"] >= MAX_NATIVE:
                raise RuntimeError("finite native endpoint bound exceeded")
            counts["attempted_native"] += 1
            native = screen.native_forward(model, row, tokens["vocab_size"])
            writer.add(identity, native)
            counts["saved_native"] += 1
            return native

        for job in planned["natural_jobs"]:
            native, captures[job["endpoint_id"]] = capture(
                model,
                job,
                old_natural[job["endpoint_id"]],
                old_references[job["row_id"]],
                mass_hooks,
                query_hooks,
                lambda identity, row=rows[job["row_id"]]: forward(identity, row),
            )
            if not torch.equal(
                native, original_reader.read(job["row_id"] + "::step00")
            ):
                raise ValueError("natural baseline native replay changed")
        writer.flush()
        old.save_pt(root / "NATURAL.pt", captures)
        computed = rebuild_kernels(planned, inputs, captures, old_natural)
        old.save_pt(root / "KERNELS.pt", computed)
        natural = {
            (job["case_id"], job["role"], job["field"]): captures[job["endpoint_id"]]
            for job in planned["natural_jobs"]
        }
        old_predictions = {p["id"]: p for p in inputs["predictions"]}
        jobs = sorted(
            planned["generation_jobs"],
            key=lambda job: (
                job["condition"] != "self",
                job["condition"] != "oldDcolorMass",
            ),
        )
        for job in jobs:
            row = rows[job["row_id"]]
            delta = computed[f"{job['case_id']}::{job['field']}"]["conditions"][
                job["condition"]
            ]["delta"]
            reference = natural[(job["case_id"], "recipient", job["field"])][
                "head_output"
            ]
            if job["condition"] == "oldDcolorMass":
                old_id = f"{job['case_id']}::primaryDcolor::{job['field']}"
                if not torch.equal(
                    delta, old_operators[old_id]["intervention"]["delta_head_output"]
                ):
                    raise ValueError(
                        "old color negative-control delta no longer matches"
                    )
            prediction = {
                "id": job["id"],
                "expected_answer": job["expected_answer"],
                "generated_ids": [],
                "generated_text": "",
                "step_endpoint_ids": [],
                "terminated": False,
                "truncated": False,
                "stop_reason": None,
            }
            generations.append(prediction)
            comparison, compared_reader = comparison_for(
                job, old_predictions, old_generations, original_reader, old_reader
            )
            with query_hooks.add_query_delta(
                model, 28, 0, job["original_query_position"], delta, reference
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
                    if comparison is not None and (
                        step >= len(comparison["step_endpoint_ids"])
                        or not torch.equal(
                            native,
                            compared_reader.read(comparison["step_endpoint_ids"][step]),
                        )
                    ):
                        raise ValueError(
                            "self or old-color full native trajectory changed"
                        )
                    if token in eos:
                        prediction.update(terminated=True, stop_reason="eos")
                        break
                else:
                    prediction.update(truncated=True, stop_reason="max_tokens")
            writer.flush()
            audit_greedy(
                job,
                prediction,
                trace,
                delta,
                reference,
                old.RawReader(root, writer.index),
                comparison,
                compared_reader,
                planned,
                behavior,
                tokenizer,
                eos,
            )
            if len(generations) % 9 == 0:
                print(
                    json.dumps({**counts, "completed_generations": len(generations)}),
                    flush=True,
                )
        old.save_pt(root / "TRACES.pt", traces)
        old.seal_json(root / "GENERATIONS.json", generations)
        writer.finish()
        result = evaluate(
            old.RawReader(root, old.read_json(root / "RAW_INDEX.json")),
            original_reader,
            old_reader,
            captures,
            computed,
            traces,
            generations,
            planned,
            inputs,
            old_natural,
            old_generations,
            behavior,
            tokenizer,
            eos,
            routing.exact_equal,
        )
        if (
            counts["saved_native"] != result["native_forward_count"]
            or counts["attempted_native"] != counts["saved_native"]
        ):
            raise ValueError("native endpoint and physical forward counts differ")
        old.seal_json(root / "RESULT.json", result)
        old.seal_json(root / "COUNTS.json", counts)
        check(output)
        if (
            sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
            > CAP - 1024**2
        ):
            raise RuntimeError("complete decomposition evidence exceeds one GB")
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
            ("NATURAL", captures),
            ("KERNELS", computed),
            ("TRACES", traces),
        ):
            if values and not (root / f"{name}.pt").exists():
                old.save_pt(root / f"PARTIAL_{name}.pt", values)
        if not (root / "GENERATIONS.json").exists():
            old.seal_json(root / "PARTIAL_GENERATIONS.json", generations)
        old.seal_json(
            root / "failure.json",
            {
                "phase": "decomposition",
                "traceback": trace,
                "flush_error": flush_error,
                "counts": counts,
            },
        )
        raise


def rescore(output):
    output = Path(output)
    if sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) > CAP:
        raise ValueError("decomposition evidence exceeds one GB")
    (
        _,
        routing,
        _,
        _,
        _,
        behavior,
        old,
        tokens,
        prior_reg,
        inputs,
        registration,
        planned,
    ) = check(output)
    root = output / "decomposition"
    old.verify_file(root / "CUSTODY.json")
    custody = old.read_json(root / "CUSTODY.json")
    if custody["registration_sha256"] != old.verify_file(output / "REGISTRATION.json"):
        raise ValueError("decomposition custody changed registration")
    for name, digest in custody["artifact_sha256"].items():
        if Path(name).name != name:
            raise ValueError("decomposition custody escaped its namespace")
        old.verify_file(root / name, digest)
    captures, computed, traces = (
        torch.load(root / name, map_location="cpu", weights_only=True)
        for name in ("NATURAL.pt", "KERNELS.pt", "TRACES.pt")
    )
    reader = old.RawReader(root, old.read_json(root / "RAW_INDEX.json"))
    for identity in reader.index["endpoints"]:
        native = reader.read(identity)
        if (
            native.dtype != torch.float32
            or tuple(native.shape) != (tokens["vocab_size"],)
            or not bool(torch.isfinite(native).all())
        ):
            raise ValueError("saved endpoint is not complete native finite FP32")
    original_root = Path(prior_reg["behavior_root"]) / "qwen"
    original_reader = old.RawReader(
        original_root, old.read_json(original_root / "RAW_INDEX.json")
    )
    old_root = Path(registration["validation_root"]) / "validation"
    old_reader = old.RawReader(old_root, old.read_json(old_root / "RAW_INDEX.json"))
    old_natural = torch.load(
        old_root / "NATURAL.pt", map_location="cpu", weights_only=True
    )
    old_generations = {g["id"]: g for g in old.read_json(old_root / "GENERATIONS.json")}
    tokenizer, eos = (
        old.load_tokenizer(tokens["snapshot"]),
        set(tokens["eos_token_ids"]),
    )
    result = evaluate(
        reader,
        original_reader,
        old_reader,
        captures,
        computed,
        traces,
        old.read_json(root / "GENERATIONS.json"),
        planned,
        inputs,
        old_natural,
        old_generations,
        behavior,
        tokenizer,
        eos,
        routing.exact_equal,
    )
    counts = old.read_json(root / "COUNTS.json")
    if (
        result != old.read_json(root / "RESULT.json")
        or counts["attempted_native"] != counts["saved_native"]
        or counts["saved_native"] != result["native_forward_count"]
        or custody["native_forward_count"] != counts["saved_native"]
    ):
        raise ValueError(
            "exact kernel, whole generation, native or count rescore differs"
        )
    return {
        "exact_match": True,
        "status": result["status"],
        "native_forward_count": counts["saved_native"],
        "assessment": result["assessment"],
        "confirmation_calls": 0,
        "automatic_followup_authorized": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "check", "run", "rescore"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--validation-root", type=Path)
    parser.add_argument("--validation-source-root", type=Path)
    parser.add_argument("--deadline")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(
            args.output,
            validation_root=args.validation_root,
            validation_source_root=args.validation_source_root,
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
