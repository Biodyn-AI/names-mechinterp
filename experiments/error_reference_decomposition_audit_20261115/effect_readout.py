"""Saved first-token continuous effects; no models, tokenizers or new forwards.

E is the single canonical expected-answer token. W is the FIXED first token of
the old wrong whole answer, not an intervention-dependent competitor. Protected
questions have no W margin. These descriptive effects never change the failed
registered whole-answer gates and do not sum all acceptable answer variants.
"""

import argparse
import importlib.util
import json
import math
import os
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import audit_receipt as audit

HERE = Path(__file__).resolve().parent


def canonical_id(row):
    ids = row["expected_answer_ids"]
    if (
        not isinstance(ids, list)
        or len(ids) != 1
        or type(ids[0]) is not int
        or ids[0] < 0
    ):
        raise ValueError("canonical E must be exactly one declared answer token")
    return ids[0]


def wrong_token(old, canonical, protected):
    if protected:
        return None
    ids = old["generated_ids"]
    if (
        old.get("terminated") is not True
        or old.get("truncated") is not False
        or old.get("stop_reason") != "eos"
        or len(ids) != 2
        or any(type(i) is not int or i < 0 for i in ids)
        or ids[0] == canonical
    ):
        raise ValueError(
            "fixed old W requires a completed one-value-token plus EOS error"
        )
    return ids[0]


def vector_effect(native, baseline, expected, wrong):
    import torch

    if (
        any(
            value.dtype != torch.float32
            or value.device.type != "cpu"
            or value.ndim != 1
            or not bool(torch.isfinite(value).all())
            for value in (native, baseline)
        )
        or native.shape != baseline.shape
        or any(
            type(i) is not int or not 0 <= i < native.numel()
            for i in [expected] + ([] if wrong is None else [wrong])
        )
    ):
        raise ValueError(
            "complete finite CPU FP32 native vectors and valid token IDs required"
        )
    before, after = baseline.double(), native.double()
    p_before = float(torch.exp(before[expected] - torch.logsumexp(before, 0)))
    p_after = float(torch.exp(after[expected] - torch.logsumexp(after, 0)))
    m_before = float(before[expected] - before[wrong]) if wrong is not None else None
    m_after = float(after[expected] - after[wrong]) if wrong is not None else None
    return {
        "canonical_E_token_id": expected,
        "fixed_old_W_token_id": wrong,
        "canonical_E_probability_before": p_before,
        "canonical_E_probability_after": p_after,
        "canonical_E_probability_change": p_after - p_before,
        "canonical_E_minus_W_before": m_before,
        "canonical_E_minus_W_after": m_after,
        "canonical_E_minus_W_change": m_after - m_before if wrong is not None else None,
        "native_vector_changed": not torch.equal(
            native.view(torch.int32), baseline.view(torch.int32)
        ),
        "native_max_abs_change": float((after - before).abs().max()),
        "argmax_before": int(baseline.argmax()),
        "argmax_after": int(native.argmax()),
        "argmax_changed": int(native.argmax()) != int(baseline.argmax()),
    }


def summarize(rows):
    def mean(key):
        values = [row[key] for row in rows if row[key] is not None]
        return math.fsum(values) / len(values) if values else None

    return {
        "questions": len(rows),
        "old_error_questions": sum(r["fixed_old_W_token_id"] is not None for r in rows),
        "mean_canonical_E_minus_W_change_on_old_errors": mean(
            "canonical_E_minus_W_change"
        ),
        "mean_canonical_E_minus_W_before_on_old_errors": mean(
            "canonical_E_minus_W_before"
        ),
        "mean_canonical_E_minus_W_after_on_old_errors": mean(
            "canonical_E_minus_W_after"
        ),
        "mean_canonical_E_probability_before": mean("canonical_E_probability_before"),
        "mean_canonical_E_probability_after": mean("canonical_E_probability_after"),
        "mean_canonical_E_probability_change": mean("canonical_E_probability_change"),
        "native_vectors_changed": sum(r["native_vector_changed"] for r in rows),
        "max_abs_native_change": max(r["native_max_abs_change"] for r in rows),
        "argmax_changes": sum(r["argmax_changed"] for r in rows),
    }


def load_reader_helper(run, inputs):
    """Locate/hash just the frozen native-reader file; do not import a model stack."""
    reg = json.loads((run / "REGISTRATION.json").read_text())
    path = Path(reg["validation_root"]) / "REGISTRATION.json"
    audit.verified(path, inputs["provenance"]["registration"])
    reg = json.loads(path.read_text())
    for key in ("routing_root", "screen_root", "behavior_root"):
        path = Path(reg[key]) / "REGISTRATION.json"
        audit.verified(path)
        reg = json.loads(path.read_text())
    helper = Path(reg["dependency_root"]) / "e30_14_run.py"
    closure = json.loads((run / "SOURCES.json").read_text())
    expected = closure["validation"]["routing"]["old_screen"]["behavior"]["old"][
        helper.name
    ]
    if audit.digest(helper) != expected:
        raise ValueError("native reader differs from the frozen source closure")
    spec = importlib.util.spec_from_file_location("frozen_effect_native_reader", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, expected


def run_readout(run_root, source_root, native_proof, output):
    run, source, proof_path, target = (
        Path(p).resolve() for p in (run_root, source_root, native_proof, output)
    )
    if (
        target.is_relative_to(run)
        or target.is_relative_to(source)
        or target.exists()
        or Path(str(target) + ".sha256").exists()
    ):
        raise ValueError(
            "readout output must be new and outside sealed scientific directories"
        )
    proof_hash = audit.verified(proof_path)
    proof = json.loads(proof_path.read_text())
    if audit.digest(Path(audit.__file__)) != proof["audit_source_sha256"]:
        raise ValueError("native audit helper differs from the anchored proof")
    before = audit.snapshot(
        run, source, proof["registration_sha256"], proof["source_manifest_sha256"]
    )
    if (
        proof.get("exact_match") is not True
        or proof.get("confirmation_calls") != 0
        or any(proof.get(key) != value for key, value in before["anchors"].items())
        or proof["runtime_source_sha256"] != before["runtime_source_sha256"]
    ):
        raise ValueError(
            "complete prior native proof and exact artifact anchors required"
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.dont_write_bytecode = True
    import torch

    if torch.cuda.is_initialized():
        raise ValueError("use a fresh CPU-only readout process")
    torch.set_num_threads(1)
    stage = run / "decomposition"
    inputs, planned = (
        json.loads((run / name).read_text()) for name in ("INPUTS.json", "PLAN.json")
    )
    generations = json.loads((stage / "GENERATIONS.json").read_text())
    jobs, rows, old = (
        {item["id"]: item for item in records}
        for records in (
            planned["generation_jobs"],
            inputs["rows"],
            inputs["predictions"],
        )
    )
    if (
        len(jobs) != 162
        or len(generations) != 162
        or len({g["id"] for g in generations}) != 162
        or set(jobs) != {g["id"] for g in generations}
    ):
        raise ValueError("the complete finite162-job saved grid is required")
    helper, helper_hash = load_reader_helper(run, inputs)
    reader = helper.RawReader(stage, json.loads((stage / "RAW_INDEX.json").read_text()))
    found = {g["id"]: g for g in generations}
    baseline = {
        (j["case_id"], j["field"]): reader.read(found[j["id"]]["step_endpoint_ids"][0])
        for j in jobs.values()
        if j["condition"] == "self"
    }
    if len(baseline) != 18:
        raise ValueError("exactly18 same-field self baselines are required")
    per_question, grouped, worlds = (
        [],
        defaultdict(list),
        defaultdict(lambda: defaultdict(list)),
    )
    # Saved generation order follows the writer: baselines are already cached,
    # so this pass does not bounce back into the early self shards per question.
    for generation in generations:
        job = jobs[generation["id"]]
        identity = generation["step_endpoint_ids"][0]
        if identity != job["greedy_endpoint_ids"][0]:
            raise ValueError(
                "first native endpoint differs from its registered question"
            )
        native = reader.read(identity)
        if tuple(native.shape) != (planned["model"]["vocab_size"],):
            raise ValueError(
                "first saved native does not contain the complete vocabulary"
            )
        expected = canonical_id(rows[job["row_id"]])
        wrong = wrong_token(old[job["row_id"]], expected, job["baseline_correct"])
        self_native = baseline[(job["case_id"], job["field"])]
        if wrong is not None and int(self_native.argmax()) != wrong:
            raise ValueError(
                "the fixed old wrong answer differs from the audited self baseline"
            )
        measured = {
            "id": job["id"],
            "world_id": job["world_id"],
            "field": job["field"],
            "condition": job["condition"],
            "protected_before": job["baseline_correct"],
            **vector_effect(native, self_native, expected, wrong),
        }
        per_question.append(measured)
        grouped[job["condition"]].append(measured)
        worlds[job["condition"]][job["world_id"]].append(measured)
    if len(grouped) != 9 or any(
        len(values) != 18
        or sum(r["fixed_old_W_token_id"] is not None for r in values) != 8
        for values in grouped.values()
    ):
        raise ValueError(
            "each fixed condition needs18 questions and eight original errors"
        )
    after = audit.snapshot(
        run, source, proof["registration_sha256"], proof["source_manifest_sha256"]
    )
    if before != after or torch.cuda.is_initialized():
        raise ValueError(
            "scientific evidence changed or CUDA was initialized during readout"
        )
    result = {
        "by_condition": {key: summarize(values) for key, values in grouped.items()},
        "per_world": {
            key: {world: summarize(values) for world, values in groups.items()}
            for key, groups in worlds.items()
        },
        "per_question": per_question,
        "native_proof_sha256": proof_hash,
        **before["anchors"],
        "runtime_source_sha256": before["runtime_source_sha256"],
        "native_reader_source_sha256": helper_hash,
        "readout_sources_sha256": {
            name: audit.digest(HERE / name)
            for name in ("effect_readout.py", "test_effect_readout.py")
        },
        "audit_source_sha256": audit.digest(Path(audit.__file__)),
        "created_utc": datetime.now(UTC).isoformat(),
        "registered_whole_answer_assessment_unchanged": proof["assessment"],
        "new_research_forward_count": 0,
        "does_not_change_registered_gates": True,
        "metric_scope": "Descriptive canonical single-token E probability over the FULL vocabulary and E-minus-fixed-old-W margin on eight old errors only. Not accepted-variant sums, whole-answer accuracy, new mechanism evidence or confirmation. Six reopened development worlds.",
    }
    helper.seal_json(target, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--native-proof", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run_readout(
        args.run_root, args.source_root, args.native_proof, args.output
    )
    print(
        json.dumps(
            {"by_condition": result["by_condition"], "new_research_forward_count": 0},
            sort_keys=True,
            allow_nan=False,
        )
    )
