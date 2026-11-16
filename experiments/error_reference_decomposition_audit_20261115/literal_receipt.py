"""Independent printed-registry whole-answer receipt; no experimental imports."""

import argparse
import hashlib
import importlib.util
import itertools
import json
import string
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

PARSER_SHA256 = "01b1cd6eaeaea4fa23ad01eecdec8dd74be3338ce22372ca8a673d9e34617de8"
FIELDS = ("color", "shape", "home")
CONDITIONS = (
    "self",
    "oldDcolorMass",
    "sameFieldMass",
    "sameFieldWithinOnly",
    "sameFieldProfileWithin",
    "sameFieldRecordContribution",
    "sameFieldNonrecordResidual",
    "sameFieldFullHead",
    "normMatchedRandomFullHead",
)


def literal_parser():
    path = (
        Path(__file__).resolve().parents[1]
        / "error_reference_audit_20261115"
        / "audit_behavior.py"
    )
    if hashlib.sha256(path.read_bytes()).hexdigest() != PARSER_SHA256:
        raise ValueError("Independent literal parser differs from its pinned source")
    spec = importlib.util.spec_from_file_location("pinned_literal_parser", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.literal_answer


def normalize(text):
    return " ".join(text.strip().casefold().split()).rstrip(string.punctuation).rstrip()


def lookup(records, key="id"):
    found = {item[key]: item for item in records}
    if len(found) != len(records):
        raise ValueError("Duplicate identifiers are not complete coverage")
    return found


def completed(prediction):
    terminated, truncated = prediction.get("terminated"), prediction.get("truncated")
    ids = prediction.get("generated_ids")
    if (
        type(terminated) is not bool
        or type(truncated) is not bool
        or terminated == truncated
        or not isinstance(prediction.get("generated_text"), str)
        or not isinstance(ids, list)
        or not 1 <= len(ids) <= 8
        or any(type(token) is not int or token < 0 for token in ids)
        or prediction.get("stop_reason") != ("eos" if terminated else "max_tokens")
        or truncated
        and len(ids) != 8
    ):
        raise ValueError(
            "Explicit consistent whole-answer termination metadata is required"
        )
    return terminated and not truncated


def reconstruct(planned, inputs, generations):
    """Derive truth/baselines from printed text, not saved success or target flags."""
    parse = literal_parser()
    rows, old = lookup(inputs["rows"]), lookup(inputs["predictions"])
    jobs, new, cases = (
        lookup(planned["generation_jobs"]),
        lookup(generations),
        lookup(planned["cases"], "case_id"),
    )
    grid = {(j["case_id"], j["condition"], j["field"]) for j in jobs.values()}
    if (
        len(rows) != 36
        or set(rows) != set(old)
        or len(cases) != 6
        or len({c["world_id"] for c in cases.values()}) != 6
        or len(jobs) != 162
        or len(new) != 162
        or set(jobs) != set(new)
        or grid != set(itertools.product(cases, CONDITIONS, FIELDS))
    ):
        raise ValueError(
            "Exactly the six-world, three-field, nine-cell complete grid is required"
        )
    parsed, correct_before = {}, {}
    for identity, row in rows.items():
        label, obj, field, facts = parse(row["text"])
        if (
            field not in FIELDS
            or row["field"] != field
            or row["expected_answer"] != label
        ):
            raise ValueError(
                "Saved row field or truth differs from its literal question"
            )
        parsed[identity] = (label, obj, field, facts)
        correct_before[identity] = completed(old[identity]) and normalize(
            old[identity]["generated_text"]
        ) == normalize(label)
    baseline, used_rows = Counter(recipient_wrong=0, recipient_protected=0), set()
    for case in cases.values():
        reference = None
        for role in ("recipient", "donor"):
            for field in FIELDS:
                identity = case[f"{role}_row_ids"][field]
                label, obj, actual_field, facts = parsed[identity]
                value = (obj, facts)
                if (
                    actual_field != field
                    or rows[identity]["world_id"] != case["world_id"]
                    or reference is not None
                    and reference != value
                    or case.get("correct_object", obj) != obj
                ):
                    raise ValueError(
                        "Literal registry/entity/world differs across natural questions"
                    )
                reference = value
                used_rows.add(identity)
                if role == "donor" and not correct_before[identity]:
                    raise ValueError("The original natural donor is not fully correct")
                if role == "recipient":
                    baseline[
                        "recipient_protected"
                        if correct_before[identity]
                        else "recipient_wrong"
                    ] += 1
    if used_rows != set(rows) or dict(baseline) != {
        "recipient_wrong": 8,
        "recipient_protected": 10,
    }:
        raise ValueError(
            "Literal old answers must reproduce eight errors and ten protected facts"
        )
    if any(planned["baseline"].get(key) != count for key, count in baseline.items()):
        raise ValueError("Saved baseline differs from independent old-answer scoring")
    reports, per_question = {}, []
    for condition in CONDITIONS:
        counts, repair_worlds = (
            Counter(correct=0, repaired=0, protected_damaged=0, truncated=0),
            set(),
        )
        for job in (
            j for j in planned["generation_jobs"] if j["condition"] == condition
        ):
            identity, row_id = job["id"], job["row_id"]
            case, prediction = cases[job["case_id"]], new[identity]
            label, obj, field, _facts = parsed[row_id]
            if (
                case["recipient_row_ids"][job["field"]] != row_id
                or job["world_id"] != case["world_id"]
                or rows[row_id]["world_id"] != job["world_id"]
                or field != job["field"]
                or job["expected_answer"] != label
                or prediction["expected_answer"] != label
                or type(job["baseline_correct"]) is not bool
                or job["baseline_correct"] != correct_before[row_id]
            ):
                raise ValueError(
                    "Planned label, baseline, world or field differs from literal truth"
                )
            finished = completed(prediction)
            endpoints = prediction["step_endpoint_ids"]
            if (
                len(job["greedy_endpoint_ids"]) != 8
                or len(set(job["greedy_endpoint_ids"])) != 8
                or endpoints
                != job["greedy_endpoint_ids"][: len(prediction["generated_ids"])]
            ):
                raise ValueError(
                    "Generation endpoint metadata differs from its planned identity"
                )
            if condition == "self" and any(
                prediction[key] != old[row_id][key]
                for key in (
                    "generated_ids",
                    "generated_text",
                    "terminated",
                    "truncated",
                    "stop_reason",
                )
            ):
                raise ValueError(
                    "Self whole-answer metadata differs from the old trajectory"
                )
            correct = finished and normalize(prediction["generated_text"]) == normalize(
                label
            )
            repaired, damaged = (
                not correct_before[row_id] and correct,
                correct_before[row_id] and not correct,
            )
            counts.update(
                correct=int(correct),
                repaired=int(repaired),
                protected_damaged=int(damaged),
                truncated=int(prediction["truncated"]),
            )
            if repaired:
                repair_worlds.add(case["world_id"])
            per_question.append(
                {
                    "id": identity,
                    "condition": condition,
                    "world_id": case["world_id"],
                    "field": field,
                    "correct": bool(correct),
                    "repaired": bool(repaired),
                    "protected_damaged": bool(damaged),
                }
            )
        reports[condition] = {
            **dict(counts),
            "count": 18,
            "initial_errors": 8,
            "initial_protected": 10,
            "repair_worlds": sorted(repair_worlds),
            "diagnostic_passed": counts["repaired"] >= 4
            and len(repair_worlds) >= 3
            and counts["protected_damaged"] <= 1,
        }
    branches = [
        name for name in CONDITIONS if name not in ("self", "normMatchedRandomFullHead")
    ]
    advantage = (
        reports["sameFieldFullHead"]["repaired"]
        - reports["normMatchedRandomFullHead"]["repaired"]
    )
    assessment = {
        "conditions": reports,
        "per_question": per_question,
        "all_fixed_branches_failed": all(
            not reports[name]["diagnostic_passed"] for name in branches
        ),
        "full_head_beats_matched_random_by_two_repairs": advantage >= 2,
        "full_head_repair_advantage_over_matched_random": advantage,
        "branches_for_possible_new_plan": [
            name
            for name in branches
            if reports[name]["diagnostic_passed"]
            and (name != "sameFieldFullHead" or advantage >= 2)
        ],
        "confirmation_authorized": False,
        "new_entity_mechanism_established": False,
    }
    return {
        "independent_literal_whole_answer_audit": True,
        "independent_worlds": 6,
        "generations": 162,
        "baseline": dict(baseline),
        "assessment": assessment,
        "native_proof_claimed": False,
        "scope": "Printed-registry labels, old baselines, whole-answer metadata and diagnostic counts only. Native argmax/EOS-token, hidden-state and delta proof belongs to the separate anchored CPU audit. Six reopened development worlds are not confirmation.",
    }


def compare_assessment(independent, reported):
    left, right = (
        {k: v for k, v in value.items() if k not in ("scope", "per_question")}
        for value in (independent, reported)
    )
    if left != right or lookup(independent["per_question"]) != lookup(
        reported["per_question"]
    ):
        raise ValueError(
            "Reported assessment differs from independent literal whole-answer counts"
        )
    return True


def verified_json(path, expected=None):
    content = Path(path).read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if (
        digest != Path(str(path) + ".sha256").read_text().strip()
        or expected is not None
        and digest != expected
    ):
        raise ValueError(f"Input hash mismatch: {Path(path).name}")
    return json.loads(content), digest


def run_audit(metadata_dir, output):
    root, target = Path(metadata_dir), Path(output)
    sidecar = Path(str(target) + ".sha256")
    if target.exists() or sidecar.exists():
        raise FileExistsError("Literal audit output is single-use")
    registration, reg_hash = verified_json(root / "REGISTRATION.json")
    records, hashes = {}, {"registration_sha256": reg_hash}
    for name, key in (
        ("PLAN.json", "plan_sha256"),
        ("INPUTS.json", "inputs_sha256"),
        ("SOURCES.json", "source_manifest_sha256"),
    ):
        records[name], hashes[key] = verified_json(
            root / name, registration["input_sha256"][name]
        )
    stage = root / "decomposition"
    custody, hashes["custody_sha256"] = verified_json(stage / "CUSTODY.json")
    if custody["registration_sha256"] != reg_hash:
        raise ValueError("Custody is attached to another registration")
    for name, key in (
        ("GENERATIONS.json", "generations_sha256"),
        ("RESULT.json", "result_sha256"),
    ):
        records[name], hashes[key] = verified_json(
            stage / name, custody["artifact_sha256"][name]
        )
    report = reconstruct(
        records["PLAN.json"], records["INPUTS.json"], records["GENERATIONS.json"]
    )
    report["reported_assessment_match"] = compare_assessment(
        report["assessment"], records["RESULT.json"]["assessment"]
    )
    report.update(
        **hashes,
        literal_parser_source_sha256=PARSER_SHA256,
        literal_audit_source_sha256=hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        audited_utc=datetime.now(UTC).isoformat(),
    )
    content = (
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    with target.open("xb") as stream:
        stream.write(content)
    with sidecar.open("x") as stream:
        stream.write(hashlib.sha256(content).hexdigest() + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    receipt = run_audit(arguments.metadata_dir, arguments.output)
    print(
        json.dumps(
            {
                "reported_assessment_match": receipt["reported_assessment_match"],
                "baseline": receipt["baseline"],
                "all_fixed_branches_failed": receipt["assessment"][
                    "all_fixed_branches_failed"
                ],
            },
            sort_keys=True,
        )
    )
