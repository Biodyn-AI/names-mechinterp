"""Independent printed-registry grading of six-world whole-answer validation.

No experimental planner, grader, tokenizer or model is imported. Reverse targets
come from the original recipient's observed screen error, not saved target IDs.
"""

import argparse
import hashlib
import inspect
import itertools
import json
import string
from collections import Counter, defaultdict
from pathlib import Path

from audit_behavior import literal_answer

FIELDS = ("color", "shape", "home")
CONDITIONS = (
    "primaryDcolor",
    "selfRfield",
    "frozenDiscoveryMean",
    "Rcolor",
    "reverseRscreen",
    "cyclicDcolor",
)
COUNT_KEYS = (
    "observed_fields",
    "task_correct",
    "predicted_target_hits",
    "repaired_initial_errors",
    "protected_damaged",
    "other_listed_wrong",
    "malformed_or_truncated",
)


def normalize(text):
    return " ".join(text.strip().casefold().split()).rstrip(string.punctuation).rstrip()


def finished(prediction):
    if (
        type(prediction.get("terminated")) is not bool
        or type(prediction.get("truncated")) is not bool
        or prediction["terminated"] == prediction["truncated"]
        or not isinstance(prediction.get("generated_text"), str)
    ):
        raise ValueError(
            "Explicit consistent whole-answer completion metadata is required"
        )
    return prediction["terminated"] and not prediction["truncated"]


def lookup(records, key="id"):
    result = {record[key]: record for record in records}
    if len(result) != len(records):
        raise ValueError("Duplicate artifact identifiers")
    return result


def audit(planned, rows, old_generations, new_generations):
    cases = lookup(planned["cases"], "case_id")
    jobs, printed, old, new = map(
        lookup, (planned["generation_jobs"], rows, old_generations, new_generations)
    )
    grid = {(j["case_id"], j["condition"], j["field"]) for j in jobs.values()}
    if (
        len(cases) != 6
        or len({c["world_id"] for c in cases.values()}) != 6
        or len(jobs) != 108
        or len(new) != 108
        or set(new) != set(jobs)
        or grid != set(itertools.product(cases, CONDITIONS, FIELDS))
    ):
        raise ValueError("Exactly the complete six-world 108-job grid is required")
    literal = {
        identity: literal_answer(row["text"])
        for identity, row in printed.items()
        if identity in old
    }
    truths, baseline = {}, Counter(recipient_wrong=0, recipient_protected=0)
    for cid, case in cases.items():
        screen_id = case["screen_row_id"]
        _screen_label, entity, screen_field, facts = literal[screen_id]
        if (
            screen_field not in FIELDS
            or screen_id != case["recipient_row_ids"][screen_field]
        ):
            raise ValueError("The old screen must be a recipient field question")
        if any(
            len({normalize(v[field]) for v in facts.values()}) != 4 for field in FIELDS
        ):
            raise ValueError("Printed field values must uniquely identify each record")
        screen = old[screen_id]
        wrong = [
            obj
            for obj in "ABCD"
            if normalize(facts[obj][screen_field])
            == normalize(screen["generated_text"])
        ]
        if not finished(screen) or len(wrong) != 1 or wrong[0] == entity:
            raise ValueError(
                "Original recipient screen answer must select one distinct wrong object"
            )
        if (
            case.get("correct_object", entity) != entity
            or case.get("reverse_object", wrong[0]) != wrong[0]
        ):
            raise ValueError(
                "Saved E or reverse W differs from independent literal reconstruction"
            )
        protected = {}
        for field in FIELDS:
            for role in ("donor", "recipient"):
                identity = case[f"{role}_row_ids"][field]
                label, obj, parsed_field, parsed_facts = literal[identity]
                if (
                    obj != entity
                    or parsed_field != field
                    or parsed_facts != facts
                    or printed[identity].get("expected_answer", label) != label
                ):
                    raise ValueError(
                        "Field/alias questions do not share the same literal registry and entity"
                    )
                correct = finished(old[identity]) and normalize(
                    old[identity]["generated_text"]
                ) == normalize(label)
                if role == "donor" and not correct:
                    raise ValueError("All original donor attributes must be correct")
                if role == "recipient":
                    protected[field] = correct
                    baseline[
                        "recipient_protected" if correct else "recipient_wrong"
                    ] += 1
        truths[cid] = {
            "entity": entity,
            "wrong": wrong[0],
            "facts": facts,
            "protected": protected,
        }
    if dict(baseline) != {"recipient_wrong": 8, "recipient_protected": 10}:
        raise ValueError(
            "Independent old answers must reproduce eight wrong and ten protected facts"
        )
    by_condition = {
        condition: Counter(dict.fromkeys(COUNT_KEYS, 0)) for condition in CONDITIONS
    }
    by_field, by_world, joint, success = (
        defaultdict(Counter),
        defaultdict(Counter),
        defaultdict(set),
        defaultdict(set),
    )
    cyclic_objects, outcomes = set(), []
    for identity, job in jobs.items():
        cid, condition, field = job["case_id"], job["condition"], job["field"]
        case, truth, generated = cases[cid], truths[cid], new[identity]
        role = (
            "donor" if condition in ("reverseRscreen", "cyclicDcolor") else "recipient"
        )
        row_id = case[f"{role}_row_ids"][field]
        if job["row_id"] != row_id or job.get("destination_role", role) != role:
            raise ValueError(
                "Counterfactual destination does not match its literal question"
            )
        label, entity, parsed_field, facts = literal[row_id]
        target_object = (
            truth["wrong"]
            if condition == "reverseRscreen"
            else "BCDA"["ABCD".index(entity)]
            if condition == "cyclicDcolor"
            else entity
        )
        target = (
            old[row_id]["generated_text"]
            if condition == "selfRfield"
            else facts[target_object][field]
        )
        if condition == "selfRfield":
            target_object = next(
                (
                    obj
                    for obj in "ABCD"
                    if normalize(facts[obj][field]) == normalize(target)
                ),
                None,
            )
        if (
            parsed_field != field
            or job["task_answer"] != label
            or job["predicted_answer"] != target
            or job.get("predicted_object", target_object) != target_object
            or generated.get("task_answer", label) != label
            or generated.get("predicted_target_answer", target) != target
        ):
            raise ValueError(
                "Saved factual truth or steering countertarget differs from literal reconstruction"
            )
        complete = finished(generated)
        answer = normalize(generated["generated_text"])
        correct, hit = (
            complete and answer == normalize(label),
            complete and answer == normalize(target),
        )
        if condition == "selfRfield" and any(
            generated.get(key) != old[row_id].get(key)
            for key in (
                "generated_ids",
                "generated_text",
                "terminated",
                "truncated",
                "stop_reason",
            )
        ):
            raise ValueError(
                "Self control differs from the complete old IDs/text/termination trajectory"
            )
        protected = role == "recipient" and truth["protected"][field]
        listed = complete and answer in {normalize(v[field]) for v in facts.values()}
        flags = {
            "observed_fields": 1,
            "task_correct": int(correct),
            "predicted_target_hits": int(hit),
            "repaired_initial_errors": int(
                role == "recipient" and not protected and correct
            ),
            "protected_damaged": int(protected and not correct),
            "other_listed_wrong": int(listed and not correct and not hit),
            "malformed_or_truncated": int(not listed),
        }
        for counter in (
            by_condition[condition],
            by_field[(condition, field)],
            by_world[(condition, case["world_id"])],
        ):
            counter.update(flags)
        if condition == "primaryDcolor" and flags["repaired_initial_errors"]:
            success["repair"].add(cid)
        if hit:
            joint[(condition, cid)].add(field)
            success[condition].add(cid)
            if condition == "cyclicDcolor":
                cyclic_objects.add(target_object)
        outcomes.append(
            {
                "id": identity,
                "world_id": case["world_id"],
                "field": field,
                "literal_object": entity,
                "literal_answer": label,
                "independent_target_object": target_object,
                "independent_target_answer": target,
                "answer": answer,
                "task_correct": bool(correct),
                "predicted_hit": bool(hit),
            }
        )
    for condition, counts in by_condition.items():
        counts["intended_fields"] = 18
        counts["three_field_joint_predicted_worlds"] = sum(
            len(joint[(condition, cid)]) == 3 for cid in cases
        )
    values = {
        "primary_repaired": by_condition["primaryDcolor"]["repaired_initial_errors"],
        "primary_repair_worlds": len(success["repair"]),
        "protected_damaged": by_condition["primaryDcolor"]["protected_damaged"],
        "mean_repaired": by_condition["frozenDiscoveryMean"]["repaired_initial_errors"],
        "reverse_hits": by_condition["reverseRscreen"]["predicted_target_hits"],
        "reverse_worlds": len(success["reverseRscreen"]),
        "cyclic_hits": by_condition["cyclicDcolor"]["predicted_target_hits"],
        "cyclic_worlds": len(success["cyclicDcolor"]),
        "cyclic_success_object_count": len(cyclic_objects),
    }
    repair = (
        values["primary_repaired"] >= 4
        and values["primary_repair_worlds"] >= 3
        and values["protected_damaged"] <= 1
        and values["primary_repaired"] - values["mean_repaired"] >= 2
    )
    address = (
        values["reverse_hits"] >= 9
        and values["reverse_worlds"] >= 3
        and values["cyclic_hits"] >= 9
        and values["cyclic_worlds"] >= 3
        and len(cyclic_objects) >= 2
    )
    return {
        "independent_literal_whole_answer_audit": True,
        "independent_worlds": 6,
        "generations": 108,
        "baseline": dict(baseline),
        "by_condition": {
            key: {**dict(counts), "task_accuracy": counts["task_correct"] / 18}
            for key, counts in by_condition.items()
        },
        "by_field": {
            condition: {field: dict(by_field[(condition, field)]) for field in FIELDS}
            for condition in CONDITIONS
        },
        "by_world": {
            condition: {
                case["world_id"]: dict(by_world[(condition, case["world_id"])])
                for case in cases.values()
            }
            for condition in CONDITIONS
        },
        "practical_counts": values,
        "repair_replication_passed": repair,
        "address_specificity_passed": address,
        "strong_candidate_passed": repair and address,
        "cyclic_success_object_ids": sorted(cyclic_objects),
        "outcomes": outcomes,
        "scope": "Six behavior-screened development worlds, not 108 independent cases or confirmation. Reverse/cyclic hits are not factual correctness. Native-logit/delta/position proof requires the separate replay audit.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "validation-plan",
        "rows",
        "old-generations",
        "new-generations",
        "output",
    ):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    sidecar = Path(str(args.output) + ".sha256")
    if args.output.exists() or sidecar.exists():
        raise FileExistsError("Independent literal audit output must be new")
    paths = [
        args.validation_plan,
        args.rows,
        args.old_generations,
        args.new_generations,
    ]
    content = [path.read_bytes() for path in paths]
    digests = [hashlib.sha256(blob).hexdigest() for blob in content]
    for path, digest in zip(paths, digests):
        checksum = Path(str(path) + ".sha256")
        if checksum.exists() and checksum.read_text().strip() != digest:
            raise ValueError("Input checksum differs from its sealed sidecar")
    report = audit(*(json.loads(blob) for blob in content))
    report["input_sha256"] = {str(path): digest for path, digest in zip(paths, digests)}
    report["audit_source_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    report["literal_parser_source_sha256"] = hashlib.sha256(
        Path(inspect.getsourcefile(literal_answer)).read_bytes()
    ).hexdigest()
    encoded = (
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    with args.output.open("xb") as output:
        output.write(encoded)
    with sidecar.open("x") as checksum:
        checksum.write(hashlib.sha256(encoded).hexdigest() + "\n")


if __name__ == "__main__":
    main()
