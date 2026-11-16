"""Independent complete-answer audit derived from the printed object lists.

Only the historical, hash-pinned prompt generator and literal parser are loaded.
The new study's selection, grading and statistical-analysis code is not imported.
Native score vectors, token decoding and intervention arithmetic require the
separate native audit; this receipt does not claim to recheck those quantities.
"""

import argparse
import hashlib
import importlib.util
import itertools
import json
import string
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

PARSER_SHA256 = "01b1cd6eaeaea4fa23ad01eecdec8dd74be3338ce22372ca8a673d9e34617de8"
GENERATOR_SHA256 = "186e73abf1e541aaa1a23bc7728f3a4ccd61cde95fb319e23bc00e45bf669200"
SELECTION_SEED = "reference-causal-discovery-20261115-v1"
FIELDS = ("color", "shape", "home")
FOCUS = ("none", "coref_0", "coref_1", "distractor")
CONDITIONS = ("selfRfield", "primaryDcolor", "frozenDiscoveryMean", "Rcolor")


def pinned_module(path, expected, name):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Pinned independent source changed: {path.name}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize(text):
    return " ".join(text.strip().casefold().split()).rstrip(string.punctuation).rstrip()


def lookup(records, key="id"):
    found = {record[key]: record for record in records}
    if len(found) != len(records):
        raise ValueError("Duplicate identifiers are not complete coverage")
    return found


def completed(prediction):
    """A convenient correct prefix never becomes a finished correct answer."""
    terminated, truncated = prediction.get("terminated"), prediction.get("truncated")
    ids, endpoints = prediction.get("generated_ids"), prediction.get(
        "step_endpoint_ids"
    )
    if (
        type(terminated) is not bool
        or type(truncated) is not bool
        or terminated == truncated
        or not isinstance(prediction.get("generated_text"), str)
        or not isinstance(ids, list)
        or not 1 <= len(ids) <= 8
        or any(type(token) is not int or token < 0 for token in ids)
        or not isinstance(endpoints, list)
        or len(endpoints) != len(ids)
        or len(set(endpoints)) != len(endpoints)
        or any(not isinstance(endpoint, str) or not endpoint for endpoint in endpoints)
        or prediction.get("stop_reason") != ("eos" if terminated else "max_tokens")
        or truncated
        and len(ids) != 8
    ):
        raise ValueError(
            "Explicit consistent complete-answer completion metadata required"
        )
    return terminated and not truncated


def checked_inputs(rows, predictions, repo_root):
    """Authenticate coverage, then derive truth from each printed question."""
    repo = Path(repo_root)
    generator = pinned_module(
        repo / "experiments/error_reference_study_20261114/error_protocol.py",
        GENERATOR_SHA256,
        "literal_confirmation_generator",
    )
    parser = pinned_module(
        repo / "experiments/error_reference_audit_20261115/audit_behavior.py",
        PARSER_SHA256,
        "literal_confirmation_parser",
    ).literal_answer
    declared = {
        row["id"]: row
        for row in generator.build_study()["rows"]
        if row["split"] == "confirmation"
    }
    found, old = lookup(rows), lookup(predictions)
    worlds = Counter(row["world_id"] for row in rows)
    if (
        len(found) != 6144
        or len(worlds) != 64
        or set(worlds.values()) != {96}
        or set(found) != set(declared)
        or set(old) != set(found)
    ):
        raise ValueError(
            "Exactly all 64 worlds and 6144 confirmation questions required for coverage"
        )
    parsed, outcomes = {}, {}
    for identity, row in found.items():
        if any(row.get(key) != value for key, value in declared[identity].items()):
            raise ValueError("A row differs from the historical frozen generator")
        truth, target, field, facts = parser(row["text"])
        prediction = old[identity]
        if (
            row["expected_answer"] != truth
            or row["field"] != field
            or prediction.get("expected_answer") != truth
        ):
            raise ValueError("Saved truth or field differs from the printed question")
        finished = completed(prediction)
        answer = normalize(prediction["generated_text"])
        correct = finished and answer == normalize(truth)
        selected = None
        if finished and field in FIELDS:
            selected = next(
                (obj for obj in "ABCD" if normalize(facts[obj][field]) == answer), None
            )
        parsed[identity] = (truth, target, field, facts)
        outcomes[identity] = {"correct": bool(correct), "selected_object": selected}
    return found, old, parsed, outcomes


def reconstruct_behavior(rows, predictions, repo_root):
    """Independently count factual errors and equivalent-name disagreements."""
    found, old, parsed, outcomes = checked_inputs(rows, predictions, repo_root)
    counts = Counter(correct=0, count=0)
    facts = Counter(wrong_object=0, unlisted_or_malformed=0)
    by_kind, pairs, wrong_worlds = defaultdict(Counter), defaultdict(dict), set()
    summaries = {
        world: {
            "fact_correct": 0,
            "facts": 0,
            "wrong_object": 0,
            "coref_pairs": 0,
            "wrong_object_coref": 0,
        }
        for world in {row["world_id"] for row in rows}
    }
    for identity, row in found.items():
        outcome, field = outcomes[identity], parsed[identity][2]
        correct, selected = outcome["correct"], outcome["selected_object"]
        counts.update(correct=int(correct), count=1)
        by_kind[row["probe_kind"]].update(correct=int(correct), count=1)
        if field in FIELDS:
            summaries[row["world_id"]]["fact_correct"] += correct
            summaries[row["world_id"]]["facts"] += 1
        if field in FIELDS and not correct:
            facts[
                "wrong_object" if selected is not None else "unlisted_or_malformed"
            ] += 1
            if selected is not None:
                wrong_worlds.add(row["world_id"])
                summaries[row["world_id"]]["wrong_object"] += 1
        if field in FIELDS and row["query_alias_index"] in (0, 1):
            key = (row["world_id"], row["name_swap"], row["focus_condition"], field)
            pairs[key][row["query_alias_index"]] = (
                correct,
                selected,
                normalize(old[identity]["generated_text"]),
            )
    if len(pairs) != 1536 or any(set(pair) != {0, 1} for pair in pairs.values()):
        raise ValueError("All 1536 paired naming comparisons are required")
    pair_counts = Counter(
        pair_count=1536,
        discordant_one_correct=0,
        discordant_wrong_object=0,
        different_answers=0,
        both_correct=0,
        both_wrong=0,
    )
    discordant_worlds = set()
    for key, pair in pairs.items():
        first, second = pair[0], pair[1]
        wrong = second if first[0] else first
        summaries[key[0]]["coref_pairs"] += 1
        pair_counts["discordant_one_correct"] += first[0] != second[0]
        pair_counts["different_answers"] += first[2] != second[2]
        pair_counts["both_correct"] += first[0] and second[0]
        pair_counts["both_wrong"] += not first[0] and not second[0]
        if first[0] != second[0] and wrong[1] is not None:
            pair_counts["discordant_wrong_object"] += 1
            discordant_worlds.add(key[0])
            summaries[key[0]]["wrong_object_coref"] += 1
    return {
        "independent_literal_behavior_audit": True,
        "worlds": 64,
        "overall": dict(counts),
        "by_probe_kind": {kind: dict(values) for kind, values in by_kind.items()},
        "fact_errors": dict(facts),
        "coreference": dict(pair_counts),
        "wrong_object_worlds": len(wrong_worlds),
        "wrong_object_coreference_worlds": len(discordant_worlds),
        "world_summaries": dict(sorted(summaries.items())),
    }


def selection_hash(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def independently_select_cases(rows, predictions, repo_root):
    """Reconstruct one old-rule context in every newly eligible world.

    Every donor must give all three facts AND the permanent identifier correctly.
    A recipient must give at least one completed wrong-object factual answer.
    Once eligible, all its wrong facts count, not just the convenient error that
    qualified it. All three properties are retained for preservation measurement.
    """
    found, _old, parsed, outcomes = checked_inputs(rows, predictions, repo_root)
    index = {
        (
            row["world_id"],
            row["name_swap"],
            row["focus_condition"],
            row["probe_kind"],
            row["query_alias_index"],
            row["field"],
        ): identity
        for identity, row in found.items()
    }
    eligible = defaultdict(list)
    for world in sorted({row["world_id"] for row in rows}):
        for swap, focus, recipient in itertools.product((0, 1), FOCUS, (0, 1)):
            group, donor = (world, swap, focus), 1 - recipient
            donor_rows = {
                field: index[group + ("fact", donor, field)] for field in FIELDS
            }
            recipient_rows = {
                field: index[group + ("fact", recipient, field)] for field in FIELDS
            }
            donor_identity = index[group + ("identity", donor, None)]
            if not all(
                outcomes[identity]["correct"]
                for identity in [*donor_rows.values(), donor_identity]
            ):
                continue
            reference = parsed[donor_identity][1]
            if any(
                parsed[identity][1] != reference
                for identity in [*donor_rows.values(), *recipient_rows.values()]
            ):
                raise ValueError(
                    "Equivalent printed names do not resolve to one object"
                )
            wrong = {
                field: outcomes[identity]["selected_object"]
                for field, identity in recipient_rows.items()
                if not outcomes[identity]["correct"]
                and outcomes[identity]["selected_object"] is not None
            }
            if not wrong:
                continue
            case_id = f"{world}:swap{swap}:{focus}:recipient{recipient}"
            screen = min(
                wrong,
                key=lambda field: selection_hash([SELECTION_SEED, case_id, field]),
            )
            eligible[world].append(
                {
                    "case_id": case_id,
                    "world_id": world,
                    "correct_object": reference,
                    "screen_field": screen,
                    "reverse_object": wrong[screen],
                    "wrong_object_by_field": wrong,
                    "initial_wrong_fields": [
                        field
                        for field in FIELDS
                        if not outcomes[recipient_rows[field]]["correct"]
                    ],
                    "protected_correct_fields": [
                        field
                        for field in FIELDS
                        if outcomes[recipient_rows[field]]["correct"]
                    ],
                    "donor_row_ids": donor_rows,
                    "recipient_row_ids": recipient_rows,
                    "screen_row_id": recipient_rows[screen],
                }
            )
    ordered = sorted(
        eligible, key=lambda world: selection_hash([SELECTION_SEED, "world", world])
    )
    return [
        min(
            eligible[world],
            key=lambda case: selection_hash(
                [SELECTION_SEED, "condition", case["case_id"]]
            ),
        )
        for world in ordered
    ]


def reconstruct_repair(cases, rows, baseline, generations, repo_root):
    """Audit the unchanged four-condition complete-answer grid independently."""
    selected = independently_select_cases(rows, baseline, repo_root)
    if cases != selected:
        raise ValueError(
            "Selected cases differ from independent full-cohort eligibility"
        )
    found, old, parsed, outcomes = checked_inputs(rows, baseline, repo_root)
    case_lookup, generated = lookup(cases, "case_id"), lookup(generations)
    grid = {
        (record["case_id"], record["condition"], record["field"]): record
        for record in generations
    }
    if (
        len(generations) != len(cases) * len(CONDITIONS) * len(FIELDS)
        or len(grid) != len(generations)
        or set(grid) != set(itertools.product(case_lookup, CONDITIONS, FIELDS))
    ):
        raise ValueError(
            "Every selected case needs the complete four-condition generation grid"
        )
    initial = Counter(
        recipient_wrong=0, recipient_protected=0, recipient_fields=0, donor_correct=0
    )
    for case in cases:
        for field in FIELDS:
            correct = outcomes[case["recipient_row_ids"][field]]["correct"]
            initial["recipient_protected" if correct else "recipient_wrong"] += 1
            initial["recipient_fields"] += 1
            initial["donor_correct"] += 1
    counts = {
        condition: Counter(
            correct=0,
            count=0,
            repaired_initial_errors=0,
            protected_damaged=0,
            truncated=0,
        )
        for condition in CONDITIONS
    }
    repair_worlds = {condition: set() for condition in CONDITIONS}
    per_world = {
        condition: {
            case["world_id"]: {
                "repaired": 0,
                "initial_errors": len(case["initial_wrong_fields"]),
                "protected": len(case["protected_correct_fields"]),
                "damaged": 0,
                "correct": 0,
            }
            for case in cases
        }
        for condition in CONDITIONS
    }
    for record in generated.values():
        case, field, condition = (
            case_lookup[record["case_id"]],
            record["field"],
            record["condition"],
        )
        row_id = case["recipient_row_ids"][field]
        truth = parsed[row_id][0]
        if (
            record.get("row_id") != row_id
            or record.get("world_id") != case["world_id"]
            or found[row_id]["world_id"] != case["world_id"]
            or record.get("expected_answer") != truth
        ):
            raise ValueError(
                "Repair world, row, field or truth differs from printed question"
            )
        finished = completed(record)
        if condition == "selfRfield" and any(
            record[key] != old[row_id][key]
            for key in (
                "generated_text",
                "generated_ids",
                "terminated",
                "truncated",
                "stop_reason",
            )
        ):
            raise ValueError("Self replay differs from its complete baseline answer")
        correct = finished and normalize(record["generated_text"]) == normalize(truth)
        before = outcomes[row_id]["correct"]
        repaired, damaged = not before and correct, before and not correct
        counts[condition].update(
            correct=int(correct),
            count=1,
            repaired_initial_errors=int(repaired),
            protected_damaged=int(damaged),
            truncated=int(record["truncated"]),
        )
        world = per_world[condition][case["world_id"]]
        world["repaired"] += repaired
        world["damaged"] += damaged
        world["correct"] += correct
        if repaired:
            repair_worlds[condition].add(case["world_id"])
    for condition in CONDITIONS:
        for values in per_world[condition].values():
            values["repair_fraction"] = values["repaired"] / values["initial_errors"]
    return {
        "independent_literal_repair_audit": True,
        "eligible_worlds": len(cases),
        "generations": len(generations),
        "baseline": dict(initial),
        "by_condition": {
            condition: {
                **dict(counts[condition]),
                "repair_worlds": sorted(repair_worlds[condition]),
                "by_world": per_world[condition],
            }
            for condition in CONDITIONS
        },
    }


def compare_behavior(independent, reported):
    """Compare exact counts only; this parser is not a second statistics engine."""
    if reported.get("independent_worlds") != 64 or reported.get("prompts") != 6144:
        raise ValueError(
            "Reported behavioral sample size differs from literal coverage"
        )
    for key in (
        "fact_errors",
        "wrong_object_worlds",
        "wrong_object_coreference_worlds",
        "world_summaries",
    ):
        if reported.get(key) != independent[key]:
            raise ValueError(f"Reported behavioral counts differ: {key}")
    for kind, counts in independent["by_probe_kind"].items():
        if any(
            reported.get("by_probe_kind", {}).get(kind, {}).get(key) != value
            for key, value in counts.items()
        ):
            raise ValueError(
                "Reported question-type counts differ from literal answers"
            )
    if set(reported.get("by_probe_kind", {})) != set(independent["by_probe_kind"]):
        raise ValueError("Reported question types differ from literal coverage")
    if any(
        reported.get("coreference", {}).get(key) != value
        for key, value in independent["coreference"].items()
    ):
        raise ValueError("Reported paired-name counts differ from literal answers")
    return True


def compare_repair(independent, reported):
    if reported.get("eligible_worlds") != independent["eligible_worlds"]:
        raise ValueError("Reported repair cohort differs from independent eligibility")
    if set(reported.get("by_condition", {})) != set(CONDITIONS):
        raise ValueError(
            "Reported repair conditions differ from the fixed four conditions"
        )
    for condition, counts in independent["by_condition"].items():
        declared = reported["by_condition"][condition]
        expected = {
            "complete_answers": counts["count"],
            "correct": counts["correct"],
            "repaired_initial_errors": counts["repaired_initial_errors"],
            "protected_damaged": counts["protected_damaged"],
            "initial_errors": independent["baseline"]["recipient_wrong"],
            "protected_facts": independent["baseline"]["recipient_protected"],
            "by_world": counts["by_world"],
        }
        if any(declared.get(key) != value for key, value in expected.items()):
            raise ValueError(
                f"Reported repair counts differ from literal answers: {condition}"
            )
    return True


def verified_json(path, expected=None):
    path = Path(path)
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if (
        digest != Path(str(path) + ".sha256").read_text().strip()
        or expected is not None
        and digest != expected
    ):
        raise ValueError(f"Input hash mismatch: {path.name}")
    return json.loads(content), digest


def file_sha256(path):
    """Hash large native artifacts without keeping a whole shard in memory."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verified_stage(study_root, name, registration_hash):
    """Authenticate saved custody and the separately produced native receipt."""
    root = Path(study_root) / name
    custody, custody_hash = verified_json(root / "CUSTODY.json")
    if custody.get("registration_sha256") != registration_hash:
        raise ValueError("Stage custody refers to another registration")
    for artifact, digest in custody["artifact_sha256"].items():
        if Path(artifact).name != artifact:
            raise ValueError("Custody artifact escaped the stage namespace")
        path = root / artifact
        if file_sha256(path) != digest:
            raise ValueError(f"Custody hash mismatch: {artifact}")
    records, hashes = {}, {"custody_sha256": custody_hash}
    for artifact in ("GENERATIONS.json", "RESULT.json") + (
        ("CASES.json", "EXECUTION.json") if name == "repair" else ()
    ):
        if artifact not in custody["artifact_sha256"]:
            raise ValueError("Required result artifact is absent from stage custody")
        records[artifact], hashes[artifact] = verified_json(
            root / artifact, custody["artifact_sha256"][artifact]
        )
    native, hashes["native_audit_sha256"] = verified_json(root / "NATIVE_AUDIT.json")
    if (
        native.get("exact_match") is not True
        or native.get("custody_sha256") != custody_hash
        or native.get("generations") != len(records["GENERATIONS.json"])
        or name != "repair"
        and (native.get("required_rows") != 6144 or native.get("required_worlds") != 64)
        or name == "repair"
        and native.get("cases") != len(records["CASES.json"])
    ):
        raise ValueError(
            "Separate native audit is missing, incomplete or bound to another stage"
        )
    return records, hashes


def run_audit(study_root, repo_root, output):
    """Write a single-use receipt after both models and the fixed repair finish."""
    root, repo, target = Path(study_root), Path(repo_root).resolve(), Path(output)
    sidecar = Path(str(target) + ".sha256")
    if target.exists() or sidecar.exists():
        raise FileExistsError("Literal confirmation audit output is single-use")
    registration, registration_hash = verified_json(root / "REGISTRATION.json")
    registered, hashes = {}, {"registration_sha256": registration_hash}
    for name, digest in registration["input_sha256"].items():
        if Path(name).name != name:
            raise ValueError("Registered input escaped the study namespace")
        registered[name], hashes[name] = verified_json(root / name, digest)
    if not {
        "PLAN.json",
        "ROWS.json",
        "WORLDS.json",
        "CODE.json",
        "TOKENS-qwen.json",
        "TOKENS-phi.json",
    } <= set(registered):
        raise ValueError("Complete registered study inputs are required")
    # The receipt checks its own registered source as well as both historical
    # pure sources. Relocation is allowed; changing source bytes is not.
    for relative, digest in registered["CODE.json"].items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Registered source escaped the repository namespace")
        if (
            hashlib.sha256((repo / path).read_bytes()).hexdigest() != digest
            or hashlib.sha256((root / "sources" / path).read_bytes()).hexdigest()
            != digest
        ):
            raise ValueError("Registered scientific source bytes changed")
    own_path = "experiments/error_reference_confirmation_20261116/literal_audit.py"
    if (
        registered["CODE.json"].get(own_path)
        != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    ):
        raise ValueError(
            "Independent audit source was not part of the pre-run source freeze"
        )
    behavior, predictions, stages = {}, {}, {}
    for model in ("qwen", "phi"):
        records, stages[model] = verified_stage(root, model, registration_hash)
        predictions[model] = records["GENERATIONS.json"]
        behavior[model] = reconstruct_behavior(
            registered["ROWS.json"], predictions[model], repo
        )
        compare_behavior(behavior[model], records["RESULT.json"])
    records, stages["repair"] = verified_stage(root, "repair", registration_hash)
    if (
        records["EXECUTION.json"].get("baseline_audit_sha256")
        != stages["qwen"]["native_audit_sha256"]
    ):
        raise ValueError("Repair execution is bound to another baseline audit")
    repair = reconstruct_repair(
        records["CASES.json"],
        registered["ROWS.json"],
        predictions["qwen"],
        records["GENERATIONS.json"],
        repo,
    )
    compare_repair(repair, records["RESULT.json"])
    receipt = {
        "independent_literal_confirmation_audit": True,
        "all_reported_counts_match": True,
        "behavior": behavior,
        "repair": repair,
        "input_sha256": hashes,
        "stages": stages,
        "protocol_commit": registration["protocol_commit"],
        "literal_parser_source_sha256": PARSER_SHA256,
        "generator_source_sha256": GENERATOR_SHA256,
        "literal_audit_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "audited_utc": datetime.now(UTC).isoformat(),
        "native_proof_claimed": False,
        "statistical_intervals_independently_recomputed": False,
        "scope": "Printed labels, full confirmation coverage, case eligibility, complete answers and exact behavioral/repair counts. Native receipts are authenticated but their tensor calculations are not repeated here. Statistical intervals require separate review.",
    }
    content = (
        json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    with target.open("xb") as stream:
        stream.write(content)
    with sidecar.open("x") as stream:
        stream.write(hashlib.sha256(content).hexdigest() + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    receipt = run_audit(arguments.study_root, arguments.repo_root, arguments.output)
    print(
        json.dumps(
            {
                "all_reported_counts_match": receipt["all_reported_counts_match"],
                "eligible_worlds": receipt["repair"]["eligible_worlds"],
            },
            sort_keys=True,
        )
    )
