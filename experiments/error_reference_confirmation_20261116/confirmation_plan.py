"""A new, model-free confirmation plan; old development guardrails stay closed.

The causal question is deliberately narrow: estimate the already-fixed repair,
not certify the unsuccessful discovery route or search for another component.
This module neither loads a model nor provisions a paid resource.
"""

import gzip
import hashlib
import importlib.util
import json
import math
import random
import string
from collections import Counter, defaultdict
from pathlib import Path

STUDY_ID = "reference-confirmation-20261116-v1"
SEED = "reference-causal-discovery-20261115-v1"
FIELDS = ("color", "shape", "home")
FOCUS = ("none", "coref_0", "coref_1", "distractor")
CONDITIONS = ("selfRfield", "primaryDcolor", "frozenDiscoveryMean", "Rcolor")
REPO = Path(__file__).resolve().parents[2]
HISTORICAL = Path("data/original/cloud_reference_error_metadata_20261115")
GENERATOR = Path("experiments/error_reference_study_20261114/error_protocol.py")
FROZEN_MEAN = (
    0.17506461211422675,
    0.6395382948200473,
    0.024789365765038294,
    0.16060772730068765,
)
INPUT_SHA256 = {
    "generator.py": "186e73abf1e541aaa1a23bc7728f3a4ccd61cde95fb319e23bc00e45bf669200",
    "WORLDS.json": "bf97679eef752f00430c165a3595fb04e490a3ec28a956f84bf92fc1335a88b7",
    "ROWS.json": "6dced70e9b20ba7059ea530d1200fd86d7e88a2d59de0e74fb51a90e38363085",
    "TOKENS-qwen.json": "986de881d50dd97830a8b7c872e84fcdbea0cd3eae83fc85ace13a9a6bfb33c6",
    "TOKENS-phi.json": "88b0f0a3e761366a8905fd7b77b9fe5d057893913ff9959d6fe0979374d2fd55",
    "MODELS.json": "55204f97c88fd3e01ad23a86fca4d05ee17fb5882328826258a20ebaadfb44dd",
    "PROTOCOL.json": "740fb5dc20129e2b51548e057716a4809b146a1be786f593a5d865ccbf322396",
    "old_validation_PLAN.json": "ea5d8256e18b9cd8f5ab7d5364b0c80609b31b5f4902b9f75dc27e0d8941fc4d",
    "old_validation_RESULT.json": "ffadf2f4098bd079542599055c0e8a30898d76754ce60ba25becde047a51cbbb",
    "old_decomposition_RESULT.json": "c468bcf6581e11ccded7ac1e611bc543f764882646a45e640eb8d9de73e9d77d",
}
SAMPLING_ASSUMPTION = (
    "Intervals use an independent-world sampling approximation for this artificial "
    "task generator. Balanced margins and unique field assignments couple the "
    "generated worlds, so these are model-based reference intervals, not "
    "unconditional guarantees. Fixed-sample counts are exact descriptive results. "
    "Neither repeated questions nor the two models create additional independent worlds."
)


def build_plan():
    """Return all scientific choices before any reserved model answer is opened."""
    return {
        "study_id": STUDY_ID,
        "purpose": "Independent behavioral repetition and estimation of one unchanged failed repair candidate.",
        "old_positive_mechanism_gate_passed": False,
        "old_results_preserved": "The 1/8 development repair and failed follow-up remain failed; this new plan does not promote either to success.",
        "models": {
            "qwen": {
                "model_id": "Qwen/Qwen3-4B-Instruct-2507",
                "revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
            },
            "phi": {
                "model_id": "microsoft/Phi-3.5-mini-instruct",
                "revision": "2fe192450127e6a83f7441aef6e3ca586c338b77",
            },
        },
        "expected_input_sha256": dict(INPUT_SHA256),
        "behavior": {
            "worlds": 64,
            "questions_per_model": 6144,
            "factual_questions_per_model": 4608,
            "equivalent_name_pairs_per_model": 1536,
            "primary": "Strict correct-versus-finished-wrong-object disagreement counts, pair rates and affected-world prevalence, separately for both models.",
            "same_worlds_for_both_models": True,
            "publication_or_mechanism_pass_gate": None,
            "reporting": "Keep the complete fixed grid, all errors, all worlds and both models; no favorable subgroup replaces the primary result.",
        },
        "selected_head": {"layer_index": 28, "head_index": 0},
        "frozen_discovery_mean_Dcolor": list(FROZEN_MEAN),
        "conditions": list(CONDITIONS),
        "causal_model": "qwen",
        "case_selection": {
            "seed": SEED,
            "eligibility": "Donor strictly correct on all three facts and its identity; recipient has at least one finished wrong-object fact. All initially wrong recipient facts count, not only the screening fact.",
            "one_context_per_eligible_world": True,
            "selection": "Old SHA256 world/context/screen-field ordering, applied to every eligible confirmation world using only unchanged answers.",
            "replacements_or_outcome_based_stopping": False,
            "scope": "Conditional on eligible correct-source/wrong-recipient contexts in the same 64 worlds used for behavior; not a second independent dataset.",
        },
        "repair": {
            "primary": "Copy the correct donor's color-question record shares unchanged across the three recipient property questions.",
            "preserved": "Recipient total record attention, within-record relative token weights, nonrecord attention and recipient value vectors.",
            "generation": "One first-step delta per destination question, reused at its original query position for all recomputed greedy steps; no cache or teacher forcing.",
            "source_sees_color": True,
            "no_field_blind_or_general_entity_mechanism_claim": True,
            "baseline": "Saved complete unmodified behavior, not an extra condition chosen after intervention outcomes.",
        },
        "numeric": {
            "max_query_head_drift": 2e-5,
            "first_step_bitwise_equivalence": True,
            "self_whole_generation_bitwise": True,
            "source_record_mass_at_least": 1e-4,
            "recipient_each_record_mass_at_least": 1e-8,
            "unsupported_or_failed_oracle": "Stop the registered repair stage and retain the failure; no smoothing, case replacement, head search or relaxed tolerance.",
        },
        "useful_world_balanced_repair_rate": 0.50,
        "minimum_eligible_worlds_for_usefulness_inference": 20,
        "usefulness_estimand": "Mean across eligible worlds of the fraction of that world's initially wrong recipient facts repaired. This world-balanced estimand differs from the earlier pooled 4/8 engineering target; both are reported, but the old gate is not redefined.",
        "usefulness_decision": "With at least 20 eligible worlds: the targeted repair fraction is excluded only if the conservative 95% bounded-world upper interval is below 0.50; above-threshold repair benefit alone is supported only if its lower interval is at least 0.50; otherwise uncertain. Benefit alone is not a useful overall fix if it damages correct answers or lacks an advantage over the frozen-mean control. With fewer than 20 worlds report imprecision, not a qualified candidate.",
        "overall_utility": "Report point-estimate practical requirements separately: world-balanced repair at least 0.50, advantage over frozen mean at least 0.25, and protected-world damage at most 0.10. The 0.25 advantage is a newly specified world-balanced comparison, not a redefinition of the earlier pooled 2/8 gate. Separate individual 95% intervals do not establish a simultaneous 95% overall-utility claim; no positive overall claim is automatically authorized.",
        "preservation": "Report damaged protected facts and worlds separately. Compute an exact one-sided 95% binomial-reference upper bound on affected-world risk. A 10% risk threshold is reported, never assumed from zero observed damage.",
        "statistics": {
            "behavior_bootstrap": {
                "resamples": 2000,
                "seed": 20261116,
                "unit": "world",
                "scope": "descriptive",
            },
            "repair_bounds": "Hoeffding bounded-world mean intervals on [0,1]; paired primary-minus-frozen-mean intervals on [-1,1], under the stated sampling approximation.",
            "world_prevalence": "Two-sided 95% Clopper-Pearson binomial reference bounds, including nonzero bounds after zero observed events.",
            "family_scope": "Each interval is individual, not simultaneous across models, endpoints or controls. No family-wide confidence or significance claim.",
            "sampling_assumption": SAMPLING_ASSUMPTION,
        },
        "generation": {
            "max_new_tokens": 8,
            "batch_size": 1,
            "dtype": "float32",
            "attention": "eager",
            "tf32": False,
            "cache": False,
            "answer_rule": "Complete decoded response with case/whitespace/final punctuation normalization; unfinished responses and explanations remain errors.",
        },
        "model_limits_bytes": {
            "qwen": 32_000_000_000,
            "phi": 8_000_000_000,
            "repair": 5_000_000_000,
        },
        "execution": "Publish and freeze this protocol, complete source closure, input digests and model pins before the first pretrained confirmation forward. Require separate bounded spending approval, a recorded deadline and sufficient persistent storage. Do not alter archived runners or their development-only guards.",
        "completion": "Report completed, failed and incomplete stages regardless of outcome. A compute deadline or insufficient sample is not evidence that the repair never works; no further search is automatically authorized.",
    }


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _authenticated(content, name):
    if hashlib.sha256(content).hexdigest() != INPUT_SHA256[name]:
        raise ValueError(f"Frozen input checksum mismatch: {name}")
    return json.loads(content)


def _study(repo_root=REPO):
    source = Path(repo_root) / GENERATOR
    content = source.read_bytes()
    if hashlib.sha256(content).hexdigest() != INPUT_SHA256["generator.py"]:
        raise ValueError("Frozen generator checksum mismatch")
    # Only the checksum-authenticated, model-free local source is executable.
    specification = importlib.util.spec_from_file_location("_frozen_confirmation_generator", source)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module.build_study()


def confirmation_inputs(repo_root):
    """Authenticate historical bytes, then expose exactly the untouched split.

    Token dictionaries retain all 9,216 rows so their historical hashes remain
    verifiable. The runner, rather than mutating these inputs, selects the 6,144
    confirmation rows and reattaches locally verified cache paths separately.
    """
    root, study = Path(repo_root), _study(repo_root)
    historical = root / HISTORICAL
    worlds = _authenticated((historical / "WORLDS.json").read_bytes(), "WORLDS.json")
    if worlds != study["worlds"]:
        raise ValueError("Frozen world data differ from their authenticated generator")
    _authenticated(_json_bytes(study["rows"]), "ROWS.json")
    for name, expected in (("PROTOCOL.json", study["protocol"]), ("MODELS.json", None)):
        parsed = _authenticated((historical / name).read_bytes(), name)
        if expected is not None and parsed != expected:
            raise ValueError("Historical protocol differs from its original generator")
    tokens = {}
    for model in ("qwen", "phi"):
        name = f"TOKENS-{model}.json"
        if (historical / f"{name}.sha256").read_text().strip() != INPUT_SHA256[name]:
            raise ValueError("Historical token sidecar checksum changed")
        with gzip.open(historical / f"{name}.gz", "rb") as handle:
            tokens[model] = _authenticated(handle.read(), name)
        token_rows = {row["id"]: row for row in tokens[model]["rows"]}
        if len(token_rows) != 9216 or len(tokens[model]["rows"]) != 9216:
            raise ValueError("Every historical token row is required exactly once")
        for row in study["rows"]:
            if row["id"] not in token_rows or any(
                token_rows[row["id"]].get(key) != value for key, value in row.items()
            ):
                raise ValueError("Historical token rows differ from frozen model-free prompts")
    validation_root = root / "data/original/cloud_reference_validation_metadata_20261115"
    prior = _authenticated((validation_root / "PLAN.json").read_bytes(), "old_validation_PLAN.json")
    failed = _authenticated(
        (validation_root / "validation/RESULT.json").read_bytes(), "old_validation_RESULT.json"
    )
    decomposition = _authenticated(
        (
            root
            / "data/original/cloud_reference_decomposition_metadata_20261115/decomposition/RESULT.json"
        ).read_bytes(),
        "old_decomposition_RESULT.json",
    )
    if (
        prior["selected_head"] != {"layer_index": 28, "head_index": 0}
        or prior["frozen_discovery_mean_Dcolor"] != list(FROZEN_MEAN)
        or failed["assessment"]["strong_candidate_passed"] is not False
        or decomposition["assessment"]["all_fixed_branches_failed"] is not True
    ):
        raise ValueError("Prior fixed candidate or failed outcome changed")
    return {
        "worlds": [world for world in worlds if world["split"] == "confirmation"],
        "rows": [row for row in study["rows"] if row["split"] == "confirmation"],
        "tokens": tokens,
        "input_sha256": dict(INPUT_SHA256),
    }


def normalize(value):
    return " ".join(value.strip().casefold().split()).rstrip(string.punctuation).rstrip()


def _correct(row, prediction):
    if (
        type(prediction.get("terminated")) is not bool
        or type(prediction.get("truncated")) is not bool
        or prediction["terminated"] == prediction["truncated"]
        or not isinstance(prediction.get("generated_text"), str)
    ):
        raise ValueError("Explicit consistent termination metadata and answer text required")
    correct = prediction["terminated"] and normalize(prediction["generated_text"]) == normalize(
        row["expected_answer"]
    )
    if "expected_answer" in prediction and prediction["expected_answer"] != row["expected_answer"]:
        raise ValueError("Saved expected answer differs from the frozen prompt")
    if "correct" in prediction and prediction["correct"] != correct:
        raise ValueError("Saved correctness differs from strict whole-answer scoring")
    return bool(correct)


def _grid(rows, predictions, worlds):
    declared = _study()
    expected_rows = {row["id"]: row for row in declared["rows"] if row["split"] == "confirmation"}
    expected_worlds = {
        world["world_id"]: world for world in declared["worlds"] if world["split"] == "confirmation"
    }
    by_row = {row["id"]: row for row in rows}
    predicted = {prediction["id"]: prediction for prediction in predictions}
    by_world = {world["world_id"]: world for world in worlds}
    if (
        len(rows) != 6144
        or len(by_row) != 6144
        or set(by_row) != set(expected_rows)
        or len(predictions) != 6144
        or set(predicted) != set(expected_rows)
        or len(worlds) != 64
        or by_world != expected_worlds
        or any(
            any(row.get(key) != value for key, value in expected_rows[identity].items())
            for identity, row in by_row.items()
        )
    ):
        raise ValueError("Exactly the complete unchanged 64-world confirmation grid is required")
    correctness = {identity: _correct(row, predicted[identity]) for identity, row in by_row.items()}
    return by_row, predicted, by_world, correctness


def _rate(successes, count):
    return {"correct": successes, "count": count, "accuracy": successes / count if count else None}


def _bootstrap(values):
    if not values:
        return {"mean": None, "ci95": None, "worlds": 0, "resamples": 2000}
    rng, count = random.Random(20261116), len(values)
    samples = sorted(sum(values[rng.randrange(count)] for _ in values) / count for _ in range(2000))

    def quantile(fraction):
        position = fraction * (len(samples) - 1)
        low, high = math.floor(position), math.ceil(position)
        return samples[low] + (samples[high] - samples[low]) * (position - low)

    return {
        "mean": sum(values) / count,
        "ci95": [quantile(0.025), quantile(0.975)],
        "worlds": count,
        "resamples": 2000,
        "scope": "descriptive world bootstrap",
    }


def _binomial_cdf(successes, count, probability):
    return sum(
        math.comb(count, index) * probability**index * (1 - probability) ** (count - index)
        for index in range(successes + 1)
    )


def _upper_binomial(successes, count, alpha):
    if successes == count:
        return 1.0
    low, high = 0.0, 1.0
    for _ in range(80):
        midpoint = (low + high) / 2
        if _binomial_cdf(successes, count, midpoint) > alpha:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2


def _binary_interval(successes, count):
    if not count:
        return {"events": successes, "worlds": 0, "lower": None, "upper": None, "confidence": 0.95}
    return {
        "events": successes,
        "worlds": count,
        "lower": 0.0 if not successes else 1 - _upper_binomial(count - successes, count, 0.025),
        "upper": _upper_binomial(successes, count, 0.025),
        "confidence": 0.95,
        "method": "Clopper-Pearson binomial reference interval under the stated world-sampling approximation",
    }


def _bounded_interval(values, low=0.0, high=1.0):
    count = len(values)
    if not count:
        return {"mean": None, "ci95": None, "worlds": 0}
    mean = sum(values) / count
    radius = (high - low) * math.sqrt(math.log(2 / 0.05) / (2 * count))
    return {
        "mean": mean,
        "ci95": [max(low, mean - radius), min(high, mean + radius)],
        "worlds": count,
        "method": "Hoeffding bounded-world interval under the stated independent-world approximation",
    }


def score_behavior(rows, predictions, worlds):
    """Score the complete new behavioral sample without a publication pass flag."""
    by_row, predicted, by_world, correctness = _grid(rows, predictions, worlds)
    counts, totals, fact_errors = Counter(), Counter(), Counter()
    summaries = {
        identity: {
            "fact_correct": 0,
            "facts": 0,
            "wrong_object": 0,
            "coref_pairs": 0,
            "wrong_object_coref": 0,
        }
        for identity in by_world
    }
    selected, pairs = {}, defaultdict(dict)
    for identity, row in by_row.items():
        kind, world = row["probe_kind"], by_world[row["world_id"]]
        totals[kind] += 1
        counts[kind] += correctness[identity]
        if kind != "fact":
            continue
        summary = summaries[row["world_id"]]
        summary["facts"] += 1
        summary["fact_correct"] += correctness[identity]
        answer = normalize(predicted[identity]["generated_text"])
        matches = [
            obj for obj in "ABCD" if normalize(world["objects"][obj][row["field"]]) == answer
        ]
        selected[identity] = matches[0] if predicted[identity]["terminated"] and matches else None
        if not correctness[identity]:
            wrong_object = selected[identity] is not None
            fact_errors["wrong_object" if wrong_object else "unlisted_or_malformed"] += 1
            summary["wrong_object"] += wrong_object
        if row["query_alias_index"] in (0, 1):
            pairs[(row["world_id"], row["name_swap"], row["focus_condition"], row["field"])][
                row["query_alias_index"]
            ] = identity
    coreference = Counter()
    for key, pair in pairs.items():
        first, second = pair[0], pair[1]
        discordant = correctness[first] != correctness[second]
        wrong = first if not correctness[first] else second
        wrong_object_pair = discordant and selected[wrong] is not None
        coreference["pair_count"] += 1
        coreference["discordant_one_correct"] += discordant
        coreference["discordant_wrong_object"] += wrong_object_pair
        coreference["different_answers"] += normalize(
            predicted[first]["generated_text"]
        ) != normalize(predicted[second]["generated_text"])
        coreference["both_correct"] += correctness[first] and correctness[second]
        coreference["both_wrong"] += not correctness[first] and not correctness[second]
        summaries[key[0]]["coref_pairs"] += 1
        summaries[key[0]]["wrong_object_coref"] += wrong_object_pair
    wrong_worlds = sum(summary["wrong_object"] > 0 for summary in summaries.values())
    paired_worlds = sum(summary["wrong_object_coref"] > 0 for summary in summaries.values())
    ordered = [summary for _, summary in sorted(summaries.items())]
    return {
        "study_id": STUDY_ID,
        "scope": "All 64 confirmation worlds; both models share them, not 128 independent worlds.",
        "independent_worlds": 64,
        "prompts": 6144,
        "by_probe_kind": {kind: _rate(counts[kind], totals[kind]) for kind in sorted(totals)},
        "fact_errors": {
            name: fact_errors[name] for name in ("wrong_object", "unlisted_or_malformed")
        },
        "coreference": dict(coreference),
        "wrong_object_worlds": wrong_worlds,
        "wrong_object_coreference_worlds": paired_worlds,
        "world_summaries": dict(sorted(summaries.items())),
        "world_intervals": {
            "fact_accuracy": _bootstrap(
                [summary["fact_correct"] / summary["facts"] for summary in ordered]
            ),
            "wrong_object_coreference_rate": _bootstrap(
                [summary["wrong_object_coref"] / summary["coref_pairs"] for summary in ordered]
            ),
        },
        "world_prevalence_intervals": {
            "wrong_object": _binary_interval(wrong_worlds, 64),
            "wrong_object_coreference": _binary_interval(paired_worlds, 64),
        },
        "sampling_assumption": SAMPLING_ASSUMPTION,
        "interval_scope": "Individual reference intervals, not simultaneous across models/endpoints; bootstrap intervals are descriptive and zero width is not a zero-risk guarantee.",
        "mechanism_identified": False,
    }


def select_cases(rows, predictions, worlds):
    """Use only baseline answers and the original deterministic selection rule."""
    by_row, predicted, by_world, correctness = _grid(rows, predictions, worlds)
    lookup = {
        (
            row["world_id"],
            row["name_swap"],
            row["focus_condition"],
            row["probe_kind"],
            row["query_alias_index"],
            row["field"],
        ): row
        for row in by_row.values()
    }
    eligible = defaultdict(list)
    for world_id, world in by_world.items():
        for swap in (0, 1):
            for focus in FOCUS:
                for recipient in (0, 1):
                    donor, group = 1 - recipient, (world_id, swap, focus)
                    donor_rows = {field: lookup[group + ("fact", donor, field)] for field in FIELDS}
                    recipient_rows = {
                        field: lookup[group + ("fact", recipient, field)] for field in FIELDS
                    }
                    donor_identity = lookup[group + ("identity", donor, None)]
                    if not all(
                        correctness[row["id"]] for row in [*donor_rows.values(), donor_identity]
                    ):
                        continue
                    wrong = {}
                    for field, row in recipient_rows.items():
                        prediction = predicted[row["id"]]
                        matches = [
                            obj
                            for obj in "ABCD"
                            if normalize(world["objects"][obj][field])
                            == normalize(prediction["generated_text"])
                        ]
                        if (
                            not correctness[row["id"]]
                            and prediction["terminated"]
                            and len(matches) == 1
                        ):
                            wrong[field] = matches[0]
                    if not wrong:
                        continue
                    identity = f"{world_id}:swap{swap}:{focus}:recipient{recipient}"
                    screen = min(wrong, key=lambda field: _hash([SEED, identity, field]))
                    eligible[world_id].append(
                        {
                            "case_id": identity,
                            "world_id": world_id,
                            "correct_object": world["base_object_id"],
                            "screen_field": screen,
                            "reverse_object": wrong[screen],
                            "wrong_object_by_field": wrong,
                            "initial_wrong_fields": [
                                field
                                for field in FIELDS
                                if not correctness[recipient_rows[field]["id"]]
                            ],
                            "protected_correct_fields": [
                                field
                                for field in FIELDS
                                if correctness[recipient_rows[field]["id"]]
                            ],
                            "donor_row_ids": {
                                field: row["id"] for field, row in donor_rows.items()
                            },
                            "recipient_row_ids": {
                                field: row["id"] for field, row in recipient_rows.items()
                            },
                            "screen_row_id": recipient_rows[screen]["id"],
                        }
                    )
    return [
        min(eligible[world], key=lambda case: _hash([SEED, "condition", case["case_id"]]))
        for world in sorted(eligible, key=lambda identity: _hash([SEED, "world", identity]))
    ]


def assess_repair(cases, rows, baseline_predictions, generations):
    """Assess every registered recipient fact, with no sample or condition rescue."""
    study = _study()
    worlds = [world for world in study["worlds"] if world["split"] == "confirmation"]
    expected_cases = select_cases(rows, baseline_predictions, worlds)
    if cases != expected_cases:
        raise ValueError("Frozen selected cases differ from complete unchanged behavior")
    by_row = {row["id"]: row for row in rows}
    baseline = {prediction["id"]: prediction for prediction in baseline_predictions}
    intended = {
        (case["case_id"], condition, field)
        for case in cases
        for condition in CONDITIONS
        for field in FIELDS
    }
    observed = {
        (generation["case_id"], generation["condition"], generation["field"]): generation
        for generation in generations
    }
    if len(generations) != len(intended) or set(observed) != intended:
        raise ValueError("Complete four-condition, three-field generation grid required")
    by_condition = {}
    world_repairs = {}
    for condition in CONDITIONS:
        repaired_total = damaged_total = correct_total = 0
        world_values, world_reports = [], {}
        damaged_worlds = protected_worlds = 0
        for case in cases:
            repaired, damaged, correct = 0, 0, 0
            for field in FIELDS:
                row = by_row[case["recipient_row_ids"][field]]
                generation = observed[(case["case_id"], condition, field)]
                if ("row_id" in generation and generation["row_id"] != row["id"]) or (
                    "world_id" in generation and generation["world_id"] != case["world_id"]
                ):
                    raise ValueError("Repair destination changed")
                correct_now = _correct(row, generation)
                if condition == "selfRfield":
                    original = baseline[row["id"]]
                    if any(
                        generation.get(key) != original.get(key)
                        for key in ("generated_text", "terminated", "truncated")
                    ) or (
                        "generated_ids" in generation
                        and "generated_ids" in original
                        and generation["generated_ids"] != original["generated_ids"]
                    ):
                        raise ValueError("Exact self replay failed")
                correct += correct_now
                repaired += field in case["initial_wrong_fields"] and correct_now
                damaged += field in case["protected_correct_fields"] and not correct_now
            initial_errors, protected = len(case["initial_wrong_fields"]), len(
                case["protected_correct_fields"]
            )
            fraction = repaired / initial_errors
            world_values.append(fraction)
            protected_worlds += protected > 0
            damaged_worlds += damaged > 0
            repaired_total += repaired
            damaged_total += damaged
            correct_total += correct
            world_reports[case["world_id"]] = {
                "repaired": repaired,
                "initial_errors": initial_errors,
                "repair_fraction": fraction,
                "protected": protected,
                "damaged": damaged,
                "correct": correct,
            }
        initial_errors = sum(len(case["initial_wrong_fields"]) for case in cases)
        protected = sum(len(case["protected_correct_fields"]) for case in cases)
        damage_upper = (
            _upper_binomial(damaged_worlds, protected_worlds, 0.05) if protected_worlds else None
        )
        by_condition[condition] = {
            "complete_answers": 3 * len(cases),
            "correct": correct_total,
            "repaired_initial_errors": repaired_total,
            "initial_errors": initial_errors,
            "pooled_repair_rate": repaired_total / initial_errors if initial_errors else None,
            "protected_damaged": damaged_total,
            "protected_facts": protected,
            "pooled_fact_damage_rate": damaged_total / protected if protected else None,
            "world_balanced_repair": _bounded_interval(world_values),
            "protected_world_damage": {
                "damaged_worlds": damaged_worlds,
                "protected_worlds": protected_worlds,
                "upper_one_sided_95": damage_upper,
                "risk_below_10_percent_demonstrated": damage_upper is not None
                and damage_upper < 0.10,
                "method": "Exact one-sided binomial reference bound under the stated sampling approximation",
            },
            "by_world": world_reports,
        }
        world_repairs[condition] = world_values
    primary = by_condition["primaryDcolor"]["world_balanced_repair"]
    if not cases:
        conclusion = "no_eligible_cases"
    elif len(cases) < 20:
        conclusion = "imprecise_low_eligible_world_count"
    elif primary["ci95"][1] < 0.50:
        conclusion = "useful_benefit_excluded"
    elif primary["ci95"][0] >= 0.50:
        conclusion = "repair_benefit_only_above_threshold"
    else:
        conclusion = "uncertain"
    differences = [
        first - second
        for first, second in zip(
            world_repairs["primaryDcolor"], world_repairs["frozenDiscoveryMean"]
        )
    ]
    paired_interval = _bounded_interval(differences, -1, 1)
    primary_damage = by_condition["primaryDcolor"]["protected_world_damage"]
    damage_point = (
        primary_damage["damaged_worlds"] / primary_damage["protected_worlds"]
        if primary_damage["protected_worlds"]
        else None
    )
    point_targets = bool(
        len(cases) >= 20
        and primary["mean"] >= 0.50
        and paired_interval["mean"] >= 0.25
        and damage_point is not None
        and damage_point <= 0.10
    )
    return {
        "study_id": STUDY_ID,
        "eligible_worlds": len(cases),
        "scope": "Conditional fixed-repair test on every eligible confirmation world; not an unconditional repair rate or a new mechanism discovery.",
        "by_condition": by_condition,
        "usefulness": {
            "conclusion": conclusion,
            "threshold": 0.50,
            "minimum_worlds": 20,
            "estimand": "Mean per-world fraction of initially wrong recipient facts repaired",
            "preservation_is_separate": True,
            "old_development_gate_redefined": False,
        },
        "primary_minus_frozen_mean": paired_interval,
        "overall_repair_utility": {
            "status": "not_established_by_separate_intervals",
            "minimum_advantage_over_frozen_mean": 0.25,
            "maximum_protected_world_damage": 0.10,
            "point_estimates_meet_all_practical_targets": point_targets,
            "simultaneous_95_percent_claim": False,
            "interpretation": "Correcting old errors alone does not establish a useful repair. Benefit, comparison-control advantage and preservation are separate endpoints; these individual reference intervals are not a joint confidence statement.",
        },
        "sampling_assumption": SAMPLING_ASSUMPTION,
        "interval_scope": "Each interval is individual, not a simultaneous claim across conditions/endpoints. Pooled facts are repeated measurements, not independent trials.",
        "old_positive_mechanism_gate_passed": False,
        "new_entity_mechanism_established": False,
        "further_search_authorized": False,
    }
