"""Strict whole-answer scoring and descriptive, world-clustered error analysis."""

import math
import random
import string
from collections import Counter, defaultdict

import error_protocol as protocol


def normalize(value):
    return (
        " ".join(value.strip().casefold().split()).rstrip(string.punctuation).rstrip()
    )


def _rate(values):
    return {
        "correct": sum(values),
        "count": len(values),
        "accuracy": sum(values) / len(values) if values else None,
    }


def world_interval(values):
    """Percentile bootstrap of independent worlds, never individual questions."""
    if not values:
        raise ValueError("At least one independent world is required")
    rng, count = random.Random(20261114), len(values)
    samples = sorted(
        sum(values[rng.randrange(count)] for _ in values) / count for _ in range(2000)
    )

    def quantile(fraction):
        position = fraction * (len(samples) - 1)
        low, high = math.floor(position), math.ceil(position)
        return samples[low] + (samples[high] - samples[low]) * (position - low)

    return {
        "mean": sum(values) / count,
        "ci95": [quantile(0.025), quantile(0.975)],
        "worlds": count,
        "resamples": 2000,
        "unit": "world",
    }


def _cue_objects(row, world):
    """Overlapping cue matches are predictions, not assigned causal explanations."""
    alias_index = row["query_alias_index"]
    line_position = world["alias_order"].index(alias_index)
    focus_index = {"none": None, "coref_0": 0, "coref_1": 1, "distractor": 2}[
        row["focus_condition"]
    ]
    return {
        "focus_object": None
        if focus_index is None
        else world["alias_targets"][focus_index],
        "alias_line_record_position": world["object_order"][line_position],
        "alias_line_stable_id_ordinal": "ABCD"[line_position],
        "first_record": world["object_order"][0],
        "last_record": world["object_order"][-1],
    }


def score(rows, predictions, worlds):
    """Score the complete development grid; this function opens no held-out data."""
    row_lookup = {r["id"]: r for r in rows}
    prediction_lookup = {p["id"]: p for p in predictions}
    world_lookup = {w["world_id"]: w for w in worlds}
    if (
        len(row_lookup) != len(rows)
        or len(prediction_lookup) != len(predictions)
        or set(row_lookup) != set(prediction_lookup)
        or len(world_lookup) != len(worlds)
        or {r["world_id"] for r in rows} != set(world_lookup)
        or len(rows) != 3072
        or len(worlds) != 32
        or any(r["split"] != "development" for r in rows)
        or any(w["split"] != "development" for w in worlds)
        or set(Counter(r["world_id"] for r in rows).values()) != {96}
    ):
        raise ValueError(
            "Analysis requires exactly the complete 32-world development grid"
        )
    # Rebuild the model-free rows to prevent a saved expected-answer field from
    # silently defining success. Tokenizer-added metadata is deliberately ignored.
    declared = protocol.build_study()
    declared_rows = {
        r["id"]: r for r in declared["rows"] if r["split"] == "development"
    }
    declared_worlds = [w for w in declared["worlds"] if w["split"] == "development"]
    if world_lookup != {w["world_id"]: w for w in declared_worlds}:
        raise ValueError("World facts differ from the frozen model-free generator")
    for identity, declared_row in declared_rows.items():
        if identity not in row_lookup or any(
            row_lookup[identity].get(k) != v for k, v in declared_row.items()
        ):
            raise ValueError(
                "Rendered row or expected label differs from the frozen generator"
            )

    outcomes, by_kind, by_field, by_template = (
        {},
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
    )
    world_stats = {
        identity: {
            "accuracy": [],
            "fact_accuracy": [],
            "identity_accuracy": [],
            "status_accuracy": [],
            "wrong_object": 0,
            "discordant_coref": 0,
            "wrong_object_coref": 0,
            "coref_pairs": 0,
        }
        for identity in world_lookup
    }
    fact_errors, cue_matches = Counter(), Counter()
    for row in rows:
        prediction, world = prediction_lookup[row["id"]], world_lookup[row["world_id"]]
        if (
            type(prediction.get("terminated")) is not bool
            or type(prediction.get("truncated")) is not bool
            or prediction["terminated"] == prediction["truncated"]
        ):
            raise ValueError(
                "Each complete generation needs explicit, consistent termination metadata"
            )
        answer = normalize(prediction["generated_text"])
        finished = prediction["terminated"] and not prediction["truncated"]
        correct = finished and answer == normalize(row["expected_answer"])
        if (
            "expected_answer" in prediction
            and prediction["expected_answer"] != row["expected_answer"]
        ):
            raise ValueError("Saved expected answer changed")
        if "correct" in prediction and prediction["correct"] != correct:
            raise ValueError("Saved correctness differs from whole-answer scoring")
        outcome = {"correct": bool(correct), "answer": answer, "selected_object": None}
        outcomes[row["id"]] = outcome
        by_kind[row["probe_kind"]].append(bool(correct))
        by_template[str(row["template"])].append(bool(correct))
        stats = world_stats[row["world_id"]]
        stats["accuracy"].append(bool(correct))
        stats[row["probe_kind"] + "_accuracy"].append(bool(correct))
        if row["probe_kind"] != "fact":
            continue
        by_field[row["field"]].append(bool(correct))
        matches = [
            obj
            for obj in "ABCD"
            if normalize(world["objects"][obj][row["field"]]) == answer
        ]
        if finished and matches:
            outcome["selected_object"] = matches[0]
        if not correct:
            if outcome["selected_object"] is None:
                fact_errors["unlisted_or_malformed"] += 1
            else:
                fact_errors["wrong_object"] += 1
                stats["wrong_object"] += 1
                for cue, obj in _cue_objects(row, world).items():
                    cue_matches[cue] += obj == outcome["selected_object"]

    coref, groups = Counter(), defaultdict(dict)
    focus_groups, swap_groups = defaultdict(dict), defaultdict(dict)
    for row in rows:
        condition = (row["world_id"], row["name_swap"], row["focus_condition"])
        if row["probe_kind"] == "fact" and row["query_alias_index"] in (0, 1):
            groups[condition + (row["field"],)][row["query_alias_index"]] = outcomes[
                row["id"]
            ]
        key = (
            row["world_id"],
            row["name_swap"],
            row["probe_kind"],
            row["query_alias_index"],
            row["field"],
        )
        focus_groups[key][row["focus_condition"]] = outcomes[row["id"]]
        swap_key = (
            row["world_id"],
            row["focus_condition"],
            row["probe_kind"],
            row["query_alias_index"],
            row["field"],
        )
        swap_groups[swap_key][row["name_swap"]] = outcomes[row["id"]]
    for key, pair in groups.items():
        first, second = pair[0], pair[1]
        discordant = first["correct"] != second["correct"]
        wrong = first if not first["correct"] else second
        wrong_object_discordance = discordant and wrong["selected_object"] is not None
        coref["pair_count"] += 1
        coref["different_answers"] += first["answer"] != second["answer"]
        coref["discordant_one_correct"] += discordant
        coref["discordant_wrong_object"] += wrong_object_discordance
        coref["both_correct"] += first["correct"] and second["correct"]
        coref["both_wrong"] += not first["correct"] and not second["correct"]
        coref["first_only_wrong"] += not first["correct"] and second["correct"]
        coref["second_only_wrong"] += first["correct"] and not second["correct"]
        world_stats[key[0]]["discordant_coref"] += discordant
        world_stats[key[0]]["wrong_object_coref"] += wrong_object_discordance
        world_stats[key[0]]["coref_pairs"] += 1

    def comparison(pairs):
        counts = Counter(
            {
                "count": 0,
                "changed": 0,
                "correct_to_wrong": 0,
                "wrong_to_correct": 0,
                "wrong_to_different_wrong": 0,
            }
        )
        for first, second in pairs:
            changed = first["answer"] != second["answer"]
            counts["count"] += 1
            counts["changed"] += changed
            counts["correct_to_wrong"] += first["correct"] and not second["correct"]
            counts["wrong_to_correct"] += not first["correct"] and second["correct"]
            counts["wrong_to_different_wrong"] += (
                changed and not first["correct"] and not second["correct"]
            )
        return dict(counts)

    focus = {
        condition: comparison(
            (group["none"], group[condition]) for group in focus_groups.values()
        )
        for condition in protocol.FOCUS
        if condition != "none"
    }
    swaps = {
        "all_focus_conditions": comparison(
            (group[0], group[1]) for group in swap_groups.values()
        ),
        "no_focus_only": comparison(
            (group[0], group[1])
            for key, group in swap_groups.items()
            if key[1] == "none"
        ),
    }
    wrong_worlds = sum(s["wrong_object"] > 0 for s in world_stats.values())
    discordant_worlds = sum(s["wrong_object_coref"] > 0 for s in world_stats.values())
    gate = protocol.build_protocol()["feasibility_gate"]
    passed = (
        wrong_worlds >= gate["wrong_object_worlds_at_least"]
        and coref["discordant_wrong_object"] >= gate["discordant_coref_pairs_at_least"]
        and discordant_worlds >= gate["discordant_coref_worlds_at_least"]
    )
    intervals = {
        metric: world_interval(
            [sum(s[metric]) / len(s[metric]) for _, s in sorted(world_stats.items())]
        )
        for metric in (
            "accuracy",
            "fact_accuracy",
            "identity_accuracy",
            "status_accuracy",
        )
    }
    intervals["coref_discordance"] = world_interval(
        [
            s["discordant_coref"] / s["coref_pairs"]
            for _, s in sorted(world_stats.items())
        ]
    )
    return {
        "study_id": protocol.STUDY_ID,
        "overall": _rate([o["correct"] for o in outcomes.values()]),
        "by_probe_kind": {k: _rate(v) for k, v in sorted(by_kind.items())},
        "by_field": {k: _rate(v) for k, v in sorted(by_field.items())},
        "by_template": {k: _rate(v) for k, v in sorted(by_template.items())},
        "fact_errors": {
            key: fact_errors[key] for key in ("wrong_object", "unlisted_or_malformed")
        },
        "coreference": dict(coref),
        "focus_comparisons": focus,
        "name_swap_comparisons": swaps,
        "cue_matches_on_wrong_object": dict(cue_matches),
        "cue_match_interpretation": "Overlapping descriptive matches, not causal attribution or a fitted retrieval-mixture explanation.",
        "world_intervals": intervals,
        "interval_scope": "Descriptive development bootstrap only. A zero observed rate and degenerate bootstrap interval do not establish a zero population risk or a conservative damage upper bound.",
        "world_summaries": {
            key: {
                metric: _rate(values) if isinstance(values, list) else values
                for metric, values in stats.items()
            }
            for key, stats in sorted(world_stats.items())
        },
        "feasibility": {
            "passed": passed,
            "wrong_object_worlds": wrong_worlds,
            "discordant_coref_pairs": coref["discordant_wrong_object"],
            "discordant_coref_worlds": discordant_worlds,
            "criteria": gate,
            "interpretation": "Feasibility, not evidence of novelty or a paper-ready result.",
        },
        "causal_calls_authorized": False,
        "confirmation_model_calls_authorized": False,
    }
