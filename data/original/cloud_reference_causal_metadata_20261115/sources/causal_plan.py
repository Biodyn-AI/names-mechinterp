"""Bounded, model-free discovery planning; never authorization or inference.

Inputs must first pass file/hash and native-logit verification outside this
module. The independent rescore receipt is checked here, not recreated from raw
files. Donors stop before any requested field. This excludes copying a requested
answer, not transferring a bundle of facts: routing/value factorization remains
necessary. Query-specific patches do not establish preservation of other aliases
merely by leaving those other queries unpatched. All screening is exploratory.
"""

import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "error_reference_study_20261114")
)
import error_analysis
import error_protocol

RAW_CAP = 1_000_000_000
SEED = "reference-causal-discovery-20261115-v1"


def _hash(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _verify(inputs):
    analysis = error_analysis.score(
        inputs["rows"], inputs["predictions"], inputs["worlds"]
    )
    receipt, custody = inputs["independent_rescore"], inputs["custody"]
    forwards = sum(len(p["generated_ids"]) for p in inputs["predictions"])
    if (
        analysis != inputs["behavioral_result"]
        or not analysis["feasibility"]["passed"]
        or receipt.get("exact_match") is not True
        or receipt.get("native_steps") != forwards
        or custody.get("physical_forward_count") != forwards
        or custody.get("completed_rows") != 3072
        or receipt.get("confirmation_calls") != 0
        or receipt.get("causal_calls") != 0
    ):
        raise ValueError(
            "A complete, independently rescored passed development gate is required"
        )
    for name in ("registration_sha256", "custody_sha256", "rescore_sha256"):
        value = inputs["provenance"][name]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError("Verified provenance must include SHA-256 receipt anchors")
    if (
        custody.get("registration_sha256")
        != inputs["provenance"]["registration_sha256"]
    ):
        raise ValueError("Behavioral custody is not bound to its registered inputs")
    model = inputs["model"]
    if (
        any(
            type(model[k]) is not int or model[k] <= 0
            for k in ("layers", "heads", "head_dim", "o_proj_in_features", "vocab_size")
        )
        or model["layers"] < 3
        or model["o_proj_in_features"] != model["heads"] * model["head_dim"]
    ):
        raise ValueError("Declared attention geometry is invalid")
    for field in error_protocol.FIELDS:
        seen = set()
        for label in error_protocol.BANKS[field]:
            ids = inputs["label_token_ids"][field].get(label)
            if (
                not isinstance(ids, list)
                or not ids
                or len(set(ids)) != len(ids)
                or any(
                    type(i) is not int or not 0 <= i < model["vocab_size"] for i in ids
                )
                or seen.intersection(ids)
            ):
                raise ValueError(
                    "Every field value needs disjoint complete-label one-token IDs"
                )
            seen.update(ids)


def _prefix(rows):
    """Require the exact same causal prefix for every future field and identity."""
    prefixes, positions, lengths = set(), set(), set()
    for row in rows:
        position = row["alias_end_position"]
        if type(position) is not int or not 0 <= position < len(row["input_ids"]):
            raise ValueError("Invalid alias-marker position")
        start, end = row["alias_end_span"]
        rendered = row["rendered_prompt"]
        if rendered.count(row["text"]) != 1:
            raise ValueError("Rendered question must occur exactly once")
        shift = rendered.index(row["text"])
        offsets = row["offset_mapping"]
        marker_tokens = [
            i
            for i, (left, right) in enumerate(offsets)
            if left < shift + end and right > shift + start
        ]
        if (
            len(offsets) != len(row["input_ids"])
            or not marker_tokens
            or marker_tokens[-1] != position
            or offsets[position][1] > shift + row["field_start"]
        ):
            raise ValueError(
                "Prefix includes a future field or differs from its marker"
            )
        prefix = row["input_ids"][: position + 1]
        if any(type(i) is not int or i < 0 for i in prefix):
            raise ValueError("Invalid prefix token IDs")
        prefixes.add(tuple(prefix))
        positions.add(position)
        if row["probe_kind"] == "fact":
            lengths.add((row["field"], len(row["input_ids"])))
    if len(prefixes) != 1 or len(positions) != 1:
        raise ValueError("Fields or identity do not share an exact alias prefix")
    return list(next(iter(prefixes))), positions.pop(), dict(lengths)


def plan(inputs):
    """Build an unexecuted plan from complete verified development measurements."""
    _verify(inputs)
    rows = {
        (
            r["world_id"],
            r["name_swap"],
            r["focus_condition"],
            r["probe_kind"],
            r["query_alias_index"],
            r["field"],
        ): r
        for r in inputs["rows"]
    }
    predictions = {p["id"]: p for p in inputs["predictions"]}

    def correct(row):
        prediction = predictions[row["id"]]
        return (
            prediction["terminated"]
            and not prediction["truncated"]
            and error_analysis.normalize(prediction["generated_text"])
            == error_analysis.normalize(row["expected_answer"])
        )

    eligible, rejections = {}, Counter()
    for world in inputs["worlds"]:
        for swap in (0, 1):
            for focus in error_protocol.FOCUS:
                for recipient in (0, 1):
                    donor = 1 - recipient
                    group = (world["world_id"], swap, focus)
                    donor_rows = [
                        rows[group + ("fact", donor, field)]
                        for field in error_protocol.FIELDS
                    ]
                    identity = rows[group + ("identity", donor, None)]
                    if not all(correct(r) for r in donor_rows + [identity]):
                        rejections["donor_not_all_correct"] += 1
                        continue
                    recipient_rows = [
                        rows[group + ("fact", recipient, field)]
                        for field in error_protocol.FIELDS
                    ]
                    wrong = {}
                    for row in recipient_rows:
                        answer = predictions[row["id"]]
                        if (
                            not correct(row)
                            and answer["terminated"]
                            and not answer["truncated"]
                        ):
                            matches = [
                                obj
                                for obj in "ABCD"
                                if error_analysis.normalize(
                                    world["objects"][obj][row["field"]]
                                )
                                == error_analysis.normalize(answer["generated_text"])
                            ]
                            if len(matches) == 1:
                                wrong[row["field"]] = matches[0]
                    if not wrong:
                        rejections["no_wrong_listed_field"] += 1
                        continue
                    donor_prefix, position, geometry = _prefix(donor_rows + [identity])
                    recipient_identity = rows[group + ("identity", recipient, None)]
                    recipient_prefix, other_position, other_geometry = _prefix(
                        recipient_rows + [recipient_identity]
                    )
                    if position != other_position or geometry != other_geometry:
                        raise ValueError("Co-referring query geometry differs")
                    case_id = (
                        f"{world['world_id']}:swap{swap}:{focus}:recipient{recipient}"
                    )
                    screen_field = min(wrong, key=lambda f: _hash([SEED, case_id, f]))
                    case = {
                        "case_id": case_id,
                        "world_id": world["world_id"],
                        "name_swap": swap,
                        "focus_condition": focus,
                        "donor_alias_index": donor,
                        "recipient_alias_index": recipient,
                        "donor_prefix_ids": donor_prefix,
                        "recipient_prefix_ids": recipient_prefix,
                        "alias_end_position": position,
                        "screen_field": screen_field,
                        "screen_row_id": rows[
                            group + ("fact", recipient, screen_field)
                        ]["id"],
                        "donor_row_ids": {r["field"]: r["id"] for r in donor_rows},
                        "recipient_row_ids": {
                            r["field"]: r["id"] for r in recipient_rows
                        },
                        "wrong_object_by_field": wrong,
                        "observed_wrong_objects_coherent": len(set(wrong.values()))
                        == 1,
                        "reverse_predicted_object": wrong[screen_field],
                        "reverse_predicted_answers": {
                            field: world["objects"][wrong[screen_field]][field]
                            for field in error_protocol.FIELDS
                        },
                        "correct_object": world["base_object_id"],
                    }
                    eligible.setdefault(world["world_id"], []).append(case)
    selected_worlds = sorted(eligible, key=lambda w: _hash([SEED, "world", w]))[:8]
    enough = len(selected_worlds) == 8
    cases = (
        [
            min(eligible[w], key=lambda c: _hash([SEED, "condition", c["case_id"]]))
            for w in selected_worlds
        ]
        if enough
        else []
    )
    layers, model = list(range(inputs["model"]["layers"] - 1)), inputs["model"]
    layer_jobs = [
        {
            "endpoint_id": f"layer::{c['case_id']}::l{layer:02d}",
            "case_id": c["case_id"],
            "row_id": c["screen_row_id"],
            "layer_index": layer,
            "head_index": None,
            "donor_role": "donor",
        }
        for c in cases
        for layer in layers
    ]
    captures = [
        {
            "endpoint_id": f"capture::{c['case_id']}::{role}",
            "case_id": c["case_id"],
            "role": role,
            "input_ids": c[f"{role}_prefix_ids"],
            "alias_end_position": c["alias_end_position"],
            "layers": layers,
            "forward_scope": "backbone_prefix_only_no_logits",
        }
        for c in cases
        for role in ("donor", "recipient")
    ]
    clean_jobs = [
        {
            "endpoint_id": f"clean::{c['case_id']}",
            "case_id": c["case_id"],
            "row_id": c["screen_row_id"],
        }
        for c in cases
    ]
    self_jobs = (
        [
            {
                "endpoint_id": f"layer-self::{case['case_id']}::l{layer:02d}",
                "case_id": case["case_id"],
                "row_id": case["screen_row_id"],
                "layer_index": layer,
                "head_index": None,
                "donor_role": "recipient",
            }
            for case in cases
            for layer in layers
        ]
        if enough
        else []
    )
    # Reserve the complete fixed next-stage grid and self-replacement oracles,
    # even though its two layer identities are not chosen until screening ends.
    head_count = 2 * model["heads"] * len(cases)
    oracle_count = len(cases) * len(layers) + 2 * model["heads"] if enough else 0
    endpoints = len(cases) + len(layer_jobs) + head_count + oracle_count
    estimated = (
        endpoints * model["vocab_size"] * 4
        + len(captures) * len(layers) * model["o_proj_in_features"] * 4
        + 64 * 1024**2
    )
    if estimated > RAW_CAP:
        raise ValueError(
            "Prospective complete discovery artifacts exceed the one-GB cap"
        )
    return {
        "study_id": SEED,
        "status": "ready_for_root_audit" if enough else "insufficient_eligible_worlds",
        "input_sha256": _hash(inputs),
        "provenance": inputs["provenance"],
        "model": model,
        "eligibility": {
            "candidate_conditions": len(inputs["worlds"]) * 16,
            "eligible_conditions": sum(map(len, eligible.values())),
            "eligible_worlds": len(eligible),
            "required_worlds": 8,
            "rejections": dict(rejections),
            "screen_world_ids": [c["world_id"] for c in cases],
            "unused_eligible_world_ids": sorted(
                set(eligible) - {c["world_id"] for c in cases},
                key=lambda w: _hash([SEED, "world", w]),
            ),
        },
        "cases": cases,
        "prefix_capture_jobs": captures,
        "clean_jobs": clean_jobs,
        "layer_self_jobs": self_jobs,
        "layer_jobs": layer_jobs,
        "layers": layers,
        "label_token_ids": inputs["label_token_ids"],
        "continuation": {
            "top_layers": 2,
            "all_heads_per_selected_layer": model["heads"],
            "maximum_selected_heads": 2,
            "selection_scope": "development discovery only",
            "positive_gain_stop": "Continue only candidates with strictly positive mean correct-minus-wrong margin gain; zero or negative best gain stops expansion.",
            "reverse_prediction_scope": "The screen field's particular wrong object predicts all three reverse answers prospectively. Mixed observed field errors do not establish a shared wrong record.",
        },
        "storage": {
            "estimated_bytes": estimated,
            "cap_bytes": RAW_CAP,
            "reserved_native_endpoints": endpoints,
            "reserved_prefix_backbone_forwards": len(captures),
            "reserved_head_jobs": head_count,
            "reserved_self_oracles": oracle_count,
        },
        "metric": "logsumexp(correct whole-label token IDs) - logsumexp(predicted wrong-object whole-label token IDs); retain full native logits",
        "scope": "First-token diagnostic only, not complete-answer repair, causal identification, or a paper-ready mechanism. Routing/value factorization and frozen confirmation remain required. Accepted label IDs must be exhaustive tokenizer-decoded whole-label sets verified by the caller; this pure planner does not authenticate external files or tokenizers.",
        "execution_authorized": False,
        "confirmation_calls_authorized": False,
    }


def _ranking(jobs, measurements, keys):
    expected = {j["endpoint_id"]: j for j in jobs}
    measured = {m["endpoint_id"]: m["correct_minus_wrong_gain"] for m in measurements}
    if (
        len(measured) != len(measurements)
        or set(expected) != set(measured)
        or not expected
    ):
        raise ValueError("Exactly every registered screening endpoint is required")
    totals, counts = Counter(), Counter()
    for endpoint, value in measured.items():
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("Screening gains must be finite numbers")
        key = tuple(expected[endpoint][k] for k in keys)
        totals[key] += value
        counts[key] += 1
    return [
        (key, totals[key] / counts[key])
        for key in sorted(totals, key=lambda k: (-totals[k] / counts[k], k))
    ]


def select_layers(planned, measurements):
    """Exploratory selection only; never read any confirmation outcome."""
    selected = [
        key[0]
        for key, gain in _ranking(planned["layer_jobs"], measurements, ("layer_index",))
        if gain > 0
    ][:2]
    jobs = [
        {
            "case_id": case["case_id"],
            "row_id": case["screen_row_id"],
            "donor_role": "donor",
            "layer_index": layer,
            "head_index": head,
            "endpoint_id": f"head::{case['case_id']}::l{layer:02d}::h{head:02d}",
        }
        for case in planned["cases"]
        for layer in selected
        for head in range(planned["model"]["heads"])
    ]
    self_jobs = [
        {
            **job,
            "donor_role": "recipient",
            "endpoint_id": f"head-self::l{job['layer_index']:02d}::h{job['head_index']:02d}",
        }
        for job in jobs
        if job["case_id"] == planned["cases"][0]["case_id"]
    ]
    return {
        **planned,
        "status": "ready_for_root_audit"
        if selected
        else "stopped_no_positive_layer_gain",
        "selected_layers": selected,
        "head_jobs": jobs,
        "head_self_jobs": self_jobs,
    }


def select_heads(planned, measurements):
    """Return at most two discovered heads, without authorizing further calls."""
    selected = [
        key
        for key, gain in _ranking(
            planned["head_jobs"], measurements, ("layer_index", "head_index")
        )
        if gain > 0
    ][:2]
    return {
        **planned,
        "status": "ready_for_root_audit"
        if selected
        else "stopped_no_positive_head_gain",
        "selected_heads": [
            {"layer_index": layer, "head_index": head} for layer, head in selected
        ],
    }
