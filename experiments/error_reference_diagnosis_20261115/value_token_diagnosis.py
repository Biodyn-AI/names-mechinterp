"""Saved-data diagnosis of one frozen head, with no model execution or writes.

Value-bearing BPE locations are not semantically pure value vectors. Attention
fractions alone cannot establish which information a token's V vector carries.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch

FIELDS = ("color", "shape", "home")
LAYER, HEAD = 28, 0
EXPECTED = {
    "cloud_reference_error_metadata_20261115/TOKENS-qwen.json": "986de881d50dd97830a8b7c872e84fcdbea0cd3eae83fc85ace13a9a6bfb33c6",
    "cloud_reference_error_metadata_20261115/WORLDS.json": "bf97679eef752f00430c165a3595fb04e490a3ec28a956f84bf92fc1335a88b7",
    "cloud_reference_routing_metadata_20261115/routing/NATURAL.pt": "cc13cc84640966b8508e6e7df6c8040caab2fdc3d571d44e395e59c6eb5fbfe3",
    "cloud_reference_validation_metadata_20261115/validation/NATURAL.pt": "589f70df0d15084bacf7d2ff8af9eee46713ae0c3a2879b6e51c11130bb91f1d",
    "cloud_reference_routing_metadata_20261115/routing/CUSTODY.json": "c88be9ba6bdb3cf9a36b6b68decd84a79b00ef8241b05d8d78712c94f8bc9491",
    "cloud_reference_validation_metadata_20261115/validation/CUSTODY.json": "0d4cddeb451317855f46844aba16f4966d9c5237e031b08301377575e96e78d2",
}


def verify(path, expected=None):
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if (
        digest != Path(str(path) + ".sha256").read_text().strip()
        or expected is not None
        and digest != expected
    ):
        raise ValueError(f"Input hash mismatch: {path.name}")
    return digest


def load_verified(path, *, custody_digest=None):
    """No torch deserialization until both custody and artifact hashes match."""
    path = Path(path)
    custody = path.parent / "CUSTODY.json"
    verify(custody, custody_digest)
    expected = json.loads(custody.read_text())["artifact_sha256"][path.name]
    verify(path, expected)
    return torch.load(path, map_location="cpu", weights_only=True)


def value_positions(row, world):
    text, rendered = row["text"], row["rendered_prompt"]
    offsets = row["offset_mapping"]
    if rendered.count(text) != 1 or len(offsets) != len(row["input_ids"]):
        raise ValueError("Literal text and offset geometry differ")
    shift, masks = rendered.index(text), {}
    for record in re.finditer(r"(?m)^Object ([A-D])(?: \| | has )[^\n]*", text):
        obj = record[1]
        if obj in masks:
            raise ValueError("Duplicate object record")
        positions = [
            i
            for i, (left, right) in enumerate(offsets)
            if left < shift + record.end() and right > shift + record.start()
        ]
        fields = {}
        for field in FIELDS:
            value = world["objects"][obj][field]
            matches = list(
                re.finditer(rf"\b{field}\s+({re.escape(value)})(?!\w)", record[0])
            )
            if len(matches) != 1:
                raise ValueError(
                    "Printed field value does not match frozen world facts"
                )
            left, right = [
                shift + record.start() + index for index in matches[0].span(1)
            ]
            tokens = [
                i
                for i, (start, end) in enumerate(offsets)
                if start < right and end > left
            ]
            if not tokens or not set(tokens) <= set(positions):
                raise ValueError("Value token leaves its declared record")
            fields[field] = tokens
        all_values = [p for values in fields.values() for p in values]
        if len(all_values) != len(set(all_values)):
            raise ValueError("A token overlaps two different field values")
        masks[obj] = {"record": positions, "values": fields}
    all_records = [p for values in masks.values() for p in values["record"]]
    if set(masks) != set("ABCD") or len(all_records) != len(set(all_records)):
        raise ValueError("Four disjoint record lines are required")
    return masks


def describe(row, world, attention, target, wrong):
    if (
        row["field"] not in FIELDS
        or attention.dtype != torch.float32
        or attention.ndim != 1
        or len(attention) != len(row["input_ids"])
        or not bool(torch.isfinite(attention).all())
        or bool((attention < 0).any())
        or abs(float(attention.double().sum()) - 1) > 1e-6
    ):
        raise ValueError("A complete normalized native attention row is required")
    masks, records = value_positions(row, world), {}
    for obj in "ABCD":
        record = masks[obj]
        mass = float(attention[record["record"]].double().sum())
        values = {
            f: float(attention[record["values"][f]].double().sum()) for f in FIELDS
        }
        requested = values[row["field"]]
        records[obj] = {
            "record_mass": mass,
            "field_value_masses": values,
            "requested_value_mass": requested,
            "other_field_value_mass": sum(
                v for f, v in values.items() if f != row["field"]
            ),
            "requested_value_fraction_of_record": requested / mass if mass else None,
            "requested_fraction_of_value_tokens": requested / sum(values.values())
            if sum(values.values())
            else None,
            "field_value_token_positions": record["values"],
        }
    return {
        "correct_object": target,
        "reference_wrong_object": wrong,
        "requested_field": row["field"],
        "record_total": sum(r["record_mass"] for r in records.values()),
        "records": records,
    }


def _stats(values):
    return (
        {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
        }
        if values
        else None
    )


def build_report(base):
    base, anchors = Path(base), {}
    # Check every binary input before either is deserialized.
    for relative, expected in EXPECTED.items():
        anchors[relative] = verify(base / relative, expected)

    def read(relative):
        anchors[relative] = verify(base / relative)
        return json.loads((base / relative).read_text())

    rows = {
        r["id"]: r
        for r in read("cloud_reference_error_metadata_20261115/TOKENS-qwen.json")[
            "rows"
        ]
        if r["split"] == "development"
    }
    worlds = {
        w["world_id"]: w
        for w in read("cloud_reference_error_metadata_20261115/WORLDS.json")
        if w["split"] == "development"
    }
    routing = "cloud_reference_routing_metadata_20261115"
    validation = "cloud_reference_validation_metadata_20261115"
    discovery_result = read(routing + "/routing/RESULT.json")
    discovery_measurements = {
        m["endpoint_id"]: m for m in discovery_result["measurements"]
    }
    generated = {g["id"]: g for g in read(validation + "/validation/GENERATIONS.json")}
    table = []
    for phase, root, sub, job_key in (
        ("discovery", routing, "routing", "capture_jobs"),
        ("validation", validation, "validation", "natural_jobs"),
    ):
        plan = read(root + "/PLAN.json")
        cases = {c["case_id"]: c for c in plan["cases"]}
        path = base / root / sub / "NATURAL.pt"
        raw = load_verified(
            path, custody_digest=EXPECTED[root + "/" + sub + "/CUSTODY.json"]
        )
        for job in plan[job_key]:
            row, case = rows[job["row_id"]], cases[job["case_id"]]
            world = worlds[case["world_id"]]
            wrong = (
                case.get("reverse_object")
                or case["wrong_object_by_field"][case["screen_field"]]
            )
            attention = raw[job["endpoint_id"]]["layers"][LAYER]["attention_rows"][HEAD]
            values = describe(row, world, attention, case["correct_object"], wrong)
            masks = value_positions(row, world)
            if {o: masks[o]["record"] for o in "ABCD"} != job["record_regions"]:
                raise ValueError(
                    "Diagnostic record masks differ from the registered intervention"
                )
            baseline_wrong, repaired = False, None
            if phase == "discovery" and job["role"] == "Rscreen":
                baseline_wrong = True
                repaired = discovery_measurements[
                    f"route::{case['case_id']}::l28::h00::Dcolor"
                ]["argmax_correct"]
            if phase == "validation" and job["role"] == "recipient":
                baseline_wrong = row["field"] in case["initial_wrong_fields"]
                output = generated[f"{case['case_id']}::primaryDcolor::{row['field']}"]
                repaired = (
                    output["terminated"]
                    and output["generated_text"].strip().casefold()
                    == row["expected_answer"].casefold()
                    if baseline_wrong
                    else None
                )
            table.append(
                {
                    "phase": phase,
                    "case_id": case["case_id"],
                    "world_id": case["world_id"],
                    "row_id": row["id"],
                    "role": job["role"],
                    "format": row["template"],
                    "baseline_wrong": baseline_wrong,
                    "primary_repaired": repaired,
                    **values,
                }
            )
        del raw
    summaries, errors = {}, []
    for phase in ("discovery", "validation"):
        selected = [r for r in table if r["phase"] == phase and r["baseline_wrong"]]
        correct = [
            r["records"][r["correct_object"]]["requested_value_fraction_of_record"]
            for r in selected
        ]
        wrong = [
            r["records"][r["reference_wrong_object"]][
                "requested_value_fraction_of_record"
            ]
            for r in selected
        ]
        summaries[phase] = {
            "initial_error_questions": len(selected),
            "worlds": len({r["world_id"] for r in selected}),
            "primary_repairs": sum(r["primary_repaired"] for r in selected),
            "correct_record_requested_value_fraction": _stats(correct),
            "wrong_record_requested_value_fraction": _stats(wrong),
            "correct_record_other_field_value_fraction": _stats(
                [
                    r["records"][r["correct_object"]]["other_field_value_mass"]
                    / r["records"][r["correct_object"]]["record_mass"]
                    for r in selected
                ]
            ),
            "correct_record_requested_fraction_of_value_tokens": _stats(
                [
                    r["records"][r["correct_object"]][
                        "requested_fraction_of_value_tokens"
                    ]
                    for r in selected
                ]
            ),
        }
        for r in selected:
            e, w = (
                r["records"][r["correct_object"]],
                r["records"][r["reference_wrong_object"]],
            )
            errors.append(
                {
                    "phase": phase,
                    "world_id": r["world_id"],
                    "field": r["requested_field"],
                    "correct_object": r["correct_object"],
                    "wrong_object": r["reference_wrong_object"],
                    "primary_repaired": r["primary_repaired"],
                    "correct_record_mass": e["record_mass"],
                    "correct_requested_value_mass": e["requested_value_mass"],
                    "correct_other_field_value_mass": e["other_field_value_mass"],
                    "correct_requested_fraction": e[
                        "requested_value_fraction_of_record"
                    ],
                    "wrong_requested_fraction": w["requested_value_fraction_of_record"],
                }
            )
    return {
        "diagnosis_id": "fixed-head-literal-value-attention-20261115-v1",
        "layer": LAYER,
        "head": HEAD,
        "input_sha256": anchors,
        "analysis_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "phase_summaries": summaries,
        "initial_error_table": errors,
        "all_natural_contexts": table,
        "model_calls": 0,
        "confirmation_outcomes_read": False,
        "interpretation": "Exploratory attention-location diagnosis only. Values include every BPE token overlapping the literal value word, including boundary-bearing tokens; field-key tokens are excluded. Requested-value concentration cannot prove the represented V content or causal importance.",
        "geometry_limit": "NATURAL.pt stores attention rows and record-V hashes, not numeric V matrices, head outputs, output projections or intervention deltas. Delta amplitude/direction and downstream readout alignment are not identifiable from these inputs alone. Existing saved operator tensors would be a separate hash-checked CPU analysis, not grounds to assume an explanation.",
        "diagnostic_conclusion": "Requested-field value-bearing token concentration does not drop in failed validation. This does not support a simple wrong-field attention-location explanation. Small off-field weights could still carry large or influential vectors, so attention fractions do not prove semantic isolation or causal irrelevance.",
        "falsifiable_next_test": "If separately authorized, keep this exact head and test a preregistered gold-blind decomposition of total record-attention budget and normalized four-record allocation: budget-only, allocation-only, both, and self/control sources. If budget-only performs like the joint change, do not call that an address-specific entity repair. Joint superiority plus predicted record-specific steering would support the interaction hypothesis; failure would reject it. This report neither authorizes that test nor chooses new heads, worlds or thresholds.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    report = build_report(args.base)
    print(
        json.dumps(
            report["phase_summaries"] if args.summary else report,
            sort_keys=True,
            allow_nan=False,
        )
    )
