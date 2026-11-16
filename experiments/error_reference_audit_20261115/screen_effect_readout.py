"""CPU readout of saved first-token interventions; no model loading or forwards."""

import argparse
import importlib
import math
import string
import sys
from pathlib import Path

import torch


def _normalize(text):
    return " ".join(text.strip().casefold().split()).rstrip(string.punctuation).rstrip()


def _native(native, vocab_size):
    if (
        native.dtype != torch.float32
        or native.device.type != "cpu"
        or tuple(native.shape) != (vocab_size,)
        or not bool(torch.isfinite(native).all())
    ):
        raise ValueError("Saved native logits must be complete finite CPU float32")
    return native


def token_stats(native, correct_ids, wrong_ids, object_labels, tokenizer, vocab_size):
    native = _native(native, vocab_size).to(torch.float64)
    if (
        not correct_ids
        or not wrong_ids
        or set(correct_ids) & set(wrong_ids)
        or any(
            type(i) is not int or not 0 <= i < vocab_size
            for i in correct_ids + wrong_ids
        )
    ):
        raise ValueError(
            "Accepted whole-label IDs must be nonempty, valid and disjoint"
        )
    probabilities, top = torch.softmax(native, 0), int(native.argmax())
    decoded = tokenizer.decode([top], skip_special_tokens=False)
    matches = [
        obj
        for obj, label in object_labels.items()
        if _normalize(decoded) == _normalize(label)
    ]
    if len(matches) > 1:
        raise ValueError("Listed field values must be unambiguous")
    return {
        "argmax_token_id": top,
        "argmax_decoded_token": decoded,
        "argmax_listed_object": matches[0] if matches else None,
        "correct_first_token": top in correct_ids,
        "correct_label_probability": float(probabilities[correct_ids].sum()),
        "predicted_wrong_label_probability": float(probabilities[wrong_ids].sum()),
        "correct_minus_wrong_margin": float(
            torch.logsumexp(native[correct_ids], 0)
            - torch.logsumexp(native[wrong_ids], 0)
        ),
    }


def screen_readout(reader, planned, inputs, result, tokenizer):
    """Selected effects are descriptive in-sample maxima, never confirmation."""
    if len(planned["cases"]) != 8:
        raise ValueError(
            "The registered eight independent screening cases are required"
        )
    rows = {row["id"]: row for row in inputs["rows"]}
    worlds = {world["world_id"]: world for world in inputs["worlds"]}
    cases, clean, labels = {}, {}, {}
    for case in planned["cases"]:
        identity, field = case["case_id"], case["screen_field"]
        world, row = worlds[case["world_id"]], rows[case["screen_row_id"]]
        object_labels = {obj: world["objects"][obj][field] for obj in "ABCD"}
        expected, wrong = (
            row["expected_answer"],
            object_labels[case["reverse_predicted_object"]],
        )
        if expected != object_labels[case["correct_object"]]:
            raise ValueError("Correct label differs from the registered world")
        correct_ids, wrong_ids = (
            inputs["label_token_ids"][field][value] for value in (expected, wrong)
        )
        for value, ids in ((expected, correct_ids), (wrong, wrong_ids)):
            if any(
                _normalize(tokenizer.decode([token], skip_special_tokens=False))
                != _normalize(value)
                for token in ids
            ):
                raise ValueError(
                    "Accepted IDs do not decode to the frozen complete label"
                )
        labels[identity] = (correct_ids, wrong_ids, object_labels)
        clean[identity] = token_stats(
            reader.read(f"clean::{identity}"),
            *labels[identity],
            tokenizer,
            inputs["model"]["vocab_size"],
        )
        cases[identity] = {
            "case_id": identity,
            "world_id": case["world_id"],
            "field": field,
            "correct_object": case["correct_object"],
            "correct_label": expected,
            "predicted_wrong_object": case["reverse_predicted_object"],
            "predicted_wrong_label": wrong,
            "correct_label_token_ids": correct_ids,
            "predicted_wrong_label_token_ids": wrong_ids,
            "clean": clean[identity],
        }
    targets, all_changes = [], {}
    for kind, jobs, selections in (
        (
            "layer",
            planned["layer_jobs"],
            [(layer, None) for layer in result["selected_layers"]],
        ),
        (
            "head",
            result["head_jobs"],
            [
                (head["layer_index"], head["head_index"])
                for head in result["selected_heads"]
            ],
        ),
    ):
        measurements = {
            item["endpoint_id"]: item["correct_minus_wrong_gain"]
            for item in result[kind + "_measurements"]
        }
        if len(measurements) != len(jobs) or set(measurements) != {
            job["endpoint_id"] for job in jobs
        }:
            raise ValueError(
                "Saved margin measurement coverage differs from registered jobs"
            )
        changes, changed_cases = 0, set()
        for job in jobs:
            top = int(
                _native(
                    reader.read(job["endpoint_id"]), inputs["model"]["vocab_size"]
                ).argmax()
            )
            changed = top != clean[job["case_id"]]["argmax_token_id"]
            changes += changed
            if changed:
                changed_cases.add(job["case_id"])
        all_changes[kind] = {
            "job_count": len(jobs),
            "argmax_changed_count": changes,
            "cases_with_any_argmax_change": sorted(changed_cases),
        }
        for layer, head in selections:
            selected = [
                job
                for job in jobs
                if job["layer_index"] == layer and job["head_index"] == head
            ]
            if len(selected) != 8 or {job["case_id"] for job in selected} != set(cases):
                raise ValueError("Selected target must cover every case exactly once")
            effects = []
            for job in selected:
                identity = job["case_id"]
                patched = token_stats(
                    reader.read(job["endpoint_id"]),
                    *labels[identity],
                    tokenizer,
                    inputs["model"]["vocab_size"],
                )
                margin = (
                    patched["correct_minus_wrong_margin"]
                    - clean[identity]["correct_minus_wrong_margin"]
                )
                if not math.isclose(
                    margin,
                    measurements[job["endpoint_id"]],
                    rel_tol=1e-10,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        "Recomputed native margin differs from saved measurement"
                    )
                effects.append(
                    {
                        **cases[identity],
                        "endpoint_id": job["endpoint_id"],
                        "patched": patched,
                        "correct_label_probability_change": patched[
                            "correct_label_probability"
                        ]
                        - clean[identity]["correct_label_probability"],
                        "correct_minus_wrong_margin_change": margin,
                    }
                )
            summary = {
                "case_count": 8,
                "clean_correct_first_token_count": sum(
                    e["clean"]["correct_first_token"] for e in effects
                ),
                "patched_correct_first_token_count": sum(
                    e["patched"]["correct_first_token"] for e in effects
                ),
                "argmax_token_changed_count": sum(
                    e["clean"]["argmax_token_id"] != e["patched"]["argmax_token_id"]
                    for e in effects
                ),
                "mean_correct_label_probability_change": sum(
                    e["correct_label_probability_change"] for e in effects
                )
                / 8,
                "mean_correct_minus_wrong_margin_change": sum(
                    e["correct_minus_wrong_margin_change"] for e in effects
                )
                / 8,
            }
            targets.append(
                {
                    "kind": kind,
                    "layer_index": layer,
                    "head_index": head,
                    "summary": summary,
                    "per_case": effects,
                }
            )
    return {
        "schema": "saved-screen-first-token-effect-readout-20261115-v1",
        "selected_targets": targets,
        "all_screen_argmax_changes": all_changes,
        "model_calls": 0,
        "complete_answer_repair_established": False,
        "caveats": [
            "Layer/head selection and measurement reuse the same eight development worlds; selection bias remains and maxima are not confirmation.",
            "A correct first token or a token flip does not establish a completed-answer repair, preservation, reverse induction or a new mechanism.",
            "Correct-label probability sums every frozen whole-label accepted token ID over the complete native vocabulary using CPU float64 softmax.",
            "An argmax token that is not a complete listed label remains unclassified; it is not silently treated as a completed wrong-object answer.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--causal-source-root", required=True, type=Path)
    parser.add_argument("--causal-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise FileExistsError("Derived output must be new; never overwrite evidence")
    torch.set_num_threads(1)
    sys.path.insert(0, str(args.causal_source_root.resolve()))
    causal = importlib.import_module("causal_run")
    if Path(causal.__file__).resolve().parent != args.causal_source_root.resolve():
        raise ValueError("Causal module escaped the explicitly supplied source root")
    _, _, _, old, tokens, registration, inputs, planned = causal.check(args.causal_root)
    root = args.causal_root / "screen"
    anchors = {
        "registration_sha256": old.verify_file(args.causal_root / "REGISTRATION.json"),
        "registered_inputs": registration["input_sha256"],
        "custody_sha256": old.verify_file(root / "CUSTODY.json"),
    }
    custody = old.read_json(root / "CUSTODY.json")
    if custody["registration_sha256"] != anchors["registration_sha256"]:
        raise ValueError("Screen custody differs from its registration")
    for name, digest in custody["artifact_sha256"].items():
        if Path(name).name != name:
            raise ValueError("Artifact escaped its custody namespace")
        old.verify_file(root / name, digest)
    reader = old.RawReader(root, old.read_json(root / "RAW_INDEX.json"))
    report = screen_readout(
        reader,
        planned,
        inputs,
        old.read_json(root / "RESULT.json"),
        old.load_tokenizer(tokens["snapshot"]),
    )
    report["input_hash_anchors"] = {
        **anchors,
        "screen_artifacts": custody["artifact_sha256"],
        "readout_source_sha256": old.sha256(Path(__file__)),
    }
    old.seal_json(args.output, report)


if __name__ == "__main__":
    main()
