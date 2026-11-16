"""Synthetic model-free contracts for the finite answer-time routing test."""

import copy
import hashlib

import pytest
import routing_plan


def _fixture():
    fields = ("color", "shape", "home")
    banks = {
        "color": ["red", "blue", "green", "brown"],
        "shape": ["circle", "square", "triangle", "oval"],
        "home": ["forest", "castle", "farm", "bridge"],
    }
    worlds, rows, cases = [], [], []
    for number, target in enumerate(["B"] * 6 + ["A", "D"]):
        world_id, case_id = f"development:{number:03d}", f"case:{number}"
        objects = {
            obj: {f: banks[f][i] for f in fields} for i, obj in enumerate("ABCD")
        }
        worlds.append(
            {"world_id": world_id, "split": "development", "objects": objects}
        )
        order = list("ABCD")[number % 4 :] + list("ABCD")[: number % 4]
        registry = "\n".join(
            f"Object {o} | color {objects[o]['color']} | shape {objects[o]['shape']} | home {objects[o]['home']}."
            for o in order
        )
        case = {
            "case_id": case_id,
            "world_id": world_id,
            "correct_object": target,
            "screen_field": "shape" if number < 5 else "home",
            "donor_row_ids": {},
            "recipient_row_ids": {},
        }
        for role, alias in (("donor", "apple"), ("recipient", "bat")):
            for field in fields:
                text = (
                    registry
                    + f"\nRequested alias: {alias}.\n[ALIAS_END]\nRequested field: {field}.\nAnswer:"
                )
                offsets = []
                start = 0
                for line in registry.splitlines():
                    offsets.append([start, start + len(line)])
                    start += len(line) + 1
                offsets += [
                    [start, text.index("[ALIAS_END]")],
                    [text.index("[ALIAS_END]"), len(text)],
                ]
                identity = f"{case_id}:{role}:{field}"
                case[f"{role}_row_ids"][field] = identity
                rows.append(
                    {
                        "id": identity,
                        "world_id": world_id,
                        "split": "development",
                        "field": field,
                        "probe_kind": "fact",
                        "text": text,
                        "rendered_prompt": text,
                        "offset_mapping": offsets,
                        "input_ids": [10 + "ABCD".index(o) for o in order]
                        + [20 if role == "donor" else 21, 30 + fields.index(field)],
                        "alias_end_span": [
                            text.index("[ALIAS_END]"),
                            text.index("[ALIAS_END]") + 11,
                        ],
                        "expected_answer": objects[target][field],
                    }
                )
        case["screen_row_id"] = case["recipient_row_ids"][case["screen_field"]]
        case["wrong_object_by_field"] = {case["screen_field"]: "C"}
        cases.append(case)
    return {
        "base_plan": {
            "cases": cases,
            "model": {
                "model_id": "Qwen/Qwen3-4B-Instruct-2507",
                "revision": "pinned",
                "layers": 3,
                "heads": 2,
                "head_dim": 4,
                "vocab_size": 1000,
                "o_proj_in_features": 8,
            },
            "label_token_ids": {
                f: {value: [100 + i] for i, value in enumerate(banks[f])}
                for f in fields
            },
        },
        "rows": rows,
        "worlds": worlds,
        "prior_audit": {
            "exact_match": True,
            "confirmation_calls": 0,
            "physical_counts": {"native": 1144, "backbone_prefix": 16},
        },
        "provenance": {
            "marker_plan_sha256": "a" * 64,
            "marker_rescore_sha256": "b" * 64,
        },
    }


def _captures(planned):
    cases = {c["case_id"]: c for c in planned["cases"]}
    result = []
    for job in planned["capture_jobs"]:
        case = cases[job["case_id"]]
        raw = [0.05] * 4
        e, w = (
            "ABCD".index(case["correct_object"]),
            "ABCD".index(case["wrong_object_by_field"][case["screen_field"]]),
        )
        if job["role"] in ("Dcolor", "Dscreen"):
            raw[e], raw[w] = 0.35, 0.05
        else:
            raw[e], raw[w] = 0.05, 0.35
        masses = [
            [[0.1] * 4 for _ in range(planned["model"]["heads"])]
            for _ in range(planned["model"]["layers"])
        ]
        masses[0][0] = raw
        result.append(
            {
                "endpoint_id": job["endpoint_id"],
                "record_masses": masses,
                "argmax_correct": job["role"] in ("Dcolor", "Dscreen"),
                "record_value_hashes": {
                    str(layer): hashlib.sha256(
                        f"{case['case_id']}:{layer}".encode()
                    ).hexdigest()
                    for layer in range(planned["model"]["layers"])
                },
            }
        )
    return result


def test_plan_has_exact_frozen_cases_full_factorial_and_finite_budget():
    inputs = _fixture()
    planned = routing_plan.plan(inputs)
    assert planned == routing_plan.plan(inputs)
    assert planned["cases"] == inputs["base_plan"]["cases"]
    assert len(planned["capture_jobs"]) == 32
    assert len({j["endpoint_id"] for j in planned["capture_jobs"]}) == 32
    assert planned["storage"]["max_native_endpoints"] == 112
    assert planned["storage"]["estimated_bytes"] < 500_000_000
    assert planned["target_object_counts"] == {"B": 6, "A": 1, "D": 1}
    assert planned["execution_authorized"] is False
    assert planned["further_validation_authorized"] is False
    for job in planned["capture_jobs"]:
        assert sorted(
            p for positions in job["record_regions"].values() for p in positions
        ) == [0, 1, 2, 3]


def test_failed_audit_duplicate_case_and_changed_record_tokens_fail_closed():
    for mutation in ("audit", "case", "tokens", "geometry"):
        inputs = _fixture()
        if mutation == "audit":
            inputs["prior_audit"]["exact_match"] = False
        elif mutation == "case":
            inputs["base_plan"]["cases"][1] = inputs["base_plan"]["cases"][0]
        elif mutation == "tokens":
            inputs["rows"][0]["input_ids"][0] = 999
        else:
            inputs["rows"][0]["input_ids"].append(999)
            inputs["rows"][0]["offset_mapping"].append([0, 0])
        with pytest.raises(ValueError):
            routing_plan.plan(inputs)


def test_same_field_selection_and_five_gold_blind_conditions():
    planned = routing_plan.plan(_fixture())
    captures = _captures(planned)
    selected = routing_plan.select_heads(planned, captures)
    assert selected["selected_heads"][0]["layer_index"] == 0
    assert selected["selected_heads"][0]["head_index"] == 0
    assert len(selected["selected_heads"]) == 1
    assert len(selected["intervention_jobs"]) == 8 * 5
    for job in selected["intervention_jobs"]:
        assert sum(job["desired_profile"]) == pytest.approx(1)
        assert len(job["desired_profile"]) == 4
    profiles = routing_plan.desired_profiles(planned, captures, 0, 0)
    relabeled = copy.deepcopy(planned)
    for case in relabeled["cases"]:
        case["correct_object"] = "D"
        case["wrong_object_by_field"] = {case["screen_field"]: "A"}
    assert profiles == routing_plan.desired_profiles(relabeled, captures, 0, 0)


def test_color_difference_alone_cannot_select_a_head():
    planned = routing_plan.plan(_fixture())
    captures = _captures(planned)
    for job, captured in zip(planned["capture_jobs"], captures):
        if job["role"] in ("Dscreen", "Rscreen"):
            captured["record_masses"][0][0] = [0.1] * 4
    selected = routing_plan.select_heads(planned, captures)
    assert selected["selected_heads"] == []
    assert selected["intervention_jobs"] == []


def test_unsupported_heads_stop_without_changing_support_thresholds():
    planned = routing_plan.plan(_fixture())
    captures = _captures(planned)
    recipient = next(
        c for j, c in zip(planned["capture_jobs"], captures) if j["role"] == "Rscreen"
    )
    recipient["record_masses"][0][0][0] = 0.0
    selected = routing_plan.select_heads(planned, captures)
    assert selected["selected_heads"] == []
    assert selected["unsupported_heads"] == 1


def test_value_invariance_and_complete_capture_grid_are_required():
    planned = routing_plan.plan(_fixture())
    captures = _captures(planned)
    with pytest.raises(ValueError):
        routing_plan.select_heads(planned, captures[:-1])
    captures[1]["record_value_hashes"]["0"] = "d" * 64
    with pytest.raises(ValueError):
        routing_plan.select_heads(planned, captures)


def test_top_two_bound_and_numeric_capture_failures():
    planned = routing_plan.plan(_fixture())
    captures = _captures(planned)
    for capture in captures:
        capture["record_masses"][0][1] = capture["record_masses"][0][0][:]
        capture["record_masses"][1][0] = capture["record_masses"][0][0][:]
    selected = routing_plan.select_heads(planned, captures)
    assert [
        (h["layer_index"], h["head_index"]) for h in selected["selected_heads"]
    ] == [(0, 0), (0, 1)]
    assert len(selected["intervention_jobs"]) == 80
    captures[0]["record_masses"][0][0][0] = float("nan")
    with pytest.raises(ValueError):
        routing_plan.select_heads(planned, captures)
    inputs = _fixture()
    inputs["base_plan"]["model"]["vocab_size"] = 10_000_000
    with pytest.raises(ValueError):
        routing_plan.plan(inputs)


def test_unreproduced_natural_first_token_is_qualification_failure():
    planned = routing_plan.plan(_fixture())
    captures = _captures(planned)
    captures[0]["argmax_correct"] = False
    with pytest.raises(routing_plan.QualificationError):
        routing_plan.select_heads(planned, captures)


def test_practical_gate_compares_only_default_id_control_not_Rcolor():
    selected = routing_plan.select_heads(
        routing_plan.plan(_fixture()), _captures(routing_plan.plan(_fixture()))
    )
    measured = []
    for job in selected["intervention_jobs"]:
        gain = {
            "Dcolor": 0.8,
            "Dscreen": 0.9,
            "Rcolor": 0.8,
            "selfRscreen": 0.0,
            "LOOWmeanDcolor": 0.2,
        }[job["condition"]]
        measured.append(
            {
                "endpoint_id": job["endpoint_id"],
                "correct_minus_wrong_gain": gain,
                "argmax_correct": job["condition"] != "selfRscreen"
                and job["case_id"] in ("case:0", "case:1"),
                "clean_argmax_correct": False,
            }
        )
    result = routing_plan.assess(selected, measured)
    assert result["passed"] is True
    assert result["head_results"][0]["newly_correct"] == 2
    assert result["further_validation_authorized"] is False
    broken = copy.deepcopy(measured)
    for job, value in zip(selected["intervention_jobs"], broken):
        if job["condition"] == "LOOWmeanDcolor":
            value["correct_minus_wrong_gain"] = 0.6
    assert routing_plan.assess(selected, broken)["passed"] is False
    broken[0]["clean_argmax_correct"] = True
    with pytest.raises(ValueError):
        routing_plan.assess(selected, broken)
