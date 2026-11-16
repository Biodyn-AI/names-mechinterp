"""Model-free specification tests, written before the new generator."""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import error_protocol as protocol


def test_fresh_world_counts_and_reproducibility():
    study = protocol.build_study()
    assert study == protocol.build_study()
    assert Counter(w["split"] for w in study["worlds"]) == {
        "development": 32,
        "confirmation": 64,
    }
    assert Counter(r["split"] for r in study["rows"]) == {
        "development": 3072,
        "confirmation": 6144,
    }
    assert len({r["id"] for r in study["rows"]}) == 9216
    for field in protocol.FIELDS:
        assignments = [
            tuple(w["objects"][obj][field] for obj in "ABCD") for w in study["worlds"]
        ]
        assert len(set(assignments)) == 96


def test_layout_and_name_balance_counts_worlds_not_questions():
    study = protocol.build_study()
    for split, count in protocol.SPLITS.items():
        worlds = [w for w in study["worlds"] if w["split"] == split]
        assert Counter(w["template"] for w in worlds) == {0: count // 2, 1: count // 2}
        assert Counter(w["base_object_id"] for w in worlds) == {
            obj: count // 4 for obj in "ABCD"
        }
        record_alias_positions = Counter(
            (w["object_order"].index(w["base_object_id"]), w["alias_order"].index(0))
            for w in worlds
        )
        assert len(record_alias_positions) == 16
        assert set(record_alias_positions.values()) == {count // 16}
        for alias_index in range(4):
            assert Counter(w["alias_order"].index(alias_index) for w in worlds) == {
                i: count // 4 for i in range(4)
            }
            assert Counter(w["aliases"][alias_index] for w in worlds) == {
                name: count // 8 for name in protocol.ALIASES
            }
        assert (
            sum(w["alias_order"].index(0) < w["alias_order"].index(1) for w in worlds)
            == count // 2
        )


def test_labels_and_field_blind_prefixes():
    study = protocol.build_study()
    worlds = {w["world_id"]: w for w in study["worlds"]}
    prefixes = defaultdict(set)
    for row in study["rows"]:
        world = worlds[row["world_id"]]
        left, right = row["alias_end_span"]
        assert row["text"][left:right] == "[ALIAS_END]"
        assert right <= row["field_start"]
        if row["query_alias_index"] is not None:
            obj = world["alias_targets"][row["query_alias_index"]]
            expected = (
                obj if row["field"] is None else world["objects"][obj][row["field"]]
            )
            assert row["expected_answer"] == expected
            key = (
                row["world_id"],
                row["name_swap"],
                row["focus_condition"],
                row["query_alias_index"],
            )
            prefixes[key].add(row["text"][:right])
        else:
            assert row["expected_answer"] == world["status_word"]
    assert all(len(values) == 1 for values in prefixes.values())


def test_name_swaps_change_labels_not_binding_or_facts():
    study = protocol.build_study()
    grouped = defaultdict(dict)
    for row in study["rows"]:
        key = (
            row["world_id"],
            row["focus_condition"],
            row["probe_kind"],
            row["query_alias_index"],
            row["field"],
        )
        grouped[key][row["name_swap"]] = row
    for pair in grouped.values():
        first, second = pair[0], pair[1]
        assert first["expected_answer"] == second["expected_answer"]
        assert first["text"] != second["text"]
        assert first["aliases"][:2] == list(reversed(second["aliases"][:2]))
        assert first["aliases"][2:] == second["aliases"][2:]


def test_old_gate_and_confirmation_are_not_rescued():
    design = protocol.build_protocol()
    assert design["old_positive_editing_study_reopened"] is False
    assert design["confirmation_model_calls_authorized"] is False
    assert design["causal_model_calls_authorized"] is False
    assert design["feasibility_gate"] == {
        "wrong_object_worlds_at_least": 12,
        "discordant_coref_pairs_at_least": 24,
        "discordant_coref_worlds_at_least": 8,
    }


def test_token_only_selection_has_complete_reproducible_provenance():
    receipt = json.loads((Path(__file__).parent / "ALIAS_SELECTION.json").read_text())

    def digest(value):
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    design = protocol.build_protocol()["token_only_alias_selection"]
    assert receipt["model_outputs_used"] is False
    assert len(receipt["candidates"]) == receipt["candidate_count"] == 96
    assert tuple(item["word"] for item in receipt["selected"]) == protocol.ALIASES
    assert (
        digest(receipt["source_candidate_banks"]["banks"])
        == design["candidate_banks_sha256"]
    )
    assert (
        digest(receipt["selection_digest"]["canonical_payload"])
        == design["selection_receipt_sha256"]
    )
    assert receipt["chosen_group"]["length_signature"] == design["signature"]
