"""Pure prospective design tests: no model, tokenizer, network, or drive access."""

import json
import unittest
from collections import Counter

import e30_14_protocol as protocol


def synthetic_facts():
    return {
        "population_reference": "2020_census",
        "facts": {
            p["entity_id"]: {
                "capital": "Capital " + p["entity_id"],
                "largest_city": "Largest " + p["entity_id"],
            }
            for p in protocol.build_entity_panel()
        },
    }


class ProtocolTests(unittest.TestCase):
    def test_all_states_have_one_disjoint_role_and_exposed_states_are_excluded(self):
        panel = protocol.build_entity_panel()
        splits = protocol.build_splits()
        self.assertEqual(len(panel), 50)
        self.assertEqual(
            {key: len(value) for key, value in splits.items()},
            {"fit": 20, "validation": 6, "test": 20, "legacy_excluded": 4},
        )
        flattened = [entity for entities in splits.values() for entity in entities]
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(set(flattened), {row["entity_id"] for row in panel})
        self.assertEqual(
            set(splits["legacy_excluded"]), {"california", "florida", "oregon", "texas"}
        )
        self.assertEqual(splits, protocol.build_splits())

    def test_training_never_contains_test_states_question_or_wording(self):
        splits = protocol.build_splits()
        prompts = protocol.build_prompts()
        self.assertEqual(len(prompts), len({p["prompt_id"] for p in prompts}))
        self.assertEqual(
            Counter(p["split"] for p in prompts),
            {"fit": 40, "validation": 24, "test": 160},
        )
        for prompt in prompts:
            self.assertIn(prompt["entity_id"], splits[prompt["split"]])
            if prompt["split"] != "test":
                self.assertEqual(prompt["relation_id"], "capital")
                self.assertNotEqual(prompt["frame"], "C")
            if prompt["split"] == "fit":
                self.assertEqual(prompt["frame"], "A")
            self.assertEqual(prompt["entity_marker_text"], "?")
            self.assertEqual(prompt["prompt"].count("?"), 1)
            self.assertIn(prompt["entity_text"] + "?", prompt["prompt"])

    def test_test_pairs_cover_every_unseen_state_without_self_pairs(self):
        pairs = protocol.build_test_pairs()
        self.assertEqual(len(pairs), len(set(pairs)))
        self.assertEqual(len(pairs), 40)
        self.assertTrue(all(base != donor for base, donor in pairs))
        for side in (0, 1):
            self.assertEqual(
                Counter(pair[side] for pair in pairs),
                {entity: 2 for entity in protocol.build_splits()["test"]},
            )

    def test_semantic_qualification_uses_only_legacy_identities(self):
        prompts = protocol.build_qualification_prompts(synthetic_facts())
        self.assertEqual(len(prompts), 32)
        self.assertEqual(
            {p["entity_id"] for p in prompts},
            set(protocol.build_splits()["legacy_excluded"]),
        )
        self.assertTrue(
            all(
                p["relation_id"] in ("capital", "largest_city")
                and p["frame"] in ("A", "B")
                and p["answer"]
                for p in prompts
            )
        )
        self.assertEqual(
            Counter((p["relation_id"], p["form"]) for p in prompts),
            {
                (relation, form): 8
                for relation in ("capital", "largest_city")
                for form in ("full", "abbr")
            },
        )
        self.assertFalse(
            {p["prompt_id"] for p in prompts}
            & {p["prompt_id"] for p in protocol.build_prompts()}
        )

    def test_primary_grid_and_role_references_are_fixed(self):
        cells = protocol.build_comparison_cells()
        prompts = {p["prompt_id"]: p for p in protocol.build_prompts()}
        self.assertEqual(len(cells), 640)
        self.assertEqual(len({cell["cell_id"] for cell in cells}), 640)
        self.assertEqual(sum(cell["primary_question_arm"] for cell in cells), 160)
        self.assertTrue(all(cell["primary_stratum"] is None for cell in cells))
        for cell in cells:
            source_base = prompts[cell["source_base_prompt_id"]]
            target_base = prompts[cell["base_prompt_id"]]
            self.assertEqual(source_base["entity_id"], target_base["entity_id"])
            self.assertEqual(source_base["frame"], "A")
            self.assertEqual(target_base["frame"], "C")
            self.assertNotEqual(source_base["form"], target_base["form"])
            if cell["primary_question_arm"]:
                self.assertEqual(source_base["relation_id"], "capital")
                self.assertEqual(target_base["relation_id"], "largest_city")
            self.assertIsNone(cell["length_matched_wrong_entity_id"])

    def test_registration_is_json_safe_and_not_execution_authority(self):
        record = protocol.build_protocol()
        for value in (
            record,
            protocol.build_prompts(),
            protocol.build_comparison_cells(),
        ):
            self.assertEqual(json.loads(json.dumps(value, allow_nan=False)), value)
        self.assertFalse(record["execution_authorized"])
        self.assertFalse(record["tokenization_qualification_authorized"])
        self.assertEqual(record["primary_site_families"], ["entity_marker", "query"])
        self.assertEqual(record["primary_dose"], 1.0)
        self.assertEqual(record["secondary_doses"], [0.5, 2.0])
        self.assertEqual(record["shuffle_count"], 32)
        self.assertEqual(record["logical_endpoint_count_per_model"], 14400)
        self.assertIn("identity_plus_correction", record["conditions"])
        self.assertIn("diagnostic", record["identity_correction_rule"])
        self.assertFalse(record["abstract_entity_proven_by_local_success"])
        self.assertEqual(
            record["prospective_models"],
            ["openai-community/gpt2-medium", "Qwen/Qwen3-4B-Instruct-2507"],
        )
        self.assertEqual(record["specificity_min_pairs_per_direction"], 8)
        self.assertEqual(record["specificity_min_entities_per_direction"], 6)

    def test_primary_stratum_requires_distinct_answers_for_both_entities(self):
        facts = synthetic_facts()
        cells = protocol.annotate_comparison_cells(facts)
        self.assertEqual(sum(c["primary_stratum"] for c in cells), 160)
        self.assertTrue(all(c["first_token_distinct_stratum"] is None for c in cells))
        entity = protocol.build_splits()["test"][0]
        facts["facts"][entity]["largest_city"] = facts["facts"][entity]["capital"]
        cells = protocol.annotate_comparison_cells(facts)
        self.assertEqual(sum(c["primary_stratum"] for c in cells), 144)
        self.assertTrue(
            all(
                not c["primary_stratum"]
                for c in cells
                if entity in (c["base_entity_id"], c["donor_entity_id"])
            )
        )
        tokens = {
            p["prompt_id"]: 1 if p["relation_id"] == "capital" else 2
            for p in protocol.build_prompts()
        }
        cells = protocol.annotate_comparison_cells(facts, answer_first_token_ids=tokens)
        self.assertEqual(sum(c["first_token_distinct_stratum"] for c in cells), 144)

    def test_fact_manifest_missing_or_noncanonical_inputs_are_rejected(self):
        for invalid in (
            {},
            {"population_reference": "latest", "facts": {}},
            {"population_reference": "2020_census", "facts": {}},
        ):
            with self.assertRaises(ValueError):
                protocol.annotate_comparison_cells(invalid)
        facts = synthetic_facts()
        facts["facts"][protocol.build_splits()["test"][0]]["capital"] = ""
        with self.assertRaises(ValueError):
            protocol.annotate_comparison_cells(facts)

    def test_specificity_coverage_is_inconclusive_below_registered_floor(self):
        cells = protocol.annotate_comparison_cells(synthetic_facts())
        for cell in cells:
            cell["effect_informative"] = True
        report = protocol.specificity_coverage(
            cells, eligibility_field="primary_stratum"
        )
        self.assertEqual(report["status"], "sufficient")
        self.assertEqual(len(report["groups"]), 4)
        for cell in cells:
            cell["effect_informative"] = False
        report = protocol.specificity_coverage(
            cells, eligibility_field="primary_stratum"
        )
        self.assertEqual(report["status"], "inconclusive")
        self.assertTrue(all(g["informative_pair_count"] == 0 for g in report["groups"]))

    def test_mutating_a_returned_record_does_not_change_design(self):
        first = protocol.build_protocol()
        first["rank_grid"].append(19)
        self.assertEqual(protocol.build_protocol()["rank_grid"], [2, 4, 8])

    def test_layer_eligibility_excludes_only_structurally_inert_final_marker(self):
        self.assertEqual(protocol.eligible_layer_indices("entity_marker", 4), [0, 1, 2])
        self.assertEqual(protocol.eligible_layer_indices("query", 4), [0, 1, 2, 3])
        self.assertEqual(protocol.eligible_layer_indices("entity_marker", 1), [])
        for family, count in (("marker", 4), ("query", 0), ("query", True)):
            with self.assertRaises(ValueError):
                protocol.eligible_layer_indices(family, count)


if __name__ == "__main__":
    unittest.main()
