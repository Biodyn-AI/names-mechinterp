"""Synthetic outcome-summary tests; no model or real experiment data."""

import math
import unittest
from copy import deepcopy

from e30_14_summary import summarize_scores

PROTOCOL = {
    "effect_floor": 1e-6,
    "shuffle_count": 2,
    "specificity_min_pairs_per_direction": 8,
    "specificity_min_entities_per_direction": 6,
}


def synthetic_cells():
    cells = []
    for site in ("entity_marker", "query"):
        for source, target in (("full", "abbr"), ("abbr", "full")):
            for index in range(8):
                errors = {
                    "base": 1.0,
                    "matched": 0.0,
                    "natural_donor": 0.1,
                    "mapped_ridge": 0.02,
                    "unmapped_delta": 0.4,
                    "delta_same_form": 0.01,
                    "reverse": 0.9,
                    "random": 0.9,
                    "wrong_donor": 0.9,
                    "length_matched_wrong_donor": 0.9,
                    "shuffle_00": 0.8,
                    "shuffle_01": 0.7,
                }
                scores = {
                    name: {
                        "local": {"denominator": 1.0, "error": error},
                        "natural": {
                            "denominator": 1.0,
                            "error": 0.1 if name == "matched" else error,
                        },
                    }
                    for name, error in errors.items()
                }
                cells.append(
                    {
                        "model_id": "model",
                        "site_family": site,
                        "source_form": source,
                        "target_form": target,
                        "base_entity_id": f"e{index}",
                        "donor_entity_id": f"e{(index + 1) % 8}",
                        "primary_question_arm": True,
                        "primary_stratum": True,
                        "first_token_distinct_stratum": True,
                        "length_matched_wrong_entity_id": "other",
                        "source_relation_id": "capital",
                        "target_relation_id": "largest_city",
                        "control_dose_diagnostics": {
                            name: {
                                "original_norm": 1.0,
                                "actual_norm": 1.0,
                                "aligned_norm": 1.0,
                                "degenerate_control": False,
                            }
                            for name in (
                                "random",
                                "wrong_donor",
                                "length_matched_wrong_donor",
                                "shuffle_00",
                                "shuffle_01",
                            )
                        },
                        "scores": scores,
                    }
                )
    return cells


def primary(summary, site="entity_marker", stratum="full_city_distinct"):
    return summary["models"]["model"]["sites"][site]["strata"][stratum]


class SummaryTests(unittest.TestCase):
    def test_summary_does_not_mutate_saved_cell_scores(self):
        cells = synthetic_cells()
        original = deepcopy(cells)
        summarize_scores(cells, PROTOCOL)
        self.assertEqual(cells, original)

    def test_exact_weighting_not_mean_recovery(self):
        cells = synthetic_cells()[:2]
        cells[0]["scores"]["mapped_ridge"]["local"] = {"denominator": 1.0, "error": 0.0}
        cells[1]["scores"]["mapped_ridge"]["local"] = {"denominator": 9.0, "error": 9.0}
        result = primary(summarize_scores(cells, PROTOCOL))
        self.assertAlmostEqual(result["local"]["mapped_ridge"]["recovery"], 0.1)

    def test_gates_directions_nulls_and_leave_one_entity_out(self):
        result = summarize_scores(synthetic_cells(), PROTOCOL)
        group = primary(result)
        self.assertEqual(group["nonshuffle_status"], "passes_conditional_metric_gates")
        self.assertTrue(group["gates"]["length_matched_specificity"])
        self.assertEqual(len(group["shuffle_improvement"]), 2)
        self.assertEqual(len(group["leave_one_entity_out"]), 8)
        self.assertTrue(
            all(row["cell_count"] == 12 for row in group["leave_one_entity_out"])
        )
        self.assertEqual(set(group["directions"]), {"full_to_abbr", "abbr_to_full"})
        self.assertFalse(result["abstract_entity_claim_authorized"])

    def test_missing_or_tiny_specificity_coverage_is_inconclusive(self):
        cells = synthetic_cells()
        for cell in cells:
            cell["length_matched_wrong_entity_id"] = None
            cell["scores"].pop("length_matched_wrong_donor")
        result = primary(summarize_scores(cells, PROTOCOL))
        self.assertEqual(
            result["nonshuffle_status"], "inconclusive_specificity_coverage"
        )
        cells = synthetic_cells()
        for cell in cells:
            for condition in cell["scores"].values():
                condition["local"]["denominator"] = 0.0
                condition["local"]["error"] = 0.0
        result = primary(summarize_scores(cells, PROTOCOL))
        self.assertIsNone(result["local"]["mapped_ridge"]["recovery"])
        self.assertEqual(
            result["nonshuffle_status"], "inconclusive_specificity_coverage"
        )

    def test_local_success_is_not_enough_when_natural_reference_is_weak(self):
        cells = synthetic_cells()
        for cell in cells:
            cell["scores"]["matched"]["natural"]["error"] = 0.8
        result = primary(summarize_scores(cells, PROTOCOL))
        self.assertFalse(result["gates"]["natural_sufficiency"])
        self.assertEqual(
            result["nonshuffle_status"], "does_not_meet_conditional_metric_gates"
        )

    def test_answer_copy_subset_and_null_completeness_are_explicit(self):
        cells = synthetic_cells()
        for cell in cells:
            cell["first_token_distinct_stratum"] = False
            cell["scores"].pop("shuffle_01")
        result = summarize_scores(cells, PROTOCOL)
        self.assertEqual(
            primary(result, stratum="first_token_distinct")["nonshuffle_status"],
            "inconclusive_specificity_coverage",
        )
        self.assertIsNone(primary(result)["shuffle_improvement"][1])

    def test_nonfinite_and_negative_measurements_fail(self):
        for value in (math.nan, math.inf, -1.0, True):
            cells = synthetic_cells()
            cells[0]["scores"]["mapped_ridge"]["local"]["error"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                summarize_scores(cells, PROTOCOL)

    def test_degenerate_or_missing_null_diagnostics_are_not_qualified_benchmarks(self):
        for missing in (False, True):
            cells = synthetic_cells()
            if missing:
                cells[0]["control_dose_diagnostics"].pop("shuffle_00")
            else:
                cells[0]["control_dose_diagnostics"]["shuffle_00"][
                    "degenerate_control"
                ] = True
            group = primary(summarize_scores(cells, PROTOCOL))
            self.assertIsNone(group["shuffle_improvement"][0])
            self.assertFalse(group["shuffle_benchmark_complete"])

    def test_degenerate_specificity_control_prevents_success_but_retains_scores(self):
        cells = synthetic_cells()
        cells[0]["control_dose_diagnostics"]["random"]["degenerate_control"] = True
        group = primary(summarize_scores(cells, PROTOCOL))
        self.assertEqual(
            group["nonshuffle_status"], "inconclusive_specificity_controls"
        )
        self.assertAlmostEqual(group["local"]["mapped_ridge"]["recovery"], 0.98)


if __name__ == "__main__":
    unittest.main()
