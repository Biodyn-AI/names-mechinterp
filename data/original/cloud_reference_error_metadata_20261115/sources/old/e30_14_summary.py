"""Conservative, stdlib-only summaries of already-scored held-out cells.

This module cannot qualify a model, run a patch, or authorize an abstract entity
claim. Global shuffled benchmarks across models remain the driver's responsibility.
"""

import math
from collections import defaultdict


def _finite_nonnegative(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("score errors/effects must be finite nonnegative numbers")


def _aggregate(rows, condition, scope, floor):
    metrics = [
        row["scores"][condition][scope] for row in rows if condition in row["scores"]
    ]
    denominator = math.fsum(metric["denominator"] for metric in metrics)
    error = math.fsum(metric["error"] for metric in metrics)
    return {
        "cell_count": len(metrics),
        "denominator_sum": denominator,
        "error_sum": error,
        "recovery": None if denominator < floor else 1 - error / denominator,
    }


def _pack(rows, floor):
    conditions = sorted({condition for row in rows for condition in row["scores"]})
    return {
        scope: {
            condition: _aggregate(rows, condition, scope, floor)
            for condition in conditions
        }
        for scope in ("local", "natural")
    }


def _recovery(packed, scope, condition):
    return packed[scope].get(condition, {}).get("recovery")


def _gap(first, second):
    return None if first is None or second is None else first - second


def _coverage(rows, protocol):
    result = {}
    for source, target in (("full", "abbr"), ("abbr", "full")):
        selected = [
            r
            for r in rows
            if r["source_form"] == source
            and r["scores"]["mapped_ridge"]["local"]["denominator"]
            >= protocol["effect_floor"]
        ]
        pairs = {(r["base_entity_id"], r["donor_entity_id"]) for r in selected}
        entities = {entity for pair in pairs for entity in pair}
        result[f"{source}_to_{target}"] = {
            "informative_pair_count": len(pairs),
            "unique_entity_count": len(entities),
            "sufficient": len(pairs) >= protocol["specificity_min_pairs_per_direction"]
            and len(entities) >= protocol["specificity_min_entities_per_direction"],
        }
    return result


def _dose_certification(rows, condition, floor, *, informative_only):
    uncertified = []
    for index, row in enumerate(rows):
        if (
            informative_only
            and row["scores"]["mapped_ridge"]["local"]["denominator"] < floor
        ):
            continue
        diagnostic = row.get("control_dose_diagnostics", {}).get(condition, {})
        try:
            norms = [
                diagnostic[key]
                for key in ("original_norm", "actual_norm", "aligned_norm")
            ]
            valid = all(
                type(value) in (int, float) and math.isfinite(value) and value > 0
                for value in norms
            )
            valid = valid and diagnostic.get("degenerate_control") is False
            valid = valid and math.isclose(
                norms[1], norms[2], rel_tol=1e-6, abs_tol=1e-8
            )
        except (KeyError, TypeError, AttributeError):
            valid = False
        if not valid:
            uncertified.append(row.get("cell_id", f"row_{index}"))
    return {"certified": not uncertified, "uncertified_cell_ids": uncertified}


def _primary_summary(rows, protocol):
    floor = protocol["effect_floor"]
    packed = _pack(rows, floor)
    directions = {}
    for source, target in (("full", "abbr"), ("abbr", "full")):
        directions[f"{source}_to_{target}"] = _pack(
            [r for r in rows if r["source_form"] == source], floor
        )
    coverage = _coverage(rows, protocol)
    length_rows = [
        r
        for r in rows
        if r.get("length_matched_wrong_entity_id") is not None
        and "length_matched_wrong_donor" in r["scores"]
    ]
    length_coverage = _coverage(length_rows, protocol)
    length_gaps = {}
    for source, target in (("full", "abbr"), ("abbr", "full")):
        subset = [r for r in length_rows if r["source_form"] == source]
        pair = _pack(subset, floor)
        length_gaps[f"{source}_to_{target}"] = _gap(
            _recovery(pair, "local", "mapped_ridge"),
            _recovery(pair, "local", "length_matched_wrong_donor"),
        )
    mapped = _recovery(packed, "local", "mapped_ridge")
    unmapped = _recovery(packed, "local", "unmapped_delta")
    control_gaps = {
        name: _gap(mapped, _recovery(packed, "local", name))
        for name in ("unmapped_delta", "reverse", "random", "wrong_donor")
    }
    same_form = _recovery(packed, "local", "delta_same_form")
    natural = _recovery(packed, "natural", "matched")
    dose_diagnostics = {
        name: _dose_certification(rows, name, floor, informative_only=True)
        for name in ("random", "wrong_donor")
    }
    dose_diagnostics["length_matched_wrong_donor"] = _dose_certification(
        length_rows, "length_matched_wrong_donor", floor, informative_only=True
    )
    gates = {
        "coverage": all(v["sufficient"] for v in coverage.values()),
        "length_coverage": all(v["sufficient"] for v in length_coverage.values()),
        "mapped_combined": mapped is not None and mapped >= 0.90,
        "mapped_each_direction": all(
            _recovery(value, "local", "mapped_ridge") is not None
            and _recovery(value, "local", "mapped_ridge") >= 0.80
            for value in directions.values()
        ),
        "control_gaps": all(
            value is not None and value >= 0.10 for value in control_gaps.values()
        ),
        "same_form": same_form is not None and same_form >= 0.90,
        "natural_sufficiency": natural is not None and natural >= 0.80,
        "length_matched_specificity": all(
            value is not None and value >= 0.10 for value in length_gaps.values()
        ),
        "norm_matched_specificity": all(
            diagnostic["certified"] for diagnostic in dose_diagnostics.values()
        ),
    }
    if not gates["coverage"] or not gates["length_coverage"]:
        status = "inconclusive_specificity_coverage"
    elif not gates["norm_matched_specificity"]:
        status = "inconclusive_specificity_controls"
    elif all(gates.values()):
        status = "passes_conditional_metric_gates"
    else:
        status = "does_not_meet_conditional_metric_gates"
    shuffle_improvement = []
    shuffle_dose_diagnostics = []
    for index in range(protocol["shuffle_count"]):
        name = f"shuffle_{index:02d}"
        value = _recovery(packed, "local", name)
        certification = _dose_certification(rows, name, floor, informative_only=False)
        shuffle_dose_diagnostics.append(certification)
        complete = (
            bool(rows)
            and all(name in row["scores"] for row in rows)
            and certification["certified"]
        )
        shuffle_improvement.append(_gap(value, unmapped) if complete else None)
    influence = []
    for entity in sorted(
        {r[key] for r in rows for key in ("base_entity_id", "donor_entity_id")}
    ):
        subset = [
            r for r in rows if entity not in (r["base_entity_id"], r["donor_entity_id"])
        ]
        scores = _pack(subset, floor)
        influence.append(
            {
                "removed_entity_id": entity,
                "cell_count": len(subset),
                "mapped_local": _recovery(scores, "local", "mapped_ridge"),
                "unmapped_local": _recovery(scores, "local", "unmapped_delta"),
                "mapped_natural": _recovery(scores, "natural", "mapped_ridge"),
            }
        )
    return {
        **packed,
        "cell_count": len(rows),
        "directions": directions,
        "coverage": coverage,
        "length_coverage": length_coverage,
        "control_gaps": control_gaps,
        "length_matched_control_gaps": length_gaps,
        "gates": gates,
        "nonshuffle_status": status,
        "control_dose_diagnostics": dose_diagnostics,
        "mapped_minus_unmapped_improvement": _gap(mapped, unmapped),
        "shuffle_improvement": shuffle_improvement,
        "shuffle_dose_diagnostics": shuffle_dose_diagnostics,
        "shuffle_benchmark_complete": all(
            value is not None for value in shuffle_improvement
        ),
        "leave_one_entity_out": influence,
    }


def summarize_scores(cell_scores, protocol):
    """Group fixed strata and preserve inconclusive coverage and reference scope."""
    rows = list(cell_scores)
    if not rows:
        raise ValueError("cell_scores must be nonempty")
    for row in rows:
        if row.get("site_family") not in ("entity_marker", "query") or row.get(
            "source_form"
        ) not in ("full", "abbr"):
            raise ValueError("invalid model site or name-form direction")
        for required in (
            "mapped_ridge",
            "unmapped_delta",
            "delta_same_form",
            "matched",
            "reverse",
            "random",
            "wrong_donor",
        ):
            if required not in row.get("scores", {}):
                raise ValueError("missing core causal condition")
        for scores in row["scores"].values():
            for scope in ("local", "natural"):
                try:
                    _finite_nonnegative(scores[scope]["denominator"])
                    _finite_nonnegative(scores[scope]["error"])
                except (KeyError, TypeError) as error:
                    raise ValueError("invalid scored-cell schema") from error
    models = defaultdict(list)
    for row in rows:
        models[row.get("model_id", "unspecified")].append(row)
    result = {}
    for model, model_rows in sorted(models.items()):
        sites = {}
        for site in ("entity_marker", "query"):
            site_rows = [r for r in model_rows if r["site_family"] == site]
            strata = {
                "full_city_distinct": _primary_summary(
                    [r for r in site_rows if r.get("primary_stratum") is True], protocol
                ),
                "first_token_distinct": _primary_summary(
                    [
                        r
                        for r in site_rows
                        if r.get("first_token_distinct_stratum") is True
                    ],
                    protocol,
                ),
            }
            questions = {}
            for source in ("capital", "largest_city"):
                for target in ("capital", "largest_city"):
                    subset = [
                        r
                        for r in site_rows
                        if r.get("source_relation_id") == source
                        and r.get("target_relation_id") == target
                    ]
                    questions[f"{source}_to_{target}"] = _pack(
                        subset, protocol["effect_floor"]
                    )
            sites[site] = {
                "strata": strata,
                "descriptive_question_groups": questions,
                "all_cells_descriptive": _pack(site_rows, protocol["effect_floor"]),
                "global_shuffled_comparison_pending": True,
            }
        families = [
            stratum for site in sites.values() for stratum in site["strata"].values()
        ]
        maxima = []
        for index in range(protocol["shuffle_count"]):
            values = [family["shuffle_improvement"][index] for family in families]
            maxima.append(
                max(values) if all(value is not None for value in values) else None
            )
        result[model] = {"sites": sites, "shuffle_maximum_improvement_by_index": maxima}
    return {
        "schema_version": "e30_14_conditional_summary_v1",
        "models": result,
        "abstract_entity_claim_authorized": False,
        "global_shuffled_comparison_pending": True,
        "scope": "Conditional metric gates only; driver must also verify model qualification, numerical integrity, and the maximum finite shuffled benchmark across all primary models/sites/strata. No exact permutation p-value.",
        "uncertainty": "Leave-one-entity-out removes every pair touching that entity; pair counts are not independent sample sizes.",
    }
