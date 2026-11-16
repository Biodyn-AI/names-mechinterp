"""Pure prospective held-out-state alignment design, not execution authority.

This file builds text and comparison records only. Model selection, verified
answer facts, token offsets, numerical qualification, and the final registration
remain separate prerequisites. No old experiment or external drive is accessed.
"""

import hashlib
import json
from pathlib import Path

_STATES = (
    ("Alabama", "AL"),
    ("Alaska", "AK"),
    ("Arizona", "AZ"),
    ("Arkansas", "AR"),
    ("California", "CA"),
    ("Colorado", "CO"),
    ("Connecticut", "CT"),
    ("Delaware", "DE"),
    ("Florida", "FL"),
    ("Georgia", "GA"),
    ("Hawaii", "HI"),
    ("Idaho", "ID"),
    ("Illinois", "IL"),
    ("Indiana", "IN"),
    ("Iowa", "IA"),
    ("Kansas", "KS"),
    ("Kentucky", "KY"),
    ("Louisiana", "LA"),
    ("Maine", "ME"),
    ("Maryland", "MD"),
    ("Massachusetts", "MA"),
    ("Michigan", "MI"),
    ("Minnesota", "MN"),
    ("Mississippi", "MS"),
    ("Missouri", "MO"),
    ("Montana", "MT"),
    ("Nebraska", "NE"),
    ("Nevada", "NV"),
    ("New Hampshire", "NH"),
    ("New Jersey", "NJ"),
    ("New Mexico", "NM"),
    ("New York", "NY"),
    ("North Carolina", "NC"),
    ("North Dakota", "ND"),
    ("Ohio", "OH"),
    ("Oklahoma", "OK"),
    ("Oregon", "OR"),
    ("Pennsylvania", "PA"),
    ("Rhode Island", "RI"),
    ("South Carolina", "SC"),
    ("South Dakota", "SD"),
    ("Tennessee", "TN"),
    ("Texas", "TX"),
    ("Utah", "UT"),
    ("Vermont", "VT"),
    ("Virginia", "VA"),
    ("Washington", "WA"),
    ("West Virginia", "WV"),
    ("Wisconsin", "WI"),
    ("Wyoming", "WY"),
)
_LEGACY = {"california", "florida", "oregon", "texas"}
CONDITIONS = (
    "base",
    "natural_donor",
    "matched",
    "self",
    "unmapped_delta",
    "mapped_ridge",
    "mapped_orthogonal",
    "identity_plus_correction",
    "reverse",
    "random",
    "wrong_donor",
    "length_matched_wrong_donor",
    "delta_same_form",
    "replacement",
)


def _hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_entity_panel():
    return [
        {"entity_id": name.lower().replace(" ", "_"), "name": name, "abbr": abbr}
        for name, abbr in _STATES
    ]


def build_splits():
    eligible = [
        p["entity_id"] for p in build_entity_panel() if p["entity_id"] not in _LEGACY
    ]
    ordered = sorted(eligible, key=lambda entity: _hash("e30_14_split_v1::" + entity))
    return {
        "fit": ordered[:20],
        "validation": ordered[20:26],
        "test": ordered[26:],
        "legacy_excluded": sorted(_LEGACY),
    }


def eligible_layer_indices(site_family, layer_count):
    """Exclude a nonquery patch after the final cross-token computation."""
    if (
        site_family not in ("entity_marker", "query")
        or type(layer_count) is not int
        or layer_count < 1
    ):
        raise ValueError("valid site family and positive integer layer count required")
    return list(
        range(layer_count - 1 if site_family == "entity_marker" else layer_count)
    )


def _prompt_id(split, frame, form, relation, entity):
    return f"e30_14::{split}::{frame}::{form}::{relation}::{entity}"


def build_prompts():
    panel = {p["entity_id"]: p for p in build_entity_panel()}
    prompts = []
    for split, entities in build_splits().items():
        if split == "legacy_excluded":
            continue
        frames = (
            ("A",)
            if split == "fit"
            else ("A", "B")
            if split == "validation"
            else ("A", "C")
        )
        relations = ("capital", "largest_city") if split == "test" else ("capital",)
        for entity in entities:
            for form in ("full", "abbr"):
                name = panel[entity]["name" if form == "full" else "abbr"]
                for relation in relations:
                    for frame in frames:
                        noun = (
                            "the capital city of"
                            if relation == "capital"
                            else "the largest city by population in"
                        )
                        question = (
                            f"What is {noun} {name}?"
                            if frame == "A"
                            else f"Name {noun} {name}?"
                            if frame == "B"
                            else f"Identify {noun} {name}?"
                        )
                        prompts.append(
                            {
                                "prompt_id": _prompt_id(
                                    split, frame, form, relation, entity
                                ),
                                "split": split,
                                "entity_id": entity,
                                "entity_text": name,
                                "form": form,
                                "frame": frame,
                                "relation_id": relation,
                                "entity_marker_text": "?",
                                "prompt": f"Context: a U.S. state. Answer with only the city name.\nQuestion: {question}\nAnswer:",
                            }
                        )
    return prompts


def build_qualification_prompts(facts_record=None):
    """Keep semantic screening on old identities, outside all fresh splits."""
    facts = _facts_data(facts_record) if facts_record is not None else None
    panel = {p["entity_id"]: p for p in build_entity_panel()}
    prompts = []
    for entity in sorted(_LEGACY):
        for form in ("full", "abbr"):
            name = panel[entity]["name" if form == "full" else "abbr"]
            for relation in ("capital", "largest_city"):
                for frame in ("A", "B"):
                    noun = (
                        "the capital city of"
                        if relation == "capital"
                        else "the largest city by population in"
                    )
                    question = (
                        f"What is {noun} {name}?"
                        if frame == "A"
                        else f"Name {noun} {name}?"
                    )
                    prompts.append(
                        {
                            "prompt_id": _prompt_id(
                                "qualification", frame, form, relation, entity
                            ),
                            "split": "qualification",
                            "entity_id": entity,
                            "entity_text": name,
                            "form": form,
                            "frame": frame,
                            "relation_id": relation,
                            "entity_marker_text": "?",
                            "answer": None
                            if facts is None
                            else facts[entity][relation],
                            "prompt": f"Context: a U.S. state. Answer with only the city name.\nQuestion: {question}\nAnswer:",
                        }
                    )
    return prompts


def build_test_pairs():
    entities = build_splits()["test"]
    return [
        (base, entities[(index + offset) % len(entities)])
        for index, base in enumerate(entities)
        for offset in (1, 7)
    ]


def build_shuffles():
    training = build_splits()["fit"]
    return [
        {
            "shuffle_index": index,
            "target_entity_ids": sorted(
                training,
                key=lambda entity: _hash(f"e30_14_shuffle_v1::{index}::{entity}"),
            ),
        }
        for index in range(32)
    ]


def build_comparison_cells():
    """Build logical cells; token qualification later fixes both selected layers."""
    cells = []
    entities = build_splits()["test"]
    for site in ("entity_marker", "query"):
        for source_form, target_form in (("full", "abbr"), ("abbr", "full")):
            for source_relation in ("capital", "largest_city"):
                for target_relation in ("capital", "largest_city"):
                    for base, donor in build_test_pairs():
                        wrong = next(
                            entity
                            for entity in sorted(entities)
                            if entity not in (base, donor)
                        )
                        cells.append(
                            {
                                "cell_id": f"{site}::{source_form}_to_{target_form}::{source_relation}_to_{target_relation}::{base}_to_{donor}",
                                "site_family": site,
                                "primary_question_arm": source_relation == "capital"
                                and target_relation == "largest_city",
                                "primary_stratum": None,
                                "first_token_distinct_stratum": None,
                                "base_entity_id": base,
                                "donor_entity_id": donor,
                                "source_form": source_form,
                                "target_form": target_form,
                                "source_relation_id": source_relation,
                                "target_relation_id": target_relation,
                                "base_prompt_id": _prompt_id(
                                    "test", "C", target_form, target_relation, base
                                ),
                                "donor_prompt_id": _prompt_id(
                                    "test", "C", target_form, target_relation, donor
                                ),
                                "source_base_prompt_id": _prompt_id(
                                    "test", "A", source_form, source_relation, base
                                ),
                                "source_donor_prompt_id": _prompt_id(
                                    "test", "A", source_form, source_relation, donor
                                ),
                                "same_form_source_base_prompt_id": _prompt_id(
                                    "test", "A", target_form, source_relation, base
                                ),
                                "same_form_source_donor_prompt_id": _prompt_id(
                                    "test", "A", target_form, source_relation, donor
                                ),
                                "wrong_source_donor_prompt_id": _prompt_id(
                                    "test", "A", source_form, source_relation, wrong
                                ),
                                "length_matched_wrong_entity_id": None,
                            }
                        )
    return cells


def _facts_data(record):
    """Validate supplied fixed facts; never invent missing facts or aliases."""
    expected = {p["entity_id"] for p in build_entity_panel()}
    if (
        not isinstance(record, dict)
        or record.get("population_reference") != "2020_census"
        or not isinstance(record.get("facts"), dict)
        or set(record["facts"]) != expected
    ):
        raise ValueError("fixed facts require 2020_census and exactly fifty state IDs")
    for row in record["facts"].values():
        if not isinstance(row, dict) or any(
            not isinstance(row.get(key), str) or not row[key].strip()
            for key in ("capital", "largest_city")
        ):
            raise ValueError(
                "every state requires nonempty canonical capital/largest_city strings"
            )
        if (
            "capital_largest_city_distinct" in row
            and type(row["capital_largest_city_distinct"]) is not bool
        ):
            raise ValueError("canonical city-distinct flags must be booleans")
    json.dumps(record, allow_nan=False)
    return record["facts"]


def load_fixed_facts(path):
    """Return supplied JSON and its byte hash; caller freezes the provenance."""
    data = Path(path).read_bytes()
    record = json.loads(data)
    _facts_data(record)
    return record, hashlib.sha256(data).hexdigest()


def annotate_comparison_cells(
    facts_record, *, answer_first_token_ids=None, length_matched_wrong_entities=None
):
    """Resolve factual strata before outcomes; token IDs come from qualification."""
    facts = _facts_data(facts_record)
    fact_digest = _hash(
        json.dumps(facts_record, sort_keys=True, separators=(",", ":"), allow_nan=False)
    )
    matched = length_matched_wrong_entities or {}
    cells = build_comparison_cells()
    for cell in cells:
        base, donor = cell["base_entity_id"], cell["donor_entity_id"]
        full_distinct = []
        exact_distinct = []
        for entity in (base, donor):
            row = facts[entity]
            exact_distinct.append(row["capital"] != row["largest_city"])
            full_distinct.append(
                row.get(
                    "capital_largest_city_distinct",
                    " ".join(row["capital"].casefold().split())
                    != " ".join(row["largest_city"].casefold().split()),
                )
            )
        cell["facts_canonical_sha256"] = fact_digest
        cell["full_city_exact_distinct"] = all(exact_distinct)
        cell["full_city_distinct"] = all(full_distinct)
        cell["primary_stratum"] = (
            cell["primary_question_arm"] and cell["full_city_distinct"]
        )
        cell["first_token_distinct"] = None
        if answer_first_token_ids is not None:
            keys = (
                "source_base_prompt_id",
                "base_prompt_id",
                "source_donor_prompt_id",
                "donor_prompt_id",
            )
            try:
                tokens = [answer_first_token_ids[cell[key]] for key in keys]
            except (KeyError, TypeError) as error:
                raise ValueError(
                    "answer first-token qualification is incomplete"
                ) from error
            if any(type(token) is not int or token < 0 for token in tokens):
                raise ValueError("answer first-token IDs must be nonnegative integers")
            cell["first_token_distinct"] = (
                tokens[0] != tokens[1] and tokens[2] != tokens[3]
            )
            cell["first_token_distinct_stratum"] = (
                cell["primary_stratum"] and cell["first_token_distinct"]
            )
        wrong = matched.get(cell["cell_id"])
        if wrong is not None and (
            wrong not in build_splits()["test"] or wrong in (base, donor)
        ):
            raise ValueError(
                "length-matched wrong donor must be a distinct test entity"
            )
        cell["length_matched_wrong_entity_id"] = wrong
        cell["length_matched_specificity_eligible"] = (
            cell["primary_stratum"] and wrong is not None
        )
    return cells


def specificity_coverage(cells, *, eligibility_field):
    """Apply fixed coverage floors after scoring, without deleting any cells."""
    if eligibility_field not in (
        "primary_stratum",
        "first_token_distinct_stratum",
        "length_matched_specificity_eligible",
    ):
        raise ValueError("unknown specificity eligibility field")
    groups = []
    for site in ("entity_marker", "query"):
        for source_form in ("full", "abbr"):
            selected = [
                c
                for c in cells
                if c["site_family"] == site
                and c["source_form"] == source_form
                and c.get(eligibility_field) is True
                and c.get("effect_informative") is True
            ]
            pairs = {(c["base_entity_id"], c["donor_entity_id"]) for c in selected}
            entities = {entity for pair in pairs for entity in pair}
            groups.append(
                {
                    "site_family": site,
                    "source_form": source_form,
                    "informative_pair_count": len(pairs),
                    "unique_entity_count": len(entities),
                    "sufficient": len(pairs) >= 8 and len(entities) >= 6,
                }
            )
    return {
        "eligibility_field": eligibility_field,
        "groups": groups,
        "status": "sufficient"
        if all(g["sufficient"] for g in groups)
        else "inconclusive",
    }


def build_protocol():
    return {
        "schema_version": "e30_14_heldout_alignment_design_v1",
        "status": "draft_requires_model_fact_token_and_runtime_registration",
        "execution_authorized": False,
        "tokenization_qualification_authorized": False,
        "splits": build_splits(),
        "prompt_count": 224,
        "qualification_prompt_count": 32,
        "prospective_models": [
            "openai-community/gpt2-medium",
            "Qwen/Qwen3-4B-Instruct-2507",
        ],
        "primary_site_families": ["entity_marker", "query"],
        "layer_selection": "One layer per site family, shared across both form directions; select using validation geometry only, never held-out questions, states, or causal outputs.",
        "causally_eligible_layers": "Entity-marker candidates exclude the final transformer block, because no later cross-token operation can carry that marker patch to the query. Query candidates include all blocks. Keep all-layer captures; this exclusion is architectural, not outcome-selected.",
        "fit_rule": "Exactly 20 fit-state vectors per form, capital question wording A only. Fit means, singular vectors, regularization scale, and mappings from these vectors only.",
        "validation_rule": "Six separate states; compare A-source-form donor-minus-base differences against B-target-form differences, capital only. Minimize summed squared prediction error divided by summed target-difference energy, combined across both form directions.",
        "selection_ties": "Smaller rank, larger ridge fraction, then earlier layer; no test-data tie breaking.",
        "rank_grid": [2, 4, 8],
        "ridge_fraction_grid": [0.01, 0.1, 1.0],
        "ridge_formula": "Center X/Y on fit means. X=U diag(s) V.T; alpha=ridge_fraction*mean(s[:min(19,width)]**2). W=V_r diag(s_r/(s_r**2+alpha)) U_r.T Y; rank at most r.",
        "orthogonal_secondary": "Train-identified low-rank partial Procrustes only, with chosen rank; no arbitrary completion of unobserved ambient directions and no separate test tuning.",
        "identity_correction_rule": "One diagnostic identity-plus-low-rank-correction map: fit centered X -> Y-X using only twenty fit states, then apply source_delta + predicted_correction. At the already-fixed primary physical layer/site, independently select rank/ridge from the same grid using validation geometry jointly across both form directions, comparing the corrected delta to the full target delta with full target-energy denominator. Preserve source components outside the learned correction subspace. Exactly one added condition per cell; no extra doses/nulls, no confirmatory success rescue or abstract-entity claim. A strong diagnostic needs separate future confirmation.",
        "no_pair_inflation": "Twenty fit states provide at most nineteen independent centered entity directions, even if many pair differences are reported.",
        "primary_question": "Source wording A capital -> target wording C largest city; largest-city relation and wording C never enter fitting or mapping selection.",
        "primary_answer_stratum": "Within the full capital-to-city question arm, BOTH base and donor must have different canonical full-city capital versus largest-city answers in the fixed 2020 Census facts. Report literal-string distinctness and canonical full-city distinctness separately.",
        "first_token_distinct_stratum": "Also report the nested subset where BOTH base and donor source-capital versus target-city first-answer-token IDs differ under the model-specific frozen token qualification. Missing token IDs remain unresolved, never guessed.",
        "facts_interface": "JSON population_reference='2020_census', facts={all fifty snake_case state IDs:{capital:canonical city string,largest_city:canonical city string,optional capital_largest_city_distinct:boolean}}, with source/provenance metadata. load_fixed_facts returns record plus byte SHA256; annotate_comparison_cells consumes it and optional prompt_id->answer-first-token-ID mapping.",
        "secondary_questions": [
            "capital_to_capital",
            "largest_city_to_largest_city",
            "largest_city_to_capital",
        ],
        "test_pairs": [list(pair) for pair in build_test_pairs()],
        "primary_question_arm_cell_count": 160,
        "primary_stratum_cell_count": None,
        "secondary_question_cell_count": 480,
        "conditions": list(CONDITIONS),
        "shuffle_count": 32,
        "shuffles": build_shuffles(),
        "primary_dose": 1.0,
        "secondary_doses": [0.5, 2.0],
        "dose_rule": "Apply selected map at unit gain; half/double doses are robustness diagnostics, not a test-selected rescue. Never match a mapped delta to the held-out true target-delta norm.",
        "causal_patch": "Target-base state + dose*W(source-donor minus source-base).",
        "local_reference": "Whole target-donor state replacement at the same selected site.",
        "natural_reference": "Unmodified target-donor prompt, reported independently from local reference.",
        "control_rule": "Unmapped difference, same-form difference, whole source-donor replacement, self, reverse mapped difference, norm-matched isotropic noise, mapped wrong donor, and length-matched mapped wrong donor.",
        "shuffle_rule": "Shuffle only the twenty fit-state target labels. Keep the real map's selected physical site/layer, but EACH shuffle independently selects from the SAME rank/ridge grid using validation geometry jointly across both form directions, with the same tie rule. Never use test states/outcomes. Norm-match each null mapped difference to the real mapped difference on the same source input, never to the true target difference. Record zero/null norms.",
        "control_dose_diagnostics_required": "Every shuffled/random/wrong/length-matched perturbation records original_norm,actual_norm,aligned_norm,degenerate_control. A missing/degenerate shuffled-dose record makes that benchmark inconclusive, never a qualified null advantage. Missing/degenerate specificity controls on informative affected cases prevent specificity success, without dropping scores.",
        "control_norm_match_tolerance": "Certify only positive finite original/actual/aligned norms, degenerate_control=false, and actual_norm matching aligned_norm within relative 1e-6 plus absolute 1e-8 tolerance. Zero requested doses are explicit degeneracies, not matched nonzero null controls.",
        "length_position_control": "At model-free token qualification, choose the lexicographically first wrong test entity matching intended donor source AND target token-span lengths, marker positions, query positions, and full prompt lengths. Record unavailable cases explicitly; never choose using predictions.",
        "entity_marker_rule": "A common verified token immediately after the complete name, not an arbitrary final name subword; record all token spans/offsets/position IDs. Names may have multiple tokens.",
        "position_reporting": "Report equal-length/position and unequal-length/position strata without selecting the primary sample; keep all forty pairs and both form directions.",
        "specificity_min_pairs_per_direction": 8,
        "specificity_min_entities_per_direction": 6,
        "informative_pair_rule": "A pair counts toward specificity coverage only when its own local base-to-matched divergence is at least the registered 1e-6 effect floor; retain all pairs in the declared aggregate scores.",
        "specificity_coverage_rule": "For EACH site and EACH form direction, require at least eight informative ordered pairs spanning at least six unique entities, separately in the full-city-distinct, first-token-distinct, and available length-matched-wrong-donor strata. Insufficient coverage makes that success/specificity claim inconclusive; never discard cells or relax floors.",
        "length_specificity_success_rule": "Where coverage is sufficient, mapped local recovery must exceed the length-matched wrong-donor control by at least .10 in EACH form direction on exactly those matched cases. Otherwise report position specificity inconclusive, not success.",
        "metric": "Full-vocabulary first-prediction-token Jensen-Shannon divergence, scored in float64 from native float32 logits.",
        "recovery": "1-sum(JS(intervention,reference))/sum(JS(base,reference)); references never mixed.",
        "effect_floor": 1e-6,
        "primary_success_rule": "On the prequalified full-city-distinct primary stratum at a selected site, combined mapped local recovery >=.90, each form direction >=.80, and >=.10 above unmapped/reverse/random/wrong-donor controls; beat all 32 maximum-null improvement benchmarks. Apply the same transfer and coverage gates to the nested distinct-first-token stratum before excluding answer-copying. Length-matched specificity and numerical gates must also pass.",
        "meaningful_natural_scope_gate": "Also require combined same-form local recovery >=.90 and matched-to-natural recovery >=.80. Otherwise describe any result as local-coordinate alignment only; never discard failed cells.",
        "multiplicity": "Two confirmatory site families per inferential model, with predeclared full-city/first-token-distinct strata. Each shared shuffle produces the maximum mapped-minus-unmapped recovery improvement across all primary sites/models/strata; compare each real improvement to these maxima. This is a finite shuffled benchmark, NOT an exact permutation p-value. Orthogonal maps, doses, directions, secondary questions, and other layers remain diagnostic, not extra uncorrected discoveries.",
        "uncertainty_scope": "Report each held-out entity and leave-one-entity-out sensitivity, removing every pair touching that entity. Pair counts are not independent entity sample sizes; shuffled-label rankings are finite null benchmarks, not unrestricted population inference.",
        "geometric_diagnostics": "Held-out delta error/cosine, mapped/source/target norms, train-subspace energy coverage, and shuffled-map norms. Never use held-out target geometry to tune or rescale the map.",
        "semantic_qualification": "Legacy GPT-2 is a computational bridge, not a behavior-qualified semantic model. The preselected Qwen3-4B-Instruct-2507 is screened only on the four previously exposed states, BOTH relations A/B and both name forms: thirty-two fixed prompts. No fresh state or wording C outcomes select models or revise prompts. Largest-city questions appear in legacy behavior screening, but never in map fitting or geometry selection.",
        "qualification_success_rule": "Greedy complete city-name accuracy >=28/32 overall and >=7/8 within each relation-by-form combination, aggregated across wordings; all eight CAPITAL-only abbreviation/full-name nearest-output comparisons must match the same state. Report the eight largest-city nearest comparisons as diagnostics. Generate at most twelve new tokens, stop at EOS/newline, normalize only case, surrounding whitespace and terminal punctuation. A failure is reported, not repaired by trying prompts/models on test data.",
        "qualification_fact_rule": "All capital/largest-city answers and canonical aliases are supplied by the independently verified fixed 2020 Census fact JSON, never filled by this module. Verify/hash copied provenance before final registration. Qualification is separate from the six fresh geometry-validation states.",
        "facts_model_and_runtime_required": True,
        "raw_storage_limit_bytes": 15000000000,
        "logical_endpoint_count_per_model": 14400,
        "storage_rule": "32 shuffled controls only on primary-question-arm cells, fourteen conditions on all cells including diagnostic identity_plus_correction, half/double mapped-ridge dose only on primary-question-arm cells; deduplicate exact repeated operands. Compute native endpoint/state storage before registration; exceeding 15GB total is a planning failure, not permission to prune after outcomes.",
        "abstract_entity_proven_by_local_success": False,
        "interpretation": "Positive held-out causal transfer supports a low-capacity, question-portable coordinate translation, not a pure entity proof. Failure cannot exclude nonlinear, higher-rank, distributed, or context-specific entity representations.",
        "future_path_study": "Not authorized here. Direct-versus-composed maps require matched reconstruction/noise controls; cycle discrepancies alone cannot violate the reconstruction-error triangle bound or prove intrinsic context dependence.",
        "required_before_abstract_entity_claim": "Separate selective-preservation tests on unrelated entities and dynamic same-spelling rebinding tests; neither is implemented or authorized as part of this core alignment study.",
        "resource_gate": "No model/tokenizer qualification, paid pod, or credential use until specific user approval and a separate immutable registration.",
    }
