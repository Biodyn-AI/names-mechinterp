"""Fresh reference-error study. No inference and no reopening of older gates."""

import hashlib
import itertools
import random
from collections import defaultdict

STUDY_ID = "reference-errors-20261114-v1"
SEED = "reference-errors-fresh-worlds-20261114-v1"
SPLITS = {"development": 32, "confirmation": 64}
FIELDS = ("color", "shape", "home")
FOCUS = ("none", "coref_0", "coref_1", "distractor")
# Selected using only both pinned tokenizers, not model answers. Categories are
# mixed within every split rather than being confounded with development/test.
ALIASES = ("apple", "bat", "anchor", "date", "cat", "basket", "fig", "dog")
BANKS = {
    "color": ("red", "blue", "green", "yellow", "black", "white", "gray", "brown"),
    "shape": (
        "circle",
        "square",
        "triangle",
        "rectangle",
        "diamond",
        "oval",
        "star",
        "heart",
    ),
    "home": (
        "forest",
        "castle",
        "river",
        "station",
        "tower",
        "farm",
        "bridge",
        "market",
    ),
    "nameplate": ("Aster", "Cedar", "Elm", "Fir", "Grove", "Hazel", "Iris", "Maple"),
}


def _rng(label):
    return random.Random(int.from_bytes(hashlib.sha256(label.encode()).digest(), "big"))


def build_protocol():
    return {
        "study_id": STUDY_ID,
        "seed": SEED,
        "splits": dict(SPLITS),
        "independent_unit": "A newly sampled world; name swaps, Focus conditions and questions are repeated measures, not new worlds.",
        "prompts_per_world": 96,
        "development_prompt_count": 3072,
        "confirmation_prompt_count": 6144,
        "old_positive_editing_study_reopened": False,
        "confirmation_model_calls_authorized": False,
        "causal_model_calls_authorized": False,
        "purpose": "Replicate and explain wrong-object selection, not demonstrate a successful general entity edit.",
        "names": list(ALIASES),
        "fields": list(FIELDS),
        "banks": {key: list(value) for key, value in BANKS.items()},
        "token_only_alias_selection": {
            "models_order": [
                "Qwen/Qwen3-4B-Instruct-2507",
                "microsoft/Phi-3.5-mini-instruct",
            ],
            "candidate_banks_sha256": "0c28bbe5419abea857590f12a887c43246624b6978888686fe446be74fafcd11",
            "selection_receipt_sha256": "9e07ad87fbe84155350e12f9786d4f4f01b27fc9ee26f074a92bdef86126422d",
            "signature": [6, 5, 10, 9, 8, 7, 15, 14],
            "rule": "Lexicographically smallest common input-frame signature with at least eight candidates and two categories; round-robin fruit/animal/object preserving declared candidate order. Full rendered geometry still must pass before inference.",
            "outputs_used": False,
        },
        "counterbalance": {
            "between_worlds": "Within each split: all 16 target-record/first-alias-line position combinations equally often; each object ID and each alias-line position equally often; both co-reference line orders equally often; each of eight lexical names equally often in every semantic alias role; two wording formats equally often. Other pairwise combinations are randomized, not promised fully factorial.",
            "within_world": "Two consistent swaps of the co-referring names crossed with no Focus, first co-referring name, second co-referring name, and unrelated name. Facts and bindings do not change.",
            "facts": "Four distinct values per field, with each complete ordered object-to-field assignment unique across the 96 new worlds. Value lexemes recur; worlds are not lexically disjoint.",
        },
        "feasibility_gate": {
            "wrong_object_worlds_at_least": 12,
            "discordant_coref_pairs_at_least": 24,
            "discordant_coref_worlds_at_least": 8,
        },
        "gate_interpretation": "Discordant pairs must have one correct answer and one completed answer selecting another listed object in the requested field; malformed/truncated discordance does not qualify. Per-model feasibility only, not significance, novelty or publication readiness. Failure stops mechanism search for that model without prompt rescue. One passing model permits only one-model exploratory work.",
        "answer_rule": "Normalize case, whitespace and final punctuation of the entire decoded answer. Explanations or extra non-whitespace lines are not silently truncated. Keep malformed answers as errors.",
        "statistics": "Report all cases, world-level rates and deterministic 2,000-resample percentile intervals resampling whole worlds. Paired comparisons retain every repeat of the sampled world. Development intervals are descriptive after earlier exploratory results.",
        "causal_next_stage_required_before_confirmation": [
            "Freeze a narrowly localized component, strength, eligible-case rule and particular predicted wrong object using development only.",
            "A single donor computation captured at ALIAS_END before seeing any requested field must be reused unchanged for color, shape and home.",
            "Test matched restoration and reverse induction of the particular wrong object's corresponding values; mere accuracy loss is insufficient.",
            "Report unconditional performance and protected unrelated facts/identity/status alongside selected-error repair.",
            "Compare self, norm-matched random, input-name swap, other-object donor and broad-state replacement controls.",
            "Freeze quantitative lexical/record-position/Focus retrieval-mixture baselines before opening all 64 confirmation worlds.",
            "Separate second-model replication and wording-specific failures; no architectural-generalization claim from one passing model.",
        ],
        "stop_rules": "Stop a model's mechanism search if the feasibility gate fails. Stop a broad novelty claim if counterbalancing removes the effect, known retrieval mixtures suffice, or only answer copying/broad disruption repairs it. Retain failed and negative runs.",
    }


def _layouts(count, rng):
    """Balanced permutations, not rotated copies counted as independent worlds."""
    extras = []
    for order in ((0, 1, 2, 3), (0, 3, 2, 1)):
        extras.extend(order[i:] + order[:i] for i in range(4))
    orders = list(itertools.permutations(range(4))) * (count // 24)
    orders += extras * ((count % 24) // 8)
    by_position = defaultdict(list)
    for order in orders:
        by_position[order.index(0)].append(list(order))
    for group in by_position.values():
        rng.shuffle(group)
    layouts = []
    for alias_position in range(4):
        for record_position in range(4):
            for block in range(count // 16):
                layouts.append(
                    {
                        "alias_order": by_position[alias_position].pop(),
                        "base_record_position": record_position,
                        "base_object_id": "ABCD"[
                            (alias_position + record_position + block) % 4
                        ],
                    }
                )
    rng.shuffle(layouts)
    # Randomize wording separately: a Latin formula would create a deterministic
    # higher-order wording/ID/position coupling despite balanced marginal counts.
    templates = [0, 1] * (count // 2)
    rng.shuffle(templates)
    for layout, template in zip(layouts, templates):
        layout["template"] = template
    return layouts


def _prefix(world, names, focus_condition):
    template = world["template"]
    lines = [
        "Use only this fictional registry. Object IDs stay fixed; nameplates are display labels. Each alias points to its listed object ID. Focus is a highlighted alias. Answer the explicitly requested alias and field with only the value, without explanation."
        if template == 0
        else "This registry defines imaginary objects. An ID is permanent and a nameplate is just a label. Resolve a requested alias through its binding, then report the requested field only. Focus marks an alias for attention. Do not add an explanation."
    ]
    for obj in world["object_order"]:
        facts = world["objects"][obj]
        if template == 0:
            lines.append(
                f"Object {obj} | nameplate {facts['nameplate']} | color {facts['color']} | shape {facts['shape']} | home {facts['home']}."
            )
        else:
            lines.append(
                f"Object {obj} has nameplate {facts['nameplate']}, color {facts['color']}, shape {facts['shape']}, and home {facts['home']}."
            )
    for index in world["alias_order"]:
        lines.append(f"Alias {names[index]} -> Object {world['alias_targets'][index]}.")
    focus_index = {"none": None, "coref_0": 0, "coref_1": 1, "distractor": 2}[
        focus_condition
    ]
    focus_name = "none" if focus_index is None else names[focus_index]
    lines += [
        f"Status word: {world['status_word']}.",
        "[WORLD_END]",
        f"Focus alias: {focus_name}.",
        "[FOCUS_END]",
    ]
    return "\n".join(lines) + "\n"


def _rows(world):
    probes = [("fact", alias, field) for alias in (0, 1, 2) for field in FIELDS]
    probes += [("identity", 0, None), ("identity", 1, None), ("status", None, None)]
    for swap in (0, 1):
        names = list(world["aliases"])
        if swap:
            names[:2] = reversed(names[:2])
        for focus in FOCUS:
            prefix = _prefix(world, names, focus)
            for kind, index, field in probes:
                label = (
                    field
                    if kind == "fact"
                    else "object ID"
                    if kind == "identity"
                    else "status word"
                )
                if kind == "status":
                    expected = world["status_word"]
                else:
                    obj = world["alias_targets"][index]
                    expected = obj if field is None else world["objects"][obj][field]
                if world["template"] == 0:
                    subject = (
                        "source: the registry"
                        if index is None
                        else f"alias: {names[index]}"
                    )
                    question_prefix = f"Requested {subject}.\n[ALIAS_END]\n"
                    query = f"Requested field: {label}.\nAnswer:"
                else:
                    subject = (
                        "the registry status"
                        if index is None
                        else f"Alias {names[index]}"
                    )
                    question_prefix = f"Read {subject}.\n[ALIAS_END]\n"
                    query = (
                        f"Return {'the' if index is None else 'its'} {label}.\nAnswer:"
                    )
                text = prefix + question_prefix + query
                marker_start = text.index("[ALIAS_END]")
                yield {
                    "id": f"{world['world_id']}:swap{swap}:{focus}:{kind}:{index}:{field}",
                    "world_id": world["world_id"],
                    "split": world["split"],
                    "template": world["template"],
                    "name_swap": swap,
                    "focus_condition": focus,
                    "probe_kind": kind,
                    "field": field,
                    "query_alias_index": index,
                    "aliases": list(names),
                    "expected_answer": expected,
                    "text": text,
                    "alias_end_span": [marker_start, marker_start + len("[ALIAS_END]")],
                    "field_start": len(prefix + question_prefix),
                }


def build_study():
    """Construct frozen development/held-out cases without reading any answers."""
    worlds, rows = [], []
    seen = {field: set() for field in FIELDS}
    for split, count in SPLITS.items():
        rng = _rng(f"{SEED}:{split}")
        layouts = _layouts(count, rng)
        shifts = list(range(8)) * (count // 8)
        rng.shuffle(shifts)
        for number, (layout, shift) in enumerate(zip(layouts, shifts)):
            base = layout["base_object_id"]
            other = rng.sample([obj for obj in "ABCD" if obj != base], 3)
            order = rng.sample(other, 3)
            order.insert(layout["base_record_position"], base)
            objects = {obj: {} for obj in "ABCD"}
            for field in FIELDS:
                while True:
                    values = tuple(rng.sample(BANKS[field], 4))
                    if values not in seen[field]:
                        break
                seen[field].add(values)
                for obj, value in zip("ABCD", values):
                    objects[obj][field] = value
            for obj, nameplate in zip("ABCD", rng.sample(BANKS["nameplate"], 4)):
                objects[obj]["nameplate"] = nameplate
            world = {
                "world_id": f"{split}:{number:03d}",
                "split": split,
                "template": layout["template"],
                "base_object_id": base,
                "objects": objects,
                "object_order": order,
                "alias_order": layout["alias_order"],
                "aliases": [ALIASES[(shift + j) % 8] for j in range(4)],
                "alias_targets": [base, base, other[0], other[1]],
                "unaliased_object_id": other[2],
                "status_word": "ready" if number % 2 == 0 else "clear",
            }
            worlds.append(world)
            rows.extend(_rows(world))
    return {"protocol": build_protocol(), "worlds": worlds, "rows": rows}
