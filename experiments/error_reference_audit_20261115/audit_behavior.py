"""Independent literal-prompt grading; imports no experimental scoring code."""

import argparse
import hashlib
import json
import re
import string
from collections import Counter, defaultdict
from pathlib import Path


def _clean(text):
    return (
        re.sub(r"\s+", " ", text.strip().casefold()).rstrip(string.punctuation).rstrip()
    )


def literal_answer(text):
    """Reconstruct labels from printed records and bindings, not saved worlds."""
    expressions = (
        r"Object ([A-D]) \| nameplate (\w+) \| color (\w+) \| shape (\w+) \| home (\w+)\.",
        r"Object ([A-D]) has nameplate (\w+), color (\w+), shape (\w+), and home (\w+)\.",
    )
    matches = [match for pattern in expressions for match in re.findall(pattern, text)]
    if len(matches) != 4 or {record[0] for record in matches} != set("ABCD"):
        raise ValueError("Exactly four distinct printed records are required")
    facts = {
        obj: dict(zip(("nameplate", "color", "shape", "home"), values))
        for obj, *values in matches
    }
    entries = re.findall(r"Alias (\w+) -> Object ([A-D])\.", text)
    if len(entries) != 4 or len(dict(entries)) != 4:
        raise ValueError("Exactly four unambiguous printed bindings are required")
    bindings = dict(entries)
    before, after = text.split("[ALIAS_END]")
    request = re.fullmatch(
        r"\n(?:Requested field: |Return (?:its|the) )(color|shape|home|object ID|status word)\.\nAnswer:",
        after,
    )
    if request is None:
        raise ValueError("Unrecognized literal question")
    field = request[1]
    if field == "status word":
        statuses = re.findall(r"Status word: (\w+)\.", before)
        if len(statuses) != 1:
            raise ValueError("Unambiguous printed status required")
        return statuses[0], None, None, facts
    aliases = re.findall(r"(?:Requested alias: |Read Alias )(\w+)\.\n$", before)
    if len(aliases) != 1 or aliases[0] not in bindings:
        raise ValueError("Question does not resolve to a listed alias")
    obj = bindings[aliases[0]]
    return (
        (obj, obj, None, facts)
        if field == "object ID"
        else (facts[obj][field], obj, field, facts)
    )


def audit(rows, predictions, registered_rows_sha256):
    """Check complete native-run metadata independently of its RESULT file."""
    row_bytes = Path(rows).read_bytes()
    if hashlib.sha256(row_bytes).hexdigest() != registered_rows_sha256:
        raise ValueError("Local literal rows do not match the cloud registration")
    selected = [row for row in json.loads(row_bytes) if row["split"] == "development"]
    lookup = {item["id"]: item for item in predictions}
    if (
        len(selected) != 3072
        or len(lookup) != len(predictions)
        or len(predictions) != 3072
        or set(lookup) != {r["id"] for r in selected}
    ):
        raise ValueError("Complete development-only question coverage required")
    counts, by_kind, wrong_worlds, pairs = (
        Counter(),
        defaultdict(Counter),
        set(),
        defaultdict(dict),
    )
    worlds = Counter(row["world_id"] for row in selected)
    if len(worlds) != 32 or set(worlds.values()) != {96}:
        raise ValueError("Repeated questions must belong to exactly 32 complete worlds")
    for row in selected:
        item = lookup[row["id"]]
        expected, _target, field, facts = literal_answer(row["text"])
        if expected != row["expected_answer"] or item["expected_answer"] != expected:
            raise ValueError("Saved label differs from the literal registry")
        if (
            type(item.get("terminated")) is not bool
            or type(item.get("truncated")) is not bool
            or item["terminated"] == item["truncated"]
        ):
            raise ValueError("Explicit completion metadata required")
        finished = item["terminated"] and not item["truncated"]
        answer = _clean(item["generated_text"])
        correct = finished and answer == _clean(expected)
        counts["correct"] += correct
        counts["count"] += 1
        by_kind[row["probe_kind"]]["correct"] += correct
        by_kind[row["probe_kind"]]["count"] += 1
        selected_object = None
        if field is not None and finished:
            selected_object = next(
                (obj for obj in "ABCD" if _clean(facts[obj][field]) == answer), None
            )
        if field is not None and not correct:
            counts[
                "wrong_object"
                if selected_object is not None
                else "unlisted_or_malformed"
            ] += 1
            if selected_object is not None:
                wrong_worlds.add(row["world_id"])
        if field is not None and row["query_alias_index"] in (0, 1):
            pairs[(row["world_id"], row["name_swap"], row["focus_condition"], field)][
                row["query_alias_index"]
            ] = (correct, selected_object)
    eligible, eligible_worlds = 0, set()
    for key, pair in pairs.items():
        first, second = pair[0], pair[1]
        wrong = second if first[0] else first
        if first[0] != second[0] and wrong[1] is not None:
            eligible += 1
            eligible_worlds.add(key[0])
    return {
        "independent_literal_label_audit": True,
        "worlds": 32,
        "counts": dict(counts),
        "by_probe_kind": {key: dict(value) for key, value in by_kind.items()},
        "wrong_object_worlds": len(wrong_worlds),
        "eligible_wrong_object_coref_pairs": eligible,
        "eligible_wrong_object_coref_worlds": len(eligible_worlds),
        "feasibility_passed": len(wrong_worlds) >= 12
        and eligible >= 24
        and len(eligible_worlds) >= 8,
        "native_logit_audit_scope": "Separate remote rescore is required; this audit independently reconstructs printed labels and behavioral counts only.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", required=True, type=Path)
    parser.add_argument("--generations", required=True, type=Path)
    parser.add_argument("--registration", required=True, type=Path)
    args = parser.parse_args()
    registration = json.loads(args.registration.read_text())
    print(
        json.dumps(
            audit(
                args.rows,
                json.loads(args.generations.read_text()),
                registration["input_sha256"]["ROWS.json"],
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
