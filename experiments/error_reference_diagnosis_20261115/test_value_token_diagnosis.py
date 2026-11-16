"""Literal offset masks, not semantic interpretation of value vectors."""

import hashlib
import json

import pytest
import torch
import value_token_diagnosis as diagnosis


def fixture(format_number=0):
    objects = {
        "A": {"color": "red", "shape": "circle", "home": "forest"},
        "B": {"color": "blue", "shape": "square", "home": "castle"},
        "C": {"color": "green", "shape": "oval", "home": "farm"},
        "D": {"color": "brown", "shape": "star", "home": "bridge"},
    }
    lines = []
    for obj, values in objects.items():
        if format_number == 0:
            lines.append(
                f"Object {obj} | color {values['color']} | shape {values['shape']} | home {values['home']}."
            )
        else:
            lines.append(
                f"Object {obj} has color {values['color']}, shape {values['shape']}, and home {values['home']}."
            )
    text = "\n".join(lines) + "\nRequested alias: apple.\nRequested field: shape."
    rendered = "HEADER\n" + text
    row = {
        "id": "fixture",
        "text": text,
        "rendered_prompt": rendered,
        "field": "shape",
        "input_ids": list(range(len(rendered))),
        "offset_mapping": [[i, i + 1] for i in range(len(rendered))],
    }
    return row, {"objects": objects}


@pytest.mark.parametrize("format_number", [0, 1])
def test_value_positions_match_the_actual_printed_values_in_both_formats(format_number):
    row, world = fixture(format_number)
    masks = diagnosis.value_positions(row, world)
    for obj in "ABCD":
        for field in diagnosis.FIELDS:
            positions = masks[obj]["values"][field]
            assert (
                "".join(row["rendered_prompt"][p] for p in positions)
                == world["objects"][obj][field]
            )
            assert set(positions) <= set(masks[obj]["record"])


def test_attention_fractions_keep_requested_and_other_fields_separate():
    row, world = fixture()
    masks = diagnosis.value_positions(row, world)
    attention = torch.zeros(len(row["input_ids"]), dtype=torch.float32)
    requested = masks["A"]["values"]["shape"]
    other = masks["A"]["values"]["color"]
    wrong = masks["B"]["values"]["shape"]
    attention[requested] = 0.60 / len(requested)
    attention[other] = 0.10 / len(other)
    attention[wrong] = 0.30 / len(wrong)
    result = diagnosis.describe(row, world, attention, "A", "B")
    assert result["records"]["A"][
        "requested_value_fraction_of_record"
    ] == pytest.approx(6 / 7)
    assert result["records"]["A"]["other_field_value_mass"] == pytest.approx(0.1)
    assert result["records"]["B"]["requested_value_mass"] == pytest.approx(0.3)
    assert result["records"]["D"]["requested_value_fraction_of_record"] is None


def test_inconsistent_printed_value_or_offsets_fail_closed():
    row, world = fixture()
    world["objects"]["A"]["shape"] = "triangle"
    with pytest.raises(ValueError):
        diagnosis.value_positions(row, world)
    row, world = fixture()
    row["offset_mapping"] = row["offset_mapping"][:-1]
    with pytest.raises(ValueError):
        diagnosis.value_positions(row, world)


def test_custody_and_digest_are_checked_before_tensor_loading(tmp_path, monkeypatch):
    path = tmp_path / "NATURAL.pt"
    path.write_bytes(b"not even a torch file")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "NATURAL.pt.sha256").write_text(digest + "\n")
    custody = tmp_path / "CUSTODY.json"
    custody.write_text(json.dumps({"artifact_sha256": {"NATURAL.pt": "0" * 64}}))
    (tmp_path / "CUSTODY.json.sha256").write_text(
        hashlib.sha256(custody.read_bytes()).hexdigest() + "\n"
    )
    monkeypatch.setattr(
        torch,
        "load",
        lambda *a, **k: pytest.fail("Unsafe load before custody verification"),
    )
    with pytest.raises(ValueError):
        diagnosis.load_verified(path)
