"""Import checks and regression counts for the separately evaluated 64 worlds."""

import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "package_confirmation_data", ROOT / "tools/package_confirmation_data.py"
)
packaging = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(packaging)


def sealed(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    path.with_name(path.name + ".sha256").write_text(
        hashlib.sha256(content).hexdigest() + "\n"
    )


def test_import_preserves_original_bytes_and_deterministic_compression(tmp_path):
    source = tmp_path / "source"
    sealed(source / "ROWS.json", b"[]\n")
    sealed(source / "TOKENS-qwen.json", b'{"rows": []}\n')
    receipt = packaging.import_metadata(source, tmp_path / "first")
    packaging.import_metadata(source, tmp_path / "second")
    assert (tmp_path / "first/ROWS.json").read_bytes() == b"[]\n"
    for folder in ("first", "second"):
        assert (
            gzip.decompress((tmp_path / folder / "TOKENS-qwen.json.gz").read_bytes())
            == b'{"rows": []}\n'
        )
    assert (tmp_path / "first/TOKENS-qwen.json.gz").read_bytes() == (
        tmp_path / "second/TOKENS-qwen.json.gz"
    ).read_bytes()
    assert (
        receipt["files"]["TOKENS-qwen.json.gz"]["representation"]
        == "deterministic gzip of exact original bytes"
    )
    with pytest.raises(FileExistsError):
        packaging.import_metadata(source, tmp_path / "first")


@pytest.mark.parametrize("problem", ["hash", "personal", "native"])
def test_unsafe_or_unverified_import_is_rejected_before_copy(tmp_path, problem):
    source = tmp_path / "source"
    if problem == "native":
        sealed(source / "raw-0000.safetensors", b"not metadata")
    else:
        sealed(
            source / "ROWS.json",
            b"/Users/private/path" if problem == "personal" else b"[]",
        )
        if problem == "hash":
            (source / "ROWS.json.sha256").write_text("0" * 64)
    target = tmp_path / "target"
    with pytest.raises(ValueError):
        packaging.import_metadata(source, target)
    assert not target.exists()


def test_completed_confirmation_package_keeps_development_separate():
    folder = ROOT / "data/processed/confirmation"
    assert len(json.loads((folder / "ROWS.json").read_text())) == 6144
    worlds = json.loads((folder / "WORLDS.json").read_text())
    assert len(worlds) == 64 and {w["split"] for w in worlds} == {"confirmation"}
    qwen = json.loads((folder / "qwen/RESULT.json").read_text())
    phi = json.loads((folder / "phi/RESULT.json").read_text())
    assert qwen["by_probe_kind"]["fact"]["correct"] == 3656
    assert phi["by_probe_kind"]["fact"]["correct"] == 3782
    assert qwen["coreference"]["discordant_wrong_object"] == 265
    assert phi["coreference"]["discordant_wrong_object"] == 166
    repair = json.loads((folder / "repair/RESULT.json").read_text())
    assert repair["eligible_worlds"] == 30
    assert repair["by_condition"]["primaryDcolor"]["repaired_initial_errors"] == 16
    assert repair["by_condition"]["primaryDcolor"]["initial_errors"] == 40
    assert repair["by_condition"]["primaryDcolor"]["protected_damaged"] == 0
    assert repair["usefulness"]["conclusion"] == "uncertain"
    scope = json.loads((ROOT / "data/processed/package_scope.json").read_text())
    assert scope["worlds"] == 32 and scope["confirmation_worlds"] == 64
    assert scope["reserved_confirmation_outcomes_included"] is True
    assert len(json.loads((ROOT / "data/processed/worlds.json").read_text())) == 32
