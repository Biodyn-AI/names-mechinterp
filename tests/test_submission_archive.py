"""Submission archives must be portable and contain only deliberate inputs."""

import runpy
from pathlib import Path
from zipfile import ZipFile

import pytest

MODULE = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/package_submission.py"))


def test_archive_bytes_are_deterministic_and_paths_are_relative(tmp_path):
    source = tmp_path / "source.tex"
    source.write_text("Editable source\n")
    first, second = tmp_path / "first.zip", tmp_path / "second.zip"
    MODULE["write_archive"](first, [(source, "latex/source.tex")])
    MODULE["write_archive"](second, [(source, "latex/source.tex")])
    assert first.read_bytes() == second.read_bytes()
    with ZipFile(first) as archive:
        assert archive.namelist() == ["latex/source.tex"]
        assert archive.read("latex/source.tex") == b"Editable source\n"


@pytest.mark.parametrize("unsafe", ["/absolute.tex", "../outside.tex", "latex/../../outside.tex"])
def test_archive_rejects_paths_outside_its_root(tmp_path, unsafe):
    source = tmp_path / "source.tex"
    source.write_text("source")
    with pytest.raises(ValueError, match="relative"):
        MODULE["write_archive"](tmp_path / "unsafe.zip", [(source, unsafe)])
