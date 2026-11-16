"""The temporary forced SSH command must expose only approved archive bytes."""

import hashlib
import importlib.util
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "native_archive_reader", Path(__file__).resolve().parents[1] / "tools/native_archive_reader.py"
)
reader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reader)
NOW = datetime(2026, 11, 16, 19, tzinfo=timezone.utc).timestamp()


def fixture(tmp_path, monkeypatch):
    root = tmp_path / "publication"
    root.mkdir()
    monkeypatch.setattr(reader, "ALLOWED_ROOTS", frozenset((str(root),)))
    archive = root / "names-development-part001.tar.gz"
    content = b"checked binary archive\x00\xff"
    archive.write_bytes(content)
    value = {"expires_utc": "2026-11-16T21:45:00Z", "allowed_roots": [str(root)],
             "archives": [{"name": archive.name, "path": str(archive),
                           "size_bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}]}
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps(value))
    return root, archive, content, value, catalog


def test_probe_and_exact_binary_stream(tmp_path, monkeypatch):
    _, archive, content, _, catalog = fixture(tmp_path, monkeypatch)
    output = io.BytesIO()
    reader.serve(catalog, "probe", output, clock=lambda: NOW)
    assert output.getvalue() == b"names-native-reader-v1\n"
    output = io.BytesIO()
    reader.serve(catalog, "get " + archive.name, output, clock=lambda: NOW)
    assert output.getvalue() == content


@pytest.mark.parametrize("command", ["", "ls", "get ../secret", "get /etc/passwd", "get names-development-part002.tar.gz", "get names-development-part001.tar.gz; id", "get names-development-part001.tar.gz\nprobe"])
def test_no_shell_listing_or_unapproved_files(tmp_path, monkeypatch, command):
    _, _, _, _, catalog = fixture(tmp_path, monkeypatch)
    output = io.BytesIO()
    with pytest.raises(ValueError):
        reader.serve(catalog, command, output, clock=lambda: NOW)
    assert not output.getvalue()


@pytest.mark.parametrize("problem", ["expired", "late_expiry", "unknown_root", "escape", "wrong_size", "partial", "bad_sha", "duplicate", "symlink", "ancestor_symlink"])
def test_catalog_and_file_boundaries_fail_closed(tmp_path, monkeypatch, problem):
    root, archive, _, value, catalog = fixture(tmp_path, monkeypatch)
    if problem == "expired":
        value["expires_utc"] = "2026-11-16T18:00:00Z"
    elif problem == "late_expiry":
        value["expires_utc"] = "2026-11-16T23:00:00Z"
    elif problem == "unknown_root":
        value["allowed_roots"] = [str(tmp_path)]
    elif problem == "escape":
        value["archives"][0]["path"] = "/etc/passwd"
    elif problem == "wrong_size":
        value["archives"][0]["size_bytes"] += 1
    elif problem == "partial":
        value["archives"][0]["name"] += ".partial"
    elif problem == "bad_sha":
        value["archives"][0]["sha256"] = "not a checksum"
    elif problem == "duplicate":
        value["archives"].append(value["archives"][0])
    elif problem == "symlink":
        real = root / "real"
        archive.rename(real)
        archive.symlink_to(real)
    else:
        moved = tmp_path / "moved"
        root.rename(moved)
        root.symlink_to(moved, target_is_directory=True)
    catalog.write_text(json.dumps(value))
    with pytest.raises((ValueError, FileNotFoundError)):
        reader.serve(catalog, "probe", io.BytesIO(), clock=lambda: NOW)


def test_expiry_is_checked_during_stream(tmp_path, monkeypatch):
    _, archive, _, _, catalog = fixture(tmp_path, monkeypatch)
    calls = iter((NOW, NOW + 4 * 3600))
    with pytest.raises(ValueError):
        reader.serve(catalog, "get " + archive.name, io.BytesIO(), clock=lambda: next(calls))


def test_absolute_expiry_alarm_interrupts_even_a_stalled_write():
    with pytest.raises(ValueError, match="expired"):
        reader.expired_alarm(None, None)
