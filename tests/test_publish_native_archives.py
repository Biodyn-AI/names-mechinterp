"""Uploader safeguards with mocked GitHub/SSH; no remote work is performed."""

import base64
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "publish_native_archives", ROOT / "tools/publish_native_archives.py"
)
publisher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publisher)

KEY = base64.b64encode(b"\0\0\0\x0bssh-ed25519\0\0\0\x20" + bytes(32)).decode()
PAYLOAD = b"synthetic archive bytes"
ITEM = {
    "name": "names-confirmation-qwen-part001.tar.gz",
    "size_bytes": len(PAYLOAD),
    "sha256": hashlib.sha256(PAYLOAD).hexdigest(),
}


def inputs():
    return publisher.validate_inputs(
        "194.68.245.86",
        "22071",
        f"[194.68.245.86]:22071 ssh-ed25519 {KEY}\n",
        json.dumps([ITEM]),
    )


@pytest.mark.parametrize(
    "change",
    [
        "host",
        "port",
        "known_host",
        "multiple_hosts",
        "name",
        "size",
        "sha",
        "duplicate",
    ],
)
def test_untrusted_dispatch_values_are_rejected(change):
    host, port, known, manifest = (
        "194.68.245.86",
        "22071",
        f"[194.68.245.86]:22071 ssh-ed25519 {KEY}\n",
        [dict(ITEM)],
    )
    if change == "host":
        host = "194.68.245.86; command"
    elif change == "port":
        port = "22071 -oProxyCommand=bad"
    elif change == "known_host":
        known = known.replace("22071", "22")
    elif change == "multiple_hosts":
        known += known
    elif change == "name":
        manifest[0]["name"] = "../names-confirmation-bad.tar.gz"
    elif change == "size":
        manifest[0]["size_bytes"] = 2 * 1024**3
    elif change == "sha":
        manifest[0]["sha256"] = "not-a-hash"
    else:
        manifest.append(dict(ITEM))
    with pytest.raises(ValueError):
        publisher.validate_inputs(host, port, known, json.dumps(manifest))


def test_existing_asset_must_match_server_digest_size_and_name():
    asset = {
        **ITEM,
        "size": ITEM["size_bytes"],
        "digest": "sha256:" + ITEM["sha256"],
        "id": 10,
        "state": "uploaded",
    }
    assert publisher.asset_record(asset, ITEM)["id"] == 10
    for wrong in (
        {"size": ITEM["size_bytes"] + 1},
        {"digest": "sha256:" + "0" * 64},
        {"name": "other.tar.gz"},
    ):
        with pytest.raises(ValueError):
            publisher.asset_record({**asset, **wrong}, ITEM)


@pytest.mark.parametrize("bad", ["hash", "oversize", "short"])
def test_failed_stream_verification_removes_only_its_new_partial(tmp_path, bad):
    content = (
        b"x" * len(PAYLOAD)
        if bad == "hash"
        else PAYLOAD + b"extra"
        if bad == "oversize"
        else PAYLOAD[:-1]
    )
    command = [
        sys.executable,
        "-c",
        "import sys; sys.stdout.buffer.write(" + repr(content) + ")",
    ]
    target = tmp_path / "derived.partial"
    with pytest.raises((ValueError, RuntimeError)):
        publisher.download(command, target, ITEM, publisher.time.monotonic() + 10)
    assert not target.exists()
    target.write_bytes(b"pre-existing file")
    with pytest.raises(FileExistsError):
        publisher.download(command, target, ITEM, publisher.time.monotonic() + 10)
    assert target.read_bytes() == b"pre-existing file"


def test_valid_stream_is_written_without_loading_archive_contents(tmp_path):
    command = [
        sys.executable,
        "-c",
        "import sys; sys.stdout.buffer.write(" + repr(PAYLOAD) + ")",
    ]
    target = tmp_path / "derived.partial"
    publisher.download(command, target, ITEM, publisher.time.monotonic() + 10)
    assert target.read_bytes() == PAYLOAD


def test_public_key_format_is_exact_and_no_private_bytes_are_accepted():
    publisher.validate_public_key(
        f"ssh-ed25519 {KEY} names-native-reader-42\n".encode(), "42"
    )
    for value in (b"PRIVATE KEY SECRET", f"ssh-ed25519 {KEY} wrong-comment\n".encode()):
        with pytest.raises(ValueError):
            publisher.validate_public_key(value, "42")


def fake_network(
    monkeypatch, *, existing=None, broken_download=False, missing_public_digest=False
):
    state = {
        "assets": list(existing or []),
        "deleted": [],
        "uploaded": [],
        "private_contents": b"PRIVATE-KEY-SECRET-DO-NOT-PRINT",
    }

    def github(method, endpoint, *, file=None, **kwargs):
        if method == "GET" and endpoint.endswith("/447421086"):
            return {"id": 447421086, "tag_name": publisher.TAG, "draft": True}
        if method == "GET" and "/releases/assets/" in endpoint:
            asset = next(
                a for a in state["assets"] if a["id"] == int(endpoint.rsplit("/", 1)[1])
            )
            return (
                {**asset, "digest": None}
                if missing_public_digest and asset["name"].endswith(".pub")
                else asset
            )
        if method == "GET":
            return [list(state["assets"])]
        if method == "DELETE":
            identity = int(endpoint.rsplit("/", 1)[1])
            state["deleted"].append(identity)
            state["assets"][:] = [a for a in state["assets"] if a["id"] != identity]
            return None
        assert method == "POST" and file is not None
        content = Path(file).read_bytes()
        assert state["private_contents"] not in content
        name = endpoint.split("?name=", 1)[1]
        asset = {
            "name": name,
            "size": len(content),
            "digest": "sha256:" + hashlib.sha256(content).hexdigest(),
            "id": len(state["uploaded"]) + 100,
            "state": "uploaded",
        }
        state["assets"].append(asset)
        state["uploaded"].append((name, content))
        return asset

    def create_key(directory, run_id):
        private = directory / "reader_ed25519"
        private.write_bytes(state["private_contents"])
        public = private.with_suffix(".pub")
        public.write_text(f"ssh-ed25519 {KEY} names-native-reader-{run_id}\n")
        return private, public

    def download(command, target, item, deadline):
        assert command[-1] == "get " + ITEM["name"]
        assert "StrictHostKeyChecking=yes" in command
        assert "SendEnv=-*" in command and "ForwardAgent=no" in command
        assert state["private_contents"].decode() not in " ".join(command)
        if broken_download:
            raise ValueError("synthetic archive mismatch")
        target.write_bytes(PAYLOAD)

    monkeypatch.setattr(publisher, "github", github)
    monkeypatch.setattr(publisher, "create_key", create_key)
    monkeypatch.setattr(publisher, "wait_for_probe", lambda *args: None)
    monkeypatch.setattr(publisher, "download", download)
    monkeypatch.setattr(publisher.time, "sleep", lambda duration: None)
    return state


def test_controller_uploads_only_verified_archives_cleans_public_key_and_never_prints_private(
    tmp_path, monkeypatch, capsys
):
    state = fake_network(monkeypatch)
    receipt = publisher.publish(inputs(), "42", tmp_path)
    assert receipt["status"] == "complete" and len(receipt["archives"]) == 1
    assert [n for n, _ in state["uploaded"]] == [
        "names-native-reader-42.pub",
        ITEM["name"],
        "names-native-upload-receipt-42.json",
    ]
    assert state["deleted"] == [100]
    assert not list(tmp_path.iterdir())
    assert "PRIVATE-KEY-SECRET" not in capsys.readouterr().out


def test_failed_download_still_removes_own_public_key_and_publishes_failure_receipt(
    tmp_path, monkeypatch
):
    state = fake_network(monkeypatch, broken_download=True)
    with pytest.raises(ValueError):
        publisher.publish(inputs(), "42", tmp_path)
    assert state["deleted"] == [100]
    assert [n for n, _ in state["uploaded"]] == [
        "names-native-reader-42.pub",
        "names-native-upload-receipt-42.json",
    ]
    assert not list(tmp_path.iterdir())


def test_existing_matching_archive_skips_reader_but_mismatch_stops_before_writes(
    tmp_path, monkeypatch
):
    existing = {
        "name": ITEM["name"],
        "size": ITEM["size_bytes"],
        "digest": "sha256:" + ITEM["sha256"],
        "id": 12,
        "state": "uploaded",
    }
    state = fake_network(monkeypatch, existing=[existing])
    publisher.publish(inputs(), "42", tmp_path)
    assert [n for n, _ in state["uploaded"]] == ["names-native-upload-receipt-42.json"]
    state = fake_network(
        monkeypatch, existing=[{**existing, "digest": "sha256:" + "0" * 64}]
    )
    with pytest.raises(ValueError):
        publisher.publish(inputs(), "43", tmp_path)
    assert state["uploaded"] == [] and state["deleted"] == []


def test_public_key_upload_is_cleaned_even_when_server_digest_verification_fails(
    tmp_path, monkeypatch
):
    state = fake_network(monkeypatch, missing_public_digest=True)
    with pytest.raises(RuntimeError, match="server digest"):
        publisher.publish(inputs(), "42", tmp_path)
    assert state["deleted"] == [100]
    assert not list(tmp_path.iterdir())


def test_occupied_public_key_name_is_never_deleted_or_replaced(tmp_path, monkeypatch):
    occupied = {
        "name": "names-native-reader-42.pub",
        "size": 1,
        "digest": "sha256:" + "0" * 64,
        "id": 77,
        "state": "uploaded",
    }
    state = fake_network(monkeypatch, existing=[occupied])
    with pytest.raises(ValueError, match="already occupied"):
        publisher.publish(inputs(), "42", tmp_path)
    assert state["uploaded"] == [] and state["deleted"] == []


def test_lost_public_key_post_response_is_cleaned_only_after_exact_digest_match(
    tmp_path, monkeypatch
):
    state = fake_network(monkeypatch)
    original = publisher.github

    def lost_response(method, endpoint, **kwargs):
        value = original(method, endpoint, **kwargs)
        if method == "POST" and "?name=names-native-reader-" in endpoint:
            raise RuntimeError("synthetic lost response")
        return value

    monkeypatch.setattr(publisher, "github", lost_response)
    with pytest.raises(RuntimeError, match="lost response"):
        publisher.publish(inputs(), "42", tmp_path)
    assert state["deleted"] == [100]


def test_workflow_is_manual_standard_runner_only_and_inputs_are_not_shell_code():
    workflow = (ROOT / ".github/workflows/publish-native-archives.yml").read_text()
    assert "workflow_dispatch:" in workflow and "runs-on: ubuntu-latest" in workflow
    assert "timeout-minutes: 90" in workflow and "contents: write" in workflow
    assert "persist-credentials: false" in workflow
    assert "actions/checkout@d23441a48e516b6c34aea4fa41551a30e30af803" in workflow
    assert "run: python3 tools/publish_native_archives.py" in workflow
    assert not any(
        "${{" in line
        for line in workflow.splitlines()
        if line.strip().startswith("run:")
    )
    assert "actions/upload-artifact" not in workflow and "actions/cache" not in workflow
    assert "self-hosted" not in workflow
    assert "github.event.repository.private == false" in workflow


def test_github_binary_upload_has_explicit_length_without_exposing_credentials(
    tmp_path, monkeypatch
):
    archive = tmp_path / "archive.tar.gz"
    archive.write_bytes(PAYLOAD)
    observed = []

    def run(command, **kwargs):
        observed.append(command)
        return type("Response", (), {"returncode": 0, "stdout": b'{"id": 1}'})()

    monkeypatch.setattr(publisher.subprocess, "run", run)
    publisher.github("POST", "https://uploads.github.com/approved-test", file=archive)
    assert f"Content-Length: {len(PAYLOAD)}" in observed[0]
    assert "Content-Type: application/octet-stream" in observed[0]
    assert "--hostname" in observed[0] and "github.com" in observed[0]
    assert not any("Authorization" in part or "TOKEN" in part for part in observed[0])


def test_main_refuses_accidental_local_execution_before_any_github_request(monkeypatch):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.setattr(
        publisher,
        "github",
        lambda *args, **kwargs: pytest.fail("no GitHub request permitted"),
    )
    with pytest.raises(RuntimeError, match="only in the fixed repository"):
        publisher.main()
