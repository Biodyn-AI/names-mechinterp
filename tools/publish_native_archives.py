"""Stream approved cloud archives to one existing draft GitHub release.

Run only on the public repository's standard GitHub-hosted runner. The ephemeral
private SSH key stays in RUNNER_TEMP; the public asset lets the owner authorize
its tightly restricted reader. No scientific source or remote archive is edited.
"""

import base64
import hashlib
import ipaddress
import json
import os
import re
import selectors
import signal
import subprocess
import tempfile
import time
from pathlib import Path

REPOSITORY = "Biodyn-AI/names-mechinterp"
RELEASE_ID = 447421086
TAG = "native-data-20261116-v1"
API = f"repos/{REPOSITORY}/"
ED25519_PREFIX = b"\0\0\0\x0bssh-ed25519\0\0\0\x20"
ARCHIVE_NAME = re.compile(
    r"(?:names-confirmation-[a-z0-9]+(?:-[a-z0-9]+)*|names-development-part[0-9]{3})\.tar\.gz"
)


def validate_inputs(host, port, known_hosts, manifest):
    # This operation is authorized only for the currently owned source Pod.
    if (
        host != "194.68.245.86"
        or str(ipaddress.IPv4Address(host)) != host
        or port != "22071"
    ):
        raise ValueError("source address is not the approved cloud reader")
    fields = known_hosts.strip().split()
    if (
        len(known_hosts.strip().splitlines()) != 1
        or len(fields) != 3
        or fields[:2] != [f"[{host}]:{port}", "ssh-ed25519"]
    ):
        raise ValueError("exactly one matching pinned ED25519 host key is required")
    try:
        key = base64.b64decode(fields[2], validate=True)
    except ValueError as error:
        raise ValueError("invalid pinned host key") from error
    if len(key) != 51 or not key.startswith(ED25519_PREFIX):
        raise ValueError("invalid pinned ED25519 host key")
    items = json.loads(manifest)
    if not isinstance(items, list) or not 1 <= len(items) <= 32:
        raise ValueError("one to 32 approved archives are required")
    names = set()
    for item in items:
        if (
            not isinstance(item, dict)
            or set(item) != {"name", "size_bytes", "sha256"}
            or not isinstance(item["name"], str)
            or not ARCHIVE_NAME.fullmatch(item["name"])
            or len(item["name"]) > 160
            or item["name"] in names
            or type(item["size_bytes"]) is not int
            or not 0 < item["size_bytes"] < 2 * 1024**3
            or not isinstance(item["sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
        ):
            raise ValueError(
                "archive names, sizes and SHA-256 values must match the strict catalog schema"
            )
        names.add(item["name"])
    return {
        "host": host,
        "port": port,
        "known_hosts": " ".join(fields) + "\n",
        "archives": items,
    }


def github(method, endpoint, *, file=None, timeout=180):
    command = ["gh", "api", "--hostname", "github.com", "--method", method, endpoint]
    if file is not None:
        # gh maps this header to Request.ContentLength; raw file readers otherwise
        # have no inferred length, and GitHub's upload endpoint requires it.
        command += [
            "-H",
            "Content-Type: application/octet-stream",
            "-H",
            f"Content-Length: {Path(file).stat().st_size}",
            "--input",
            str(file),
        ]
    elif method == "GET" and endpoint.endswith("assets?per_page=100"):
        command += ["--paginate", "--slurp"]
    process = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if process.returncode:
        # Do not forward client diagnostics, headers, environment or key contents.
        raise RuntimeError(f"GitHub {method} request failed")
    return json.loads(process.stdout) if process.stdout.strip() else None


def asset_record(asset, item):
    if (
        asset.get("name") != item["name"]
        or type(asset.get("size")) is not int
        or asset["size"] != item["size_bytes"]
        or asset.get("digest") != "sha256:" + item["sha256"]
        or asset.get("state") != "uploaded"
        or type(asset.get("id")) is not int
        or asset["id"] <= 0
    ):
        raise ValueError(
            "existing or uploaded release asset does not match the approved digest and size"
        )
    return {**item, "id": asset["id"], "digest": asset["digest"]}


def validate_public_key(content, run_id):
    try:
        text = content.decode("ascii")
        match = re.fullmatch(
            r"ssh-ed25519 ([A-Za-z0-9+/=]+) names-native-reader-" + run_id + r"\n", text
        )
        key = base64.b64decode(match[1], validate=True) if match else b""
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError("unexpected generated public key") from error
    if len(key) != 51 or not key.startswith(ED25519_PREFIX):
        raise ValueError("unexpected generated public key")


def create_key(directory, run_id):
    private = directory / "reader_ed25519"
    process = subprocess.run(
        [
            "ssh-keygen",
            "-q",
            "-t",
            "ed25519",
            "-N",
            "",
            "-C",
            f"names-native-reader-{run_id}",
            "-f",
            str(private),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=30,
        check=False,
    )
    if process.returncode:
        raise RuntimeError("ephemeral SSH key generation failed")
    return private, private.with_suffix(".pub")


def ssh_command(inputs, private, hosts, command):
    return [
        "ssh",
        "-F",
        "/dev/null",
        "-T",
        "-i",
        str(private),
        "-p",
        inputs["port"],
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "IdentityAgent=none",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        f"UserKnownHostsFile={hosts}",
        "-o",
        "GlobalKnownHostsFile=/dev/null",
        "-o",
        "ForwardAgent=no",
        "-o",
        "SendEnv=-*",
        "-o",
        "ClearAllForwardings=yes",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=2",
        "root@" + inputs["host"],
        command,
    ]


def wait_for_probe(command, deadline):
    while time.monotonic() < deadline:
        try:
            process = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=min(20, max(1, deadline - time.monotonic())),
                check=False,
            )
            if (
                process.returncode == 0
                and process.stdout == b"names-native-reader-v1\n"
            ):
                return
        except subprocess.TimeoutExpired:
            pass
        time.sleep(min(10, max(0, deadline - time.monotonic())))
    raise TimeoutError("read-only SSH authorization was not ready within ten minutes")


def download(command, target, item, deadline):
    """Bound disk use to one expected archive; retain no failed derived partial."""
    created, process = False, None
    try:
        with target.open("xb") as handle:
            created = True
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            digest, count = hashlib.sha256(), 0
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("archive download deadline passed")
                    if not selector.select(min(10, remaining)):
                        continue
                    chunk = os.read(process.stdout.fileno(), 1024 * 1024)
                    if not chunk:
                        break
                    count += len(chunk)
                    if count > item["size_bytes"]:
                        raise ValueError(
                            "archive stream exceeded its approved byte count"
                        )
                    handle.write(chunk)
                    digest.update(chunk)
            if (
                process.wait(timeout=max(1, deadline - time.monotonic())) != 0
                or count != item["size_bytes"]
                or digest.hexdigest() != item["sha256"]
            ):
                raise ValueError(
                    "archive stream did not match its approved SHA-256 and size"
                )
    except BaseException:
        if created:
            target.unlink(missing_ok=True)
        raise
    finally:
        if process is not None:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
            process.stdout.close()


def publish(inputs, run_id, temp_root):
    if not re.fullmatch(r"[0-9]{1,24}", run_id):
        raise ValueError("a numeric GitHub workflow run ID is required")
    deadline = time.monotonic() + 80 * 60
    created_assets = {}

    def api(method, endpoint, *, file=None):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("upload-session deadline passed")
        return github(
            method, endpoint, file=file, timeout=min(1200 if file else 180, remaining)
        )

    def draft():
        release = api("GET", API + f"releases/{RELEASE_ID}")
        if (
            release.get("id") != RELEASE_ID
            or release.get("tag_name") != TAG
            or release.get("draft") is not True
        ):
            raise ValueError("the fixed draft release identity or state changed")

    def assets():
        pages = api("GET", API + f"releases/{RELEASE_ID}/assets?per_page=100")
        found = [asset for page in pages for asset in page]
        if len({a["name"] for a in found}) != len(found):
            raise ValueError("duplicate release asset names")
        return {asset["name"]: asset for asset in found}

    def upload(path, item):
        draft()
        existing = assets().get(item["name"])
        if existing is not None:
            return asset_record(existing, item)
        uploaded = api(
            "POST",
            f"https://uploads.github.com/{API}releases/{RELEASE_ID}/assets?name={item['name']}",
            file=path,
        )
        if (
            uploaded.get("name") != item["name"]
            or type(uploaded.get("id")) is not int
            or uploaded["id"] <= 0
        ):
            raise RuntimeError("GitHub returned an unexpected upload identity")
        created_assets[item["name"]] = uploaded["id"]
        for _ in range(30):
            confirmed = api("GET", API + f"releases/assets/{uploaded['id']}")
            if confirmed.get("digest") is not None:
                return asset_record(confirmed, item)
            time.sleep(2)
        raise RuntimeError("GitHub did not supply a verifiable server digest")

    draft()
    existing, complete, pending = assets(), [], []
    for item in inputs["archives"]:
        if item["name"] in existing:
            complete.append(asset_record(existing[item["name"]], item))
        else:
            pending.append(item)
    key_name = f"names-native-reader-{run_id}.pub"
    if key_name in existing:
        raise ValueError("temporary public-key asset name is already occupied")
    with tempfile.TemporaryDirectory(
        prefix=f"names-native-upload-{run_id}-", dir=temp_root
    ) as owned:
        directory, key_asset, key_item, failure = Path(owned), None, None, None
        receipt = {
            "run_id": run_id,
            "release_id": RELEASE_ID,
            "archives": complete,
            "scope": "Verified archive-byte transfer only; no native mathematical replay or model execution",
        }
        try:
            if pending:
                private, public = create_key(directory, run_id)
                public_content = public.read_bytes()
                validate_public_key(public_content, run_id)
                key_item = {
                    "name": key_name,
                    "size_bytes": len(public_content),
                    "sha256": hashlib.sha256(public_content).hexdigest(),
                }
                key_asset = upload(public, key_item)
                hosts = directory / "known_hosts"
                with hosts.open("x", encoding="ascii") as handle:
                    handle.write(inputs["known_hosts"])
                wait_for_probe(
                    ssh_command(inputs, private, hosts, "probe"),
                    min(deadline, time.monotonic() + 600),
                )
                for item in pending:
                    partial = directory / (item["name"] + ".partial")
                    download(
                        ssh_command(inputs, private, hosts, "get " + item["name"]),
                        partial,
                        item,
                        deadline,
                    )
                    complete.append(upload(partial, item))
                    partial.unlink()
        except BaseException as error:  # noqa: BLE001 - re-raised after exact-key cleanup and receipt
            failure = error
        finally:
            # Work stops at 80 minutes; retain a small, bounded cleanup window
            # before the workflow's independent 90-minute hard limit.
            deadline = max(deadline, time.monotonic() + 180)
            key_id = created_assets.get(key_name) or (
                key_asset["id"] if key_asset else None
            )
            # A lost POST response can still be resolved by the unique public key's
            # complete server digest. Never delete an unverified namesake asset.
            if key_id is None and key_item is not None:
                try:
                    candidate = assets().get(key_name)
                    if candidate is not None:
                        key_id = asset_record(candidate, key_item)["id"]
                except BaseException as lookup_error:  # noqa: BLE001 - retain cancellation until cleanup finishes
                    failure = failure or lookup_error
            if key_id is not None:
                receipt["temporary_public_key_asset_id"] = key_id
                try:
                    api("DELETE", API + f"releases/assets/{key_id}")
                    receipt["temporary_public_key_removed"] = True
                except BaseException as cleanup_error:  # noqa: BLE001 - failure is recorded and re-raised
                    receipt["temporary_public_key_removed"] = False
                    failure = failure or cleanup_error
            receipt["status"] = "failed" if failure else "complete"
            if failure:
                receipt["error_type"] = type(failure).__name__
            content = (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode()
            receipt_path = directory / "receipt.json"
            with receipt_path.open("xb") as handle:
                handle.write(content)
            upload(
                receipt_path,
                {
                    "name": f"names-native-upload-receipt-{run_id}.json",
                    "size_bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                },
            )
        if failure:
            raise failure
        return receipt


def main():
    if (
        os.environ.get("GITHUB_ACTIONS") != "true"
        or os.environ.get("GITHUB_REPOSITORY") != REPOSITORY
    ):
        raise RuntimeError(
            "run this publisher only in the fixed repository's GitHub workflow"
        )

    def interrupted(signum, frame):
        raise InterruptedError("native upload was interrupted")

    signal.signal(signal.SIGTERM, interrupted)
    inputs = validate_inputs(
        os.environ["NATIVE_SOURCE_HOST"],
        os.environ["NATIVE_SOURCE_PORT"],
        os.environ["NATIVE_KNOWN_HOSTS"],
        os.environ["NATIVE_ARCHIVE_MANIFEST"],
    )
    receipt = publish(
        inputs, os.environ["GITHUB_RUN_ID"], Path(os.environ["RUNNER_TEMP"])
    )
    print(
        json.dumps(
            {"status": receipt["status"], "verified_archives": len(receipt["archives"])}
        )
    )


if __name__ == "__main__":
    main()
